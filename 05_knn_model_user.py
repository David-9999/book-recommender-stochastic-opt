import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------
# Paths
cwd = os.getcwd()

X_path = os.path.join(cwd, "dataset/books_features_X.npz")
meta_npz_path = os.path.join(cwd, "dataset/books_features_meta.npz")
books_path = os.path.join(cwd, "dataset/books_features_metadata.csv")
ratings_path = os.path.join(cwd, "dataset/ratings.csv")

# ------------------------------------------
# Hyperparameters
K_ITEM_NEIGHBORS = 11       # neighbors per rated book
TOP_N_RECOMMEND = 30        # final recommendations

# ------------------------------------------
# Rating weight function (strongly boost high ratings, penalize low)
def rating_weight(r):
    r = np.clip(r, 0, 5)
    a = 2.0
    b = 0.5
    return a * (np.exp(b * (r - 2.5)) - 1)

# ------------------------------------------
# Load item features
print("Loading item features...")
X = load_npz(X_path)

meta_npz = np.load(meta_npz_path, allow_pickle=True)
feature_ids = meta_npz["ids"]        # use 'id' as canonical key
book_ids = meta_npz["book_ids"]      # original book_id from source (optional)

# Map id → row index
id_to_idx = {bid: i for i, bid in enumerate(feature_ids)}

# ------------------------------------------
# Load book metadata
books_df = pd.read_csv(books_path).reset_index(drop=True)
assert X.shape[0] == len(books_df), "Feature / metadata mismatch!"

# ------------------------------------------
# Train item–item KNN
print("Training item-item KNN...")
knn = NearestNeighbors(
    n_neighbors=K_ITEM_NEIGHBORS,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)
knn.fit(X)
print("KNN ready.")

# ------------------------------------------
# Load user ratings
print("Loading user ratings...")
ratings = pd.read_csv(ratings_path)

# Ensure ratings use 'id' for mapping
#ratings = ratings.rename(columns={"book_id": "id"})
ratings = ratings[ratings["book_id"].isin(id_to_idx)]

# Build user history: user_id → list of (feature_idx, rating)
user_history = defaultdict(list)
for _, row in ratings.iterrows():
    user_history[row["user_id"]].append(
        (id_to_idx[row["book_id"]], row["rating"])
    )

# ------------------------------------------
# Recommend books for a user
def recommend_for_user(user_id, top_n=TOP_N_RECOMMEND):
    liked_items = user_history.get(user_id, [])
    liked_indices = {i for i, _ in liked_items}

    candidate_scores = defaultdict(float)

    for item_idx, rating in liked_items:
        distances, neighbors = knn.kneighbors(
            X[item_idx],
            n_neighbors=K_ITEM_NEIGHBORS + 1
        )

        for d, j in zip(distances[0][1:], neighbors[0][1:]):
            if j in liked_indices:
                continue
            sim = 1.0 - d
            candidate_scores[j] += sim * rating_weight(rating)

    if not candidate_scores:
        return []

    ranked = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    for idx, score in ranked[:top_n]:
        row = books_df.iloc[idx]
        recommendations.append({
            "book_id": int(feature_ids[idx]),   # canonical id
            "title": row["title"],
            "authors": row["authors"],
            "average_rating": float(row["average_rating"]),
            "score": float(score)
        })

    return recommendations

# ------------------------------------------
# Pretty print recommendations
def print_user_recommendations(user_id, top_n=TOP_N_RECOMMEND):
    print(f"\nUSER: {user_id}")
    print("-" * 100)

    liked = user_history.get(user_id, [])
    print(f"User rated {len(liked)} books. Here are some of rated books:")
    print(f"{'Title':<45} | {'Authors':<30} | {'Rating':<6}")
    print("-" * 100)

    sample_size = min(len(liked), 10)

    for idx, r in random.sample(liked, sample_size):
        print(f"{books_df.iloc[idx]['title'][:45]:<45} | "
            f"{books_df.iloc[idx]['authors'][:30]:<30} | "
            f"{r:<8.2f}")

 
    print("\nRECOMMENDATIONS")
    print(f"{'Title':<45} | {'Authors':<30} | {'Rating':<8} | {'Score':<8}")
    print("-" * 100)

    recs = recommend_for_user(user_id, top_n)

    for r in recs:
        print(
            f"{r['title'][:45]:<45} | "
            f"{r['authors'][:30]:<30} | "
            f"{r['average_rating']:<8.2f} | "
            f"{r['score']:<8.3f}"
        )

# ------------------------------------------
# Example run
if __name__ == "__main__":
    #example_user = 314
    example_user = 999999 # vendo's user ID
    print_user_recommendations(example_user, top_n=TOP_N_RECOMMEND)
