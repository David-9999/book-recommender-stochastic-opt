import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

from scipy.sparse import load_npz, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# ------------------------------------------
# Paths
cwd = os.getcwd()
X_path = os.path.join(cwd, "dataset/books_features_X.npz")
meta_npz_path = os.path.join(cwd, "dataset/books_features_meta.npz")
ratings_path = os.path.join(cwd, "dataset/ratings.csv")

OUTPUT_FILE = "weights_schg.dat"

# ------------------------------------------
# Evaluation parameters
K_ITEM_NEIGHBORS = 11
TOP_K_EVAL = 10

MIN_RATINGS_PER_USER = 3
USER_SAMPLE_SIZE = 2000
RANDOM_SEED = 42

PROGRESS_EVERY = 100

# ------------------------------------------
# Rating weight function
def rating_weight(r):
    r = np.clip(r, 0, 5)
    a = 2.0
    b = 0.5
    return a * (np.exp(b * (r - 2.5)) - 1)

# ------------------------------------------
# Load feature matrix
print("Loading feature matrix...")
X_full = load_npz(X_path)

meta_npz = np.load(meta_npz_path, allow_pickle=True)
feature_ids = meta_npz["ids"]

# ------------------------------------------
# Load ratings
print("Loading ratings...")
ratings = pd.read_csv(ratings_path)
ratings = ratings[ratings["book_id"].isin(feature_ids)]

id_to_idx = {bid: i for i, bid in enumerate(feature_ids)}

user_history = defaultdict(list)
for _, row in ratings.iterrows():
    user_history[row["user_id"]].append(
        (id_to_idx[row["book_id"]], row["rating"])
    )

# Keep users with enough ratings
user_history = {
    u: items for u, items in user_history.items()
    if len(items) >= MIN_RATINGS_PER_USER
}

# Sample users
rng = np.random.default_rng(RANDOM_SEED)
users = list(user_history.keys())
if len(users) > USER_SAMPLE_SIZE:
    users = rng.choice(users, USER_SAMPLE_SIZE, replace=False)
    user_history = {u: user_history[u] for u in users}

print(f"Users used for evaluation: {len(user_history)}")

# ------------------------------------------
# Feature blocks
BLOCK_SIZES = {
    "title": 768,
    "tags": 750,
    "authors": 1001,
    "lang": 26,
    "numeric": 3
}

def split_blocks(X):
    blocks = {}
    start = 0
    for name, size in BLOCK_SIZES.items():
        blocks[name] = X[:, start:start + size]
        start += size
    return blocks

X_blocks = split_blocks(X_full)

# ------------------------------------------
# Build weighted matrix
def build_weighted_X(weights):
    parts = [X_blocks[k] * w for k, w in weights.items()]
    Xw = hstack(parts).tocsr()
    return normalize(Xw, axis=1)

# ------------------------------------------
# Evaluation function (LEAVE-ONE-OUT)
def evaluate_weights(weights):
    start_time = time.time()

    Xw = build_weighted_X(weights)

    knn = NearestNeighbors(
        n_neighbors=K_ITEM_NEIGHBORS + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    knn.fit(Xw)

    distances, neighbors = knn.kneighbors(Xw)

    hits = 0
    users_eval = 0

    for i, (user, items) in enumerate(user_history.items(), 1):
        items = list(items)

        # Leave-one-out
        test_pos = rng.integers(len(items))
        test_item, _ = items[test_pos]

        train_items = [
            (idx, r)
            for j, (idx, r) in enumerate(items)
            if j != test_pos
        ]

        seen = {idx for idx, _ in train_items}
        scores = defaultdict(float)

        for item_idx, rating in train_items:
            for d, j in zip(
                distances[item_idx][1:],
                neighbors[item_idx][1:]
            ):
                if j in seen:
                    continue
                scores[j] += (1 - d) * rating_weight(rating)

        if not scores:
            continue

        top_k = set(
            idx for idx, _ in sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_K_EVAL]
        )

        # Check if the single hidden item is in Top K
        hits += int(test_item in top_k)
        users_eval += 1

        if i % PROGRESS_EVERY == 0:
            print(f"  evaluated {i}/{len(user_history)} users")

    elapsed = time.time() - start_time

    hit_rate = hits / users_eval
    recall = hit_rate  # LOO: Recall@K == Hit@K

    return hit_rate, recall, elapsed

# ------------------------------------------
# Prepare output file
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w") as f:
        f.write(
            "timestamp,title,tags,authors,lang,numeric,"
            "hit@10,recall@10,time_s\n"
        )

# ------------------------------------------
# SCHG search parameters
"""
initial_weights = {
    "title": 0.6,
    "tags": 1.0,
    "authors": 0.7,
    "lang": 0.2,
    "numeric": 0.01
}
"""
# SCHG search parameters
initial_weights = {
    "title": 1.0,
    "tags": 1.0,
    "authors": 1.0,
    "lang": 1.0,
    "numeric": 1.0
}

delta = {
    "title": 0.05,
    "tags": 0.05,
    "authors": 0.05,
    "lang": 0.01,
    "numeric": 0.001
}

MAX_ITER = 250

weights_best = initial_weights.copy()
hit_best, recall_best, _ = evaluate_weights(weights_best)

print("\nStarting SCHG weight search...")
print(f"Initial Hit@10={hit_best:.4f}")

for iteration in range(1, MAX_ITER + 1):
    # Random perturbation of each weight
    weights_new = {
        k: max(0.0, weights_best[k] + rng.uniform(-delta[k], delta[k]))
        for k in weights_best
    }

    hit_new, recall_new, t_new = evaluate_weights(weights_new)

    if hit_new > hit_best:
        weights_best = weights_new
        hit_best = hit_new
        recall_best = recall_new
        print(f"[Iter {iteration}] New best Hit@10={hit_best:.4f} | weights={weights_best}")

    # Log every iteration
    with open(OUTPUT_FILE, "a") as f:
        f.write(
            f"{datetime.now().isoformat()},"
            f"{weights_new['title']},"
            f"{weights_new['tags']},"
            f"{weights_new['authors']},"
            f"{weights_new['lang']},"
            f"{weights_new['numeric']},"
            f"{hit_new:.6f},"
            f"{recall_new:.6f},"
            f"{t_new:.2f}\n"
        )

print("\nSCHG finished!")
print(f"Best weights: {weights_best}")
print(f"Best Hit@10={hit_best:.4f}, Recall@10={recall_best:.4f}")
