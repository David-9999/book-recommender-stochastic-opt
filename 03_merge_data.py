import pandas as pd
import numpy as np
import os

# ------------------------------------------
# Paths
cwd = os.getcwd()

books_path = os.path.join(cwd, "dataset/books.csv")
book_tags_path = os.path.join(cwd, "dataset/book_tags.csv")
tags_path = os.path.join(cwd, "dataset/tags_preprocessed.csv")
output_metadata_path = os.path.join(cwd, "dataset/books_features_metadata.csv")
output_book_tags_path = os.path.join(cwd, "dataset/book_tags_clean.csv")

# ------------------------------------------
# Load data
books = pd.read_csv(books_path)
book_tags = pd.read_csv(book_tags_path)
tags = pd.read_csv(tags_path)

print("Loaded datasets:")
print("Books:", books.shape)
print("Book tags:", book_tags.shape)
print("Tags:", tags.shape)

# ------------------------------------------
# Keep only needed book columns
books = books[
    [
        "id",
        "book_id",
        "title",
        "authors",
        "language_code",
        "average_rating",
        "ratings_count"
    ]
].copy()

# ------------------------------------------
# Merge book_tags with cleaned tag names
book_tags_named = book_tags.merge(tags[['tag_id', 'tag_name']], on='tag_id', how='inner')

# ------------------------------------------
# Aggregate tag counts per book
book_tag_counts = (
    book_tags_named
    .groupby(['goodreads_book_id', 'tag_name'], as_index=False)['count']
    .sum()
)

# ------------------------------------------
# Top-5 tags per book for inspection
top_tags = (
    book_tag_counts
    .sort_values(['goodreads_book_id', 'count'], ascending=[True, False])
    .groupby('goodreads_book_id', group_keys=False)
    .head(5)
)

top_tags_pivot = (
    top_tags
    .groupby('goodreads_book_id')['tag_name']
    .apply(list)
    .reset_index()
)

for i in range(5):
    top_tags_pivot[f'top_{i+1}_tag'] = top_tags_pivot['tag_name'].apply(
        lambda x: x[i] if len(x) > i else None
    )

top_tags_pivot = top_tags_pivot.drop(columns=['tag_name'])

# ------------------------------------------
# Merge metadata with top tags
books_features = books.merge(top_tags_pivot, left_on='book_id', right_on='goodreads_book_id', how='left')

books_features = books_features.drop(columns=['goodreads_book_id'])

# ------------------------------------------
# Clean metadata
books_features['language_code'] = books_features['language_code'].fillna('unknown')
books_features['ratings_count'] = books_features['ratings_count'].fillna(0)
books_features['average_rating'] = books_features['average_rating'].fillna(0)

# Popularity feature
books_features['log_ratings_count'] = np.log1p(books_features['ratings_count'])

# ------------------------------------------
# Save outputs
books_features.to_csv(output_metadata_path, index=False)

book_tag_counts[['goodreads_book_id', 'tag_name']].to_csv(output_book_tags_path, index=False)

print("\nSaved:")
print(f"- {output_metadata_path}")
print(f"- {output_book_tags_path}")