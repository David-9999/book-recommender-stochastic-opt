import os
import re
import numpy as np
import pandas as pd
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MultiLabelBinarizer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz

# ------------------------------------------
# Paths
cwd = os.getcwd()

metadata_path = os.path.join(cwd, "dataset/books_features_metadata.csv")
tags_path = os.path.join(cwd, "dataset/book_tags_clean.csv")

X_out_path = os.path.join(cwd, "dataset/books_features_X.npz")
meta_out_path = os.path.join(cwd, "dataset/books_features_meta.npz")

# ------------------------------------------
# Load data
meta = pd.read_csv(metadata_path)
book_tags = pd.read_csv(tags_path)

# ------------------------------------------
# Author normalization helpers
def normalize_author(name: str) -> str:
    name = name.lower()
    name = re.sub(r'\.', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def split_and_normalize_authors(author_str: str):
    if pd.isna(author_str):
        return []
    return [
        normalize_author(a)
        for a in author_str.split(',')
        if a.strip()
    ]

# ------------------------------------------
# Title embeddings
print("\nEmbedding titles...")
model = SentenceTransformer("all-distilroberta-v1")

X_title = model.encode(
    meta["title"].fillna("").tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normalize dense block
X_title = X_title / np.linalg.norm(X_title, axis=1, keepdims=True)
X_title = csr_matrix(X_title)

# ------------------------------------------
# Tag TF-IDF
print("\nBuilding tag TF-IDF...")

tag_docs = (
    book_tags
    .groupby('goodreads_book_id')['tag_name']
    .apply(lambda tags: " ".join(tags))
)

tag_docs = meta['book_id'].map(tag_docs).fillna("") # Aligning with metadata & filling missing

vectorizer = TfidfVectorizer(
    min_df=5,
    max_features=750, # This can be adjusted
    ngram_range=(1, 1)
)

X_tags = vectorizer.fit_transform(tag_docs)
X_tags = normalize(X_tags, norm="l2", axis=1)

# ------------------------------------------
# Author multi-hot
print("\nEncoding authors...")

meta['author_list'] = meta['authors'].apply(split_and_normalize_authors)

author_counter = Counter(
    a for authors in meta['author_list'] for a in authors
)

print("\nTop 20 authors:")
for author, count in author_counter.most_common(20):
    print(f"{author:30s} {count}")

TOP_AUTHORS = 1000 # This can be adjusted
top_authors = set(a for a, _ in author_counter.most_common(TOP_AUTHORS))

meta['author_list_filtered'] = meta['author_list'].apply(
    lambda authors: [
        a if a in top_authors else 'OTHER'
        for a in authors
    ]
)

author_mlb = MultiLabelBinarizer(sparse_output=True)
X_authors = author_mlb.fit_transform(meta['author_list_filtered'])
X_authors = normalize(X_authors, axis=1)

# ------------------------------------------
# Language one-hot
lang_encoder = OneHotEncoder(
    sparse_output=True,
    handle_unknown='ignore'
)
X_lang = lang_encoder.fit_transform(meta[['language_code']])
X_lang = normalize(X_lang, axis=1)

# ------------------------------------------
# Numeric features
# Handle missing values
meta['average_rating_missing'] = meta['average_rating'].isna().astype(int)
meta['average_rating'] = meta['average_rating'].fillna(meta['average_rating'].mean())
meta['log_ratings_count'] = np.log1p(meta['ratings_count'].fillna(0))

scaler = MinMaxScaler()
X_numeric = scaler.fit_transform(meta[['average_rating', 'log_ratings_count', 'average_rating_missing']])
X_numeric = normalize(X_numeric, axis=1)
X_numeric = csr_matrix(X_numeric.astype(np.float32))

# ------------------------------------------
# Combine features
print("\nCombining features...")

# Use any weights as starting point. These can be adjusted by optimization later.
"""
BLOCK_WEIGHTS = {
    "title": 0.53075,
    "tags": 0.98150,
    "authors": 0.74466,
    "lang": 0.17989,
    "numeric": 0.00948
}
"""
BLOCK_WEIGHTS = {
    "title": 1.0,
    "tags": 1.0,
    "authors": 1.0,
    "lang": 1.0,
    "numeric": 1.0
}
# Multiply each block by its corresponding weight
X_title *= BLOCK_WEIGHTS["title"]
X_tags *= BLOCK_WEIGHTS["tags"]
X_authors *= BLOCK_WEIGHTS["authors"]
X_lang *= BLOCK_WEIGHTS["lang"]
X_numeric *= BLOCK_WEIGHTS["numeric"]


X_final = hstack([
    X_title,
    X_tags,
    X_authors,
    X_lang,
    X_numeric
]).tocsr()

print("Final feature shape:", X_final.shape)

# ------------------------------------------
# Save sparse matrix + metadata
save_npz(X_out_path, X_final)

np.savez_compressed(
    meta_out_path,
    book_ids=meta['book_id'].values.astype(np.int32),
    ids=meta['id'].values.astype(np.int32),
    tag_vocab=vectorizer.get_feature_names_out(),
    author_vocab=author_mlb.classes_,
    language_vocab=lang_encoder.categories_[0],
    block_sizes=np.array([
        X_title.shape[1],
        X_tags.shape[1],
        X_authors.shape[1],
        X_lang.shape[1],
        X_numeric.shape[1]
    ]),
    block_weights=np.array([
        BLOCK_WEIGHTS["title"],
        BLOCK_WEIGHTS["tags"],
        BLOCK_WEIGHTS["authors"],
        BLOCK_WEIGHTS["lang"],
        BLOCK_WEIGHTS["numeric"]
    ])
)


print("\nFeature dimensions:")
print(f"Title: {X_title.shape[1]}")
print(f"Tags: {X_tags.shape[1]}")
print(f"Authors: {X_authors.shape[1]}")
print(f"Language: {X_lang.shape[1]}")
print(f"Numeric: {X_numeric.shape[1]}")

print("\nSaved:")
print(f"- {X_out_path}")
print(f"- {meta_out_path}")
