import pandas as pd
import os
import kagglehub
import shutil

#------------------------------------------
# Paths to dataset
cwd = os.getcwd()
target_dir = os.path.join(cwd, "dataset")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"Created directory: {target_dir}")

print("Downloading dataset...")
download_path = kagglehub.dataset_download("zygmunt/goodbooks-10k")

print(f"Moving files to {target_dir}...")
for filename in os.listdir(download_path):
    source_file = os.path.join(download_path, filename)
    target_file = os.path.join(target_dir, filename)
    
    shutil.copy2(source_file, target_file)

#------------------------------------------
books_path = os.path.join(cwd, "dataset/books.csv")
book_tags_path = os.path.join(cwd, "dataset/book_tags.csv")
tags_path = os.path.join(cwd, "dataset/tags.csv")
ratings_path = os.path.join(cwd, "dataset/ratings.csv")

#------------------------------------------
# Load CSVs
books = pd.read_csv(books_path)
book_tags = pd.read_csv(book_tags_path)
tags = pd.read_csv(tags_path)
rating = pd.read_csv(ratings_path)

#------------------------------------------
# Display basic information about the datasets
print("Books Dataset Info:")
print(books.info())
print("\nBook Tags Dataset Info:")
print(book_tags.info())
print("\nTags Dataset Info:")
print(tags.info())
print("\nRatings Dataset Info:")
print(rating.info())
print("------------------------------------------")

#------------------------------------------
print("\nBooks Dataset Sample:")
print(books.head(10))
print("\nBook Tags Dataset Sample:")
print(book_tags.head())
print("\nTags Dataset Sample:")
print(tags.head())
print("\nRatings Dataset Sample:")
print(rating.head())
print("------------------------------------------")
