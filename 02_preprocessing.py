import pandas as pd
import re

# ------------------------------------------
# Load CSV
tags_df = pd.read_csv("dataset/tags.csv")
output_path = "dataset/tags_preprocessed.csv"

# ------------------------------------------
# Months to remove
months = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

# ------------------------------------------
# Junk patterns
junk_regex = [
    r"\bto[\s-]?read\b",
    r"\bhave[\s-]?read\b",
    r"\bcurrently[\s-]?reading\b",
    r"\bread\b",
    r"\bdefault\b",
    r"\bfavorite(s)?\b",
    r"\bkindle\b",
    r"\be[-\s]?book\b",
    r"\bebook\b",
    r"\baudio(book)?\b",
    r"\bpaperback\b",
    r"\bhardcopy\b",
    r"\blibrary\b",
    r"\bavailable\b",
    r"\bown(ed)?\b",
    r"\bwant(ed)?\b",
    r"\bno[-\s]?copy\b",
    r"\bwatched\b",
    r"\bmovie\b",
    r"\bfilm\b",
    r"\btv\b",
    r"\bgift\b",
    r"\bbuy\b",
    r"\bbought\b",
    r"\bgiving\b",
    r"\breceived\b",
    r"\bwish[-\s]?list\b",
    r"\bwishlist\b",
    r"\bauthor\b",
    r"\bbooks\b",
]

# ------------------------------------------
# Synonym mapping
synonym_patterns = {
    "young-adult": [
        r"\bya\b",
        r"young[\s-]?adult(s)?",
        r"youngadult"
    ],
    "science-fiction": [
        r"sci[\s-]?fi",
        r"science[\s-]?fiction",
        r"\bsf\b"
    ],
    "fantasy": [
        r"\bfantasy\b",
        r"\bfantasía\b",
        r"\bfantastique\b"
    ],
    "mystery": [r"\bmystery\b"],
    "romance": [r"\bromance\b"],
    "historical-fiction": [r"historical[\s-]?fiction"],
    "non-fiction": [r"non[\s-]?fiction", r"nonfiction"],
    "graphic-novel": [
        r"graphic[\s-]?novel", 
        r"comic[\s-]?book"
    ],
    "horror": [r"\bhorror\b"],
    "thriller": [r"\bthriller\b"],
    "biography": [
        r"\bbiography\b", 
        r"\bmemoir\b"
    ],
    "self-help": [r"self[\s-]?help"],
    "classics": [
        r"\bclassic(s)?\b", 
        r"\bclàssics\b"
    ],
    "dystopian": [r"\bdystopian\b", r"dystopia"],
    "adventure": [r"\badventure\b"],
    "children": [
        r"\bchildren\b", 
        r"\bchildrens\b", 
        r"\bkids\b"
    ],
    "poetry": [r"\bpoetry\b", r"\bpoem\b"],
    "history": [r"\bhistory\b"],
    "science": [r"\bscience\b"],
    "philosophy": [r"\bphilosophy\b"],
    "religion": [
        r"\breligion\b", 
        r"\bspirituality\b"
    ],
    "travel": [r"\btravel\b"],
    "cooking": [r"\bcooking\b", r"\bcookbook\b"],
    "music": [r"\bmusic\b", r"música"],
    "business": [
        r"\bbusiness\b", 
        r"\bentrepreneurship\b"
    ],
    "comics": [r"\bcomics\b"]
}

# ------------------------------------------
# Clean leading dashes / prefixes
def clean_leading(tag):
    tag = str(tag).strip().lower()
    tag = tag.lstrip("-")
    if tag.startswith("a-a-"):
        tag = tag[4:]
    elif tag.startswith("aa-"):
        tag = tag[3:]
    elif tag.startswith("a-"):
        tag = tag[2:]
    return tag

# ------------------------------------------
# Validation function
def is_valid_tag(tag):
    if not tag:
        return False

    # Remove numeric / code-like
    if re.fullmatch(r"[-\d]+", tag):
        return False

    # Explicit junk states
    if any(x in tag for x in ["not-finish", "not-finished", "on-my", "paused"]):
        return False

    # Junk regex
    if any(re.search(p, tag) for p in junk_regex):
        return False

    # Month names
    if any(month in tag for month in months):
        return False

    # Allowed characters: letters, spaces, hyphens (keep other languages!)
    for c in tag:
        if not (c.isalpha() or c in [" ", "-"]):
            return False

    return True

# ------------------------------------------
# Synonym mapping
def map_synonyms(tag):
    for canonical, patterns in synonym_patterns.items():
        for pattern in patterns:
            if re.search(pattern, tag):
                return canonical
    return tag

# ------------------------------------------
# Apply preprocessing
tags_df["tag_name"] = tags_df["tag_name"].apply(clean_leading)

tags_df_clean = tags_df[
    tags_df["tag_name"].apply(is_valid_tag)
].copy()

tags_df_clean["tag_name"] = tags_df_clean["tag_name"].apply(map_synonyms)

# Normalize hyphens and spaces
tags_df_clean["tag_name"] = tags_df_clean["tag_name"].str.replace(r"-+", "-", regex=True)
tags_df_clean["tag_name"] = tags_df_clean["tag_name"].str.replace(r"\s+", "-", regex=True)
tags_df_clean["tag_name"] = tags_df_clean["tag_name"].str.rstrip("-")

# Remove single-character tags
tags_df_clean = tags_df_clean[tags_df_clean["tag_name"].str.len() > 1]

# ------------------------------------------
# Frequency-based pruning.
MIN_TAG_FREQ = 1

tag_counts = tags_df_clean["tag_name"].value_counts()
valid_tags = tag_counts[tag_counts >= MIN_TAG_FREQ].index

tags_df_clean = tags_df_clean[tags_df_clean["tag_name"].isin(valid_tags)]

# ------------------------------------------
# Drop duplicates
tags_df_clean = tags_df_clean.drop_duplicates(subset=["tag_id", "tag_name"]).reset_index(drop=True)

# ------------------------------------------
# Save outputs
tags_df_clean.to_csv(output_path, index=False)
print(f"Preprocessed tags saved to {output_path}")

"""
unique_tags = tags_df_clean["tag_name"].dropna().unique()
with open("unique_tags.txt", "w") as f:
    for tag in unique_tags:
        f.write(f"{tag}\n")
"""