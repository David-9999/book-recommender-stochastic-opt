# Stochastic Book Recommender

A hybrid book recommendation system that optimizes feature weights using **Stochastic Hill Climbing** to maximize recommendation accuracy (**Hit@10**). 

This project integrating **BERT (RoBERTa) embeddings** for semantic title analysis and **TF-IDF** for user-generated tags, achieving a +33% performance boost over uniform weight baselines.

## Project Structure

The pipeline is organized into sequentially numbered scripts to ensure reproducibility:

1.  **`01_load_data.py`**: Downloads and initializes the Goodreads-10k dataset.
2.  **`02_preprocessing.py`**: Performs tag sanitization, including junk pattern removal, synonym mapping, and frequency-based pruning.
3.  **`03_merge_data.py`**: Integrates processed metadata, authors, and cleaned tags into a unified master dataset.
4.  **`04_feature_build.py`**: Constructs the high-dimensional feature space. 
    * Generates 768-dim **BERT embeddings** for titles.
    * Applies **TF-IDF** to sanitized tags.
    * Encodes authors (Multi-Label Binarizer) and languages (One-Hot).
5.  **`05_knn_model_user.py`**: The core recommendation engine. Implements the weighted user preference aggregation logic and the non-linear rating weight function.
6.  **`06_knn_model_user_weights_LOO_grid.py`**: Initial optimization phase using **Grid Search** to find a high-performing weight baseline across a sample of 2,000 users.
7.  **`07_knn_model_user_weights_LOO_SHC.py`**: Advanced optimization using **Stochastic Hill Climbing**. Perturbs feature weights to maximize the Leave-One-Out (LOO) Hit@10 metric.



## Key Features

* **Hybrid Semantic Search**: Combines deep learning latent spaces (Title) with TF-IDF (Tags).
* **Automated Weight Tuning**: Learns that "not all features are created equal", automatically prioritizing Tags and Authors over less predictive numerical metadata.
* **Custom Scoring**: A non-linear transformation function $w(r)$ that aggressively rewards 5-star ratings and penalizes disliked content.

## Results

| Approach | Optimization | Hit@10 | Improvement |
| :--- | :--- | :--- | :--- |
| Uniform Weights | Manual Baseline | 0.1315 | - |
| **Optimized Weights** | **Grid Search + SHC** | **0.1750** | **+33%** |

## Requirements

* Python 3.8+
* `scikit-learn`
* `sentence-transformers` (all-distilroberta-v1)
* `pandas`, `numpy`, `scipy`, `kagglehub`

##Future Improvements
* **Refined Validation Strategy**: Instead of random Leave-One-Out (LOO), transition to a "Highest-Rated-Out" approach. This ensures the model is optimized to predict books a user is statistically likely to love, rather than just any book they happened to interact with.

* **Advanced Evaluation Metrics**: Incorporate NDCG (Normalized Discounted Cumulative Gain) or MRR (Mean Reciprocal Rank) to measure not just if a book was recommended, but how high it ranked in the top-10 list.

* **Enhanced NLP Pipeline**: Implement more aggressive tag denoising using NER (Named Entity Recognition) or LLM-based summarization to filter out low-signal user tags that current frequency-based pruning might miss.

* **Temporal Weighting**: Introduce a "Recency Decay" factor to give higher priority to a user's more recent reading habits, reflecting evolving tastes over time.

---
*Developed by David Vrba as part of 2-INF-150 Machine Learning course.*
