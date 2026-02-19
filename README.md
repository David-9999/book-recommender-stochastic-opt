# 📚 Stochastic Book Recommender

A **hybrid book recommendation system** that optimizes feature weights using **Stochastic Hill Climbing (SHC)** to maximize recommendation accuracy (**Hit@10**).  
It combines **BERT (RoBERTa) embeddings** for semantic title analysis with **TF-IDF** for user-generated tags, achieving a **+33% performance boost** over uniform weight baselines.

---

## ⚡ Key Features

- **Hybrid Semantic Search**: Combines deep learning embeddings for titles with TF-IDF for tags.
- **Automated Weight Tuning**: Learns which features matter most (Tags & Authors prioritized).
- **Custom Scoring**: Non-linear rating function rewards 5-star content and penalizes disliked books.
- **High Reproducibility**: Sequentially numbered scripts make running the pipeline straightforward.

---

## 🏗 Project Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_load_data.py` | Download & initialize Goodreads-10k dataset |
| 2 | `02_preprocessing.py` | Clean & sanitize tags; remove junk patterns & map synonyms |
| 3 | `03_merge_data.py` | Merge metadata, authors, and cleaned tags |
| 4 | `04_feature_build.py` | Generate features: 768-dim BERT embeddings, TF-IDF tags, author/language encoding |
| 5 | `05_knn_model_user.py` | Core recommendation engine using weighted KNN aggregation |
| 6 | `06_knn_model_user_weights_LOO_grid.py` | Initial weight optimization with Grid Search (sample of 2,000 users) |
| 7 | `07_knn_model_user_weights_LOO_SHC.py` | Advanced weight tuning with **Stochastic Hill Climbing** to maximize Hit@10 |

---

## 📊 Results

| Approach | Optimization | Hit@10 | Improvement |
| :--- | :--- | :--- | :--- |
| Uniform Weights | Manual Baseline | 0.1315 | - |
| **Optimized Weights** | **Grid Search + SHC** | **0.1750** | **+33%** |

---

## ⚙️ Requirements

- Python 3.8+
- `scikit-learn`, `sentence-transformers` (`all-distilroberta-v1`)
- `pandas`, `numpy`, `scipy`, `kagglehub`


🔮 Future Improvements

- **Refined Validation**: Switch from random LOO to "Highest-Rated-Out" for better prediction of favorite books.
- **Advanced Metrics**: Add NDCG / MRR for ranking-aware evaluation.
- **Enhanced NLP Pipeline**: Use NER or LLM-based summarization for cleaner tag features.
- **Temporal Weighting**: Prioritize recent user interactions for evolving taste modeling.

---
*Developed by David Vrba as part of 2-INF-150 Machine Learning course.*
