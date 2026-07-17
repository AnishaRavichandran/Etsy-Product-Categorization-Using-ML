# Product Categorization for E-Commerce using Logistic Regression

A machine learning project that classifies e-commerce products into hierarchical categories (top-level and bottom-level) using TF-IDF text features and logistic regression.

## Overview

Product categorization is a core problem for e-commerce platforms, powering search, recommendations, and inventory management. This project builds a two-stage classification pipeline that predicts:

- **Top-level (coarse) category** — e.g. Electronics, Clothing
- **Bottom-level (fine-grained) category** — the more specific sub-category within each top-level group

Rather than reaching for deep learning, this project shows that a well-preprocessed, feature-engineered logistic regression pipeline can achieve strong, interpretable results at low computational cost.

## Results

| Task | F1 Score |
|---|---|
| Top-level category prediction | 0.8619 |
| Bottom-level category prediction | 0.8093 |

- Top-level categories consistently outperform bottom-level categories, reflecting the reduced granularity and clearer distinctions between major product types.
- Lemmatization and stop word removal gave the largest single improvement to macro-F1 (7–10%).
- Misclassifications mostly occur between semantically related categories (e.g. within the same top-level group).

## Approach

**1. Preprocessing**
- Lowercasing and case normalization
- Stop word removal
- Tokenization
- Lemmatization
- Emoji / special character removal
- Combining title, description, and tags into a single text field

**2. Feature Extraction**
- TF-IDF vectorization (unigrams + bigrams)
- Vocabulary size capped for dimensionality control
- Minimum document frequency threshold to filter noisy terms

**3. Modeling**
- Two-stage hierarchical classification:
  1. A coarse classifier predicts the top-level category
  2. A fine classifier predicts the bottom-level category, trained within each coarse group
- Multinomial logistic regression with:
  - L2 regularization
  - Class weight balancing (to handle category imbalance)
  - One-vs-rest multi-class strategy

**4. Evaluation**
- Per-class F1-score
- Macro-average F1
- Confusion matrices for error analysis
- UMAP visualizations of predicted top and bottom categories



## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (TF-IDF, LogisticRegression, pipelines)
- NLTK (stopwords, lemmatization, tokenization)
- Matplotlib (visualizations)
- UMAP (dimensionality reduction for category visualization)


