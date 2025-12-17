# StackOverflow Question Analysis & Inference

This project implements advanced data mining and deep learning techniques to analyze StackOverflow questions. It focuses on semantic tag clustering, multi-label tag prediction, and question score regression using embeddings generated via `qwen3-embedding:8b`.

## Project Overview

The dataset consists of **100,000 questions** from StackExchange (StackOverflow), including metadata such as titles, bodies, tags, and scores.

Key objectives:
1.  **Exploratory Data Analysis (EDA)**: Understanding distributions of tags, scores, and answer acceptance.
2.  **Tag Space Reduction**: Clustering over 22,000 unique tags into manageable semantic centroids.
3.  **Tag Prediction**: Developing Neural Networks to predict tag clusters from text embeddings.
4.  **Score Prediction**: Estimating question quality (score) based on content.

## Methodology

### 1. Embeddings & Dimensionality Reduction
- **Embeddings**: Generated using `qwen3-embedding:8b`, resulting in 4096-dimensional vectors for titles and question bodies.
- **Clustering**: Addressed the high cardinality of tags (22,753 unique tags) using **Recursive Spherical K-Means** [@Schubert_2021].
    - Reduced tags to **100 semantic centroids**.
    - Orphan tags were assigned to the nearest centroid based on cosine similarity.

### 2. Tag Prediction (Multi-Label Classification)
We compared several approaches to predict the tag centroids for a given question:

- **XGBoost Baseline**: Gradient boosting with class weights to handle imbalance.
- **Dual-Stream Fusion Network (DSF)**: A deep learning architecture processing Title and Body embeddings in separate streams.
    - **Fusion Mechanisms**:
        - **MHSA**: Multi-Head Self-Attention fusion inspired by fake news detection models [@YANG2024112358].
        - **Cross-Attention**: Using Title embeddings to query Body embeddings.
    - **Loss Function**: **Asymmetric Loss (ASL)** [@benbaruch2021asymmetriclossmultilabelclassification] was crucial for handling the extreme class imbalance of tags.
    - **Regularization**: Implemented Manifold Mixup and Dropout.

### 3. Score Prediction
We attempted to regress the question score directly from embeddings using:
- Single-stream MLPs (Title or Body only).
- Dual-stream architectures (DSF-MHSA, DSF-CrossAttn).

## Results

### Tag Prediction Performance
The Dual-Stream Fusion network with Cross-Attention achieved the best results, outperforming the XGBoost baseline.

| Model | F1 Score (Weighted) | Notes |
|-------|---------------------|-------|
| **XGBoost** | 0.6882 | Baseline, computationally expensive on dense embeddings. |
| **Baseline MLP** | 0.6790 | Simple Multi-Layer Perceptron. |
| **DSF (MHSA Fusion)** | 0.7110 | Dual-stream with Self-Attention. |
| **DSF (Cross-Attention)** | **0.7196** | Best performance, Title queries Body. |

### Score Prediction Performance
Score prediction proved to be a difficult task using embeddings alone.

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| **DSF-MHSA** | **0.199** | 134.24 |
| **DSF-CrossAttn** | 0.186 | 135.27 |
| **Body MLP** | 0.158 | 137.57 |

## Setup

Create a virtual environment and install dependencies:
```bash
just install
```

## References

- <a id="Schubert_2021">**[@Schubert_2021]**</a>: Schubert, E., Lang, A., & Feher, G. (2021). *Accelerating Spherical k-Means*. In Similarity Search and Applications (pp. 217–231). Springer International Publishing. [https://doi.org/10.1007/978-3-030-89657-7_17](https://doi.org/10.1007/978-3-030-89657-7_17)
- <a id="benbaruch2021asymmetriclossmultilabelclassification">**[@benbaruch2021asymmetriclossmultilabelclassification]**</a>: Ben-Baruch, E., Ridnik, T., Zamir, N., Noy, A., Friedman, I., Protter, M., & Zelnik-Manor, L. (2021). *Asymmetric Loss For Multi-Label Classification*. arXiv preprint arXiv:2009.14119. [https://arxiv.org/abs/2009.14119](https://arxiv.org/abs/2009.14119)
- <a id="YANG2024112358">**[@YANG2024112358]**</a>: Yang, Y., Liu, J., Yang, Y., & Cen, L. (2024). *Dual-stream fusion network with multi-head self-attention for multi-modal fake news detection*. Applied Soft Computing, 167, 112358. [https://doi.org/10.1016/j.asoc.2024.112358](https://doi.org/10.1016/j.asoc.2024.112358)
