#  Introduction

This project investigates how deep learning and embedding methods can be applied in analysis and prediction of properties derived from StackExchange (StackOverflow) questions. Using the StackExchenge API, we collected a dataset of 100,000 questions together with their metadata, textual context, and tag information. The analysis focuses on transforming the high-dimensional, sparse and highly imbalance data into representation suitable for learning.

A major technical challenge addressed was managing the extreme cardinality and high dimensionality of the data: specifically, compressing over 22,000 unique question tags into a meaningful, lower-dimensional space, and then leveraging 4096-dimensional embeddings for both multi-label classification (tag prediction) and high-variance regression (question score prediction). To accomplish this, we embedded all tags, titles and question bodies using the `qwen3-embedding:8b` model. For tags, we have applied a series of dimensionality reduction and clustering strategies. After evaluating UMAP [@mcinnes2020umapuniformmanifoldapproximation] + HDBSCAN [@hdbscan], Birch [@Zhang1997], and agglomerative approaches, we adopted **Recursive Spherical K-Means**, which produced 100 tag centroids that preserve coverage of all original tags.

The core objectives pursued throughout this project were:

1.  **Data Exploration and Feature Engineering (EDA):** To quantify the sparsity and distribution characteristics of key features such as tag frequency, temporal metrics related to answer acceptance, and question scoring.
2.  **Dimensionality Reduction:** To overcome computational limits and improve model tractability by intelligently reducing the space of 4096-dimensional embeddings and the semantic space of 22,753 unique tags.
3.  **Tag Classification:** To design and evaluate neural network architectures capable of predicting question categories based on textual input from the title and body embeddings.
4.  **Score Regression:** To assess the intrinsic predictability of question quality (measured by score) directly from the learned semantic embeddings.

The methodology relied heavily on the `qwen3-embedding:8b` model for vector representation and incorporates robust machine learning techniques such as **Recursive Spherical K-Means** for unsupervised clustering, and **Asymmetric Loss (ASL)** and **Cross-Attention** mechanisms for enhanced deep learning performance.

