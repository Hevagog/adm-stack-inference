#  Introduction

This project investigates the application of advanced deep learning and data mining methodologies to high-dimensional text embeddings derived from StackExchange (StackOverflow) questions. The analysis focuses on transforming the inherently complex, sparse structure of raw textual and tagging data into dense, manageable vector spaces suitable for sophisticated tasks.

The primary technical challenge addressed was managing the extreme cardinality and high dimensionality of the data: specifically, compressing over 22,000 unique question tags into a meaningful, lower-dimensional space, and then leveraging 4096-dimensional embeddings for both multi-label classification (tag prediction) and high-variance regression (question score prediction).

The core objectives pursued throughout this research were:

1.  **Data Exploration and Feature Engineering (EDA):** To quantify the sparsity and distribution characteristics of key features such as tag frequency, temporal metrics related to answer acceptance, and question scoring.
2.  **Dimensionality Reduction:** To overcome computational limits and improve model tractability by intelligently reducing the space of 4096-dimensional embeddings and the semantic space of 22,753 unique tags.
3.  **Tag Classification:** To design and evaluate neural network architectures capable of predicting question categories based on textual input from the title and body embeddings.
4.  **Score Regression:** To assess the intrinsic predictability of question quality (measured by score) directly from the learned semantic embeddings.

The methodology relied heavily on the `qwen3-embedding:8b` model for vector representation and incorporated robust machine learning techniques suchibilities as **Recursive Spherical K-Means** for unsupervised clustering, and **Asymmetric Loss (ASL)** and **Cross-Attention** mechanisms for enhanced deep learning performance.

