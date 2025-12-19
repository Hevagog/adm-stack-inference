# The Methodologies

The path to generating robust predictive models required significant methodological investment, particularly in handling the immense size and complexity of the embedded text data.

## Embedding Generation

To transform the textual data (titles, tags, and question bodies) into numerical feature representations, we utilized **`qwen3-embedding:8b`**, an open-source model capable of generating **4096-dimensional vectors** [@qwen_embedding].
Textual data (titles and question bodies) was transformed into **4096-dimensional vectors** using the open-source embedding model **`qwen3-embedding:8b`**.

We implemented two distinct embedding strategies:

### 1. Global Document Embedding (Baseline)

- This approach served as the baseline due to its simplicity and relative computational speed.
- Embed each question body, title and tag individually into an embedding vector.
- This method assumes that the semantic context of the entire document can be effectively compressed into a single 4096-dimensional vector (using `float64` precision) without significant information loss.
- Computation required approximately 6 hours using the `ollama` library [@ollama]. The resulting dataset occupied 3.3 GB of storage.

### 2. Sequential Token Embedding

- To address potential information loss in the baseline approach, we hypothesized that a single vector might fail to capture complex dependencies in longer texts.
- Instead of pooling the text into one vector, we maintained a sequence of embeddings to preserve token-level knowledge. We defined a fixed sequence length of 4 tokens for the title and 32 tokens for the body, resulting in a distinct embedding vector for each token.
- Implementation changes:
    - Due to the limitations of `ollama` in the terms of appropriate tokenizer for our model, as well as inefficient resource management during embedding, we've switched to *Hugging Face* `transformers` library.
    - Because of the exponential increase in data size and compute time, we reduced the floating-point precision from `float64` $\to$ `float32`.
- These optimizations reduced the estimated compute time from 40 hours to approximately 13 hours.
- The resulting embeddings required 27 GB of space. Due to memory constraints preventing the dataset from being loaded entirely into RAM, we utilized the Hierarchical Data Format (HDF5) for efficient storage and access [@hdf5].

## Embedding Analysis

After the embedding phase, we wanted to verify the validy of the topology and density of the resulting embedding space. Specifically, we wanted to verify that the embeddings captured sufficient semantic overlap between questions to facilitate meaningful clustering. 

To measure this, we analyzed the *Nearest-Neighbour Cosine Similarity*. Using a
subset of the question embeddings, we performed the following steps:

- Normalized the embeddings (the resulting embeddings from `ollama`
theoretically should be normalized, but it's better to normalize it anyway,
since it does not affect the data).
- We utilized the `NearestNeighbors` algorithm (from `scikit-learn`) to locate the closest non-identical neighbor ($k=1$) for each data point.
- We calculated the cosine similarity for these pairs, defined as $1 - \text{cosine\_distance}$.

![Nearest-Neighbour cosine similarity distribution](img/03/tags-nn-cosine-sim.png){#fig:tags-nn-cosine-sim width=60%}

As we can see in Figure [@fig:tags-nn-cosine-sim], the distribution of nearest-neighbor similarities is approximately Gaussian and centered around 0.65.

- The lack of data points near 0.0 indicates that very few questions are "isolated" in the vector space; almost every question has a semantically related counterpart.
- The unimodal distribution suggests a well-structured manifold where local neighborhoods are consistent. This confirms that the embedding model (`qwen3-embedding`) successfully mapped semantically similar questions to adjacent regions in the high-dimensional space, providing a strong foundation for the subsequent clustering phases.

# Dimensionality Reduction and Clustering

The high dimensionality (4096 dimensions) and high cardinality (22,753 unique tags) presented immediate barriers to traditional clustering techniques, not to mention classification task. As we still wanted to retain semantic richness in the tag space, we explored various dimensionality reduction and clustering strategies.

### Initial tries (HDBSCAN and UMAP)

1.  **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)** was initially rejected because memory requirements exceeded the available 64GB of RAM.
2.  **UMAP (Uniform Manifold Approximation and Projection)** was adopted as a non-linear dimension reduction technique (e.g., reducing to 5 components), coupled with HDBSCAN clustering. Optimization of UMAP and HDBSCAN parameters was attempted using **Optuna**, guided by unsupervised metrics like the Caliński-Harabasz index, Silhouette score, and Cluster persistence.

This combined approach failed to yield coherent results. The clusters were highly imbalanced, with large numbers of points assigned to a single noise cluster or small, trivial clusters (visible in [@fig:naive-umap] and [@fig:labels-umap-hdbscan]). This failure was theorized to stem from the violation of UMAP’s underlying assumption that data should be uniformly distributed on a Riemannian manifold, as dense embeddings often exhibit varying local geometry.

![Naiive UMAP reducing tag embeddings to 2 components](img/03/naive-umap.png){#fig:naive-umap width=60%}

![Histogram of labels resulting from UMAP reduction coupled with HDBSCAN](img/03/labels-umap-hdbscan.png){#fig:labels-umap-hdbscan width=60%}


### Recursive Spherical K-Means Clustering

To achieve a stable reduction of the tag space to **100 semantic centroids**, a **Recursive Embedding and Clustering (REAC)** approach, inspired by industry techniques, was implemented.

1.  **Tag Filtration:** Tags with a low frequency (threshold $< 100$) were initially removed, resulting in a dataset reduced to **411 unique tags** across 90,205 records.
2.  **Clustering:** **Recursive Spherical K-Means** was employed on these 411 tags. Spherical K-Means is particularly advantageous for textual embeddings as it optimizes distance based on **cosine similarity**. This process successfully yielded the target 100 centroids. Unlike in the case with HDBSCAN, this time the resulting clusters showed a stable distribution of tags per centroid [@fig:cluster-sizes-per-centroid] and focused on general ideas connecting the grouped tags, e.g.: the cluster that could have been described as "databases" included: `database`, `mongodb`, `sql`, `postgresql`, `mysql`, etc..
3.  **Orphan Assignment:** The 9,787 infrequent "orphan tags" were assigned to their nearest centroid based on cosine similarity, ensuring that all 22,753 original tags were mapped to the final 100 classes.

![Distribution of Sizes of Clusters found using Recursive Spherical K-Means](img/03/cluster-sizes-per-centroid.png){#fig:cluster-sizes-per-centroid width=60%}
