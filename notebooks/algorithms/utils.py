import numpy as np
import re
from collections import Counter


def find_representative_tags(
    tree_builder,
    tags,
    n_centroids=100,
    similarity_metric="euclidean",
):
    if similarity_metric not in ["euclidean", "cosine"]:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    root_centroids = np.asarray(tree_builder.tree["root"]["centroids"])

    # Handle case where actual clusters might be fewer than requested n_centroids
    actual_n_centroids = root_centroids.shape[0]

    cluster_means = {}
    for i in range(actual_n_centroids):
        cluster_means[i] = root_centroids[i]

    all_embeddings = tree_builder.norm_embeddings
    N = len(tags)
    D = all_embeddings.shape[1]

    cluster_idxs = np.full(N, -1, dtype=int)
    pattern_root_child = re.compile(r"root_c(\d+)")

    for i in range(N):
        # tag_to_path gives the leaf path, e.g., "root_c5_c2" or "root_c0"
        tag_path = tree_builder.tag_to_path.get(i, "")

        # We extract the first number after root_c to identify the Level 1 cluster
        m = pattern_root_child.search(tag_path)
        if m:
            cluster_idxs[i] = int(m.group(1))

    # Initialize result dictionary
    h1_t = {idx: None for idx in range(n_centroids)}

    # Loop through clusters to find representatives
    for idx in range(n_centroids):
        if idx not in cluster_means:
            continue

        mean = cluster_means[idx].reshape(1, D)

        mask = cluster_idxs == idx
        if not mask.any():
            continue

        true_indices = np.nonzero(mask)[0]
        candidate_embs = all_embeddings[mask]  # (m, D)

        if similarity_metric == "euclidean":
            diffs = candidate_embs - mean
            dists = np.linalg.norm(diffs, axis=1)
            argmin = int(np.argmin(dists))
            closest_dist = float(dists[argmin])
        else:  # cosine
            mean_norm = mean / (np.linalg.norm(mean) + 1e-10)
            cand_norm = candidate_embs / (
                np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
            )

            cosine_sims = np.dot(cand_norm, mean_norm.T).squeeze()
            if cosine_sims.ndim == 0:
                cosine_sims = np.array([cosine_sims])  # single item case

            argmin = int(np.argmax(cosine_sims))
            closest_dist = float(1.0 - cosine_sims[argmin])

        chosen_tag_idx = true_indices[argmin]
        closest_tag_name = tags[chosen_tag_idx]
        closest_tag_emb = all_embeddings[chosen_tag_idx]

        cluster_tag_names = [tags[i] for i in true_indices]

        top_popular_tags = []
        tag_counter = Counter(cluster_tag_names)
        for tag, _ in tag_counter.most_common(5):
            top_popular_tags.append(tag)

        h1_t[idx] = {
            "closest_tag": closest_tag_name,
            "closest_dist": closest_dist,
            "cluster_mean": cluster_means[idx],
            "tag_embedding": closest_tag_emb,
            "cluster_names": cluster_tag_names,
            "cluster_size": len(cluster_tag_names),
            "top_5_popular_tags": top_popular_tags,
        }

    return h1_t
