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
    h1_t = {centroid_idx: [] for centroid_idx in range(n_centroids)}

    cluster_means = {}
    pattern_cluster = re.compile(r"_c(\d+)$")
    for key in tree_builder.tree["root"]["children"]:
        m = pattern_cluster.search(key)
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx < n_centroids:
            centroids = np.asarray(tree_builder.tree[key]["centroids"])
            if centroids.size == 0:
                continue
            cluster_means[idx] = np.mean(centroids, axis=0)

    N = len(tags)
    tag_paths = [None] * N
    cluster_idxs = np.full(N, -1, dtype=int)
    embeddings = []

    pattern_any_c = re.compile(r"_c(\d+)")

    for i in range(N):
        tag_path = tree_builder.tag_to_path[i]
        tag_paths[i] = tag_path

        m = pattern_any_c.search(tag_path)
        if m:
            cluster_idxs[i] = int(m.group(1))
        else:
            cluster_idxs[i] = -1  # ignore

        emb = np.asarray(tree_builder.tree[tag_path]["centroid"])
        embeddings.append(emb)

    embeddings = np.stack(embeddings)  # shape (N, D)
    D = embeddings.shape[1]

    h1_t = {idx: None for idx in range(n_centroids)}

    for idx in range(n_centroids):
        if idx not in cluster_means:
            continue

        mean = cluster_means[idx].reshape(1, D)  # (1, D)
        mask = cluster_idxs == idx
        if not mask.any():
            continue

        candidate_embs = embeddings[mask]  # (m, D)

        if similarity_metric == "euclidean":
            diffs = candidate_embs - mean  # broadcasting (m, D)
            dists = np.linalg.norm(diffs, axis=1)  # (m,)
            argmin = int(np.argmin(dists))
            closest_dist = float(dists[argmin])
        else:  # cosine similarity
            mean_norm = mean / (np.linalg.norm(mean) + 1e-10)  # (1, D)
            candidate_norms = candidate_embs / (
                np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
            )  # (m, D)
            cosine_sims = np.dot(candidate_norms, mean_norm.T).squeeze()  # (m,)
            argmin = int(np.argmax(cosine_sims))
            closest_dist = float(1.0 - cosine_sims[argmin])

        true_indices = np.nonzero(mask)[0]
        chosen_tag_idx = true_indices[argmin]
        closest_tag_name = tags[chosen_tag_idx]
        closest_tag_emb = embeddings[chosen_tag_idx]

        cluster_tag_indices = true_indices
        cluster_tag_names = [tags[i] for i in cluster_tag_indices]

        top_popular_tags = []
        tag_counter = Counter(cluster_tag_names)
        for tag, _ in tag_counter.most_common(5):
            top_popular_tags.append(tag)

        h1_t[idx] = {
            "closest_tag": closest_tag_name,
            "closest_dist": closest_dist,
            "cluster_mean": cluster_means[idx],
            "tag_embedding": closest_tag_emb,
            "cluster_size": len(cluster_tag_names),
            "top_5_popular_tags": top_popular_tags,
        }

    return h1_t
