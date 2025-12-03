import numpy as np
from sklearn.cluster import KMeans
from .spherical_kmeans import SphericalKMeans


class DirectHierarchicalLabelTree:
    def __init__(
        self,
        tag_embeddings,
        tag_names,
        branching_factors=[100, 100],
        use_spherical_clusters=True,
    ):
        """
        Args:
            branching_factors: hierarchy structure [level1_clusters, level2_clusters]
        """
        self.tag_embeddings = tag_embeddings
        self.tag_names = tag_names
        self.branching_factors = branching_factors
        self.tree = {}
        self.tag_to_path = {}
        self._use_spherical = use_spherical_clusters
        self.norm_embeddings = tag_embeddings.astype(np.float32)
        self.norm_embeddings = self.norm_embeddings / (
            np.linalg.norm(self.norm_embeddings, axis=1, keepdims=True) + 1e-12
        )

    def build_tree(self):
        self._recursive_cluster(
            embeddings=self.norm_embeddings,
            tag_indices=np.arange(len(self.tag_names)),
            level=0,
            parent_id="root",
        )
        return self.tree

    def _recursive_cluster(self, embeddings, tag_indices, level, parent_id):

        # Base case: reached max depth or too few tags
        if level >= len(self.branching_factors) or len(tag_indices) <= 5:
            # Store leaf tags
            leaf_tags = [self.tag_names[i] for i in tag_indices]
            self.tree[parent_id] = {
                "is_leaf": True,
                "tags": leaf_tags,
                "tag_indices": tag_indices,
                "centroid": embeddings.mean(axis=0),
            }
            # Record paths for all tags in this leaf
            for idx in tag_indices:
                self.tag_to_path[idx] = parent_id
            return

        n_clusters = min(self.branching_factors[level], len(tag_indices))
        if self._use_spherical:
            model = SphericalKMeans(
                n_clusters=n_clusters,
                n_init=20,
                max_iter=500,
                random_state=42,
                memory_efficient=True,
                verbose=False,
            )
            cluster_labels = model.fit_predict(embeddings)
        else:
            model = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=20,
                max_iter=500,
                random_state=42,
            )

            cluster_labels = model.fit_predict(embeddings)

        # Store node information
        self.tree[parent_id] = {
            "is_leaf": False,
            "n_clusters": n_clusters,
            "centroids": model.cluster_centers_,
            "children": [],
        }

        # Recursively process each cluster
        for cluster_id in range(n_clusters):
            child_id = f"{parent_id}_c{cluster_id}"
            self.tree[parent_id]["children"].append(child_id)

            mask = cluster_labels == cluster_id
            cluster_tag_indices = tag_indices[mask]
            cluster_embeddings = embeddings[mask]

            self._recursive_cluster(
                cluster_embeddings, cluster_tag_indices, level + 1, child_id
            )

    def get_cluster_path(self, tag_name):
        tag_idx = self.tag_names.index(tag_name)
        return self.tag_to_path.get(tag_idx, None)

    def visualize_tree_statistics(self):
        print(f"Total tags: {len(self.tag_names)}")

        level_counts = {0: 1}
        for node_id in self.tree:
            if node_id == "root":
                continue
            level = node_id.count("_c")
            level_counts[level] = level_counts.get(level, 0) + 1

        for level, count in sorted(level_counts.items()):
            print(f"Level {level}: {count} clusters")
