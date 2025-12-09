import numpy as np


class DualEncoderMatcher:
    def __init__(self, question_embeddings, tag_embeddings, tag_names):
        self.question_embeddings = question_embeddings
        self.tag_embeddings = tag_embeddings
        self.tag_names = tag_names

        self.question_embeddings_norm = self._normalize(question_embeddings)
        self.tag_embeddings_norm = self._normalize(tag_embeddings)

    def _normalize(self, embeddings):
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def predict_top_k(self, question_idx, k=5):
        similarities = (
            self.question_embeddings_norm[question_idx] @ self.tag_embeddings_norm.T
        )

        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_tags = [self.tag_names[i] for i in top_k_indices]
        top_k_scores = similarities[top_k_indices]

        return list(zip(top_k_tags, top_k_scores))

    def batch_predict(self, question_indices, k=5):
        similarities = (
            self.question_embeddings_norm[question_indices] @ self.tag_embeddings_norm.T
        )

        top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]

        predictions = []
        for i, _ in enumerate(question_indices):
            top_tags = [self.tag_names[idx] for idx in top_k_indices[i]]
            top_scores = similarities[i, top_k_indices[i]]
            predictions.append(list(zip(top_tags, top_scores)))

        return predictions
