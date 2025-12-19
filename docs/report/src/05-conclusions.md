#  Conclusions and Future Work

The overall project successfully navigated several data processing and modeling complexities inherent in large-scale textual data.

## Key Successes and Insights

*   **Robust Tag Reduction:** The greatest technical victory in the data preparation phase was the establishment of the **Recursive Spherical K-Means** pipeline, effectively reducing the 22,753 original tags to 100 semantically coherent centroids. This step was vital for making multi-label classification computationally feasible after the failure of unsupervised methods like UMAP/HDBSCAN optimization.
*   **Optimal Classifier Architecture:** The use of **Dual-Stream Fusion Networks** demonstrated a clear advantage over both traditional ML (XGBoost) and simple MLPs for multi-label classification. The best results were achieved by the **DSF with Cross-Attention Fusion (F1 Weighted 0.7196)**, incorporating both **Asymmetric Loss** to manage label imbalance and **Manifold Mixup** to improve generalization.
*   **Score Prediction Difficulty:** The consistently low $R^2$ values across all regression experiments confirm that predicting Stack Overflow scores from purely semantic content embeddings is inherently difficult due to the social/external factors influencing a question’s eventual popularity (score).

## Summary of Best Results

| Task | Best Model | Key Metric | Result |
| :--- | :--- | :--- | :--- |
| **Tag Prediction (Classification)** | **DSF Cross-Attention** | F1 Weighted | **0.7196** |
| **Score Prediction (Regression)** | **DSF-MHSA Regressor** | Test $R^2$ | **0.199** |

## Future Work

Future research should focus on mitigating the score prediction problem by incorporating features that capture non-textual quality signals, such as user reputation or time-of-day posting biases, which were outside the scope of this content-based embedding analysis. For classification, exploring even deeper transformer architectures for sequence-aware embedding generation could further enhance performance beyond the current pooled embedding methods.
