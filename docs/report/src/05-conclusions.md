#  Conclusions and Future Work

The overall project successfully navigated several data processing and modeling complexities inherent in large-scale textual data.

## Key Successes and Insights

*   **Robust Tag Reduction:** The greatest technical victory in the data preparation phase was the establishment of the **Recursive Spherical K-Means** pipeline, effectively reducing the 22,753 original tags to 100 semantically coherent centroids. This step was vital for making multi-label classification computationally feasible after the failure of unsupervised methods like UMAP/HDBSCAN optimization.
*   **Optimal Classifier Architecture:** The use of **Dual-Stream Fusion Networks** demonstrated a clear advantage over both traditional ML (XGBoost) and simple MLPs for multi-label classification. The best results were achieved by the **DSF with Cross-Attention Fusion (F1 Weighted 0.7196)**, incorporating both **Asymmetric Loss** to manage label imbalance and **Manifold Mixup** to improve generalization.
*   **Inductive Bias vs. Data Volume:** The ultimate limitation observed across 
our experiments in tag prediction, was the trade-off between model flexibility 
and dataset size. As noted in [Performance Analysis](#pa), Transformer-based attention mechanisms lack the inductive bias of simpler architectures (like CNNs or MLPs). They are extremely flexible but highly "data-hungry".
* Our dataset of $\approx$ 100,000 questions was insufficient to constrain the vast search space of the fully sequence-aware models, leading them to memorize noise rather than learn robust generalized features. Consequently, the DSF with Cross-Attention represented the optimal "sweet spot": it was complex enough to model semantic interactions via attention, but structured enough (compressing the body into a single embedding) to avoid the overfitting pitfalls of full sequence modeling.
*   **Score Prediction Difficulty:** The consistently low $R^2$ values across all regression experiments confirm that predicting Stack Overflow scores from purely semantic content embeddings is inherently difficult due to the social/external factors influencing a question’s eventual popularity (score).

## Summary of Best Results

| Task | Best Model | Key Metric | Result |
| :--- | :--- | :--- | :--- |
| **Tag Prediction (Classification)** | **DSF Cross-Attention** | F1 Weighted | **0.7196** |
| **Score Prediction (Regression)** | **DSF-MHSA Regressor** | Test $R^2$ | **0.199** |

## Future Work

Future research should focus on mitigating the score prediction problem by incorporating features that capture non-textual quality signals, such as user reputation or time-of-day posting biases, which were outside the scope of this content-based embedding analysis. For classification, acquiring dataset of size around 10000000 could drastically enhance performance beyond the current pooled embedding methods.
