# SCORE PREDICTION — SUMMARY

## Approach

We began with a **classical NLP regression pipeline** to predict question scores.  
Steps included:

- Text preprocessing  
  - Lowercasing + whitespace normalization  
  - Regex tokenization preserving technical tokens (e.g. `c++`, `c#`)  
  - Stopword removal with a technical whitelist  
  - Stemming

- Feature engineering  
  - `question_length_words`, `num_code_blocks`, `num_code_lines`  
  - binary flags such as presence of “?” and presence of code

- Modeling  
  - Baselines (mean predictor, question length regression)  
  - Bag-of-Words + Ridge
  - TF-IDF (unigrams, bigrams)
  - TF-IDF → **SVD (200 dims)** → Ridge
  - TF-IDF → SVD + numeric features → Gradient Boosting

## Baseline Results (Validation)

| Model | RMSE | Train Time |
|------|------|------------|
| **TF-IDF → SVD → Ridge** | **152.31** | ~27s |
| TF-IDF bigrams + Ridge | 155.16 | 32s |
| TF-IDF unigrams + Ridge | 157.50 | 10s |
| BOW + Ridge | 347.79 | 77s |
| Mean baseline | 154.86 | - |

**Interpretation:**  
Even the best baseline model barely improves over predicting the mean score → score prediction is extremely hard using plain text only.

## Enhanced Modeling

To address the **heavy right-skew**, we introduced a **log-transformed target** and tested additional models: LinearSVR, SGD(Huber), RandomForest, GradientBoosting, and **MLP on SVD features**.

We also kept:

- the exact same preprocessing and data splits  
- the same TF-IDF and SVD configurations 

### Additional findings:

- **Log-score** stabilizes training but does not improve generalization.  
- **SVD + MLP** becomes the strongest single enhanced model.  
- Simple **ensembling** (mean of 3 best models) further improves validation RMSE.

## Enhanced Results (Validation + Test)

| Model | RMSE |
|------|------|
| **Mean Ensemble (SVD+MLP, SVD+Ridge, SVD+RF)** | **149.25** |
| SVD + MLP (raw) | 150.23 |
| SVD + Ridge | 152.39 |
| SVD + RandomForest | 154.98 |
| LinearSVR / SGD | 157–265 |
