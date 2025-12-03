# Score Prediction from Embeddings

## Data

- Goal: regress Stack Overflow question scores directly from dense representations instead of sparse, text-heavy pipelines.
- Data sources:
  - `stackexchange_reduced_tags_embeddings.pkl` - 4,096‑dimensional `title_embedding` and `question_text_embedding`.
  - `stackexchange_dataset.csv` - `question_score`, `num_tags`.
- Preprocessing:
  - Join embeddings with metadata on `question_id`, drop rows without scores, cast everything to `float32`.
  - Clip tag counts to `[0, 5]` and normalize to `[0, 1]` when fed into neural nets.
  - 80/10/10 train/validation/test split with fixed seed (42).

---

## Single-Stream Regressors

- Baseline: `SimpleEmbeddingRegressor`.
  - Inputs: either the body (`question_text_embedding`) or the title embedding alone.
  - Architecture:
    - Linear (4096 → 1024) + LayerNorm + GELU + Dropout(0.3).
    - Linear (1024 → 512) + GELU + Dropout(0.3).
    - Linear (512 → 1) with squeeze.
  - Loss: `MSE`.
  - Optimizer: `AdamW`, lr $1e^{-3}$, weight decay $1e^{-4}$.
  - Training: mini-batch size 256, up to 50 epochs (same notebook supports runs with or without early stopping).
  - Outputs: per-question score predictions; metrics logged as RMSE/MAE/R² on both validation and held-out test splits.

---

## Dual-Stream DSF Regressors

- Shared ingredients:
  - Stream-specific projection stacks (Linear 4096 → 1024, LayerNorm, GELU, Dropout).
  - `AdamW` (lr $5e^{-4}$), weight decay $1e^{-4}$, batch size 128, 50 epochs.
- Variants:
  1. **DSF-MHSA Regressor**
     - Fuse stacked title/body streams via Multi-Head Self-Attention (8 heads) + residual LayerNorm.
     - Final regressor head: Linear(2049 → 1024) → GELU → Dropout(0.4) → Linear(1024 → 1).
  2. **DSF-CrossAttention Regressor**
     - Treat title embedding as the query attending over the body embedding (cross-attention).
     - Concatenate attended title features with raw body projection before the regression head (BatchNorm → ReLU → Dropout(0.5) → Linear → 1).
- Loss: `MSE` (regression), metrics identical to single-stream setup.

---

## Results

| Model | Features | Validation RMSE | Validation MAE | Validation R² | Test RMSE | Test MAE | Test R² |
|-------|----------|-----------------|----------------|---------------|-----------|----------|---------|
| DSF-MHSA | title + body embeddings | 146.77 | 29.47 | 0.263 | 134.24 | 30.40 | 0.199 |
| DSF-CrossAttention | title + body embeddings | 158.12 | 30.22 | 0.145 | 135.27 | 29.80 | 0.186 |
| Body MLP | question_text_embedding | 156.48 | 33.61 | 0.163 | 137.57 | 33.34 | 0.158 |
| Title MLP | title_embedding | 155.33 | 27.66 | 0.175 | 138.73 | 28.20 | 0.144 |
| Mean baseline | Constant | 171.02 | 36.79 | -0.000 | 149.97 | 37.43 | -0.000 |

---

## Results  
![Predicted vs actual scores (filtered range)](img/predicted-actual-filtered.png)
