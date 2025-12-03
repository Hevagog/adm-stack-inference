import torch
import torch.nn as nn
import torch.nn.functional as F


class DSF_MHSA_Classifier(nn.Module):
    """Inspired by 'Dual-stream fusion network with multi-head self-attention for multi-modal
    fake news detection' https://www.sciencedirect.com/science/article/pii/S1568494624011323
    Instead of processing text and images as in the paper above, we will map two streams:
    - question_text embeddings,
    - title embeddings,

    and fuse them using multi-head self-attention to capture correlation between the two modalities.
    As a fusion, we will use the paper's proposed MHSA Fusion strategy to let the Title "attend" to the question body and vice versa.
    """

    def __init__(
        self, input_dim=4096, hidden_dim=1024, num_classes=100, num_heads=8, dropout=0.2
    ):
        super().__init__()

        # Dual Stream Projections (for efficiency, reducing input dim → hidden dim)
        self.title_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.body_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Multi-Head Self-Attention Fusion
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)

        # Classifier Head
        # Input is hidden_dim * 2 (concatenated streams) + 1 (num_tags feature)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, title_emb, body_emb, num_tags):
        t_feat = self.title_proj(title_emb)
        b_feat = self.body_proj(body_emb)  # (Batch, 1024)

        seq = torch.stack([t_feat, b_feat], dim=1)  # (Batch, 2, 1024)

        attn_output, _ = self.mhsa(seq, seq, seq)
        seq = self.layer_norm_fusion(seq + attn_output)

        fused_feat = torch.cat([seq[:, 0, :], seq[:, 1, :]], dim=1)  # (Batch, 2048)

        features = torch.cat([fused_feat, num_tags], dim=1)

        logits = self.classifier(features)

        return logits
