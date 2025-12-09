import torch
import torch.nn as nn


class DSF_CrossAttn_Classifier(nn.Module):
    def __init__(
        self, input_dim=4096, hidden_dim=1024, num_classes=100, num_heads=8, dropout=0.5
    ):
        super().__init__()

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

        # Cross-Attention: Query = Title, Key/Value = Body
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, title_emb, body_emb, num_tags):
        t_feat = self.title_proj(title_emb)
        b_feat = self.body_proj(body_emb)

        # Reshape for MultiheadAttention (N, SeqLen=1, Dim)
        query = t_feat.unsqueeze(1)  # Title is the Query
        key_val = b_feat.unsqueeze(1)  # Body is the Context

        # Cross Attention: "Given this Title, which parts of the Body Embedding are relevant?"
        attn_output, attention_weights = self.cross_attn(
            query, key_val, key_val
        )  # (N, 1, 1024)

        # We add the attention context TO the original title features
        t_contextual = self.layer_norm(t_feat + attn_output.squeeze(1))  # (N, 1024)

        fused = torch.cat([t_contextual, b_feat], dim=1)  # (N, 2048)

        features = torch.cat([fused, num_tags], dim=1)  # (N, 2049)

        return self.classifier(features), attention_weights
