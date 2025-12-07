import torch
import torch.nn as nn
import torch.nn.functional as F


class DSF_Sequence_Aware_Classifier(nn.Module):
    def __init__(
        self, input_dim=4096, hidden_dim=1024, num_classes=100, num_heads=8, dropout=0.5
    ):
        super().__init__()

        # --- STREAM 1: Title (Query) ---
        # Input: (Batch, 4096) -> Output: (Batch, 1024)
        self.title_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # --- STREAM 2: Body Sequence (Key/Value) ---
        # Input: (Batch, 32, 4096) -> Output: (Batch, 32, 1024)
        # This time we're using Conv1d to capture local patterns (phrases) in the body sequence.
        # This was because the network overfitted too much with Linear layers on the body sequence.
        self.body_proj = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- SELF-ATTENTION LAYER ---
        # Self-Attention on Body Sequence to better represent it before Cross-Attention, since we have 32 embeddings for the body.
        self.body_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )

        # --- CROSS-ATTENTION ---
        # Query = Title
        # Key/Value = Body Sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm_title = nn.LayerNorm(hidden_dim)

        # --- NUM TAGS PROJECTION ---
        # Since single scalar input can be easily tyreated as a noise, we project it to a higher dimension so it can be 'easily heard'
        self.num_tags_proj = nn.Sequential(nn.Linear(1, 32), nn.ReLU())

        # --- CLASSIFIER ---
        # Input = Enriched Title (1024) + Global Body Context (1024) + Num Tags (1) -> 32
        fusion_dim = hidden_dim * 2 + 32  # 1024 + 1024 + 1 projected to 32

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def masked_mean_pooling(self, sequence, mask):
        """
        Pools the sequence into a single vector, ignoring padded tokens.
        Args:
            sequence: (Batch, Seq_Len, Dim)
            mask: (Batch, Seq_Len) - True indicates Padding (Ignore), False indicates Real
        """
        input_mask_expanded = (~mask).unsqueeze(-1).float()  # (B, S, 1)
        sum_embeddings = torch.sum(sequence * input_mask_expanded, dim=1)  # (B, Dim)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)  # (B, 1)

        return sum_embeddings / sum_mask

    def forward(self, title_emb, body_seq, padding_mask, num_tags):
        """
        Args:
            title_emb: (Batch, 4096)
            body_seq: (Batch, 32, 4096)
            padding_mask: (Batch, 32) - True where padding exists
            num_tags: (Batch, 1)
        """
        t_feat = self.title_proj(title_emb)  # (B, 1024)
        b_feat_conv = self.body_proj(body_seq.permute(0, 2, 1))
        b_feat_seq = b_feat_conv.permute(0, 2, 1)  # (B, 32, 1024)

        # We keep b_att_feat_seq for Cross-Attention, but we won't use it for pooling later to get global body context.
        b_att_feat_seq = self.body_self_attn(
            b_feat_seq, src_key_padding_mask=padding_mask
        )

        query = t_feat.unsqueeze(1)
        attn_output, attn_weights = self.cross_attn(
            query=query,
            key=b_att_feat_seq,
            value=b_att_feat_seq,
            key_padding_mask=padding_mask,
        )

        t_enriched = self.norm_title(t_feat + attn_output.squeeze(1))

        b_pooled = self.masked_mean_pooling(b_feat_seq, padding_mask)

        n_tags_proj = self.num_tags_proj(num_tags)
        fused = torch.cat([t_enriched, b_pooled, n_tags_proj], dim=1)

        return self.classifier(fused), attn_weights
