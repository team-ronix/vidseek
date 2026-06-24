import torch.nn as nn
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """Single encoder layer - 'Attention Is All You Need', Section 3.1.

    Each layer has two sublayers:
      1. Multi-head self-attention
      2. Position-wise feed-forward network
    Both are wrapped with Add & Norm: LayerNorm(x + Dropout(sublayer(x)))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn  = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # sublayer 1: self-attention + Add & Norm
        attn_out = self.attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # sublayer 2: FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x
