import torch.nn as nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention - 'Attention Is All You Need', Section 3.2.2.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
    where head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

    Projects Q, K, V into h independent d_k-dimensional subspaces,
    runs Scaled Dot-Product Attention in each, then concatenates and
    projects back to d_model with W_O.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention(dropout)

    def split_heads(self, x):
        """(B, T, d_model) -> (B, h, T, d_k)"""
        B = x.size(0)
        return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """(B, h, T, d_k) -> (B, T, d_model)"""
        B = x.size(0)
        return x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))

        attn_out, _ = self.attention(q, k, v, mask)

        return self.w_o(self.combine_heads(attn_out))
