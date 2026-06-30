import math

import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
    # Dividing by sqrt(d_k) keeps attention scores from becoming too large and prevents softmax saturation.

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # d_k = embedding dimension of each Query/Key vector in one attention head
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.dropout(self.softmax(scores))
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    # Runs h attention heads in parallel, each learning different token relationships.
    # Outputs are concatenated and projected back to d_model.

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention(dropout)

    def split_heads(self, x):
        # (B, T, d_model) -> (B, heads, T, d_k)
        B = x.size(0)
        return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # (B, heads, T, d_k) -> (B, T, d_model)
        B = x.size(0)
        return x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))
        attn_out, _ = self.attention(q, k, v, mask)
        return self.w_o(self.combine_heads(attn_out))
