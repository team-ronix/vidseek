import math
import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention — 'Attention Is All You Need', Section 3.2.1.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    Dropout is applied to the attention weights before the weighted sum,
    as specified in Section 3.2.2 of the paper.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)

        # (B, h, T, d_k) x (B, h, d_k, T) -> (B, h, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.dropout(self.softmax(scores))

        output = torch.matmul(attn_weights, v)
        return output, attn_weights
