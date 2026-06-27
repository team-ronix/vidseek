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