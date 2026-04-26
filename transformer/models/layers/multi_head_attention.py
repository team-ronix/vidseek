import torch
import torch.nn as nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention()

    def split_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))

        attn_output, _ = self.attention(q, k, v, mask)

        output = self.w_o(self.combine_heads(attn_output))
        return output
