import torch.nn as nn

from models.attention import MultiHeadAttention
from models.embedding import TransformerEmbedding
from models.feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    # Self-attention + FFN, each wrapped with residual + layer norm (Add & Norm).

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn  = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class Encoder(nn.Module):
    # Stacks n_layers encoder blocks on top of the embedding layer.

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, src_mask=None):
        # x: (batch, seq_len) token ids -> (batch, seq_len, d_model) contextual vectors
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
