import torch.nn as nn
from models.embedding.transformer_embedding import TransformerEmbedding
from models.blocks.encoder_layer import EncoderLayer


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
        # x: (batch, seq_len) token ids → (batch, seq_len, d_model) contextual vectors
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
