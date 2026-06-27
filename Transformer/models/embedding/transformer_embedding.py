import math
import torch.nn as nn
from models.embedding.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    # Token embedding + positional encoding, combined before the encoder.

    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.d_model   = d_model

    def forward(self, x):
        # scale by sqrt(d_model) as in the paper — keeps token and position signals balanced
        tok = self.token_emb(x) * math.sqrt(self.d_model)
        return self.pos_enc(tok)
