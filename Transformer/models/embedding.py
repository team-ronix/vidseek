import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # Fixed sinusoidal encoding - not learned, just adds position info to each token.
    # Even dims use sin, odd dims use cos, each at a different frequency.

    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)  # saved in state_dict but not trained
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)


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
