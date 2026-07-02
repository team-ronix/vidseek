from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder


class Transformer(nn.Module):
    # Encoder-only transformer for sentence embeddings.
    # Encodes a token sequence -> mean/max pool -> L2 normalize -> sentence vector.

    def __init__(
        self,
        vocab_size: int,
        d_model: int   = 384,
        n_layers: int  = 6,
        n_heads: int   = 6,
        d_ff: int      = 1536,
        max_len: int   = 512,
        dropout: float = 0.1,
        pooling: str   = "mean",   # "mean" or "max"
    ):
        super().__init__()
        assert pooling in ("mean", "max"), (
            f"pooling must be 'mean' or 'max', got '{pooling}'"
        )
        self.pooling = pooling
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, d_model)  mask: (B, T) — 1=real token, 0=padding
        m = mask.unsqueeze(-1).float()
        if self.pooling == "mean":
            # average only over real tokens, ignore padding
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        # max pooling: set padding positions to -inf so they never win
        hidden = hidden * m + (1.0 - m) * (-1e9)
        return hidden.max(dim=1).values

    def encode(
        self,
        input_ids: torch.Tensor,                          # (B, T)
        attention_mask: Optional[torch.Tensor] = None,    # (B, T)
        normalize: bool = True,
    ) -> torch.Tensor:                                    # (B, d_model)


        # expand mask for multi-head attention: (B, 1, 1, T)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        hidden   = self.encoder(input_ids, src_mask)    # (B, T, d_model)
        emb      = self._pool(hidden, attention_mask)   # (B, d_model)

        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        # calling model(ids) is the same as model.encode(ids)
        return self.encode(input_ids, attention_mask, normalize)
