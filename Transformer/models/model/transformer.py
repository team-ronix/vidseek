import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model.encoder import Encoder


class Transformer(nn.Module):
    """Encoder-only Transformer producing L2-normalized sentence embeddings.

    Architecture: 'Attention Is All You Need' (Vaswani et al., 2017)
    Training:     Multiple Negatives Ranking Loss on positive sentence pairs
    Pooling:      mean (default) or max over non-padding token positions

    Typical usage
    -------------
    model = Transformer(vocab_size=10_000)

    # Encode a batch → normalized embeddings (cosine sim = dot product)
    emb = model.encode(input_ids)               # (B, d_model), unit vectors

    # Similarity between two batches
    sim = (emb_a * emb_b).sum(-1)               # (B,) cosine similarity
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int   = 256,
        n_layers: int  = 4,
        n_heads: int   = 8,
        d_ff: int      = 512,
        max_len: int   = 512,
        dropout: float = 0.1,
        pooling: str   = "mean",   # "mean" | "max"
    ):
        super().__init__()
        assert pooling in ("mean", "max"), (
            f"pooling must be 'mean' or 'max', got '{pooling}'"
        )
        self.pooling = pooling
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)

    # ── pooling ───────────────────────────────────────────────────────────────

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, d)  mask: (B, T) - 1=real token, 0=padding"""
        m = mask.unsqueeze(-1).float()
        if self.pooling == "mean":
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        # max - padding positions are set to -∞ so they never win
        hidden = hidden * m + (1.0 - m) * (-1e9)
        return hidden.max(dim=1).values

    # ── public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,                     # (B, T)
        attention_mask: torch.Tensor | None = None,  # (B, T)
        normalize: bool = True,
    ) -> torch.Tensor:                               # (B, d_model)
        """Encode token ids into fixed-size sentence vectors.

        normalize=True (default) returns L2-normalized embeddings so that
        cosine similarity reduces to a plain dot product:
            cos_sim(a, b) = a · b   when ||a|| = ||b|| = 1

        Pass normalize=False to get raw (unscaled) pooled vectors, e.g.
        when the downstream loss handles normalization itself.
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        hidden   = self.encoder(input_ids, src_mask)         # (B, T, d_model)
        emb      = self._pool(hidden, attention_mask)        # (B, d_model)

        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass - returns sentence embeddings (not similarity scores).

        Calling model(ids) is equivalent to model.encode(ids).
        """
        return self.encode(input_ids, attention_mask, normalize)
