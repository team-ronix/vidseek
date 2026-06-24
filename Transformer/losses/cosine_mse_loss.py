import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineMSELoss(nn.Module):
    """MSE regression between predicted cosine similarity and gold scores.

    Rescales binary [0, 1] labels (or continuous scores) to [-1, 1] to match
    the natural cosine similarity range, so the model can actually reach the
    target for both positive (→ +1) and negative (→ -1) pairs.
    """

    def forward(
        self,
        emb_a:  torch.Tensor,   # (B, d) — L2 normalized
        emb_b:  torch.Tensor,   # (B, d) — L2 normalized
        scores: torch.Tensor,   # (B,)   — targets in [0, 1]
    ) -> torch.Tensor:
        sim     = (emb_a * emb_b).sum(-1)   # (B,) cosine similarity in [-1, 1]
        targets = scores * 2.0 - 1.0        # [0, 1] → [-1, 1]
        return F.mse_loss(sim, targets)
