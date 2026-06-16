import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet Margin Loss for sentence embeddings.

    Mathematical formulation
    ────────────────────────
    Given L2-normalized (anchor a, positive p, negative n):

    Cosine distance:  d(x, y) = 1 − cos_sim(x, y)

    Loss per triplet:
        ℓ = max( d(a, p) − d(a, n) + margin,  0 )
          = max( sim(a, n) − sim(a, p) + margin,  0 )

    The loss is zero — no gradient — when the positive is already at least
    `margin` more similar than the negative:
        sim(a, p) − sim(a, n) ≥ margin  →  ℓ = 0

    Geometric picture
    ─────────────────
    All sentences live on the unit sphere (after L2 normalization).
    The loss nudges a closer to p and further from n until a cone of width
    `margin` separates them on the sphere surface.

    Comparison with MNR Loss
    ────────────────────────
    · Triplet uses 1 negative per anchor per step.
    · MNR uses (B−1) negatives per anchor per step — far more gradient signal.
    · Triplet suffers from "easy negative collapse": once simple triplets are
      satisfied the loss is 0 everywhere and training stalls. Hard negative
      mining is required to keep learning.
    · Use TripletLoss when you have pre-mined hard negatives, otherwise
      prefer MultipleNegativesRankingLoss.

    Hyperparameter guidance
    ────────────────────────
    · margin = 0.3–0.5  — start here; increase if embeddings collapse
    · Cosine space: margins are in [-2, 2] (d = 1 − sim ∈ [0, 2])
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor:   torch.Tensor,   # (B, d) — L2 normalized
        positive: torch.Tensor,   # (B, d) — L2 normalized
        negative: torch.Tensor,   # (B, d) — L2 normalized
    ) -> torch.Tensor:
        sim_pos = F.cosine_similarity(anchor, positive, dim=-1)   # (B,)
        sim_neg = F.cosine_similarity(anchor, negative, dim=-1)   # (B,)

        # sim_neg − sim_pos + margin > 0  ⟹  non-zero loss
        loss = torch.clamp(sim_neg - sim_pos + self.margin, min=0.0)
        return loss.mean()
