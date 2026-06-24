import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Online Contrastive Loss for L2-normalized sentence embeddings.

    positive pairs (label=1): push cos_sim → +1   loss = (1 - sim)²
    negative pairs (label=0): push cos_sim below margin  loss = relu(sim - margin)²

    Reference: Hadsell et al., 2006
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb_a:  torch.Tensor,   # (B, d) — L2 normalized
        emb_b:  torch.Tensor,   # (B, d) — L2 normalized
        labels: torch.Tensor,   # (B,)   — 1.0 positive, 0.0 negative
    ) -> torch.Tensor:
        sim = (emb_a * emb_b).sum(-1)                            # (B,) cosine similarity
        pos = (1.0 - sim).pow(2)
        neg = torch.clamp(sim - self.margin, min=0.0).pow(2)
        return (labels * pos + (1.0 - labels) * neg).mean()
