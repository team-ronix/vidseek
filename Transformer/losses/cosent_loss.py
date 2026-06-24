import torch
import torch.nn as nn


class CoSENTLoss(nn.Module):
    """CoSENT: Cosine Sentence Embedding Loss (Su et al., 2022).

    Pairwise ranking loss over the full batch:
        L = log(1 + Σ_{(p,q): score_p > score_q} exp(scale · (sim_q - sim_p)))

    For every pair (p, q) where pair p has a higher gold score than pair q,
    we penalise the case where sim_q ≥ sim_p (wrong ranking order).

    Works with both continuous scores (STS) and binary labels (0/1 from AllNLI).
    With binary labels the loss reduces to: all (pos, neg) pair combinations in
    the batch are ranked, which gives B_pos × B_neg training signals per step.
    """

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        emb_a:  torch.Tensor,   # (B, d) — L2 normalized
        emb_b:  torch.Tensor,   # (B, d) — L2 normalized
        scores: torch.Tensor,   # (B,)   — similarity targets (higher = more similar)
    ) -> torch.Tensor:
        sim = (emb_a * emb_b).sum(-1)  # (B,) cosine similarities

        # sim_diff[p, q] = sim[q] - sim[p]  (positive when sim[q] > sim[p])
        sim_diff = sim.unsqueeze(0) - sim.unsqueeze(1)   # (B, B)

        # score_diff[p, q] = score[p] - score[q]  (positive when score[p] > score[q])
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (B, B)

        # Penalise pairs where score[p] > score[q] but sim[q] >= sim[p]
        mask = (score_diff > 0).float()
        loss_terms = torch.exp(sim_diff * self.scale) * mask
        return torch.log(1.0 + loss_terms.sum())
