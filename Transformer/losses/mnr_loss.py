import torch
import torch.nn as nn


class MultipleNegativesRankingLoss(nn.Module):
    # InfoNCE / MNR loss for sentence pairs.
    # Builds a BxB cosine similarity matrix - every other positive in the batch
    # serves as a hard negative, so a bigger batch = more training signal.

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature   = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor) -> torch.Tensor:
        # cosine sim matrix (B, B) - diagonal holds the matching pairs
        sim = torch.matmul(anchor_emb, positive_emb.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)

        # bidirectional loss: anchor -> positive and positive -> anchor
        loss = (self.cross_entropy(sim, labels) + self.cross_entropy(sim.T, labels)) / 2
        return loss
