import torch
import torch.nn as nn


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss (InfoNCE / NT-Xent variant).

    Given a batch of B (anchor, positive) pairs encoded as L2-normalized
    vectors, every other positive in the batch acts as a negative for each
    anchor - for free, with no extra curation.

    Mathematical formulation

    Let A ∈ ℝ^{B×d} be the anchor embeddings and P ∈ ℝ^{B×d} the positive
    embeddings, both L2-normalized so cosine similarity = dot product.

    Similarity matrix  S ∈ ℝ^{B×B}:
        S_ij = A_i · P_j   (= cos_sim(a_i, p_j))

    For anchor a_i the model should assign highest similarity to its true
    positive p_i (diagonal entry S_ii) and low similarity to all imposters
    (off-diagonal row entries).

    Probability of the correct positive:
        P(i | a_i) = exp(S_ii / τ) / Σ_j exp(S_ij / τ)

    Loss (average negative log-likelihood):
        L = -(1/B) Σ_i log P(i | a_i)
          = CrossEntropy( S / τ,  [0, 1, 2, …, B-1] )

    Temperature τ controls sharpness:
        · τ = 0.05  (default) - tight clusters, hard discrimination
        · τ = 1.0             - softer, more lenient
        Lower τ is generally better for semantic search.

    In-batch negatives
    ──────────────────
    A batch of size B provides B negative signals per anchor (B² total).
    Larger batches → more negatives → harder task → better embeddings.
    Recommended batch size ≥ 32 (64–256 ideal).

    Relationship to other losses

    · τ → ∞   : reduces toward MSE on similarity scores
    · τ → 0   : approaches hard max (winner-take-all)
    · Symmetric version (+ CrossEntropy on columns): SimCLR / NT-Xent
    · Asymmetric (rows only): DPR, E5, most dense-retrieval models
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature  = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        anchor_emb:   torch.Tensor,   # (B, d) - L2 normalized
        positive_emb: torch.Tensor,   # (B, d) - L2 normalized
    ) -> torch.Tensor:
        # (B, B) - since vectors are normalized, dot product = cosine similarity
        sim = torch.matmul(anchor_emb, positive_emb.T) / self.temperature

        # Each row i should peak at column i (and vice versa for columns)
        labels = torch.arange(sim.size(0), device=sim.device)

        loss_a2p = self.cross_entropy(sim,   labels)
        loss_p2a = self.cross_entropy(sim.T, labels)
        return (loss_a2p + loss_p2a) / 2
