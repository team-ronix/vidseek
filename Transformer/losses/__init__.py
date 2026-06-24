from losses.mnr_loss import MultipleNegativesRankingLoss
from losses.triplet_loss import TripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.cosent_loss import CoSENTLoss
from losses.cosine_mse_loss import CosineMSELoss

__all__ = [
    "MultipleNegativesRankingLoss",
    "TripletLoss",
    "ContrastiveLoss",
    "CoSENTLoss",
    "CosineMSELoss",
]
