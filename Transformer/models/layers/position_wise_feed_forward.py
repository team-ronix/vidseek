import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    # Two-layer MLP applied to each token position independently.
    # d_ff is usually 4x d_model (e.g. 384 → 1536 → 384).

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.gelu    = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))
