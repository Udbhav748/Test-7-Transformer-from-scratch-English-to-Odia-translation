import torch.nn as nn
from torch import Tensor

from configs.base import D_FF, D_MODEL, DROPOUT


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF, dropout: float = DROPOUT):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)
