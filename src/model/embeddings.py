import math

import torch
import torch.nn as nn
from torch import Tensor

from configs.base import D_MODEL, DROPOUT, MAX_LEN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = D_MODEL, dropout: float = DROPOUT, max_len: int = MAX_LEN + 16):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = D_MODEL):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, ids: Tensor) -> Tensor:
        return self.embedding(ids) * math.sqrt(self.d_model)


class Embeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        dropout: float = DROPOUT,
        max_len: int = MAX_LEN + 16,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

    def forward(self, ids: Tensor) -> Tensor:
        return self.positional_encoding(self.token_embedding(ids))
