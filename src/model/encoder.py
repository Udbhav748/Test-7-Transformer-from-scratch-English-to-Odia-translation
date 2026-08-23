import torch.nn as nn
from torch import Tensor

from configs.base import D_FF, D_MODEL, DROPOUT, N_ENCODER_LAYERS, N_HEADS
from src.model.attention import MultiHeadAttention
from src.model.embeddings import Embeddings
from src.model.feedforward import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        n_layers: int = N_ENCODER_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, src_ids: Tensor, src_mask: Tensor) -> Tensor:
        x = self.embeddings(src_ids)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
