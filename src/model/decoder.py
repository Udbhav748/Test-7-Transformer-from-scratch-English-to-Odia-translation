import torch.nn as nn
from torch import Tensor

from configs.base import D_FF, D_MODEL, DROPOUT, N_DECODER_LAYERS, N_HEADS
from src.model.attention import MultiHeadAttention
from src.model.embeddings import Embeddings
from src.model.feedforward import PositionwiseFeedForward


class DecoderBlock(nn.Module):
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

        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor,
        src_mask: Tensor,
    ) -> Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        n_layers: int = N_DECODER_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, d_model, dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        tgt_ids: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor,
        src_mask: Tensor,
    ) -> Tensor:
        x = self.embeddings(tgt_ids)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x
