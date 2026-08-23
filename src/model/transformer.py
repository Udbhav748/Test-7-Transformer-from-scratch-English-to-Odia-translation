import torch.nn as nn
from torch import Tensor

from configs.base import (
    D_FF,
    D_MODEL,
    DROPOUT,
    N_DECODER_LAYERS,
    N_ENCODER_LAYERS,
    N_HEADS,
    PAD_ID,
    TIE_OUTPUT_PROJECTION,
)
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.masks import (
    make_cross_attn_mask,
    make_decoder_self_attn_mask,
    make_padding_mask,
)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        n_encoder_layers: int = N_ENCODER_LAYERS,
        n_decoder_layers: int = N_DECODER_LAYERS,
        dropout: float = DROPOUT,
        pad_id: int = PAD_ID,
        tie_output_projection: bool = TIE_OUTPUT_PROJECTION,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_encoder_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, n_decoder_layers, dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        if tie_output_projection:
            self.output_projection.weight = self.decoder.embeddings.token_embedding.embedding.weight

    def forward(self, src_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        src_mask = make_padding_mask(src_ids, self.pad_id)
        tgt_mask = make_decoder_self_attn_mask(tgt_ids, self.pad_id)
        cross_mask = make_cross_attn_mask(src_ids, self.pad_id)

        encoder_output = self.encoder(src_ids, src_mask)
        decoder_output = self.decoder(tgt_ids, encoder_output, tgt_mask, cross_mask)
        return self.output_projection(decoder_output)
