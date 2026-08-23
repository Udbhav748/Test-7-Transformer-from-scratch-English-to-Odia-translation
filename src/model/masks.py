"""Attention masks for the encoder-decoder transformer.

Convention used throughout this module and consumed by attention.py:
masks are boolean tensors where True means "attend / keep" and False
means "masked out". The attention module is responsible for turning
False positions into a large negative bias before softmax.
"""

import torch
from torch import Tensor


def make_padding_mask(ids: Tensor, pad_id: int) -> Tensor:
    return (ids != pad_id).unsqueeze(1).unsqueeze(1)


def make_causal_mask(size: int) -> Tensor:
    return torch.tril(torch.ones(size, size, dtype=torch.bool))


def make_decoder_self_attn_mask(tgt_ids: Tensor, pad_id: int) -> Tensor:
    T = tgt_ids.size(1)
    causal = make_causal_mask(T).to(tgt_ids.device)
    padding = make_padding_mask(tgt_ids, pad_id)
    return causal.unsqueeze(0).unsqueeze(0) & padding


def make_cross_attn_mask(src_ids: Tensor, pad_id: int) -> Tensor:
    return make_padding_mask(src_ids, pad_id)


# NaN-from-all-masked-row note: softmax over a row that is entirely
# masked (all False) would produce NaN, since every logit becomes -inf
# and exp(-inf) sums to zero. That can't happen for the mask shapes
# built here as long as every example has at least one non-pad token.
# Decoder self-attention: query position i is causally allowed to see
# key position i, and real (non-pad) target sequences always have a
# non-pad token at position 0 (<SOS>), so row i for any real query
# always has at least one unmasked key (itself or an earlier real
# token). Cross-attention: every row is masked identically by the
# source padding pattern, so as long as the source sequence has at
# least one non-pad token, no row is fully masked. Fully-empty
# sequences (all pad) are not a case this codebase produces.
