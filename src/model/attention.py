import math

import torch
import torch.nn as nn
from torch import Tensor

from configs.base import D_MODEL, N_HEADS

_MASK_FILL_VALUE = -1e9


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        x = x.view(B, L, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, L, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, H * D)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(~mask, _MASK_FILL_VALUE)

        attn = torch.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach()
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)
