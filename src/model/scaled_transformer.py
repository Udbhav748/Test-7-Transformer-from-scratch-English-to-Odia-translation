"""Scaled Transformer (Pre-LN + weight tying) used for the 11.5M-parameter model.

Moved out of app.py so it can be reused by evaluation scripts, not just the
Streamlit dashboard.
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            device = x.device
            needed = seq_len + 16
            pe = torch.zeros(needed, self.pe.size(1), device=device)
            position = torch.arange(0, needed, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.pe.size(1), 2, dtype=torch.float32, device=device)
                * (-math.log(10000.0) / self.pe.size(1))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe
        return self.dropout(x + self.pe[:seq_len, :].unsqueeze(0))


class ScaledEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = ScaledPositionalEncoding(d_model, dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding(self.token_embedding(ids) * self.scale)


class ScaledMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.last_attn_weights = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape
        q_s = self.q_proj(q).view(B, Lq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k_s = self.k_proj(k).view(B, Lk, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v_s = self.v_proj(v).view(B, Lk, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach()
        out = torch.matmul(attn, v_s).permute(0, 2, 1, 3).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)


class ScaledFeedForward(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class ScaledPreLNEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = ScaledMultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ScaledFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, src_mask))
        norm_x = self.norm2(x)
        x = x + self.dropout2(self.ffn(norm_x))
        return x


class ScaledPreLNDecoderBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = ScaledMultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = ScaledMultiHeadAttention(d_model, n_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = ScaledFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, tgt_mask))
        norm_x = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(norm_x, enc_out, enc_out, src_mask))
        norm_x = self.norm3(x)
        x = x + self.dropout3(self.ffn(norm_x))
        return x


class EnhancedScaledTransformer(nn.Module):
    def __init__(
        self,
        en_vocab_size: int = 8000,
        or_vocab_size: int = 8000,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.encoder_emb = ScaledEmbeddings(en_vocab_size, d_model, dropout)
        self.decoder_emb = ScaledEmbeddings(or_vocab_size, d_model, dropout)
        self.encoder_blocks = nn.ModuleList([ScaledPreLNEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)])
        self.encoder_final_norm = nn.LayerNorm(d_model)
        self.decoder_blocks = nn.ModuleList([ScaledPreLNDecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)])
        self.decoder_final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, or_vocab_size, bias=False)

        if tie_weights:
            self.output_proj.weight = self.decoder_emb.token_embedding.weight

    def make_src_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
        return (src_ids != self.pad_id).unsqueeze(1).unsqueeze(1)

    def make_tgt_mask(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        T = tgt_ids.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_ids.device))
        pad = (tgt_ids != self.pad_id).unsqueeze(1).unsqueeze(1)
        return causal.unsqueeze(0).unsqueeze(0) & pad

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src_ids)
        tgt_mask = self.make_tgt_mask(tgt_ids)
        x = self.encoder_emb(src_ids)
        for b in self.encoder_blocks:
            x = b(x, src_mask)
        x = self.encoder_final_norm(x)

        y = self.decoder_emb(tgt_ids)
        for b in self.decoder_blocks:
            y = b(y, x, tgt_mask, src_mask)
        y = self.decoder_final_norm(y)
        return self.output_proj(y)
