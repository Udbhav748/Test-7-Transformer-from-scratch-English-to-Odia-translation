"""English to Odia Neural Machine Translation Platform

Enterprise-grade research and demonstration dashboard for from-scratch
Sequence-to-Sequence Transformers (Baseline and Scaled GPU architectures).
"""

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# -----------------------------------------------------------------------------
# Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="English -> Odia Neural Machine Translation",
    page_icon="https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f310.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = Path(__file__).resolve().parent

# Color Palette (Linear / Stripe / Vercel Enterprise aesthetic)
PRIMARY = "#2563EB"       # Classic Sapphire Blue
SLATE_900 = "#0F172A"     # Deepest slate
SLATE_800 = "#1E293B"     # Card slate
SLATE_700 = "#334155"     # Border slate
TEXT_MAIN = "#0F172A"     # Primary text
TEXT_MUTED = "#64748B"    # Secondary text
TEXT_LIGHT = "#94A3B8"    # Subtle labels
BORDER_SUBTLE = "#E2E8F0" # Crisp hairline borders
BG_SURFACE = "#FFFFFF"    # Pure white surface
BG_PAGE = "#F8FAFC"       # Cool soft gray canvas
SUCCESS = "#059669"       # Emerald status

PRO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Oriya:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Navbar / App Header */
.app-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.12);
}
.app-title-group h1 {
    font-size: 1.25rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.app-title-group p {
    font-size: 0.82rem;
    color: #94A3B8;
    margin: 0.2rem 0 0 0;
    font-weight: 400;
}
.nav-badges {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.28);
    color: #34D399;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    letter-spacing: 0.02em;
}
.status-dot {
    width: 6px;
    height: 6px;
    background: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
}
.tag-badge {
    background: #1E293B;
    border: 1px solid #334155;
    color: #E2E8F0;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.8rem;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.92rem;
    font-weight: 600;
    padding: 0.75rem 0.2rem;
    color: #64748B;
    border-bottom: 2px solid transparent;
    transition: color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0F172A;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
}

/* Workbench Cards */
.workbench-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02);
    overflow: hidden;
    height: 100%;
}
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.1rem;
    border-bottom: 1px solid #F1F5F9;
    background: #F8FAFC;
}
.panel-title {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
}
.panel-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
}

/* Translation Result Containers */
.target-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.15s ease;
}
.target-box:hover {
    border-color: #CBD5E1;
}
.target-box.highlight {
    border-left: 4px solid #2563EB;
}
.target-box.scaled-box {
    border-left: 4px solid #4F46E5;
}
.target-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
}
.model-pill {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.18rem 0.55rem;
    border-radius: 4px;
}
.model-pill.baseline {
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
}
.model-pill.scaled {
    background: #EEF2FF;
    color: #4338CA;
    border: 1px solid #C7D2FE;
}
.odia-text {
    font-family: 'Noto Sans Oriya', sans-serif !important;
    font-size: 1.3rem;
    font-weight: 600;
    color: #0F172A;
    line-height: 1.85;
    margin: 0;
}
.meta-chip-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
    padding-top: 0.6rem;
    border-top: 1px solid #F1F5F9;
}
.meta-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
}

/* Prompt Pills */
.prompt-chips-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.65rem 0 0.85rem 0;
}
.prompt-chip {
    font-size: 0.76rem;
    font-weight: 500;
    color: #334155;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.prompt-chip:hover {
    background: #E2E8F0;
    color: #0F172A;
}

/* Section Headings */
.view-heading {
    font-size: 1.08rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
    margin: 1.4rem 0 0.75rem 0;
}

/* KPI Cards */
.kpi-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);
}
.kpi-title {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.1;
    font-family: 'JetBrains Mono', 'Inter', monospace;
}
.kpi-subtext {
    font-size: 0.76rem;
    color: #059669;
    font-weight: 600;
    margin-top: 0.35rem;
}
.kpi-subtext.neutral {
    color: #64748B;
    font-weight: 500;
}

/* Narrative callout */
.narrative-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #334155;
    margin-bottom: 1.25rem;
}

/* Clean Form Controls */
div[data-baseweb="textarea"] {
    border-radius: 8px !important;
    border-color: #CBD5E1 !important;
}
div[data-baseweb="textarea"]:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
}

/* Control Toolbar Card */
.control-toolbar {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.3rem 0.8rem 1.3rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
}
.toolbar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.65rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid #F1F5F9;
}
.toolbar-title {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
}
.toolbar-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
}

/* Attention Card full width */
.attention-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    margin-top: 1.4rem;
}
.attention-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.9rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #F1F5F9;
}

div[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border-color: #CBD5E1 !important;
}
</style>
"""
st.markdown(PRO_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Import
# -----------------------------------------------------------------------------
try:
    from project_data import (
        ATTENTION_EXAMPLES,
        DATA_STATS,
        EDA_RESULTS,
        EVAL_RESULTS,
        HYPERPARAMS,
        LENGTH_QUALITY_RESULTS,
        MODEL_COMPARISON,
        SCALED_EVAL_RESULTS,
        SCALED_HYPERPARAMS,
        SCALED_TRAINING_HISTORY,
        TOKENIZER_STATS,
        TRAINING_HISTORY,
    )
    DATA_LOADED = True
except Exception:
    DATA_LOADED = False
    HYPERPARAMS, DATA_STATS, TOKENIZER_STATS = {}, {}, {}
    TRAINING_HISTORY, EVAL_RESULTS, EDA_RESULTS = [], {}, {}
    SCALED_TRAINING_HISTORY, SCALED_EVAL_RESULTS, MODEL_COMPARISON = [], {}, {}
    LENGTH_QUALITY_RESULTS = {}

# -----------------------------------------------------------------------------
# Scaled Transformer Definition (Pre-LN + Weight Tying)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Cached Loaders & Inference
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_baseline_components():
    from configs.base import CHECKPOINT_DIR
    from src.training.train import build_model
    from src.training.checkpoint import load_checkpoint
    from src.tokenization.tokenizer_utils import load_tokenizer
    from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH

    model = build_model()
    load_checkpoint(CHECKPOINT_DIR / "kaggle_run_best.pt", model)
    model.eval()
    en_tok = load_tokenizer(EN_TOKENIZER_PATH)
    or_tok = load_tokenizer(OR_TOKENIZER_PATH)
    return model, en_tok, or_tok

@st.cache_resource(show_spinner=False)
def load_scaled_components():
    ckpt_path = ROOT_DIR / "checkpoints" / "scaled_model_best.pt"
    en_tok_path = ROOT_DIR / "tokenizers" / "scaled_en_bpe.json"
    or_tok_path = ROOT_DIR / "tokenizers" / "scaled_or_bpe.json"

    model = EnhancedScaledTransformer()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    en_tok = Tokenizer.from_file(str(en_tok_path))
    or_tok = Tokenizer.from_file(str(or_tok_path))
    return model, en_tok, or_tok

def run_greedy_decode(model, src_tensor, max_len=64, no_repeat_size=3):
    from configs.base import EOS_ID, SOS_ID
    from src.inference.repetition import banned_ngram_tokens

    tgt_ids = torch.tensor([[SOS_ID]], dtype=torch.long, device=src_tensor.device)
    for _ in range(max_len - 1):
        with torch.no_grad():
            logits = model(src_tensor, tgt_ids)
        next_logits = logits[:, -1, :].clone()

        if no_repeat_size > 0:
            banned = banned_ngram_tokens(tgt_ids[0].tolist(), no_repeat_size)
            for token_id in banned:
                next_logits[:, token_id] = float("-inf")

        next_id = next_logits.argmax(dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break
    return tgt_ids[0].tolist()

def run_beam_search(model, src_tensor, beam_width=4, length_penalty=0.6, max_len=64, no_repeat_size=3):
    from configs.base import EOS_ID, SOS_ID
    from src.inference.repetition import banned_ngram_tokens

    device = src_tensor.device
    beams = [([SOS_ID], 0.0)]
    completed = []

    for _ in range(max_len - 1):
        candidates = []
        for seq, cum_logprob in beams:
            if seq[-1] == EOS_ID:
                completed.append((seq, cum_logprob))
                continue

            tgt_ids = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(src_tensor, tgt_ids)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

            if no_repeat_size > 0:
                banned = banned_ngram_tokens(seq, no_repeat_size)
                for token_id in banned:
                    log_probs[token_id] = float("-inf")

            top_logprobs, top_ids = log_probs.topk(beam_width)
            for logprob, token_id in zip(top_logprobs.tolist(), top_ids.tolist()):
                candidates.append((seq + [token_id], cum_logprob + logprob))

        if not candidates:
            break

        candidates.sort(key=lambda c: c[1] / (len(c[0]) ** length_penalty), reverse=True)
        beams = candidates[:beam_width]

    if not completed:
        completed = beams
    completed.sort(key=lambda c: c[1] / (len(c[0]) ** length_penalty), reverse=True)
    return completed[0][0]

def run_attention_decode(model, src_tensor, is_scaled=False, max_len=64):
    from configs.base import EOS_ID, SOS_ID
    from src.inference.repetition import banned_ngram_tokens

    device = src_tensor.device
    tgt_ids = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    attention_rows = []

    for _ in range(max_len - 1):
        with torch.no_grad():
            logits = model(src_tensor, tgt_ids)
        next_logits = logits[:, -1, :].clone()

        banned = banned_ngram_tokens(tgt_ids[0].tolist(), 3)
        for token_id in banned:
            next_logits[:, token_id] = float("-inf")

        next_id = next_logits.argmax(dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_id], dim=1)

        if is_scaled:
            cross_attn = model.decoder_blocks[-1].cross_attn.last_attn_weights
        else:
            cross_attn = model.decoder.layers[-1].cross_attn.last_attn_weights

        if cross_attn is not None:
            row = cross_attn.mean(dim=1)[0, -1, :]
            attention_rows.append(row.tolist())

        if next_id.item() == EOS_ID:
            break

    return tgt_ids[0].tolist(), attention_rows

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-navbar">
        <div class="app-title-group">
            <h1>English &rarr; Odia Neural Machine Translation</h1>
            <p>From-scratch PyTorch Sequence-to-Sequence Transformer research platform &bull; AI4Bharat Samanantar Corpus</p>
        </div>
        <div class="nav-badges">
            <span class="status-indicator"><span class="status-dot"></span>Models Ready</span>
            <span class="tag-badge">Baseline: 4.0M</span>
            <span class="tag-badge">Scaled: 11.5M</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top-Level Tabs
# -----------------------------------------------------------------------------
tab_translate, tab_comparison, tab_benchmarks, tab_arch = st.tabs(
    [
        "Translator",
        "Model Comparison",
        "Training & Benchmarks",
        "Architecture & Linguistics",
    ]
)

# =============================================================================
# TAB 1: TRANSLATOR (DeepL / Vercel style dual-pane console)
# =============================================================================
with tab_translate:
    # -------------------------------------------------------------------------
    # Top Control Toolbar: Architecture, Decoding, and Inspector Toggles
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="control-toolbar">
            <div class="toolbar-header">
                <span class="toolbar-title">Workbench Configuration &amp; Inference Pipeline</span>
                <span class="toolbar-meta">Runtime: PyTorch CPU &bull; Dynamic Positional Encodings Active</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.6, 1.3, 1.1], gap="medium")
    with ctrl_col1:
        st.markdown("<div style='font-size:0.73rem;font-weight:700;color:#64748B;margin-bottom:0.25rem;text-transform:uppercase;letter-spacing:0.04em;'>Model Architecture</div>", unsafe_allow_html=True)
        model_mode = st.selectbox(
            "Evaluation Model",
            options=["Scaled GPU Model (11.5M - Recommended)", "Baseline Model (4.0M - §5.6 Baseline)", "Side-by-Side Dual Evaluation"],
            index=0,
            label_visibility="collapsed",
        )
    with ctrl_col2:
        st.markdown("<div style='font-size:0.73rem;font-weight:700;color:#64748B;margin-bottom:0.25rem;text-transform:uppercase;letter-spacing:0.04em;'>Decoding Algorithm</div>", unsafe_allow_html=True)
        decoding_choice = st.selectbox(
            "Decoding Strategy",
            options=["Beam Search (k=4, α=0.6)", "Greedy Search (Argmax)"],
            index=0,
            label_visibility="collapsed",
        )
    with ctrl_col3:
        st.markdown("<div style='font-size:0.73rem;font-weight:700;color:#64748B;margin-bottom:0.25rem;text-transform:uppercase;letter-spacing:0.04em;'>Inspection Tools</div>", unsafe_allow_html=True)
        t_c1, t_c2 = st.columns(2)
        with t_c1:
            show_subwords = st.toggle("Tokens", value=True, help="Display subword token counts and expansion ratio")
        with t_c2:
            show_attention_map = st.toggle("Heatmap", value=False, help="Render full-width cross-attention alignment matrix")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Quick Evaluation Presets
    # -------------------------------------------------------------------------
    if "source_text" not in st.session_state:
        st.session_state.source_text = "The weather is very nice today."

    def apply_preset(prompt: str):
        st.session_state.source_text = prompt

    st.markdown("<div style='font-size:0.73rem;font-weight:700;color:#64748B;margin:0.8rem 0 0.35rem 0;text-transform:uppercase;letter-spacing:0.05em;'>Quick Evaluation Presets:</div>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    with chip_col1:
        st.button("Daily / Weather", on_click=apply_preset, args=("The weather is very nice today.",), use_container_width=True)
    with chip_col2:
        st.button("Morphology / Compound", on_click=apply_preset, args=("The former minister was present at the meeting.",), use_container_width=True)
    with chip_col3:
        st.button("Education / Concept", on_click=apply_preset, args=("Education is essential for everyone.",), use_container_width=True)
    with chip_col4:
        st.button("News / Public Domain", on_click=apply_preset, args=("The police reached the spot immediately.",), use_container_width=True)

    # -------------------------------------------------------------------------
    # Dual-Pane Translation Workbench
    # -------------------------------------------------------------------------
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(
            """
            <div class="panel-header">
                <span class="panel-title">Source Language: English</span>
                <span class="panel-meta">Latin Script</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user_text = st.text_area(
            "English Input",
            key="source_text",
            height=165,
            label_visibility="collapsed",
            placeholder="Type or paste English text to translate...",
        )
        char_count = len(user_text)
        word_count = len(user_text.split()) if user_text else 0
        st.markdown(
            f"<div style='font-family:JetBrains Mono;font-size:0.74rem;color:#64748B;text-align:right;margin-top:0.4rem;'>"
            f"{word_count} words &bull; {char_count} characters"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Attention and translation state containers
    attn_matrix = None
    attn_src_toks = None
    attn_tgt_toks = None
    attn_model_name = "Scaled GPU Model (11.5M)"

    with col_output:
        st.markdown(
            """
            <div class="panel-header">
                <span class="panel-title">Target Language: Odia (ଓଡ଼ିଆ)</span>
                <span class="panel-meta">Brahmic Script</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if user_text.strip():
            # Load appropriate models
            need_base = "Baseline" in model_mode or "Side-by-Side" in model_mode
            need_scaled = "Scaled" in model_mode or "Side-by-Side" in model_mode

            base_m, base_en_t, base_or_t = None, None, None
            scaled_m, scaled_en_t, scaled_or_t = None, None, None

            if need_base:
                base_m, base_en_t, base_or_t = load_baseline_components()
            if need_scaled:
                scaled_m, scaled_en_t, scaled_or_t = load_scaled_components()

            # -------------------------------------------------------------
            # Execution: Side-by-Side View
            # -------------------------------------------------------------
            if "Side-by-Side" in model_mode and base_m and scaled_m:
                # 1. Scaled Model Execution
                s_t0 = time.time()
                s_ids = scaled_en_t.encode(user_text).ids
                s_src = torch.tensor([s_ids])
                if show_attention_map:
                    s_out, attn_matrix = run_attention_decode(scaled_m, s_src, is_scaled=True, max_len=96)
                    attn_model_name = "Scaled GPU Model (11.5M)"
                    attn_src_toks = [scaled_en_t.decode([i], skip_special_tokens=False) or "?" for i in s_ids]
                    attn_tgt_toks = [scaled_or_t.decode([i], skip_special_tokens=False) or "?" for i in s_out[1:]]
                elif "Beam" in decoding_choice:
                    s_out = run_beam_search(scaled_m, s_src, beam_width=4, max_len=96)
                else:
                    s_out = run_greedy_decode(scaled_m, s_src, max_len=96)
                s_text = scaled_or_t.decode(s_out, skip_special_tokens=True).strip()
                s_latency = (time.time() - s_t0) * 1000

                # 2. Baseline Model Execution
                from src.tokenization.tokenizer_utils import encode as b_encode, decode as b_decode
                b_t0 = time.time()
                b_ids = b_encode(base_en_t, user_text)
                b_src = torch.tensor([b_ids])
                if "Beam" in decoding_choice:
                    b_out = run_beam_search(base_m, b_src, beam_width=4, max_len=64)
                else:
                    b_out = run_greedy_decode(base_m, b_src, max_len=64)
                b_text = b_decode(base_or_t, b_out)
                b_latency = (time.time() - b_t0) * 1000

                # Render Scaled Card
                st.markdown(
                    f"""
                    <div class="target-box scaled-box">
                        <div class="target-header">
                            <span class="model-pill scaled">Scaled GPU Model (11.5M)</span>
                            <span style="font-family:JetBrains Mono;font-size:0.72rem;color:#64748B;">{s_latency:.1f}ms</span>
                        </div>
                        <p class="odia-text">{s_text}</p>
                        <div class="meta-chip-row">
                            <span class="meta-chip">Tokens: {len(s_ids)} in &rarr; {len(s_out)} out</span>
                            <span class="meta-chip">Ratio: {len(s_out)/max(1,len(s_ids)):.2f}&times;</span>
                            <span class="meta-chip">Pre-LN &bull; SWA</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Render Baseline Card
                st.markdown(
                    f"""
                    <div class="target-box highlight">
                        <div class="target-header">
                            <span class="model-pill baseline">Baseline Model (4.0M)</span>
                            <span style="font-family:JetBrains Mono;font-size:0.72rem;color:#64748B;">{b_latency:.1f}ms</span>
                        </div>
                        <p class="odia-text">{b_text}</p>
                        <div class="meta-chip-row">
                            <span class="meta-chip">Tokens: {len(b_ids)} in &rarr; {len(b_out)} out</span>
                            <span class="meta-chip">Ratio: {len(b_out)/max(1,len(b_ids)):.2f}&times;</span>
                            <span class="meta-chip">Post-LN &bull; Course §5.6</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------------------
            # Execution: Single Model View
            # -------------------------------------------------------------
            else:
                is_scaled = "Scaled" in model_mode
                active_model = scaled_m if is_scaled else base_m
                card_class = "target-box scaled-box" if is_scaled else "target-box highlight"
                pill_class = "model-pill scaled" if is_scaled else "model-pill baseline"
                model_label = "Scaled GPU Model (11.5M)" if is_scaled else "Baseline Model (4.0M)"
                max_l = 96 if is_scaled else 64

                t0 = time.time()

                if is_scaled:
                    src_ids = scaled_en_t.encode(user_text).ids
                    src_t = torch.tensor([src_ids])
                    if show_attention_map:
                        out_ids, attn_matrix = run_attention_decode(active_model, src_t, is_scaled=True, max_len=max_l)
                        attn_model_name = "Scaled GPU Model (11.5M)"
                        attn_src_toks = [scaled_en_t.decode([i], skip_special_tokens=False) or "?" for i in src_ids]
                        attn_tgt_toks = [scaled_or_t.decode([i], skip_special_tokens=False) or "?" for i in out_ids[1:]]
                    elif "Beam" in decoding_choice:
                        out_ids = run_beam_search(active_model, src_t, beam_width=4, max_len=max_l)
                    else:
                        out_ids = run_greedy_decode(active_model, src_t, max_len=max_l)
                    translated_text = scaled_or_t.decode(out_ids, skip_special_tokens=True).strip()
                else:
                    from src.tokenization.tokenizer_utils import encode as b_encode, decode as b_decode
                    src_ids = b_encode(base_en_t, user_text)
                    src_t = torch.tensor([src_ids])
                    if show_attention_map:
                        out_ids, attn_matrix = run_attention_decode(active_model, src_t, is_scaled=False, max_len=max_l)
                        attn_model_name = "Baseline Model (4.0M)"
                        attn_src_toks = [base_en_t.decode([i], skip_special_tokens=False) or "?" for i in src_ids]
                        attn_tgt_toks = [base_or_t.decode([i], skip_special_tokens=False) or "?" for i in out_ids[1:]]
                    elif "Beam" in decoding_choice:
                        out_ids = run_beam_search(active_model, src_t, beam_width=4, max_len=max_l)
                    else:
                        out_ids = run_greedy_decode(active_model, src_t, max_len=max_l)
                    translated_text = b_decode(base_or_t, out_ids)

                latency = (time.time() - t0) * 1000
                expansion = len(out_ids) / max(1, len(src_ids))

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="target-header">
                            <span class="{pill_class}">{model_label}</span>
                            <span style="font-family:JetBrains Mono;font-size:0.72rem;color:#64748B;">{latency:.1f}ms</span>
                        </div>
                        <p class="odia-text">{translated_text}</p>
                        <div class="meta-chip-row">
                            <span class="meta-chip">Source: {len(src_ids)} subwords</span>
                            <span class="meta-chip">Target: {len(out_ids)} subwords</span>
                            <span class="meta-chip">Expansion: {expansion:.2f}&times;</span>
                            <span class="meta-chip">{decoding_choice}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if show_subwords:
                    st.caption(f"Encoded into {len(src_ids)} source tokens, decoded into {len(out_ids)} Brahmic subwords.")
        else:
            st.info("Enter an English sentence above to inspect translation.")

    # -------------------------------------------------------------------------
    # Dedicated Full-Width Cross-Attention Alignment Inspector
    # -------------------------------------------------------------------------
    if show_attention_map and user_text.strip() and attn_matrix and attn_src_toks and attn_tgt_toks:
        st.markdown(
            f"""
            <div class="attention-card">
                <div class="attention-card-header">
                    <div>
                        <div style="font-size:0.92rem;font-weight:700;color:#0F172A;letter-spacing:-0.01em;">
                            Cross-Attention Alignment Heatmap &bull; {attn_model_name}
                        </div>
                        <div style="font-size:0.75rem;color:#64748B;margin-top:2px;">
                            Decoder cross-attention weight distribution (X: Source English subwords &rarr; Y: Generated Odia subwords).
                        </div>
                    </div>
                    <div style="display:flex;gap:0.4rem;align-items:center;">
                        <span class="tag-badge">Multi-Head Mean</span>
                        <span class="tag-badge">Decoder Final Block</span>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        rows = []
        for ti, t_str in enumerate(attn_tgt_toks):
            if ti >= len(attn_matrix):
                break
            for si, s_str in enumerate(attn_src_toks):
                if si >= len(attn_matrix[ti]):
                    break
                rows.append({
                    "Target Subword (Odia)": f"{ti+1:02d}: {t_str}",
                    "Source Token (English)": f"{si+1:02d}: {s_str}",
                    "t_idx": ti,
                    "s_idx": si,
                    "weight": float(attn_matrix[ti][si]),
                })

        if rows:
            hdf = pd.DataFrame(rows)
            max_w = max(0.35, float(hdf["weight"].max()))
            h_chart = (
                alt.Chart(hdf)
                .mark_rect(stroke="#FFFFFF", strokeWidth=0.5)
                .encode(
                    x=alt.X(
                        "Source Token (English):N",
                        sort=alt.SortField("s_idx"),
                        axis=alt.Axis(labelAngle=-30, labelFontSize=11, titleFontSize=12, labelColor="#334155"),
                        title="Source English Tokens (Input)",
                    ),
                    y=alt.Y(
                        "Target Subword (Odia):N",
                        sort=alt.SortField("t_idx"),
                        axis=alt.Axis(labelFontSize=11, titleFontSize=12, labelColor="#334155"),
                        title="Generated Odia Subwords (Target)",
                    ),
                    color=alt.Color(
                        "weight:Q",
                        scale=alt.Scale(scheme="blues", domain=[0, max_w]),
                        title="Attention Weight",
                    ),
                    tooltip=[
                        alt.Tooltip("Source Token (English)", title="Source Token"),
                        alt.Tooltip("Target Subword (Odia)", title="Generated Subword"),
                        alt.Tooltip("weight:Q", format=".4f", title="Attention Weight"),
                    ],
                )
                .properties(height=max(260, 22 * len(attn_tgt_toks)), background="#FFFFFF")
            )
            st.altair_chart(h_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2: MODEL COMPARISON (Option A Study)
# =============================================================================
with tab_comparison:
    st.markdown('<div class="view-heading">Empirical Comparison: Baseline (§5.6) vs. Scaled GPU Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="narrative-card">
            To assess the scaling headroom of our from-scratch PyTorch Transformer, we conducted an empirical
            investigation comparing the course-compliant <strong>Baseline Model</strong> (4.0M parameters, CPU-trained)
            against an <strong>Option A Scaled GPU Model</strong> (11.5M parameters, NVIDIA Tesla T4 GPU) featuring
            Pre-LayerNorm residuals, embedding-to-output weight tying, and Stochastic Weight Averaging (SWA).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Clean Executive Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">Model Capacity</div>
                <div class="kpi-value">11.5M</div>
                <div class="kpi-subtext">+7.5M params (2.87&times;)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">Validation Loss Floor</div>
                <div class="kpi-value">3.56</div>
                <div class="kpi-subtext">-0.40 cross-entropy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">Training Wall Time</div>
                <div class="kpi-value">23.0m</div>
                <div class="kpi-subtext">Tesla T4 GPU (FP16 AMP)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">Weight Tying Savings</div>
                <div class="kpi-value">2.05M</div>
                <div class="kpi-subtext neutral">Linear projection tied to Emb</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # 4-Panel Master Benchmark Figure
    fig_full = ROOT_DIR / "reports" / "figures" / "full_model_comparison.png"
    if fig_full.exists():
        st.image(str(fig_full), caption="Figure 1: Comprehensive Comparative Study Dashboard (Loss Curves, Parameter Allocations, Learning Rate Schedules & Sample BLEU)", use_container_width=True)

    fig_c1, fig_c2 = st.columns(2)
    fig_loss = ROOT_DIR / "reports" / "figures" / "comparison_loss_curves.png"
    fig_param = ROOT_DIR / "reports" / "figures" / "comparison_param_breakdown.png"
    with fig_c1:
        if fig_loss.exists():
            st.image(str(fig_loss), caption="Figure 2: Convergence Trajectories & Loss Basin Floor", use_container_width=True)
    with fig_c2:
        if fig_param.exists():
            st.image(str(fig_param), caption="Figure 3: Layer-by-Layer Parameter Distribution & Weight Tying", use_container_width=True)

    # Architectural Spec Table
    st.markdown('<div class="view-heading">Architectural & Hyperparameter Specification Matrix</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame([
        {"Specification": "Architecture Pattern", "Baseline (§5.6 Compliant)": "Post-LN Vanilla Transformer", "Scaled GPU Model (Option A)": "Pre-LN Deep Transformer", "Impact": "Prevents vanishing gradients in deep stacks"},
        {"Specification": "Layer Depth (N_enc + N_dec)", "Baseline (§5.6 Compliant)": "2 + 2 = 4 layers", "Scaled GPU Model (Option A)": "4 + 4 = 8 layers", "Impact": "2× hierarchical feature extraction"},
        {"Specification": "Hidden Model Dim (d_model)", "Baseline (§5.6 Compliant)": "128", "Scaled GPU Model (Option A)": "256", "Impact": "2× representational embedding width"},
        {"Specification": "Attention Heads (n_heads)", "Baseline (§5.6 Compliant)": "4 heads (d_k=32)", "Scaled GPU Model (Option A)": "8 heads (d_k=32)", "Impact": "2× multi-aspect syntactic attention"},
        {"Specification": "Feed-Forward Dim (d_ff)", "Baseline (§5.6 Compliant)": "512", "Scaled GPU Model (Option A)": "1024", "Impact": "2× non-linear sublayer capacity"},
        {"Specification": "Output Head Weight Tying", "Baseline (§5.6 Compliant)": "Untied (Independent linear head)", "Scaled GPU Model (Option A)": "Tied to target token embedding", "Impact": "Saves 2,048,000 redundant parameters"},
        {"Specification": "Vocabulary Budget", "Baseline (§5.6 Compliant)": "4,000 En / 4,000 Or", "Scaled GPU Model (Option A)": "8,000 En / 8,000 Or", "Impact": "Captures Odia multi-syllable compounds"},
        {"Specification": "Learning Rate Schedule", "Baseline (§5.6 Compliant)": "Noam Warmup + Inverse Sqrt", "Scaled GPU Model (Option A)": "Warmup (1200 st) + Cosine Anneal", "Impact": "Smooth decay down to 1e-6 floor"},
        {"Specification": "Checkpoint Selection", "Baseline (§5.6 Compliant)": "Single Best Epoch", "Scaled GPU Model (Option A)": "Stochastic Weight Averaging (Top 3)", "Impact": "Flatter loss basin and generalizability"},
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Qualitative Test Set Comparison
    if MODEL_COMPARISON and "qualitative_comparison" in MODEL_COMPARISON:
        st.markdown('<div class="view-heading">Qualitative Test Set Translation Evaluation</div>', unsafe_allow_html=True)
        st.caption("Side-by-side held-out test translations illustrating the elimination of repetition loops and improved compound word synthesis:")
        st.dataframe(pd.DataFrame(MODEL_COMPARISON["qualitative_comparison"]), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: TRAINING & BENCHMARKS
# =============================================================================
with tab_benchmarks:
    st.markdown('<div class="view-heading">Training Trajectory & Quantitative Test Benchmarks</div>', unsafe_allow_html=True)

    # Interactive Loss Curve
    if TRAINING_HISTORY:
        h_df = pd.DataFrame(TRAINING_HISTORY)
        melted = h_df.melt(id_vars="epoch", value_vars=["train_loss", "val_loss"], var_name="Split", value_name="CrossEntropyLoss")
        melted["Split"] = melted["Split"].map({"train_loss": "Train Loss", "val_loss": "Validation Loss"})

        loss_chart = (
            alt.Chart(melted)
            .mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=36, filled=True))
            .encode(
                x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("CrossEntropyLoss:Q", title="Cross-Entropy Loss", scale=alt.Scale(zero=False)),
                color=alt.Color("Split:N", scale=alt.Scale(domain=["Train Loss", "Validation Loss"], range=["#2563EB", "#D97706"])),
                tooltip=["epoch:Q", "Split:N", alt.Tooltip("CrossEntropyLoss:Q", format=".4f")],
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#F1F5F9", labelColor=TEXT_MUTED, titleColor=SLATE_900)
            .properties(height=340, background="#FFFFFF", title="Baseline Model 40-Epoch Training Trajectory")
        )
        st.altair_chart(loss_chart, use_container_width=True)

    # Test Set Metrics
    st.markdown('<div class="view-heading">Official Evaluation Metrics (SacreBLEU)</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    bleu_score = EVAL_RESULTS.get("bleu_score", 2.60) if EVAL_RESULTS else 2.60
    signature = EVAL_RESULTS.get("bleu_signature", "n/a") if EVAL_RESULTS else "n/a"
    dec_sec = EVAL_RESULTS.get("decode_seconds", 36.1) if EVAL_RESULTS else 36.1

    with m1:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">Held-Out Test BLEU</div>
                <div class="kpi-value">{bleu_score:.2f}</div>
                <div class="kpi-subtext neutral">SacreBLEU Standard Metric</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">Test Corpus Size</div>
                <div class="kpi-value">2,000</div>
                <div class="kpi-subtext neutral">Independent held-out split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">Greedy Decoding Time</div>
                <div class="kpi-value">{dec_sec:.1f}s</div>
                <div class="kpi-subtext neutral">Across 2,000 test sentences</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sequence Length Quality Analysis
    if LENGTH_QUALITY_RESULTS:
        st.markdown('<div class="view-heading">Quality Degradation Across Sequence Lengths</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="narrative-card">
                Empirical evaluation across all 2,000 test sentences shows that BLEU degrades on longer source sequences
                due to degenerate repetition loops in greedy decoding. Implementing <strong>no-repeat n-gram blocking (n=3)</strong>
                and <strong>length-penalized beam search</strong> effectively eliminates these degenerate loops.
            </div>
            """,
            unsafe_allow_html=True,
        )
        buckets = LENGTH_QUALITY_RESULTS.get("buckets_by_word_len", [])
        if buckets:
            b_df = pd.DataFrame(buckets)
            bc1, bc2 = st.columns(2)
            with bc1:
                c_bleu = (
                    alt.Chart(b_df)
                    .mark_bar(color="#2563EB", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("bucket:N", title="Sentence Length (Words)", sort=[b["bucket"] for b in buckets]),
                        y=alt.Y("mean_bleu:Q", title="Mean Sentence BLEU"),
                        tooltip=["bucket", "count", alt.Tooltip("mean_bleu:Q", format=".2f")],
                    )
                    .properties(height=260, title="BLEU Score vs. Source Length")
                )
                st.altair_chart(c_bleu, use_container_width=True)
            with bc2:
                c_rep = (
                    alt.Chart(b_df)
                    .mark_bar(color="#DC2626", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("bucket:N", title="Sentence Length (Words)", sort=[b["bucket"] for b in buckets]),
                        y=alt.Y("repetition_rate_pct:Q", title="Degenerate Repetition Rate (%)"),
                        tooltip=["bucket", "count", alt.Tooltip("repetition_rate_pct:Q", format=".1f")],
                    )
                    .properties(height=260, title="Repetition Rate vs. Source Length")
                )
                st.altair_chart(c_rep, use_container_width=True)

# =============================================================================
# TAB 4: ARCHITECTURE & LINGUISTICS
# =============================================================================
with tab_arch:
    st.markdown('<div class="view-heading">Brahmic Script Tokenization & Subword Asymmetry</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="narrative-card">
            <strong>The Brahmic Script Asymmetry:</strong> Odia is written in an abugida script where consonants carry
            inherent vowels and combine with dependent vowel signs (ମାତ୍ରା) and consonant conjuncts (ଯୁକ୍ତାକ୍ଷର).
            Because subword tokenization operates on multi-byte UTF-8 sequences, an Odia sentence consistently decomposes into
            <strong>2.5&times; to 3.0&times; more subwords</strong> than its English counterpart for the same semantic content.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Subword distribution chart
    en_stats = TOKENIZER_STATS.get("english", {}) if isinstance(TOKENIZER_STATS, dict) else {}
    or_stats = TOKENIZER_STATS.get("odia", {}) if isinstance(TOKENIZER_STATS, dict) else {}
    stats_order = ["mean", "median", "p90", "p95", "p99", "max"]
    plot_rows = []
    for s in stats_order:
        if s in en_stats:
            plot_rows.append({"metric": s, "Language": "English", "Subwords": en_stats.get(s)})
        if s in or_stats:
            plot_rows.append({"metric": s, "Language": "Odia", "Subwords": or_stats.get(s)})

    if plot_rows:
        pdf = pd.DataFrame(plot_rows)
        token_chart = (
            alt.Chart(pdf)
            .mark_bar(size=18, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("metric:N", sort=stats_order, title="Percentile / Metric", axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("Language:N", sort=["English", "Odia"]),
                y=alt.Y("Subwords:Q", title="Subword Token Count"),
                color=alt.Color("Language:N", scale=alt.Scale(domain=["English", "Odia"], range=["#2563EB", "#D97706"])),
                tooltip=["Language", "metric", "Subwords"],
            )
            .properties(height=300, background="#FFFFFF", title="Empirical Subword Length Distribution Comparison")
        )
        st.altair_chart(token_chart, use_container_width=True)

    # Pipeline Pillars
    st.markdown('<div class="view-heading">Data Pipeline & Engineering Methodology</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">1. Text Normalization</div>
                <div style="font-size:0.86rem;line-height:1.6;color:#334155;margin-top:0.4rem;">
                    &bull; <strong>Unicode NFC:</strong> Eliminates duplicate BPE tokens from decomposing code points.<br/>
                    &bull; <strong>Joiner Preservation:</strong> Retains ZWJ/ZWNJ for Indic conjuncts.<br/>
                    &bull; <strong>Noise Filtration:</strong> Removes ZWSP, BOM, and malformed pairs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">2. Byte-Level BPE</div>
                <div style="font-size:0.86rem;line-height:1.6;color:#334155;margin-top:0.4rem;">
                    &bull; <strong>0.0% UNK Guarantee:</strong> Raw byte fallback prevents out-of-vocabulary crashes.<br/>
                    &bull; <strong>Dual Vocabularies:</strong> Dedicated vocab budgets (4k/8k) per script.<br/>
                    &bull; <strong>Template Wrapping:</strong> Automatic atomic <code>&lt;SOS&gt;</code> / <code>&lt;EOS&gt;</code> tagging.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="kpi-container">
                <div class="kpi-title">3. Regularization &amp; Decoding</div>
                <div style="font-size:0.86rem;line-height:1.6;color:#334155;margin-top:0.4rem;">
                    &bull; <strong>Label Smoothing (0.1):</strong> Calibrates cross-entropy loss against overconfidence.<br/>
                    &bull; <strong>Beam Search (k=4):</strong> Length-normalized search over output space.<br/>
                    &bull; <strong>3-Gram Repetition Block:</strong> Dynamic masking prevents degenerate subword loops.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-top:2.5rem;border-top:1px solid #E2E8F0;padding-top:1rem;text-align:center;color:#94A3B8;font-size:0.78rem;'>English &rarr; Odia Neural Machine Translation Platform &bull; Academic Research Implementation &bull; PyTorch</div>", unsafe_allow_html=True)
