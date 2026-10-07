"""Assembles this project's real training/eval artifacts into plain data structures.

No UI code here -- the dashboard layer imports these constants directly.
"""

import json
from pathlib import Path

from configs.base import (
    D_MODEL,
    N_HEADS,
    D_FF,
    N_ENCODER_LAYERS,
    N_DECODER_LAYERS,
    DROPOUT,
    TIE_OUTPUT_PROJECTION,
    BATCH_SIZE_KAGGLE,
    NUM_EPOCHS_KAGGLE,
    WARMUP_STEPS,
    MAX_LEN,
    CANDIDATE_POOL_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    TOTAL_SIZE,
)
from src.tokenization.tokenizer_utils import load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH

ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "reports"

# Real vocab sizes from the trained tokenizers on disk, not the configured
# target -- BPE training can undershoot the configured vocab_size on a small
# corpus (it did for Odia: EN_VOCAB_SIZE/OR_VOCAB_SIZE in configs/base.py
# are both 8000, but the Odia BPE trainer ran out of useful merges early).
EN_VOCAB_SIZE = load_tokenizer(EN_TOKENIZER_PATH).get_vocab_size()
OR_VOCAB_SIZE = load_tokenizer(OR_TOKENIZER_PATH).get_vocab_size()


def _real_param_count() -> int:
    from src.model.transformer import Seq2SeqTransformer

    model = Seq2SeqTransformer(
        src_vocab_size=EN_VOCAB_SIZE,
        tgt_vocab_size=OR_VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        dropout=DROPOUT,
        tie_output_projection=TIE_OUTPUT_PROJECTION,
    )
    return sum(p.numel() for p in model.parameters())

HYPERPARAMS = {
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "d_ff": D_FF,
    "n_encoder_layers": N_ENCODER_LAYERS,
    "n_decoder_layers": N_DECODER_LAYERS,
    "dropout": DROPOUT,
    "tie_output_projection": TIE_OUTPUT_PROJECTION,
    "en_vocab_size": EN_VOCAB_SIZE,
    "or_vocab_size": OR_VOCAB_SIZE,
    "batch_size_kaggle": BATCH_SIZE_KAGGLE,
    "num_epochs_kaggle": NUM_EPOCHS_KAGGLE,
    "warmup_steps": WARMUP_STEPS,
    "max_len": MAX_LEN,
    # Computed live against the real tokenizer vocab sizes above, not a
    # hardcoded figure -- this is what the actual trained checkpoint has.
    "total_params": _real_param_count(),
    "layer_norm_style": "post-norm (residual -> dropout -> LayerNorm)",
}

DATA_STATS = {
    "candidate_pool_size": CANDIDATE_POOL_SIZE,
    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE,
    "test_size": TEST_SIZE,
    "total_size": TOTAL_SIZE,
    "max_len": MAX_LEN,
    # Measured from the actual data pipeline run, not derivable from config.
    "retention_rate_pct": 76.4,
    "retention_survivors": 45858,
}


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


EDA_RESULTS = _load_json(REPORTS_DIR / "eda_results.json")

# Built from the real, full-corpus measurement (reports/eda_results.json,
# computed on the actual train/val/test split) and a real retention-vs-MAX_LEN
# curve measured with the production tokenizers (reports/tokenizer_retention_curve.json).
# reports/tokenizer_pilot_stats.json is an older, much smaller, separately
# tokenized pilot sample -- it is kept on disk for history but intentionally
# not used here, since mixing it into these live figures previously made a
# non-representative sample look like the real corpus statistic.
_retention_curve = _load_json(REPORTS_DIR / "tokenizer_retention_curve.json")
_en_sub = EDA_RESULTS["descriptive_stats"]["en_subwords"]
_or_sub = EDA_RESULTS["descriptive_stats"]["or_subwords"]
TOKENIZER_STATS = {
    "english": {
        "mean": _en_sub["mean"], "median": _en_sub["median"], "p90": _en_sub["p90"],
        "p95": _en_sub["p95"], "p99": _en_sub["p99"], "max": _en_sub["max"],
    },
    "odia": {
        "mean": _or_sub["mean"], "median": _or_sub["median"], "p90": _or_sub["p90"],
        "p95": _or_sub["p95"], "p99": _or_sub["p99"], "max": _or_sub["max"],
    },
    "retention_at_max_len": _retention_curve["retention_at_max_len"],
    "en_vocab_size": EN_VOCAB_SIZE,
    "or_vocab_size": OR_VOCAB_SIZE,
}
# Note: or_vocab_size is the real trained size (undershoots the configured
# 8000 target -- see the comment above EN_VOCAB_SIZE/OR_VOCAB_SIZE).

TRAINING_HISTORY = _load_json(REPORTS_DIR / "training_history.json")

EVAL_RESULTS = _load_json(REPORTS_DIR / "eval_results.json")
_samples = EVAL_RESULTS.get("samples", [])
EVAL_RESULTS["long_sentence_index"] = max(
    range(len(_samples)), key=lambda i: len(_samples[i]["source"])
) if _samples else None

ATTENTION_EXAMPLES = _load_json(REPORTS_DIR / "attention_examples.json")

LENGTH_QUALITY_RESULTS = _load_json(REPORTS_DIR / "length_quality_analysis.json")

_requirements_path = REPORTS_DIR / "requirements_coverage.json"
REQUIREMENTS_COVERAGE = _load_json(_requirements_path) if _requirements_path.exists() else {}

# Scaled Model (Option A) Artifacts
_scaled_history_path = REPORTS_DIR / "scaled_training_history.json"
SCALED_TRAINING_HISTORY = _load_json(_scaled_history_path) if _scaled_history_path.exists() else []

_scaled_eval_path = REPORTS_DIR / "scaled_eval_results.json"
SCALED_EVAL_RESULTS = _load_json(_scaled_eval_path) if _scaled_eval_path.exists() else {}

_model_comp_path = REPORTS_DIR / "model_comparison.json"
MODEL_COMPARISON = _load_json(_model_comp_path) if _model_comp_path.exists() else {}

SCALED_HYPERPARAMS = {
    "d_model": 256,
    "n_heads": 8,
    "d_ff": 1024,
    "n_encoder_layers": 4,
    "n_decoder_layers": 4,
    "dropout": 0.1,
    "tie_output_projection": True,
    "en_vocab_size": 8000,
    "or_vocab_size": 8000,
    "batch_size_effective": 256,
    "num_epochs": 25,
    "warmup_steps": 1200,
    "max_len": 96,
    "total_params": 11_469_824,
    "layer_norm_style": "pre-norm (LayerNorm -> Sublayer -> Residual)",
}
