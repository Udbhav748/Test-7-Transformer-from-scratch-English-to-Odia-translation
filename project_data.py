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
    EN_VOCAB_SIZE,
    OR_VOCAB_SIZE,
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

ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "reports"

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
    # Measured from the trained model, not derivable from config alone.
    "total_params": 4_005_696,
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
    "retention_rate_pct": 77.0,
    "retention_survivors": 44673,
}


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_tokenizer_pilot_stats = _load_json(REPORTS_DIR / "tokenizer_pilot_stats.json")
TOKENIZER_STATS = {
    **_tokenizer_pilot_stats,
    "en_vocab_size": EN_VOCAB_SIZE,
    "or_vocab_size": OR_VOCAB_SIZE,
}

TRAINING_HISTORY = _load_json(REPORTS_DIR / "training_history.json")

EVAL_RESULTS = _load_json(REPORTS_DIR / "eval_results.json")
_samples = EVAL_RESULTS.get("samples", [])
EVAL_RESULTS["long_sentence_index"] = max(
    range(len(_samples)), key=lambda i: len(_samples[i]["source"])
) if _samples else None
