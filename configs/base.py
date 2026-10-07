"""Frozen hyperparameters and paths for the Test-7 En->Odia transformer.

Numbers below are measured, not assumed. A small early pilot tokenizer
(reports/tokenizer_pilot_stats.json, ~500 sentences) measured English
mean/median 17.5/13 subwords and Odia mean/median 49.8/39 subwords; that
pilot is not the production tokenizer and should not be read as the real
corpus statistics. The real, full-corpus measurement, from the trained
production tokenizers (reports/eda_results.json), is English mean/median
13.1/12.0 subwords, Odia mean/median 35.4/34.0 subwords, with 76.4%
pair-retention at MAX_LEN=64. Re-run scripts/run_eda.py and check that
file before changing TOTAL_SIZE, CANDIDATE_POOL_SIZE, or MAX_LEN.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TOKENIZER_DIR = REPO_ROOT / "tokenizers"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
REPORTS_DIR = REPO_ROOT / "reports"

# --- Dataset ---
HF_DATASET_ID = "ai4bharat/samanantar"
HF_DATASET_CONFIG = "or"
RANDOM_SEED = 42

# Word-count filter applied before tokenization (cheap first pass).
MIN_WORDS = 3
MAX_WORDS = 60

# Final target counts (post MAX_LEN filtering).
TRAIN_SIZE = 36_000
VAL_SIZE = 2_000
TEST_SIZE = 2_000
TOTAL_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE  # 40,000

# Over-collection target before the MAX_LEN filter, to net TOTAL_SIZE
# pairs after ~78% measured retention at MAX_LEN=64, once the
# TOKENIZER_POOL_SIZE pairs reserved for tokenizer training are removed.
CANDIDATE_POOL_SIZE = 75_000
# Random candidates reserved for BPE training only; split.py never lets
# these pairs (or any candidate sharing a sentence with them) reach val/test.
TOKENIZER_POOL_SIZE = 15_000

# --- Tokenization ---
EN_VOCAB_SIZE = 8_000
OR_VOCAB_SIZE = 8_000
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
MAX_LEN = 64  # subword tokens, including <SOS>/<EOS>

# --- Model (implements assignment section 5.6) ---
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
DROPOUT = 0.1
TIE_OUTPUT_PROJECTION = False  # assignment specifies "linear+softmax"; tying is an optional extra

# --- Training ---
BATCH_SIZE_KAGGLE = 128
BATCH_SIZE_LOCAL_SMOKE = 8
# Bumped from 18: the first real run's val loss was still decreasing every
# epoch with no sign of plateauing (1.9257 at epoch 18), so more epochs on
# the same d=128/heads=4/N=2 architecture is real headroom, not just noise.
NUM_EPOCHS_KAGGLE = 40
NUM_EPOCHS_LOCAL_SMOKE = 3
ADAM_BETAS = (0.9, 0.98)
ADAM_EPS = 1e-9
WARMUP_STEPS = 900
GRAD_CLIP_NORM = 1.0

# --- Inference ---
GREEDY_MAX_DECODE_LEN = MAX_LEN

# --- Evaluation ---
NUM_SAMPLE_TRANSLATIONS = 5
LONG_SENTENCE_PERCENTILE = 0.90
