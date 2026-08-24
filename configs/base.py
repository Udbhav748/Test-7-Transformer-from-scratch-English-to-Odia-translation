"""Frozen hyperparameters and paths for the Test-7 En->Odia transformer.

Numbers below are measured, not assumed -- see the MAX_LEN retention
measurement against a real Samanantar sample (English mean/median 17.5/13
subwords, Odia mean/median 49.8/39 subwords, 78.0% pair-retention at
MAX_LEN=64) before changing SUBSET_SIZE or MAX_LEN.
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
# pairs after ~78% measured retention at MAX_LEN=64.
CANDIDATE_POOL_SIZE = 58_000

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
LABEL_SMOOTHING = 0.1

# --- Inference ---
GREEDY_MAX_DECODE_LEN = MAX_LEN
BEAM_WIDTH = 4
BEAM_LENGTH_PENALTY = 0.6
# Blocks a repeat of any n-gram of this size already generated in the same
# sequence -- a standard inference-time fix for the degenerate repetition
# loops small greedy-decoded models fall into, with no retraining needed.
NO_REPEAT_NGRAM_SIZE = 3

# --- Evaluation ---
NUM_SAMPLE_TRANSLATIONS = 5
LONG_SENTENCE_PERCENTILE = 0.90
