import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NB_PATH = REPO_ROOT / "notebooks" / "kaggle" / "en_or_transformer.ipynb"

# Dependency order: every module here is read verbatim from disk (the exact
# code that already passed local unit tests) and reconstructed byte-for-byte
# on Kaggle, rather than hand-duplicated as separate literal cells, so there
# is no way for the notebook to silently drift from what was tested locally.
MODULES = [
    "configs/base.py",
    "src/data/clean.py",
    "src/data/download.py",
    "src/tokenization/train_tokenizer.py",
    "src/tokenization/tokenizer_utils.py",
    "src/data/split.py",
    "src/data/dataset.py",
    "src/model/masks.py",
    "src/model/embeddings.py",
    "src/model/attention.py",
    "src/model/feedforward.py",
    "src/model/encoder.py",
    "src/model/decoder.py",
    "src/model/transformer.py",
    "src/training/lr_schedule.py",
    "src/training/checkpoint.py",
    "src/training/train.py",
    "src/inference/greedy_decode.py",
    "src/evaluation/bleu.py",
    "src/evaluation/sample_translations.py",
]

# scripts/run_evaluation.py is embedded and run verbatim (not reimplemented
# inline) so the notebook's evaluation is provably the exact same code path
# as `python scripts/run_evaluation.py` on the CLI -- one evaluation
# implementation, not two that could silently drift apart.
SCRIPTS = [
    "scripts/run_evaluation.py",
]


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.strip().split("\n")]}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def read_module_texts():
    texts = {}
    for rel_path in MODULES + SCRIPTS:
        full_path = REPO_ROOT / rel_path
        texts[rel_path] = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
    return texts


def build_materialize_cell(texts):
    packed = {path: base64.b64encode(src.encode("utf-8")).decode("ascii") for path, src in texts.items()}
    packed_literal = json.dumps(packed)
    return code(f"""
import base64
from pathlib import Path

REPO_ROOT = Path("/kaggle/working")
_FILES_B64 = {packed_literal}

for rel_path, b64_src in _FILES_B64.items():
    dest = REPO_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64_src))

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"materialized {{len(_FILES_B64)}} files under {{REPO_ROOT}}")
""")


def main():
    texts = read_module_texts()

    cells = [
        md("""# Test-7 -- From-Scratch Transformer, English to Odia
Encoder-decoder transformer implemented per assignment section 5.6
(d_model=128, heads=4, N=2 encoder/decoder blocks). This notebook reconstructs
the exact source files developed and unit-tested locally, then runs the real
training pass on GPU, then evaluates the best checkpoint with plain greedy
decoding (the graded headline path -- not beam search, which is bonus
exploration kept in `extras/`)."""),
        code("""
# Pinned to the exact versions in requirements.txt, not whatever Kaggle's
# image happens to have, so the notebook installs the same dependencies
# that were used for local development and testing. Intentionally not the
# full requirements.txt -- that also pulls in UI/test-only packages this
# notebook doesn't need.
!pip install -q datasets==5.0.1 tokenizers==0.22.2 sacrebleu==2.6.0
"""),
        build_materialize_cell(texts),
        md("""## Data
Load the pre-processed dataset from the attached Kaggle dataset input if
present; otherwise fall back to running the download/clean/tokenize/split
pipeline inline (requires internet, enabled on this kernel)."""),
        code("""
import shutil
from pathlib import Path

KAGGLE_INPUT_CANDIDATES = list(Path("/kaggle/input").glob("*/processed"))
DATA_PROCESSED_DIR = Path("/kaggle/working/data/processed")
TOKENIZER_DIR = Path("/kaggle/working/tokenizers")

if KAGGLE_INPUT_CANDIDATES:
    src_dir = KAGGLE_INPUT_CANDIDATES[0]
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for f in src_dir.glob("*.parquet"):
        shutil.copy(f, DATA_PROCESSED_DIR / f.name)
    input_root = src_dir.parent
    tok_src = input_root / "tokenizers"
    if tok_src.exists():
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        for f in tok_src.glob("*.json"):
            shutil.copy(f, TOKENIZER_DIR / f.name)
    print(f"copied processed data + tokenizers from {src_dir.parent}")
else:
    print("no Kaggle dataset input found, running the pipeline inline")
    from src.data.download import main as download_main
    from src.tokenization.train_tokenizer import main as train_tokenizer_main
    from src.data.split import main as split_main

    download_main()
    train_tokenizer_main()
    split_main()
"""),
        md("## Train (Kaggle GPU profile: full dataset, larger batch size, full epoch count)"),
        code("""
import torch
from configs.base import (
    BATCH_SIZE_KAGGLE,
    DATA_PROCESSED_DIR,
    NUM_EPOCHS_KAGGLE,
)
from src.training.train import train

def select_device():
    # torch.cuda.is_available() only checks driver presence, not whether the
    # installed torch build actually has compiled kernels for the assigned
    # GPU's compute architecture (older cards like the P100 can fail with
    # "no kernel image is available" on a build that dropped that arch) --
    # a real matmul probe catches that, is_available() alone would not.
    if not torch.cuda.is_available():
        return "cpu"
    try:
        probe = torch.randn(8, 8, device="cuda")
        _ = probe @ probe
        torch.cuda.synchronize()
        return "cuda"
    except Exception as e:
        print(f"CUDA reported available but unusable ({e}); falling back to CPU")
        return "cpu"

device = select_device()
print(f"training on device: {device}")

model, history = train(
    train_path=DATA_PROCESSED_DIR / "train.parquet",
    val_path=DATA_PROCESSED_DIR / "val.parquet",
    batch_size=BATCH_SIZE_KAGGLE,
    num_epochs=NUM_EPOCHS_KAGGLE,
    checkpoint_name="kaggle_run",
    device=device,
)
"""),
        md("""## Evaluation -- Test-7 graded path
Runs `scripts/run_evaluation.py` exactly as embedded above (the identical
file that `python scripts/run_evaluation.py` runs on the CLI, not a
reimplementation): loads the best-validation checkpoint and the English/Odia
tokenizers, runs plain greedy decoding (`src/inference/greedy_decode.py` --
not beam search) over the full 2,000-pair test split, computes corpus BLEU
and chrF++, and selects exactly 5 sample translations, one of them
deterministically the >=90th-percentile-length source sentence."""),
        code("""
!cd /kaggle/working && python scripts/run_evaluation.py
"""),
        md("""## Done
Checkpoints are written under `/kaggle/working/checkpoints/` and the
evaluation results (BLEU, chrF++, and the 5 samples) under
`/kaggle/working/reports/eval_results.json`; both appear in this kernel's
Output tab for download once the run finishes."""),
    ]

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(make_notebook(cells), indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH} ({len(cells)} cells, {len(texts)} source files packaged)")


if __name__ == "__main__":
    main()
