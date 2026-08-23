import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from configs.base import CHECKPOINT_DIR, DATA_PROCESSED_DIR, REPORTS_DIR
from src.evaluation.bleu import corpus_bleu
from src.evaluation.sample_translations import select_samples, translate_samples
from src.inference.greedy_decode import greedy_decode
from src.tokenization.tokenizer_utils import encode, load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model
import pandas as pd


def main():
    model = build_model()
    load_checkpoint(CHECKPOINT_DIR / "kaggle_run_best.pt", model)
    model.eval()

    en_tok = load_tokenizer(EN_TOKENIZER_PATH)
    or_tok = load_tokenizer(OR_TOKENIZER_PATH)

    test_df = pd.read_parquet(DATA_PROCESSED_DIR / "test.parquet")

    start = time.time()
    hyp_ids, ref_ids = [], []
    for src_text, tgt_text in zip(test_df["src"], test_df["tgt"]):
        src_ids = torch.tensor([encode(en_tok, src_text)])
        hyp_ids.append(greedy_decode(model, src_ids))
        ref_ids.append(encode(or_tok, tgt_text))
    elapsed = time.time() - start

    bleu = corpus_bleu(hyp_ids, ref_ids, or_tok)
    print(f"decoded {len(test_df)} test examples in {elapsed:.1f}s")
    print(f"BLEU: {bleu.score:.2f}  ({bleu})")

    rows = select_samples(test_path=DATA_PROCESSED_DIR / "test.parquet", n=5)
    samples = translate_samples(model, rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "bleu_score": bleu.score,
        "bleu_signature": str(bleu),
        "num_test_examples": len(test_df),
        "decode_seconds": elapsed,
        "samples": samples,
    }
    out_path = REPORTS_DIR / "eval_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    for s in samples:
        print("SRC:", s["source"])
        print("REF:", s["reference"])
        print("HYP:", s["hypothesis"])
        print("---")


if __name__ == "__main__":
    main()
