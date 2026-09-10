import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from tokenizers import Tokenizer

from configs.base import DATA_PROCESSED_DIR, LONG_SENTENCE_PERCENTILE, NUM_SAMPLE_TRANSLATIONS, REPORTS_DIR
from src.evaluation.bleu import corpus_bleu, corpus_chrf
from src.inference.beam_search import beam_search_decode
from src.inference.greedy_decode import greedy_decode
from src.model.scaled_transformer import EnhancedScaledTransformer
from src.tokenization.tokenizer_utils import decode, encode

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "scaled_model_best.pt"
EN_TOKENIZER_PATH = REPO_ROOT / "tokenizers" / "scaled_en_bpe.json"
OR_TOKENIZER_PATH = REPO_ROOT / "tokenizers" / "scaled_or_bpe.json"
MAX_LEN = 96


def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = ckpt["config"]
    model = EnhancedScaledTransformer(
        en_vocab_size=cfg["src_vocab_size"],
        or_vocab_size=cfg["tgt_vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_encoder_layers=cfg["n_encoder_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        dropout=cfg["dropout"],
        tie_weights=cfg["tie_weights"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    en_tok = Tokenizer.from_file(str(EN_TOKENIZER_PATH))
    or_tok = Tokenizer.from_file(str(OR_TOKENIZER_PATH))

    test_df = pd.read_parquet(DATA_PROCESSED_DIR / "test.parquet")

    # Full-corpus BLEU/chrF++ uses greedy decode -- same methodology as the
    # baseline's official eval (scripts/run_evaluation.py), and ~10x faster
    # than beam search on CPU since beam search here isn't batched. Beam
    # search is still used below for the 5 qualitative samples.
    start = time.time()
    hyp_ids, ref_ids = [], []
    for src_text, tgt_text in zip(test_df["src"], test_df["tgt"]):
        src_ids = torch.tensor([encode(en_tok, src_text)])
        hyp_ids.append(greedy_decode(model, src_ids, max_len=MAX_LEN))
        ref_ids.append(encode(or_tok, tgt_text))
    elapsed = time.time() - start

    bleu = corpus_bleu(hyp_ids, ref_ids, or_tok)
    chrf = corpus_chrf(hyp_ids, ref_ids, or_tok)

    print(f"decoded {len(test_df)} test examples in {elapsed:.1f}s (greedy decode)")
    print(f"BLEU: {bleu.score:.2f}  ({bleu})")
    print(f"chrF++: {chrf.score:.2f}  ({chrf})")

    # 5 sample translations (same selection logic as the baseline script)
    src_word_len = test_df["src"].str.split().apply(len)
    long_threshold = src_word_len.quantile(LONG_SENTENCE_PERCENTILE)
    long_candidates = test_df[src_word_len >= long_threshold]
    long_row = long_candidates.sample(1, random_state=42).iloc[0]
    rest = test_df.drop(long_candidates.index)
    representative = rest.sample(NUM_SAMPLE_TRANSLATIONS - 1, random_state=42)
    rows = list(representative.itertuples(index=False)) + [long_row]

    samples = []
    for row in rows:
        src_ids = torch.tensor([encode(en_tok, row.src)])
        hyp = beam_search_decode(model, src_ids, max_len=MAX_LEN)
        samples.append({
            "source": row.src,
            "reference": row.tgt,
            "hypothesis": decode(or_tok, hyp),
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "bleu_score": bleu.score,
        "bleu_signature": str(bleu),
        "chrf_score": chrf.score,
        "chrf_signature": str(chrf),
        "num_test_examples": len(test_df),
        "decode_seconds": elapsed,
        "decoding": "greedy (corpus metrics); beam_search_k4 (qualitative samples)",
        "samples": samples,
    }
    out_path = REPORTS_DIR / "scaled_eval_results.json"

    # preserve any existing fields (e.g. earlier greedy/beam qualitative samples) not overwritten above
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        for k, v in existing.items():
            out.setdefault(k, v)
        out["bleu_score"] = bleu.score
        out["bleu_signature"] = str(bleu)
        out["chrf_score"] = chrf.score
        out["chrf_signature"] = str(chrf)
        out["num_test_examples"] = len(test_df)
        out["decode_seconds"] = elapsed
        out["samples"] = samples

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    for s in samples:
        print("SRC:", s["source"])
        print("REF:", s["reference"])
        print("HYP:", s["hypothesis"])
        print("---")


if __name__ == "__main__":
    main()
