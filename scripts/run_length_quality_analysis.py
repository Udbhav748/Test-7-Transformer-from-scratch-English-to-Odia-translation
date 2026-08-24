import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import sacrebleu
import torch

from configs.base import CHECKPOINT_DIR, DATA_PROCESSED_DIR, REPORTS_DIR
from src.evaluation.bleu import _postprocess
from src.inference.greedy_decode import greedy_decode
from src.tokenization.tokenizer_utils import encode, load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model

WORD_LEN_BUCKETS = [(3, 5), (6, 8), (9, 11), (12, 15), (16, 20), (21, None)]
SUBWORD_LEN_BUCKETS = [(0, 15), (16, 25), (26, 35), (36, 50), (51, 64)]


def bucket_label(lo, hi):
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def has_repeated_bigram(ids, min_repeats=3):
    if len(ids) < 2:
        return False
    counts = {}
    for i in range(len(ids) - 1):
        bigram = (ids[i], ids[i + 1])
        counts[bigram] = counts.get(bigram, 0) + 1
        if counts[bigram] >= min_repeats:
            return True
    return False


def summarize_buckets(buckets, lengths, bleu_scores, repetitive_flags):
    lengths = np.asarray(lengths)
    bleu_scores = np.asarray(bleu_scores)
    repetitive_flags = np.asarray(repetitive_flags)

    rows = []
    for lo, hi in buckets:
        if hi is None:
            mask = lengths >= lo
        else:
            mask = (lengths >= lo) & (lengths <= hi)
        count = int(mask.sum())
        if count == 0:
            mean_bleu = 0.0
            rep_rate = 0.0
        else:
            mean_bleu = round(float(bleu_scores[mask].mean()), 3)
            rep_rate = round(float(repetitive_flags[mask].mean()) * 100, 2)
        rows.append(
            {
                "bucket": bucket_label(lo, hi),
                "count": count,
                "mean_bleu": mean_bleu,
                "repetition_rate_pct": rep_rate,
            }
        )
    return rows


def pearson_r(a, b) -> float:
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def main():
    model = build_model()
    load_checkpoint(CHECKPOINT_DIR / "kaggle_run_best.pt", model)
    model.eval()

    en_tok = load_tokenizer(EN_TOKENIZER_PATH)
    or_tok = load_tokenizer(OR_TOKENIZER_PATH)

    test_df = pd.read_parquet(DATA_PROCESSED_DIR / "test.parquet")

    source_word_lens = []
    source_subword_lens = []
    sentence_bleus = []
    is_repetitive_flags = []

    start = time.time()
    for src_text, tgt_text in zip(test_df["src"], test_df["tgt"]):
        src_token_ids = encode(en_tok, src_text)
        src_ids = torch.tensor([src_token_ids])
        hyp_ids = greedy_decode(model, src_ids)

        hyp_text = _postprocess(or_tok, hyp_ids)
        ref_text = _postprocess(or_tok, encode(or_tok, tgt_text))

        bleu = sacrebleu.sentence_bleu(hyp_text, [ref_text], smooth_method="exp")

        hyp_subword_ids = encode(or_tok, hyp_text)

        source_word_lens.append(len(src_text.split()))
        source_subword_lens.append(len(src_token_ids))
        sentence_bleus.append(round(bleu.score, 3))
        is_repetitive_flags.append(has_repeated_bigram(hyp_subword_ids))
    elapsed = time.time() - start
    print(f"decoded {len(test_df)} test examples in {elapsed:.1f}s")

    buckets_by_word_len = summarize_buckets(
        WORD_LEN_BUCKETS, source_word_lens, sentence_bleus, is_repetitive_flags
    )
    buckets_by_subword_len = summarize_buckets(
        SUBWORD_LEN_BUCKETS, source_subword_lens, sentence_bleus, is_repetitive_flags
    )

    out = {
        "num_examples": len(test_df),
        "overall_mean_sentence_bleu": round(float(np.mean(sentence_bleus)), 3),
        "overall_repetition_rate_pct": round(float(np.mean(is_repetitive_flags)) * 100, 2),
        "pearson_r_wordlen_bleu": pearson_r(source_word_lens, sentence_bleus),
        "pearson_r_subwordlen_bleu": pearson_r(source_subword_lens, sentence_bleus),
        "buckets_by_word_len": buckets_by_word_len,
        "buckets_by_subword_len": buckets_by_subword_len,
        "scatter_source_word_len": source_word_lens,
        "scatter_sentence_bleu": sentence_bleus,
        "scatter_is_repetitive": is_repetitive_flags,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "length_quality_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    print(f"overall_mean_sentence_bleu: {out['overall_mean_sentence_bleu']}")
    print(f"overall_repetition_rate_pct: {out['overall_repetition_rate_pct']}")
    print(f"pearson_r_wordlen_bleu: {out['pearson_r_wordlen_bleu']}")
    print(f"pearson_r_subwordlen_bleu: {out['pearson_r_subwordlen_bleu']}")
    print("buckets_by_word_len:")
    for row in buckets_by_word_len:
        print(f"  {row}")
    print("buckets_by_subword_len:")
    for row in buckets_by_subword_len:
        print(f"  {row}")


if __name__ == "__main__":
    main()
