import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from configs.base import CANDIDATE_POOL_SIZE, DATA_PROCESSED_DIR, REPORTS_DIR, UNK_ID
from src.tokenization.tokenizer_utils import encode, load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH

EN_PUNCT_RE = re.compile(r"[^a-z0-9']+")
OR_PUNCT_RE = re.compile(r"[।,.!?\"'()\[\]\-—–:;]+")

EN_ENDING_CHARS = {".", "?", "!", ","}
OR_ENDING_CHARS = {"।", "?", "!"}


def word_count(text: str) -> int:
    return len(text.split())


def avg_word_len(text: str) -> float:
    wc = word_count(text)
    if wc == 0:
        return 0.0
    chars = len(text.replace(" ", ""))
    return round(chars / wc, 3)


def clean_words(text: str, is_english: bool):
    if is_english:
        cleaned = EN_PUNCT_RE.sub(" ", text.lower())
    else:
        cleaned = OR_PUNCT_RE.sub(" ", text)
    return cleaned.split()


def build_word_counter(texts, is_english: bool, stopwords: set = frozenset()) -> Counter:
    counts = Counter()
    for text in texts:
        for token in clean_words(text, is_english):
            if token and token not in stopwords:
                counts[token] += 1
    return counts


def top_words_full(counter: Counter, total: int, n: int):
    return [
        [word, count, round(100 * count / total, 2) if total else 0.0]
        for word, count in counter.most_common(n)
    ]


def ending_punct_counts(texts, keep_chars: set) -> dict:
    counts = Counter()
    for text in texts:
        stripped = text.rstrip()
        last = stripped[-1] if stripped else ""
        counts[last if last in keep_chars else "other"] += 1
    result = {ch: counts.get(ch, 0) for ch in keep_chars}
    result["other"] = counts.get("other", 0)
    return result


def pct_with_digits(texts) -> float:
    n = len(texts)
    if n == 0:
        return 0.0
    hits = sum(1 for t in texts if re.search(r"[0-9]", t))
    return round(100 * hits / n, 2)


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(a)), 3),
        "std": round(float(np.std(a)), 3),
        "min": round(float(np.min(a)), 3),
        "p25": round(float(np.percentile(a, 25)), 3),
        "median": round(float(np.median(a)), 3),
        "p75": round(float(np.percentile(a, 75)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(np.max(a)), 3),
    }


def pearson_r(a, b) -> float:
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


EN_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "is", "was", "on", "for", "with",
    "that", "it", "as", "at", "by", "his", "her", "he", "she", "this", "are",
    "be", "has", "have", "had", "not", "but", "from", "will", "said",
}


def main():
    train = pd.read_parquet(DATA_PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(DATA_PROCESSED_DIR / "test.parquet")
    full = pd.concat([train, val, test], ignore_index=True)

    en_tok = load_tokenizer(EN_TOKENIZER_PATH)
    or_tok = load_tokenizer(OR_TOKENIZER_PATH)

    en_word_counts = full["src"].apply(word_count).tolist()
    or_word_counts = full["tgt"].apply(word_count).tolist()

    en_encoded = full["src"].apply(lambda t: encode(en_tok, t)).tolist()
    or_encoded = full["tgt"].apply(lambda t: encode(or_tok, t)).tolist()
    en_subword_counts = [len(ids) for ids in en_encoded]
    or_subword_counts = [len(ids) for ids in or_encoded]
    en_unk_counts = [ids.count(UNK_ID) for ids in en_encoded]
    or_unk_counts = [ids.count(UNK_ID) for ids in or_encoded]

    subword_ratio = [
        round(o / e, 4) if e else 0.0 for e, o in zip(en_subword_counts, or_subword_counts)
    ]

    en_char_counts = full["src"].apply(len).tolist()
    or_char_counts = full["tgt"].apply(len).tolist()
    en_avg_word_len = full["src"].apply(avg_word_len).tolist()
    or_avg_word_len = full["tgt"].apply(avg_word_len).tolist()

    en_total_subwords = sum(en_subword_counts)
    or_total_subwords = sum(or_subword_counts)
    en_unk_rate_pct = round(100 * sum(en_unk_counts) / en_total_subwords, 4) if en_total_subwords else 0.0
    or_unk_rate_pct = round(100 * sum(or_unk_counts) / or_total_subwords, 4) if or_total_subwords else 0.0

    en_counter = build_word_counter(full["src"], is_english=True, stopwords=EN_STOPWORDS)
    or_counter = build_word_counter(full["tgt"], is_english=False)
    en_total_words = sum(en_counter.values())
    or_total_words = sum(or_counter.values())

    vocab_stats = {
        "en_total_words": en_total_words,
        "en_unique_words": len(en_counter),
        "en_ttr": round(len(en_counter) / en_total_words, 4) if en_total_words else 0.0,
        "or_total_words": or_total_words,
        "or_unique_words": len(or_counter),
        "or_ttr": round(len(or_counter) / or_total_words, 4) if or_total_words else 0.0,
    }

    result = {
        "num_examples": len(full),
        "split_composition": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "en_word_counts": en_word_counts,
        "or_word_counts": or_word_counts,
        "en_subword_counts": en_subword_counts,
        "or_subword_counts": or_subword_counts,
        "subword_ratio": subword_ratio,
        "top_words_en": en_counter.most_common(20),
        "top_words_or": or_counter.most_common(20),
        "en_char_counts": en_char_counts,
        "or_char_counts": or_char_counts,
        "en_avg_word_len": en_avg_word_len,
        "or_avg_word_len": or_avg_word_len,
        "en_unk_counts": en_unk_counts,
        "or_unk_counts": or_unk_counts,
        "en_unk_rate_pct": en_unk_rate_pct,
        "or_unk_rate_pct": or_unk_rate_pct,
        "en_ending_punct": ending_punct_counts(full["src"], EN_ENDING_CHARS),
        "or_ending_punct": ending_punct_counts(full["tgt"], OR_ENDING_CHARS),
        "en_pct_with_digits": pct_with_digits(full["src"]),
        "or_pct_with_digits": pct_with_digits(full["tgt"]),
        "vocab_stats": vocab_stats,
        "length_correlation": {
            "pearson_r_words": pearson_r(en_word_counts, or_word_counts),
            "pearson_r_subwords": pearson_r(en_subword_counts, or_subword_counts),
        },
        "cleaning_funnel": {
            "candidate_pool_after_cleaning": CANDIDATE_POOL_SIZE,
            "survived_max_len_filter": 44_673,
            "final_sampled": len(full),
        },
        "descriptive_stats": {
            "en_words": describe(en_word_counts),
            "or_words": describe(or_word_counts),
            "en_subwords": describe(en_subword_counts),
            "or_subwords": describe(or_subword_counts),
            "subword_ratio": describe(subword_ratio),
            "en_chars": describe(en_char_counts),
            "or_chars": describe(or_char_counts),
        },
        "top_words_en_full": top_words_full(en_counter, en_total_words, 30),
        "top_words_or_full": top_words_full(or_counter, or_total_words, 30),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "eda_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")

    print("English words: mean", round(statistics.mean(en_word_counts), 2), "median", statistics.median(en_word_counts))
    print("Odia words: mean", round(statistics.mean(or_word_counts), 2), "median", statistics.median(or_word_counts))
    print("English subwords: mean", round(statistics.mean(en_subword_counts), 2))
    print("Odia subwords: mean", round(statistics.mean(or_subword_counts), 2))
    print("subword ratio: mean", round(statistics.mean(subword_ratio), 3), "median", statistics.median(subword_ratio))
    print("top EN:", result["top_words_en"][:10])
    print("top OR:", result["top_words_or"][:10])
    print("EN unk rate pct:", en_unk_rate_pct, "OR unk rate pct:", or_unk_rate_pct)
    print("vocab stats:", vocab_stats)
    print("length correlation:", result["length_correlation"])


if __name__ == "__main__":
    main()
