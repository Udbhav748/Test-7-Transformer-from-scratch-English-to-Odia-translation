import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from configs.base import DATA_PROCESSED_DIR, REPORTS_DIR
from src.tokenization.tokenizer_utils import encode, load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH

EN_PUNCT_RE = re.compile(r"[^a-z0-9']+")
OR_PUNCT_RE = re.compile(r"[।,.!?\"'()\[\]\-—–:;]+")


def word_count(text: str) -> int:
    return len(text.split())


def top_words(texts, is_english: bool, n: int = 20, stopwords: set = frozenset()):
    counts = Counter()
    for text in texts:
        if is_english:
            cleaned = EN_PUNCT_RE.sub(" ", text.lower())
        else:
            cleaned = OR_PUNCT_RE.sub(" ", text)
        for token in cleaned.split():
            if token and token not in stopwords:
                counts[token] += 1
    return counts.most_common(n)


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
    en_subword_counts = full["src"].apply(lambda t: len(encode(en_tok, t))).tolist()
    or_subword_counts = full["tgt"].apply(lambda t: len(encode(or_tok, t))).tolist()
    subword_ratio = [
        round(o / e, 4) if e else 0.0 for e, o in zip(en_subword_counts, or_subword_counts)
    ]

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
        "top_words_en": top_words(full["src"], is_english=True, stopwords=EN_STOPWORDS),
        "top_words_or": top_words(full["tgt"], is_english=False),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "eda_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")

    import statistics
    print("English words: mean", round(statistics.mean(en_word_counts), 2), "median", statistics.median(en_word_counts))
    print("Odia words: mean", round(statistics.mean(or_word_counts), 2), "median", statistics.median(or_word_counts))
    print("English subwords: mean", round(statistics.mean(en_subword_counts), 2))
    print("Odia subwords: mean", round(statistics.mean(or_subword_counts), 2))
    print("subword ratio: mean", round(statistics.mean(subword_ratio), 3), "median", statistics.median(subword_ratio))
    print("top EN:", result["top_words_en"][:10])
    print("top OR:", result["top_words_or"][:10])


if __name__ == "__main__":
    main()
