import sys

import pandas as pd
from datasets import load_dataset

from configs.base import (
    CANDIDATE_POOL_SIZE,
    DATA_RAW_DIR,
    HF_DATASET_CONFIG,
    HF_DATASET_ID,
)
from src.data.clean import clean_text, passes_word_count

# parquet over tsv: candidate/processed text can contain arbitrary punctuation
# (including literal tab or newline bytes from scraped web text), which a
# tab-separated format would need to escape; parquet round-trips it exactly
CANDIDATES_PATH = DATA_RAW_DIR / "candidates.parquet"


def collect_candidates(pool_size: int = CANDIDATE_POOL_SIZE) -> list[dict]:
    stream = load_dataset(HF_DATASET_ID, HF_DATASET_CONFIG, split="train", streaming=True)

    seen_src = set()
    candidates = []
    exhausted = True
    for example in stream:
        src = clean_text(example["src"])
        tgt = clean_text(example["tgt"])

        if not src or not tgt:
            continue
        if src in seen_src:
            continue
        if not (passes_word_count(src) and passes_word_count(tgt)):
            continue

        seen_src.add(src)
        candidates.append({"src": src, "tgt": tgt})

        if len(candidates) >= pool_size:
            exhausted = False
            break

    if exhausted:
        print(
            f"WARNING: stream exhausted before reaching CANDIDATE_POOL_SIZE="
            f"{pool_size}; collected only {len(candidates)} candidates.",
            file=sys.stderr,
        )

    return candidates


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CANDIDATES_PATH.exists():
        existing = pd.read_parquet(CANDIDATES_PATH)
        print(f"{CANDIDATES_PATH} already exists with {len(existing)} candidates; skipping download.")
        return

    candidates = collect_candidates()

    df = pd.DataFrame(candidates, columns=["src", "tgt"])
    df.to_parquet(CANDIDATES_PATH, index=False)
    print(f"saved {len(df)} candidates to {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
