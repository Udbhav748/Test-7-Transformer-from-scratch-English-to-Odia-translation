import random
import sys

import pandas as pd

from configs.base import (
    DATA_PROCESSED_DIR,
    MAX_LEN,
    RANDOM_SEED,
    TEST_SIZE,
    TOTAL_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from src.data.download import CANDIDATES_PATH
from src.tokenization.train_tokenizer import (
    EN_TOKENIZER_PATH,
    OR_TOKENIZER_PATH,
    select_tokenizer_pool,
)
from src.tokenization.tokenizer_utils import encode, load_tokenizer

ESTIMATED_RETENTION = 0.780

TRAIN_PATH = DATA_PROCESSED_DIR / "train.parquet"
VAL_PATH = DATA_PROCESSED_DIR / "val.parquet"
TEST_PATH = DATA_PROCESSED_DIR / "test.parquet"


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(CANDIDATES_PATH)

    # Tokenizer training text must never overlap val/test, so drop the reserved
    # pool and every candidate sharing any sentence with it before selection.
    tokenizer_pool = select_tokenizer_pool(candidates)
    tokenizer_texts = set(tokenizer_pool["src"]) | set(tokenizer_pool["tgt"])
    shares_tokenizer_text = candidates["src"].isin(tokenizer_texts) | candidates["tgt"].isin(
        tokenizer_texts
    )
    excluded_count = int(shares_tokenizer_text.sum())
    candidates = candidates[~shares_tokenizer_text].reset_index(drop=True)
    pool_size = len(candidates)

    en_tok = load_tokenizer(EN_TOKENIZER_PATH)
    or_tok = load_tokenizer(OR_TOKENIZER_PATH)

    kept_rows = []
    for src, tgt in zip(candidates["src"], candidates["tgt"]):
        en_ids = encode(en_tok, src)
        or_ids = encode(or_tok, tgt)
        if len(en_ids) <= MAX_LEN and len(or_ids) <= MAX_LEN:
            kept_rows.append({"src": src, "tgt": tgt})

    survivors = pd.DataFrame(kept_rows, columns=["src", "tgt"])
    survivors = survivors.drop_duplicates(subset="src").reset_index(drop=True)
    survivor_count = len(survivors)
    retention_rate = survivor_count / pool_size if pool_size else 0.0

    if survivor_count < TOTAL_SIZE:
        print(
            f"WARNING: only {survivor_count} pairs survived MAX_LEN={MAX_LEN} filtering, "
            f"below TOTAL_SIZE={TOTAL_SIZE}. Observed retention rate {retention_rate:.1%} vs "
            f"the {ESTIMATED_RETENTION:.1%} estimate. Increase CANDIDATE_POOL_SIZE and re-run "
            "download/train_tokenizer/split.",
            file=sys.stderr,
        )
        final = survivors
    else:
        rng = random.Random(RANDOM_SEED)
        indices = list(range(survivor_count))
        rng.shuffle(indices)
        final = survivors.iloc[indices[:TOTAL_SIZE]].reset_index(drop=True)

    final_count = len(final)
    n_train = min(TRAIN_SIZE, final_count)
    n_val = min(VAL_SIZE, max(final_count - n_train, 0))
    n_test = max(final_count - n_train - n_val, 0)

    train_df = final.iloc[:n_train].reset_index(drop=True)
    val_df = final.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = final.iloc[n_train + n_val:n_train + n_val + n_test].reset_index(drop=True)

    train_src = set(train_df["src"])
    val_src = set(val_df["src"])
    test_src = set(test_df["src"])
    assert not (train_src & val_src)
    assert not (train_src & test_src)
    assert not (val_src & test_src)

    held_out_texts = val_src | test_src | set(val_df["tgt"]) | set(test_df["tgt"])
    assert not (held_out_texts & tokenizer_texts)

    train_df.to_parquet(TRAIN_PATH, index=False)
    val_df.to_parquet(VAL_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print(f"tokenizer pool size (excluded from splits): {len(tokenizer_pool)}")
    print(f"candidates removed (tokenizer pool rows + text overlap): {excluded_count}")
    print(f"candidate pool size: {pool_size}")
    print(f"post-filter survivor count (MAX_LEN={MAX_LEN}): {survivor_count}")
    print(
        f"observed retention rate: {retention_rate:.1%} "
        f"(throwaway-sample estimate was {ESTIMATED_RETENTION:.1%})"
    )
    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"saved to {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")


if __name__ == "__main__":
    main()
