import pandas as pd
import torch

from configs.base import DATA_PROCESSED_DIR, LONG_SENTENCE_PERCENTILE, NUM_SAMPLE_TRANSLATIONS
from src.inference.greedy_decode import greedy_decode
from src.tokenization.tokenizer_utils import decode, encode, load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH


def select_samples(test_path=DATA_PROCESSED_DIR / "test.parquet", n=NUM_SAMPLE_TRANSLATIONS, seed=42):
    df = pd.read_parquet(test_path).reset_index(drop=True)

    # long sentence is drawn from the filtered test split, never the raw
    # corpus -- a raw-corpus-derived long example could have been dropped
    # during MAX_LEN filtering and would not be a valid test example
    src_word_len = df["src"].str.split().apply(len)
    long_threshold = src_word_len.quantile(LONG_SENTENCE_PERCENTILE)
    long_candidates = df[src_word_len >= long_threshold]
    long_row = long_candidates.sample(1, random_state=seed).iloc[0]

    rest = df.drop(long_candidates.index)
    representative = rest.sample(n - 1, random_state=seed)

    return list(representative.itertuples(index=False)) + [long_row]


def translate_samples(model, rows, device="cpu"):
    en_tok = load_tokenizer(EN_TOKENIZER_PATH)
    or_tok = load_tokenizer(OR_TOKENIZER_PATH)

    results = []
    for row in rows:
        src_ids = torch.tensor([encode(en_tok, row.src)], device=device)
        hyp_ids = greedy_decode(model, src_ids)
        results.append({
            "source": row.src,
            "reference": row.tgt,
            "hypothesis": decode(or_tok, hyp_ids),
        })
    return results
