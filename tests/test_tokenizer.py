import pandas as pd
import pytest

from configs.base import DATA_PROCESSED_DIR, EOS_ID, PAD_ID, SOS_ID, UNK_ID
from src.data.clean import nfc_normalize
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH
from src.tokenization.tokenizer_utils import decode, encode, load_tokenizer

TRAIN_PATH = DATA_PROCESSED_DIR / "train.parquet"


@pytest.fixture(scope="module")
def en_tok():
    return load_tokenizer(EN_TOKENIZER_PATH)


@pytest.fixture(scope="module")
def or_tok():
    return load_tokenizer(OR_TOKENIZER_PATH)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_parquet(TRAIN_PATH)


def test_special_token_ids_en(en_tok):
    assert en_tok.token_to_id("<PAD>") == PAD_ID
    assert en_tok.token_to_id("<SOS>") == SOS_ID
    assert en_tok.token_to_id("<EOS>") == EOS_ID
    assert en_tok.token_to_id("<UNK>") == UNK_ID


def test_special_token_ids_or(or_tok):
    assert or_tok.token_to_id("<PAD>") == PAD_ID
    assert or_tok.token_to_id("<SOS>") == SOS_ID
    assert or_tok.token_to_id("<EOS>") == EOS_ID
    assert or_tok.token_to_id("<UNK>") == UNK_ID


def test_roundtrip_english(en_tok, train_df):
    for text in train_df["src"].head(10):
        normalized = nfc_normalize(text)
        ids = encode(en_tok, normalized)
        assert decode(en_tok, ids) == normalized


def test_roundtrip_odia(or_tok, train_df):
    for text in train_df["tgt"].head(10):
        normalized = nfc_normalize(text)
        ids = encode(or_tok, normalized)
        assert decode(or_tok, ids) == normalized


def test_odia_encoding_wraps_with_sos_eos(or_tok, train_df):
    text = nfc_normalize(train_df["tgt"].iloc[0])
    ids = encode(or_tok, text)
    assert ids[0] == SOS_ID
    assert ids[-1] == EOS_ID
