import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.inference.repetition import banned_ngram_tokens
from src.training.train import build_model


def test_banned_ngram_tokens_blocks_the_completing_token():
    generated = [1, 5, 6, 7, 5, 6]
    banned = banned_ngram_tokens(generated, ngram_size=3)
    assert 7 in banned


def test_banned_ngram_tokens_empty_when_no_repeat_yet():
    assert banned_ngram_tokens([1, 5, 6], ngram_size=3) == set()


def test_banned_ngram_tokens_short_sequence_returns_empty():
    assert banned_ngram_tokens([1], ngram_size=3) == set()


def test_greedy_decode_never_repeats_ngram_on_random_model():
    from src.inference.greedy_decode import greedy_decode

    model = build_model()
    model.eval()
    src_ids = torch.tensor([[1, 40, 80, 120, 2]])

    ids = greedy_decode(model, src_ids, max_len=40, no_repeat_ngram_size=3)

    seen = set()
    for i in range(len(ids) - 2):
        gram = tuple(ids[i:i + 3])
        assert gram not in seen, f"repeated 3-gram {gram} in {ids}"
        seen.add(gram)
