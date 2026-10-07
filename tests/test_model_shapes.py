import torch

from configs.base import PAD_ID
from src.model.transformer import Seq2SeqTransformer

SRC_VOCAB_SIZE = 50
TGT_VOCAB_SIZE = 50


def _random_ids(batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    return torch.randint(4, vocab_size, (batch_size, seq_len))


def _assert_valid_output(logits: torch.Tensor, batch_size: int, tgt_len: int, vocab_size: int):
    assert logits.shape == (batch_size, tgt_len, vocab_size)
    assert logits.dtype in (torch.float32, torch.float64)
    assert torch.isfinite(logits).all()


def test_shapes_various_batch_and_seq_lengths():
    model = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    model.eval()

    cases = [
        (1, 5, 5),
        (2, 8, 6),
        (4, 12, 10),
        (3, 6, 12),
    ]
    with torch.no_grad():
        for batch_size, src_len, tgt_len in cases:
            src_ids = _random_ids(batch_size, src_len, SRC_VOCAB_SIZE)
            tgt_ids = _random_ids(batch_size, tgt_len, TGT_VOCAB_SIZE)
            logits = model(src_ids, tgt_ids)
            _assert_valid_output(logits, batch_size, tgt_len, TGT_VOCAB_SIZE)


def test_shapes_with_right_padding():
    model = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    model.eval()

    batch_size, src_len, tgt_len = 3, 10, 8
    src_ids = _random_ids(batch_size, src_len, SRC_VOCAB_SIZE)
    tgt_ids = _random_ids(batch_size, tgt_len, TGT_VOCAB_SIZE)

    src_ids[0, 6:] = PAD_ID
    src_ids[1, 8:] = PAD_ID
    tgt_ids[0, 5:] = PAD_ID
    tgt_ids[2, 6:] = PAD_ID

    with torch.no_grad():
        logits = model(src_ids, tgt_ids)
    _assert_valid_output(logits, batch_size, tgt_len, TGT_VOCAB_SIZE)


def test_forward_returns_raw_logits_not_probabilities():
    model = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    model.eval()

    src_ids = _random_ids(2, 7, SRC_VOCAB_SIZE)
    tgt_ids = _random_ids(2, 5, TGT_VOCAB_SIZE)

    with torch.no_grad():
        logits = model(src_ids, tgt_ids)

    assert logits.dtype.is_floating_point
    row_sums = logits.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert (logits < 0).any()


def test_param_count_sanity_bound():
    # Synthetic 8000/8000 vocab, used only as an upper-bound sanity check that
    # works without the trained tokenizers present (e.g. a fresh clone or CI).
    # This is NOT the real model's parameter count -- see
    # test_param_count_matches_real_tokenizers below, or reports/write_up.md,
    # for that (the real Odia tokenizer undershoots this configured size).
    model = Seq2SeqTransformer(src_vocab_size=8000, tgt_vocab_size=8000)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20_000_000


def test_param_count_matches_real_tokenizers():
    # Reports the model's real size against the tokenizers actually on disk.
    # Skipped where they aren't present (e.g. a fresh clone before running
    # the pipeline) rather than silently reporting a synthetic number.
    from src.tokenization.tokenizer_utils import load_tokenizer
    from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH

    if not (EN_TOKENIZER_PATH.exists() and OR_TOKENIZER_PATH.exists()):
        import pytest
        pytest.skip("trained tokenizers not present")

    en_vocab = load_tokenizer(EN_TOKENIZER_PATH).get_vocab_size()
    or_vocab = load_tokenizer(OR_TOKENIZER_PATH).get_vocab_size()
    model = Seq2SeqTransformer(src_vocab_size=en_vocab, tgt_vocab_size=or_vocab)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"real total params (en_vocab={en_vocab}, or_vocab={or_vocab}): {total_params}")
    assert total_params < 20_000_000
