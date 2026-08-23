import torch
import torch.nn as nn

from configs.base import PAD_ID
from src.model.attention import MultiHeadAttention
from src.model.masks import (
    make_causal_mask,
    make_cross_attn_mask,
    make_decoder_self_attn_mask,
)
from src.model.transformer import Seq2SeqTransformer

SRC_VOCAB_SIZE = 40
TGT_VOCAB_SIZE = 40


def test_causal_leak_invariance_full_model():
    torch.manual_seed(1)
    model = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    model.eval()

    T = 10
    t = 5
    src_ids = torch.randint(4, SRC_VOCAB_SIZE, (1, 7))

    tgt_a = torch.randint(4, TGT_VOCAB_SIZE, (1, T))
    tgt_b = tgt_a.clone()
    tgt_b[:, t + 1 :] = torch.randint(4, TGT_VOCAB_SIZE, (1, T - t - 1))
    assert not torch.equal(tgt_a[:, t + 1 :], tgt_b[:, t + 1 :]) or T - t - 1 == 0

    with torch.no_grad():
        logits_a = model(src_ids, tgt_a)
        logits_b = model(src_ids, tgt_b)

    assert torch.allclose(logits_a[:, : t + 1, :], logits_b[:, : t + 1, :], atol=1e-5)


def test_causal_mask_negative_control_has_teeth():
    """Constructive proof that the invariance test above is not vacuous.

    Runs the identical divergent-suffix invariance check through a single
    self-attention layer twice: once with the real causal mask (must be
    invariant, matching production wiring) and once with a deliberately
    broken "mask" that permits full bidirectional attention (must NOT be
    invariant). If a real causal-mask bug ever reintroduced this exact
    kind of leak, the production test would fail the same way this
    broken configuration does here.
    """
    torch.manual_seed(2)
    d_model, n_heads = 16, 2
    vocab_size = 20
    T, t = 10, 5

    embedding = nn.Embedding(vocab_size, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    mha.eval()

    tgt_a = torch.randint(4, vocab_size, (1, T))
    tgt_b = tgt_a.clone()
    tgt_b[:, t + 1 :] = torch.randint(4, vocab_size, (1, T - t - 1))

    def run(tgt_ids, mask):
        x = embedding(tgt_ids)
        with torch.no_grad():
            return mha(x, x, x, mask)

    correct_causal_mask = make_causal_mask(T).unsqueeze(0).unsqueeze(0)
    out_a_correct = run(tgt_a, correct_causal_mask)
    out_b_correct = run(tgt_b, correct_causal_mask)
    assert torch.allclose(out_a_correct[:, : t + 1, :], out_b_correct[:, : t + 1, :], atol=1e-5)

    broken_mask = torch.ones(1, 1, T, T, dtype=torch.bool)
    out_a_broken = run(tgt_a, broken_mask)
    out_b_broken = run(tgt_b, broken_mask)
    assert not torch.allclose(out_a_broken[:, : t + 1, :], out_b_broken[:, : t + 1, :], atol=1e-5)


def test_decoder_self_attn_mask_explicit_coverage():
    pad_id = PAD_ID
    tgt_ids = torch.tensor([[5, 6, 7, 8, pad_id, pad_id]])
    T = tgt_ids.size(1)

    mask = make_decoder_self_attn_mask(tgt_ids, pad_id)
    assert mask.shape == (1, 1, T, T)
    mask = mask[0, 0]

    for i in range(T):
        for j in range(T):
            if j > i:
                assert not mask[i, j], f"future key {j} not blocked at query {i}"
            elif tgt_ids[0, j].item() == pad_id:
                assert not mask[i, j], f"pad key {j} not blocked at query {i}"
            else:
                assert mask[i, j], f"valid causal key {j} wrongly blocked at query {i}"


def test_cross_attn_mask_explicit_coverage():
    pad_id = PAD_ID
    src_ids = torch.tensor([[5, 6, 7, pad_id, pad_id]])
    S = src_ids.size(1)

    mask = make_cross_attn_mask(src_ids, pad_id)
    assert mask.shape == (1, 1, 1, S)
    mask = mask[0, 0, 0]

    for j in range(S):
        is_pad = src_ids[0, j].item() == pad_id
        if is_pad:
            assert not mask[j], f"pad source position {j} not blocked"
        else:
            assert mask[j], f"non-pad source position {j} wrongly blocked"

    # cross-attn mask has no query dimension of its own ([B,1,1,S]); it
    # broadcasts identically across every decoder query position, so the
    # per-position checks above already cover "for every query".
