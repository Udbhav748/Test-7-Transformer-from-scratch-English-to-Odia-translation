import torch

from configs.base import EOS_ID, GREEDY_MAX_DECODE_LEN, NO_REPEAT_NGRAM_SIZE, SOS_ID
from src.inference.repetition import banned_ngram_tokens


@torch.no_grad()
def translate_with_attention(
    model,
    src_ids: torch.Tensor,
    max_len: int = GREEDY_MAX_DECODE_LEN,
    no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
) -> dict:
    """Greedy-decodes src_ids exactly like greedy_decode, additionally
    recording the last decoder layer's cross-attention row for each
    generated token (i.e. how that token attended over the source).

    The <SOS> step has no generated token to attach a row to, so
    attention_matrix has one row per token generated after <SOS> --
    the same tokens present in generated_ids[1:].
    """
    device = src_ids.device
    was_training = model.training
    model.eval()

    tgt_ids = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    attention_rows = []
    for _ in range(max_len - 1):
        logits = model(src_ids, tgt_ids)
        next_logits = logits[:, -1, :].clone()

        banned = banned_ngram_tokens(tgt_ids[0].tolist(), no_repeat_ngram_size)
        for token_id in banned:
            next_logits[:, token_id] = float("-inf")

        next_id = next_logits.argmax(dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_id], dim=1)

        cross_attn = model.decoder.layers[-1].cross_attn.last_attn_weights
        # [B, n_heads, Q_len, K_len] -> average heads -> last query row
        row = cross_attn.mean(dim=1)[0, -1, :]
        attention_rows.append(row.tolist())

        if next_id.item() == EOS_ID:
            break

    model.train(was_training)
    return {
        "generated_ids": tgt_ids[0].tolist(),
        "attention_matrix": attention_rows,
    }
