import torch

from configs.base import EOS_ID, GREEDY_MAX_DECODE_LEN, NO_REPEAT_NGRAM_SIZE, SOS_ID
from src.inference.repetition import banned_ngram_tokens


@torch.no_grad()
def greedy_decode(
    model,
    src_ids: torch.Tensor,
    max_len: int = GREEDY_MAX_DECODE_LEN,
    no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
) -> list[int]:
    device = src_ids.device
    was_training = model.training
    model.eval()

    tgt_ids = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits = model(src_ids, tgt_ids)
        next_logits = logits[:, -1, :].clone()

        banned = banned_ngram_tokens(tgt_ids[0].tolist(), no_repeat_ngram_size)
        for token_id in banned:
            next_logits[:, token_id] = float("-inf")

        next_id = next_logits.argmax(dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break

    model.train(was_training)
    return tgt_ids[0].tolist()
