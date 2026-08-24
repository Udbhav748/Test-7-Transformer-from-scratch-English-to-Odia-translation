import torch
import torch.nn.functional as F

from configs.base import (
    BEAM_LENGTH_PENALTY,
    BEAM_WIDTH,
    EOS_ID,
    GREEDY_MAX_DECODE_LEN,
    NO_REPEAT_NGRAM_SIZE,
    SOS_ID,
)
from src.inference.repetition import banned_ngram_tokens


def _normalized_score(cum_logprob: float, length: int, length_penalty: float) -> float:
    return cum_logprob / (length ** length_penalty)


@torch.no_grad()
def beam_search_decode(
    model,
    src_ids: torch.Tensor,
    beam_width: int = BEAM_WIDTH,
    max_len: int = GREEDY_MAX_DECODE_LEN,
    length_penalty: float = BEAM_LENGTH_PENALTY,
    no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
) -> list[int]:
    device = src_ids.device
    was_training = model.training
    model.eval()

    beams = [([SOS_ID], 0.0)]
    completed = []

    for _ in range(max_len - 1):
        candidates = []
        for seq, cum_logprob in beams:
            if seq[-1] == EOS_ID:
                completed.append((seq, cum_logprob))
                continue

            tgt_ids = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(src_ids, tgt_ids)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

            banned = banned_ngram_tokens(seq, no_repeat_ngram_size)
            for token_id in banned:
                log_probs[token_id] = float("-inf")

            top_logprobs, top_ids = log_probs.topk(beam_width)

            for logprob, token_id in zip(top_logprobs.tolist(), top_ids.tolist()):
                candidates.append((seq + [token_id], cum_logprob + logprob))

        if not candidates:
            break

        candidates.sort(key=lambda c: _normalized_score(c[1], len(c[0]), length_penalty), reverse=True)
        beams = candidates[:beam_width]

        if all(seq[-1] == EOS_ID for seq, _ in beams):
            completed.extend(beams)
            beams = []
            break

    completed.extend(beams)
    model.train(was_training)

    best_seq, _ = max(
        completed, key=lambda c: _normalized_score(c[1], len(c[0]), length_penalty)
    )
    return best_seq
