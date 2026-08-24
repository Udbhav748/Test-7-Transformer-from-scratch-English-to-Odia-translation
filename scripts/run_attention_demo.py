import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from configs.base import CHECKPOINT_DIR, REPORTS_DIR
from src.inference.attention_extraction import translate_with_attention
from src.tokenization.tokenizer_utils import decode, encode, load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model

# The first 3 "samples" entries in reports/eval_results.json are the
# representative (non-long) picks from select_samples(); the 5th is the
# dedicated long-sentence example. Reused here for consistency across the
# dashboard rather than drawing fresh sentences.
EXAMPLE_SOURCES = [
    "Shutting down might cause them to lose unsaved work.",
    "Bihar Chief Minister and JD(U) chief Nitish Kumar.",
    "Students will be focussed.",
]


def main():
    model = build_model()
    load_checkpoint(CHECKPOINT_DIR / "kaggle_run_best.pt", model)
    model.eval()

    en_tok = load_tokenizer(EN_TOKENIZER_PATH)
    or_tok = load_tokenizer(OR_TOKENIZER_PATH)

    examples = []
    for src_text in EXAMPLE_SOURCES:
        src_ids_list = encode(en_tok, src_text)
        src_ids = torch.tensor([src_ids_list])
        source_tokens = en_tok.encode(src_text).tokens

        result = translate_with_attention(model, src_ids)
        generated_ids = result["generated_ids"]
        attention_matrix = result["attention_matrix"]

        # generated_ids[0] is <SOS>, which has no attention row (see
        # translate_with_attention); hypothesis_tokens therefore lines up
        # with generated_ids[1:], one token per attention_matrix row.
        hypothesis_tokens = [or_tok.id_to_token(i) for i in generated_ids[1:]]
        hypothesis_text = decode(or_tok, generated_ids)

        expected_shape = (len(hypothesis_tokens), len(source_tokens))
        actual_shape = (len(attention_matrix), len(attention_matrix[0]) if attention_matrix else 0)
        print(f"SRC: {src_text}")
        print(f"HYP: {hypothesis_text}")
        print(f"attention matrix shape: {actual_shape} (expected {expected_shape})")
        assert actual_shape == expected_shape, "attention matrix shape mismatch"

        row_sums = [sum(row) for row in attention_matrix]
        max_dev = max(abs(s - 1.0) for s in row_sums) if row_sums else 0.0
        print(f"max |row_sum - 1.0| = {max_dev:.6f}")
        assert max_dev < 1e-3, "attention rows do not sum to ~1.0"
        print("---")

        examples.append({
            "source_text": src_text,
            "hypothesis_text": hypothesis_text,
            "source_tokens": source_tokens,
            "hypothesis_tokens": hypothesis_tokens,
            "attention_matrix": attention_matrix,
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "attention_examples.json"
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
