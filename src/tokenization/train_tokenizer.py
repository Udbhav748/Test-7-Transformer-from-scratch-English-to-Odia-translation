import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from configs.base import (
    EN_VOCAB_SIZE,
    EOS_ID,
    OR_VOCAB_SIZE,
    PAD_ID,
    SOS_ID,
    SPECIAL_TOKENS,
    TOKENIZER_DIR,
    UNK_ID,
)
from src.data.download import CANDIDATES_PATH

EN_TOKENIZER_PATH = TOKENIZER_DIR / "en_bpe.json"
OR_TOKENIZER_PATH = TOKENIZER_DIR / "or_bpe.json"


def _train_one(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tok.train_from_iterator(texts, trainer=trainer)

    tok.post_processor = processors.TemplateProcessing(
        single="<SOS> $A <EOS>",
        special_tokens=[("<SOS>", tok.token_to_id("<SOS>")), ("<EOS>", tok.token_to_id("<EOS>"))],
    )

    assert tok.token_to_id("<PAD>") == PAD_ID
    assert tok.token_to_id("<SOS>") == SOS_ID
    assert tok.token_to_id("<EOS>") == EOS_ID
    assert tok.token_to_id("<UNK>") == UNK_ID

    return tok


def main() -> None:
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    if EN_TOKENIZER_PATH.exists() and OR_TOKENIZER_PATH.exists():
        print(
            f"{EN_TOKENIZER_PATH} and {OR_TOKENIZER_PATH} already exist; skipping training. "
            "Delete them to force a retrain."
        )
        return

    df = pd.read_parquet(CANDIDATES_PATH)

    en_tok = _train_one(df["src"].tolist(), EN_VOCAB_SIZE)
    en_tok.save(str(EN_TOKENIZER_PATH))
    print(f"saved English tokenizer ({en_tok.get_vocab_size()} tokens) to {EN_TOKENIZER_PATH}")

    or_tok = _train_one(df["tgt"].tolist(), OR_VOCAB_SIZE)
    or_tok.save(str(OR_TOKENIZER_PATH))
    print(f"saved Odia tokenizer ({or_tok.get_vocab_size()} tokens) to {OR_TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
