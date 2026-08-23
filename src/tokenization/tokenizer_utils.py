from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    text = tok.decode(ids, skip_special_tokens=True)
    # add_prefix_space=True on the pre-tokenizer injects a synthetic leading
    # space before tokenization so the first word is treated like any other;
    # that space round-trips back on decode and must be dropped here
    if text.startswith(" "):
        text = text[1:]
    return text
