import re
import unicodedata

from configs.base import MAX_WORDS, MIN_WORDS

_ZWSP = "​"
_BOM = "﻿"
_ZWJ = "‍"
_ZWNJ = "‌"
_ZW_JOINERS = _ZWJ + _ZWNJ

_EDGE_JOINER_RUN = re.compile(f"^[{_ZW_JOINERS}]+|[{_ZW_JOINERS}]+$")
_INTERIOR_JOINER_RUN = re.compile(f"[{_ZW_JOINERS}]{{2,}}")
_WHITESPACE_RUN = re.compile(r"\s+")


def nfc_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_zero_width(text: str) -> str:
    text = text.replace(_ZWSP, "").replace(_BOM, "")
    text = _EDGE_JOINER_RUN.sub("", text)
    # a run of 2+ ZWJ/ZWNJ in the interior is a scraping artifact; a lone
    # one is meaningful Indic conjunct-formation and must survive
    text = _INTERIOR_JOINER_RUN.sub(lambda m: m.group(0)[0], text)
    return text


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    return normalize_whitespace(strip_zero_width(nfc_normalize(text)))


def passes_word_count(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> bool:
    n = len(text.split())
    return min_words <= n <= max_words


def inspect_nfc_anomalies(texts: list[str], sample_size: int = 500) -> dict:
    sample = texts[:sample_size]
    changed = [t for t in sample if unicodedata.normalize("NFC", t) != t]
    non_idempotent = [
        t for t in sample
        if unicodedata.normalize("NFC", t) != unicodedata.normalize("NFC", unicodedata.normalize("NFC", t))
    ]
    return {
        "sample_size": len(sample),
        "changed_by_nfc": len(changed),
        "changed_fraction": len(changed) / len(sample) if sample else 0.0,
        "non_idempotent_count": len(non_idempotent),
    }
