import re
import unicodedata

from configs.base import MAX_WORDS, MIN_WORDS

_ZWSP = "​"
_BOM = "﻿"
_ZWJ = "‍"
_ZWNJ = "‌"
_ZW_JOINERS = _ZWJ + _ZWNJ

# \s* around the joiner run (not just the run itself) so a joiner artifact
# separated from the true string boundary by whitespace -- common in real
# scraped text, e.g. "  ‍‍Hello world‌  \n" -- still counts
# as an edge joiner. Without it, a joiner preceded/followed only by
# whitespace before the real boundary would be missed here and then
# survive as the literal first/last character after normalize_whitespace
# strips the whitespace around it. Any whitespace this consumes is
# re-normalized by the normalize_whitespace() call that always follows
# this function in clean_text().
_EDGE_JOINER_RUN = re.compile(f"^\\s*[{_ZW_JOINERS}]+|[{_ZW_JOINERS}]+\\s*$")
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
    # changed_by_nfc / changed_fraction: how much of the sample was NOT
    # already in NFC form before normalization touched it -- this is the
    # meaningful corpus-quality signal (low values mean the source corpus
    # was already close to NFC-clean). Deliberately does NOT report an
    # "idempotency" check (NFC(x) vs NFC(NFC(x))): NFC is idempotent by
    # mathematical definition, so that check is always 0 regardless of
    # input and proves nothing about the corpus -- it was removed rather
    # than kept as a vacuous, trust-me diagnostic.
    sample = texts[:sample_size]
    changed = [t for t in sample if unicodedata.normalize("NFC", t) != t]
    return {
        "sample_size": len(sample),
        "changed_by_nfc": len(changed),
        "changed_fraction": len(changed) / len(sample) if sample else 0.0,
    }
