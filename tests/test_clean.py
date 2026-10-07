import unicodedata

from src.data.clean import (
    clean_text,
    nfc_normalize,
    normalize_whitespace,
    passes_word_count,
    strip_zero_width,
)

ZWSP = "​"
BOM = "﻿"
ZWJ = "‍"
ZWNJ = "‌"


def test_nfc_normalize_composes_decomposed_form():
    # "e" + combining acute accent (U+0301), decomposed form
    decomposed = "é"
    composed = nfc_normalize(decomposed)
    assert composed == "é"  # precomposed "é"
    assert unicodedata.is_normalized("NFC", composed)


def test_nfc_normalize_is_idempotent():
    text = "ଓଡିଆ"  # ଓଡ଼ିଆ ("Odia"), already NFC
    once = nfc_normalize(text)
    twice = nfc_normalize(once)
    assert once == twice == text


def test_strip_zero_width_removes_zwsp_and_bom_anywhere():
    text = f"{BOM}a{ZWSP}b{ZWSP}c"
    assert strip_zero_width(text) == "abc"


def test_strip_zero_width_strips_edge_joiners():
    # a lone joiner at the very start or end has nothing to join and is
    # not meaningful Indic typography -- must be stripped
    assert strip_zero_width(f"{ZWJ}abc") == "abc"
    assert strip_zero_width(f"abc{ZWNJ}") == "abc"
    assert strip_zero_width(f"{ZWJ}{ZWNJ}abc{ZWNJ}{ZWJ}") == "abc"


def test_strip_zero_width_preserves_lone_interior_joiner():
    # a single interior ZWJ/ZWNJ is meaningful Indic conjunct-formation
    # control and must survive untouched
    assert strip_zero_width(f"a{ZWJ}b") == f"a{ZWJ}b"
    assert strip_zero_width(f"a{ZWNJ}b") == f"a{ZWNJ}b"


def test_strip_zero_width_collapses_interior_run_to_one_char():
    # a run of 2+ interior joiners is a scraping artifact, not meaningful
    # typography -- collapsed down to a single character, not removed
    # entirely and not left as a multi-character run
    assert strip_zero_width(f"a{ZWJ}{ZWJ}b") == f"a{ZWJ}b"
    assert strip_zero_width(f"a{ZWJ}{ZWNJ}{ZWJ}b") == f"a{ZWJ}b"


def test_normalize_whitespace_collapses_and_strips():
    assert normalize_whitespace("  a   b\tc\n\nd  ") == "a b c d"


def test_clean_text_end_to_end():
    text = f"{BOM}  {ZWJ}{ZWJ}Hello{ZWSP}  world{ZWNJ}  \n\n"
    assert clean_text(text) == "Hello world"


def test_passes_word_count_boundaries():
    assert passes_word_count("a b c", min_words=3, max_words=5)  # exactly min
    assert passes_word_count("a b c d e", min_words=3, max_words=5)  # exactly max
    assert not passes_word_count("a b", min_words=3, max_words=5)  # below min
    assert not passes_word_count("a b c d e f", min_words=3, max_words=5)  # above max
