# Test-7: From-Scratch Transformer, English -> Odia

## Architecture (assignment section 5.6)

Embedding + sinusoidal positional encoding -> N encoder blocks (self-attention -> FFN, each with
residual connection + LayerNorm) -> N decoder blocks (masked self-attention -> cross-attention ->
FFN, each with residual + LayerNorm) -> linear projection + softmax.

| hyperparameter | value |
|---|---|
| d_model | 128 |
| attention heads | 4 |
| feed-forward dim | 512 |
| encoder blocks (N) | 2 |
| decoder blocks (N) | 2 |
| dropout | 0.1 |
| output projection | separate `Linear(128, vocab)`, not tied to the target embedding (kept untied for strict compliance with "linear+softmax" as specified; weight tying is implemented as an easy constructor flag but is not the default) |
| total parameters | 4,005,696 |

Layer normalization is post-norm (residual -> dropout -> LayerNorm), matching the original
Vaswani et al. ordering the assignment is quoting, not the more recent pre-norm variant.

## Data

Source: `ai4bharat/samanantar`, English-Odia config. 58,000 candidate pairs were streamed and
cleaned (NFC Unicode normalization on both languages, zero-width-character cleanup that preserves
linguistically meaningful ZWJ/ZWNJ conjunct formation while stripping true junk like ZWSP/BOM,
whitespace normalization, and a 3-60 word count filter). Both English and Odia sides were then
tokenized with their trained subword tokenizers and any pair where either side exceeded `MAX_LEN=64`
subword tokens (including `<SOS>`/`<EOS>`) was dropped rather than truncated.

Retention at `MAX_LEN=64` measured on the real 58k candidate pool: **77.0%** (44,673 survivors),
consistent with an earlier 8,000-pair pilot measurement of 78.0%. From the survivors, 40,000 pairs
were seed-shuffled and split **36,000 train / 2,000 validation / 2,000 test**, with no source
sentence appearing in more than one split.

## Tokenization and Odia-specific notes (extra credit)

Two independent byte-level BPE tokenizers were trained (English, Odia), 8,000-token vocabulary
each, rather than one shared vocabulary — English and Odia share almost no Unicode code points, so
a shared BPE vocabulary would waste capacity relative to two per-language vocabularies at the same
total size. Special tokens `<PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3` are identical across both
tokenizers, applied via a `TemplateProcessing` post-processor that automatically wraps every
sequence as `<SOS> ... <EOS>`.

Subword tokenization mattered far more for Odia than for English in this project, for reasons that
go beyond a generic "morphologically rich language" caveat:

- **Script encoding cost.** Odia is a multi-byte UTF-8 script (Brahmic-derived, built from
  independent vowels, consonants, and combining vowel signs/virama sequences), so byte-level BPE
  starts from a much longer raw byte sequence per sentence than English's single-byte ASCII text.
  A pilot measurement on a small sample tokenizer showed a stark asymmetry: English sentences
  averaged 17.5 subword tokens (median 13) versus **49.8 for Odia (median 39)** at the same 8k
  vocabulary size and comparable sentence content. This asymmetry is exactly why `MAX_LEN=64` — a
  limit generous by English standards — still drops roughly a quarter of pairs: it is almost always
  the Odia side, not the English side, that exceeds the limit.
- **Morphology.** Odia is suffixation-heavy (case marking, postpositions, verb agreement all attach
  to the stem rather than appearing as separate words), so a fixed-size BPE vocabulary has to spend
  more of its budget capturing productive suffix patterns instead of whole words, which a purely
  frequency-driven BPE merge process does imperfectly at 8k merges trained on a few tens of
  thousands of sentences — a corpus size that is adequate but not large by subword-tokenizer
  training standards.
- **Conjunct/virama sequences and normalization.** Consonant conjuncts and vowel signs can, in
  principle, be represented by more than one equivalent Unicode byte sequence; NFC normalization
  before tokenizer training and before every encode call is what keeps "the same" Odia character
  from being learned and encoded as two different token sequences. An idempotency check on 500 Odia
  samples found 0 anomalies, i.e. NFC was already effectively normalizing this corpus's Odia text
  correctly, but this is corpus-dependent and should not be assumed without checking.

The practical consequence for translation quality: the decoder has to get many more subword
decisions right per sentence on the Odia side than an English-only intuition would suggest, and
BPE merge quality on a modest-sized, morphologically dense corpus is a real, first-order factor in
output quality here — not a minor implementation detail.

## Training and the causal-mask bug

Teacher forcing (`decoder_input = tgt[:, :-1]`, `target = tgt[:, 1:]`), cross-entropy loss with
`ignore_index=PAD_ID`, Adam (betas 0.9/0.98, eps 1e-9) with a Noam warmup schedule (900 warmup
steps), gradient clipping at norm 1.0.

The assignment specifically warns: *"if val loss is suspiciously perfect, your decoder is
peeking."* This was tested directly, not just watched for informally: with the model in `eval()`
mode (dropout off), the decoder was run twice on identical target-token prefixes but different
tokens after a cut position `t`, and the logits at every position `<= t` were asserted identical
between the two runs — proof the decoder's output at each step cannot depend on any future token.
A second, constructive check confirmed this test actually has teeth: the same invariance check was
run through a deliberately broken (all-True / no causal restriction) mask, and it failed as
expected, ruling out a vacuously-passing test. Explicit position-by-position mask coverage tests
additionally confirmed that future positions are always blocked, `<PAD>` positions are always
blocked regardless of causal position, and valid past/current non-pad positions are never
incorrectly blocked. All of this is in `tests/test_masks.py` and passes.

### Real training run (headline)

Training ran on Kaggle with a Tesla T4 GPU (Kaggle's CPU was not used for this run). The headline
run uses the spec-only code: no √d_model embedding scaling, plain cross-entropy with pad ignored, and
plain greedy decoding. It trained for 40 epochs on 36,000 pairs with batch size 128. Each epoch took
about 14 seconds.

| epoch | train loss | val loss |
|---|---|---|
| 1 | 4.9830 | 3.1863 |
| 10 | 1.6819 | 1.6899 |
| 20 | 1.4720 | 1.5901 |
| 30 | 1.3730 | 1.5646 |
| 39 | 1.3146 | **1.5455** |
| 40 | 1.3093 | 1.5489 |

Train and validation loss fall together with no sudden drop, which is the expected pattern for a
correctly masked decoder. The evaluated checkpoint is the best-validation one, epoch 39. Full history:
`reports/training_history.json`.

## Deviations from the assignment spec

The headline run follows the spec. The following are not in the headline run:

- **Scaled comparison model** (d=256, 8 heads, 4+4 blocks, Pre-LN, weight tying). It goes beyond the
  "start small" guidance, so it lives in `extras/model/`. It is not the section 5.6 architecture.
- **Beam search and n-gram repetition blocking.** Both are bonus or optional code in `extras/`. Neither
  is used in the headline evaluation.
- **Earlier runs** used label smoothing (0.1) and 3-gram blocking. Their numbers (BLEU 2.19 and 2.60)
  are not spec results and are no longer in the headline.
- **Tokenizer data hygiene.** Tokenizers train on a 15,000-pair pool that the split excludes from
  validation and test. The split asserts this, and the candidate pool is 75,000 so the 40,000-pair
  target still survives the length filter.


## Evaluation

Headline run: spec-only greedy decoding on the 2,000-pair test split, using `reports/eval_results.json`.
BLEU and chrF++ use sacrebleu, with identical postprocessing (strip special tokens, decode, normalize
whitespace) on hypotheses and references.

| metric | score |
|---|---|
| BLEU | **2.84** (`22.5 / 5.5 / 1.5 / 0.4` n-gram precisions, brevity penalty 0.950, hyp/ref length ratio 0.951) |
| chrF++ | **24.41** |
| test examples | 2,000 (decoded in 294 s) |

### 5 sample translations

| # | Source (English) | Reference (Odia) | Model output (greedy) |
|---|---|---|---|
| 1 | Chennai Super Kings made the cut. | ଚେନ୍ନଇ ସୁପର କିଙ୍ଗ୍‌ସ ଟସ୍ ଜିତି ଫିଲ୍‌ଡିଂ କରିଥିଲା। | ସୁପର ଚେନ୍ନାଇ ସୁପରକୁଟିଏ ସୁପର ମ୍ୟାଚ୍‌ରେ ସୁପରିକଳ୍ପ କରିଥିଲେ । |
| 2 | He died of excessive bleeding on the spot. | ପ୍ରଚୁର ରକ୍ତସ୍ରାବ ଯୋଗୁଁ ଘଟଣାସ୍ଥଳରେ ହିଁ ତାଙ୍କ ମୃତ୍ୟୁ ଘଟିଥିଲା। | ଘଟଣାସ୍ଥଳରେ ସେଠାରେ ସେଠାରେ ସେଠାରେ ପହଞ୍ଚିଥିଲା। |
| 3 | This, though, was not planned. | ତେବେ ଏହା ଆଦୌ ଯୋଜନାବଦ୍ଧ ନଥିଲା। | ତେବେ ଏହା କୌଣସି କାର୍ଯ୍ୟକାରୀ ହୋଇନଥିଲା। |
| 4 | Those injured have been admitted to a nearby hospital. | ଆହତ ଅବସ୍ଥାରେ ଉଦ୍ଧାର ହୋଇଥିବା ଶ୍ରମିକମାନଙ୍କୁ ନିକଟସ୍ଥ ଡାକ୍ତରଖାନାରେ ଭର୍ତ୍ତି କରାଯାଇଛି। | ସେମାନଙ୍କୁ ନିକଟସ୍ଥ ହସ୍ପିଟାଲରେ ଭର୍ତ୍ତି କରାଯାଇଛି। |
| 5 | **(long, ≥90th percentile)** On account of heavy rains in the city, the schools and colleges of Mumbai are shut. | ଲଗାଣ ବର୍ଷା ଯୋଗୁଁ ମୁମ୍ବାଇରେ ସ୍କୁଲ୍‌ ଓ କଲେଜ ବନ୍ଦ ରହିଛି ।  | ମୁମ୍ବାଇରେ ପ୍ରବଳ ବର୍ଷା ହେବାରୁ ପ୍ରବଳ ବର୍ଷା ହେବାରୁ ପ୍ରବଳ ବର୍ଷା ହୋଇଛି ।  |

Samples 3 and 4 are close in meaning to the reference. Samples 2 and 5 repeat a word or phrase, which
is the greedy loop failure the length analysis measures.

### Length vs quality (all 2,000 test pairs)

From `reports/length_quality_analysis.json`:

| source word count | pairs | mean sentence BLEU |
|---|---|---|
| 3–5 | 483 | 10.27 |
| 6–8 | 691 | 7.69 |
| 9–11 | 483 | 7.12 |
| 12–15 | 249 | 5.68 |
| 16–20 | 79 | 4.68 |
| 21+ | 15 | 3.40 |

Overall mean sentence BLEU is 7.77. The repetition rate across all outputs is 37.4%, and the
correlation with source word length is −0.20. Quality falls as sentences get longer.

### Discussion: limitations

The main limitation is greedy decoding on longer inputs. The model's greedy output repeats words or
phrases (samples 2 and 5), and the length study shows this rising with sentence length. Beam search
and n-gram blocking can reduce the loops, and both are in `extras/`, but they are not part of the
headline result and beam search has not been re-measured on this checkpoint.

The capacity limit remains: a 4M-parameter model (`d=128`, 2 decoder layers) trained on 36,000 pairs.
That size is deliberate under the "must fit class compute" constraint. It also explains the low
absolute BLEU, which is expected for a model this small trained from scratch.
