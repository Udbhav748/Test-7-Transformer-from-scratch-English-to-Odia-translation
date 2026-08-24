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

### Real training run

Ground-truth training happened on Kaggle (this machine is CPU-only — Intel i3-1125G4, 4 cores,
~8GB RAM, no CUDA/MPS). The first Kaggle run attempt failed immediately: Kaggle assigned a Tesla
P100 GPU, but its compute architecture (sm_60) is no longer supported by the pre-installed PyTorch
build on that image, so the very first `forward()` call raised `CUDA error: no kernel image is
available for execution on the device`. The notebook was made robust to this by probing GPU
usability with an actual matmul at runtime (not just `torch.cuda.is_available()`, which only checks
driver presence) and falling back to CPU automatically. Training ran on Kaggle's CPU as a result.

An initial 18-epoch run (train loss 5.0468 -> 1.7868, val loss 3.2268 -> 1.9257) established that
the setup was correct and the causal mask was healthy, but validation loss had not plateaued —
it was still improving every single epoch. That run was extended to **40 epochs** with **label
smoothing (0.1)** added to the loss (a standard regularizer that softens the one-hot training
target; it changes the achievable loss floor, so the two runs' raw loss values are not directly
comparable). The architecture was not changed — `d=128, heads=4, N=2` throughout, per the
assignment's compute constraint. The final run took roughly 410-455 seconds/epoch, about 4.9 hours
total, over the full 36,000-pair train split at batch size 128.

![training and validation loss curve](figures/loss_curve.png)

| epoch | train loss | val loss |
|---|---|---|
| 1 | 5.5902 | 4.0746 |
| 10 | 3.0340 | 3.0447 |
| 20 | 2.8551 | 2.9212 |
| 30 | 2.7690 | 2.8715 |
| 40 | 2.7144 | 2.8464 |

Train and validation loss decrease together, smoothly, with validation consistently at or above
training and no discontinuous drop — the expected shape for a correctly-masked decoder, and the
opposite of the "suspiciously perfect" pattern the assignment warns is a leakage symptom. The curve
flattens over the last ~10 epochs (val loss 2.8715 -> 2.8464), indicating the model is approaching
what this architecture/data combination can achieve rather than being cut off mid-improvement.

## Evaluation

BLEU (sacrebleu, computed with identical postprocessing — strip special tokens, decode, normalize
whitespace — applied to both hypotheses and references) over the full 2,000-pair test split, using
greedy decoding with 3-gram repetition blocking (a decode-time addition described below):

**BLEU = 2.60** (`22.2/4.9/1.7/0.5` n-gram precisions, brevity penalty 0.829, hypothesis/reference
length ratio 0.842)

| run | epochs | label smoothing | repetition blocking | BLEU |
|---|---|---|---|---|
| baseline | 18 | no | no | 2.19 |
| enhanced | 40 | yes (0.1) | yes (n=3) | **2.60** |

A ~19% relative BLEU improvement from three changes that all stay within the assignment's
`d=128, heads=4, N=2` compute constraint: none of them add model capacity. This is still a low
score in absolute terms — expected for a deliberately small (4M-parameter) from-scratch transformer
trained on 36,000 sentence pairs, with no pretraining and CPU-only training limiting how much
compute the run could use. It is in line with what small from-scratch NMT systems trained on under
100k pairs typically produce.

### 5 sample translations

| Source (English) | Reference (Odia) | Model output (greedy, repetition-blocked) |
|---|---|---|
| Shutting down might cause them to lose unsaved work. | ବନ୍ଦ କରିବା ଫଳରେ ହୁଏତ ସେମାନେ ତାଙ୍କର ଅସଂରକ୍ଷିତ କାର୍ଯ୍ୟକୁ ହରାଇପାରନ୍ତି। | କାର୍ଯ୍ୟ କାମ କରିବାରେ ସେମାନଙ୍କୁ ସମବେଦନା ଜଣାପଡିଛି । |
| Bihar Chief Minister and JD(U) chief Nitish Kumar. | ବିହାର ମୁଖ୍ୟମନ୍ତ୍ରୀ ତଥା ଜେଡିୟୁ ମୁଖ୍ୟ ନୀତୀଶ କୁମାର ଜଣେ ଅତି ଚତୁର ରାଜନେତା। | ବିହାରରେ ମୁଖ୍ୟମନ୍ତ୍ରୀ ନୀତୀତିରୀତ୍ୱାରୀ ରାତିକାତାଷ୍ରେ ଅଛନ୍ତି । |
| Students will be focussed. | ଛାତ୍ରଛାତ୍ରୀମାନେ ଉତ୍ସୃଖଳିତ ହେବେ । | ଛାତ୍ରଛାତାତିରୀମାନେ ଶିକ୍ଷାରୀଙ୍କୁ କଡ଼ା ହେବ । |
| There is nothing on the ground. | ଜମି ବାଡ଼ି କିଛି ନାହିଁ । | କିଛି ବି କି ବାର୍ଯ୍ୟାଳୟ ନୁହେଁ। |
| **(long sentence, >=90th percentile source length)** BJP media cell head and Rajya Sabha member Anil Baluni dismissed the charge. | ଭାଜପା ନ୍ୟାସନାଲ ମିଡିଆ ମୁଖ୍ୟ ତଥା ରାଜ୍ୟସଭା ସାଂସଦ ଅନିଲ ବାଲୁନିଙ୍କୁ ଏହି ବଙ୍ଗଳା ଦିଆଯାଇଛି । | ରାଜ୍ୟସଭାରେ ବିଜେପି ଓ ମୁଖ୍ୟ ବିଜୟୀ କରିଛିଜି । |

### Discussion: the long-sentence example and limitations

The clearest change from the baseline checkpoint is on the long sentence itself: the earlier
18-epoch, no-smoothing, pure-greedy checkpoint produced a pure repetition loop
(`ବିଜେପି ଓ ବିଜେପି ଓ ବିର ମିଧାନସଭାରେ ବିଜେପି ଓ...`, "BJP and BJP and..." repeating). The enhanced
checkpoint instead produces a short, grammatically complete sentence that stays on-topic (BJP,
Rajya Sabha) without looping. Two independent changes contributed to this: more training/label
smoothing gave the model better-calibrated probabilities, and decode-time 3-gram repetition
blocking (`src/inference/repetition.py`) directly forbids the decoder from re-emitting an n-gram it
has already produced, regardless of how confident it is in doing so.

This is real progress, but it does not mean the underlying capacity limitation is solved — it means
the most visually obvious symptom of it is suppressed. A full-test-set measurement
(`reports/length_quality_analysis.json`, all 2,000 test examples, not just this one anecdote) still
shows a clear decline in translation quality as source sentences get longer: mean per-sentence BLEU
falls from 10.09 (3-5 word sentences) to 7.82 (6-8 words) to 6.69 (9-11 words) to 4.73 (12-15 words)
to 3.14 (16-20 words) to 2.80 (21+ words), a Pearson correlation of -0.235 against source word
length. A broader repetition-signature detector (any bigram recurring 3+ times anywhere in the
output, not just consecutively) still climbs from 16.3% on the shortest sentences to 76.0% on the
longest — repetition blocking prevents the exact-loop failure mode but cannot give the model
semantic content it never had the capacity or data to learn in the first place. The two
compounding causes remain (1) limited model capacity (`d=128`, only 2 decoder layers), deliberate
per the "must fit class compute" constraint, and (2) a 36,000-pair training set, an order of
magnitude smaller than production NMT systems use. Beam search (`src/inference/beam_search.py`,
implemented as the assignment's bonus item) is available as a further mitigation and is exposed
live in the dashboard's Translate tab alongside an attention-weight heatmap for inspecting exactly
which source tokens the decoder relied on for any given translation.
