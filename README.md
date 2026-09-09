# Test-7 — Transformer from Scratch: English → Odia Translation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-from--scratch-ee4c2c)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b)
![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)

A sequence-to-sequence Transformer built **from scratch in PyTorch** (no `nn.Transformer`) for
English → Odia machine translation, trained on the [AI4Bharat Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar)
corpus. Includes an interactive Streamlit dashboard for translating text, comparing two trained
models side by side, and inspecting every training/evaluation result.

**Author:** Udbhav Narawat — English → Odia NMT project for [Course Name]

## Contents

- [Results](#results)
- [Requirement Coverage](#requirement-coverage)
- [Limitations](#limitations)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Run the Dashboard](#run-the-dashboard)
- [Reproduce the Pipeline](#reproduce-the-pipeline)
- [Notebooks](#notebooks)
- [Testing](#testing)
- [Data & Acknowledgments](#data--acknowledgments)

## Results

Two models were trained and are compared throughout the dashboard and write-up:

| | Baseline | Scaled |
|---|---|---|
| Parameters | 4,005,696 | 11,469,824 |
| Layers (enc + dec) | 2 + 2 | 4 + 4 |
| Hidden size | 128 | 256 |
| Attention heads | 4 | 8 |
| Hardware | Kaggle CPU | Tesla T4 GPU |
| Epochs | 40 | 25 |
| Training data | 36,000 pairs | 60,000 pairs |
| Best validation loss | 2.85 | 3.56 |
| Test BLEU (greedy) | 2.60 | 0.31 |

The scaled model is architecturally stronger (Pre-LN, weight tying, more capacity) but was trained
for fewer epochs, so its raw greedy-decode BLEU is lower — beam search and repetition blocking
close most of that gap, as shown live in the dashboard's Model Comparison tab. Full analysis is in
[`reports/write_up.md`](reports/write_up.md).

## Requirement Coverage

Every requirement from the assignment spec is checked off against the actual code in
[`reports/requirements_coverage.json`](reports/requirements_coverage.json):

| Category | Requirements | Status |
|---|---|---|
| Data preparation | 6 | ✅ all met |
| Architecture (spec §5.6) | 7 | ✅ all met |
| Training | 5 | ✅ all met (1 exceeded) |
| Inference | 3 | ✅ all met (2 exceeded) |
| Evaluation | 4 | ✅ all met (1 exceeded) |
| Odia-specific / extra credit | 3 | ✅ all met (2 exceeded) |
| **Total** | **28** | **28/28 — 6 exceeded spec** |

"Exceeded" items include things the spec didn't strictly require but the project does anyway:
beam search, decode-time repetition blocking, Odia Unicode normalization, and an explicit
causal-mask bug check on the training curves.

## Limitations

- **Small model, trained from scratch, on limited compute.** The baseline is 4M parameters trained
  on a CPU; the scaled model is 11.5M parameters trained on a single GPU for 25 epochs. Production
  translation systems (e.g. AI4Bharat IndicTrans2, Meta NLLB-200) use 600M–1B+ parameters trained
  on tens of millions of sentence pairs — the BLEU scores here (2.60 / 0.31) reflect that gap in
  scale, not a bug in the implementation (all 18 tests pass, including an explicit causal-mask
  leak check).
- **Quality drops sharply on longer sentences.** Both models are prone to falling into repetition
  loops as source length increases (see the Training & Benchmarks tab) — beam search with n-gram
  blocking largely fixes this, greedy decoding does not.
- **The scaled model's raw BLEU is lower than the baseline's** because it was trained for fewer
  epochs (25 vs. 40) on a tighter time budget. With beam search it closes most of that gap — see
  the Model Comparison tab and `reports/write_up.md` for the full discussion.

## Screenshots

### Translator — empty state
![Translator tab, empty](docs/screenshots/01_translator_empty.png)

### Translator — live translation
Typing a sentence runs it through the model live, with token counts and decoding settings shown alongside the output.
![Translator tab, translated output](docs/screenshots/02_translator_result.png)

### Model Comparison
Baseline vs. scaled model, side by side — parameter counts, loss curves, architecture table, and example translations.
![Model Comparison tab](docs/screenshots/03_model_comparison.png)

### Training & Benchmarks
Training loss curve, BLEU score, and how translation quality changes with sentence length — including a confusion matrix testing whether sentence length alone predicts repetition failures.
![Training & Benchmarks tab](docs/screenshots/04_training_benchmarks.png)

### Architecture & Linguistics
Why Odia needs more subword tokens than English, and how the text pipeline works.
![Architecture & Linguistics tab](docs/screenshots/05_architecture_linguistics.png)

## Project Structure

```
app.py                    Streamlit dashboard (entry point)
project_data.py           Loads reports/checkpoints for the dashboard

src/                       Core library
├── data/                  Download, clean, split the parallel corpus
├── tokenization/          Byte-level BPE tokenizer training/loading
├── model/                 Transformer: attention, encoder, decoder, embeddings, masks
├── training/               Training loop, LR schedule, checkpointing
├── inference/               Greedy decode, beam search, repetition blocking, attention extraction
└── evaluation/             BLEU scoring, sample translation selection

configs/base.py            Frozen hyperparameters and paths
scripts/                   Pipeline entry points (run_eda.py, run_evaluation.py, ...)
tests/                     pytest unit + smoke tests

notebooks/                  Training notebooks
├── model_parameters_and_results.ipynb   Full walkthrough: architecture, data, training, eval
├── train_scaled_transformer_colab.ipynb  Trains the scaled model on a free Colab GPU
└── kaggle/                 Notebooks + metadata used to train on Kaggle GPU/CPU

reports/                    Results (tracked in git)
├── write_up.md              Full technical write-up
├── requirements_coverage.json
├── eval_results.json / scaled_eval_results.json
├── training_history.json / scaled_training_history.json
├── length_quality_analysis.json
└── figures/                  Generated comparison charts (gitignored, regenerate via scripts)

docs/screenshots/           Dashboard screenshots used in this README

data/, tokenizers/, checkpoints/   Generated locally by the pipeline (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with four tabs: Translator, Model Comparison, Training &
Benchmarks, and Architecture & Linguistics.

## Reproduce the Pipeline

1. `python -m src.data.download` — download raw pairs into `data/raw/`
2. `python -m src.data.split` — clean + split into `data/processed/{train,val,test}.parquet`
3. `python -m src.tokenization.train_tokenizer` — train the English/Odia BPE tokenizers
4. Train the model — full runs were done on Kaggle (see `notebooks/kaggle/`); for a quick local
   correctness check use `python scripts/run_local_smoke_test.py`
5. `python scripts/run_evaluation.py` — BLEU + sample translations
6. `python scripts/run_length_quality_analysis.py` — BLEU/repetition vs. sentence length
7. `python scripts/run_attention_demo.py` — attention-weight extraction for the dashboard
8. `python scripts/run_eda.py` — corpus/tokenizer statistics

## Notebooks

- **`notebooks/model_parameters_and_results.ipynb`** — the main results notebook: architecture
  breakdown, dataset stats, training curves, BLEU evaluation, sample translations, length-vs-quality
  analysis, attention visualization, and a baseline-vs-scaled comparison.
- **`notebooks/train_scaled_transformer_colab.ipynb`** — trains the scaled (11.5M param) model on a
  free Google Colab T4 GPU in about 20–25 minutes.
- **`notebooks/kaggle/`** — the notebooks actually run on Kaggle to produce the checkpoints used in
  this repo (`en_or_transformer.ipynb` for the baseline, `scaled_en_or_transformer.ipynb` for the
  scaled model), plus their Kaggle kernel metadata.

## Testing

```bash
pytest tests/
```

18 tests covering causal masking, model output shapes, repetition blocking, tokenizer round-trips,
and a full training-loop smoke test (loss must actually decrease and the checkpoint must reload
correctly).

![pytest run — 18 passed](docs/screenshots/06_pytest_run.png)

## Data & Acknowledgments

Training data is the English–Odia (`or`) config of
[AI4Bharat Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) (Ramesh et al., 2021),
a large-scale parallel corpus for Indic languages.
