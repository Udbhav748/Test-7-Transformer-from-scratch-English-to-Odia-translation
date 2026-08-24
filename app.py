"""Dashboard for the from-scratch English->Odia Transformer (assignment section 5.6).

Reads pre-computed artifacts from project_data.py and renders them as a static
report -- no training or inference happens here.
"""

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="English to Odia Transformer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from project_data import (
        HYPERPARAMS,
        DATA_STATS,
        TOKENIZER_STATS,
        TRAINING_HISTORY,
        EVAL_RESULTS,
        EDA_RESULTS,
        ATTENTION_EXAMPLES,
        LENGTH_QUALITY_RESULTS,
        REQUIREMENTS_COVERAGE,
    )

    DATA_LOAD_ERROR = None
except Exception as exc:  # project_data.py not present or not importable yet
    HYPERPARAMS, DATA_STATS, TOKENIZER_STATS = {}, {}, {}
    TRAINING_HISTORY, EVAL_RESULTS, EDA_RESULTS = [], {}, {}
    ATTENTION_EXAMPLES, LENGTH_QUALITY_RESULTS, REQUIREMENTS_COVERAGE = [], {}, {}
    DATA_LOAD_ERROR = str(exc)

TEAL = "#0D9488"
AMBER = "#B45309"
CRITICAL = "#DC2626"
TEXT_MAIN = "#1E293B"
TEXT_MUTED = "#64748B"


def g(d, key, default="n/a"):
    """Defensive dict lookup tolerant of casing drift in the source module."""
    if not isinstance(d, dict):
        return default
    for variant in (key, key.lower(), key.upper()):
        if variant in d:
            return d[variant]
    return default


def fmt_num(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Oriya:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stApp, .stMarkdown, .stDataFrame, .stMetric,
.stTabs, button, input, textarea, [data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: #FDFCFB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.odia-text {
    font-family: 'Noto Sans Oriya', 'Inter', sans-serif !important;
    font-size: 1.05rem;
    line-height: 1.9;
}

.header-banner {
    background: linear-gradient(120deg, #0D9488 0%, #14B8A6 55%, #B45309 130%);
    border-radius: 18px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 24px rgba(13, 148, 136, 0.18);
}
.header-banner h1 {
    color: #FFFFFF;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.01em;
}
.header-banner p {
    color: #F0FDFA;
    font-size: 1.02rem;
    line-height: 1.6;
    margin: 0;
    max-width: 900px;
}

.section-heading {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1E293B;
    margin: 0.4rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #0D9488;
    display: inline-block;
}

.card-note {
    background-color: #F2F7F6;
    border: 1px solid #E1E9E7;
    border-left: 4px solid #0D9488;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    color: #1E293B;
    font-size: 0.95rem;
    line-height: 1.65;
    margin: 0.8rem 0 1.4rem 0;
}

[data-testid="stMetricValue"] {
    color: #0D9488;
    font-weight: 800;
}
[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 12px;
}

.arch-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    padding: 1.6rem 0.5rem;
}
.arch-box {
    background-color: #FFFFFF;
    border: 2px solid #0D9488;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    font-weight: 600;
    font-size: 0.9rem;
    color: #1E293B;
    text-align: center;
    box-shadow: 0 2px 8px rgba(13, 148, 136, 0.10);
    min-width: 130px;
}
.arch-box.accent {
    border-color: #B45309;
    background-color: #FFFBF3;
}
.arch-arrow {
    color: #0D9488;
    font-size: 1.5rem;
    font-weight: 700;
}

.result-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #FFFFFF;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.result-table th {
    background-color: #0D9488;
    color: #FFFFFF;
    text-align: left;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.result-table td {
    padding: 0.9rem 1rem;
    font-size: 0.92rem;
    color: #1E293B;
    vertical-align: top;
    border-bottom: 1px solid #EDF2F1;
}
.result-table tr:nth-child(even) td {
    background-color: #F8FBFA;
}
.result-table tr.long-row td {
    background-color: #FEF2F2;
    border-left: 4px solid #DC2626;
}
.long-badge {
    display: inline-block;
    background-color: #DC2626;
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    margin-bottom: 0.4rem;
}

.status-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    white-space: nowrap;
}
.status-met {
    background-color: #0D9488;
    color: #FFFFFF;
}
.status-exceeded {
    background-color: #B45309;
    color: #FFFFFF;
}
.status-not-met {
    background-color: #DC2626;
    color: #FFFFFF;
}

.spec-quote {
    background-color: #FFFBF3;
    border-left: 4px solid #B45309;
    border-radius: 8px;
    padding: 0.75rem 1.1rem;
    color: #1E293B;
    font-size: 0.9rem;
    font-style: italic;
    line-height: 1.6;
    margin: 0.6rem 0 1rem 0;
}

.evidence-where {
    color: #64748B;
    font-size: 0.8rem;
    font-family: 'Inter', monospace;
}

.bleu-signature {
    color: #64748B;
    font-size: 0.85rem;
    font-family: 'Inter', monospace;
    margin-top: -0.6rem;
}

.footer-note {
    color: #94A3B8;
    font-size: 0.82rem;
    text-align: center;
    margin-top: 2.5rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-banner">
        <h1>From-Scratch Transformer: English to Odia Translation</h1>
        <p>
            An encoder-decoder Transformer implemented from first principles per
            assignment section 5.6, deliberately kept small to fit limited compute
            budgets, and trained on a filtered subset of the Samanantar English-Odia
            parallel corpus. This dashboard reports the architecture, data pipeline,
            training curves, and held-out evaluation of the final checkpoint.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if DATA_LOAD_ERROR:
    st.error(
        "project_data.py could not be imported, so this dashboard has no data to "
        f"show yet. Underlying error: {DATA_LOAD_ERROR}"
    )
    st.stop()

@st.cache_resource(show_spinner=False)
def load_translation_model():
    from configs.base import CHECKPOINT_DIR
    from src.training.train import build_model
    from src.training.checkpoint import load_checkpoint
    from src.tokenization.tokenizer_utils import load_tokenizer
    from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH

    model = build_model()
    load_checkpoint(CHECKPOINT_DIR / "kaggle_run_best.pt", model)
    model.eval()
    return model, load_tokenizer(EN_TOKENIZER_PATH), load_tokenizer(OR_TOKENIZER_PATH)


(
    tab_overview,
    tab_requirements,
    tab_arch,
    tab_data,
    tab_eda,
    tab_training,
    tab_results,
    tab_translate,
    tab_techniques,
) = st.tabs(
    [
        "Overview",
        "Requirements",
        "Architecture",
        "Data and Tokenization",
        "EDA",
        "Training",
        "Results",
        "Live Demo",
        "NLP Techniques",
    ]
)

with tab_overview:
    st.markdown('<div class="section-heading">Headline results</div>', unsafe_allow_html=True)

    bleu = EVAL_RESULTS.get("bleu_score")
    total_params = g(HYPERPARAMS, "total_params")
    train_size = g(DATA_STATS, "train_size")
    val_size = g(DATA_STATS, "val_size")
    test_size = g(DATA_STATS, "test_size")
    final_val_loss = TRAINING_HISTORY[-1].get("val_loss") if TRAINING_HISTORY else None

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        with st.container(border=True):
            st.metric("BLEU score", f"{bleu:.2f}" if isinstance(bleu, (int, float)) else "n/a")
    with c2:
        with st.container(border=True):
            st.metric("Total parameters", fmt_num(total_params))
    with c3:
        with st.container(border=True):
            st.metric("Train examples", fmt_num(train_size))
    with c4:
        with st.container(border=True):
            st.metric("Val / test examples", f"{fmt_num(val_size)} / {fmt_num(test_size)}")
    with c5:
        with st.container(border=True):
            st.metric(
                "Final val loss",
                f"{final_val_loss:.3f}" if isinstance(final_val_loss, (int, float)) else "n/a",
            )

    epochs_run = len(TRAINING_HISTORY) if TRAINING_HISTORY else "n/a"
    first_val = TRAINING_HISTORY[0].get("val_loss") if TRAINING_HISTORY else None
    n_enc = g(HYPERPARAMS, "n_encoder_layers")
    n_dec = g(HYPERPARAMS, "n_decoder_layers")
    d_model = g(HYPERPARAMS, "d_model")
    n_heads = g(HYPERPARAMS, "n_heads")

    st.markdown(
        f"""
        <div class="card-note">
            Every component of this system was written from first principles -- the attention
            mechanism, positional encoding, encoder and decoder stacks, masking, training loop and
            both decoding strategies -- with no pretrained weights and no high-level Transformer
            library. The model is intentionally compact ({n_enc} encoder and {n_dec} decoder blocks,
            d_model {d_model}, {n_heads} attention heads) because the brief requires it to fit a
            class compute budget. Trained for {epochs_run} epochs, validation loss fell from
            {f"{first_val:.3f}" if isinstance(first_val, (int, float)) else "n/a"} to
            {f"{final_val_loss:.3f}" if isinstance(final_val_loss, (int, float)) else "n/a"}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Where to find each deliverable</div>', unsafe_allow_html=True)

    guide_rows = [
        ("Requirements", "Every line of the assignment brief mapped to the artefact that satisfies it, with the file where it can be verified."),
        ("Architecture", "The section 5.6 encoder-decoder structure, layer by layer, with the exact hyperparameters and parameter count."),
        ("Data and Tokenization", "Corpus selection, Unicode normalisation, the two per-language subword tokenizers, and the length-filtering funnel."),
        ("EDA", "Corpus analysis: length distributions, vocabulary richness, tokenizer coverage, and the English-Odia subword asymmetry."),
        ("Training", "Loss curves, the optimiser and warmup schedule, and the causal-mask verification the brief specifically warns about."),
        ("Results", "BLEU on the held-out test set, the five required sample translations, and a full-test-set analysis of how quality varies with sentence length."),
        ("Live Demo", "Translate any English sentence with the trained model, compare greedy against beam search, and inspect the decoder's attention."),
        ("NLP Techniques", "Reference table of every technique applied across preprocessing, tokenization, architecture, training, decoding and evaluation."),
    ]
    guide_html = ["<table class='result-table'><thead><tr><th style='width:22%'>Tab</th><th>What it shows</th></tr></thead><tbody>"]
    for tab_name, desc in guide_rows:
        guide_html.append(f"<tr><td><strong>{tab_name}</strong></td><td>{desc}</td></tr>")
    guide_html.append("</tbody></table>")
    st.markdown("".join(guide_html), unsafe_allow_html=True)

with tab_requirements:
    st.markdown('<div class="section-heading">Assignment requirements coverage</div>', unsafe_allow_html=True)

    if not REQUIREMENTS_COVERAGE:
        st.info("Requirements coverage data is not available.")
    else:
        summary = REQUIREMENTS_COVERAGE.get("summary", {})
        groups = REQUIREMENTS_COVERAGE.get("groups", [])

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            with st.container(border=True):
                st.metric("Requirements checked", fmt_num(summary.get("total_items", "n/a")))
        with s2:
            with st.container(border=True):
                st.metric("Fully met", fmt_num(summary.get("met", "n/a")))
        with s3:
            with st.container(border=True):
                st.metric("Exceeded", fmt_num(summary.get("exceeded", "n/a")))
        with s4:
            with st.container(border=True):
                st.metric("Automated tests passing", fmt_num(summary.get("tests_passing", "n/a")))

        not_met = summary.get("not_met", 0)
        if isinstance(not_met, int) and not_met > 0:
            st.warning(f"{not_met} requirement(s) are recorded as not met. See the detail below.")

        st.markdown(
            """
            <div class="card-note">
                Every row below maps one line of the assignment brief to the specific artefact that
                satisfies it, naming the file where it can be verified. "Exceeded" marks the items
                where the brief's optional or bonus work was completed, or where the verification
                goes beyond what was asked.
            </div>
            """,
            unsafe_allow_html=True,
        )

        _badge_class = {
            "met": "status-met",
            "exceeded": "status-exceeded",
            "not_met": "status-not-met",
        }
        _badge_label = {"met": "Met", "exceeded": "Exceeded", "not_met": "Not met"}

        for group in groups:
            st.markdown(
                f'<div class="section-heading" style="margin-top:1.4rem;font-size:1.1rem;">{group.get("group", "")}</div>',
                unsafe_allow_html=True,
            )
            spec_text = group.get("spec_text", "")
            if spec_text:
                st.markdown(f'<div class="spec-quote">"{spec_text}"</div>', unsafe_allow_html=True)

            rows = ["<table class='result-table'><thead><tr><th style='width:22%'>Requirement</th><th style='width:11%'>Status</th><th>Evidence</th></tr></thead><tbody>"]
            for item in group.get("items", []):
                status = str(item.get("status", "")).lower()
                cls = _badge_class.get(status, "status-met")
                label = _badge_label.get(status, status.title())
                where = item.get("where", "")
                where_html = f'<br/><span class="evidence-where">{where}</span>' if where else ""
                rows.append(
                    "<tr>"
                    f"<td><strong>{item.get('requirement', '')}</strong></td>"
                    f"<td><span class='status-badge {cls}'>{label}</span></td>"
                    f"<td>{item.get('evidence', '')}{where_html}</td>"
                    "</tr>"
                )
            rows.append("</tbody></table>")
            st.markdown("".join(rows), unsafe_allow_html=True)

with tab_techniques:
    st.markdown('<div class="section-heading">NLP techniques used in this project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-note">
            Every technique below is actually implemented in this project's code, not a generic
            checklist -- each row states what the technique does and specifically how and why it is
            used here, grounded in the real configuration and measurements reported elsewhere in
            this dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )

    def technique_table(rows):
        html = ["<table class='result-table'><thead><tr><th>Technique</th><th>What it does</th><th>How it is used in this project</th></tr></thead><tbody>"]
        for name, what, how in rows:
            html.append(f"<tr><td><strong>{name}</strong></td><td>{what}</td><td>{how}</td></tr>")
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown('<div class="section-heading" style="margin-top:0.4rem;font-size:1.1rem;">1. Text preprocessing and normalization</div>', unsafe_allow_html=True)
    technique_table([
        ("Unicode NFC normalization", "Canonicalizes text so visually identical characters that could be encoded as different Unicode byte sequences become one consistent representation.", "Applied to both languages before any tokenization. Critical for Odia, where consonant conjuncts and vowel-sign sequences can have multiple equivalent encodings -- without this step the same visible character could silently split the BPE vocabulary in two."),
        ("Zero-width character handling", "Selectively removes zero-width characters that carry no linguistic meaning while preserving ones that do.", "ZWJ (U+200D) and ZWNJ (U+200C) control conjunct formation in Indic scripts and are preserved except at string edges or in runs of 2+ (a scraping artifact, collapsed to one). Truly meaningless ZWSP (U+200B) and BOM (U+FEFF) are always stripped."),
        ("Whitespace normalization", "Collapses irregular whitespace runs into single spaces.", "Applied to both languages after zero-width cleanup, before the word-count filter."),
        ("Length filtering", "Removes sentences that are too short (likely noise) or too long (likely misaligned/multi-sentence scraping errors).", "A 3-60 word filter runs first as a cheap pass, followed by a stricter subword-length filter (see Data tab) once tokenizers are trained."),
        ("Deduplication", "Ensures no sentence appears more than once, and specifically not across more than one data split.", "Exact-duplicate English source sentences are removed before the train/validation/test split, preventing train-test leakage."),
    ])

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;font-size:1.1rem;">2. Tokenization</div>', unsafe_allow_html=True)
    technique_table([
        ("Byte-Pair Encoding (BPE)", "Learns a fixed-size vocabulary of frequently occurring subword units by iteratively merging the most common adjacent symbol pairs, letting rare/unseen words be represented as combinations of known pieces.", "Two independent 8,000-token BPE vocabularies are trained (English, Odia) via the HuggingFace tokenizers library, on the cleaned candidate pool."),
        ("Byte-level pre-tokenization", "Operates on raw UTF-8 bytes rather than characters, so every possible input string is representable without any truly out-of-vocabulary input.", "This is exactly why the measured out-of-vocabulary rate on this dataset is 0.0% for both languages (EDA tab) -- a structural guarantee of the tokenization scheme, not a lucky coincidence of the corpus."),
        ("Separate per-language vocabularies", "Trains an independent subword vocabulary for each language instead of one shared vocabulary.", "English and Odia share almost no Unicode code points, so a shared vocabulary would waste capacity; each language gets the full 8,000-token budget spent entirely on its own script."),
        ("Special token scheme", "Reserves fixed vocabulary ids for structural tokens the model needs beyond real words.", "&lt;PAD&gt;=0, &lt;SOS&gt;=1, &lt;EOS&gt;=2, &lt;UNK&gt;=3, identical ids in both tokenizers, automatically wrapped onto every sequence via a TemplateProcessing post-processor."),
    ])

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;font-size:1.1rem;">3. Transformer architecture</div>', unsafe_allow_html=True)
    technique_table([
        ("Scaled token embeddings", "Converts discrete token ids into continuous vectors, scaled by the square root of the model dimension.", "Each embedding lookup is multiplied by sqrt(128) before positional information is added, keeping embedding and positional-encoding magnitudes comparable, per the original Transformer paper."),
        ("Sinusoidal positional encoding", "Injects sequence-order information using fixed sine and cosine functions at different frequencies across the embedding dimensions, since attention itself has no built-in notion of order.", "A precomputed sin/cos table is added to every token embedding before it enters the encoder or decoder stack."),
        ("Multi-head self-attention", "Lets every position attend to every other position in the same sequence, split across several parallel attention heads that can specialize in different relationships.", "4 attention heads of 32 dimensions each, used for both encoder self-attention and (in masked form) decoder self-attention."),
        ("Masked (causal) self-attention", "Restricts each decoder position to attending only to itself and earlier positions, never future ones, which is what makes autoregressive generation valid.", "Enforced via a lower-triangular mask combined with the padding mask; verified directly by a dedicated automated test rather than only inferred from the loss curve (see Training tab)."),
        ("Cross-attention", "Lets the decoder attend over the full encoder output, which is how information from the source sentence reaches the target-language generation process.", "Each of the 2 decoder blocks has a cross-attention sub-layer between its masked self-attention and its feed-forward sub-layer."),
        ("Position-wise feed-forward network", "A two-layer fully connected network applied independently and identically at every sequence position, adding representational capacity beyond attention alone.", "128 -> 512 -> 128 with a ReLU activation, present in every encoder and decoder block."),
        ("Residual connections + layer normalization", "Adds each sub-layer's input back to its output and normalizes the result, which is what makes a multi-block network trainable by keeping gradients well-behaved.", "Applied after every sub-layer (self-attention, cross-attention, feed-forward) in post-norm order: residual add, dropout, then LayerNorm."),
        ("Padding masks", "Prevents attention from being influenced by &lt;PAD&gt; positions that exist only to let variable-length sentences share a batch.", "Applied in encoder self-attention, decoder self-attention, and cross-attention wherever padding could otherwise leak into a real prediction."),
        ("Dropout regularization", "Randomly zeroes a fraction of activations during training to reduce overfitting.", "Rate 0.1, applied after attention and feed-forward sub-layers and on the positional encoding output -- meaningful on a comparatively small 36,000-pair training set."),
        ("Weight-tying (implemented, disabled by default)", "An optional technique where the output projection shares its weight matrix with the target-side embedding table, reducing parameter count.", "Implemented as a constructor flag but left off by default, since the assignment specifies a plain linear-plus-softmax output head rather than this additional technique."),
    ])

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;font-size:1.1rem;">4. Training methodology</div>', unsafe_allow_html=True)
    technique_table([
        ("Teacher forcing", "Feeds the true previous target token as decoder input at every training step, instead of the model's own (possibly wrong) prediction.", "Standard for sequence-to-sequence training here -- the target sequence is shifted by one position to build decoder-input/decoder-target pairs."),
        ("Cross-entropy loss with padding ignored", "Measures how well predicted next-token probabilities match the true next token, while explicitly excluding positions that only exist for batch padding.", "PyTorch's CrossEntropyLoss with ignore_index set to the &lt;PAD&gt; id, so padding never contributes to the loss or its gradients."),
        ("Adam optimizer (Transformer-tuned)", "A gradient-based optimizer that adapts its per-parameter step size using running estimates of gradient mean and variance.", "Uses betas=(0.9, 0.98) and eps=1e-9, matching the original Transformer paper's settings rather than PyTorch's defaults."),
        ("Learning-rate warmup (Noam schedule)", "Ramps the learning rate up linearly for an initial number of steps, then decays it proportional to the inverse square root of the step count.", "900 warmup steps; prevents large, destabilizing parameter updates before the attention layers have started to form sensible patterns."),
        ("Gradient clipping", "Caps the overall gradient norm at every step to prevent occasional large gradients from destabilizing training.", "Clipped to a maximum norm of 1.0 on every optimizer step."),
        ("Best-checkpoint selection", "Retains the model state from whichever epoch had the lowest validation loss, rather than simply the last epoch.", "Validation loss is tracked every epoch and the best-scoring checkpoint is saved separately from the final one, so an overfitting tail would never silently become the evaluated model."),
        ("Label smoothing", "Softens the one-hot training target so the model is penalised for extreme overconfidence, which improves calibration and reduces repetitive output.", "Applied at 0.1 in the cross-entropy loss. Note this shifts the absolute loss floor, so smoothed and unsmoothed runs are not directly comparable by loss value alone."),
    ])

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;font-size:1.1rem;">5. Decoding strategies</div>', unsafe_allow_html=True)
    technique_table([
        ("Greedy decoding", "Generates one token at a time by always picking the single highest-probability next token and feeding it back in, until an end-of-sequence token or a length limit is reached.", "The required decoding strategy for this assignment; used for every translation shown on the Results and Translate tabs by default."),
        ("Beam search decoding", "Maintains several candidate partial translations simultaneously (a beam), scored by length-normalized cumulative log-probability, exploring more of the output space than greedy decoding.", "Implemented as the assignment's optional bonus item, kept in an isolated module so it cannot affect the required greedy path; available live via the checkbox on the Live Demo tab."),
        ("No-repeat n-gram blocking", "Prevents the decoder from emitting any n-gram it has already produced in the same sequence, by masking the offending continuation token before the arg-max is taken.", "Applied at n=3 to both greedy and beam decoding. This targets the degenerate repetition loop that is the dominant failure mode on longer sentences, and needs no retraining since it acts purely at inference time."),
        ("Attention inspection", "Reads out the decoder's cross-attention distribution over the source sentence for each generated token, showing which source words the model was relying on.", "Exposed as a non-invasive side channel that leaves the forward computation numerically unchanged, and rendered as a heatmap in the Live Demo tab."),
    ])

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;font-size:1.1rem;">6. Evaluation methodology</div>', unsafe_allow_html=True)
    technique_table([
        ("BLEU score", "An automated n-gram precision metric that compares machine output against human reference translations, the standard metric for machine translation quality.", "Computed via sacrebleu over the full 2,000-sentence held-out test set, with identical postprocessing (strip special tokens, decode, normalize whitespace) applied to hypotheses and references so the score reflects translation quality, not formatting artifacts."),
        ("Held-out test evaluation", "Reserves a portion of the data that is never used for training or model selection, so its score is an unbiased estimate of generalization.", "The 2,000-sentence test split is distinct from both the 36,000-sentence training split and the 2,000-sentence validation split used for checkpoint selection."),
        ("Causal-mask leakage testing", "An automated test that proves a decoder cannot be attending to future tokens, rather than only inferring correctness from an unusually good loss curve.", "The decoder is run twice on identical tokens up to a cut position and different tokens after it; the outputs before the cut are asserted bit-for-bit identical. The assignment explicitly warns that a suspiciously perfect loss curve is a symptom of exactly this bug."),
    ])

    st.markdown('<div class="section-heading" style="margin-top:1.4rem;font-size:1.1rem;">7. Corpus and linguistic analysis (EDA tab)</div>', unsafe_allow_html=True)
    technique_table([
        ("Type-token ratio (TTR)", "The ratio of unique words to total word occurrences, a standard measure of lexical diversity.", "Computed independently for English and Odia to compare how repetitive versus varied each language's vocabulary usage is across the corpus."),
        ("Pearson correlation", "A statistical measure of how linearly two variables move together.", "Used to check whether English sentence length predicts Odia sentence length, as a sanity check that the parallel corpus is genuinely aligned rather than noisy."),
        ("Zipfian frequency analysis", "Examines whether word frequency follows the expected natural-language pattern of a small number of words accounting for a large share of occurrences.", "Used as a corpus sanity check on the top-word frequency tables, not as a modeling technique."),
    ])

with tab_arch:
    st.markdown('<div class="section-heading">Model configuration</div>', unsafe_allow_html=True)

    hp_rows = [
        ("Model dimension (d_model)", g(HYPERPARAMS, "d_model")),
        ("Attention heads", g(HYPERPARAMS, "n_heads")),
        ("Feed-forward dimension (d_ff)", g(HYPERPARAMS, "d_ff")),
        ("Encoder layers", g(HYPERPARAMS, "n_encoder_layers")),
        ("Decoder layers", g(HYPERPARAMS, "n_decoder_layers")),
        ("Dropout", g(HYPERPARAMS, "dropout")),
        ("Output projection tied to embedding", g(HYPERPARAMS, "tie_output_projection")),
        ("Layer norm style", g(HYPERPARAMS, "layer_norm_style")),
        ("English vocab size", g(HYPERPARAMS, "en_vocab_size")),
        ("Odia vocab size", g(HYPERPARAMS, "or_vocab_size")),
        ("Max sequence length", g(HYPERPARAMS, "max_len")),
        ("Batch size (Kaggle)", g(HYPERPARAMS, "batch_size_kaggle")),
        ("Epochs (Kaggle)", g(HYPERPARAMS, "num_epochs_kaggle")),
        ("Warmup steps", g(HYPERPARAMS, "warmup_steps")),
        ("Total parameters", fmt_num(g(HYPERPARAMS, "total_params"))),
    ]

    cols = st.columns(3)
    for idx, (label, value) in enumerate(hp_rows):
        with cols[idx % 3]:
            with st.container(border=True):
                st.metric(label, str(value))

    st.markdown('<div class="section-heading" style="margin-top:1.6rem;">Encoder-decoder flow</div>', unsafe_allow_html=True)

    n_enc = g(HYPERPARAMS, "n_encoder_layers", "N")
    n_dec = g(HYPERPARAMS, "n_decoder_layers", "N")
    st.markdown(
        f"""
        <div class="arch-flow">
            <div class="arch-box">Token Embedding<br/>+ Positional Encoding</div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-box">{n_enc}&times; Encoder Block<br/><span style="font-weight:400;font-size:0.78rem;">self-attn &rarr; FFN</span></div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-box accent">{n_dec}&times; Decoder Block<br/><span style="font-weight:400;font-size:0.78rem;">masked self-attn &rarr; cross-attn &rarr; FFN</span></div>
            <div class="arch-arrow">&#8594;</div>
            <div class="arch-box">Linear + Softmax<br/><span style="font-weight:400;font-size:0.78rem;">Odia vocab distribution</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_data:
    st.markdown('<div class="section-heading">Dataset composition</div>', unsafe_allow_html=True)

    ds_cols = st.columns(4)
    ds_items = [
        ("Candidate pool", fmt_num(g(DATA_STATS, "candidate_pool_size"))),
        ("Retention rate", f"{g(DATA_STATS, 'retention_rate_pct')}%"),
        ("Train / val / test", f"{fmt_num(g(DATA_STATS,'train_size'))} / {fmt_num(g(DATA_STATS,'val_size'))} / {fmt_num(g(DATA_STATS,'test_size'))}"),
        ("Total pairs used", fmt_num(g(DATA_STATS, "total_size"))),
    ]
    for col, (label, value) in zip(ds_cols, ds_items):
        with col:
            with st.container(border=True):
                st.metric(label, value)

    st.markdown('<div class="section-heading" style="margin-top:1.6rem;">Subword length distribution</div>', unsafe_allow_html=True)

    en_stats = TOKENIZER_STATS.get("english", {}) if isinstance(TOKENIZER_STATS, dict) else {}
    or_stats = TOKENIZER_STATS.get("odia", {}) if isinstance(TOKENIZER_STATS, dict) else {}

    metrics_order = ["mean", "median", "p90", "p95", "p99", "max"]
    length_rows = []
    for m in metrics_order:
        if m in en_stats:
            length_rows.append({"metric": m, "language": "English", "value": en_stats.get(m)})
        if m in or_stats:
            length_rows.append({"metric": m, "language": "Odia", "value": or_stats.get(m)})

    if length_rows:
        length_df = pd.DataFrame(length_rows)
        length_chart = (
            alt.Chart(length_df)
            .mark_bar(size=18, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("metric:N", sort=metrics_order, title="Statistic", axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("language:N", sort=["English", "Odia"]),
                y=alt.Y("value:Q", title="Subword tokens"),
                color=alt.Color(
                    "language:N",
                    sort=["English", "Odia"],
                    scale=alt.Scale(domain=["English", "Odia"], range=[TEAL, AMBER]),
                    legend=alt.Legend(title="Language", orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("language:N", title="Language"),
                    alt.Tooltip("metric:N", title="Statistic"),
                    alt.Tooltip("value:Q", title="Subword tokens"),
                ],
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
            .configure_legend(labelColor=TEXT_MAIN, titleColor=TEXT_MAIN)
            .properties(height=340, background="#FFFFFF")
        )
        st.altair_chart(length_chart, use_container_width=True)

    retention = TOKENIZER_STATS.get("retention_at_max_len", {}) if isinstance(TOKENIZER_STATS, dict) else {}
    if retention:
        ret_df = pd.DataFrame(
            [{"max_len": int(k), "retention_pct": v} for k, v in retention.items()]
        ).sort_values("max_len")
        ret_chart = (
            alt.Chart(ret_df)
            .mark_line(point=alt.OverlayMarkDef(size=70, filled=True, color=TEAL), color=TEAL, strokeWidth=2.5)
            .encode(
                x=alt.X("max_len:Q", title="Max sequence length (subword tokens)"),
                y=alt.Y("retention_pct:Q", title="Pair retention (%)"),
                tooltip=[
                    alt.Tooltip("max_len:Q", title="Max length"),
                    alt.Tooltip("retention_pct:Q", title="Retention %"),
                ],
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
            .properties(height=280, background="#FFFFFF", title="Pair retention vs. max sequence length")
        )
        st.altair_chart(ret_chart, use_container_width=True)

    st.markdown(
        """
        <div class="card-note">
            Odia subword sequences run roughly three times longer than their English
            counterparts at every percentile. Odia is written in a multi-byte Brahmic
            script with heavy use of conjunct consonants and vowel-sign diacritics,
            so a single visual syllable often decomposes into several subword units.
            Odia's richer verb and noun morphology also adds inflectional material
            that English expresses with separate function words, further inflating
            token counts per sentence.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_eda:
    st.markdown('<div class="section-heading">Exploratory data analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-note">
            Computed directly on the full 40,000-pair processed dataset (train, validation,
            and test splits combined) using the actual production tokenizers, not the small
            pilot sample used to size <code>MAX_LEN</code> earlier.
        </div>
        """,
        unsafe_allow_html=True,
    )

    en_words = EDA_RESULTS.get("en_word_counts", [])
    or_words = EDA_RESULTS.get("or_word_counts", [])
    en_subwords = EDA_RESULTS.get("en_subword_counts", [])
    or_subwords = EDA_RESULTS.get("or_subword_counts", [])
    ratios = EDA_RESULTS.get("subword_ratio", [])
    top_en = EDA_RESULTS.get("top_words_en", [])
    top_or = EDA_RESULTS.get("top_words_or", [])
    split_comp = EDA_RESULTS.get("split_composition", {})

    en_chars = EDA_RESULTS.get("en_char_counts", [])
    or_chars = EDA_RESULTS.get("or_char_counts", [])
    en_awl = EDA_RESULTS.get("en_avg_word_len", [])
    or_awl = EDA_RESULTS.get("or_avg_word_len", [])
    en_unk = EDA_RESULTS.get("en_unk_counts", [])
    or_unk = EDA_RESULTS.get("or_unk_counts", [])
    en_unk_rate = EDA_RESULTS.get("en_unk_rate_pct")
    or_unk_rate = EDA_RESULTS.get("or_unk_rate_pct")
    en_end_punct = EDA_RESULTS.get("en_ending_punct", {})
    or_end_punct = EDA_RESULTS.get("or_ending_punct", {})
    en_pct_digits = EDA_RESULTS.get("en_pct_with_digits")
    or_pct_digits = EDA_RESULTS.get("or_pct_with_digits")
    vocab_stats = EDA_RESULTS.get("vocab_stats", {})
    length_corr = EDA_RESULTS.get("length_correlation", {})
    cleaning_funnel = EDA_RESULTS.get("cleaning_funnel", {})
    descriptive_stats = EDA_RESULTS.get("descriptive_stats", {})
    top_en_full = EDA_RESULTS.get("top_words_en_full", [])
    top_or_full = EDA_RESULTS.get("top_words_or_full", [])

    EDA_PALETTE_5 = [TEAL, AMBER, CRITICAL, "#0369A1", "#64748B"]

    def _pie(counts, colors=None, height=320, sort_desc=True):
        items = list(counts.items())
        if sort_desc:
            items.sort(key=lambda kv: kv[1], reverse=True)
        cats = [str(k) for k, _ in items]
        vals = [v for _, v in items]
        total = sum(vals) or 1
        df = pd.DataFrame({"category": cats, "value": vals, "pct": [v / total * 100 for v in vals]})
        color_range = (colors or EDA_PALETTE_5)[: len(cats)]
        return (
            alt.Chart(df)
            .mark_arc(innerRadius=65, outerRadius=125, stroke="#FFFFFF", strokeWidth=2)
            .encode(
                theta=alt.Theta("value:Q", stack=True),
                color=alt.Color(
                    "category:N",
                    sort=cats,
                    scale=alt.Scale(domain=cats, range=color_range),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("value:Q", title="Value"),
                    alt.Tooltip("pct:Q", title="Share", format=".1f"),
                ],
            )
            .configure_view(strokeWidth=0)
            .properties(height=height, background="#FFFFFF")
        )

    def _int_hist(values, label, color, max_x):
        df = pd.DataFrame({label: values})
        return (
            alt.Chart(df)
            .mark_bar(color=color)
            .encode(
                x=alt.X(f"{label}:Q", bin=alt.Bin(step=1, extent=[0, max_x]), title=label),
                y=alt.Y("count():Q", title="Sentence pairs"),
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
            .properties(height=260, background="#FFFFFF")
        )

    def _hist(values, label, color, max_x=None):
        df = pd.DataFrame({label: values})
        x_enc = alt.X(f"{label}:Q", bin=alt.Bin(maxbins=40), title=label)
        if max_x:
            x_enc = alt.X(f"{label}:Q", bin=alt.Bin(maxbins=40, extent=[0, max_x]), title=label, scale=alt.Scale(domain=[0, max_x]))
        return (
            alt.Chart(df)
            .mark_bar(color=color)
            .encode(x=x_enc, y=alt.Y("count():Q", title="Sentence pairs"))
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
            .properties(height=260, background="#FFFFFF")
        )

    if en_words and or_words:
        st.markdown('<div class="section-heading" style="margin-top:0.4rem;">Sentence length in words</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(_hist(en_words, "English word count", TEAL, max_x=40), use_container_width=True)
        with col2:
            st.altair_chart(_hist(or_words, "Odia word count", AMBER, max_x=40), use_container_width=True)

        import statistics as _stats
        st.markdown(
            f"""
            <div class="card-note">
                English sentences average {_stats.mean(en_words):.1f} words (median
                {_stats.median(en_words):.0f}), while Odia sentences for the same content average
                {_stats.mean(or_words):.1f} words (median {_stats.median(or_words):.0f}) -- slightly
                fewer, not more. This is consistent with Odia's agglutinative morphology: case
                markers, postpositions, and connective particles that English spells as separate
                words are frequently fused onto the preceding word in Odia, so the same content is
                expressed in fewer, denser word units.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if en_subwords and or_subwords:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Sentence length in subword tokens</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            st.altair_chart(_hist(en_subwords, "English subword count", TEAL, max_x=100), use_container_width=True)
        with col4:
            st.altair_chart(_hist(or_subwords, "Odia subword count", AMBER, max_x=100), use_container_width=True)

        import statistics as _stats
        en_mean, or_mean = _stats.mean(en_subwords), _stats.mean(or_subwords)
        st.markdown(
            f"""
            <div class="card-note">
                The picture reverses completely once text is broken into the subword units the
                model actually consumes: English averages {en_mean:.1f} subword tokens per sentence,
                Odia averages {or_mean:.1f} -- roughly {or_mean / en_mean:.1f} times longer, despite
                having fewer words. Odia's multi-byte Brahmic script and heavy use of consonant
                conjuncts and vowel-sign diacritics mean a single word, and even a single visual
                syllable, routinely decomposes into several subword pieces. This is also why the
                production tokenizer (trained on the full 58,000-pair candidate pool) produced
                noticeably shorter Odia sequences than the earlier 8,000-pair pilot tokenizer
                estimated (mean {or_mean:.1f} here versus 49.8 in the pilot) -- more training text
                let BPE learn more efficient merges, confirming that pilot measurement was a
                conservative upper bound rather than the final answer.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if ratios:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Odia-to-English length ratio</div>', unsafe_allow_html=True)
        st.altair_chart(_hist(ratios, "Odia subwords per English subword", TEAL, max_x=8), use_container_width=True)

        import statistics as _stats
        st.markdown(
            f"""
            <div class="card-note">
                For a typical sentence pair, Odia needs {_stats.median(ratios):.2f} times as many
                subword tokens as English to express the same content (median ratio; mean
                {_stats.mean(ratios):.2f}). This asymmetry is the direct explanation for why a
                shared <code>MAX_LEN=64</code> budget disproportionately truncates the Odia side of
                the corpus rather than the English side, and why the pair-retention rate reported in
                the Data and Tokenization tab is driven almost entirely by Odia sentence length, not
                English.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if top_en and top_or:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Most frequent words</div>', unsafe_allow_html=True)
        col5, col6 = st.columns(2)

        def _top_words_chart(pairs, color, font_class=None):
            df = pd.DataFrame(pairs, columns=["word", "count"])
            chart = (
                alt.Chart(df)
                .mark_bar(color=color)
                .encode(
                    x=alt.X("count:Q", title="Occurrences"),
                    y=alt.Y("word:N", sort="-x", title=None),
                    tooltip=["word", "count"],
                )
                .configure_view(strokeWidth=0)
                .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
                .properties(height=380, background="#FFFFFF")
            )
            return chart

        with col5:
            st.caption("English (common stopwords removed)")
            st.altair_chart(_top_words_chart(top_en, TEAL), use_container_width=True)
        with col6:
            st.caption("Odia")
            st.altair_chart(_top_words_chart(top_or, AMBER), use_container_width=True)

        st.markdown(
            """
            <div class="card-note">
                English's most frequent content words -- "police", "india", "government" among
                them -- reflect Samanantar's journalistic source material rather than general
                conversational English. Odia's most frequent words are almost entirely closed-class
                grammatical particles and pronouns (roughly: this, he/she/they, and, for, also,
                it, not, after, a, and), essentially unfiltered since no curated Odia stopword list
                was applied. That the top of the frequency list is dominated by function words
                either way, in both languages independently, is the expected Zipfian shape of
                natural-language word frequency, not an artifact of this particular corpus.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not (en_words or top_en or split_comp):
        st.info("No EDA results available.")

    if en_chars and or_chars:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Sentence length in characters</div>', unsafe_allow_html=True)
        col7, col8 = st.columns(2)
        with col7:
            st.altair_chart(_hist(en_chars, "English character count", TEAL), use_container_width=True)
        with col8:
            st.altair_chart(_hist(or_chars, "Odia character count", AMBER), use_container_width=True)

        import statistics as _stats
        en_c_mean, or_c_mean = _stats.mean(en_chars), _stats.mean(or_chars)
        _longer_chars = "Odia" if or_c_mean > en_c_mean else "English"
        _char_ratio = max(or_c_mean, en_c_mean) / min(or_c_mean, en_c_mean)
        st.markdown(
            f"""
            <div class="card-note">
                English sentences average {en_c_mean:.1f} characters (median {_stats.median(en_chars):.0f}),
                against {or_c_mean:.1f} characters (median {_stats.median(or_chars):.0f}) for the matching
                Odia sentences -- {_longer_chars} runs about {_char_ratio:.2f}&times; longer in raw character
                count for the same content. Odia's Brahmic script represents syllables with a base consonant
                plus combining vowel signs and, for consonant clusters, additional conjunct glyphs, which
                tends to inflate character counts relative to a Latin alphabet even when word counts (see
                above) do not move the same way.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if en_awl and or_awl:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Average word length (characters)</div>', unsafe_allow_html=True)
        col9, col10 = st.columns(2)
        with col9:
            st.altair_chart(_hist(en_awl, "English avg word length", TEAL), use_container_width=True)
        with col10:
            st.altair_chart(_hist(or_awl, "Odia avg word length", AMBER), use_container_width=True)

        import statistics as _stats
        en_awl_mean, or_awl_mean = _stats.mean(en_awl), _stats.mean(or_awl)
        _longer_words = "Odia" if or_awl_mean > en_awl_mean else "English"
        _awl_diff_pct = abs(or_awl_mean - en_awl_mean) / min(en_awl_mean, or_awl_mean) * 100
        st.markdown(
            f"""
            <div class="card-note">
                The average English word in this corpus is {en_awl_mean:.2f} characters long, versus
                {or_awl_mean:.2f} for Odia -- {_longer_words} words run about {_awl_diff_pct:.0f}% longer
                per word on average. Read together with the character-count and word-count charts above,
                this is where the character-level difference between the two languages actually originates:
                a shift in per-word length rather than in how many words a sentence contains, consistent
                with case markers and postpositions being fused onto word stems in Odia rather than left as
                separate tokens.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if en_unk or or_unk or en_unk_rate is not None or or_unk_rate is not None:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Tokenizer coverage (UNK rate)</div>', unsafe_allow_html=True)

        if en_unk and or_unk:
            max_unk = max(max(en_unk), max(or_unk), 1)
            col11, col12 = st.columns(2)
            with col11:
                st.altair_chart(_int_hist(en_unk, "English UNK tokens per sentence", TEAL, max_unk), use_container_width=True)
            with col12:
                st.altair_chart(_int_hist(or_unk, "Odia UNK tokens per sentence", AMBER, max_unk), use_container_width=True)

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            with st.container(border=True):
                st.metric("English UNK rate", f"{en_unk_rate:.3f}%" if isinstance(en_unk_rate, (int, float)) else "n/a")
        with mcol2:
            with st.container(border=True):
                st.metric("Odia UNK rate", f"{or_unk_rate:.3f}%" if isinstance(or_unk_rate, (int, float)) else "n/a")

        _rate_desc = []
        if isinstance(en_unk_rate, (int, float)):
            _rate_desc.append(f"English at {en_unk_rate:.3f}%")
        if isinstance(or_unk_rate, (int, float)):
            _rate_desc.append(f"Odia at {or_unk_rate:.3f}%")
        _rate_text = " and ".join(_rate_desc) if _rate_desc else "not yet computed"
        st.markdown(
            f"""
            <div class="card-note">
                Overall &lt;UNK&gt; coverage sits at {_rate_text} of tokens. A low UNK rate (well under 1%)
                means the BPE tokenizer's learned merge table already covers almost every character and
                subword pattern actually occurring in this corpus, so the model rarely has to fall back to
                an uninformative unknown-token placeholder during training or inference. A meaningfully
                higher UNK rate would instead point to vocabulary size being too small for the script's
                character inventory, or to the tokenizer having been trained on a different data
                distribution than it is now being applied to -- worth checking against the vocab sizes
                reported in the Architecture tab if this number ever climbs.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if split_comp:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Split composition (proportions)</div>', unsafe_allow_html=True)
        preferred = ["train", "val", "test"]
        ordered_split = {k: split_comp[k] for k in preferred if k in split_comp}
        ordered_split.update({k: v for k, v in split_comp.items() if k not in ordered_split})
        st.altair_chart(_pie(ordered_split, colors=[TEAL, AMBER, CRITICAL], sort_desc=False), use_container_width=True)
        _split_total = sum(split_comp.values()) or 1
        _split_breakdown = ", ".join(f"{k} {v / _split_total * 100:.1f}%" for k, v in ordered_split.items())
        st.markdown(
            f"""
            <div class="card-note">
                The {fmt_num(g(DATA_STATS, 'total_size'))} sampled pairs split as {_split_breakdown} -- a
                standard train-dominant split that leaves enough held-out data in validation and test to
                get a stable loss curve and a meaningful BLEU estimate without sacrificing training signal.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if en_end_punct or or_end_punct:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Sentence-ending punctuation</div>', unsafe_allow_html=True)
        col13, col14 = st.columns(2)
        with col13:
            st.caption("English")
            if en_end_punct:
                st.altair_chart(_pie(en_end_punct, colors=EDA_PALETTE_5), use_container_width=True)
        with col14:
            st.caption("Odia")
            if or_end_punct:
                st.altair_chart(_pie(or_end_punct, colors=EDA_PALETTE_5), use_container_width=True)

        _en_top_punct = max(en_end_punct, key=en_end_punct.get) if en_end_punct else None
        _or_top_punct = max(or_end_punct, key=or_end_punct.get) if or_end_punct else None
        _punct_bits = []
        if _en_top_punct:
            _punct_bits.append(f"English sentences most commonly end in '{_en_top_punct}'.")
        if _or_top_punct:
            _punct_bits.append(f"Odia sentences most commonly end in '{_or_top_punct}', the Odia sentence-final danda where the corpus retains native punctuation rather than a Latin period.")
        st.markdown(
            f"""
            <div class="card-note">
                {' '.join(_punct_bits)}
                A high share of "other" endings in either language usually indicates truncated, list-like,
                or otherwise noisy source sentences that survived cleaning without a terminal mark, which is
                useful context when interpreting outliers in the length distributions above.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if en_words and or_words and length_corr:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">English vs Odia sentence length (word count)</div>', unsafe_allow_html=True)
        scatter_df = pd.DataFrame({"English words": en_words, "Odia words": or_words})
        scatter_chart = (
            alt.Chart(scatter_df)
            .mark_circle(size=16, opacity=0.18, color=TEAL)
            .encode(
                x=alt.X("English words:Q", title="English word count"),
                y=alt.Y("Odia words:Q", title="Odia word count"),
                tooltip=[alt.Tooltip("English words:Q"), alt.Tooltip("Odia words:Q")],
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
            .properties(height=380, background="#FFFFFF")
        )
        st.altair_chart(scatter_chart, use_container_width=True)

        r_words = length_corr.get("pearson_r_words")
        r_subwords = length_corr.get("pearson_r_subwords")
        if isinstance(r_words, (int, float)) and isinstance(r_subwords, (int, float)):
            _r_desc = "strong" if r_words >= 0.7 else "moderate" if r_words >= 0.4 else "weak"
            st.markdown(
                f"""
                <div class="card-note">
                    Pearson correlation between English and Odia sentence length is
                    r = {r_words:.3f} in words and r = {r_subwords:.3f} in subword tokens, a {_r_desc}
                    positive relationship. This is exactly the shape a genuinely parallel, consistently
                    translated corpus should have: longer source sentences produce longer target
                    sentences and vice versa, rather than target length being decoupled from source
                    length. A weak or near-zero correlation here would instead be a red flag for
                    misaligned pairs, machine-generated filler, or truncated sentences slipping through
                    the cleaning pipeline.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if cleaning_funnel:
        _funnel_pool = cleaning_funnel.get("candidate_pool_after_cleaning")
        _funnel_survived = cleaning_funnel.get("survived_max_len_filter")
        _funnel_final = cleaning_funnel.get("final_sampled")
        if all(isinstance(v, (int, float)) for v in (_funnel_pool, _funnel_survived, _funnel_final)):
            st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Data cleaning funnel</div>', unsafe_allow_html=True)
            stages = ["Candidate pool after cleaning", "Survived MAX_LEN filter", "Final sampled"]
            funnel_df = pd.DataFrame({"stage": stages, "count": [_funnel_pool, _funnel_survived, _funnel_final]})
            funnel_chart = (
                alt.Chart(funnel_df)
                .mark_bar(color=TEAL, size=70, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("stage:N", sort=stages, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("count:Q", title="Sentence pairs"),
                    tooltip=[alt.Tooltip("stage:N", title="Stage"), alt.Tooltip("count:Q", title="Pairs", format=",")],
                )
                .configure_view(strokeWidth=0)
                .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
                .properties(height=300, background="#FFFFFF")
            )
            st.altair_chart(funnel_chart, use_container_width=True)

            _lost_len_filter = _funnel_pool - _funnel_survived
            _lost_len_pct = (_lost_len_filter / _funnel_pool * 100) if _funnel_pool else 0.0
            _kept_final_pct = (_funnel_final / _funnel_survived * 100) if _funnel_survived else 0.0
            st.markdown(
                f"""
                <div class="card-note">
                    Of the {fmt_num(_funnel_pool)} candidate pairs remaining after text cleaning,
                    {fmt_num(_lost_len_filter)} ({_lost_len_pct:.1f}%) are dropped by the
                    <code>MAX_LEN</code> subword-length filter, leaving {fmt_num(_funnel_survived)}
                    pairs, of which {fmt_num(_funnel_final)} ({_kept_final_pct:.1f}% of survivors) are
                    kept in the final sampled dataset. As the subword length-ratio analysis above shows,
                    Odia sequences run several times longer than English at the same content, so a single
                    shared <code>MAX_LEN</code> budget disproportionately filters out pairs where the Odia
                    side is too long rather than the English side -- the funnel here is effectively an
                    Odia-length filter wearing an English-length filter's name.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if descriptive_stats:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Descriptive statistics</div>', unsafe_allow_html=True)
        _row_labels = {
            "en_words": "English words",
            "or_words": "Odia words",
            "en_subwords": "English subwords",
            "or_subwords": "Odia subwords",
            "subword_ratio": "Subword ratio (Odia / English)",
            "en_chars": "English characters",
            "or_chars": "Odia characters",
        }
        _col_order = ["mean", "std", "min", "p25", "median", "p75", "p90", "p95", "p99", "max"]
        _desc_rows = {
            label: {col: descriptive_stats[key].get(col) for col in _col_order}
            for key, label in _row_labels.items()
            if key in descriptive_stats
        }
        if _desc_rows:
            desc_df = pd.DataFrame(_desc_rows).T[_col_order].round(2)
            st.dataframe(desc_df, use_container_width=True)

            st.markdown(
                """
                <div class="card-note">
                    Percentiles rather than mean/std alone matter here because sentence length is
                    right-skewed: the gap between the median and the p99/max columns shows how long the
                    tail of unusually long sentences runs for each measure, which is what actually drives
                    truncation risk under a fixed <code>MAX_LEN</code> rather than the mean length.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if vocab_stats:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Vocabulary richness</div>', unsafe_allow_html=True)
        vocab_df = pd.DataFrame(
            {
                "Total words": [vocab_stats.get("en_total_words"), vocab_stats.get("or_total_words")],
                "Unique words": [vocab_stats.get("en_unique_words"), vocab_stats.get("or_unique_words")],
                "Type-token ratio": [vocab_stats.get("en_ttr"), vocab_stats.get("or_ttr")],
            },
            index=["English", "Odia"],
        )
        if pd.api.types.is_numeric_dtype(vocab_df["Type-token ratio"]):
            vocab_df["Type-token ratio"] = vocab_df["Type-token ratio"].round(4)
        st.dataframe(vocab_df, use_container_width=True)

        en_ttr = vocab_stats.get("en_ttr")
        or_ttr = vocab_stats.get("or_ttr")
        if isinstance(en_ttr, (int, float)) and isinstance(or_ttr, (int, float)):
            _higher_lang = "Odia" if or_ttr > en_ttr else "English"
            if _higher_lang == "Odia":
                _morph_note = (
                    "Odia is agglutinative and inflects nouns and verbs for case, number, and honorific "
                    "register by adding suffixes to a shared stem, so the same underlying vocabulary "
                    "surfaces as many more distinct word forms -- exactly what a higher type-token ratio "
                    "measures."
                )
            else:
                _morph_note = (
                    "even though Odia's morphology is more inflected than English's, the raw word-form "
                    "count here comes out lower relative to English, which given the corpus size is more "
                    "consistent with a smaller effective vocabulary in this particular sample than with "
                    "morphology alone."
                )
            st.markdown(
                f"""
                <div class="card-note">
                    Type-token ratio (unique words / total words) is {en_ttr:.4f} for English and
                    {or_ttr:.4f} for Odia -- {_higher_lang} shows the higher lexical diversity of the two
                    on this corpus. {_morph_note}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if top_en_full and top_or_full:
        st.markdown('<div class="section-heading" style="margin-top:1.4rem;">Top 30 words, full detail</div>', unsafe_allow_html=True)
        col15, col16 = st.columns(2)
        with col15:
            st.caption("English (common stopwords removed)")
            en_full_df = pd.DataFrame(top_en_full, columns=["word", "count", "percentage"])
            en_full_df["percentage"] = en_full_df["percentage"].round(2)
            st.dataframe(en_full_df, use_container_width=True, hide_index=True)
        with col16:
            st.caption("Odia")
            or_full_df = pd.DataFrame(top_or_full, columns=["word", "count", "percentage"])
            or_full_df["percentage"] = or_full_df["percentage"].round(2)
            st.dataframe(or_full_df, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="card-note">
                Extending the ranking from the top 20 shown in the charts above to the top 30 words in
                each language: the English list is topped by "{top_en_full[0][0]}" at
                {top_en_full[0][2]:.2f}% of all (non-stopword) word occurrences, and the Odia list by
                "{top_or_full[0][0]}" at {top_or_full[0][2]:.2f}%. A steep drop-off from rank 1 to rank
                30 is the expected Zipfian pattern for natural-language word frequency; a flat,
                near-uniform distribution across the top 30 instead would be a sign of a corpus dominated
                by boilerplate or templated sentences rather than natural text variety.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_training:
    st.markdown('<div class="section-heading">Loss curves</div>', unsafe_allow_html=True)

    if TRAINING_HISTORY:
        hist_df = pd.DataFrame(TRAINING_HISTORY)
        long_df = hist_df.melt(id_vars="epoch", value_vars=["train_loss", "val_loss"], var_name="split", value_name="loss")
        split_label = {"train_loss": "Train loss", "val_loss": "Validation loss"}
        long_df["split"] = long_df["split"].map(split_label)

        loss_chart = (
            alt.Chart(long_df)
            .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=45, filled=True))
            .encode(
                x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("loss:Q", title="Loss"),
                color=alt.Color(
                    "split:N",
                    sort=["Train loss", "Validation loss"],
                    scale=alt.Scale(domain=["Train loss", "Validation loss"], range=[TEAL, AMBER]),
                    legend=alt.Legend(title="", orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("epoch:Q", title="Epoch"),
                    alt.Tooltip("split:N", title="Split"),
                    alt.Tooltip("loss:Q", title="Loss", format=".4f"),
                ],
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
            .configure_legend(labelColor=TEXT_MAIN)
            .properties(height=380, background="#FFFFFF")
        )
        st.altair_chart(loss_chart, use_container_width=True)
    else:
        st.info("No training history available.")

    _n_epochs = len(TRAINING_HISTORY) if TRAINING_HISTORY else 0
    st.markdown(
        f"""
        <div class="card-note">
            Train and validation loss decrease smoothly together across all {_n_epochs}
            epochs with no divergence between them. This is the expected shape for
            a correctly masked decoder: if validation loss had dropped suspiciously
            below training loss, it would suggest the causal mask was leaking
            future tokens into the decoder's self-attention during training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading" style="margin-top:1.6rem;">Causal-mask verification</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spec-quote">
            "watch for the causal mask bug (if val loss is suspiciously perfect, your decoder is
            peeking!)"
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="card-note">
            A smooth loss curve is only circumstantial evidence, so the masking is verified
            mechanically instead of by eye. With dropout disabled, the decoder is run twice on target
            sequences that are identical up to a cut position and different after it; the output
            logits at every position up to the cut are asserted to be numerically identical. If any
            future token could influence an earlier prediction, that assertion fails.
            <br/><br/>
            A test that passes is only meaningful if it can also fail, so a negative control runs the
            same check against a deliberately broken, non-causal mask and confirms it does fail --
            proving the test has real diagnostic power rather than passing vacuously. Three further
            tests assert the mask tensors directly: future positions always blocked, padding always
            blocked, and valid past positions never blocked.
        </div>
        """,
        unsafe_allow_html=True,
    )

    mask_tests = [
        ("Leak invariance", "Outputs before the cut position are unchanged by future target tokens.", "tests/test_masks.py"),
        ("Negative control", "The same check fails against a deliberately non-causal mask, proving the test has teeth.", "tests/test_masks.py"),
        ("Decoder mask coverage", "Future and padding positions blocked; valid past positions allowed.", "tests/test_masks.py"),
        ("Cross-attention mask", "Source padding blocked for every decoder query position.", "tests/test_masks.py"),
    ]
    mt_html = ["<table class='result-table'><thead><tr><th style='width:22%'>Check</th><th style='width:11%'>Result</th><th>What it proves</th></tr></thead><tbody>"]
    for name, proves, where in mask_tests:
        mt_html.append(
            "<tr>"
            f"<td><strong>{name}</strong></td>"
            "<td><span class='status-badge status-met'>Passing</span></td>"
            f"<td>{proves}<br/><span class='evidence-where'>{where}</span></td>"
            "</tr>"
        )
    mt_html.append("</tbody></table>")
    st.markdown("".join(mt_html), unsafe_allow_html=True)

with tab_results:
    st.markdown('<div class="section-heading">Evaluation</div>', unsafe_allow_html=True)

    bleu = EVAL_RESULTS.get("bleu_score")
    signature = EVAL_RESULTS.get("bleu_signature", "n/a")
    num_examples = EVAL_RESULTS.get("num_test_examples", "n/a")

    r1, r2 = st.columns([1, 2])
    with r1:
        with st.container(border=True):
            st.metric("BLEU score", f"{bleu:.2f}" if isinstance(bleu, (int, float)) else "n/a")
            st.markdown(f'<div class="bleu-signature">{signature}</div>', unsafe_allow_html=True)
    with r2:
        with st.container(border=True):
            st.metric("Test examples evaluated", fmt_num(num_examples))
            decode_seconds = EVAL_RESULTS.get("decode_seconds")
            if isinstance(decode_seconds, (int, float)):
                st.caption(f"Greedy decoding over the test set took {decode_seconds:,.1f} seconds.")

    st.markdown('<div class="section-heading" style="margin-top:1.6rem;">Sample translations</div>', unsafe_allow_html=True)

    samples = EVAL_RESULTS.get("samples", [])
    long_idx = EVAL_RESULTS.get("long_sentence_index")

    if samples:
        rows_html = ["<table class='result-table'><thead><tr><th>Source (English)</th><th>Reference (Odia)</th><th>Hypothesis (Odia)</th></tr></thead><tbody>"]
        for i, sample in enumerate(samples):
            is_long = (i == long_idx)
            row_class = "long-row" if is_long else ""
            badge = '<span class="long-badge">Long sentence example</span><br/>' if is_long else ""
            source = sample.get("source", "")
            reference = sample.get("reference", "")
            hypothesis = sample.get("hypothesis", "")
            rows_html.append(
                f"<tr class='{row_class}'>"
                f"<td>{badge}{source}</td>"
                f"<td class='odia-text'>{reference}</td>"
                f"<td class='odia-text'>{hypothesis}</td>"
                f"</tr>"
            )
        rows_html.append("</tbody></table>")
        st.markdown("".join(rows_html), unsafe_allow_html=True)

        if long_idx is not None:
            st.caption(
                "The highlighted row is the longest source sentence in the sample set. "
                "Its hypothesis shows degenerate repetition of syllables and stems, a known "
                "failure mode for small greedy-decoded Transformers as output length grows."
            )
    else:
        st.info("No sample translations available.")

    if LENGTH_QUALITY_RESULTS:
        st.markdown('<div class="section-heading" style="margin-top:1.6rem;">Does quality actually degrade with length?</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-note">
                The single long-sentence example above is anecdotal. To check whether it reflects a
                real pattern, every one of the 2,000 test-set sentences was decoded and scored with
                per-sentence BLEU, then grouped by source sentence length.
            </div>
            """,
            unsafe_allow_html=True,
        )

        lq_overall_bleu = LENGTH_QUALITY_RESULTS.get("overall_mean_sentence_bleu")
        lq_rep_rate = LENGTH_QUALITY_RESULTS.get("overall_repetition_rate_pct")
        r_word = LENGTH_QUALITY_RESULTS.get("pearson_r_wordlen_bleu")
        r_subword = LENGTH_QUALITY_RESULTS.get("pearson_r_subwordlen_bleu")

        lq1, lq2, lq3 = st.columns(3)
        with lq1:
            with st.container(border=True):
                st.metric("Mean per-sentence BLEU", f"{lq_overall_bleu:.2f}" if isinstance(lq_overall_bleu, (int, float)) else "n/a")
        with lq2:
            with st.container(border=True):
                st.metric("Repetition rate (full test set)", f"{lq_rep_rate:.1f}%" if isinstance(lq_rep_rate, (int, float)) else "n/a")
        with lq3:
            with st.container(border=True):
                st.metric("Correlation: length vs. BLEU", f"{r_word:.2f}" if isinstance(r_word, (int, float)) else "n/a")

        buckets = LENGTH_QUALITY_RESULTS.get("buckets_by_word_len", [])
        if buckets:
            bucket_df = pd.DataFrame(buckets)
            bleu_bar = (
                alt.Chart(bucket_df)
                .mark_bar(color=TEAL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("bucket:N", sort=[b["bucket"] for b in buckets], title="Source length (words)"),
                    y=alt.Y("mean_bleu:Q", title="Mean per-sentence BLEU"),
                    tooltip=["bucket", "count", "mean_bleu"],
                )
                .configure_view(strokeWidth=0)
                .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
                .properties(height=280, background="#FFFFFF", title="Mean BLEU by source sentence length")
            )
            rep_bar = (
                alt.Chart(bucket_df)
                .mark_bar(color=CRITICAL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("bucket:N", sort=[b["bucket"] for b in buckets], title="Source length (words)"),
                    y=alt.Y("repetition_rate_pct:Q", title="Repetition rate (%)"),
                    tooltip=["bucket", "count", "repetition_rate_pct"],
                )
                .configure_view(strokeWidth=0)
                .configure_axis(gridColor="#EDF2F1", domainColor="#CBD5E1", labelColor=TEXT_MUTED, titleColor=TEXT_MAIN)
                .properties(height=280, background="#FFFFFF", title="Degenerate-repetition rate by source sentence length")
            )
            lqc1, lqc2 = st.columns(2)
            with lqc1:
                st.altair_chart(bleu_bar, use_container_width=True)
            with lqc2:
                st.altair_chart(rep_bar, use_container_width=True)

        shortest = buckets[0] if buckets else {}
        longest = buckets[-1] if buckets else {}
        st.markdown(
            f"""
            <div class="card-note">
                The pattern holds across all 2,000 test sentences, not just the one example above:
                mean BLEU falls from {shortest.get('mean_bleu', 'n/a')} on the shortest sentences
                ({shortest.get('bucket', 'n/a')} words) to {longest.get('mean_bleu', 'n/a')} on the
                longest ({longest.get('bucket', 'n/a')} words), while the degenerate-repetition rate
                climbs from {shortest.get('repetition_rate_pct', 'n/a')}% to
                {longest.get('repetition_rate_pct', 'n/a')}% over the same range. The length-BLEU
                correlation itself is only moderate (r = {r_word if r_word is not None else 'n/a'}
                for word length, {r_subword if r_subword is not None else 'n/a'} for subword length)
                because the relationship is not linear -- most of the degradation happens by
                roughly 12-15 words, after which quality is already poor. The repetition-rate trend
                is the sharper signal: it points at degenerate repetition specifically, not just
                generic quality loss, as the dominant failure mode on longer sentences.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_translate:
    st.markdown('<div class="section-heading">Translate a sentence</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-note">
            Type an English sentence below and the actual trained checkpoint from
            this project translates it into Odia using greedy decoding, live.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        translate_model, en_tokenizer, or_tokenizer = load_translation_model()
        model_load_error = None
    except Exception as exc:
        translate_model, en_tokenizer, or_tokenizer = None, None, None
        model_load_error = str(exc)

    if model_load_error:
        st.warning(f"Translation model could not be loaded: {model_load_error}")
    else:
        user_text = st.text_input(
            "English sentence",
            value="The weather is very nice today.",
            key="live_translate_input",
        )
        show_beam = st.checkbox("Also show beam search output", value=False)
        show_attention = st.checkbox("Show attention heatmap", value=False)

        if user_text.strip():
            import torch

            from src.inference.greedy_decode import greedy_decode
            from src.tokenization.tokenizer_utils import encode as tok_encode, decode as tok_decode

            source_ids = tok_encode(en_tokenizer, user_text)
            src_tensor = torch.tensor([source_ids])

            with st.spinner("Translating"):
                greedy_ids = greedy_decode(translate_model, src_tensor)
            greedy_text = tok_decode(or_tokenizer, greedy_ids)

            st.markdown(
                f'<div class="card-note odia-text" style="font-size:1.3rem;">{greedy_text}</div>',
                unsafe_allow_html=True,
            )

            if show_beam:
                from src.inference.beam_search import beam_search_decode

                with st.spinner("Running beam search"):
                    beam_ids = beam_search_decode(translate_model, src_tensor)
                beam_text = tok_decode(or_tokenizer, beam_ids)
                st.caption("Beam search (width 4):")
                st.markdown(f'<div class="odia-text">{beam_text}</div>', unsafe_allow_html=True)

            st.caption(
                f"Encoded as {len(source_ids)} English subword tokens, including SOS/EOS."
            )

            if show_attention:
                from src.inference.attention_extraction import translate_with_attention

                with st.spinner("Extracting attention weights"):
                    attn_result = translate_with_attention(translate_model, src_tensor)

                attn_matrix = attn_result["attention_matrix"]
                generated_ids = attn_result["generated_ids"]
                # byte-level BPE's raw .tokens()/id_to_token() strings are not
                # human-readable (they use a byte<->unicode remapping); decoding
                # one id at a time through the tokenizer's own decoder recovers
                # the real text each subword piece represents
                src_tokens = [
                    en_tokenizer.decode([i], skip_special_tokens=False) or "?"
                    for i in source_ids
                ]
                hyp_tokens = [
                    or_tokenizer.decode([i], skip_special_tokens=False) or "?"
                    for i in generated_ids[1:]
                ]

                heat_rows = []
                for hi, hyp_tok in enumerate(hyp_tokens):
                    if hi >= len(attn_matrix):
                        break
                    for si, src_tok in enumerate(src_tokens):
                        if si >= len(attn_matrix[hi]):
                            break
                        heat_rows.append({
                            "hyp_label": f"{hi}: {hyp_tok}",
                            "src_label": f"{si}: {src_tok}",
                            "hyp_order": hi,
                            "src_order": si,
                            "weight": attn_matrix[hi][si],
                        })

                if heat_rows:
                    heat_df = pd.DataFrame(heat_rows)
                    heatmap = (
                        alt.Chart(heat_df)
                        .mark_rect()
                        .encode(
                            x=alt.X("src_label:N", sort=alt.SortField("src_order"), title="English source tokens", axis=alt.Axis(labelAngle=-40)),
                            y=alt.Y("hyp_label:N", sort=alt.SortField("hyp_order"), title="Odia generated tokens", axis=alt.Axis(labelFont="Noto Sans Oriya")),
                            color=alt.Color("weight:Q", scale=alt.Scale(scheme="teals"), title="Attention weight"),
                            tooltip=["src_label", "hyp_label", alt.Tooltip("weight:Q", format=".3f")],
                        )
                        .configure_view(strokeWidth=0)
                        .properties(height=max(220, 26 * len(hyp_tokens)), background="#FFFFFF")
                    )
                    st.altair_chart(heatmap, use_container_width=True)
                    st.caption(
                        "Each row is one generated Odia subword token; each column is one English "
                        "source subword token. Darker cells mean the decoder attended more strongly "
                        "to that source token while generating that output token."
                    )
        else:
            st.info("Type a sentence above to see its Odia translation.")

        st.markdown(
            """
            <div class="card-note" style="margin-top:1.4rem;">
                This model is deliberately small and trained on a modest corpus, so
                translation quality is limited, especially on longer or more complex
                sentences -- see the Results tab for the measured BLEU score and a
                worked example of the failure mode.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="footer-note">English-Odia Transformer &middot; trained from scratch per assignment section 5.6</div>', unsafe_allow_html=True)
