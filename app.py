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
    )

    DATA_LOAD_ERROR = None
except Exception as exc:  # project_data.py not present or not importable yet
    HYPERPARAMS, DATA_STATS, TOKENIZER_STATS = {}, {}, {}
    TRAINING_HISTORY, EVAL_RESULTS = [], {}
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

tab_overview, tab_arch, tab_data, tab_training, tab_results = st.tabs(
    ["Overview", "Architecture", "Data and Tokenization", "Training", "Results"]
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

    st.markdown(
        """
        <div class="card-note">
            The model is a compact 2-encoder / 2-decoder Transformer trained for 18
            epochs on a Kaggle GPU session. It scores a low but non-degenerate BLEU,
            reflecting the constraints of training a translation model from scratch
            on a small subset and a small vocabulary rather than fine-tuning a
            pretrained multilingual model.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown(
        """
        <div class="card-note">
            Train and validation loss decrease smoothly together across all 18
            epochs with no divergence between them. This is the expected shape for
            a correctly masked decoder: if validation loss had dropped suspiciously
            below training loss, it would suggest the causal mask was leaking
            future tokens into the decoder's self-attention during training.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

st.markdown('<div class="footer-note">English-Odia Transformer &middot; trained from scratch per assignment section 5.6</div>', unsafe_allow_html=True)
