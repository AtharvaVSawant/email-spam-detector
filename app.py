"""
Email Spam Detector — Streamlit App
Bidirectional LSTM · NLP · Interactive inference + model metrics
"""

import os
import sys
import time
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:         #0d0f14;
    --surface:    #161920;
    --border:     #252830;
    --accent:     #6c63ff;
    --accent-alt: #ff4d6d;
    --success:    #23d18b;
    --warning:    #f0a500;
    --text:       #e8eaf0;
    --muted:      #7a7f9a;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem !important;
}

/* ── Inputs ── */
textarea, .stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Result box ── */
.result-spam {
    background: rgba(255, 77, 109, 0.12);
    border: 1px solid var(--accent-alt);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-ham {
    background: rgba(35, 209, 139, 0.10);
    border: 1px solid var(--success);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-label {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-sub {
    color: var(--muted);
    font-size: 0.95rem;
}
.tag {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 0.6rem;
}
.tag-spam { background: rgba(255,77,109,0.25); color: #ff4d6d; }
.tag-ham  { background: rgba(35,209,139,0.20); color: #23d18b; }

/* ── Example pills ── */
.example-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model…")
def load_pipeline():
    """Load prediction pipeline once and cache it."""
    from src.pipeline.predict_pipeline import PredictPipeline
    return PredictPipeline()


def gauge_chart(prob: float, is_spam: bool) -> go.Figure:
    color = "#ff4d6d" if is_spam else "#23d18b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"color": color, "size": 40, "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#7a7f9a"}, "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161920",
            "bordercolor": "#252830",
            "steps": [
                {"range": [0, 30],  "color": "rgba(35,209,139,0.12)"},
                {"range": [30, 60], "color": "rgba(240,165,0,0.10)"},
                {"range": [60, 100],"color": "rgba(255,77,109,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8, "value": prob * 100
            },
        },
        title={"text": "Spam Probability", "font": {"color": "#7a7f9a", "size": 14}},
    ))
    fig.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font_color="#e8eaf0", height=260, margin=dict(t=40, b=10, l=20, r=20)
    )
    return fig


def bar_chart(spam_prob: float) -> go.Figure:
    ham_prob = 1 - spam_prob
    fig = go.Figure(go.Bar(
        x=["Ham (Safe)", "Spam"],
        y=[round(ham_prob * 100, 2), round(spam_prob * 100, 2)],
        marker_color=["#23d18b", "#ff4d6d"],
        text=[f"{ham_prob*100:.1f}%", f"{spam_prob*100:.1f}%"],
        textposition="auto",
        textfont={"color": "#0d0f14", "family": "Space Grotesk", "size": 14},
    ))
    fig.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#161920",
        font_color="#e8eaf0", height=220,
        xaxis=dict(showgrid=False, color="#7a7f9a"),
        yaxis=dict(showgrid=True, gridcolor="#252830", color="#7a7f9a", range=[0, 110], ticksuffix="%"),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


def training_history_chart(history: dict) -> go.Figure:
    epochs = list(range(1, len(history["train_acc"]) + 1))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Accuracy", "Loss"),
                        horizontal_spacing=0.12)
    # Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train Acc",
                             line=dict(color="#6c63ff", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val Acc",
                             line=dict(color="#23d18b", width=2, dash="dot")), row=1, col=1)
    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss",
                             line=dict(color="#f0a500", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss",
                             line=dict(color="#ff4d6d", width=2, dash="dot")), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="#0d0f14", plot_bgcolor="#161920",
        font=dict(color="#e8eaf0", family="Space Grotesk"),
        legend=dict(bgcolor="#161920", bordercolor="#252830"),
        height=300, margin=dict(t=30, b=10, l=10, r=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#252830", color="#7a7f9a", title_text="Epoch")
    fig.update_yaxes(showgrid=True, gridcolor="#252830", color="#7a7f9a")
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SpamShield AI")
    st.markdown("*Bidirectional LSTM · NLP pipeline*")
    st.divider()

    page = st.radio("Navigate", ["🔍 Detector", "📊 Model Metrics", "🏋️ Train Model"],
                    label_visibility="collapsed")
    st.divider()

    st.markdown("**Architecture**")
    st.markdown("""
    - Embedding (128-dim)
    - BiLSTM × 2
    - Dense + Dropout
    - Sigmoid output
    """)
    st.markdown("**Dataset**")
    st.markdown("SMS Spam Collection  \n~5,500 messages")

    st.divider()
    st.markdown(
        "<span style='color:#7a7f9a;font-size:0.8rem'>Built with TensorFlow + Streamlit</span>",
        unsafe_allow_html=True
    )


# ── Page: Detector ────────────────────────────────────────────────────────────
if "🔍 Detector" in page:
    st.markdown("# 🛡️ SpamShield AI")
    st.markdown("<p style='color:#7a7f9a'>Paste any email or SMS below — the LSTM model will classify it in milliseconds.</p>", unsafe_allow_html=True)
    st.divider()

    # Example messages
    EXAMPLES = {
        "🎉 Prize scam":      "Congratulations! You've won a $1,000 Walmart gift card. Click http://bit.ly/claim-now to claim your prize before it expires!",
        "💊 Pharma spam":     "Get Viag*a and other meds at lowest prices! No prescription needed. Order now at cheapmeds247.net",
        "📧 Legit email":     "Hi Sarah, just following up on yesterday's meeting. I've attached the revised proposal for your review. Let me know if you have any questions!",
        "👨‍💼 Work message":   "The standup is rescheduled to 3 PM today. Please update your tickets before joining.",
        "💰 Money scam":      "URGENT: Your bank account has been suspended. Verify immediately at secure-banking-alert.com or lose access permanently.",
    }

    col_ex = st.columns(len(EXAMPLES))
    selected_example = None
    for i, (label, msg) in enumerate(EXAMPLES.items()):
        if col_ex[i].button(label, use_container_width=True):
            selected_example = msg

    st.markdown("")

    default_text = selected_example if selected_example else ""
    user_input = st.text_area(
        "Email / SMS content",
        value=default_text,
        height=160,
        placeholder="Paste email content here…",
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 5])
    run = col_btn.button("🔍 Analyse", use_container_width=True)

    if run:
        if not user_input.strip():
            st.warning("Please enter some text to analyse.")
        else:
            try:
                pipeline = load_pipeline()
                with st.spinner("Running inference…"):
                    t0 = time.perf_counter()
                    result = pipeline.predict(user_input)
                    elapsed = (time.perf_counter() - t0) * 1000

                st.divider()

                is_spam = result["is_spam"]
                prob    = result["spam_probability"]
                conf    = result["confidence"]

                # Result card
                box_class  = "result-spam" if is_spam else "result-ham"
                emoji      = "🚨" if is_spam else "✅"
                label_text = "SPAM DETECTED" if is_spam else "LOOKS SAFE"
                tag_class  = "tag-spam" if is_spam else "tag-ham"
                tag_text   = "HIGH RISK" if prob > 0.8 else ("MODERATE RISK" if prob > 0.5 else "LOW RISK")

                st.markdown(f"""
                <div class="{box_class}">
                    <div class="result-label">{emoji} {label_text}</div>
                    <div class="result-sub">Inference time: {elapsed:.1f} ms &nbsp;·&nbsp; Confidence: {conf*100:.1f}%</div>
                    <span class="tag {tag_class}">{tag_text}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")
                col_g, col_b = st.columns(2)
                col_g.plotly_chart(gauge_chart(prob, is_spam), use_container_width=True)
                col_b.plotly_chart(bar_chart(prob), use_container_width=True)

                # Stats row
                c1, c2, c3 = st.columns(3)
                c1.metric("Spam Probability",  f"{prob*100:.1f}%")
                c2.metric("Ham Probability",   f"{result['ham_probability']*100:.1f}%")
                c3.metric("Model Confidence",  f"{conf*100:.1f}%")

            except FileNotFoundError:
                st.error(
                    "⚠️ Model not found. Head to **🏋️ Train Model** to train the LSTM first."
                )
            except Exception as e:
                st.error(f"Inference error: {e}")


# ── Page: Metrics ─────────────────────────────────────────────────────────────
elif "📊 Model Metrics" in page:
    st.markdown("# 📊 Model Metrics")
    st.markdown("<p style='color:#7a7f9a'>Training results from the last run.</p>", unsafe_allow_html=True)
    st.divider()

    metrics_path = os.path.join("artifacts", "metrics.json")
    if not os.path.exists(metrics_path):
        st.info("No metrics found. Train the model first via **🏋️ Train Model**.")
    else:
        with open(metrics_path) as f:
            m = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",        f"{m['accuracy']*100:.2f}%")
        c2.metric("ROC-AUC",         f"{m['roc_auc']:.4f}")
        c3.metric("Precision (spam)",f"{m['precision_spam']*100:.2f}%")
        c4.metric("Recall (spam)",   f"{m['recall_spam']*100:.2f}%")

        st.divider()
        st.markdown("### 📈 Training History")
        if "history" in m:
            st.plotly_chart(training_history_chart(m["history"]), use_container_width=True)

        st.divider()
        st.markdown("### 🔢 Confusion Matrix")
        if "confusion_matrix" in m:
            cm = np.array(m["confusion_matrix"])
            labels = ["Ham", "Spam"]
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Purples",
                x=labels, y=labels,
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig_cm.update_layout(
                paper_bgcolor="#0d0f14", plot_bgcolor="#161920",
                font=dict(color="#e8eaf0", family="Space Grotesk"),
                height=350, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)


# ── Page: Train ───────────────────────────────────────────────────────────────
elif "🏋️ Train Model" in page:
    st.markdown("# 🏋️ Train Model")
    st.markdown(
        "<p style='color:#7a7f9a'>This will run the full ML pipeline: ingest → transform → train → evaluate.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    uploaded = st.file_uploader(
        "Upload custom dataset (CSV with `label` and `message` columns, optional)",
        type=["csv"],
        help="Leave empty to auto-download the SMS Spam Collection dataset."
    )

    custom_path = None
    if uploaded:
        save_dir = "artifacts"
        os.makedirs(save_dir, exist_ok=True)
        custom_path = os.path.join(save_dir, "uploaded_data.csv")
        with open(custom_path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Dataset uploaded: {uploaded.name}")

    if st.button("🚀 Start Training"):
        from src.pipeline.train_pipeline import TrainPipeline
        progress = st.progress(0, text="Initialising…")
        log_box  = st.empty()

        try:
            with st.spinner("Training in progress — this may take a few minutes…"):
                progress.progress(10, text="Ingesting data…")
                pipeline_obj = TrainPipeline()

                progress.progress(30, text="Transforming text…")
                train_path, test_path = pipeline_obj.ingestion.initiate_data_ingestion(custom_path)

                X_train, X_test, y_train, y_test = pipeline_obj.transformation.fit_transform(
                    train_path, test_path
                )

                progress.progress(60, text="Training LSTM…")
                model, metrics = pipeline_obj.trainer.train(X_train, X_test, y_train, y_test)
                progress.progress(100, text="Done!")

            st.success("✅ Training complete! Navigate to **📊 Model Metrics** to view results.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            c2.metric("ROC-AUC",  f"{metrics['roc_auc']:.4f}")
            c3.metric("F1 Spam",  f"{metrics['f1_spam']:.4f}")

            # Invalidate cached model so next inference reloads
            st.cache_resource.clear()

        except Exception as e:
            st.error(f"Training failed: {e}")
            progress.empty()
