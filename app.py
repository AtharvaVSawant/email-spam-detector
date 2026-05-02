import streamlit as st
import sys
import os
import time
import random

sys.path.append(os.path.abspath("."))

# ── Uncomment when your pipeline is ready ──────────────────────────────────
# from src.pipeline.predict_pipeline import PredictPipeline

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── SESSION STATE ──────────────────────────────────────────────────────────
if "history"   not in st.session_state: st.session_state.history   = []
if "result"    not in st.session_state: st.session_state.result    = None
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

# ── THEME TOKENS ───────────────────────────────────────────────────────────
# Every colour lives here — CSS injects via f-string so zero hardcoded hex
# in any HTML snippet below.
if st.session_state.dark_mode:
    T = dict(
        # canvas
        page_bg         = "#0d0d0f",
        container_bg    = "#0d0d0f",
        surface         = "#1c1c21",
        surface_hover   = "#25252c",
        border          = "#2e2e38",
        # text — high contrast on dark background
        text_primary    = "#f0f0f5",
        text_secondary  = "#9090a8",
        text_muted      = "#55556a",
        # red accent (same in both modes — stands out either way)
        accent          = "#e24b4a",
        accent_hover    = "#ff6160",
        # badge
        badge_bg        = "#1c1c21",
        badge_border    = "#2e2e38",
        badge_text      = "#9090a8",
        # textarea
        input_bg        = "#1c1c21",
        input_border    = "#2e2e38",
        input_text      = "#f0f0f5",
        input_ph        = "#55556a",
        # ── SPAM result — dark mode ──
        # bg is very dark red, so labels must be LIGHT red for contrast
        spam_bg         = "#1a0a0a",
        spam_border     = "#7f1d1d",
        spam_icon_bg    = "#3b0e0e",
        spam_label      = "#fca5a5",   # light red  ✓ contrast on dark bg
        spam_sub        = "#f87171",
        spam_bar_track  = "#3b0e0e",
        spam_bar_fill   = "#ef4444",
        spam_bar_val    = "#fca5a5",
        spam_pill_bg    = "#3b0e0e",
        spam_pill_text  = "#fca5a5",
        # ── HAM result — dark mode ──
        # bg is very dark green, so labels must be LIGHT green
        ham_bg          = "#091a0e",
        ham_border      = "#166534",
        ham_icon_bg     = "#052e16",
        ham_label       = "#86efac",   # light green ✓ contrast on dark bg
        ham_sub         = "#4ade80",
        ham_bar_track   = "#052e16",
        ham_bar_fill    = "#22c55e",
        ham_bar_val     = "#86efac",
        ham_pill_bg     = "#052e16",
        ham_pill_text   = "#86efac",
        # history
        hist_row_bg     = "#1c1c21",
        hist_row_border = "#2e2e38",
        hist_text       = "#9090a8",
        hist_conf       = "#55556a",
        # misc
        divider         = "#2e2e38",
        footer_text     = "#3a3a4a",
        charcount_col   = "#55556a",
        # toggle
        toggle_icon     = "☀️",
        toggle_label    = "Light mode",
    )
else:
    T = dict(
        page_bg         = "#ffffff",
        container_bg    = "#ffffff",
        surface         = "#f9fafb",
        surface_hover   = "#f3f4f6",
        border          = "#e5e7eb",
        text_primary    = "#111827",
        text_secondary  = "#6b7280",
        text_muted      = "#9ca3af",
        accent          = "#e24b4a",
        accent_hover    = "#c23a39",
        badge_bg        = "#ffffff",
        badge_border    = "#e5e7eb",
        badge_text      = "#6b7280",
        input_bg        = "#ffffff",
        input_border    = "#e5e7eb",
        input_text      = "#111827",
        input_ph        = "#9ca3af",
        # ── SPAM result — light mode ──
        # bg is light pink, so labels must be DARK red for contrast
        spam_bg         = "#fff1f1",
        spam_border     = "#fca5a5",
        spam_icon_bg    = "#fecaca",
        spam_label      = "#991b1b",   # dark red   ✓ contrast on light bg
        spam_sub        = "#7f1d1d",
        spam_bar_track  = "#fecaca",
        spam_bar_fill   = "#e24b4a",
        spam_bar_val    = "#7f1d1d",
        spam_pill_bg    = "#fecaca",
        spam_pill_text  = "#7f1d1d",
        # ── HAM result — light mode ──
        # bg is light green, so labels must be DARK green
        ham_bg          = "#f0fdf4",
        ham_border      = "#86efac",
        ham_icon_bg     = "#bbf7d0",
        ham_label       = "#166534",   # dark green  ✓ contrast on light bg
        ham_sub         = "#14532d",
        ham_bar_track   = "#bbf7d0",
        ham_bar_fill    = "#16a34a",
        ham_bar_val     = "#14532d",
        ham_pill_bg     = "#bbf7d0",
        ham_pill_text   = "#14532d",
        hist_row_bg     = "#f9fafb",
        hist_row_border = "#e5e7eb",
        hist_text       = "#6b7280",
        hist_conf       = "#9ca3af",
        divider         = "#e5e7eb",
        footer_text     = "#d1d5db",
        charcount_col   = "#9ca3af",
        toggle_icon     = "🌙",
        toggle_label    = "Dark mode",
    )

# ── GLOBAL CSS (all colours injected from T dict) ──────────────────────────
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif !important;
    background-color: {T['page_bg']} !important;
}}
.stApp {{ background-color: {T['page_bg']} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 680px !important;
    background: {T['container_bg']} !important;
}}

/* native streamlit text */
p, div, span, label, .stMarkdown {{
    color: {T['text_primary']} !important;
}}

/* ── Badge ────────────────────────────────────────────────── */
.sd-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {T['badge_text']};
    background: {T['badge_bg']};
    border: 0.5px solid {T['badge_border']};
    padding: 4px 13px;
    border-radius: 100px;
    margin-bottom: 1.3rem;
}}
.sd-badge .dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e;
    animation: blink 2s infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1;}} 50%{{opacity:.25;}} }}

/* ── Headline ─────────────────────────────────────────────── */
.sd-title {{
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 800;
    line-height: 1.05;
    color: {T['text_primary']};
    margin-bottom: .4rem;
}}
.sd-title .acc {{ color: {T['accent']}; }}
.sd-sub {{ font-size:15px; color:{T['text_secondary']}; margin-bottom:2rem; }}

/* ── All stButtons ───────────────────────────────────────── */
div.stButton > button {{
    width: 100%;
    background: {T['surface']} !important;
    color: {T['text_secondary']} !important;
    border: 0.5px solid {T['border']} !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 0 !important;
    transition: all .18s !important;
}}
div.stButton > button:hover {{
    background: {T['surface_hover']} !important;
    border-color: {T['accent']} !important;
    color: {T['accent']} !important;
}}

/* Analyze button wrapper overrides to red */
.analyze-btn div.stButton > button {{
    background: {T['accent']} !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    padding: 13px 0 !important;
}}
.analyze-btn div.stButton > button:hover {{
    background: {T['accent_hover']} !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}}

/* ── Textarea ─────────────────────────────────────────────── */
textarea {{
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 12px !important;
    border: 1px solid {T['input_border']} !important;
    background: {T['input_bg']} !important;
    color: {T['input_text']} !important;
    transition: border-color .2s !important;
}}
textarea::placeholder {{ color: {T['input_ph']} !important; }}
textarea:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 3px {T['accent']}22 !important;
}}

/* ── Char count ──────────────────────────────────────────── */
.sd-char {{
    text-align: right;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: {T['charcount_col']};
    margin-top: -8px;
    margin-bottom: 12px;
}}

/* ── Result card ─────────────────────────────────────────── */
.sd-result {{
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.5rem;
    animation: pop .35s cubic-bezier(.34,1.56,.64,1);
}}
@keyframes pop {{
    from {{ opacity:0; transform: translateY(14px) scale(.97); }}
    to   {{ opacity:1; transform: translateY(0) scale(1); }}
}}
.sd-result.spam {{ background:{T['spam_bg']}; border:1px solid {T['spam_border']}; }}
.sd-result.ham  {{ background:{T['ham_bg']};  border:1px solid {T['ham_border']};  }}

.sd-result-top {{ display:flex; align-items:center; gap:13px; margin-bottom:1.1rem; }}
.sd-result-icon {{
    width:46px; height:46px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:22px; flex-shrink:0;
}}
.sd-result.spam .sd-result-icon {{ background:{T['spam_icon_bg']}; }}
.sd-result.ham  .sd-result-icon {{ background:{T['ham_icon_bg']}; }}

.sd-result-label {{ font-weight:800; font-size:20px; line-height:1.1; }}
.sd-result.spam .sd-result-label {{ color:{T['spam_label']}; }}
.sd-result.ham  .sd-result-label {{ color:{T['ham_label']}; }}

.sd-result-sub {{ font-family:'DM Mono',monospace; font-size:12px; margin-top:3px; }}
.sd-result.spam .sd-result-sub {{ color:{T['spam_sub']}; }}
.sd-result.ham  .sd-result-sub {{ color:{T['ham_sub']}; }}

.sd-bar-label {{
    font-family:'DM Mono',monospace; font-size:11px;
    text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px;
}}
.sd-result.spam .sd-bar-label {{ color:{T['spam_label']}; }}
.sd-result.ham  .sd-bar-label {{ color:{T['ham_label']}; }}

.sd-bar-bg {{ height:6px; border-radius:100px; overflow:hidden; }}
.sd-result.spam .sd-bar-bg {{ background:{T['spam_bar_track']}; }}
.sd-result.ham  .sd-bar-bg {{ background:{T['ham_bar_track']}; }}

.sd-bar-fill {{ height:100%; border-radius:100px; }}
.sd-result.spam .sd-bar-fill {{ background:{T['spam_bar_fill']}; }}
.sd-result.ham  .sd-bar-fill {{ background:{T['ham_bar_fill']}; }}

.sd-bar-val {{
    font-family:'DM Mono',monospace; font-size:13px; font-weight:500;
    margin-top:6px; text-align:right;
}}
.sd-result.spam .sd-bar-val {{ color:{T['spam_bar_val']}; }}
.sd-result.ham  .sd-bar-val {{ color:{T['ham_bar_val']}; }}

/* ── History ─────────────────────────────────────────────── */
.sd-divider {{ height:.5px; background:{T['divider']}; margin:1.6rem 0; }}
.sd-hist-title {{
    font-family:'DM Mono',monospace; font-size:11px;
    text-transform:uppercase; letter-spacing:.08em;
    color:{T['text_muted']}; margin-bottom:.9rem;
}}
.sd-hist-item {{
    display:flex; align-items:center; gap:12px;
    padding:10px 14px;
    background:{T['hist_row_bg']};
    border-radius:8px;
    border:.5px solid {T['hist_row_border']};
    margin-bottom:7px;
}}
.sd-pill {{
    font-family:'DM Mono',monospace; font-size:10px; font-weight:500;
    padding:3px 9px; border-radius:100px; flex-shrink:0;
    text-transform:uppercase; letter-spacing:.05em;
}}
.sd-pill.spam {{ background:{T['spam_pill_bg']}; color:{T['spam_pill_text']}; }}
.sd-pill.ham  {{ background:{T['ham_pill_bg']};  color:{T['ham_pill_text']}; }}
.sd-hist-text {{
    font-size:13px; color:{T['hist_text']};
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;
}}
.sd-hist-conf {{
    font-family:'DM Mono',monospace; font-size:11px;
    color:{T['hist_conf']}; flex-shrink:0;
}}

/* ── Footer ──────────────────────────────────────────────── */
.sd-footer {{
    display:flex; align-items:center; gap:8px; margin-top:2.5rem;
    font-family:'DM Mono',monospace; font-size:11px; color:{T['footer_text']};
}}
.sd-footer-dot {{ width:3px; height:3px; background:{T['footer_text']}; border-radius:50%; }}

/* streamlit warning / info boxes */
.stAlert {{ background:{T['surface']} !important; border-color:{T['border']} !important; }}
</style>
""", unsafe_allow_html=True)

# ── TOGGLE ROW ─────────────────────────────────────────────────────────────
tog_col, _ = st.columns([1, 4])
with tog_col:
    if st.button(f"{T['toggle_icon']}  {T['toggle_label']}", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="sd-badge">
    <div class="dot"></div>
    LSTM · Model v2.4 · Active
</div>
<div class="sd-title">Detect <span class="acc">Spam</span><br>in seconds.</div>
<div class="sd-sub">Powered by a fine-tuned LSTM classifier. Paste any message below.</div>
""", unsafe_allow_html=True)

# ── EXAMPLE QUICK-FILL ─────────────────────────────────────────────────────
EXAMPLES = {
    "🎁  Promo bait":    "Congratulations! You've won a $1,000 gift card. Click here to claim now before it expires.",
    "💬  Legit message": "Hey, are we still on for lunch tomorrow? Let me know!",
    "🎣  Phishing":      "URGENT: Your bank account has been compromised. Verify now at: bit.ly/secure-verify",
}

selected_example = None
c1, c2, c3 = st.columns(3)
for col, (lbl, txt) in zip([c1, c2, c3], EXAMPLES.items()):
    if col.button(lbl, use_container_width=True):
        selected_example = txt

# ── TEXT AREA ──────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="Message",
    value=selected_example or "",
    height=150,
    placeholder="Paste message here — email, SMS, notification...",
    label_visibility="collapsed",
)
st.markdown(f'<div class="sd-char">{len(user_input)} chars</div>', unsafe_allow_html=True)

# ── ANALYZE BUTTON ─────────────────────────────────────────────────────────
st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
analyze = st.button("🔍  Analyze message", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── INFERENCE ──────────────────────────────────────────────────────────────
if analyze:
    if not user_input.strip():
        st.warning("Please enter a message before analyzing.")
    else:
        with st.spinner("Running inference…"):
            time.sleep(1.1)

            # ── Replace with your real pipeline ───────────────────────────
            # model  = PredictPipeline()
            # result = model.predict(user_input)
            # label  = result["label"]            # "spam" | "ham"
            # prob   = result["spam_probability"] # float 0–1
            # ──────────────────────────────────────────────────────────────

            # Mock (delete when using real model)
            SPAM_WORDS = ["win","winner","congratulation","free","click here",
                          "urgent","claim","gift","prize","verify","bank",
                          "compromised","bit.ly","offer","limited","expire"]
            is_spam = any(w in user_input.lower() for w in SPAM_WORDS)
            prob    = random.uniform(0.72, 0.97) if is_spam else random.uniform(0.02, 0.18)
            label   = "spam" if is_spam else "ham"

        st.session_state.result = {"label": label, "prob": prob}
        st.session_state.history.insert(0, {
            "text":  user_input,
            "label": label,
            "conf":  prob if label == "spam" else 1 - prob,
        })

# ── RESULT CARD ────────────────────────────────────────────────────────────
if st.session_state.result:
    r     = st.session_state.result
    label = r["label"]
    prob  = r["prob"]
    cls   = "spam" if label == "spam" else "ham"
    icon  = "🚨"   if label == "spam" else "✅"
    head  = "Spam Detected" if label == "spam" else "Safe Message"
    sub   = (f"Spam probability · {prob*100:.1f}%"
             if label == "spam"
             else f"Ham probability · {(1-prob)*100:.1f}%")
    bar   = int(prob * 100) if label == "spam" else int((1 - prob) * 100)

    st.markdown(f"""
    <div class="sd-result {cls}">
      <div class="sd-result-top">
        <div class="sd-result-icon">{icon}</div>
        <div>
          <div class="sd-result-label">{head}</div>
          <div class="sd-result-sub">{sub}</div>
        </div>
      </div>
      <div class="sd-bar-label">Confidence</div>
      <div class="sd-bar-bg">
        <div class="sd-bar-fill" style="width:{bar}%;"></div>
      </div>
      <div class="sd-bar-val">{bar}%</div>
    </div>
    """, unsafe_allow_html=True)

# ── HISTORY ────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-hist-title">Recent checks</div>', unsafe_allow_html=True)

    for item in st.session_state.history[:5]:
        preview = item["text"][:75] + ("…" if len(item["text"]) > 75 else "")
        st.markdown(f"""
        <div class="sd-hist-item">
          <span class="sd-pill {item['label']}">{item['label']}</span>
          <span class="sd-hist-text">{preview}</span>
          <span class="sd-hist-conf">{item['conf']*100:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑  Clear history"):
        st.session_state.history = []
        st.session_state.result  = None
        st.rerun()

# ── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sd-footer">
  <span>Built with LSTM + Streamlit</span>
  <div class="sd-footer-dot"></div>
  <span>Model v2.4</span>
</div>
""", unsafe_allow_html=True)