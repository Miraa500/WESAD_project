import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import io

st.set_page_config(
    page_title="WESAD Stress Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #EEF3F5 0%, #E4ECEF 100%);
}

section[data-testid="stSidebar"] {
    background: #123C4D;
}
section[data-testid="stSidebar"] * {
    color: #EAF3F3 !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.06);
    border: 1.5px dashed rgba(255,255,255,0.35);
    border-radius: 12px;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

/* Hero title */
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #123C4D;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.0rem;
    color: #4A6572;
    max-width: 680px;
    line-height: 1.5;
    margin-bottom: 0.4rem;
}

.pulse-divider {
    width: 100%;
    height: 26px;
    margin: 1rem 0 1.4rem 0;
}

/* Section card */
.section-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    box-shadow: 0 2px 14px rgba(18, 60, 77, 0.06);
    border: 1px solid rgba(18, 60, 77, 0.06);
    margin-bottom: 1.3rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.section-card:hover {
    box-shadow: 0 8px 24px rgba(18, 60, 77, 0.10);
}

.section-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2A9D8F;
    margin-bottom: 0.3rem;
}

.section-heading {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #123C4D;
    margin-bottom: 0.7rem;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
    color: white;
}

/* Streamlit widget overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    background: #F4F9F9;
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    color: #4A6572;
}
.stTabs [aria-selected="true"] {
    background: #146C6D !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(135deg, #146C6D 0%, #0F4C5C 100%);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 1.6rem;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(20, 108, 109, 0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(20, 108, 109, 0.32);
}

div[data-testid="stMetric"] {
    background: #F4F9F9;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #2A9D8F;
    transition: transform 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] { color: #4A6572 !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: #123C4D !important; font-family: 'Fraunces', serif !important; }

div[data-testid="stAlert"] { border-radius: 10px; }
footer, .stCaption { color: #7A8C94 !important; }
</style>
""", unsafe_allow_html=True)

LABEL_NAMES = {0: "Baseline", 1: "Stress", 2: "Amusement"}
LABEL_COLORS = {0: "#2A9D8F", 1: "#E76F51", 2: "#F4A261"}
CHANNEL_NAMES = ["ACC_x", "ACC_y", "ACC_z", "ECG", "EMG", "EDA", "Temp", "Resp"]
CHANNEL_COLORS = ["#8AB6D6", "#B8A9C9", "#6FB3A0", "#E76F51", "#9A8C98", "#2A9D8F", "#F4A261", "#457B9D"]

if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_wesad_model.h5")

model = load_model()

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="hero-title">🫀 WESAD Stress Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">A subject-independent CNN model that classifies psychological '
    'state — Baseline, Stress, or Amusement — from 8-channel physiological signals '
    '(ACC, ECG, EMG, EDA, Temperature, Respiration).</div>',
    unsafe_allow_html=True
)
st.markdown("""
<svg class="pulse-divider" viewBox="0 0 600 26" xmlns="http://www.w3.org/2000/svg">
  <polyline points="0,13 180,13 200,3 215,23 230,13 600,13"
    fill="none" stroke="#2A9D8F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR — DATA SOURCE
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ Input Setup")
    option = st.radio(
        "Data source",
        ["Sample from dataset (demo)", "Upload a .npy file"],
        label_visibility="visible"
    )

    window = None
    true_label_str = None

    if option == "Sample from dataset (demo)":
        X = np.load("data/X.npy")
        y = np.load("data/y.npy")
        sample_idx = st.slider("Sample index", 0, len(X) - 1, 0)
        window = X[sample_idx]
        true_label_map = {1: "Baseline", 2: "Stress", 3: "Amusement"}
        true_label_str = true_label_map.get(y[sample_idx], "Unknown")
        st.info(f"True label: **{true_label_str}**")
    else:
        uploaded_file = st.file_uploader("Upload .npy — shape (700, 8)", type=["npy"])
        if uploaded_file is not None:
            window = np.load(uploaded_file)
            if window.shape != (700, 8):
                st.error(f"Shape must be (700, 8), got {window.shape}")
                window = None

    st.markdown("---")
    st.markdown("### 👁️ Channel Visibility")
    visible_channels = st.multiselect(
        "Channels to plot", CHANNEL_NAMES, default=CHANNEL_NAMES
    )

    st.markdown("---")
    if st.session_state.history:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# =====================================================
# TABS
# =====================================================
tab_predict, tab_history = st.tabs(["🔍 Predict", f"📜 History ({len(st.session_state.history)})"])

with tab_predict:
    if window is not None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Step 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Signal Preview</div>', unsafe_allow_html=True)

        fig = go.Figure()
        for i, ch_name in enumerate(CHANNEL_NAMES):
            if ch_name in visible_channels:
                fig.add_trace(go.Scatter(
                    y=window[:, i], mode="lines", name=ch_name,
                    line=dict(color=CHANNEL_COLORS[i], width=1.6),
                    hovertemplate=f"{ch_name}: %{{y:.3f}}<br>t=%{{x}}<extra></extra>"
                ))
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#4A6572", family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Time steps", gridcolor="#E4ECEF", zeroline=False),
            yaxis=dict(title="Value", gridcolor="#E4ECEF", zeroline=False),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        predict_clicked = st.button("🔍 Predict Psychological State", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_clicked:
            with st.spinner("Running inference through the CNN..."):
                input_data = np.expand_dims(window, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_class = int(np.argmax(prediction))
                confidence = float(prediction[predicted_class] * 100)

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "predicted": LABEL_NAMES[predicted_class],
                "confidence": round(confidence, 1),
                "true_label": true_label_str if true_label_str else "N/A"
            })

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-eyebrow">Step 2</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)

            badge_color = LABEL_COLORS[predicted_class]
            st.markdown(
                f'<span class="status-badge" style="background:{badge_color};">'
                f'● {LABEL_NAMES[predicted_class]} — {confidence:.1f}% confidence</span>',
                unsafe_allow_html=True
            )
            st.write("")

            col1, col2 = st.columns([1, 1.2])

            with col1:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    number={"suffix": "%", "font": {"color": "#123C4D", "size": 34}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#4A6572"},
                        "bar": {"color": badge_color},
                        "bgcolor": "#F4F9F9",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "#EDF4F4"},
                            {"range": [50, 100], "color": "#E4F0EF"},
                        ],
                    },
                    title={"text": "Confidence", "font": {"color": "#4A6572", "size": 14}}
                ))
                gauge.update_layout(
                    height=260, margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter")
                )
                st.plotly_chart(gauge, use_container_width=True)

            with col2:
                bar = go.Figure(go.Bar(
                    x=[LABEL_NAMES[i] for i in range(3)],
                    y=prediction * 100,
                    marker_color=[LABEL_COLORS[i] for i in range(3)],
                    text=[f"{v:.1f}%" for v in prediction * 100],
                    textposition="outside"
                ))
                bar.update_layout(
                    height=260, margin=dict(l=10, r=10, t=30, b=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#4A6572", family="Inter"),
                    yaxis=dict(title="Confidence (%)", range=[0, 110], gridcolor="#E4ECEF"),
                    xaxis=dict(gridcolor="#E4ECEF")
                )
                st.plotly_chart(bar, use_container_width=True)

            report_df = pd.DataFrame([{
                "timestamp": datetime.now().isoformat(),
                "predicted_state": LABEL_NAMES[predicted_class],
                "confidence_%": round(confidence, 2),
                "true_label": true_label_str if true_label_str else "N/A",
                **{f"prob_{LABEL_NAMES[i]}": round(float(prediction[i] * 100), 2) for i in range(3)}
            }])
            csv_buffer = io.StringIO()
            report_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download this result as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"wesad_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Select a sample or upload a file from the sidebar to run a prediction.")

with tab_history:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Session Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Prediction History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        counts = hist_df["predicted"].value_counts()
        trend_fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[LABEL_COLORS[[k for k, v in LABEL_NAMES.items() if v == label][0]] for label in counts.index]
        ))
        trend_fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#4A6572", family="Inter"),
            yaxis=dict(title="Count", gridcolor="#E4ECEF"),
            title="Distribution of predictions this session"
        )
        st.plotly_chart(trend_fig, use_container_width=True)

        full_csv = io.StringIO()
        hist_df.to_csv(full_csv, index=False)
        st.download_button(
            "⬇️ Download full history as CSV",
            data=full_csv.getvalue(),
            file_name="wesad_session_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No predictions yet this session. Run one from the Predict tab.")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("WESAD Psychological State Classification · CNN · Subject-Independent Evaluation · 91% Test Accuracy")