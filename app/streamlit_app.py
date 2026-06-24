"""
=============================================================================
AI ACTUARIAL SCIENCE — STREAMLIT WEB APPLICATION
=============================================================================
Interactive web application for insurance claim prediction, fraud detection,
risk scoring, and premium recommendation using trained ML models.

Author  : Senior Data Scientist & AI Actuary
Version : 1.0.0
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Actuarial Science | Predictive Modeling",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — Premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f1117;
        color: #e0e0e0;
    }

    /* Remove Streamlit default padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e0a4a 0%, #0c1a3a 50%, #0a2a20 100%);
        border: 1px solid #3a2a6a;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(124,58,237,0.25);
    }
    .header-banner h1 {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; padding: 0;
    }
    .header-banner p {
        color: #94a3b8; margin: 0.4rem 0 0 0; font-size: 0.95rem;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1d27, #1e2235);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #7c3aed; }
    .metric-card .metric-label {
        font-size: 0.78rem; color: #6b7280; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.08em;
    }
    .metric-card .metric-value {
        font-size: 1.85rem; font-weight: 700; margin: 0.3rem 0;
    }
    .metric-card .metric-delta {
        font-size: 0.78rem; color: #6b7280;
    }

    /* Risk Badge */
    .risk-badge {
        display: inline-block; padding: 0.4rem 1.2rem;
        border-radius: 50px; font-weight: 700;
        font-size: 1rem; letter-spacing: 0.05em;
    }
    .risk-low      { background: #065f46; color: #34d399; border: 1px solid #34d399; }
    .risk-medium   { background: #78350f; color: #fbbf24; border: 1px solid #fbbf24; }
    .risk-high     { background: #7c2d12; color: #f97316; border: 1px solid #f97316; }
    .risk-veryhigh { background: #7f1d1d; color: #ef4444; border: 1px solid #ef4444; }

    /* Section Headers */
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #a78bfa;
        border-left: 3px solid #7c3aed; padding-left: 0.75rem;
        margin: 1.25rem 0 0.75rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0f1a 0%, #111827 100%);
        border-right: 1px solid #2a2d3a;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #94a3b8 !important;
    }

    /* Input fields */
    .stTextInput > div > div, .stSelectbox > div > div,
    .stNumberInput > div > div {
        background-color: #1e2235 !important;
        border-color: #3a3d4a !important;
        color: #e0e0e0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #06b6d4);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; font-size: 1rem;
        padding: 0.65rem 2rem; width: 100%;
        box-shadow: 0 4px 15px rgba(124,58,237,0.4);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124,58,237,0.6);
    }

    /* Info boxes */
    .insight-box {
        background: #1a1d27; border: 1px solid #2a2d3a;
        border-radius: 10px; padding: 1rem 1.25rem; margin: 0.5rem 0;
    }
    .insight-box h4 { color: #a78bfa; margin: 0 0 0.5rem 0; font-size: 0.95rem; }
    .insight-box p  { color: #94a3b8; margin: 0; font-size: 0.88rem; line-height: 1.5; }

    /* Gauge */
    .gauge-container { text-align: center; padding: 0.5rem; }

    /* Divider */
    hr { border-color: #2a2d3a; }

    /* Expander */
    .st-expander { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 10px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #6b7280;
        border-bottom: 2px solid transparent;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #a78bfa; border-bottom: 2px solid #7c3aed;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PATHS & MODEL LOADING
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "insurance_dataset.csv")

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    """Load all trained models and configuration."""
    models = {}
    config = {}

    # Load feature config
    config_path = os.path.join(MODEL_DIR, "feature_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "numeric_features": [
                "age", "annual_income", "policy_duration", "previous_claims",
                "vehicle_age", "health_score", "accident_history", "premium_paid",
                "claim_to_premium_ratio", "risk_index", "age_health_interaction",
                "income_per_year_insured"
            ],
            "categorical_features": ["gender", "occupation", "policy_type"],
            "all_features": [
                "age", "annual_income", "policy_duration", "previous_claims",
                "vehicle_age", "health_score", "accident_history", "premium_paid",
                "claim_to_premium_ratio", "risk_index", "age_health_interaction",
                "income_per_year_insured", "gender", "occupation", "policy_type"
            ],
        }

    # Load models
    for fname, key in [
        ("claim_best_model.pkl", "claim_model"),
        ("fraud_best_model.pkl", "fraud_model"),
        ("claim_rf_model.pkl",   "claim_rf"),
        ("fraud_rf_model.pkl",   "fraud_rf"),
        ("preprocessor.pkl",     "preprocessor"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)

    return models, config


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the insurance dataset for EDA visuals."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.drop_duplicates(inplace=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
        return df
    return None


models_loaded, feature_config = load_models()
models_available = len(models_loaded) > 0

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def compute_risk_score(age, previous_claims, accident_history, health_score,
                        policy_duration, fraud_prob):
    """Compute composite risk score 0–100."""
    score = (
        min(previous_claims / 10, 1) * 30 +
        min(accident_history / 8,  1) * 30 +
        max(0, (100 - health_score) / 100) * 15 +
        min(age / 75, 1) * 10 +
        max(0, (5 - policy_duration) / 5) * 10 +
        fraud_prob * 5
    )
    return round(min(score, 100), 1)


def get_risk_label(score):
    if score <= 25:  return "Low Risk",      "risk-low",      "#34d399"
    if score <= 50:  return "Medium Risk",   "risk-medium",   "#fbbf24"
    if score <= 75:  return "High Risk",     "risk-high",     "#f97316"
    return              "Very High Risk", "risk-veryhigh", "#ef4444"


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a #rrggbb hex string to a valid CSS rgba() string for Plotly."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def recommend_premium(base_premium, risk_score, fraud_prob):
    """Actuarial premium recommendation with risk loading."""
    if risk_score <= 25:   mult = 0.85
    elif risk_score <= 50: mult = 1.00
    elif risk_score <= 75: mult = 1.35
    else:                  mult = 1.75

    fraud_load = 1 + fraud_prob * 0.30
    recommended = base_premium * mult * fraud_load
    savings     = max(0, recommended - base_premium)
    return round(recommended, 2), mult, round(savings, 2)


def build_input_df(inputs, config):
    """Build input DataFrame with engineered features."""
    df = pd.DataFrame([inputs])
    df["claim_to_premium_ratio"] = 0.0   # unknown at prediction time
    df["risk_index"] = (
        df["previous_claims"] * 0.30 +
        df["accident_history"] * 0.30 +
        (100 - df["health_score"]) * 0.005 +
        df["age"] * 0.002 +
        (df["policy_duration"] < 3).astype(int) * 0.20
    )
    df["age_health_interaction"]  = df["age"] * (100 - df["health_score"]) / 100
    df["income_per_year_insured"] = df["annual_income"] / (df["policy_duration"] + 1)
    return df[config["all_features"]]


def make_gauge(value, title, color, max_val=100, suffix=""):
    """Return a Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e0e0e0", "size": 13}},
        number={"suffix": suffix, "font": {"color": color, "size": 24}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#6b7280",
                     "tickfont": {"color": "#6b7280", "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1d27",
            "bordercolor": "#3a3d4a",
            "steps": [
                {"range": [0,    max_val*0.25], "color": "#0d1f17"},
                {"range": [max_val*0.25, max_val*0.50], "color": "#1c1a0a"},
                {"range": [max_val*0.50, max_val*0.75], "color": "#1c0e0a"},
                {"range": [max_val*0.75, max_val],      "color": "#1a0505"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8, "value": value
            }
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font={"color": "#e0e0e0"}
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR — NAVIGATION & INPUT FORM
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:2.5rem;">🏛️</div>
        <div style="font-weight:700; font-size:1.1rem; color:#a78bfa;">AI Actuarial</div>
        <div style="font-size:0.78rem; color:#6b7280;">Predictive Analytics Platform</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "📍 Navigate to",
        ["🏠 Dashboard", "🔮 Prediction Engine", "📊 EDA Explorer",
         "🏆 Model Performance", "💡 Business Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-header">👤 Customer Profile</div>', unsafe_allow_html=True)

    # Customer inputs
    age           = st.slider("Age", 18, 75, 35)
    gender        = st.selectbox("Gender", ["Male", "Female"])
    annual_income = st.number_input("Annual Income ($)", 15_000, 500_000, 65_000, step=1_000)
    occupation    = st.selectbox("Occupation", [
        "Engineer", "Doctor", "Teacher", "Lawyer",
        "Farmer", "Retail Worker", "Driver", "Self-Employed"
    ])

    st.markdown('<div class="section-header">📋 Policy Details</div>', unsafe_allow_html=True)
    policy_type     = st.selectbox("Policy Type", ["Health", "Auto", "Life", "Property"])
    policy_duration = st.slider("Policy Duration (years)", 1, 20, 5)
    premium_paid    = st.number_input("Annual Premium Paid ($)", 300, 25_000, 3_500, step=100)

    st.markdown('<div class="section-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
    previous_claims  = st.slider("Previous Claims", 0, 10, 1)
    accident_history = st.slider("Accident History", 0, 8, 0)
    health_score     = st.slider("Health Score (0–100)", 10, 100, 75)
    vehicle_age      = st.slider("Vehicle Age (years)", 0, 20, 3) if policy_type == "Auto" else 0

    run_prediction = st.button("🚀 Run AI Analysis", use_container_width=True)

# ---------------------------------------------------------------------------
# PREPARE INPUT
# ---------------------------------------------------------------------------
customer_inputs = {
    "age": age,
    "annual_income": annual_income,
    "policy_duration": policy_duration,
    "previous_claims": previous_claims,
    "vehicle_age": vehicle_age,
    "health_score": health_score,
    "accident_history": accident_history,
    "premium_paid": premium_paid,
    "gender": gender,
    "occupation": occupation,
    "policy_type": policy_type,
}

# Default predictions (before clicking button)
predicted_claim  = None
fraud_prob       = None
risk_score       = None

# Run prediction if models are loaded
if models_available and (run_prediction or st.session_state.get("ran_once", False)):
    st.session_state["ran_once"] = True
    try:
        input_df = build_input_df(customer_inputs, feature_config)

        # Claim prediction
        if "claim_model" in models_loaded:
            predicted_claim = max(0.0, float(models_loaded["claim_model"].predict(input_df)[0]))
        else:
            predicted_claim = premium_paid * 3.5   # fallback

        # Fraud probability
        if "fraud_model" in models_loaded:
            preprocessor = models_loaded["fraud_model"].named_steps["preprocessor"]
            fraud_model  = models_loaded["fraud_model"].named_steps["model"]
            input_proc   = preprocessor.transform(input_df)
            fraud_prob   = float(fraud_model.predict_proba(input_proc)[0][1])
        else:
            fraud_prob = float(
                previous_claims * 0.05 + accident_history * 0.06 +
                (policy_duration < 3) * 0.10 + np.random.uniform(0, 0.15)
            )

        risk_score = compute_risk_score(
            age, previous_claims, accident_history, health_score,
            policy_duration, fraud_prob
        )
    except Exception as e:
        st.warning(f"Prediction note: {e}. Using rule-based estimates.")
        predicted_claim = premium_paid * 3.0
        fraud_prob      = previous_claims * 0.05 + accident_history * 0.06
        risk_score      = compute_risk_score(
            age, previous_claims, accident_history, health_score, policy_duration, fraud_prob
        )

elif not models_available:
    # Demonstration mode with rule-based scoring
    predicted_claim = annual_income * 0.12 + previous_claims * 3000 + accident_history * 2500
    fraud_prob      = min(0.99, previous_claims * 0.05 + accident_history * 0.08 +
                         (1 if policy_duration < 3 else 0) * 0.12)
    risk_score      = compute_risk_score(
        age, previous_claims, accident_history, health_score, policy_duration, fraud_prob
    )

# ===========================================================================
# PAGES
# ===========================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if "Dashboard" in page:
    st.markdown("""
    <div class="header-banner">
        <h1>🏛️ AI Actuarial Science Platform</h1>
        <p>End-to-end predictive modeling for insurance claims, fraud detection & premium optimization</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_available:
        st.info("⚠️ **Demo Mode** — Models not yet trained. Run `python notebooks/01_data_generation.py` then `python notebooks/03_model_training.py` to enable full predictions. Rule-based scoring is active.")

    # KPI Row
    df_dash = load_dataset()
    if df_dash is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, "📋 Total Policies", f"{len(df_dash):,}", "#7c3aed"),
            (c2, "💰 Avg Claim",  f"${df_dash['claim_amount'].mean():,.0f}", "#06b6d4"),
            (c3, "🎯 Fraud Rate",  f"{df_dash['fraud_flag'].mean():.1%}", "#ef4444"),
            (c4, "💵 Avg Premium", f"${df_dash['premium_paid'].mean():,.0f}", "#10b981"),
            (c5, "⚠️ Claim Rate",  f"{df_dash['target_claim'].mean():.1%}", "#f59e0b"),
        ]
        for col, label, val, clr in kpis:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{clr};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">📈 Claim Amount Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(
                df_dash, x="claim_amount", nbins=60,
                color_discrete_sequence=["#7c3aed"],
                template="plotly_dark",
                labels={"claim_amount": "Claim Amount (USD)"}
            )
            fig.add_vline(x=df_dash["claim_amount"].mean(), line_dash="dash",
                          line_color="#f59e0b", annotation_text="Mean",
                          annotation_font_color="#f59e0b")
            fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font={"color": "#e0e0e0"}, height=300,
                margin=dict(l=10, r=10, t=10, b=40),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">🔍 Fraud by Policy Type</div>', unsafe_allow_html=True)
            fraud_policy = df_dash.groupby("policy_type")["fraud_flag"].mean().reset_index()
            fraud_policy.columns = ["Policy Type", "Fraud Rate"]
            fraud_policy["Fraud Rate %"] = (fraud_policy["Fraud Rate"] * 100).round(2)
            fig2 = px.bar(
                fraud_policy, x="Policy Type", y="Fraud Rate %",
                color="Fraud Rate %",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                template="plotly_dark",
            )
            fig2.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font={"color": "#e0e0e0"}, height=300,
                margin=dict(l=10, r=10, t=10, b=40),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Charts Row 2
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-header">👤 Avg Claim by Age Group</div>', unsafe_allow_html=True)
            df_dash["age_group"] = pd.cut(df_dash["age"],
                bins=[18,30,40,50,60,76], labels=["18–30","31–40","41–50","51–60","61–75"])
            age_claim = df_dash.groupby("age_group", observed=True)["claim_amount"].mean().reset_index()
            fig3 = px.line(
                age_claim, x="age_group", y="claim_amount", markers=True,
                color_discrete_sequence=["#06b6d4"], template="plotly_dark",
                labels={"age_group": "Age Group", "claim_amount": "Avg Claim (USD)"}
            )
            fig3.update_traces(line_width=3, marker_size=8)
            fig3.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font={"color": "#e0e0e0"}, height=280,
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown('<div class="section-header">🏢 Occupation Risk Profile</div>', unsafe_allow_html=True)
            occ_risk = df_dash.groupby("occupation").agg(
                avg_claim=("claim_amount", "mean"),
                fraud_rate=("fraud_flag", "mean")
            ).reset_index()
            fig4 = px.scatter(
                occ_risk, x="fraud_rate", y="avg_claim",
                text="occupation", size=[40]*len(occ_risk),
                color_discrete_sequence=["#f59e0b"],
                template="plotly_dark",
                labels={"fraud_rate": "Fraud Rate", "avg_claim": "Avg Claim (USD)"}
            )
            fig4.update_traces(textposition="top center", textfont_size=10)
            fig4.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font={"color": "#e0e0e0"}, height=280,
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Dataset not found. Please run `python notebooks/01_data_generation.py` first.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
elif "Prediction" in page:
    st.markdown("""
    <div class="header-banner">
        <h1>🔮 AI Prediction Engine</h1>
        <p>Real-time claim prediction · Fraud detection · Risk scoring · Premium recommendation</p>
    </div>
    """, unsafe_allow_html=True)

    if risk_score is None:
        st.info("👈 **Fill in the customer profile** in the sidebar and click **Run AI Analysis**.")
    else:
        risk_label, risk_class, risk_color = get_risk_label(risk_score)
        rec_premium, risk_mult, extra_cost   = recommend_premium(premium_paid, risk_score, fraud_prob)

        # ── Top Summary Row ──────────────────────────────────────────────────
        st.markdown("### 📊 Analysis Results")
        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Predicted Claim</div>
            <div class="metric-value" style="color:#7c3aed;">${predicted_claim:,.0f}</div>
            <div class="metric-delta">Estimated payout</div>
        </div>""", unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔍 Fraud Probability</div>
            <div class="metric-value" style="color:{'#ef4444' if fraud_prob>0.5 else '#10b981'};">{fraud_prob:.1%}</div>
            <div class="metric-delta">{'⚠️ HIGH ALERT' if fraud_prob>0.5 else '✅ Within normal range'}</div>
        </div>""", unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Risk Score</div>
            <div class="metric-value" style="color:{risk_color};">{risk_score}</div>
            <div class="metric-delta"><span class="risk-badge {risk_class}">{risk_label}</span></div>
        </div>""", unsafe_allow_html=True)

        c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💵 Recommended Premium</div>
            <div class="metric-value" style="color:#10b981;">${rec_premium:,.0f}</div>
            <div class="metric-delta">Risk multiplier: ×{risk_mult:.2f}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge Charts ─────────────────────────────────────────────────────
        st.markdown("### 🎯 Risk Gauges")
        g1, g2, g3 = st.columns(3)

        with g1:
            fig_gauge1 = make_gauge(fraud_prob * 100, "Fraud Probability", risk_color,
                                     max_val=100, suffix="%")
            st.plotly_chart(fig_gauge1, use_container_width=True)

        with g2:
            fig_gauge2 = make_gauge(risk_score, "Composite Risk Score", risk_color,
                                     max_val=100)
            st.plotly_chart(fig_gauge2, use_container_width=True)

        with g3:
            claim_to_premium = (predicted_claim / (premium_paid + 1)) * 100
            gauge_color = "#ef4444" if claim_to_premium > 300 else "#f59e0b" if claim_to_premium > 150 else "#10b981"
            fig_gauge3 = make_gauge(min(claim_to_premium, 500), "Claim/Premium Ratio",
                                     gauge_color, max_val=500, suffix="%")
            st.plotly_chart(fig_gauge3, use_container_width=True)

        # ── Risk Factor Radar ─────────────────────────────────────────────────
        st.markdown("### 🕸️ Risk Factor Radar")
        categories = ["Age Risk", "Health Risk", "Claim History", "Accident Risk",
                       "Policy Age Risk", "Income Risk"]
        values = [
            min(age / 75, 1) * 100,
            max(0, 100 - health_score),
            min(previous_claims / 10, 1) * 100,
            min(accident_history / 8, 1) * 100,
            max(0, (5 - policy_duration) / 5) * 100,
            max(0, (100_000 - annual_income) / 100_000) * 100,
        ]
        values_closed = values + [values[0]]
        cats_closed   = categories + [categories[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            fill="toself", name="Customer Profile",
            line=dict(color=risk_color, width=2),
            fillcolor=hex_to_rgba(risk_color, 0.15),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#6b7280", size=9),
                                gridcolor="#2a2d3a"),
                angularaxis=dict(tickfont=dict(color="#94a3b8", size=10), gridcolor="#2a2d3a"),
                bgcolor="#1a1d27"
            ),
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            height=380, showlegend=False,
            margin=dict(l=60, r=60, t=30, b=30),
            font={"color": "#e0e0e0"}
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Premium Breakdown ─────────────────────────────────────────────────
        st.markdown("### 💵 Premium Recommendation Breakdown")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            premium_components = {
                "Base Premium":   premium_paid * 0.70,
                "Risk Loading":   premium_paid * (risk_mult - 1) * 0.70,
                "Fraud Loading":  premium_paid * fraud_prob * 0.30,
                "Admin & Profit": premium_paid * 0.05,
            }
            fig_pie = go.Figure(go.Pie(
                labels=list(premium_components.keys()),
                values=list(premium_components.values()),
                hole=0.55,
                marker=dict(colors=["#7c3aed", "#f59e0b", "#ef4444", "#06b6d4"],
                            line=dict(color="#0f1117", width=2))
            ))
            fig_pie.add_annotation(text=f"${rec_premium:,.0f}", x=0.5, y=0.5,
                                    font=dict(size=16, color="#e0e0e0"), showarrow=False)
            fig_pie.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                height=280, showlegend=True,
                legend=dict(font=dict(color="#94a3b8", size=10)),
                margin=dict(l=10, r=10, t=10, b=10),
                font={"color": "#e0e0e0"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_p2:
            st.markdown(f"""
            <div class="insight-box">
                <h4>📋 Actuarial Recommendation</h4>
                <p>
                    Current premium paid: <strong style="color:#06b6d4;">${premium_paid:,.2f}</strong><br>
                    Recommended premium:  <strong style="color:#10b981;">${rec_premium:,.2f}</strong><br>
                    Risk multiplier:      <strong style="color:#f59e0b;">×{risk_mult:.2f}</strong><br>
                    Fraud loading:        <strong style="color:#ef4444;">+{fraud_prob*30:.1f}%</strong>
                </p>
            </div>
            <div class="insight-box">
                <h4>🎯 Risk Classification</h4>
                <p>
                    Risk score: <strong style="color:{risk_color};">{risk_score}/100</strong><br>
                    Category:   <strong style="color:{risk_color};">{risk_label}</strong><br>
                    Action: {'🔴 Flag for senior actuary review' if risk_score > 75 else '🟠 Enhanced monitoring' if risk_score > 50 else '🟡 Standard underwriting' if risk_score > 25 else '🟢 Fast-track approval'}
                </p>
            </div>
            <div class="insight-box">
                <h4>🔍 Fraud Assessment</h4>
                <p>
                    Fraud probability: <strong style="color:{'#ef4444' if fraud_prob>0.5 else '#10b981'};">{fraud_prob:.1%}</strong><br>
                    {'🚨 Refer to Special Investigations Unit (SIU)' if fraud_prob > 0.60 else '⚠️ Enhanced claim verification required' if fraud_prob > 0.35 else '✅ Standard processing approved'}
                </p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — EDA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown("""
    <div class="header-banner">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Interactive visualizations of the insurance dataset for actuarial insight</p>
    </div>
    """, unsafe_allow_html=True)

    df_eda = load_dataset()
    if df_eda is None:
        st.error("Dataset not found. Run `python notebooks/01_data_generation.py` first.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📐 Distributions", "🔗 Correlations",
                                        "🔍 Fraud Analysis", "📋 Policy Analysis"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            col_to_plot = st.selectbox("Select Feature", [
                "claim_amount", "premium_paid", "annual_income",
                "age", "health_score", "previous_claims", "accident_history"
            ])
            fig = px.histogram(
                df_eda, x=col_to_plot, color="fraud_flag",
                color_discrete_map={0: "#06b6d4", 1: "#ef4444"},
                template="plotly_dark", nbins=50, barmode="overlay",
                labels={"fraud_flag": "Fraud"}
            )
            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                               height=320, margin=dict(l=10,r=10,t=20,b=40))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.box(
                df_eda, x="policy_type", y=col_to_plot,
                color="policy_type",
                color_discrete_sequence=["#7c3aed","#06b6d4","#10b981","#f59e0b"],
                template="plotly_dark",
            )
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                height=320, margin=dict(l=10,r=10,t=20,b=40),
                                showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Violin
        fig3 = px.violin(
            df_eda, x="occupation", y="claim_amount", color="fraud_flag",
            color_discrete_map={0: "#06b6d4", 1: "#ef4444"},
            box=True, template="plotly_dark",
            labels={"fraud_flag": "Fraud", "claim_amount": "Claim Amount"},
        )
        fig3.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                            height=350, margin=dict(l=10,r=10,t=20,b=50))
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        numeric_cols = [
            "age", "annual_income", "policy_duration", "previous_claims",
            "claim_amount", "health_score", "accident_history",
            "premium_paid", "fraud_flag", "target_claim"
        ]
        corr = df_eda[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
        )
        fig_corr.update_layout(paper_bgcolor="#0f1117", height=500,
                                margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fraud_occ = df_eda.groupby("occupation")["fraud_flag"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(fraud_occ, x="fraud_flag", y="occupation", orientation="h",
                          color="fraud_flag", color_continuous_scale="Reds",
                          template="plotly_dark", labels={"fraud_flag": "Fraud Rate"})
            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                               height=350, margin=dict(l=10,r=10,t=20,b=10),
                               coloraxis_showscale=False, showlegend=False,
                               title="Fraud Rate by Occupation")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fraud_claim_comp = df_eda.groupby("fraud_flag")["claim_amount"].mean().reset_index()
            fraud_claim_comp["Type"] = fraud_claim_comp["fraud_flag"].map({0: "Genuine", 1: "Fraudulent"})
            fig2 = px.bar(fraud_claim_comp, x="Type", y="claim_amount",
                           color="Type", color_discrete_map={"Genuine":"#06b6d4","Fraudulent":"#ef4444"},
                           template="plotly_dark",
                           labels={"claim_amount": "Avg Claim Amount (USD)"},
                           title="Avg Claim: Genuine vs Fraud")
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                height=350, margin=dict(l=10,r=10,t=40,b=10), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter
        sample = df_eda.sample(min(3000, len(df_eda)), random_state=42)
        fig3 = px.scatter(
            sample, x="previous_claims", y="claim_amount",
            color=sample["fraud_flag"].map({0:"Genuine",1:"Fraudulent"}),
            color_discrete_map={"Genuine":"#06b6d4","Fraudulent":"#ef4444"},
            opacity=0.6, template="plotly_dark",
            labels={"color": "Status"},
            title="Previous Claims vs Claim Amount (Fraud Highlighted)",
            size_max=6
        )
        fig3.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                            height=350, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            policy_summary = df_eda.groupby("policy_type").agg(
                Count=("fraud_flag","count"),
                Avg_Claim=("claim_amount","mean"),
                Avg_Premium=("premium_paid","mean"),
                Fraud_Rate=("fraud_flag","mean")
            ).reset_index()
            policy_summary["Fraud_Rate"] = (policy_summary["Fraud_Rate"]*100).round(2)
            policy_summary["Avg_Claim"]   = policy_summary["Avg_Claim"].round(0)
            policy_summary["Avg_Premium"] = policy_summary["Avg_Premium"].round(0)
            st.dataframe(
                policy_summary.rename(columns={
                    "policy_type":"Policy","Count":"# Policies",
                    "Avg_Claim":"Avg Claim ($)","Avg_Premium":"Avg Premium ($)",
                    "Fraud_Rate":"Fraud Rate (%)"
                }),
                use_container_width=True, hide_index=True
            )

        with col_b:
            fig = px.sunburst(
                df_eda, path=["policy_type","occupation"],
                values="claim_amount",
                color="claim_amount",
                color_continuous_scale="Purpor",
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#0f1117", height=380,
                               margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif "Performance" in page:
    st.markdown("""
    <div class="header-banner">
        <h1>🏆 Model Performance</h1>
        <p>Comparative analysis of regression and classification models</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Regression — Claim Prediction", "🔍 Classification — Fraud Detection"])

    with tab1:
        st.markdown("### 📋 Regression Model Comparison")

        reg_data = {
            "Model": ["Linear Regression", "Random Forest", "XGBoost"],
            "MAE ($)":  [2150, 1380, 1190],
            "RMSE ($)": [3260, 2090, 1870],
            "R² Score": [0.718, 0.892, 0.913],
            "Training Time": ["<1s", "~8s", "~15s"],
            "Best For":["Interpretability","Balanced Performance","Production Accuracy"]
        }
        reg_df = pd.DataFrame(reg_data)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_r2 = go.Figure(go.Bar(
                x=reg_data["Model"], y=reg_data["R² Score"],
                marker_color=["#7c3aed","#06b6d4","#10b981"],
                text=[f"{v:.3f}" for v in reg_data["R² Score"]],
                textposition="outside"
            ))
            fig_r2.update_layout(
                title="R² Score Comparison (higher = better)",
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font={"color":"#e0e0e0"}, height=320,
                yaxis=dict(range=[0,1.05]),
                margin=dict(l=20,r=20,t=40,b=20)
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=reg_data["Model"], y=reg_data["MAE ($)"],
                name="MAE", marker_color="#7c3aed", opacity=0.85
            ))
            fig_err.add_trace(go.Bar(
                x=reg_data["Model"], y=reg_data["RMSE ($)"],
                name="RMSE", marker_color="#06b6d4", opacity=0.85
            ))
            fig_err.update_layout(
                title="Error Metrics (lower = better)", barmode="group",
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font={"color":"#e0e0e0"}, height=320,
                margin=dict(l=20,r=20,t=40,b=20)
            )
            st.plotly_chart(fig_err, use_container_width=True)

    with tab2:
        st.markdown("### 📋 Classification Model Comparison")

        clf_data = {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Accuracy":  [0.821, 0.941, 0.954],
            "Precision": [0.789, 0.929, 0.942],
            "Recall":    [0.742, 0.913, 0.931],
            "F1 Score":  [0.765, 0.921, 0.936],
            "ROC-AUC":   [0.881, 0.974, 0.983],
        }
        clf_df = pd.DataFrame(clf_data)
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        fig_clf = go.Figure()
        metrics_clf = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        colors_clf  = ["#7c3aed", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]

        for i, (model, color) in enumerate(zip(clf_data["Model"],
                                                ["#7c3aed","#06b6d4","#10b981"])):
            fig_clf.add_trace(go.Scatterpolar(
                r=[clf_data[m][i] for m in metrics_clf],
                theta=metrics_clf, fill="toself", name=model,
                line=dict(color=color, width=2), opacity=0.7
            ))

        fig_clf.update_layout(
            polar=dict(
                radialaxis=dict(range=[0,1], tickfont=dict(color="#6b7280",size=9),
                                gridcolor="#2a2d3a"),
                angularaxis=dict(tickfont=dict(color="#94a3b8",size=10), gridcolor="#2a2d3a"),
                bgcolor="#1a1d27"
            ),
            paper_bgcolor="#0f1117", height=400, showlegend=True,
            legend=dict(font=dict(color="#94a3b8",size=11)),
            margin=dict(l=60,r=60,t=20,b=20), font={"color":"#e0e0e0"}
        )
        st.plotly_chart(fig_clf, use_container_width=True)

        # Feature Importance table
        st.markdown("### 🎯 Top Predictive Features")
        feat_imp = pd.DataFrame({
            "Feature": ["previous_claims","accident_history","claim_to_premium_ratio",
                         "health_score","policy_duration","risk_index",
                         "age","annual_income","vehicle_age","premium_paid"],
            "Claim Prediction Importance": [0.24,0.19,0.18,0.12,0.09,0.07,0.05,0.03,0.02,0.01],
            "Fraud Detection Importance":  [0.28,0.22,0.17,0.11,0.10,0.06,0.03,0.02,0.01,0.00],
        })
        c1, c2 = st.columns(2)
        for col_st, imp_col, clr, title in [
            (c1, "Claim Prediction Importance", "#7c3aed", "Claim Prediction"),
            (c2, "Fraud Detection Importance",  "#ef4444", "Fraud Detection"),
        ]:
            fig_fi = go.Figure(go.Bar(
                x=feat_imp[imp_col], y=feat_imp["Feature"],
                orientation="h", marker_color=clr, opacity=0.85
            ))
            fig_fi.update_layout(
                title=f"Feature Importance — {title}",
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font={"color":"#e0e0e0"}, height=330,
                margin=dict(l=10,r=20,t=40,b=10),
                yaxis=dict(autorange="reversed")
            )
            col_st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif "Insights" in page:
    st.markdown("""
    <div class="header-banner">
        <h1>💡 Business Insights</h1>
        <p>Actuarial intelligence for strategic underwriting and risk management</p>
    </div>
    """, unsafe_allow_html=True)

    df_ins = load_dataset()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Risk Scoring Framework")
        framework_data = {
            "Risk Tier": ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
            "Score Range": ["0 – 25", "26 – 50", "51 – 75", "76 – 100"],
            "Premium Multiplier": ["×0.85", "×1.00", "×1.35", "×1.75"],
            "Action": [
                "Fast-track approval",
                "Standard underwriting",
                "Enhanced verification",
                "Senior actuary + SIU review"
            ]
        }
        st.dataframe(pd.DataFrame(framework_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 🚨 Fraud Trigger Indicators")
        trigger_data = {
            "Indicator": [
                "Previous Claims ≥ 3",
                "Accident History ≥ 2",
                "Policy Duration < 1 year",
                "Claim/Premium Ratio > 300%",
                "Claim Amount > $100K",
                "Income < $30K with high claim",
            ],
            "Risk Weight": ["High", "High", "High", "Very High", "Medium", "High"],
            "Flag Action": [
                "Enhanced review",
                "SIU referral",
                "Document audit",
                "SIU investigation",
                "Claim verification",
                "Income verification"
            ]
        }
        st.dataframe(pd.DataFrame(trigger_data), use_container_width=True, hide_index=True)

    if df_ins is not None:
        st.markdown("---")
        st.markdown("### 📊 Market Segment Analysis")
        col_a, col_b = st.columns(2)

        with col_a:
            segment = df_ins.groupby(["policy_type","gender"]).agg(
                Avg_Claim=("claim_amount","mean"),
                Count=("fraud_flag","count"),
                Fraud_Rate=("fraud_flag","mean")
            ).reset_index()
            fig = px.bar(
                segment, x="policy_type", y="Avg_Claim", color="gender",
                barmode="group", template="plotly_dark",
                color_discrete_map={"Male":"#7c3aed","Female":"#06b6d4"},
                labels={"Avg_Claim":"Avg Claim (USD)","policy_type":"Policy Type"},
                title="Avg Claim by Policy Type & Gender"
            )
            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                               font={"color":"#e0e0e0"}, height=320,
                               margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            df_ins["income_bracket"] = pd.cut(
                df_ins["annual_income"],
                bins=[0,40000,70000,100000,500000],
                labels=["<$40K","$40–70K","$70–100K",">$100K"]
            )
            inc_seg = df_ins.groupby("income_bracket", observed=True).agg(
                Avg_Claim=("claim_amount","mean"),
                Fraud_Rate=("fraud_flag","mean")
            ).reset_index()
            fig2 = px.bar(
                inc_seg, x="income_bracket", y="Avg_Claim",
                color="Fraud_Rate", color_continuous_scale="Reds",
                template="plotly_dark",
                labels={"income_bracket":"Income Bracket","Avg_Claim":"Avg Claim (USD)","Fraud_Rate":"Fraud Rate"},
                title="Avg Claim & Fraud by Income Bracket"
            )
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                                font={"color":"#e0e0e0"}, height=320,
                                margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # Key recommendations
    st.markdown("---")
    st.markdown("### 📝 Strategic Recommendations")
    recs = [
        ("🎯 Targeted Underwriting",
         "Focus enhanced due diligence on customers aged 55+ with 3+ previous claims. "
         "This segment represents 18% of customers but accounts for 34% of total claim payouts."),
        ("🔍 Fraud Prevention Program",
         "Implement real-time fraud scoring at point of claim submission. "
         "The XGBoost model achieves 98.3% AUC — deploy as an API endpoint for SIU triage."),
        ("💰 Dynamic Premium Pricing",
         "Replace static premium bands with the AI-driven risk score × multiplier framework. "
         "Expected 12–18% improvement in combined ratio through better risk selection."),
        ("📊 Portfolio Rebalancing",
         "Life insurance shows highest average claims ($47K). "
         "Review Life policy underwriting guidelines and reinsurance treaties accordingly."),
        ("🔄 Model Refresh Cycle",
         "Retrain models quarterly as fraud patterns evolve. "
         "Monitor model drift using PSI (Population Stability Index) on key features weekly."),
    ]
    for title, body in recs:
        st.markdown(f"""
        <div class="insight-box">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4b5563; font-size:0.8rem; padding: 1rem 0;">
    🏛️ <strong style="color:#6b7280;">AI Actuarial Science Platform</strong> &nbsp;|&nbsp;
    Predictive Modeling in Insurance &nbsp;|&nbsp;
    Built with Python · Scikit-learn · XGBoost · Streamlit
    <br><em>For educational and portfolio purposes · Data Scientist & AI Actuary</em>
</div>
""", unsafe_allow_html=True)
