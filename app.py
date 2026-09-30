"""
Streamlit Dashboard -- Financial Risk & Loan Default Analysis
==============================================================
Run with:  streamlit run app.py
Tabs:
  1. Overview       -- dataset stats & grade distribution
  2. EDA Charts     -- all generated visualizations
  3. Risk Predictor -- live logistic reg + XGBoost probability
  4. SHAP Explainer -- global importance + on-demand waterfall
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap, joblib, os, subprocess, sys

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs")
MDL  = os.path.join(BASE, "models")

# ─── Auto-setup (runs on Streamlit Cloud first boot) ─────────────────────────
def _run(script):
    subprocess.run([sys.executable, os.path.join(BASE, script)], check=True)

_needs_data   = not os.path.exists(os.path.join(BASE, "data", "loan_data_clean.csv"))
_needs_models = not os.path.exists(os.path.join(MDL,  "xgboost_model.pkl"))
_needs_charts = not os.path.exists(os.path.join(OUT,  "shap_bar.png"))

if _needs_data or _needs_models or _needs_charts:
    with st.spinner("⚙️ First-time setup: generating data & training models (≈60s)…"):
        if _needs_data:
            _run("01_generate_data.py")
            _run("02_clean_data.py")
        if _needs_models or _needs_charts:
            _run("03_eda_analysis.py")
            _run("04_model.py")
            _run("05_shap_analysis.py")
    st.success("✅ Setup complete! Reloading…")
    st.rerun()


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #0f1923 100%); }

.metric-card {
    background: linear-gradient(135deg, #1a2332, #1e2d40);
    border: 1px solid #2d4a6b;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,100,200,0.2); }
.metric-title { color: #7fb3d3; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }
.metric-value { color: #e8f4fd; font-size: 2rem; font-weight: 700; }
.metric-sub   { color: #5a8fa8; font-size: 0.72rem; margin-top: 0.3rem; }

.risk-badge-low    { background:#1b4332; color:#52b788; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.risk-badge-medium { background:#3d2b0a; color:#f4a261; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.risk-badge-high   { background:#3d0c0c; color:#e63946; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }

div[data-testid="stTabs"] button { font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Load resources ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    GRADE_ORDER = ["A","B","C","D","E","F","G"]
    df = pd.read_csv(os.path.join(BASE, "data", "loan_data_clean.csv"))
    df["loan_grade"] = pd.Categorical(df["loan_grade"], categories=GRADE_ORDER, ordered=True)
    df["grade_code"] = df["loan_grade"].cat.codes
    return df

@st.cache_resource
def load_models():
    lr  = joblib.load(os.path.join(MDL, "logistic_model.pkl"))
    xgb = joblib.load(os.path.join(MDL, "xgboost_model.pkl"))
    features = joblib.load(os.path.join(MDL, "feature_cols.pkl"))
    return lr, xgb, features

@st.cache_resource
def load_shap_explainer():
    _, xgb, _ = load_models()
    return shap.TreeExplainer(xgb)

df       = load_data()
lr, xgb, FEATURES = load_models()

FEATURE_LABELS = {
    "loan_amount":       "Loan Amount ($)",
    "interest_rate":     "Interest Rate (%)",
    "dti_ratio":         "DTI Ratio",
    "annual_income":     "Annual Income ($)",
    "loan_term":         "Loan Term (months)",
    "employment_length": "Employment Length (yrs)",
    "grade_code":        "Loan Grade",
}
GRADE_ORDER = ["A","B","C","D","E","F","G"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Loan Risk Dashboard")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"5,000 loan records | 9 features")
    st.markdown("**Models trained**")
    st.caption("✅ Logistic Regression")
    st.caption("✅ XGBoost (gradient boosting)")
    st.markdown("**Explainability**")
    st.caption("✅ SHAP TreeExplainer")
    st.markdown("---")
    grade_filter = st.multiselect(
        "Filter by Grade (Overview)",
        options=GRADE_ORDER, default=GRADE_ORDER
    )

filtered_df = df[df["loan_grade"].isin(grade_filter)]

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 EDA Charts", "🎯 Risk Predictor", "🔍 SHAP Explainer"])

# ════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("# 🏦 Financial Risk Overview")
    st.markdown("Real-time summary of the loan portfolio based on selected grades.")
    st.markdown("---")

    total    = len(filtered_df)
    defaults = filtered_df["default_status"].sum()
    rate     = defaults / total if total else 0
    avg_ir   = filtered_df["interest_rate"].mean()
    avg_dti  = filtered_df["dti_ratio"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, title, value, sub in [
        (c1, "Total Loans",    f"{total:,}",          "in selected grades"),
        (c2, "Defaults",       f"{defaults:,}",        f"{rate:.1%} of total"),
        (c3, "Avg Interest",   f"{avg_ir:.1f}%",       "annual rate"),
        (c4, "Avg DTI",        f"{avg_dti:.3f}",       "debt-to-income"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown("### Default Rate by Grade")
        grade_stats = (
            filtered_df.groupby("loan_grade", observed=True)["default_status"]
            .agg(count="count", defaults="sum", rate="mean")
            .reset_index()
        )
        grade_stats["rate_pct"] = (grade_stats["rate"]*100).round(1)
        st.dataframe(
            grade_stats[["loan_grade","count","defaults","rate_pct"]]
            .rename(columns={"loan_grade":"Grade","count":"Loans",
                             "defaults":"Defaults","rate_pct":"Default Rate (%)"}),
            width='stretch', hide_index=True
        )

    with col_r:
        st.markdown("### Income vs Default Status")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        for status, color, label in [(0,"#4fc3f7","No Default"),(1,"#ef5350","Default")]:
            vals = filtered_df[filtered_df["default_status"]==status]["annual_income"]/1000
            ax.hist(vals, bins=30, alpha=0.65, color=color, label=label, edgecolor="none")
        ax.set_xlabel("Annual Income ($k)", color="white", fontsize=9)
        ax.set_ylabel("Count", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.4)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

    st.markdown("### Raw Data Sample")
    st.dataframe(filtered_df.head(50), width='stretch', height=280)

# ════════════════════════════════════════════════════════
# TAB 2 — EDA CHARTS
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("# 📈 Exploratory Data Analysis")
    st.markdown("---")

    img_c1, img_c2 = st.columns(2)
    with img_c1:
        st.markdown("### Default Rate by Loan Grade")
        st.image(os.path.join(OUT, "default_rate_by_grade.png"), width='stretch')
    with img_c2:
        st.markdown("### Feature Correlation Heatmap")
        st.image(os.path.join(OUT, "eda_heatmap.png"), width='stretch')

    st.markdown("---")
    img_c3, img_c4 = st.columns(2)
    with img_c3:
        st.markdown("### Model ROC Curves")
        st.image(os.path.join(OUT, "roc_curve.png"), width='stretch')
    with img_c4:
        st.markdown("### Confusion Matrices")
        st.image(os.path.join(OUT, "confusion_matrices.png"), width='stretch')

# ════════════════════════════════════════════════════════
# TAB 3 — RISK PREDICTOR
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("# 🎯 Live Default Risk Predictor")
    st.markdown("Adjust the sliders below to simulate a borrower profile and get real-time default probabilities from both models.")
    st.markdown("---")

    col_in, col_out = st.columns([1.1, 1])

    with col_in:
        st.markdown("### Borrower Profile")
        grade_sel   = st.selectbox("Loan Grade", GRADE_ORDER, index=2)
        loan_amt    = st.slider("Loan Amount ($)",     1000,  45000, 12000, step=500)
        int_rate    = st.slider("Interest Rate (%)",    3.0,   32.0, 12.0,  step=0.1)
        dti         = st.slider("DTI Ratio",           0.01,   0.75, 0.20,  step=0.01)
        annual_inc  = st.slider("Annual Income ($)",  15000, 200000, 60000, step=1000)
        term        = st.selectbox("Loan Term (months)", [36, 60], index=0)
        emp_len     = st.slider("Employment Length (yrs)", 0, 12, 3)

    grade_code_val = GRADE_ORDER.index(grade_sel)
    input_row = pd.DataFrame([[loan_amt, int_rate, dti, annual_inc,
                                term, emp_len, grade_code_val]], columns=FEATURES)

    lr_prob  = lr.predict_proba(input_row)[0, 1]
    xgb_prob = xgb.predict_proba(input_row)[0, 1]
    avg_prob = (lr_prob + xgb_prob) / 2

    if avg_prob < 0.25:
        badge = '<span class="risk-badge-low">🟢 LOW RISK</span>'
        verdict = "This borrower profile presents a **low default risk**. Standard lending terms recommended."
    elif avg_prob < 0.55:
        badge = '<span class="risk-badge-medium">🟡 MEDIUM RISK</span>'
        verdict = "This borrower presents **moderate risk**. Consider higher interest rate or collateral requirement."
    else:
        badge = '<span class="risk-badge-high">🔴 HIGH RISK</span>'
        verdict = "This borrower presents **high default risk**. Loan approval not recommended under standard terms."

    with col_out:
        st.markdown("### Prediction Results")
        st.markdown(badge, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Logistic Reg", f"{lr_prob:.1%}")
        m2.metric("XGBoost",      f"{xgb_prob:.1%}")
        m3.metric("Ensemble Avg", f"{avg_prob:.1%}")

        st.markdown("---")

        # Gauge-style bar
        fig, ax = plt.subplots(figsize=(5, 1.2))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        ax.barh(["Risk"], [1], color="#1f2937", height=0.5)
        gauge_color = "#52b788" if avg_prob < 0.25 else "#f4a261" if avg_prob < 0.55 else "#e63946"
        ax.barh(["Risk"], [avg_prob], color=gauge_color, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticklabels(["0%","25%","50%","75%","100%"], color="white", fontsize=9)
        ax.tick_params(left=False, labelleft=False, colors="white")
        ax.set_title(f"Default Probability: {avg_prob:.1%}", color="white", fontsize=11, pad=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

        st.info(verdict)

# ════════════════════════════════════════════════════════
# TAB 4 — SHAP EXPLAINER
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("# 🔍 SHAP Explainability")
    st.markdown("SHAP (SHapley Additive exPlanations) reveals which features push the model toward or away from predicting default.")
    st.markdown("---")

    sh_c1, sh_c2 = st.columns(2)
    with sh_c1:
        st.markdown("### Global Feature Importance")
        st.image(os.path.join(OUT, "shap_bar.png"), width='stretch')
    with sh_c2:
        st.markdown("### SHAP Beeswarm (All Loans)")
        st.image(os.path.join(OUT, "shap_summary.png"), width='stretch')

    st.markdown("---")
    st.markdown("### 🔎 Riskiest Loan — Waterfall Explanation")
    st.caption("This shows exactly WHY the model predicted high default risk for the single riskiest borrower in the dataset.")
    st.image(os.path.join(OUT, "shap_waterfall.png"), width='stretch')

    st.markdown("---")
    st.markdown("### On-Demand SHAP for Custom Profile")
    st.caption("Uses the profile from the Risk Predictor tab. Switch over, adjust sliders, then come back here.")

    if st.button("⚡ Compute SHAP for Current Profile", type="primary"):
        with st.spinner("Running TreeExplainer..."):
            grade_sel_s  = st.session_state.get("grade_sel", "C")
            loan_amt_s   = st.session_state.get("loan_amt",  12000)
            int_rate_s   = st.session_state.get("int_rate",  12.0)
            dti_s        = st.session_state.get("dti",       0.20)
            annual_inc_s = st.session_state.get("annual_inc",60000)
            term_s       = st.session_state.get("term",      36)
            emp_len_s    = st.session_state.get("emp_len",   3)
            gc_s         = GRADE_ORDER.index(grade_sel)

            row = pd.DataFrame([[loan_amt, int_rate, dti, annual_inc,
                                  term, emp_len, grade_code_val]], columns=FEATURES)
            explainer   = load_shap_explainer()
            sv          = explainer(row)
            exp_obj     = shap.Explanation(
                values       = sv.values[0],
                base_values  = sv.base_values[0],
                data         = row.iloc[0].values,
                feature_names=[FEATURE_LABELS[f] for f in FEATURES],
            )
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fig2.patch.set_facecolor("#111827")
            shap.plots.waterfall(exp_obj, show=False)
            ax2 = plt.gca()
            ax2.set_facecolor("#111827")
            ax2.tick_params(colors="white")
            ax2.set_title("SHAP Waterfall — Your Custom Profile",
                          color="white", fontsize=12, fontweight="bold")
            for spine in ax2.spines.values(): spine.set_edgecolor("#333")
            plt.tight_layout()
            st.pyplot(fig2, width='stretch')

