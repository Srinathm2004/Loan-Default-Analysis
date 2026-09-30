# 📊 Financial Risk & Loan Default Analysis

An end-to-end machine learning project for **loan default prediction** with interactive Streamlit dashboard, SHAP explainability, and full EDA.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 🚀 Live Demo

Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) for free — see deployment instructions below.

---

## 🗂️ Project Structure

```
├── app.py                   # Streamlit dashboard (4 tabs)
├── requirements.txt         # Python dependencies
├── 01_generate_data.py      # Synthetic loan dataset generation
├── 02_clean_data.py         # Data cleaning & imputation
├── 03_eda_analysis.py       # EDA + correlation heatmap + report
├── 04_model.py              # Logistic Regression + XGBoost training
├── 05_shap_analysis.py      # SHAP explainability (beeswarm, waterfall)
├── data/
│   ├── loan_data_raw.csv    # Raw dataset (5,000 rows, with nulls)
│   └── loan_data_clean.csv  # Cleaned & typed dataset
├── models/
│   ├── logistic_model.pkl   # Trained Logistic Regression pipeline
│   ├── xgboost_model.pkl    # Trained XGBoost model
│   └── feature_cols.pkl     # Feature column list
└── outputs/
    ├── default_rate_by_grade.png
    ├── eda_heatmap.png
    ├── roc_curve.png
    ├── confusion_matrices.png
    ├── shap_summary.png
    ├── shap_bar.png
    ├── shap_waterfall.png
    └── report_summary.md    # Auto-generated risk report
```

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🏠 **Overview** | KPI cards, grade table, income histogram, raw data explorer |
| 📈 **EDA Charts** | Default rate by grade, correlation heatmap, ROC curves, confusion matrices |
| 🎯 **Risk Predictor** | Live sliders → real-time default probability from both models |
| 🔍 **SHAP Explainer** | Global feature importance, beeswarm plot, per-prediction waterfall |

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 5,000 loans |
| Overall default rate | 24.7% |
| Logistic Regression AUC | 0.787 |
| XGBoost AUC | 0.778 |
| Top SHAP feature | Loan Grade (0.698) |
| 2nd SHAP feature | Interest Rate (0.596) |

---

## 🛠️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-default-analysis.git
cd loan-default-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

> **Note:** On first launch, if data/models are missing, they are auto-generated (~60 seconds).

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and set **Main file path** to `app.py`
4. Click **Deploy** — done! First boot auto-generates data & trains models.

---

## 🔍 Features

- **Synthetic dataset** with realistic correlated default probabilities
- **Data cleaning pipeline** with median/mode imputation and outlier flagging
- **EDA** with correlation heatmap and default-rate analysis by loan grade
- **Two ML models**: Logistic Regression (interpretable) + XGBoost (performance)
- **SHAP explainability**: global feature importance + individual loan waterfall
- **Interactive dashboard**: filter by grade, simulate borrower profiles, get instant predictions

---

## 📦 Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `Streamlit` · `matplotlib` · `seaborn`
