"""
Task 5 -- SHAP Explainability Analysis
========================================
Loads the trained XGBoost model and computes SHAP values to explain
which features drive individual and global default predictions.
Outputs:
  outputs/shap_summary.png      -- global feature importance (beeswarm)
  outputs/shap_bar.png          -- mean |SHAP| bar chart
  outputs/shap_waterfall.png    -- single-prediction waterfall (top risky loan)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap, joblib, os

BASE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(BASE, "outputs")
MDL   = os.path.join(BASE, "models")

# ─── Load model & data ────────────────────────────────────────────────────────
GRADE_ORDER = ["A","B","C","D","E","F","G"]
df = pd.read_csv(os.path.join(BASE, "data", "loan_data_clean.csv"))
df["loan_grade"] = pd.Categorical(df["loan_grade"], categories=GRADE_ORDER, ordered=True)
df["grade_code"] = df["loan_grade"].cat.codes

FEATURES = joblib.load(os.path.join(MDL, "feature_cols.pkl"))
xgb_model = joblib.load(os.path.join(MDL, "xgboost_model.pkl"))

X = df[FEATURES]

# ─── Compute SHAP values ──────────────────────────────────────────────────────
print("=" * 58)
print("  SHAP Explainability Analysis")
print("=" * 58)
print("  Computing SHAP values (TreeExplainer)...")

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer(X)

print(f"  SHAP matrix shape: {shap_values.values.shape}")

# Friendly display names
feature_names = {
    "loan_amount":       "Loan Amount",
    "interest_rate":     "Interest Rate",
    "dti_ratio":         "DTI Ratio",
    "annual_income":     "Annual Income",
    "loan_term":         "Loan Term",
    "employment_length": "Employment Length",
    "grade_code":        "Loan Grade",
}

# ─── Plot 1: Beeswarm Summary Plot ───────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.gcf().patch.set_facecolor("#0f1117")

shap.summary_plot(
    shap_values.values, X,
    feature_names=[feature_names[f] for f in FEATURES],
    show=False,
    plot_size=None,
    color_bar=True,
)

ax = plt.gca()
ax.set_facecolor("#0f1117")
ax.set_title("SHAP Feature Impact (Beeswarm)", color="white",
             fontsize=14, fontweight="bold", pad=12)
ax.tick_params(colors="white")
ax.set_xlabel("SHAP Value (impact on default probability)", color="white", fontsize=11)
for spine in ax.spines.values(): spine.set_edgecolor("#444")
for label in ax.get_yticklabels(): label.set_color("white")

plt.tight_layout()
summary_path = os.path.join(OUT, "shap_summary.png")
plt.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print(f"  Saved: {summary_path}")

# ─── Plot 2: Mean |SHAP| Bar Chart ───────────────────────────────────────────
mean_shap = np.abs(shap_values.values).mean(axis=0)
sorted_idx = np.argsort(mean_shap)
labels = [feature_names[FEATURES[i]] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(labels)))
bars = ax.barh(labels, mean_shap[sorted_idx], color=colors, edgecolor="#222", linewidth=0.5)

for bar, val in zip(bars, mean_shap[sorted_idx]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", ha="left", color="white", fontsize=10)

ax.set_xlabel("Mean |SHAP Value|", color="white", fontsize=12)
ax.set_title("Global Feature Importance (Mean |SHAP|)", color="white",
             fontsize=14, fontweight="bold", pad=12)
ax.tick_params(colors="white", labelsize=11)
for spine in ax.spines.values(): spine.set_edgecolor("#444")
ax.grid(axis="x", color="#333", linestyle="--", linewidth=0.5)

plt.tight_layout()
bar_path = os.path.join(OUT, "shap_bar.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {bar_path}")

# ─── Plot 3: Waterfall — highest-risk individual loan ────────────────────────
proba = xgb_model.predict_proba(X)[:, 1]
riskiest_idx = int(np.argmax(proba))

print(f"\n  Riskiest loan index: {riskiest_idx}  |  P(default)={proba[riskiest_idx]:.3f}")
print(f"  Features:")
for feat in FEATURES:
    print(f"    {feature_names[feat]:<22}: {X.iloc[riskiest_idx][feat]}")

# Build Explanation object with renamed features
exp = shap.Explanation(
    values      = shap_values.values[riskiest_idx],
    base_values = shap_values.base_values[riskiest_idx],
    data        = X.iloc[riskiest_idx].values,
    feature_names=[feature_names[f] for f in FEATURES],
)

plt.figure(figsize=(10, 5))
plt.gcf().patch.set_facecolor("#0f1117")
shap.plots.waterfall(exp, show=False)
ax = plt.gca()
ax.set_facecolor("#0f1117")
ax.set_title(f"SHAP Waterfall — Riskiest Loan (P(default)={proba[riskiest_idx]:.1%})",
             color="white", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(colors="white")
ax.set_xlabel(ax.get_xlabel(), color="white")
for spine in ax.spines.values(): spine.set_edgecolor("#444")
for label in ax.get_yticklabels(): label.set_color("white")
for label in ax.get_xticklabels(): label.set_color("white")

plt.tight_layout()
wf_path = os.path.join(OUT, "shap_waterfall.png")
plt.savefig(wf_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print(f"  Saved: {wf_path}")

# ─── Global ranking summary ───────────────────────────────────────────────────
print("\n  Global feature ranking by mean |SHAP|:")
for i in sorted_idx[::-1]:
    print(f"    {feature_names[FEATURES[i]]:<22}: {mean_shap[i]:.4f}")

print("\n" + "=" * 58)
print("  SHAP analysis complete.")
print("=" * 58)
