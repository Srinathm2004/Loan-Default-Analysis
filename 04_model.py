"""
Task 4 -- Model Training: Logistic Regression + XGBoost
=========================================================
Trains both models on the clean dataset, evaluates with:
  - Classification report (precision, recall, F1)
  - ROC-AUC score
  - Confusion matrix plot
Saves trained models to models/ for dashboard and SHAP use.
Outputs:
  outputs/roc_curve.png
  outputs/confusion_matrices.png
  models/logistic_model.pkl
  models/xgboost_model.pkl
  models/feature_cols.pkl
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (classification_report, roc_auc_score,
                                        RocCurveDisplay, confusion_matrix,
                                        ConfusionMatrixDisplay)
from xgboost                   import XGBClassifier

BASE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(BASE, "outputs")
MDL    = os.path.join(BASE, "models")
os.makedirs(MDL, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ─── Load & feature-encode ────────────────────────────────────────────────────
GRADE_ORDER = ["A","B","C","D","E","F","G"]
df = pd.read_csv(os.path.join(BASE, "data", "loan_data_clean.csv"))
df["loan_grade"] = pd.Categorical(df["loan_grade"], categories=GRADE_ORDER, ordered=True)
df["grade_code"] = df["loan_grade"].cat.codes          # A=0 … G=6

FEATURES = ["loan_amount","interest_rate","dti_ratio",
            "annual_income","loan_term","employment_length","grade_code"]
TARGET   = "default_status"

X = df[FEATURES]
y = df[TARGET]

# Save feature list for dashboard use
joblib.dump(FEATURES, os.path.join(MDL, "feature_cols.pkl"))

# ─── Train / test split (stratified) ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("=" * 62)
print("  Model Training Report")
print("=" * 62)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Default rate  train: {y_train.mean():.2%}")
print(f"  Default rate  test : {y_test.mean():.2%}")

# ─── Model 1: Logistic Regression ────────────────────────────────────────────
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
lr_pred  = lr_pipe.predict(X_test)
lr_auc   = roc_auc_score(y_test, lr_proba)

# 5-fold CV AUC
lr_cv = cross_val_score(lr_pipe, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

print(f"\n  [Logistic Regression]")
print(f"    Test ROC-AUC : {lr_auc:.4f}")
print(f"    CV  ROC-AUC  : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"\n{classification_report(y_test, lr_pred, target_names=['No Default','Default'])}")

joblib.dump(lr_pipe, os.path.join(MDL, "logistic_model.pkl"))

# ─── Model 2: XGBoost ────────────────────────────────────────────────────────
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred  = xgb_model.predict(X_test)
xgb_auc   = roc_auc_score(y_test, xgb_proba)

xgb_cv = cross_val_score(xgb_model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

print(f"  [XGBoost]")
print(f"    Test ROC-AUC : {xgb_auc:.4f}")
print(f"    CV  ROC-AUC  : {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
print(f"\n{classification_report(y_test, xgb_pred, target_names=['No Default','Default'])}")

joblib.dump(xgb_model, os.path.join(MDL, "xgboost_model.pkl"))

# ─── Plot 1: Side-by-side ROC curves ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

RocCurveDisplay.from_predictions(y_test, lr_proba,  name=f"Logistic Reg  (AUC={lr_auc:.3f})",  ax=ax).line_.set_color("#4fc3f7")
RocCurveDisplay.from_predictions(y_test, xgb_proba, name=f"XGBoost       (AUC={xgb_auc:.3f})", ax=ax).line_.set_color("#ff7043")
ax.plot([0,1],[0,1], "w--", linewidth=0.8, label="Random baseline")

ax.set_title("ROC Curves — Loan Default Prediction", color="white", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
ax.set_ylabel("True Positive Rate",  color="white", fontsize=12)
ax.tick_params(colors="white")
for spine in ax.spines.values(): spine.set_edgecolor("#444")
ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)
ax.grid(color="#333", linestyle="--", linewidth=0.5)

plt.tight_layout()
roc_path = os.path.join(OUT, "roc_curve.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {roc_path}")

# ─── Plot 2: Confusion matrices ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0f1117")

for ax, pred, proba, title in [
    (axes[0], lr_pred,  lr_proba,  "Logistic Regression"),
    (axes[1], xgb_pred, xgb_proba, "XGBoost"),
]:
    ax.set_facecolor("#0f1117")
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default","Default"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.set_xlabel(ax.get_xlabel(), color="white")
    ax.set_ylabel(ax.get_ylabel(), color="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    for text in ax.texts: text.set_color("white")

fig.suptitle("Confusion Matrices", color="white", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
cm_path = os.path.join(OUT, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {cm_path}")

print("\n" + "=" * 62)
print(f"  Models saved to: {MDL}")
print(f"  Best model: {'XGBoost' if xgb_auc > lr_auc else 'Logistic Regression'} (AUC {max(xgb_auc,lr_auc):.4f})")
print("=" * 62)
