"""
Task 1 -- Generate Synthetic Loan Default Dataset
=================================================
Generates 5,000 realistic loan records with correlated default probabilities.
Missing values (~8%) are intentionally introduced for the cleaning step.
Output: data/loan_data_raw.csv
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
np.random.seed(42)
N = 5000

# Grade definitions
GRADES = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_WEIGHTS = [0.20, 0.22, 0.20, 0.16, 0.12, 0.06, 0.04]
GRADE_BASE_DEFAULT = {
    "A": 0.03, "B": 0.07, "C": 0.13,
    "D": 0.20, "E": 0.30, "F": 0.42, "G": 0.55,
}

# Feature generation
loan_grade = np.random.choice(GRADES, size=N, p=GRADE_WEIGHTS)
grade_idx  = np.array([GRADES.index(g) for g in loan_grade])

interest_rate = np.round(
    5 + grade_idx * 3.5 + np.random.normal(0, 1.5, N), 2
).clip(3.0, 32.0)

loan_amount = np.round(
    np.random.uniform(3000, 10000, N) + grade_idx * 2500 + np.random.normal(0, 2000, N), 2
).clip(1000, 45000)

annual_income = np.round(
    np.random.normal(70000 - grade_idx * 5000, 18000, N), 2
).clip(15000, 200000)

dti_ratio = np.round(
    0.10 + grade_idx * 0.05 + np.random.normal(0, 0.06, N), 4
).clip(0.01, 0.75)

loan_term = np.random.choice([36, 60], size=N, p=[0.60, 0.40])

employment_length = np.random.choice(range(0, 13), size=N,
    p=[0.08,0.10,0.10,0.09,0.09,0.08,0.08,0.07,0.07,0.07,0.06,0.06,0.05])

# Default status (target)
dti_boost      = (dti_ratio - 0.15).clip(0) * 0.5
interest_boost = (interest_rate - 10).clip(0) * 0.008
income_penalty = ((80000 - annual_income) / 80000).clip(0) * 0.05

p_default = np.array([GRADE_BASE_DEFAULT[g] for g in loan_grade])
p_default = (p_default + dti_boost + interest_boost + income_penalty).clip(0, 0.95)
default_status = (np.random.uniform(0, 1, N) < p_default).astype(int)

# Assemble DataFrame
df = pd.DataFrame({
    "loan_id":           range(1, N + 1),
    "loan_grade":        loan_grade,
    "loan_amount":       loan_amount,
    "interest_rate":     interest_rate,
    "dti_ratio":         dti_ratio,
    "annual_income":     annual_income,
    "loan_term":         loan_term,
    "employment_length": employment_length,
    "default_status":    default_status,
})

# Introduce missing values (~8-9%)
missing_mask_dti = np.random.choice([True, False], size=N, p=[0.08, 0.92])
missing_mask_emp = np.random.choice([True, False], size=N, p=[0.09, 0.91])
df.loc[missing_mask_dti, "dti_ratio"]         = None
df.loc[missing_mask_emp, "employment_length"] = None

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "loan_data_raw.csv")
df.to_csv(out_path, index=False)

# Summary
print("=" * 55)
print("  Dataset Generated Successfully")
print("=" * 55)
print(f"  Rows            : {len(df):,}")
print(f"  Columns         : {len(df.columns)}")
print(f"  Overall default : {df['default_status'].mean():.1%}")
print(f"  Missing dti     : {df['dti_ratio'].isna().sum()} rows")
print(f"  Missing emp_len : {df['employment_length'].isna().sum()} rows")
print(f"  Saved to        : {out_path}")
print("=" * 55)
print("\nDefault rate by grade:")
print(df.groupby("loan_grade")["default_status"].mean().map("{:.1%}".format).to_string())
