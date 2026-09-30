"""
Task 2 -- Data Cleaning
========================
Loads loan_data_raw.csv, handles missing values, enforces types,
removes duplicates, flags outliers, and saves a clean dataset.
Output: data/loan_data_clean.csv
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
raw_path   = os.path.join(BASE, "data", "loan_data_raw.csv")
clean_path = os.path.join(BASE, "data", "loan_data_clean.csv")

# Load
df = pd.read_csv(raw_path)
initial_rows = len(df)
print("=" * 55)
print("  Data Cleaning Report")
print("=" * 55)
print(f"  Loaded : {initial_rows:,} rows, {len(df.columns)} columns")

# --- 1. Missing value imputation ---
dti_missing  = df["dti_ratio"].isna().sum()
emp_missing  = df["employment_length"].isna().sum()

dti_median   = df["dti_ratio"].median()
emp_mode     = int(df["employment_length"].mode()[0])

df["dti_ratio"]         = df["dti_ratio"].fillna(dti_median)
df["employment_length"] = df["employment_length"].fillna(emp_mode)

print(f"\n  [Imputation]")
print(f"    dti_ratio        : {dti_missing} nulls filled with median ({dti_median:.4f})")
print(f"    employment_length: {emp_missing} nulls filled with mode  ({emp_mode})")

# --- 2. Type enforcement ---
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
df["loan_grade"]        = pd.Categorical(df["loan_grade"], categories=GRADE_ORDER, ordered=True)
df["default_status"]    = df["default_status"].astype(int)
df["loan_term"]         = df["loan_term"].astype(int)
df["employment_length"] = df["employment_length"].astype(int)
print(f"\n  [Types enforced]")
print(f"    loan_grade     -> ordered Categorical (A < B < ... < G)")
print(f"    default_status -> int")
print(f"    loan_term      -> int")
print(f"    employment_length -> int")

# --- 3. Remove duplicates ---
dupes = df.duplicated(subset="loan_id").sum()
df = df.drop_duplicates(subset="loan_id")
print(f"\n  [Duplicates] Removed: {dupes}")

# --- 4. Outlier detection (flag only, do not drop) ---
outlier_mask = (df["annual_income"] < 5000) | (df["loan_amount"] > 200000)
df["is_outlier"] = outlier_mask.astype(int)
print(f"\n  [Outliers] Flagged: {outlier_mask.sum()} rows (kept, marked in 'is_outlier')")

# --- 5. Final null check ---
remaining_nulls = df.isnull().sum().sum()
print(f"\n  [Null Check] Remaining nulls: {remaining_nulls}")

# --- 6. Save ---
df.to_csv(clean_path, index=False)
print(f"\n  Final rows : {len(df):,}")
print(f"  Saved to   : {clean_path}")
print("=" * 55)
print("\nColumn dtypes after cleaning:")
print(df.dtypes.to_string())
print("\nBasic statistics:")
print(df[["loan_amount","interest_rate","dti_ratio","annual_income","default_status"]].describe().round(2).to_string())
