"""
=============================================================================
AI ACTUARIAL SCIENCE — MODULE 01: SYNTHETIC DATASET GENERATION
=============================================================================
Generates a realistic 10,000-record insurance dataset with correlations
that mirror real-world actuarial data patterns.

Author  : Senior Data Scientist & AI Actuary
Version : 1.0.0
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
N_RECORDS = 10_000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance_dataset.csv")

# ---------------------------------------------------------------------------
# HELPER DISTRIBUTIONS
# ---------------------------------------------------------------------------
OCCUPATIONS = [
    "Engineer", "Doctor", "Teacher", "Lawyer", "Farmer",
    "Retail Worker", "Driver", "Self-Employed"
]

OCCUPATION_INCOME_MAP = {
    "Doctor":        (110_000, 30_000),
    "Lawyer":        (95_000,  25_000),
    "Engineer":      (85_000,  20_000),
    "Self-Employed": (70_000,  35_000),
    "Teacher":       (55_000,  12_000),
    "Retail Worker": (38_000,  10_000),
    "Farmer":        (42_000,  15_000),
    "Driver":        (36_000,   8_000),
}

POLICY_TYPES   = ["Health", "Auto", "Life", "Property"]
GENDERS        = ["Male", "Female"]

# ---------------------------------------------------------------------------
# STEP 1 — DEMOGRAPHICS
# ---------------------------------------------------------------------------
print("=" * 60)
print("  AI ACTUARIAL SCIENCE — Synthetic Data Generation")
print("=" * 60)
print(f"\n[1/7] Generating {N_RECORDS:,} customer records ...")

customer_ids = [f"CUST-{str(i).zfill(6)}" for i in range(1, N_RECORDS + 1)]

ages     = np.random.randint(18, 76, size=N_RECORDS)
genders  = np.random.choice(GENDERS, size=N_RECORDS, p=[0.54, 0.46])
occupations = np.random.choice(
    OCCUPATIONS, size=N_RECORDS,
    p=[0.18, 0.10, 0.14, 0.08, 0.10, 0.16, 0.12, 0.12]
)

annual_incomes = np.array([
    max(15_000, np.random.normal(loc=OCCUPATION_INCOME_MAP[occ][0],
                                  scale=OCCUPATION_INCOME_MAP[occ][1]))
    for occ in occupations
])

# ---------------------------------------------------------------------------
# STEP 2 — POLICY INFORMATION
# ---------------------------------------------------------------------------
print("[2/7] Assigning policy information ...")

policy_types     = np.random.choice(POLICY_TYPES, size=N_RECORDS, p=[0.35, 0.30, 0.20, 0.15])
policy_durations = np.random.randint(1, 21, size=N_RECORDS)           # 1–20 years

# ---------------------------------------------------------------------------
# STEP 3 — HEALTH & VEHICLE
# ---------------------------------------------------------------------------
print("[3/7] Computing health scores and vehicle ages ...")

# Health score inversely related to age + noise
health_scores = np.clip(
    100 - (ages * 0.5) + np.random.normal(0, 10, N_RECORDS),
    a_min=10, a_max=100
).round(2)

vehicle_ages = np.where(
    policy_types == "Auto",
    np.random.randint(0, 21, size=N_RECORDS),   # 0–20 years for Auto policies
    0                                             # 0 for non-Auto
)

# ---------------------------------------------------------------------------
# STEP 4 — CLAIMS HISTORY
# ---------------------------------------------------------------------------
print("[4/7] Simulating claims history and accident records ...")

# Previous claims: older customers + lower health → more claims
prev_claims_base = (ages / 25 + (100 - health_scores) / 30).astype(int)
previous_claims  = np.clip(prev_claims_base + np.random.randint(-1, 3, N_RECORDS), 0, 10)

# Accident history correlated with age and vehicle use
accident_history = np.clip(
    (vehicle_ages * 0.15 + previous_claims * 0.3 + np.random.poisson(0.5, N_RECORDS)).astype(int),
    0, 8
)

# ---------------------------------------------------------------------------
# STEP 5 — FRAUD FLAG (realistic ~12% fraud rate)
# ---------------------------------------------------------------------------
print("[5/7] Generating fraud flags with realistic patterns ...")

# Normalize each component to 0-1 range then combine with weights
prev_claims_norm   = np.clip(previous_claims  / 10, 0, 1)
accident_norm      = np.clip(accident_history / 8,  0, 1)
low_income_flag    = (annual_incomes < 40_000).astype(float)
short_policy_flag  = (policy_durations < 3).astype(float)
random_component   = np.random.uniform(0, 1, N_RECORDS)

fraud_score = (
    prev_claims_norm  * 0.35 +
    accident_norm     * 0.30 +
    low_income_flag   * 0.10 +
    short_policy_flag * 0.10 +
    random_component  * 0.15
)
# Use percentile to fix exactly ~13% fraud rate regardless of distribution
threshold = np.percentile(fraud_score, 87)        # top 13% = fraudulent
fraud_flag = (fraud_score >= threshold).astype(int)

# ---------------------------------------------------------------------------
# STEP 6 — CLAIM AMOUNTS & PREMIUM CALCULATION
# ---------------------------------------------------------------------------
print("[6/7] Calculating claim amounts and premiums ...")

# Base claim amount driven by multiple factors
base_claim = (
    annual_incomes * 0.08
    + previous_claims * 2_500
    + accident_history * 3_000
    + (100 - health_scores) * 150
    + vehicle_ages * 400
)

# Add policy-type multipliers
policy_multipliers = {"Health": 1.4, "Auto": 1.2, "Life": 1.6, "Property": 1.3}
policy_mult = np.array([policy_multipliers[p] for p in policy_types])

# Fraudsters inflate claims ~60%
fraud_multiplier = np.where(fraud_flag == 1, 1.6, 1.0)

claim_amounts = np.clip(
    base_claim * policy_mult * fraud_multiplier + np.random.normal(0, 2_000, N_RECORDS),
    a_min=500, a_max=250_000
).round(2)

# Premium calculation: actuarial base + risk loading
risk_loading = (
    previous_claims * 0.05 +
    accident_history * 0.08 +
    (100 - health_scores) * 0.003 +
    (ages / 100) * 0.10
)
premium_base = annual_incomes * 0.025 + claim_amounts * 0.04
premiums_paid = np.clip(
    premium_base * (1 + risk_loading) + np.random.normal(0, 200, N_RECORDS),
    a_min=300, a_max=25_000
).round(2)

# ---------------------------------------------------------------------------
# STEP 7 — TARGET: CLAIM FILED (binary)
# ---------------------------------------------------------------------------
print("[7/7] Assigning binary target (claim filed) ...")

claim_prob = np.clip(
    previous_claims * 0.08 +
    accident_history * 0.10 +
    (100 - health_scores) * 0.005 +
    fraud_flag * 0.15 +
    np.random.uniform(0, 0.3, N_RECORDS),
    0, 1
)
target_claim = (claim_prob > 0.40).astype(int)

# ---------------------------------------------------------------------------
# ASSEMBLE DATAFRAME
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "customer_id":      customer_ids,
    "age":              ages,
    "gender":           genders,
    "annual_income":    annual_incomes.round(2),
    "occupation":       occupations,
    "policy_type":      policy_types,
    "policy_duration":  policy_durations,
    "previous_claims":  previous_claims,
    "claim_amount":     claim_amounts,
    "vehicle_age":      vehicle_ages,
    "health_score":     health_scores,
    "accident_history": accident_history,
    "premium_paid":     premiums_paid,
    "fraud_flag":       fraud_flag,
    "target_claim":     target_claim,
})

# ---------------------------------------------------------------------------
# INJECT MISSING VALUES (realistic ~2–4% per column)
# ---------------------------------------------------------------------------
print("\n[INFO] Injecting realistic missing values (~2–4%) ...")
for col, frac in [("health_score", 0.03), ("vehicle_age", 0.02),
                   ("annual_income", 0.015), ("accident_history", 0.025)]:
    missing_idx = np.random.choice(df.index, size=int(N_RECORDS * frac), replace=False)
    df.loc[missing_idx, col] = np.nan

# ---------------------------------------------------------------------------
# INJECT DUPLICATES (realistic ~0.5%)
# ---------------------------------------------------------------------------
n_dups = int(N_RECORDS * 0.005)
dup_rows = df.sample(n_dups, random_state=7).copy()
df = pd.concat([df, dup_rows], ignore_index=True)
print(f"[INFO] Added {n_dups} duplicate rows (total rows: {len(df):,})")

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("  DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"  Saved to  : {OUTPUT_PATH}")
print(f"  Shape     : {df.shape}")
print(f"  Fraud Rate: {df['fraud_flag'].mean():.2%}")
print(f"  Claim Rate: {df['target_claim'].mean():.2%}")
print(f"  Missing % : {df.isnull().mean().mean():.2%}")
print("=" * 60)

# Quick validation
print("\nColumn dtypes:")
print(df.dtypes)
print("\nFirst 3 records:")
print(df.head(3).to_string())
