"""
=============================================================================
AI ACTUARIAL SCIENCE — MODULE 02: EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
Professional visualizations covering distributions, correlations, and
cross-segment analysis for actuarial decision-making.

Author  : Senior Data Scientist & AI Actuary
Version : 1.0.0
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# STYLE CONFIGURATION
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4a",
    "axes.labelcolor": "#e0e0e0",
    "axes.titlecolor": "#ffffff",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "text.color": "#e0e0e0",
    "grid.color": "#2a2d3a",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4a",
    "font.family": "DejaVu Sans",
})

ACCENT_PALETTE = ["#7c3aed", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
GRADIENT_BLUE  = sns.color_palette("Blues_d", 6)
SAVE_DIR       = os.path.join(os.path.dirname(__file__), "..", "reports", "visualizations")

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance_dataset.csv")

print("=" * 65)
print("  AI ACTUARIAL SCIENCE — Exploratory Data Analysis")
print("=" * 65)
print(f"\n[INFO] Loading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Shape before cleaning: {df.shape}")

# ---------------------------------------------------------------------------
# QUICK PREPROCESSING FOR EDA
# ---------------------------------------------------------------------------
df_clean = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
df_clean = df_clean.copy()

# Fill missing numerics with median for EDA
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

print(f"[INFO] Shape after dedup & imputation: {df_clean.shape}")
print(f"[INFO] Fraud rate  : {df_clean['fraud_flag'].mean():.2%}")
print(f"[INFO] Claim rate  : {df_clean['target_claim'].mean():.2%}")

# ---------------------------------------------------------------------------
# PLOT 1 — CORRELATION HEATMAP
# ---------------------------------------------------------------------------
print("\n[1/6] Generating Correlation Heatmap ...")

numeric_cols = [
    "age", "annual_income", "policy_duration", "previous_claims",
    "claim_amount", "vehicle_age", "health_score", "accident_history",
    "premium_paid", "fraud_flag", "target_claim"
]
corr_matrix = df_clean[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor("#0f1117")

mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True  # upper triangle only

cmap = sns.diverging_palette(250, 20, sep=20, as_cmap=True)
sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap, center=0,
    annot=True, fmt=".2f", linewidths=0.5,
    linecolor="#0f1117", square=True, ax=ax,
    annot_kws={"size": 8, "color": "white"},
    cbar_kws={"shrink": 0.8}
)

ax.set_title("📊 Feature Correlation Heatmap — Insurance Dataset", pad=18, fontsize=14,
             fontweight="bold", color="white")
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)

plt.tight_layout()
out = os.path.join(SAVE_DIR, "01_correlation_heatmap.png")
plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Saved -> {out}")

# ---------------------------------------------------------------------------
# PLOT 2 — CLAIM AMOUNT DISTRIBUTION
# ---------------------------------------------------------------------------
print("[2/6] Generating Claim Amount Distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("💰 Insurance Claim Amount Distribution", fontsize=14,
             fontweight="bold", color="white", y=1.02)

for ax in axes:
    ax.set_facecolor("#1a1d27")

# Raw distribution
axes[0].hist(df_clean["claim_amount"], bins=60, color="#7c3aed", alpha=0.85,
             edgecolor="#9d5cf6", linewidth=0.5)
axes[0].set_title("Raw Claim Distribution", color="white")
axes[0].set_xlabel("Claim Amount (USD)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df_clean["claim_amount"].mean(), color="#f59e0b", linestyle="--",
                linewidth=2, label=f"Mean: ${df_clean['claim_amount'].mean():,.0f}")
axes[0].axvline(df_clean["claim_amount"].median(), color="#10b981", linestyle="--",
                linewidth=2, label=f"Median: ${df_clean['claim_amount'].median():,.0f}")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Log-transformed for actuarial modelling
log_claims = np.log1p(df_clean["claim_amount"])
axes[1].hist(log_claims, bins=60, color="#06b6d4", alpha=0.85,
             edgecolor="#38bdf8", linewidth=0.5)
axes[1].set_title("Log-Transformed Claim Distribution", color="white")
axes[1].set_xlabel("log(1 + Claim Amount)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(SAVE_DIR, "02_claim_distribution.png")
plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Saved -> {out}")

# ---------------------------------------------------------------------------
# PLOT 3 — FRAUD DISTRIBUTION
# ---------------------------------------------------------------------------
print("[3/6] Generating Fraud Distribution Analysis ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("🔍 Fraud Distribution Analysis", fontsize=14,
             fontweight="bold", color="white", y=1.02)

colors_fraud = ["#10b981", "#ef4444"]

for ax in axes:
    ax.set_facecolor("#1a1d27")

# Pie chart
fraud_counts = df_clean["fraud_flag"].value_counts()
explode = (0, 0.07)
wedges, texts, autotexts = axes[0].pie(
    fraud_counts, labels=["Genuine", "Fraudulent"],
    colors=colors_fraud, autopct="%1.1f%%", explode=explode,
    startangle=90, pctdistance=0.80,
    wedgeprops={"edgecolor": "#0f1117", "linewidth": 2}
)
for t in texts:   t.set_color("white")
for at in autotexts: at.set_color("white"); at.set_fontsize(11)
axes[0].set_title("Fraud vs Genuine Ratio")

# Fraud by policy type
fraud_by_policy = df_clean.groupby("policy_type")["fraud_flag"].mean().sort_values(ascending=False)
bars = axes[1].bar(fraud_by_policy.index, fraud_by_policy.values * 100,
                    color=ACCENT_PALETTE[:4], edgecolor="#0f1117", linewidth=0.8)
axes[1].set_title("Fraud Rate by Policy Type")
axes[1].set_xlabel("Policy Type")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].grid(axis="y", alpha=0.3)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 fontsize=9, color="white")

# Fraud by previous claims
fraud_by_claims = df_clean.groupby("previous_claims")["fraud_flag"].mean()
axes[2].plot(fraud_by_claims.index, fraud_by_claims.values * 100,
             color="#f59e0b", linewidth=2.5, marker="o", markersize=6,
             markerfacecolor="#ef4444", markeredgecolor="white", markeredgewidth=1.5)
axes[2].fill_between(fraud_by_claims.index, fraud_by_claims.values * 100,
                      alpha=0.15, color="#f59e0b")
axes[2].set_title("Fraud Rate vs Previous Claims")
axes[2].set_xlabel("Number of Previous Claims")
axes[2].set_ylabel("Fraud Rate (%)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(SAVE_DIR, "03_fraud_distribution.png")
plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Saved -> {out}")

# ---------------------------------------------------------------------------
# PLOT 4 — AGE VS CLAIM ANALYSIS
# ---------------------------------------------------------------------------
print("[4/6] Generating Age vs Claim Analysis ...")

df_clean["age_group"] = pd.cut(
    df_clean["age"],
    bins=[18, 30, 40, 50, 60, 76],
    labels=["18–30", "31–40", "41–50", "51–60", "61–75"]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("👤 Age vs Insurance Claim Analysis", fontsize=14,
             fontweight="bold", color="white", y=1.02)

for ax in axes:
    ax.set_facecolor("#1a1d27")

# Box plot by age group
age_groups = df_clean["age_group"].cat.categories.tolist()
data_by_age = [df_clean[df_clean["age_group"] == g]["claim_amount"].values for g in age_groups]

bp = axes[0].boxplot(data_by_age, patch_artist=True, notch=False,
                      labels=age_groups, whiskerprops={"color": "#aaaaaa"},
                      capprops={"color": "#aaaaaa"}, medianprops={"color": "#f59e0b", "linewidth": 2})
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(ACCENT_PALETTE[i % len(ACCENT_PALETTE)])
    patch.set_alpha(0.75)

axes[0].set_title("Claim Amount Distribution by Age Group")
axes[0].set_xlabel("Age Group")
axes[0].set_ylabel("Claim Amount (USD)")
axes[0].grid(axis="y", alpha=0.3)

# Avg claim by age group
avg_claim_age = df_clean.groupby("age_group")["claim_amount"].mean()
bars = axes[1].bar(avg_claim_age.index, avg_claim_age.values,
                    color=ACCENT_PALETTE[:5], edgecolor="#0f1117")
axes[1].set_title("Average Claim Amount by Age Group")
axes[1].set_xlabel("Age Group")
axes[1].set_ylabel("Avg Claim Amount (USD)")
axes[1].grid(axis="y", alpha=0.3)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f"${bar.get_height():,.0f}", ha="center", va="bottom",
                 fontsize=9, color="white")

plt.tight_layout()
out = os.path.join(SAVE_DIR, "04_age_vs_claim.png")
plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Saved -> {out}")

# ---------------------------------------------------------------------------
# PLOT 5 — INCOME VS CLAIM ANALYSIS
# ---------------------------------------------------------------------------
print("[5/6] Generating Income vs Claim Analysis ...")

df_clean["income_bracket"] = pd.cut(
    df_clean["annual_income"],
    bins=[0, 40_000, 70_000, 100_000, 500_000],
    labels=["<$40K", "$40K–$70K", "$70K–$100K", ">$100K"]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("💵 Annual Income vs Insurance Claim Analysis", fontsize=14,
             fontweight="bold", color="white", y=1.02)

for ax in axes:
    ax.set_facecolor("#1a1d27")

# Scatter plot (sample for readability)
sample = df_clean.sample(2000, random_state=42)
scatter_colors = sample["fraud_flag"].map({0: "#06b6d4", 1: "#ef4444"})
sc = axes[0].scatter(sample["annual_income"] / 1000, sample["claim_amount"] / 1000,
                      c=scatter_colors, alpha=0.45, s=18, linewidths=0)
axes[0].set_title("Income vs Claim Amount (Fraud highlighted)")
axes[0].set_xlabel("Annual Income ($K)")
axes[0].set_ylabel("Claim Amount ($K)")
axes[0].grid(True, alpha=0.2)
genuine_patch = mpatches.Patch(color="#06b6d4", label="Genuine")
fraud_patch   = mpatches.Patch(color="#ef4444", label="Fraudulent")
axes[0].legend(handles=[genuine_patch, fraud_patch], fontsize=9)

# Average claim by income bracket
avg_claim_inc = df_clean.groupby("income_bracket")["claim_amount"].mean()
bars = axes[1].bar(avg_claim_inc.index, avg_claim_inc.values,
                    color=["#7c3aed", "#06b6d4", "#10b981", "#f59e0b"],
                    edgecolor="#0f1117")
axes[1].set_title("Average Claim Amount by Income Bracket")
axes[1].set_xlabel("Income Bracket")
axes[1].set_ylabel("Avg Claim Amount (USD)")
axes[1].grid(axis="y", alpha=0.3)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f"${bar.get_height():,.0f}", ha="center", va="bottom",
                 fontsize=9, color="white")

plt.tight_layout()
out = os.path.join(SAVE_DIR, "05_income_vs_claim.png")
plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Saved -> {out}")

# ---------------------------------------------------------------------------
# PLOT 6 — POLICY TYPE ANALYSIS
# ---------------------------------------------------------------------------
print("[6/6] Generating Policy Type Analysis ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("📋 Insurance Policy Type Analysis", fontsize=14,
             fontweight="bold", color="white", y=1.02)

for ax in axes:
    ax.set_facecolor("#1a1d27")

# Policy type counts
policy_counts = df_clean["policy_type"].value_counts()
axes[0].bar(policy_counts.index, policy_counts.values,
            color=ACCENT_PALETTE[:4], edgecolor="#0f1117")
axes[0].set_title("Policy Distribution")
axes[0].set_xlabel("Policy Type")
axes[0].set_ylabel("Count")
axes[0].grid(axis="y", alpha=0.3)
for i, (idx, val) in enumerate(policy_counts.items()):
    axes[0].text(i, val + 30, f"{val:,}", ha="center", va="bottom",
                 fontsize=9, color="white")

# Avg claim by policy type
avg_claim_policy = df_clean.groupby("policy_type")["claim_amount"].mean().sort_values(ascending=False)
bars = axes[1].barh(avg_claim_policy.index, avg_claim_policy.values,
                     color=ACCENT_PALETTE[:4])
axes[1].set_title("Avg Claim Amount by Policy Type")
axes[1].set_xlabel("Avg Claim Amount (USD)")
axes[1].grid(axis="x", alpha=0.3)
for bar in bars:
    axes[1].text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                 f"${bar.get_width():,.0f}", va="center", fontsize=9, color="white")

# Fraud rate by policy type
fraud_rate_policy = df_clean.groupby("policy_type")["fraud_flag"].mean().sort_values(ascending=False) * 100
colors_p = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(fraud_rate_policy))]
axes[2].bar(fraud_rate_policy.index, fraud_rate_policy.values,
            color=colors_p, edgecolor="#0f1117")
axes[2].set_title("Fraud Rate by Policy Type (%)")
axes[2].set_xlabel("Policy Type")
axes[2].set_ylabel("Fraud Rate (%)")
axes[2].grid(axis="y", alpha=0.3)
for i, (idx, val) in enumerate(fraud_rate_policy.items()):
    axes[2].text(i, val + 0.1, f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, color="white")

plt.tight_layout()
out = os.path.join(SAVE_DIR, "06_policy_type_analysis.png")
plt.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Saved -> {out}")

# ---------------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  EDA COMPLETE — Summary Statistics")
print("=" * 65)
print(df_clean[["age", "annual_income", "claim_amount", "premium_paid",
                 "health_score", "previous_claims", "accident_history"]].describe().to_string())
print(f"\nAll plots saved to: {SAVE_DIR}")
print("=" * 65)
