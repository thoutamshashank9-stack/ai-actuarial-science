"""
=============================================================================
AI ACTUARIAL SCIENCE — MODULE 03: MODEL TRAINING & EVALUATION
=============================================================================
Trains regression models for claim prediction and classification models
for fraud detection. Saves best models and preprocessing pipelines.
Generates model comparison tables and business insights report.

Author  : Senior Data Scientist & AI Actuary
Version : 1.0.0
=============================================================================
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "insurance_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "visualizations")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27", "axes.edgecolor": "#3a3d4a",
    "axes.labelcolor": "#e0e0e0", "axes.titlecolor": "#ffffff",
    "text.color": "#e0e0e0", "xtick.color": "#aaaaaa", "ytick.color": "#aaaaaa",
    "grid.color": "#2a2d3a", "grid.linestyle": "--", "grid.alpha": 0.5,
    "legend.facecolor": "#1a1d27", "legend.edgecolor": "#3a3d4a",
})
COLORS = ["#7c3aed", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]

print("=" * 68)
print("  AI ACTUARIAL SCIENCE — Model Training & Evaluation")
print("=" * 68)

# ============================================================================
# SECTION 1 — LOAD & PREPROCESSING
# ============================================================================
print("\n[SECTION 1] Loading and preprocessing data ...")

df = pd.read_csv(DATA_PATH)
print(f"  Raw shape       : {df.shape}")

# --- Duplicate Removal ---
initial_len = len(df)
df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"], inplace=True)
print(f"  After dedup     : {df.shape}  (removed {initial_len - len(df)} rows)")

# --- Drop customer_id ---
df.drop(columns=["customer_id"], inplace=True)

# --- Feature Engineering ---
print("  Engineering new features ...")
df["claim_to_premium_ratio"] = df["claim_amount"] / (df["premium_paid"] + 1)
df["risk_index"] = (
    df["previous_claims"] * 0.30 +
    df["accident_history"] * 0.30 +
    (100 - df["health_score"].fillna(df["health_score"].median())) * 0.005 +
    df["age"] * 0.002 +
    (df["policy_duration"] < 3).astype(int) * 0.20
)
df["age_health_interaction"] = df["age"] * (100 - df["health_score"].fillna(50)) / 100
df["income_per_year_insured"]= df["annual_income"] / (df["policy_duration"] + 1)

# --- Define feature sets ---
CATEGORICAL_FEATURES = ["gender", "occupation", "policy_type"]
NUMERIC_FEATURES = [
    "age", "annual_income", "policy_duration", "previous_claims",
    "vehicle_age", "health_score", "accident_history", "premium_paid",
    "claim_to_premium_ratio", "risk_index", "age_health_interaction",
    "income_per_year_insured"
]

REGRESSION_TARGET    = "claim_amount"
CLASSIFICATION_TARGET = "fraud_flag"

# Drop targets from feature list
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = df[ALL_FEATURES]
y_reg = df[REGRESSION_TARGET]
y_clf = df[CLASSIFICATION_TARGET]

print(f"  Feature set     : {len(ALL_FEATURES)} features")
print(f"  Fraud rate      : {y_clf.mean():.2%}")

# --- Preprocessing Pipeline ---
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
])

# ============================================================================
# SECTION 2 — REGRESSION: CLAIM PREDICTION
# ============================================================================
print("\n" + "=" * 68)
print("  [SECTION 2] CLAIM PREDICTION — Regression Models")
print("=" * 68)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)

reg_results = {}

def evaluate_regressor(name, pipeline, X_tr, X_te, y_tr, y_te):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    mse  = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te, y_pred)
    print(f"\n  ── {name}")
    print(f"     MAE  : ${mae:,.2f}")
    print(f"     MSE  : ${mse:,.2f}")
    print(f"     RMSE : ${rmse:,.2f}")
    print(f"     R²   : {r2:.4f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "predictions": y_pred}

# Linear Regression
print("\n  Training Linear Regression ...")
lr_pipe = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
reg_results["Linear Regression"] = evaluate_regressor(
    "Linear Regression", lr_pipe, X_reg_train, X_reg_test, y_reg_train, y_reg_test
)

# Random Forest Regressor
print("\n  Training Random Forest Regressor ...")
rf_reg_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=12,
                                    min_samples_leaf=5, n_jobs=-1, random_state=42))
])
reg_results["Random Forest"] = evaluate_regressor(
    "Random Forest", rf_reg_pipe, X_reg_train, X_reg_test, y_reg_train, y_reg_test
)

# XGBoost Regressor
print("\n  Training XGBoost Regressor ...")
xgb_reg_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=7,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, verbosity=0))
])
reg_results["XGBoost"] = evaluate_regressor(
    "XGBoost", xgb_reg_pipe, X_reg_train, X_reg_test, y_reg_train, y_reg_test
)

# --- Best Regression Model ---
best_reg_name = min(reg_results, key=lambda k: reg_results[k]["RMSE"])
best_reg_pipe = {
    "Linear Regression": lr_pipe,
    "Random Forest":     rf_reg_pipe,
    "XGBoost":           xgb_reg_pipe,
}[best_reg_name]

print(f"\n  ✅ Best Regression Model: {best_reg_name}  (RMSE=${reg_results[best_reg_name]['RMSE']:,.0f})")

# Save best regression model
joblib.dump(best_reg_pipe, os.path.join(MODEL_DIR, "claim_best_model.pkl"))
joblib.dump(rf_reg_pipe,   os.path.join(MODEL_DIR, "claim_rf_model.pkl"))
joblib.dump(xgb_reg_pipe,  os.path.join(MODEL_DIR, "claim_xgb_model.pkl"))
print("  Models saved to /models/")

# ============================================================================
# SECTION 3 — CLASSIFICATION: FRAUD DETECTION
# ============================================================================
print("\n" + "=" * 68)
print("  [SECTION 3] FRAUD DETECTION — Classification Models")
print("=" * 68)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.20, stratify=y_clf, random_state=42
)

clf_results = {}

def evaluate_classifier(name, pipeline, X_tr, X_te, y_tr, y_te):
    # Fit preprocessor and get transformed data for SMOTE
    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_te_proc = preprocessor.transform(X_te)

    # Apply SMOTE for class imbalance
    sm = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_proc, y_tr)

    # Train model on SMOTE-balanced data
    model = pipeline.named_steps["model"]
    model.fit(X_tr_bal, y_tr_bal)

    y_pred      = model.predict(X_te_proc)
    y_pred_proba= model.predict_proba(X_te_proc)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_pred_proba)

    print(f"\n  ── {name}")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print(f"     ROC-AUC   : {auc:.4f}")

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "AUC": auc,
        "y_pred": y_pred, "y_pred_proba": y_pred_proba,
        "cm": confusion_matrix(y_te, y_pred)
    }

# Logistic Regression
print("\n  Training Logistic Regression ...")
lr_clf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced",
                                  solver="lbfgs", random_state=42))
])
clf_results["Logistic Regression"] = evaluate_classifier(
    "Logistic Regression", lr_clf_pipe,
    X_clf_train, X_clf_test, y_clf_train, y_clf_test
)

# Random Forest Classifier
print("\n  Training Random Forest Classifier ...")
rf_clf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=12,
                                      class_weight="balanced", n_jobs=-1, random_state=42))
])
clf_results["Random Forest"] = evaluate_classifier(
    "Random Forest", rf_clf_pipe,
    X_clf_train, X_clf_test, y_clf_train, y_clf_test
)

# XGBoost Classifier
print("\n  Training XGBoost Classifier ...")
scale_pos_weight = (y_clf_train == 0).sum() / max((y_clf_train == 1).sum(), 1)
xgb_clf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=7,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=scale_pos_weight,
                             use_label_encoder=False, eval_metric="logloss",
                             random_state=42, verbosity=0))
])
clf_results["XGBoost"] = evaluate_classifier(
    "XGBoost", xgb_clf_pipe,
    X_clf_train, X_clf_test, y_clf_train, y_clf_test
)

# --- Best Classification Model ---
best_clf_name = max(clf_results, key=lambda k: clf_results[k]["AUC"])
best_clf_pipe = {
    "Logistic Regression": lr_clf_pipe,
    "Random Forest":       rf_clf_pipe,
    "XGBoost":             xgb_clf_pipe,
}[best_clf_name]

print(f"\n  ✅ Best Classification Model: {best_clf_name}  (AUC={clf_results[best_clf_name]['AUC']:.4f})")

# Save models
joblib.dump(best_clf_pipe, os.path.join(MODEL_DIR, "fraud_best_model.pkl"))
joblib.dump(rf_clf_pipe,   os.path.join(MODEL_DIR, "fraud_rf_model.pkl"))
joblib.dump(xgb_clf_pipe,  os.path.join(MODEL_DIR, "fraud_xgb_model.pkl"))
joblib.dump(preprocessor,  os.path.join(MODEL_DIR, "preprocessor.pkl"))

# Save feature lists for app
feature_config = {
    "numeric_features":      NUMERIC_FEATURES,
    "categorical_features":  CATEGORICAL_FEATURES,
    "all_features":          ALL_FEATURES,
    "regression_target":     REGRESSION_TARGET,
    "classification_target": CLASSIFICATION_TARGET,
    "best_reg_model":        best_reg_name,
    "best_clf_model":        best_clf_name,
}
with open(os.path.join(MODEL_DIR, "feature_config.json"), "w") as f:
    json.dump(feature_config, f, indent=2)

print("  All models and configs saved.")

# ============================================================================
# SECTION 4 — MODEL COMPARISON VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 68)
print("  [SECTION 4] Generating Model Comparison Visualizations")
print("=" * 68)

# --- Regression Comparison Table ---
reg_df = pd.DataFrame({k: {m: v for m, v in metrics.items() if m != "predictions"}
                        for k, metrics in reg_results.items()}).T
reg_df = reg_df[["MAE", "MSE", "RMSE", "R2"]]
reg_df = reg_df.astype(float)

print("\n  REGRESSION MODEL COMPARISON:")
print(reg_df.to_string())

# --- Classification Comparison Table ---
clf_df = pd.DataFrame({k: {m: v for m, v in metrics.items()
                             if m not in ["y_pred", "y_pred_proba", "cm"]}
                         for k, metrics in clf_results.items()}).T
clf_df = clf_df[["Accuracy", "Precision", "Recall", "F1", "AUC"]]
clf_df = clf_df.astype(float)

print("\n  CLASSIFICATION MODEL COMPARISON:")
print(clf_df.to_string())

# --- Plot: Regression Model Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("📈 Regression Model Comparison — Claim Prediction", fontsize=14,
             fontweight="bold", color="white", y=1.02)
for ax in axes: ax.set_facecolor("#1a1d27")

models = list(reg_results.keys())
x = np.arange(len(models))
width = 0.3

rmse_vals = [reg_results[m]["RMSE"] for m in models]
mae_vals  = [reg_results[m]["MAE"]  for m in models]
r2_vals   = [reg_results[m]["R2"]   for m in models]

bars1 = axes[0].bar(x - width/2, rmse_vals, width, label="RMSE", color="#7c3aed", alpha=0.85)
bars2 = axes[0].bar(x + width/2, mae_vals,  width, label="MAE",  color="#06b6d4", alpha=0.85)
axes[0].set_title("RMSE & MAE by Model (lower = better)")
axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=10)
axes[0].set_ylabel("Error (USD)")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

bars3 = axes[1].bar(x, r2_vals, color=COLORS[:3], alpha=0.85, edgecolor="#0f1117")
axes[1].set_title("R² Score by Model (higher = better)")
axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=10)
axes[1].set_ylabel("R² Score"); axes[1].set_ylim(0, 1.05)
axes[1].grid(axis="y", alpha=0.3)
for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, color="white")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "07_regression_comparison.png"),
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# --- Plot: Classification Model Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("🔍 Classification Model Comparison — Fraud Detection", fontsize=14,
             fontweight="bold", color="white", y=1.02)
for ax in axes: ax.set_facecolor("#1a1d27")

clf_models = list(clf_results.keys())
x = np.arange(len(clf_models))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
bar_width = 0.15
offsets = np.linspace(-0.3, 0.3, len(metrics_to_plot))

for i, (metric, offset) in enumerate(zip(metrics_to_plot, offsets)):
    vals = [clf_results[m][metric] for m in clf_models]
    axes[0].bar(x + offset, vals, bar_width, label=metric, color=COLORS[i], alpha=0.85)
axes[0].set_title("All Metrics by Model")
axes[0].set_xticks(x); axes[0].set_xticklabels(clf_models, rotation=10)
axes[0].set_ylabel("Score"); axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

# Confusion matrix for best model
cm = clf_results[best_clf_name]["cm"]
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", ax=axes[1],
            linewidths=1, linecolor="#0f1117",
            annot_kws={"size": 14, "color": "white"},
            xticklabels=["Genuine", "Fraud"],
            yticklabels=["Genuine", "Fraud"])
axes[1].set_title(f"Confusion Matrix — {best_clf_name}")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "08_classification_comparison.png"),
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"  Comparison plots saved to {REPORT_DIR}")

# ============================================================================
# SECTION 5 — FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 68)
print("  [SECTION 5] Feature Importance Analysis")
print("=" * 68)

feature_names_out = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)

# --- Regression Feature Importance (RF) ---
rf_reg_model = rf_reg_pipe.named_steps["model"]
reg_importances = rf_reg_model.feature_importances_

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("🎯 Feature Importance — Claim Prediction vs Fraud Detection",
             fontsize=14, fontweight="bold", color="white", y=1.02)
for ax in axes: ax.set_facecolor("#1a1d27")

# Regression feature importance
idx_reg = np.argsort(reg_importances)[-12:]
axes[0].barh(np.array(feature_names_out)[idx_reg], reg_importances[idx_reg],
              color="#7c3aed", edgecolor="#0f1117", alpha=0.85)
axes[0].set_title("Claim Prediction (Random Forest)")
axes[0].set_xlabel("Feature Importance")
axes[0].grid(axis="x", alpha=0.3)

# Classification feature importance (RF)
X_clf_proc = preprocessor.fit_transform(X_clf_train)
rf_clf_model = rf_clf_pipe.named_steps["model"]
X_proc = preprocessor.transform(X)
rf_clf_model.fit(X_proc, y_clf)
clf_importances = rf_clf_model.feature_importances_
idx_clf = np.argsort(clf_importances)[-12:]
axes[1].barh(np.array(feature_names_out)[idx_clf], clf_importances[idx_clf],
              color="#ef4444", edgecolor="#0f1117", alpha=0.85)
axes[1].set_title("Fraud Detection (Random Forest)")
axes[1].set_xlabel("Feature Importance")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "09_feature_importance.png"),
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Feature importance plot saved.")

# ============================================================================
# SECTION 6 — BUSINESS INSIGHTS REPORT
# ============================================================================
print("\n" + "=" * 68)
print("  [SECTION 6] Business Insights Report")
print("=" * 68)

# Re-load clean df for segmentation
df_insights = pd.read_csv(DATA_PATH)
df_insights.drop_duplicates(inplace=True)
for col in df_insights.select_dtypes(include=[np.number]).columns:
    df_insights[col].fillna(df_insights[col].median(), inplace=True)

# High-Risk Segments
high_risk = df_insights[
    (df_insights["previous_claims"] >= 3) |
    (df_insights["accident_history"] >= 2) |
    (df_insights["age"] >= 55)
].copy()

fraud_prone = df_insights[df_insights["fraud_flag"] == 1].copy()

report = f"""
================================================================================
     AI ACTUARIAL SCIENCE — BUSINESS INSIGHTS REPORT
================================================================================

1. DATASET OVERVIEW
   Total Records    : {len(df_insights):,}
   Fraud Rate       : {df_insights['fraud_flag'].mean():.2%}
   Avg Claim Amount : ${df_insights['claim_amount'].mean():,.2f}
   Avg Premium Paid : ${df_insights['premium_paid'].mean():,.2f}

2. HIGH-RISK CUSTOMER SEGMENTS
   High-Risk Count  : {len(high_risk):,} ({len(high_risk)/len(df_insights)*100:.1f}% of customers)
   Avg Claim (High) : ${high_risk['claim_amount'].mean():,.2f}
   Avg Claim (Low)  : ${df_insights[~df_insights.index.isin(high_risk.index)]['claim_amount'].mean():,.2f}
   Key Triggers     : Age 55+, 3+ previous claims, 2+ accidents

3. FRAUD-PRONE PROFILES
   Total Fraudulent  : {len(fraud_prone):,}
   Avg Fraud Claim   : ${fraud_prone['claim_amount'].mean():,.2f}
   Avg Genuine Claim : ${df_insights[df_insights['fraud_flag']==0]['claim_amount'].mean():,.2f}
   Fraud Premium Avg : ${fraud_prone['premium_paid'].mean():,.2f}
   
   Top Fraud Occupations:
{fraud_prone['occupation'].value_counts().head(3).to_string()}

   Top Fraud Policy Types:
{fraud_prone['policy_type'].value_counts().head(3).to_string()}

4. PREMIUM OPTIMIZATION SUGGESTIONS
   - Risk Tier 1 (Low Risk)     : Base premium × 0.85 — reward loyal, low-claim customers
   - Risk Tier 2 (Medium Risk)  : Base premium × 1.00 — standard pricing
   - Risk Tier 3 (High Risk)    : Base premium × 1.35 — surcharge for high-claim history
   - Risk Tier 4 (Very High)    : Base premium × 1.75 + mandatory review

5. RISK SCORING FRAMEWORK (0–100 scale)
   Component Weights:
   - Previous Claims     : 30%
   - Accident History    : 30%
   - Health Score (inv)  : 15%
   - Age Factor          : 10%
   - Policy Duration     : 10%
   - Fraud History       : 5%
   
   Score Bands:
   0–25  : Low Risk    — Green
   26–50 : Medium Risk — Yellow
   51–75 : High Risk   — Orange
   76–100: Very High   — Red

6. MODEL RECOMMENDATIONS
   - Claim Prediction    : {best_reg_name} (R²={reg_results[best_reg_name]['R2']:.3f}, RMSE=${reg_results[best_reg_name]['RMSE']:,.0f})
   - Fraud Detection     : {best_clf_name} (AUC={clf_results[best_clf_name]['AUC']:.3f}, F1={clf_results[best_clf_name]['F1']:.3f})

================================================================================
"""

print(report)

report_path = os.path.join(BASE_DIR, "reports", "business_insights_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"  Business insights saved to: {report_path}")

print("\n" + "=" * 68)
print("  ✅ MODEL TRAINING & EVALUATION COMPLETE")
print("=" * 68)
print(f"  Models saved to   : {MODEL_DIR}")
print(f"  Reports saved to  : {REPORT_DIR}")
print("=" * 68)
