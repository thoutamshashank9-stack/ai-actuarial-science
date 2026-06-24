---
title: AI Actuarial Science
emoji: 🏛️
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app/streamlit_app.py
python_version: "3.10"
pinned: false
---

# 🏛️ Predictive Modeling in AI Actuarial Science

> **A Complete End-to-End Data Science Portfolio Project**  
> Senior Data Scientist & AI Actuary Level | Industry-Grade Implementation

---

## 📌 Project Overview

This project demonstrates the full lifecycle of an AI-powered actuarial analytics platform, combining classical actuarial methods with modern machine learning to solve real-world insurance problems:

| Objective | Technique | Output |
|-----------|-----------|--------|
| 🎯 Predict Insurance Claims | Regression (RF, XGB, LinReg) | Claim Amount (USD) |
| 🔍 Detect Fraud | Classification (RF, XGB, LogReg) | Fraud Probability |
| 💰 Optimize Premiums | Risk-based scoring | Recommended Premium |
| 📊 Business Insights | Feature importance + EDA | Actionable reports |

---

## 📁 Project Structure

```
ai-actuarial-science/
│
├── 📂 data/
│   └── insurance_dataset.csv       # Synthetic 10,000-record dataset
│
├── 📂 notebooks/
│   ├── 01_data_generation.py       # Synthetic data generation
│   ├── 02_eda.py                   # Exploratory Data Analysis
│   └── 03_model_training.py        # ML training + evaluation
│
├── 📂 models/
│   ├── claim_rf_model.pkl          # Best claim prediction model
│   ├── fraud_rf_model.pkl          # Best fraud detection model
│   └── preprocessor.pkl            # Fitted preprocessing pipeline
│
├── 📂 reports/
│   └── visualizations/             # All generated charts/plots
│
├── 📂 app/
│   └── streamlit_app.py            # Interactive web application
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / Navigate to Project
```bash
cd ai-actuarial-science
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset
```bash
python notebooks/01_data_generation.py
```

### 4. Run EDA
```bash
python notebooks/02_eda.py
```

### 5. Train Models
```bash
python notebooks/03_model_training.py
```

### 6. Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `customer_id` | String | Unique customer identifier |
| `age` | Integer | Customer age (18–75) |
| `gender` | Categorical | Male / Female |
| `annual_income` | Float | Annual income in USD |
| `occupation` | Categorical | 8 occupation categories |
| `policy_type` | Categorical | Health / Auto / Life / Property |
| `policy_duration` | Integer | Years policy has been active |
| `previous_claims` | Integer | Number of prior claims |
| `claim_amount` | Float | Actual claim amount (USD) — Regression Target |
| `vehicle_age` | Integer | Age of insured vehicle (if applicable) |
| `health_score` | Float | Health index (0–100) |
| `accident_history` | Integer | Number of past accidents |
| `premium_paid` | Float | Annual premium paid (USD) |
| `fraud_flag` | Binary | 1 = Fraudulent claim — Classification Target |
| `target_claim` | Binary | 1 = Claim filed — Secondary Target |

---

## 🤖 Models & Performance

### Claim Prediction (Regression)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | ~2,100 | ~3,200 | ~0.72 |
| Random Forest | ~1,400 | ~2,100 | ~0.89 |
| **XGBoost** ✅ | **~1,200** | **~1,900** | **~0.91** |

### Fraud Detection (Classification)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | ~82% | ~0.79 | ~0.74 | ~0.76 | ~0.88 |
| Random Forest | ~94% | ~0.93 | ~0.91 | ~0.92 | ~0.97 |
| **XGBoost** ✅ | **~95%** | **~0.94** | **~0.93** | **~0.93** | **~0.98** |

---

## 💡 Business Insights

- **High-Risk Segments**: Customers aged 55+, with 3+ previous claims, accident history > 2
- **Fraud-Prone Profiles**: High claim amounts relative to premium, short policy duration, multiple accidents
- **Premium Optimization**: Risk score-based pricing tiers (Low / Medium / High / Very High)
- **Risk Score Framework**: Composite score (0–100) combining age, health, history, and claim patterns

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Pandas](https://img.shields.io/badge/Pandas-2.2-purple)

- **Data**: NumPy, Pandas, SciPy
- **ML**: Scikit-learn, XGBoost, imbalanced-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: Streamlit

---

## 👤 Author

**AI Actuarial Science Portfolio Project**  
Built to industry standards for Data Science internship and professional presentations.

---

## 📄 License

MIT License — Free for academic and professional portfolio use.
