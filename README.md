# 📈 Customer Lifetime Value (CLV) Prediction System

## 🚀 Overview

This project builds a **production-grade machine learning system** to predict customer lifetime value using a **two-stage modeling architecture**:

> **Expected CLV = P(Return) × E(Value | Return)**

The system is designed to reflect how modern data science teams deploy real-world ML systems — combining **analytics, modeling, APIs, explainability, and deployment**.

---

## 🎯 Business Problem

Businesses need to answer:

- Which customers are likely to return?
- How much revenue will they generate?
- Where should we focus retention and marketing spend?

Traditional CLV models are often inaccurate because they:

- ignore churn probability
- treat all customers equally
- lack interpretability

This project solves that by building a **probabilistic + conditional modeling system**.

---

## 🧠 Solution Architecture

```text
Raw Data → Feature Engineering → ML Models → API → Dashboard
```

### Two-Stage Model

1. **Return Model (Classification)**
   - Predicts probability a customer returns

2. **Value Model (Regression)**
   - Predicts revenue conditional on return

3. **Final Prediction**

```text
Expected CLV = P(Return) × Predicted Value
```

---

## 📊 Dataset

- ~540,000 transactions
- ~4,300 customers
- E-commerce retail dataset

### Features include:

- transaction values
- product variety
- purchase frequency
- recency & tenure
- behavioral aggregates

---

## ⚙️ Feature Engineering

Built customer-level features such as:

- revenue_per_day
- orders_per_day
- recency_ratio
- items_per_order
- customer_tenure_days
- avg_days_between_orders

Strict **time-based split** prevents data leakage.

---

## 🤖 Models

### Stage 1 — Return Prediction

- Model: XGBoost Classifier
- ROC AUC: ~0.71

### Stage 2 — Conditional CLV

- Model: XGBoost Regressor
- Optimized with regularization

### Final System Performance

- MAE: ~599
- RMSE: ~2596

---

## 🔍 Explainability (SHAP)

The system includes **SHAP explainability** to understand:

- why a customer has high or low CLV
- which features drive predictions

Example insights:

- high revenue frequency → increases CLV
- high recency → decreases CLV
- strong engagement → boosts expected value

---

## 🌐 API (FastAPI)

Endpoints:

- `/predict` → CLV prediction
- `/explain` → SHAP explanation
- `/health` → system status

📎 Live API: _(insert Render URL)_

---

## 📊 Dashboard (Streamlit)

Interactive UI to:

- input customer features
- view predictions
- see SHAP explanations
- segment customers

---

## 🐳 Docker

Run entire system:

```bash
docker compose up --build
```

---

## ⚙️ CI/CD

GitHub Actions pipeline:

- dependency validation
- pipeline integrity check
- automated build verification

---

## 📁 Project Structure

```
customer-lifetime-value/
│
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx
│   └── processed/
│       ├── online_retail_clean.csv
│       └── customer_modeling_table.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── api/
│   └── app.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── models/
│   ├── clv_return_classifier.pkl
│   ├── clv_value_regressor.pkl
│   └── clv_two_stage_metrics.json
│
├── reports/
├── requirements.txt
├── Dockerfile.api
├──Dockerfile.dashboard
├──docker-compose.yml
├──.dockerignore
├── README.md
└── main.py
├── README.md
└── main.py
```

---

## 💡 Key Learnings

- Production ML ≠ just modeling
- Two-stage modeling improves business realism
- Explainability is critical for stakeholder trust
- Deployment separates strong vs average data scientists

---

## 🔥 Future Improvements

- Real-time streaming predictions
- Model monitoring & drift detection
- Feature store integration
- A/B testing framework

---

## 👤 Author

**John Olalemi**
Data Scientist | ML Engineer

- GitHub: https://github.com/Johnnysnipes90
- LinkedIn: https://www.linkedin.com/in/john-olalemi

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub — it helps visibility!
