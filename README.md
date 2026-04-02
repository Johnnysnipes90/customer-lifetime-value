# 📈 Customer Lifetime Value (CLV) Prediction System

## 🚀 Overview

This project builds a **production-grade machine learning system** to predict customer lifetime value using a **two-stage modeling architecture**:

> **Expected CLV = P(Return) × E(Value | Return)**

The system mirrors how real-world data teams deploy ML systems — combining:

- data processing
- feature engineering
- machine learning
- API deployment
- explainability
- interactive dashboards

---

## 🌐 Live Demo

- 🔗 **API (FastAPI):**
  https://customer-lifetime-value-system.onrender.com/docs

- 📊 **Dashboard (Streamlit):**
  _(Insert your Streamlit URL after deployment)_

---

## 🎯 Business Problem

Businesses need to answer:

- Which customers are likely to return?
- How much revenue will they generate?
- Where should we focus retention and marketing spend?

Traditional CLV approaches fail because they:

- ignore churn probability
- assume uniform customer behavior
- lack interpretability

This project solves that using a **probabilistic + conditional modeling approach**.

---

## 🧠 Solution Architecture

```text
Raw Data → Feature Engineering → ML Models → API → Dashboard
```

### Two-Stage Modeling Strategy

1️⃣ **Return Model (Classification)**
Predicts probability that a customer returns

2️⃣ **Value Model (Regression)**
Predicts expected revenue given the customer returns

3️⃣ **Final Output**

```text
Expected CLV = P(Return) × Predicted Value
```

---

## 📊 Dataset

- ~540,000 transactions
- ~4,300 customers
- UK-based e-commerce retail dataset

### Key Variables

- purchase frequency
- monetary value
- product diversity
- recency and tenure
- behavioral aggregates

---

## ⚙️ Feature Engineering

Key engineered features:

- revenue_per_day
- orders_per_day
- recency_ratio
- items_per_order
- customer_tenure_days
- avg_days_between_orders

✅ Strict **time-based split** ensures no data leakage

---

## 🤖 Models

### Stage 1 — Return Prediction

- Model: XGBoost Classifier
- ROC AUC: ~0.71

### Stage 2 — Conditional CLV

- Model: XGBoost Regressor
- Regularized for stability

### Final System Performance

- MAE: **~599**
- RMSE: **~2596**

👉 Significant improvement over single-stage modeling

---

## 🔍 Explainability (SHAP)

The system includes **SHAP explainability** to interpret predictions.

### What it explains:

- drivers of predicted customer value
- feature impact (increase vs decrease)
- contribution magnitude

### Example Insights:

- High engagement → increases CLV
- High recency → decreases CLV
- Frequent purchasing → boosts expected value

---

## 🌐 API (FastAPI)

### Endpoints

- `/predict` → CLV prediction
- `/explain` → SHAP explanation
- `/health` → service status

### Example Response

```json
{
  "return_probability": 0.84,
  "predicted_value_if_return": 1016.57,
  "expected_clv": 863.51
}
```

---

## 📊 Dashboard (Streamlit)

Interactive interface to:

- input customer features
- generate CLV predictions
- visualize SHAP explanations
- segment customers into value tiers

---

## 🐳 Docker

Run full system locally:

```bash
docker compose up --build
```

Access:

- API → http://localhost:8000/docs
- Dashboard → http://localhost:8501

---

## ⚙️ CI/CD

GitHub Actions pipeline:

- dependency validation
- pipeline sanity check
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

## 💡 Key Takeaways

- Production ML goes beyond modeling
- Two-stage modeling improves business realism
- Explainability builds stakeholder trust
- Deployment differentiates senior candidates

---

## 🔥 Future Improvements

- Real-time prediction pipeline
- model monitoring & drift detection
- feature store integration
- A/B testing framework

---

## 👤 Author

**John Olalemi**
Data Scientist | Machine Learning Engineer

- GitHub: https://github.com/Johnnysnipes90
- LinkedIn: https://www.linkedin.com/in/john-olalemi

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub — it helps visibility!
