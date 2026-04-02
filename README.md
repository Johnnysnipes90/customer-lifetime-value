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
├── Dockerfile
├── README.md
└── main.py