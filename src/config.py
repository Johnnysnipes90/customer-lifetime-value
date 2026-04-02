from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Input files
RAW_XLSX_FILE = RAW_DIR / "Online Retail.xlsx"

# Processed files
CLEAN_DATA_FILE = PROCESSED_DIR / "online_retail_clean.csv"
MODELING_TABLE_FILE = PROCESSED_DIR / "customer_modeling_table.csv"

# Model artifact files
RETURN_CLASSIFIER_FILE = MODELS_DIR / "clv_return_classifier.pkl"
VALUE_REGRESSOR_FILE = MODELS_DIR / "clv_value_regressor.pkl"
METRICS_FILE = MODELS_DIR / "clv_two_stage_metrics.json"

# SHAP artifact files
SHAP_EXPLAINER_FILE = MODELS_DIR / "clv_shap_explainer.pkl"
SHAP_BACKGROUND_FILE = MODELS_DIR / "clv_shap_background.csv"

# Time window settings
OBSERVATION_DAYS = 180
PREDICTION_DAYS = 90

# Final feature columns
FEATURE_COLUMNS = [
    "total_revenue",
    "total_orders",
    "total_items",
    "avg_line_revenue",
    "avg_unit_price",
    "unique_products",
    "unique_countries",
    "customer_tenure_days",
    "recency_days",
    "avg_revenue_per_order",
    "active_days",
    "avg_days_between_orders",
    "revenue_per_day",
    "orders_per_day",
    "recency_ratio",
    "items_per_order",
]

# Classification model params
CLASSIFIER_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": 0.6,
    "random_state": 42,
}

# Conditional regression params
REGRESSOR_PARAMS = {
    "n_estimators": 800,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
}
