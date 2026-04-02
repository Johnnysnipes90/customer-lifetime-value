# -----------------------------------------
# Ensure project root is in Python path
# -----------------------------------------
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_FILE

st.set_page_config(
    page_title="CLV Prediction Dashboard",
    page_icon="📈",
    layout="wide",
)

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_data
def load_metrics():
    metrics_path = Path(METRICS_FILE)
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def classify_customer(expected_clv: float) -> str:
    if expected_clv < 250:
        return "Low Value"
    if expected_clv < 1000:
        return "Medium Value"
    return "High Value"


def make_sample_payload() -> dict:
    return {
        "total_revenue": 1603.68,
        "total_orders": 3,
        "total_items": 956,
        "avg_line_revenue": 25.0575,
        "avg_unit_price": 2.877344,
        "unique_products": 50,
        "unique_countries": 1,
        "customer_tenure_days": 116,
        "recency_days": 39,
        "avg_revenue_per_order": 534.56,
        "active_days": 3,
        "avg_days_between_orders": 58.0,
        "revenue_per_day": 13.82,
        "orders_per_day": 0.0259,
        "recency_ratio": 0.3362,
        "items_per_order": 318.67,
    }


def call_prediction_api(api_base_url: str, payload: dict) -> dict:
    url = f"{api_base_url.rstrip('/')}/predict"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def call_explanation_api(api_base_url: str, payload: dict) -> dict:
    url = f"{api_base_url.rstrip('/')}/explain"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def check_api_health(api_base_url: str) -> tuple[bool, dict]:
    url = f"{api_base_url.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return True, response.json()
    except Exception as e:
        return False, {"status": "error", "detail": str(e)}


st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .sub-text {
            color: #6b7280;
            margin-bottom: 1.2rem;
        }
        .section-header {
            font-size: 1.15rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Customer Lifetime Value Prediction Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-text">Estimate expected customer lifetime value using a two-stage machine learning system served through FastAPI.</div>',
    unsafe_allow_html=True,
)

metrics = load_metrics()
if metrics:
    final_metrics = metrics.get("final_two_stage_metrics", {})
    classifier_metrics = metrics.get("classifier_metrics", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final MAE", f"{final_metrics.get('mae_raw', 0):,.2f}")
    c2.metric("Final RMSE", f"{final_metrics.get('rmse_raw', 0):,.2f}")
    c3.metric("Return ROC AUC", f"{classifier_metrics.get('roc_auc', 0):.3f}")
    c4.metric("Architecture", "API + UI")

st.divider()

st.sidebar.header("API Connection")
api_base_url = st.sidebar.text_input("FastAPI Base URL", value=DEFAULT_API_URL)

health_ok, health_payload = check_api_health(api_base_url)
if health_ok:
    st.sidebar.success("API connected")
    st.sidebar.json(health_payload)
else:
    st.sidebar.error("API not reachable")
    st.sidebar.caption(health_payload.get("detail", "Unknown error"))

st.sidebar.divider()
st.sidebar.header("Controls")

use_sample = st.sidebar.button("Load Sample Customer")
clear_inputs = st.sidebar.button("Reset to Default")

if "form_data" not in st.session_state or clear_inputs:
    st.session_state.form_data = {
        "total_revenue": 500.0,
        "total_orders": 2,
        "total_items": 50,
        "avg_line_revenue": 20.0,
        "avg_unit_price": 5.0,
        "unique_products": 10,
        "unique_countries": 1,
        "customer_tenure_days": 60,
        "recency_days": 15,
        "avg_revenue_per_order": 250.0,
        "active_days": 2,
        "avg_days_between_orders": 20.0,
        "revenue_per_day": 8.33,
        "orders_per_day": 0.03,
        "recency_ratio": 0.25,
        "items_per_order": 25.0,
    }

if use_sample:
    st.session_state.form_data = make_sample_payload()

left, right = st.columns([1.25, 1])

with left:
    st.markdown(
        '<div class="section-header">Customer Feature Inputs</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        total_revenue = st.number_input(
            "Total Revenue",
            min_value=0.0,
            value=float(st.session_state.form_data["total_revenue"]),
            step=10.0,
        )
        total_orders = st.number_input(
            "Total Orders",
            min_value=0,
            value=int(st.session_state.form_data["total_orders"]),
            step=1,
        )
        total_items = st.number_input(
            "Total Items",
            min_value=0,
            value=int(st.session_state.form_data["total_items"]),
            step=1,
        )
        avg_line_revenue = st.number_input(
            "Average Line Revenue",
            min_value=0.0,
            value=float(st.session_state.form_data["avg_line_revenue"]),
            step=1.0,
        )
        avg_unit_price = st.number_input(
            "Average Unit Price",
            min_value=0.0,
            value=float(st.session_state.form_data["avg_unit_price"]),
            step=0.5,
        )
        unique_products = st.number_input(
            "Unique Products",
            min_value=0,
            value=int(st.session_state.form_data["unique_products"]),
            step=1,
        )
        unique_countries = st.number_input(
            "Unique Countries",
            min_value=0,
            value=int(st.session_state.form_data["unique_countries"]),
            step=1,
        )
        customer_tenure_days = st.number_input(
            "Customer Tenure Days",
            min_value=0,
            value=int(st.session_state.form_data["customer_tenure_days"]),
            step=1,
        )

    with col2:
        recency_days = st.number_input(
            "Recency Days",
            min_value=0,
            value=int(st.session_state.form_data["recency_days"]),
            step=1,
        )
        avg_revenue_per_order = st.number_input(
            "Average Revenue per Order",
            min_value=0.0,
            value=float(st.session_state.form_data["avg_revenue_per_order"]),
            step=1.0,
        )
        active_days = st.number_input(
            "Active Days",
            min_value=0,
            value=int(st.session_state.form_data["active_days"]),
            step=1,
        )
        avg_days_between_orders = st.number_input(
            "Average Days Between Orders",
            min_value=0.0,
            value=float(st.session_state.form_data["avg_days_between_orders"]),
            step=1.0,
        )
        revenue_per_day = st.number_input(
            "Revenue per Day",
            min_value=0.0,
            value=float(st.session_state.form_data["revenue_per_day"]),
            step=0.5,
        )
        orders_per_day = st.number_input(
            "Orders per Day",
            min_value=0.0,
            value=float(st.session_state.form_data["orders_per_day"]),
            step=0.01,
            format="%.4f",
        )
        recency_ratio = st.number_input(
            "Recency Ratio",
            min_value=0.0,
            value=float(st.session_state.form_data["recency_ratio"]),
            step=0.01,
            format="%.4f",
        )
        items_per_order = st.number_input(
            "Items per Order",
            min_value=0.0,
            value=float(st.session_state.form_data["items_per_order"]),
            step=1.0,
        )

    predict_button = st.button(
        "Predict Customer Lifetime Value",
        type="primary",
        width="stretch",
    )

with right:
    st.markdown(
        '<div class="section-header">Input Summary</div>',
        unsafe_allow_html=True,
    )

    preview_data = {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_items": total_items,
        "avg_line_revenue": avg_line_revenue,
        "avg_unit_price": avg_unit_price,
        "unique_products": unique_products,
        "unique_countries": unique_countries,
        "customer_tenure_days": customer_tenure_days,
        "recency_days": recency_days,
        "avg_revenue_per_order": avg_revenue_per_order,
        "active_days": active_days,
        "avg_days_between_orders": avg_days_between_orders,
        "revenue_per_day": revenue_per_day,
        "orders_per_day": orders_per_day,
        "recency_ratio": recency_ratio,
        "items_per_order": items_per_order,
    }

    st.dataframe(
        pd.DataFrame([preview_data]).T.rename(columns={0: "value"}),
        width="stretch",
    )

if predict_button:
    if not health_ok:
        st.error(
            "Prediction API is not available. Please start the FastAPI service first."
        )
    else:
        try:
            prediction = call_prediction_api(api_base_url, preview_data)
            explanation = call_explanation_api(api_base_url, preview_data)

            return_probability = prediction["return_probability"]
            predicted_value_if_return = prediction["predicted_value_if_return"]
            expected_clv = prediction["expected_clv"]
            segment = classify_customer(expected_clv)
            top_factors = explanation["top_factors"]

            st.divider()
            st.markdown(
                '<div class="section-header">Prediction Results</div>',
                unsafe_allow_html=True,
            )

            r1, r2, r3 = st.columns(3)
            r1.metric("Return Probability", f"{return_probability:.2%}")
            r2.metric("Value if Return", f"{predicted_value_if_return:,.2f}")
            r3.metric("Expected CLV", f"{expected_clv:,.2f}")

            st.markdown("### Customer Value Segment")
            if segment == "High Value":
                st.success(
                    f"{segment} customer — strong commercial priority for retention and upsell."
                )
            elif segment == "Medium Value":
                st.warning(
                    f"{segment} customer — worth targeted engagement and monitoring."
                )
            else:
                st.info(
                    f"{segment} customer — likely lower priority for intensive retention spend."
                )

            st.markdown("### Prediction Interpretation")
            st.progress(
                min(max(return_probability, 0.0), 1.0),
                text=f"Return Probability: {return_probability:.2%}",
            )

            interp_col1, interp_col2 = st.columns(2)
            with interp_col1:
                st.markdown(
                    """
                    **Expected CLV** is the final business-facing estimate of future customer value.

                    It combines:
                    - the probability that the customer returns
                    - the expected revenue if the customer returns
                    """
                )

            with interp_col2:
                st.markdown(
                    f"""
                    **Model Output Summary**
                    - Return Probability: **{return_probability:.2%}**
                    - Predicted Value if Return: **{predicted_value_if_return:,.2f}**
                    - Expected CLV: **{expected_clv:,.2f}**
                    """
                )

            st.markdown("### Why the Model Predicted This Value")

            explain_df = pd.DataFrame(top_factors)
            st.dataframe(
                explain_df[["feature", "feature_value", "shap_value", "direction"]],
                width="stretch",
            )

            st.markdown("### Top Drivers")
            for row in top_factors:
                feature = row["feature"]
                direction = row["direction"]
                shap_value = row["shap_value"]
                feature_value = row["feature_value"]

                if direction == "increase":
                    st.success(
                        f"**{feature}** = {feature_value} increased the predicted value "
                        f"(SHAP: {shap_value:.4f})"
                    )
                else:
                    st.info(
                        f"**{feature}** = {feature_value} decreased the predicted value "
                        f"(SHAP: {shap_value:.4f})"
                    )

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
