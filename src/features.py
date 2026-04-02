import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_FILE,
    MODELING_TABLE_FILE,
    OBSERVATION_DAYS,
    PREDICTION_DAYS,
)
from src.utils import ensure_directory


def load_clean_data(file_path=CLEAN_DATA_FILE) -> pd.DataFrame:
    """Load cleaned transaction data."""
    return pd.read_csv(file_path, parse_dates=["invoice_date"])


def define_modeling_windows(
    df: pd.DataFrame,
    observation_days: int = OBSERVATION_DAYS,
    prediction_days: int = PREDICTION_DAYS,
) -> dict:
    """Define observation and prediction window boundaries."""
    max_date = df["invoice_date"].max().normalize()
    prediction_start = (
        max_date - pd.Timedelta(days=prediction_days) + pd.Timedelta(days=1)
    )
    observation_start = prediction_start - pd.Timedelta(days=observation_days)

    return {
        "observation_start": observation_start,
        "prediction_start": prediction_start,
        "max_date": max_date,
    }


def split_observation_future(
    df: pd.DataFrame, windows: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into observation and future windows."""
    observation_df = df[
        (df["invoice_date"] >= windows["observation_start"])
        & (df["invoice_date"] < windows["prediction_start"])
    ].copy()

    future_df = df[
        (df["invoice_date"] >= windows["prediction_start"])
        & (df["invoice_date"] <= windows["max_date"])
    ].copy()

    return observation_df, future_df


def build_customer_features(
    observation_df: pd.DataFrame, prediction_start: pd.Timestamp
) -> pd.DataFrame:
    """Aggregate customer behavior features from the observation window."""
    customer_features = (
        observation_df.groupby("customer_id")
        .agg(
            total_revenue=("revenue", "sum"),
            total_orders=("invoice_no", "nunique"),
            total_items=("quantity", "sum"),
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
            avg_line_revenue=("revenue", "mean"),
            avg_unit_price=("unit_price", "mean"),
            unique_products=("stock_code", "nunique"),
            unique_countries=("country", "nunique"),
        )
        .reset_index()
    )

    customer_features["customer_tenure_days"] = (
        customer_features["last_purchase"] - customer_features["first_purchase"]
    ).dt.days

    customer_features["recency_days"] = (
        prediction_start - customer_features["last_purchase"]
    ).dt.days

    customer_features["avg_revenue_per_order"] = (
        customer_features["total_revenue"] / customer_features["total_orders"]
    )

    active_days = (
        observation_df.groupby("customer_id")["invoice_date"]
        .nunique()
        .reset_index(name="active_days")
    )

    customer_features = customer_features.merge(
        active_days, on="customer_id", how="left"
    )

    customer_features["avg_days_between_orders"] = np.where(
        customer_features["total_orders"] > 1,
        customer_features["customer_tenure_days"]
        / (customer_features["total_orders"] - 1),
        customer_features["customer_tenure_days"],
    )

    return customer_features


def build_future_target(future_df: pd.DataFrame) -> pd.DataFrame:
    """Build future CLV target from the prediction window."""
    future_target = (
        future_df.groupby("customer_id")
        .agg(
            future_clv=("revenue", "sum"),
            future_orders=("invoice_no", "nunique"),
        )
        .reset_index()
    )
    return future_target


def add_intensity_features(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized behavioral intensity features."""
    modeling_df = modeling_df.copy()

    modeling_df["customer_tenure_days_adj"] = modeling_df[
        "customer_tenure_days"
    ].replace(0, 1)
    modeling_df["total_orders_adj"] = modeling_df["total_orders"].replace(0, 1)

    modeling_df["revenue_per_day"] = (
        modeling_df["total_revenue"] / modeling_df["customer_tenure_days_adj"]
    )
    modeling_df["orders_per_day"] = (
        modeling_df["total_orders"] / modeling_df["customer_tenure_days_adj"]
    )
    modeling_df["recency_ratio"] = (
        modeling_df["recency_days"] / modeling_df["customer_tenure_days_adj"]
    )
    modeling_df["items_per_order"] = (
        modeling_df["total_items"] / modeling_df["total_orders_adj"]
    )

    return modeling_df


def build_modeling_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    windows = define_modeling_windows(df_clean)
    observation_df, future_df = split_observation_future(df_clean, windows)

    customer_features = build_customer_features(
        observation_df=observation_df,
        prediction_start=windows["prediction_start"],
    )

    future_target = build_future_target(future_df)

    modeling_df = customer_features.merge(future_target, on="customer_id", how="left")
    modeling_df["future_clv"] = modeling_df["future_clv"].fillna(0)
    modeling_df["future_orders"] = modeling_df["future_orders"].fillna(0).astype(int)

    modeling_df = add_intensity_features(modeling_df)

    final_modeling_df = modeling_df[
        [
            "customer_id",
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
            "future_orders",
            "future_clv",
        ]
    ].copy()

    final_modeling_df["customer_id"] = final_modeling_df["customer_id"].astype(str)

    return final_modeling_df


def save_modeling_table(df: pd.DataFrame, output_path=MODELING_TABLE_FILE) -> None:
    """Save final modeling table."""
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    clean_df = load_clean_data()
    modeling_df = build_modeling_table(clean_df)
    save_modeling_table(modeling_df)
    print(f"Saved modeling table to: {MODELING_TABLE_FILE}")
    print(modeling_df.shape)
