import pandas as pd

from src.config import CLEAN_DATA_FILE, RAW_XLSX_FILE
from src.utils import ensure_directory


def load_raw_data(file_path=RAW_XLSX_FILE) -> pd.DataFrame:
    """Load raw retail data from Excel."""
    return pd.read_excel(file_path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to consistent snake_case format.
    Mirrors EDA logic exactly.
    """
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    df.rename(
        columns={
            "invoiceno": "invoice_no",
            "stockcode": "stock_code",
            "description": "description",
            "quantity": "quantity",
            "invoicedate": "invoice_date",
            "unitprice": "unit_price",
            "customerid": "customer_id",
            "country": "country",
        },
        inplace=True,
    )

    return df


def add_revenue_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction-level revenue feature."""
    df = df.copy()
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df


def analyze_and_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and remove exact duplicates across business key columns.

    Business reasoning:
    Duplicate rows inflate revenue and distort customer behavior metrics.
    """

    key_cols = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]

    duplicate_count = df.duplicated(subset=key_cols).sum()
    duplicate_pct = duplicate_count / len(df) * 100

    print("\n--- DUPLICATE ANALYSIS ---")
    print(f"Exact duplicates: {duplicate_count:,}")
    print(f"Duplicate percentage: {duplicate_pct:.2f}%")

    df_deduped = df.drop_duplicates(subset=key_cols).copy()

    print("\n--- DEDUPLICATION IMPACT ---")
    print("Before:", df.shape)
    print("After:", df_deduped.shape)
    print("Removed rows:", len(df) - len(df_deduped))

    return df_deduped


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business cleaning rules.

    Rules:
    - remove rows without customer_id (cannot model CLV)
    - remove rows without invoice_date
    - remove negative or zero quantity (returns or errors)
    - remove negative or zero unit_price
    """

    df = df.copy()

    df = df.dropna(subset=["customer_id", "invoice_date"])

    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)].copy()

    df["customer_id"] = df["customer_id"].astype("Int64").astype(str)

    print("\n--- CLEANING IMPACT ---")
    print("Final clean shape:", df.shape)

    return df.reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional time-based features."""
    df = df.copy()

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["invoice_month"] = df["invoice_date"].dt.to_period("M").astype(str)

    return df


def prepare_clean_data() -> pd.DataFrame:
    """
    Full raw → clean pipeline.

    This function mirrors the EDA process exactly and ensures:
    - reproducibility
    - consistency between notebook and production
    """

    print("Loading raw data...")
    df_raw = load_raw_data()

    print("Standardizing columns...")
    df = standardize_columns(df_raw)

    print("Adding revenue feature...")
    df = add_revenue_feature(df)

    print("Analyzing and removing duplicates...")
    df = analyze_and_remove_duplicates(df)

    print("Applying business cleaning rules...")
    df = clean_transactions(df)

    print("Adding time features...")
    df = add_time_features(df)

    return df


def save_clean_data(df: pd.DataFrame, output_path=CLEAN_DATA_FILE) -> None:
    """Save cleaned dataset."""
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    df_clean = prepare_clean_data()
    save_clean_data(df_clean)

    print("\n✅ Data preparation complete.")
    print(f"Saved to: {CLEAN_DATA_FILE}")
