import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary to a JSON file."""
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load a dictionary from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(model: Any, path: Path) -> None:
    """Save a trained model with joblib."""
    ensure_directory(path.parent)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Load a trained model with joblib."""
    return joblib.load(path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV."""
    ensure_directory(path.parent)
    df.to_csv(path, index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from CSV."""
    return pd.read_csv(path)
