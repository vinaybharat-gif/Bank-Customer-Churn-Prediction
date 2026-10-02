"""
data_loader.py
==============
Data loading, cleaning (wrangling) and feature-engineering pipeline for the
Bank Customer Churn Prediction project.

Pipeline
--------
1. load_raw()          -> read CSV and standardise column names
2. clean_data()        -> drop ID columns, fix missing values, remove duplicates,
                          cap outliers (IQR / winsorising)
3. prepare_dataset()   -> one-hot encoding, train/test split, StandardScaler
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TARGET = "churn"
CATEGORICAL_FEATURES = ["country", "gender"]
NUMERIC_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "estimated_salary",
]
BINARY_FEATURES = ["credit_card", "active_member"]
OUTLIER_COLUMNS = ["credit_score", "age", "balance", "estimated_salary"]
ID_COLUMNS = ["customer_id", "row_number", "surname"]

_CANONICAL_COLUMNS = (
    ID_COLUMNS
    + CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
    + BINARY_FEATURES
    + [TARGET]
)


def _key(name: str) -> str:
    """Normalise a column name so that 'CreditScore', 'credit_score' and
    'Credit Score' all map to the same key."""
    return re.sub(r"[\s_\-]+", "", str(name).strip().lower())


# Map of normalised-name -> canonical name (also handles the popular Kaggle
# variant of this dataset: Geography, Exited, NumOfProducts, HasCrCard ...).
_COLUMN_MAP: Dict[str, str] = {_key(c): c for c in _CANONICAL_COLUMNS}
_COLUMN_MAP.update(
    {
        "exited": TARGET,
        "geography": "country",
        "numofproducts": "products_number",
        "hascrcard": "credit_card",
        "isactivemember": "active_member",
        "rownumber": "row_number",
    }
)

DataSource = Union[str, Path, IO[bytes]]


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical snake_case names used in this project."""
    renamed = {c: _COLUMN_MAP.get(_key(c), str(c).strip()) for c in df.columns}
    return df.rename(columns=renamed)


def load_raw(source: DataSource) -> pd.DataFrame:
    """Read the CSV file (path or file-like object) and validate its schema."""
    df = standardize_columns(pd.read_csv(source))
    required = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"The dataset is missing required column(s): {', '.join(missing)}"
        )
    return df


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
def _iqr_bounds(series: pd.Series, factor: float = 1.5) -> Tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def clean_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the raw dataframe.

    Steps: remove duplicates -> drop unused ID columns -> coerce dtypes ->
    impute missing values -> cap outliers with the IQR rule.

    Returns
    -------
    (clean_df, report) where *report* is a dictionary summarising every action.
    """
    df = raw.copy()
    report: dict = {"rows_before": int(len(df))}

    # Duplicates (checked on the full original row, i.e. including customer_id)
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    if "customer_id" in df.columns:
        dup_ids = int(df.duplicated(subset=["customer_id"]).sum())
        report["duplicate_customer_ids_removed"] = dup_ids
        df = df.drop_duplicates(subset=["customer_id"])
    else:
        report["duplicate_customer_ids_removed"] = 0

    # Drop unused identifier columns
    dropped = [c for c in ID_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    report["dropped_columns"] = dropped

    # Keep only the columns we need, in a fixed order
    ordered = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES + [TARGET]
    df = df[ordered].copy()

    # Type coercion
    for col in NUMERIC_FEATURES + BINARY_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip()

    # Missing values
    report["missing_before"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    report["rows_dropped_missing_target"] = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET])
    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in BINARY_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    report["missing_after"] = int(df.isna().sum().sum())

    # Outliers: IQR winsorising (cap instead of delete -> no information loss)
    outliers = {}
    for col in OUTLIER_COLUMNS:
        low, high = _iqr_bounds(df[col])
        n_low = int((df[col] < low).sum())
        n_high = int((df[col] > high).sum())
        outliers[col] = {
            "lower_bound": round(low, 2),
            "upper_bound": round(high, 2),
            "outliers_found": n_low + n_high,
        }
        df[col] = df[col].clip(lower=low, upper=high)
    report["outliers"] = outliers

    # Final dtypes
    for col in ["credit_score", "age", "tenure", "products_number"] + BINARY_FEATURES + [TARGET]:
        df[col] = df[col].round().astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    df = df.reset_index(drop=True)
    report["rows_after"] = int(len(df))
    return df, report


# --------------------------------------------------------------------------- #
# 3. Encoding, splitting and scaling
# --------------------------------------------------------------------------- #
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode 'country' and 'gender' (drop_first avoids collinearity)."""
    features = df.drop(columns=[TARGET], errors="ignore")
    return pd.get_dummies(
        features, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int
    )


def apply_scaling(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Standardise the continuous columns; binary / dummy columns stay as 0-1."""
    X_scaled = X.copy()
    X_scaled[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
    return X_scaled


def prepare_dataset(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """
    Build model-ready data.

    The scaler is fitted on the training split only (prevents data leakage).
    """
    X = encode_features(df)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler().fit(X_train[NUMERIC_FEATURES])

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": apply_scaling(X_train, scaler),
        "X_test_scaled": apply_scaling(X_test, scaler),
        "scaler": scaler,
        "feature_columns": list(X.columns),
    }


def encode_single_customer(customer: dict, feature_columns: list) -> pd.DataFrame:
    """Turn one customer's raw inputs into the encoded layout used in training."""
    row = pd.DataFrame([customer])
    row = pd.get_dummies(row, columns=CATEGORICAL_FEATURES, dtype=int)
    return row.reindex(columns=feature_columns, fill_value=0)
