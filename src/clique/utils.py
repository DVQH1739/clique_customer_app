"""
Utilities for CLIQUE customer clustering: metrics, I/O, data pipeline.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from clique.algorithm import CLIQUE
from clique.metrics import (
    compute_calinski_harabasz,
    compute_davies_bouldin,
    compute_silhouette,
)

__all_metrics__ = [
    "compute_silhouette",
    "compute_davies_bouldin",
    "compute_calinski_harabasz",
]

# Single source of truth for feature columns
FEATURE_NAMES: list[str] = [
    "recency",
    "frequency",
    "monetary",
    "avg_basket",
    "product_diversity",
    "return_rate",
    "weekend_ratio",
    "repeat_category_rate",
]

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "recency": "Recency (days since last purchase)",
    "frequency": "Frequency (number of orders)",
    "monetary": "Monetary (total spend £)",
    "avg_basket": "Avg Basket Size (£ per order)",
    "product_diversity": "Product Diversity (unique products)",
    "return_rate": "Return Rate (fraction returned)",
    "weekend_ratio": "Weekend Ratio (fraction on weekends)",
    "repeat_category_rate": "Brand Loyalty (repeat category rate)",
}

LOG_TRANSFORM_COLS: list[str] = [
    "monetary",
    "frequency",
    "avg_basket",
    "product_diversity",
]

MODEL_VERSION: str = "v1"
RANDOM_STATE: int = 42


def run_baseline_comparison(X: np.ndarray) -> pd.DataFrame:
    """
    Train KMeans, DBSCAN, AgglomerativeClustering; return metrics DataFrame.
    """
    rows: list[dict[str, Any]] = []
    configs = [
        ("KMeans", {"n_clusters": 4}, KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)),
        ("KMeans", {"n_clusters": 5}, KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)),
        ("KMeans", {"n_clusters": 6}, KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)),
        ("DBSCAN", {"eps": 0.3, "min_samples": 5}, DBSCAN(eps=0.3, min_samples=5)),
        ("DBSCAN", {"eps": 0.5, "min_samples": 5}, DBSCAN(eps=0.5, min_samples=5)),
        (
            "AgglomerativeClustering",
            {"n_clusters": 5},
            AgglomerativeClustering(n_clusters=5),
        ),
    ]
    for algo_name, params, estimator in configs:
        t0 = time.perf_counter()
        labels = estimator.fit_predict(X)
        runtime = time.perf_counter() - t0
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        rows.append(
            {
                "algorithm": algo_name,
                "params": str(params),
                "n_clusters": n_clusters,
                "silhouette": compute_silhouette(X, labels),
                "davies_bouldin": compute_davies_bouldin(X, labels),
                "calinski_harabasz": compute_calinski_harabasz(X, labels),
                "runtime_sec": runtime,
            }
        )
    return pd.DataFrame(rows)


def save_model(
    model: CLIQUE,
    scaler: MinMaxScaler,
    profiles: dict[int, np.ndarray],
    output_dir: str = "models/",
) -> None:
    """Persist model, scaler, and cluster profiles with versioned filenames."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"clique_model_{MODEL_VERSION}.pkl")
    joblib.dump(scaler, out / f"scaler_{MODEL_VERSION}.pkl")
    joblib.dump(profiles, out / f"profiles_{MODEL_VERSION}.pkl")
    # Also write unversioned aliases for app compatibility
    joblib.dump(model, out / "clique_model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    joblib.dump(profiles, out / "profiles.pkl")


def load_model(model_dir: str = "models/") -> tuple[CLIQUE, MinMaxScaler, dict[int, np.ndarray]]:
    """Load model, scaler, and profiles from disk."""
    base = Path(model_dir)
    for name in ("clique_model.pkl", f"clique_model_{MODEL_VERSION}.pkl"):
        model_path = base / name
        if model_path.exists():
            break
    else:
        raise FileNotFoundError(f"No CLIQUE model found in {model_dir}")

    scaler_path = base / "scaler.pkl"
    if not scaler_path.exists():
        scaler_path = base / f"scaler_{MODEL_VERSION}.pkl"
    profiles_path = base / "profiles.pkl"
    if not profiles_path.exists():
        profiles_path = base / f"profiles_{MODEL_VERSION}.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    profiles = joblib.load(profiles_path)
    return model, scaler, profiles


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load Online Retail II Excel or CSV; standardize column names.

    Expected columns after load:
    InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
    UnitPrice, CustomerID, Country
    """
    if filepath.endswith((".xlsx", ".xls")):
        # Online Retail II has two year sheets; Online Retail I has one. Read every
        # sheet and concatenate so both layouts work without manual configuration.
        sheets = pd.read_excel(filepath, sheet_name=None, dtype={"Customer ID": str})
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(filepath, encoding="latin-1", dtype={"CustomerID": str})

    df.rename(
        columns={
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
            "Customer ID": "CustomerID",
        },
        inplace=True,
    )
    return df


def clean_data(
    df: pd.DataFrame, filter_uk: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean transaction data; return (clean_df, cancellations_df).
    """
    original_rows = len(df)
    print(f"Original rows: {original_rows}")

    # 1. Drop null/empty CustomerID
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df = df[df["CustomerID"].notna() & (df["CustomerID"] != "") & (df["CustomerID"] != "nan")]
    print(f"After dropping null CustomerID: {len(df)}")

    # 2. Capture cancellations FIRST. Cancellation invoices start with 'C' and carry
    #    negative quantity, so they must be separated before the Quantity > 0 filter
    #    below removes them (otherwise return_rate would always be 0).
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
    cancellations_df = df[is_cancel].copy()
    df = df[~is_cancel]
    print(f"Cancellation invoices captured: {len(cancellations_df)}")

    # 3. Valid quantity and price (negatives/zeros are returns or data-entry errors)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    print(f"After dropping invalid Quantity/UnitPrice: {len(df)}")

    # 4. Exact duplicate line items
    dup_cols = [c for c in ["InvoiceNo", "StockCode", "Quantity", "UnitPrice"] if c in df.columns]
    if dup_cols:
        df = df.drop_duplicates(subset=dup_cols)

    # 5. Parse InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[df["InvoiceDate"].notna()]

    # 6. Revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # 7. UK filter (optional) to reduce noise
    if filter_uk and "Country" in df.columns:
        df = df[df["Country"] == "United Kingdom"]

    print(f"Final rows: {len(df)}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    if len(df) > 0:
        print(
            f"Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}"
        )

    return df, cancellations_df


def build_customer_profiles(
    clean_df: pd.DataFrame,
    cancellations_df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Aggregate to one row per CustomerID with 8 RFM-style features.
    """
    if snapshot_date is None:
        snapshot_date = clean_df["InvoiceDate"].max() + timedelta(days=1)

    grouped = clean_df.groupby("CustomerID")

    recency = (snapshot_date - grouped["InvoiceDate"].max()).dt.days
    frequency = grouped["InvoiceNo"].nunique()
    monetary = grouped["Revenue"].sum()
    avg_basket = monetary / frequency.replace(0, np.nan)

    product_diversity = grouped["StockCode"].nunique()

    cancel_counts = cancellations_df.groupby("CustomerID")["InvoiceNo"].nunique()
    purchase_counts = grouped["InvoiceNo"].nunique()
    return_rate = cancel_counts / (cancel_counts + purchase_counts)
    return_rate = return_rate.fillna(0)

    clean_df = clean_df.copy()
    clean_df["is_weekend"] = clean_df["InvoiceDate"].dt.dayofweek >= 5
    weekend_ratio = clean_df.groupby("CustomerID")["is_weekend"].mean()

    clean_df["category"] = clean_df["StockCode"].astype(str).str[:2]
    cat_counts = (
        clean_df.groupby(["CustomerID", "category"]).size().reset_index(name="cnt")
    )
    top_cat = cat_counts.loc[cat_counts.groupby("CustomerID")["cnt"].idxmax()]
    total_lines = clean_df.groupby("CustomerID").size()
    repeat_category_rate = top_cat.set_index("CustomerID")["cnt"] / total_lines
    repeat_category_rate = repeat_category_rate.fillna(0)

    profiles = pd.DataFrame(
        {
            "CustomerID": recency.index,
            "recency": recency.values,
            "frequency": frequency.reindex(recency.index).values,
            "monetary": monetary.reindex(recency.index).values,
            "avg_basket": avg_basket.reindex(recency.index).fillna(0).values,
            "product_diversity": product_diversity.reindex(recency.index).values,
            "return_rate": return_rate.reindex(recency.index).fillna(0).values,
            "weekend_ratio": weekend_ratio.reindex(recency.index).fillna(0).values,
            "repeat_category_rate": repeat_category_rate.reindex(recency.index).fillna(0).values,
        }
    )

    profiles = profiles.fillna(0)
    assert not profiles[FEATURE_NAMES].isna().any().any()
    assert not np.isinf(profiles[FEATURE_NAMES].values).any()

    print(profiles[FEATURE_NAMES].describe())
    skew = profiles[FEATURE_NAMES].skew()
    for col in FEATURE_NAMES:
        if abs(skew[col]) > 1:
            print(f"  {col}: skew={skew[col]:.2f} — needs transform")

    return profiles


def preprocess(
    customer_profiles: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    log_transform_cols: list[str] | None = None,
    models_dir: str = "models/",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Log-transform, train/test split, MinMaxScaler on train only; save artifacts.
    """
    if log_transform_cols is None:
        log_transform_cols = LOG_TRANSFORM_COLS

    profiles = customer_profiles.copy()
    for col in log_transform_cols:
        if col in profiles.columns:
            profiles[col] = np.log1p(profiles[col].clip(lower=0))

    X = profiles[FEATURE_NAMES].values
    customer_ids = profiles["CustomerID"].values

    X_train, X_test, ids_train, ids_test = train_test_split(
        X, customer_ids, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out / "scaler.pkl")
    pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES).to_csv(
        out / "X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES).to_csv(
        out / "X_test_scaled.csv", index=False
    )
    profiles.to_csv(out / "customer_profiles.csv", index=False)

    assert scaler.data_min_ is not None
    assert X_train_scaled.min() >= 0.0
    assert X_train_scaled.max() <= 1.0
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, ids_train, ids_test, scaler


def validate_profile_csv(df: pd.DataFrame) -> None:
    """Raise ValueError if profile CSV does not match DATA_CONTRACT."""
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if df[FEATURE_NAMES].isna().any().any():
        raise ValueError("Null values found in feature columns")


def apply_log_transform(profiles: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to skewed columns (for inference on raw-scale profiles)."""
    out = profiles.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in out.columns:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out
