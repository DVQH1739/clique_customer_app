"""
Preprocessing: log-transform, train/test split, MinMaxScaler (fit on train only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def winsorize_features(
    profiles: pd.DataFrame,
    columns: list[str] | None = None,
    quantiles: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Clip extreme feature values before log-transform (stabilizes CLIQUE grids)."""
    cols = columns or config.FEATURE_NAMES
    qlo, qhi = quantiles or config.RETAIL_WINSORIZE_QUANTILES
    out = profiles.copy()
    for col in cols:
        if col not in out.columns:
            continue
        lo, hi = out[col].quantile([qlo, qhi])
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def _log_transform_frame(profiles: pd.DataFrame) -> pd.DataFrame:
    out = profiles.copy()
    for col in config.LOG_TRANSFORM_COLS:
        out[col] = np.log1p(out[col].clip(lower=0))
    return out


def plot_feature_distributions(
    profiles_raw: pd.DataFrame,
    profiles_transformed: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Before/after histograms for all features."""
    out = output_path or config.EDA_DISTRIBUTIONS_PNG
    config.ensure_dirs()
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle("Phân phối features: Trước (trên) vs Sau transform (dưới)", fontsize=14)
    for i, col in enumerate(config.FEATURE_NAMES):
        axes[0, i].hist(
            profiles_raw[col], bins=40, color="steelblue", edgecolor="white", linewidth=0.3
        )
        axes[0, i].set_title(col, fontsize=8)
        axes[1, i].hist(
            profiles_transformed[col], bins=40, color="coral", edgecolor="white", linewidth=0.3
        )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")


def run_synthetic(test_size: float = 0.2) -> dict[str, object]:
    """Preprocess synthetic raw CSV (stratified split + labels)."""
    config.ensure_dirs()
    if not config.RAW_CUSTOMERS_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_CUSTOMERS_CSV} missing. Run: python -m pipelines.run synthetic --step data"
        )

    raw = pd.read_csv(config.RAW_CUSTOMERS_CSV)
    profiles = _log_transform_frame(raw)

    X = profiles[config.FEATURE_NAMES].to_numpy(dtype=float)
    y = profiles["true_segment"].to_numpy(dtype=int)
    ids = profiles["CustomerID"].astype(str).to_numpy(dtype=object)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=config.FEATURE_NAMES).to_csv(
        config.X_TRAIN_SCALED_CSV, index=False
    )
    pd.DataFrame(X_test_scaled, columns=config.FEATURE_NAMES).to_csv(
        config.X_TEST_SCALED_CSV, index=False
    )
    pd.DataFrame({"CustomerID": ids_train, "true_segment": y_train}).to_csv(
        config.TRAIN_LABELS_CSV, index=False
    )
    pd.DataFrame({"CustomerID": ids_test, "true_segment": y_test}).to_csv(
        config.TEST_LABELS_CSV, index=False
    )
    profiles.to_csv(config.CUSTOMER_PROFILES_CSV, index=False)
    config.SYNTHETIC_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.SYNTHETIC_MODELS / "scaler.pkl")

    assert X_train_scaled.min() >= 0.0 and X_train_scaled.max() <= 1.0
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"Scaler -> {config.SYNTHETIC_MODELS / 'scaler.pkl'}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_test": ids_test,
        "scaler": scaler,
    }


def run_retail(
    profiles: pd.DataFrame,
    test_size: float = 0.2,
    plot_eda: bool = True,
    winsorize: bool = True,
) -> dict[str, object]:
    """Preprocess retail customer profiles (no ground-truth labels)."""
    config.ensure_dirs()
    profiles_raw = profiles.copy()
    profiles_work = profiles.copy()

    if winsorize:
        print("Step 3.0: Winsorize outliers ...")
        profiles_work = winsorize_features(profiles_work)

    print("Step 3.1: Log-transform ...")
    for col in config.LOG_TRANSFORM_COLS:
        before = profiles_work[col].skew()
        profiles_work[col] = np.log1p(profiles_work[col].clip(lower=0))
        after = profiles_work[col].skew()
        print(f"  {col:<25} skew: {before:+.2f} -> {after:+.2f}")

    X = profiles_work[config.FEATURE_NAMES].values
    ids = profiles_work["CustomerID"].astype(str).to_numpy(dtype=object)
    X_train, X_test, ids_train, ids_test = train_test_split(
        X, ids, test_size=test_size, random_state=config.RANDOM_STATE
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    assert X_train_scaled.min() >= -1e-9 and X_train_scaled.max() <= 1 + 1e-9
    print(f"  Train range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    print(f"  Test range:  [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")

    config.RETAIL_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.RETAIL_SCALER_PKL)
    pd.DataFrame(X_train_scaled, columns=config.FEATURE_NAMES).assign(
        CustomerID=ids_train
    ).to_csv(config.RETAIL_X_TRAIN_SCALED_CSV, index=False)
    pd.DataFrame(X_test_scaled, columns=config.FEATURE_NAMES).assign(
        CustomerID=ids_test
    ).to_csv(config.RETAIL_X_TEST_SCALED_CSV, index=False)

    if plot_eda:
        transformed = profiles_raw.copy()
        for col in config.LOG_TRANSFORM_COLS:
            transformed[col] = np.log1p(transformed[col].clip(lower=0))
        plot_feature_distributions(profiles_raw, transformed)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "ids_train": ids_train,
        "ids_test": ids_test,
        "scaler": scaler,
    }
