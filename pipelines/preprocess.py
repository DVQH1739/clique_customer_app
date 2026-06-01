"""
Pipeline step 2 — preprocessing.

Loads raw customer profiles, applies log-transform to skewed features, performs a
reproducible train/test split (``random_state=42``) that carries ground-truth
labels alongside the features, fits a ``MinMaxScaler`` on the TRAIN split only,
and writes processed CSVs to ``data/processed/`` plus the fitted scaler to
``models/``.

Run:
    python pipelines/preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def run(test_size: float = 0.2) -> dict[str, object]:
    """Execute preprocessing and persist all processed artifacts."""
    config.ensure_dirs()

    if not config.RAW_CUSTOMERS_CSV.exists():
        raise FileNotFoundError(
            f"Raw data not found: {config.RAW_CUSTOMERS_CSV}. "
            "Run pipelines/generate_data.py first."
        )

    raw = pd.read_csv(config.RAW_CUSTOMERS_CSV)
    profiles = raw.copy()

    # Log-transform skewed features so equal-width CLIQUE bins are populated.
    for col in config.LOG_TRANSFORM_COLS:
        profiles[col] = np.log1p(profiles[col].clip(lower=0))

    X = profiles[config.FEATURE_NAMES].values.astype(float)
    y = profiles["true_segment"].values.astype(int)
    ids = profiles["CustomerID"].values

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Persist processed features (scaled, in [0, 1]).
    pd.DataFrame(X_train_scaled, columns=config.FEATURE_NAMES).to_csv(
        config.X_TRAIN_SCALED_CSV, index=False
    )
    pd.DataFrame(X_test_scaled, columns=config.FEATURE_NAMES).to_csv(
        config.X_TEST_SCALED_CSV, index=False
    )
    # Persist labels aligned row-for-row with the scaled feature CSVs.
    pd.DataFrame(
        {"CustomerID": ids_train, "true_segment": y_train}
    ).to_csv(config.TRAIN_LABELS_CSV, index=False)
    pd.DataFrame(
        {"CustomerID": ids_test, "true_segment": y_test}
    ).to_csv(config.TEST_LABELS_CSV, index=False)
    # Persist the full log-transformed profile table for reference.
    profiles.to_csv(config.CUSTOMER_PROFILES_CSV, index=False)

    joblib.dump(scaler, config.SCALER_PKL)

    assert X_train_scaled.min() >= 0.0 and X_train_scaled.max() <= 1.0
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"Saved scaler -> {config.SCALER_PKL}")
    print(f"Processed CSVs -> {config.PROCESSED_DIR}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run()
