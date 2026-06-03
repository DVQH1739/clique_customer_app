"""
Post-pipeline validation: data contract, scaled ranges, model, and result artifacts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import config
from clique.algorithm import CLIQUE


def validate_profiles(profiles: pd.DataFrame) -> None:
    missing = set(config.FEATURE_NAMES) - set(profiles.columns)
    if missing:
        raise ValueError(f"Profiles missing columns: {missing}")
    if profiles[config.FEATURE_NAMES].isna().any().any():
        raise ValueError("Profiles contain NaN")
    if np.isinf(profiles[config.FEATURE_NAMES].values).any():
        raise ValueError("Profiles contain inf")
    if (profiles["frequency"] < config.MIN_FREQUENCY).any():
        raise ValueError(f"Profiles with frequency < {config.MIN_FREQUENCY}")


def validate_scaled_splits() -> None:
    for path in (config.X_TRAIN_SCALED_CSV, config.X_TEST_SCALED_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Missing scaled split: {path}")
        df = pd.read_csv(path)
        X = df[config.FEATURE_NAMES].values.astype(float)
        if X.min() < -1e-6 or X.max() > 1.0 + 1e-6:
            raise ValueError(f"{path.name}: scaled values outside [0, 1]")
        if np.isnan(X).any():
            raise ValueError(f"{path.name}: contains NaN")


def validate_model(model: CLIQUE, X_train: np.ndarray) -> None:
    if not model.clusters_:
        raise ValueError("CLIQUE produced no clusters")
    if model.labels_ is None:
        raise ValueError("CLIQUE labels_ not set after fit")
    n_cl = len(set(model.labels_.tolist()) - {-1})
    if n_cl < 2:
        raise ValueError("Fewer than 2 non-noise clusters")
    mask = model.labels_ >= 0
    if mask.sum() >= 2 and n_cl >= 2:
        sil = silhouette_score(X_train[mask], model.labels_[mask])
        if sil < -0.15:
            raise ValueError(f"Train silhouette unusually low: {sil:.4f}")


def validate_metrics_artifacts(model: CLIQUE, X_train: np.ndarray) -> None:
    required_csv = [
        config.GRID_SEARCH_CSV,
        config.BASELINE_COMPARISON_CSV,
        config.CLUSTER_DESCRIPTIONS_CSV,
        config.TEST_PREDICTIONS_CSV,
        config.METRICS_DIR / "clique_subspace_coverage.csv",
    ]
    for p in required_csv:
        if not p.exists() or p.stat().st_size < 10:
            raise FileNotFoundError(f"Missing or empty metrics file: {p}")

    grid = pd.read_csv(config.GRID_SEARCH_CSV)
    if "silhouette" not in grid.columns or "quality_score" not in grid.columns:
        raise ValueError("Grid search CSV missing expected columns")

    base = pd.read_csv(config.BASELINE_COMPARISON_CSV)
    cl = base[base["algorithm"] == "CLIQUE"].iloc[0]
    mask = model.labels_ >= 0
    sil = silhouette_score(X_train[mask], model.labels_[mask])
    if abs(sil - float(cl["silhouette"])) > 0.01:
        raise ValueError(
            f"CLIQUE silhouette mismatch: model={sil:.4f} csv={cl['silhouette']:.4f}"
        )

    test = pd.read_csv(config.TEST_PREDICTIONS_CSV)
    if "cluster_id" not in test.columns or len(test) < 10:
        raise ValueError("test_predictions.csv invalid")

    required_png = [
        config.EDA_DISTRIBUTIONS_PNG,
        config.BASELINE_COMPARISON_PNG,
        config.FIGURES_DIR / "subspace_heatmap.png",
        config.FIGURES_DIR / "cluster_sizes.png",
    ]
    for p in required_png:
        if not p.exists() or p.stat().st_size < 1000:
            raise FileNotFoundError(f"Missing or empty figure: {p}")


def validate_retail_pipeline(
    profiles: pd.DataFrame,
    model: CLIQUE,
    X_train: np.ndarray,
) -> None:
    """Run all checks; raises on failure."""
    if not config.ONLINE_RETAIL_XLSX.exists():
        raise FileNotFoundError(
            f"Raw workbook not found: {config.ONLINE_RETAIL_XLSX}. "
            f"Place online_retail_ii.xlsx in data/raw/."
        )
    validate_profiles(profiles)
    validate_scaled_splits()
    validate_model(model, X_train)
    validate_metrics_artifacts(model, X_train)
    print("Validation OK: data, model, metrics, and figures.")
