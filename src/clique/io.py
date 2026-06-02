"""Model persistence and profile CSV validation."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
from clique.algorithm import CLIQUE


def validate_profile_csv(df: pd.DataFrame) -> None:
    """Raise ValueError if profile CSV does not match DATA_CONTRACT."""
    missing = [c for c in config.FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if df[config.FEATURE_NAMES].isna().any().any():
        raise ValueError("Null values found in feature columns")


def apply_log_transform(profiles: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to skewed columns (inference on raw-scale profiles)."""
    out = profiles.copy()
    for col in config.LOG_TRANSFORM_COLS:
        if col in out.columns:
            out[col] = np.log1p(out[col].clip(lower=0))
    return out


def load_model(model_dir: Path | str | None = None) -> tuple[CLIQUE, MinMaxScaler, object]:
    """Load CLIQUE model, scaler, and cluster profiles (default: synthetic)."""
    base = Path(model_dir or config.SYNTHETIC_MODELS)
    model_path = base / "clique_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No CLIQUE model at {model_path}")

    scaler_path = base / "scaler.pkl"
    profiles_path = base / "profiles.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"No scaler at {scaler_path}")
    if not profiles_path.exists():
        raise FileNotFoundError(f"No profiles at {profiles_path}")

    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(profiles_path)


def save_model(
    model: CLIQUE,
    scaler: MinMaxScaler,
    profiles: object,
    *,
    model_dir: Path | None = None,
) -> None:
    """Persist model, scaler, and cluster profiles."""
    config.ensure_dirs()
    base = model_dir or config.SYNTHETIC_MODELS
    base.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, base / "clique_model.pkl")
    joblib.dump(scaler, base / "scaler.pkl")
    joblib.dump(profiles, base / "profiles.pkl")
