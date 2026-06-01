"""
Pipeline step 1 — synthetic data generation.

Produces raw, natural-scale customer profiles with a known ground-truth segment
label (so downstream evaluation can compute F1 / ROC / AUC), plus a deliberately
corrupted file for error-handling tests. Outputs land in ``data/raw/``.

Run:
    python pipelines/generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

N_CUSTOMERS = 600
N_CORRUPTED = 200

# Per-segment natural-scale parameter ranges: (low, high) uniform draws.
SEGMENTS: list[dict[str, tuple[float, float]]] = [
    {  # 0: high_value
        "recency": (1, 45),
        "frequency": (20, 60),
        "monetary": (4000, 20000),
        "product_diversity": (120, 320),
        "return_rate": (0.0, 0.08),
        "weekend_ratio": (0.05, 0.30),
        "repeat_category_rate": (0.50, 0.85),
    },
    {  # 1: at_risk
        "recency": (200, 365),
        "frequency": (1, 6),
        "monetary": (50, 800),
        "product_diversity": (1, 40),
        "return_rate": (0.15, 0.45),
        "weekend_ratio": (0.40, 0.80),
        "repeat_category_rate": (0.05, 0.30),
    },
    {  # 2: loyal_mid
        "recency": (40, 160),
        "frequency": (8, 20),
        "monetary": (900, 4000),
        "product_diversity": (40, 120),
        "return_rate": (0.03, 0.18),
        "weekend_ratio": (0.20, 0.55),
        "repeat_category_rate": (0.35, 0.65),
    },
]


def generate_raw() -> pd.DataFrame:
    """Generate natural-scale profiles with a ground-truth ``true_segment`` column."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n_per = N_CUSTOMERS // 3
    counts = [n_per, n_per, N_CUSTOMERS - 2 * n_per]

    chunks: list[pd.DataFrame] = []
    for seg_idx, (params, n) in enumerate(zip(SEGMENTS, counts)):
        data: dict[str, np.ndarray] = {}
        for feat, (lo, hi) in params.items():
            data[feat] = rng.uniform(lo, hi, size=n)
        # avg_basket derived to stay internally consistent with monetary/frequency.
        data["avg_basket"] = data["monetary"] / np.maximum(data["frequency"], 1)
        df = pd.DataFrame(data)
        df["true_segment"] = seg_idx
        df["segment_name"] = config.SEGMENT_NAMES[seg_idx]
        chunks.append(df)

    out = pd.concat(chunks, ignore_index=True)
    out = out.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    out["CustomerID"] = [f"C{i:05d}" for i in range(len(out))]

    ordered = ["CustomerID", *config.FEATURE_NAMES, "true_segment", "segment_name"]
    return out[ordered]


def generate_corrupted(raw: pd.DataFrame) -> pd.DataFrame:
    """Corrupt a copy of the raw data: missing columns, string cells, NaNs."""
    df = raw.head(N_CORRUPTED).drop(columns=["true_segment", "segment_name"]).copy()
    df = df.drop(columns=["weekend_ratio", "repeat_category_rate"])
    # Cast to object so string values can coexist with numerics (mixed dtype).
    df["monetary"] = df["monetary"].astype(object)
    df.loc[0:9, "monetary"] = "invalid"
    df.loc[10:19, "frequency"] = np.nan
    df.loc[20:24, "recency"] = np.nan
    return df


def main() -> None:
    """Generate and persist raw + corrupted datasets."""
    config.ensure_dirs()
    raw = generate_raw()
    corrupted = generate_corrupted(raw)
    raw.to_csv(config.RAW_CUSTOMERS_CSV, index=False, encoding="utf-8")
    corrupted.to_csv(config.RAW_CORRUPTED_CSV, index=False, encoding="utf-8")
    print(f"Wrote {config.RAW_CUSTOMERS_CSV} ({len(raw)} rows)")
    print(f"Wrote {config.RAW_CORRUPTED_CSV} ({len(corrupted)} rows)")
    print("Segment counts:\n", raw["segment_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
