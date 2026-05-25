"""
Generate mock customer profile CSVs for UI testing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clique.utils import FEATURE_NAMES

RNG = np.random.default_rng(42)
N_CLEAN = 500
N_CORRUPTED = 200
OUTPUT_DIR = Path(__file__).resolve().parent


def generate_clean() -> pd.DataFrame:
    """500 rows, 8 features in [0,1], 3 synthetic clusters."""
    n_per = N_CLEAN // 3
    remainder = N_CLEAN - 3 * n_per

    centers = [
        {  # high value
            "recency": (0.05, 0.15),
            "frequency": (0.70, 0.95),
            "monetary": (0.75, 0.98),
            "avg_basket": (0.70, 0.90),
            "product_diversity": (0.60, 0.85),
            "return_rate": (0.00, 0.10),
            "weekend_ratio": (0.10, 0.35),
            "repeat_category_rate": (0.55, 0.80),
        },
        {  # at risk
            "recency": (0.70, 0.95),
            "frequency": (0.15, 0.40),
            "monetary": (0.10, 0.35),
            "avg_basket": (0.15, 0.40),
            "product_diversity": (0.10, 0.35),
            "return_rate": (0.20, 0.45),
            "weekend_ratio": (0.40, 0.70),
            "repeat_category_rate": (0.10, 0.30),
        },
        {  # loyal mid-tier
            "recency": (0.20, 0.45),
            "frequency": (0.45, 0.70),
            "monetary": (0.40, 0.65),
            "avg_basket": (0.35, 0.60),
            "product_diversity": (0.35, 0.60),
            "return_rate": (0.05, 0.20),
            "weekend_ratio": (0.25, 0.50),
            "repeat_category_rate": (0.40, 0.65),
        },
    ]

    chunks: list[pd.DataFrame] = []
    counts = [n_per, n_per, n_per + remainder]
    for center, n in zip(centers, counts):
        data = {}
        for feat in FEATURE_NAMES:
            lo, hi = center[feat]
            data[feat] = RNG.uniform(lo, hi, size=n)
        chunks.append(pd.DataFrame(data))

    df = pd.concat(chunks, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["CustomerID"] = [f"C{i:05d}" for i in range(len(df))]
    cols = ["CustomerID"] + FEATURE_NAMES
    return df[cols]


def generate_corrupted() -> pd.DataFrame:
    """200 rows with missing columns, strings, and NaNs for error-handling tests."""
    df = generate_clean().head(N_CORRUPTED).copy()
    # Drop two feature columns
    df = df.drop(columns=["weekend_ratio", "repeat_category_rate"])
    # Inject string values in monetary
    df.loc[0:9, "monetary"] = "invalid"
    # Inject NaNs
    df.loc[10:19, "frequency"] = np.nan
    df.loc[20:24, "recency"] = np.nan
    return df


def main() -> None:
    """Write test_clean.csv and test_corrupted.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = generate_clean()
    corrupted = generate_corrupted()
    clean_path = OUTPUT_DIR / "test_clean.csv"
    corrupt_path = OUTPUT_DIR / "test_corrupted.csv"
    clean.to_csv(clean_path, index=False, encoding="utf-8")
    corrupted.to_csv(corrupt_path, index=False, encoding="utf-8")
    print(f"Wrote {clean_path} ({len(clean)} rows)")
    print(f"Wrote {corrupt_path} ({len(corrupted)} rows)")


if __name__ == "__main__":
    main()
