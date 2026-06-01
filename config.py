"""
Central configuration: paths, constants, and the global random seed.

This is the single source of truth for filesystem layout and reproducibility
settings. Every pipeline/script/module imports paths from here instead of
hardcoding strings.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---
RANDOM_STATE: int = 42

# --- Model hyperparameters (defaults) ---
DEFAULT_XI: int = 8
DEFAULT_TAU: float = 0.05

# --- Project root (this file lives at the project root) ---
ROOT: Path = Path(__file__).resolve().parent

# --- Data folders ---
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# --- Model artifacts ---
MODELS_DIR: Path = ROOT / "models"

# --- Results ---
RESULTS_DIR: Path = ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# --- Raw data files ---
RAW_CUSTOMERS_CSV: Path = RAW_DIR / "customers_raw.csv"
RAW_CORRUPTED_CSV: Path = RAW_DIR / "customers_corrupted.csv"
ONLINE_RETAIL_XLSX: Path = RAW_DIR / "online_retail_ii.xlsx"

# --- Processed data files ---
CUSTOMER_PROFILES_CSV: Path = PROCESSED_DIR / "customer_profiles.csv"
X_TRAIN_SCALED_CSV: Path = PROCESSED_DIR / "X_train_scaled.csv"
X_TEST_SCALED_CSV: Path = PROCESSED_DIR / "X_test_scaled.csv"
TRAIN_LABELS_CSV: Path = PROCESSED_DIR / "train_labels.csv"
TEST_LABELS_CSV: Path = PROCESSED_DIR / "test_labels.csv"

# --- Model files ---
MODEL_PKL: Path = MODELS_DIR / "clique_model.pkl"
SCALER_PKL: Path = MODELS_DIR / "scaler.pkl"
PROFILES_PKL: Path = MODELS_DIR / "profiles.pkl"

# --- Result files ---
COMPARISON_METRICS_CSV: Path = METRICS_DIR / "model_comparison.csv"
CLIQUE_CLASSIFICATION_CSV: Path = METRICS_DIR / "clique_classification_report.csv"
GRID_SEARCH_CSV: Path = METRICS_DIR / "clique_grid_search.csv"

# --- Feature schema (single source of truth) ---
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

# Ground-truth segment names for the synthetic dataset (used in evaluation).
SEGMENT_NAMES: list[str] = ["high_value", "at_risk", "loyal_mid"]


def ensure_dirs() -> None:
    """Create all output directories if they do not yet exist."""
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
