"""
Central configuration: paths, constants, and the global random seed.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---
RANDOM_STATE: int = 42

# --- Model hyperparameters (defaults) ---
DEFAULT_XI: int = 8
DEFAULT_TAU: float = 0.05

# --- Project layout ---
SRC_DIR: Path = Path(__file__).resolve().parent
ROOT: Path = SRC_DIR.parent

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DIR: Path = PROCESSED_DIR / "synthetic"
RETAIL_DIR: Path = PROCESSED_DIR / "retail"

MODELS_DIR: Path = ROOT / "models"
SYNTHETIC_MODELS: Path = MODELS_DIR / "synthetic"
RETAIL_MODELS: Path = MODELS_DIR / "retail"

RESULTS_DIR: Path = ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# --- Raw inputs ---
RAW_CUSTOMERS_CSV: Path = RAW_DIR / "customers_raw.csv"
ONLINE_RETAIL_XLSX: Path = RAW_DIR / "online_retail_ii.xlsx"

# --- Synthetic processed + models (Streamlit default) ---
CUSTOMER_PROFILES_CSV: Path = SYNTHETIC_DIR / "customer_profiles.csv"
X_TRAIN_SCALED_CSV: Path = SYNTHETIC_DIR / "X_train_scaled.csv"
X_TEST_SCALED_CSV: Path = SYNTHETIC_DIR / "X_test_scaled.csv"
TRAIN_LABELS_CSV: Path = SYNTHETIC_DIR / "train_labels.csv"
TEST_LABELS_CSV: Path = SYNTHETIC_DIR / "test_labels.csv"
MODEL_PKL: Path = SYNTHETIC_MODELS / "clique_model.pkl"
SCALER_PKL: Path = SYNTHETIC_MODELS / "scaler.pkl"
PROFILES_PKL: Path = SYNTHETIC_MODELS / "profiles.pkl"
COMPARISON_METRICS_CSV: Path = METRICS_DIR / "model_comparison.csv"
CLIQUE_CLASSIFICATION_CSV: Path = METRICS_DIR / "clique_classification_report.csv"
GRID_SEARCH_CSV: Path = METRICS_DIR / "clique_grid_search.csv"

# --- Retail processed + models ---
CLEAN_TRANSACTIONS_CSV: Path = RETAIL_DIR / "clean_transactions.csv"
CANCELLATIONS_CSV: Path = RETAIL_DIR / "cancellations.csv"
CUSTOMER_PROFILES_RAW_CSV: Path = RETAIL_DIR / "customer_profiles_raw.csv"
RETAIL_X_TRAIN_SCALED_CSV: Path = RETAIL_DIR / "X_train_scaled.csv"
RETAIL_X_TEST_SCALED_CSV: Path = RETAIL_DIR / "X_test_scaled.csv"
RETAIL_MODEL_PKL: Path = RETAIL_MODELS / "clique_model.pkl"
RETAIL_SCALER_PKL: Path = RETAIL_MODELS / "scaler.pkl"
RETAIL_PROFILES_PKL: Path = RETAIL_MODELS / "profiles.pkl"
EDA_DISTRIBUTIONS_PNG: Path = FIGURES_DIR / "EDA_distributions.png"
BASELINE_COMPARISON_CSV: Path = METRICS_DIR / "baseline_comparison.csv"
BASELINE_COMPARISON_PNG: Path = FIGURES_DIR / "baseline_comparison.png"
CLUSTER_DESCRIPTIONS_CSV: Path = METRICS_DIR / "cluster_descriptions.csv"
TEST_PREDICTIONS_CSV: Path = METRICS_DIR / "test_predictions.csv"
RETAIL_GRID_SEARCH_CSV: Path = METRICS_DIR / "retail_grid_search.csv"

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
    "recency",
    "frequency",
    "monetary",
    "avg_basket",
    "product_diversity",
]

SEGMENT_NAMES: list[str] = ["high_value", "at_risk", "loyal_mid"]


def ensure_dirs() -> None:
    """Create output directories."""
    for d in (
        RAW_DIR,
        SYNTHETIC_DIR,
        RETAIL_DIR,
        SYNTHETIC_MODELS,
        RETAIL_MODELS,
        METRICS_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
