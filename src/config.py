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



XI_GRID: list[int] = [8, 10, 12]

TAU_GRID: list[float] = [0.10, 0.12, 0.14, 0.16, 0.18]

MIN_CLUSTERS: int = 4

MAX_CLUSTERS: int = 10

MIN_COVERAGE: float = 0.85



# --- Project layout ---

SRC_DIR: Path = Path(__file__).resolve().parent

ROOT: Path = SRC_DIR.parent



DATA_DIR: Path = ROOT / "data"

RAW_DIR: Path = DATA_DIR / "raw"

PROCESSED_DIR: Path = DATA_DIR / "processed"



MODELS_DIR: Path = ROOT / "models"



RESULTS_DIR: Path = ROOT / "results"

METRICS_DIR: Path = RESULTS_DIR / "metrics"

FIGURES_DIR: Path = RESULTS_DIR / "figures"



# --- Raw input (Online Retail II Excel only) ---

ONLINE_RETAIL_XLSX: Path = RAW_DIR / "online_retail_ii.xlsx"



MIN_FREQUENCY: int = 2

WINSORIZE_QUANTILES: tuple[float, float] = (0.01, 0.99)



# --- Processed CSV caches (written by pipeline) ---

CLEAN_TRANSACTIONS_CSV: Path = PROCESSED_DIR / "clean_transactions.csv"

CANCELLATIONS_CSV: Path = PROCESSED_DIR / "cancellations.csv"

CUSTOMER_PROFILES_RAW_CSV: Path = PROCESSED_DIR / "customer_profiles_raw.csv"

X_TRAIN_SCALED_CSV: Path = PROCESSED_DIR / "X_train_scaled.csv"

X_TEST_SCALED_CSV: Path = PROCESSED_DIR / "X_test_scaled.csv"



# --- Trained models ---

MODEL_PKL: Path = MODELS_DIR / "clique_model.pkl"

SCALER_PKL: Path = MODELS_DIR / "scaler.pkl"

PROFILES_PKL: Path = MODELS_DIR / "profiles.pkl"



# --- Results ---

GRID_SEARCH_CSV: Path = METRICS_DIR / "retail_grid_search.csv"

BASELINE_COMPARISON_CSV: Path = METRICS_DIR / "baseline_comparison.csv"

BASELINE_COMPARISON_PNG: Path = FIGURES_DIR / "baseline_comparison.png"

CLUSTER_DESCRIPTIONS_CSV: Path = METRICS_DIR / "cluster_descriptions.csv"

TEST_PREDICTIONS_CSV: Path = METRICS_DIR / "test_predictions.csv"

EDA_DISTRIBUTIONS_PNG: Path = FIGURES_DIR / "EDA_distributions.png"



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





def ensure_dirs() -> None:

    """Create output directories."""

    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR):

        d.mkdir(parents=True, exist_ok=True)


