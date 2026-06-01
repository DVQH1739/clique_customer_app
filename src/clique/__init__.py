"""CLIQUE customer subspace clustering package."""

from clique.algorithm import CLIQUE
from clique.metrics import (
    align_predictions,
    clustering_agreement_metrics,
    compute_calinski_harabasz,
    compute_davies_bouldin,
    compute_silhouette,
    confusion_matrix_aligned,
    evaluate_labeling,
    intrinsic_metrics,
    majority_vote_mapping,
    supervised_metrics,
)
from clique.utils import (
    FEATURE_DISPLAY_NAMES,
    FEATURE_NAMES,
    LOG_TRANSFORM_COLS,
    build_customer_profiles,
    clean_data,
    load_model,
    load_raw_data,
    preprocess,
    run_baseline_comparison,
    save_model,
    validate_profile_csv,
)

__all__ = [
    "CLIQUE",
    "FEATURE_NAMES",
    "FEATURE_DISPLAY_NAMES",
    "LOG_TRANSFORM_COLS",
    "build_customer_profiles",
    "clean_data",
    "compute_calinski_harabasz",
    "compute_davies_bouldin",
    "compute_silhouette",
    "intrinsic_metrics",
    "supervised_metrics",
    "clustering_agreement_metrics",
    "evaluate_labeling",
    "align_predictions",
    "majority_vote_mapping",
    "confusion_matrix_aligned",
    "load_model",
    "load_raw_data",
    "preprocess",
    "run_baseline_comparison",
    "save_model",
    "validate_profile_csv",
]
