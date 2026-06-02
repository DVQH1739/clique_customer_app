"""CLIQUE subspace clustering package."""

from clique.algorithm import CLIQUE, NOISE_LABEL
from clique.io import load_model, save_model
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

__all__ = [
    "CLIQUE",
    "NOISE_LABEL",
    "load_model",
    "save_model",
    "compute_silhouette",
    "compute_davies_bouldin",
    "compute_calinski_harabasz",
    "intrinsic_metrics",
    "supervised_metrics",
    "clustering_agreement_metrics",
    "evaluate_labeling",
    "align_predictions",
    "majority_vote_mapping",
    "confusion_matrix_aligned",
]
