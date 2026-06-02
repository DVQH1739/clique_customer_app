"""
Evaluation metrics for clustering, with and without ground-truth labels.

Two families are provided:

1. Intrinsic (no labels): silhouette, Davies-Bouldin, Calinski-Harabasz.
2. External / supervised (require ground-truth segment labels):
   - Cluster-vs-truth agreement: ARI, NMI, homogeneity, completeness, V-measure.
   - Classification-style: accuracy, precision/recall/F1 (macro & weighted),
     confusion matrix.

Because CLIQUE is unsupervised, predicted cluster ids carry no inherent class
meaning. To compute classification-style metrics we map each predicted cluster
to the ground-truth label most frequent among its members (majority vote), which
handles an arbitrary number of clusters and noise (-1) gracefully.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
    v_measure_score,
)


# ---------------------------------------------------------------------------
# Intrinsic metrics (no ground truth)
# ---------------------------------------------------------------------------

def _valid_cluster_view(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return X/labels with noise (-1) removed."""
    labels = np.asarray(labels)
    mask = labels >= 0
    return X[mask], labels[mask]


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score over non-noise points; NaN if fewer than 2 clusters."""
    Xv, lv = _valid_cluster_view(X, labels)
    if len(Xv) < 2 or len(set(lv.tolist())) < 2:
        return float("nan")
    try:
        return float(silhouette_score(Xv, lv))
    except Exception:
        return float("nan")


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index (lower is better); NaN if fewer than 2 clusters."""
    Xv, lv = _valid_cluster_view(X, labels)
    if len(Xv) < 2 or len(set(lv.tolist())) < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(Xv, lv))
    except Exception:
        return float("nan")


def compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz index (higher is better); NaN if fewer than 2 clusters."""
    Xv, lv = _valid_cluster_view(X, labels)
    if len(Xv) < 2 or len(set(lv.tolist())) < 2:
        return float("nan")
    try:
        return float(calinski_harabasz_score(Xv, lv))
    except Exception:
        return float("nan")


def intrinsic_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Bundle the three intrinsic metrics into a dict."""
    return {
        "silhouette": compute_silhouette(X, labels),
        "davies_bouldin": compute_davies_bouldin(X, labels),
        "calinski_harabasz": compute_calinski_harabasz(X, labels),
    }


# ---------------------------------------------------------------------------
# Label alignment (cluster id -> ground-truth class)
# ---------------------------------------------------------------------------

def majority_vote_mapping(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[int, int]:
    """
    Map each predicted cluster id to its most frequent ground-truth label.

    Works for any number of clusters and for noise label -1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mapping: dict[int, int] = {}
    true_classes = np.unique(y_true)
    fallback = int(true_classes[np.argmax([np.sum(y_true == c) for c in true_classes])])
    for cluster in np.unique(y_pred):
        members = y_true[y_pred == cluster]
        if len(members) == 0:
            mapping[int(cluster)] = fallback
            continue
        vals, counts = np.unique(members, return_counts=True)
        mapping[int(cluster)] = int(vals[np.argmax(counts)])
    return mapping


def align_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return predicted labels re-expressed in the ground-truth label space."""
    mapping = majority_vote_mapping(y_true, y_pred)
    return np.array([mapping[int(p)] for p in y_pred], dtype=int)


def supervised_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Classification-style metrics after majority-vote alignment.

    Includes accuracy and macro/weighted precision-recall-F1. These describe how
    well the clusters, once mapped to segments, recover the ground-truth labels.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    aligned = align_predictions(y_true, y_pred)

    return {
        "accuracy": float(np.mean(aligned == y_true)),
        "precision_macro": float(
            precision_score(y_true, aligned, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, aligned, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, aligned, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, aligned, average="weighted", zero_division=0)
        ),
    }


def clustering_agreement_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """ARI, NMI, homogeneity, completeness, V-measure (label-permutation invariant)."""
    return {
        "adjusted_rand": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
    }


def confusion_matrix_aligned(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (confusion_matrix, class_labels) after majority-vote alignment."""
    y_true = np.asarray(y_true).astype(int)
    aligned = align_predictions(y_true, np.asarray(y_pred).astype(int))
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, aligned, labels=classes)
    return cm, classes


def evaluate_labeling(
    X: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray | None,
    name: str,
    params: str = "",
) -> dict[str, Any]:
    """
    Single-row evaluation combining intrinsic and (if available) supervised metrics.
    """
    y_pred = np.asarray(y_pred)
    n_clusters = len(set(y_pred.tolist()) - {-1})
    row: dict[str, Any] = {
        "algorithm": name,
        "params": params,
        "n_clusters": n_clusters,
        "noise_pct": float(100.0 * np.mean(y_pred == -1)),
    }
    row.update(intrinsic_metrics(X, y_pred))
    if y_true is not None:
        row.update(supervised_metrics(y_true, y_pred))
        row.update(clustering_agreement_metrics(y_true, y_pred))
    return row
