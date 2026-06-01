"""
Evaluation metrics for clustering, with and without ground-truth labels.

Two families are provided:

1. Intrinsic (no labels): silhouette, Davies-Bouldin, Calinski-Harabasz.
2. External / supervised (require ground-truth segment labels):
   - Cluster-vs-truth agreement: ARI, NMI, homogeneity, completeness, V-measure.
   - Classification-style: accuracy, precision/recall/F1 (macro & weighted),
     ROC-AUC (one-vs-rest), confusion matrix.

Because CLIQUE is unsupervised, predicted cluster ids carry no inherent class
meaning. To compute classification-style metrics we map each predicted cluster
to the ground-truth label most frequent among its members (majority vote), which
handles an arbitrary number of clusters and noise (-1) gracefully. ROC-AUC is
computed from nearest-centroid soft scores (softmax over negative distances to
per-class centroids), turning a hard clustering into class-probability estimates.
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
    roc_auc_score,
    silhouette_score,
    v_measure_score,
)


# ---------------------------------------------------------------------------
# Intrinsic metrics (no ground truth)
# ---------------------------------------------------------------------------

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score over non-noise points; NaN if fewer than 2 clusters."""
    labels = np.asarray(labels)
    if len(set(labels) - {-1}) < 2:
        return float("nan")
    mask = labels >= 0
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return float("nan")


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index (lower is better); NaN if fewer than 2 clusters."""
    labels = np.asarray(labels)
    if len(set(labels) - {-1}) < 2:
        return float("nan")
    mask = labels >= 0
    try:
        return float(davies_bouldin_score(X[mask], labels[mask]))
    except Exception:
        return float("nan")


def compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz index (higher is better); NaN if fewer than 2 clusters."""
    labels = np.asarray(labels)
    if len(set(labels) - {-1}) < 2:
        return float("nan")
    mask = labels >= 0
    try:
        return float(calinski_harabasz_score(X[mask], labels[mask]))
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


def _class_centroid_scores(
    X: np.ndarray, y_true: np.ndarray, aligned_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Soft class scores via nearest-centroid softmax.

    Centroids are computed from the aligned predictions so the scores reflect the
    clustering. Returns (classes, scores) where scores has shape (n_samples, n_classes).
    """
    classes = np.unique(y_true)
    global_mean = X.mean(axis=0)
    centroids = []
    for c in classes:
        pts = X[aligned_pred == c]
        centroids.append(pts.mean(axis=0) if len(pts) > 0 else global_mean)
    centroids = np.vstack(centroids)

    # Negative squared distance -> softmax for stable probability-like scores.
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    logits = -dists
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    scores = exp / exp.sum(axis=1, keepdims=True)
    return classes, scores


def supervised_metrics(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Classification-style metrics after majority-vote alignment.

    Includes accuracy, macro/weighted precision-recall-F1, and one-vs-rest ROC-AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    aligned = align_predictions(y_true, y_pred)

    out: dict[str, float] = {
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

    classes = np.unique(y_true)
    if len(classes) >= 2:
        _, scores = _class_centroid_scores(X, y_true, aligned)
        y_onehot = np.zeros((len(y_true), len(classes)))
        for j, c in enumerate(classes):
            y_onehot[:, j] = (y_true == c).astype(int)
        try:
            out["roc_auc_ovr"] = float(
                roc_auc_score(y_onehot, scores, average="macro", multi_class="ovr")
            )
        except Exception:
            out["roc_auc_ovr"] = float("nan")
    else:
        out["roc_auc_ovr"] = float("nan")
    return out


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


def roc_curve_data(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[int, dict[str, Any]]:
    """
    Per-class one-vs-rest ROC curve points and AUC.

    Returns {class_index: {"fpr": ..., "tpr": ..., "auc": ...}}.
    """
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(y_true).astype(int)
    aligned = align_predictions(y_true, np.asarray(y_pred).astype(int))
    classes, scores = _class_centroid_scores(X, y_true, aligned)
    out: dict[int, dict[str, Any]] = {}
    for j, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, scores[:, j])
            out[int(c)] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
        except Exception:
            out[int(c)] = {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": float("nan")}
    return out


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
    n_clusters = len(set(y_pred) - {-1})
    row: dict[str, Any] = {
        "algorithm": name,
        "params": params,
        "n_clusters": n_clusters,
        "noise_pct": float(100.0 * np.mean(y_pred == -1)),
    }
    row.update(intrinsic_metrics(X, y_pred))
    if y_true is not None:
        row.update(supervised_metrics(X, y_true, y_pred))
        row.update(clustering_agreement_metrics(y_true, y_pred))
    return row
