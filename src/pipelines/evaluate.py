"""
Pipeline step 4 — evaluation and reporting.

Evaluates the trained CLIQUE model against KMeans / DBSCAN / Agglomerative
baselines on the full scaled dataset using the synthetic ground-truth segments.
Writes metric tables (CSV) to ``results/metrics/`` and diagrams (PNG) to
``results/figures/``.

Metrics:
  - Intrinsic: silhouette, Davies-Bouldin, Calinski-Harabasz
  - Supervised: accuracy, precision/recall/F1 (macro & weighted)
  - Agreement: ARI, NMI, homogeneity, completeness, V-measure

Run:
    python pipelines/evaluate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless backend for PNG export
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clique.algorithm import CLIQUE
from clique.metrics import (
    align_predictions,
    confusion_matrix_aligned,
    evaluate_labeling,
)

sns.set_theme(style="whitegrid")


def _load() -> tuple[np.ndarray, np.ndarray, CLIQUE]:
    Xtr = pd.read_csv(config.X_TRAIN_SCALED_CSV)[config.FEATURE_NAMES].values.astype(float)
    Xte = pd.read_csv(config.X_TEST_SCALED_CSV)[config.FEATURE_NAMES].values.astype(float)
    ytr = pd.read_csv(config.TRAIN_LABELS_CSV)["true_segment"].values.astype(int)
    yte = pd.read_csv(config.TEST_LABELS_CSV)["true_segment"].values.astype(int)
    X_all = np.vstack([Xtr, Xte])
    y_all = np.concatenate([ytr, yte])
    model = joblib.load(config.MODEL_PKL)
    return X_all, y_all, model


def build_comparison(X: np.ndarray, y: np.ndarray, model: CLIQUE) -> pd.DataFrame:
    """Evaluate CLIQUE and baselines on the same data; return a metrics table."""
    rows: list[dict[str, object]] = []

    t0 = time.perf_counter()
    clique_pred = model.predict(X)
    clique_rt = time.perf_counter() - t0
    row = evaluate_labeling(X, clique_pred, y, "CLIQUE", f"xi={model.xi}, tau={model.tau}")
    row["runtime_sec"] = clique_rt
    rows.append(row)

    baselines = [
        ("KMeans", {"n_clusters": 4}, KMeans(n_clusters=4, random_state=config.RANDOM_STATE, n_init=10)),
        ("KMeans", {"n_clusters": 5}, KMeans(n_clusters=5, random_state=config.RANDOM_STATE, n_init=10)),
        ("KMeans", {"n_clusters": 6}, KMeans(n_clusters=6, random_state=config.RANDOM_STATE, n_init=10)),
        ("DBSCAN", {"eps": 0.3, "min_samples": 5}, DBSCAN(eps=0.3, min_samples=5)),
        ("DBSCAN", {"eps": 0.5, "min_samples": 5}, DBSCAN(eps=0.5, min_samples=5)),
        ("AgglomerativeClustering", {"n_clusters": 5}, AgglomerativeClustering(n_clusters=5)),
    ]
    for name, params, est in baselines:
        t0 = time.perf_counter()
        pred = est.fit_predict(X)
        rt = time.perf_counter() - t0
        row = evaluate_labeling(X, pred, y, name, str(params))
        row["runtime_sec"] = rt
        rows.append(row)

    cols = [
        "algorithm", "params", "n_clusters", "noise_pct",
        "accuracy", "f1_macro", "f1_weighted",
        "adjusted_rand", "nmi", "homogeneity", "completeness", "v_measure",
        "silhouette", "davies_bouldin", "calinski_harabasz", "runtime_sec",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def export_classification_report(X: np.ndarray, y: np.ndarray, model: CLIQUE) -> None:
    """Per-class precision/recall/F1 for CLIQUE (aligned predictions)."""
    pred = model.predict(X)
    aligned = align_predictions(y, pred)
    target_names = [config.SEGMENT_NAMES[c] for c in sorted(np.unique(y))]
    report = classification_report(
        y, aligned, target_names=target_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(config.CLIQUE_CLASSIFICATION_CSV)
    print(f"Classification report -> {config.CLIQUE_CLASSIFICATION_CSV}")


def export_subspace_coverage(model: CLIQUE) -> None:
    """Export discovered subspaces and their coverage."""
    rows = []
    names = model._feature_names or config.FEATURE_NAMES
    for subspace, cov in model.subspace_coverage_.items():
        n_cl = sum(1 for c in model.clusters_ if c["subspace"] == subspace)
        rows.append(
            {
                "subspace": " & ".join(names[d] for d in subspace),
                "dimensions": len(subspace),
                "n_clusters": n_cl,
                "coverage": cov,
            }
        )
    out = config.METRICS_DIR / "clique_subspace_coverage.csv"
    pd.DataFrame(rows).sort_values("coverage", ascending=False).to_csv(out, index=False)
    print(f"Subspace coverage -> {out}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_confusion(X: np.ndarray, y: np.ndarray, model: CLIQUE) -> None:
    cm, classes = confusion_matrix_aligned(y, model.predict(X))
    labels = [config.SEGMENT_NAMES[c] for c in classes]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted (aligned)")
    plt.ylabel("True segment")
    plt.title("CLIQUE Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "confusion_matrix_clique.png", dpi=150)
    plt.close()


def plot_metric_comparison(comparison: pd.DataFrame) -> None:
    """Single grouped bar chart of the key comparable metrics (all in [0, 1])."""
    metrics = [m for m in ("adjusted_rand", "nmi", "f1_macro") if m in comparison.columns]
    labels = [f"{a}\n{p}" for a, p in zip(comparison["algorithm"], comparison["params"])]
    x = np.arange(len(comparison))
    width = 0.8 / len(metrics)
    plt.figure(figsize=(11, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, comparison[metric].fillna(0), width, label=metric)
    plt.xticks(x + width * (len(metrics) - 1) / 2, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("score (higher is better)")
    plt.ylim(0, 1.05)
    plt.title("Clustering quality by algorithm (ARI / NMI / F1-macro)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "metric_comparison.png", dpi=150)
    plt.close()


def plot_subspace_heatmap(model: CLIQUE) -> None:
    df = model.get_subspace_heatmap_data()
    if df.empty:
        return
    pivot = df.pivot_table(index="dim_1", columns="dim_2", values="n_clusters", fill_value=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="g", cmap="Oranges")
    plt.title("2D Subspaces with Dense Clusters")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "subspace_heatmap.png", dpi=150)
    plt.close()


def plot_cluster_sizes(model: CLIQUE) -> None:
    if not model.clusters_:
        return
    ids = [c["id"] for c in model.clusters_]
    sizes = [c["size"] for c in model.clusters_]
    plt.figure(figsize=(8, 5))
    plt.bar([str(i) for i in ids], sizes, color="#9467bd")
    plt.xlabel("Cluster id")
    plt.ylabel("Points")
    plt.title("CLIQUE Cluster Sizes")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "cluster_sizes.png", dpi=150)
    plt.close()


def plot_top_subspace_grids(X: np.ndarray, model: CLIQUE) -> None:
    """Matplotlib grid plots (scatter + dense cells + cluster outlines) for top 2D subspaces."""
    names = model._feature_names or config.FEATURE_NAMES
    twod = [(s, c) for s, c in model.subspace_coverage_.items() if len(s) == 2]
    twod.sort(key=lambda x: -x[1])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for subspace, _cov in twod[:2]:
        dx, dy = subspace
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(X[:, dx], X[:, dy], s=8, c="grey", alpha=0.4)
        for i in range(model.xi + 1):
            v = i / model.xi
            ax.axvline(v, color="lightgray", lw=0.5, ls=":")
            ax.axhline(v, color="lightgray", lw=0.5, ls=":")
        for unit in model.dense_units_.get(subspace, []):
            ix, iy = unit["intervals"]
            ax.add_patch(
                plt.Rectangle(
                    (ix / model.xi, iy / model.xi), 1 / model.xi, 1 / model.xi,
                    facecolor="orange", alpha=0.35, edgecolor="none",
                )
            )
        for cluster in model.clusters_:
            if cluster["subspace"] != subspace:
                continue
            color = colors[cluster["id"] % len(colors)]
            for unit in cluster["units"]:
                ix, iy = unit["intervals"]
                ax.add_patch(
                    plt.Rectangle(
                        (ix / model.xi, iy / model.xi), 1 / model.xi, 1 / model.xi,
                        fill=False, edgecolor=color, lw=2.5,
                    )
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(names[dx])
        ax.set_ylabel(names[dy])
        ax.set_title(f"Subspace: {names[dx]} x {names[dy]}")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"grid_{names[dx]}_{names[dy]}.png", dpi=150)
        plt.close()


def _clear_figures() -> None:
    """Delete previously generated PNGs so no stale figures linger."""
    if config.FIGURES_DIR.exists():
        for png in config.FIGURES_DIR.glob("*.png"):
            png.unlink()


def run() -> pd.DataFrame:
    """Run full evaluation and export all artifacts."""
    config.ensure_dirs()
    _clear_figures()
    X, y, model = _load()

    comparison = build_comparison(X, y, model)
    comparison.to_csv(config.COMPARISON_METRICS_CSV, index=False)
    print(f"Comparison -> {config.COMPARISON_METRICS_CSV}")
    print(comparison.to_string(index=False))

    export_classification_report(X, y, model)
    export_subspace_coverage(model)

    plot_confusion(X, y, model)
    plot_metric_comparison(comparison)
    plot_subspace_heatmap(model)
    plot_cluster_sizes(model)
    plot_top_subspace_grids(X, model)
    print(f"Figures -> {config.FIGURES_DIR}")

    return comparison


if __name__ == "__main__":
    run()
