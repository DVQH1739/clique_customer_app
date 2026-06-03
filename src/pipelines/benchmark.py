"""
Benchmarking: baseline algorithms, test-set evaluation, figures, and reports.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clique.algorithm import CLIQUE
from clique.metrics import evaluate_labeling, intrinsic_metrics

sns.set_theme(style="whitegrid")

# Shared baseline grid (guide-aligned)
BASELINE_ESTIMATORS: list[tuple[str, str, object]] = [
    ("KMeans", "k=4", KMeans(n_clusters=4, random_state=config.RANDOM_STATE, n_init=10)),
    ("KMeans", "k=5", KMeans(n_clusters=5, random_state=config.RANDOM_STATE, n_init=10)),
    ("KMeans", "k=6", KMeans(n_clusters=6, random_state=config.RANDOM_STATE, n_init=10)),
    ("KMeans", "k=7", KMeans(n_clusters=7, random_state=config.RANDOM_STATE, n_init=10)),
    ("DBSCAN", "eps=0.2", DBSCAN(eps=0.2, min_samples=5)),
    ("DBSCAN", "eps=0.3", DBSCAN(eps=0.3, min_samples=5)),
    ("DBSCAN", "eps=0.5", DBSCAN(eps=0.5, min_samples=5)),
    ("Agglomerative", "k=5", AgglomerativeClustering(n_clusters=5)),
    ("Agglomerative", "k=6", AgglomerativeClustering(n_clusters=6)),
]


def _clear_figures() -> None:
    """Remove stale benchmark PNGs; keep EDA from preprocess."""
    if config.FIGURES_DIR.exists():
        for png in config.FIGURES_DIR.glob("*.png"):
            if png.name == config.EDA_DISTRIBUTIONS_PNG.name:
                continue
            png.unlink()


def compare_algorithms(
    X: np.ndarray,
    y_true: np.ndarray | None,
    model: CLIQUE | None = None,
    *,
    include_full_baselines: bool = True,
) -> pd.DataFrame:
    """Run CLIQUE (predict) and sklearn baselines; return metrics table."""
    rows: list[dict[str, object]] = []

    if model is not None:
        t0 = time.perf_counter()
        pred = model.predict(X)
        rt = time.perf_counter() - t0
        row = evaluate_labeling(X, pred, y_true, "CLIQUE", f"xi={model.xi}, tau={model.tau}")
        row["runtime_sec"] = rt
        rows.append(row)

    estimators = BASELINE_ESTIMATORS if include_full_baselines else BASELINE_ESTIMATORS[:6]
    for name, params, est in estimators:
        t0 = time.perf_counter()
        pred = est.fit_predict(X)
        rt = time.perf_counter() - t0
        row = evaluate_labeling(X, pred, y_true, name, params)
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


def run_baseline_comparison(X: np.ndarray) -> pd.DataFrame:
    """Intrinsic-only quick comparison for the Streamlit app."""
    rows: list[dict[str, object]] = []
    for name, params, est in BASELINE_ESTIMATORS[:6]:
        t0 = time.perf_counter()
        labels = est.fit_predict(X)
        rt = time.perf_counter() - t0
        intr = intrinsic_metrics(X, labels)
        rows.append(
            {
                "algorithm": name,
                "params": params,
                "n_clusters": len(set(labels.tolist()) - {-1}),
                "silhouette": intr["silhouette"],
                "davies_bouldin": intr["davies_bouldin"],
                "calinski_harabasz": intr["calinski_harabasz"],
                "runtime_sec": rt,
            }
        )
    return pd.DataFrame(rows)


def evaluate_test_split(
    model: CLIQUE,
    cluster_desc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict on held-out test CSV; intrinsic metrics + optional business labels."""
    test_df = pd.read_csv(config.X_TEST_SCALED_CSV)
    feat_cols = [c for c in config.FEATURE_NAMES if c in test_df.columns]
    X_test = test_df[feat_cols].values.astype(float)
    ids = test_df["CustomerID"].values if "CustomerID" in test_df.columns else np.arange(len(test_df))

    labels = model.predict(X_test)
    n_clustered = int((labels >= 0).sum())
    n_noise = int((labels == -1).sum())
    coverage = n_clustered / len(labels)

    print(f"\n--- TEST SET ({len(X_test):,} customers) ---")
    print(f"  Coverage:     {coverage:.1%}")
    print(f"  Noise:        {n_noise} ({100 * n_noise / len(labels):.1f}%)")

    if n_clustered >= 2 and len(set(labels[labels >= 0])) >= 2:
        Xv, lv = X_test[labels >= 0], labels[labels >= 0]
        sil = silhouette_score(Xv, lv, sample_size=min(500, len(Xv)), random_state=42)
        dbi = davies_bouldin_score(Xv, lv)
        print(f"  Silhouette:   {sil:.4f}")
        print(f"  Davies-Bouldin: {dbi:.4f}")

    result = pd.DataFrame({"CustomerID": ids, "cluster_id": labels, "is_noise": labels == -1})
    if cluster_desc is not None and "cluster_id" in cluster_desc.columns:
        result = result.merge(
            cluster_desc[["cluster_id", "business_label", "description"]],
            on="cluster_id",
            how="left",
        )
        result.loc[result["is_noise"], "business_label"] = "Noise"
        result.loc[result["is_noise"], "description"] = "No cluster"

    config.ensure_dirs()
    result.to_csv(config.TEST_PREDICTIONS_CSV, index=False)
    print(f"  Saved -> {config.TEST_PREDICTIONS_CSV.name}")
    print("\n--- CLUSTER DISTRIBUTION (test) ---")
    dist = (
        result.groupby("cluster_id")
        .agg(count=("CustomerID", "count"), label=("business_label", "first"))
        .sort_values("count", ascending=False)
    )
    print(dist.to_string())
    return result


def _export_subspace_coverage(model: CLIQUE) -> None:
    names = model.feature_names_ or config.FEATURE_NAMES
    rows = []
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
    print(f"Subspace coverage -> {out.name}")


def _plot_baseline_intrinsic(df: pd.DataFrame) -> None:
    """Best silhouette per algorithm — retail-style bar chart."""
    if "silhouette" not in df.columns:
        return
    best = df.loc[df.groupby("algorithm", sort=False)["silhouette"].idxmax()].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("So sánh thuật toán Clustering", fontsize=14)
    for ax, (metric, ylabel) in zip(
        axes,
        [
            ("silhouette", "Silhouette\n(cao hơn = tốt)"),
            ("davies_bouldin", "Davies-Bouldin\n(thấp hơn = tốt)"),
            ("calinski_harabasz", "Calinski-Harabasz\n(cao hơn = tốt)"),
        ],
    ):
        if metric not in best.columns:
            continue
        colors = ["#2196F3" if a == "CLIQUE" else "#90CAF9" for a in best["algorithm"]]
        bars = ax.bar(best["algorithm"], best[metric], color=colors, edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        for bar, alg in zip(bars, best["algorithm"]):
            if alg == "CLIQUE":
                bar.set_edgecolor("#1565C0")
                bar.set_linewidth(2)
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(config.BASELINE_COMPARISON_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {config.BASELINE_COMPARISON_PNG.name}")


def _plot_subspace_heatmap(model: CLIQUE) -> None:
    heat = model.get_subspace_heatmap_data()
    if heat.empty:
        return
    pivot = heat.pivot_table(index="dim_1", columns="dim_2", values="n_clusters", fill_value=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="g", cmap="Oranges")
    plt.title("2D Subspaces with Dense Clusters")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "subspace_heatmap.png", dpi=150)
    plt.close()


def _plot_cluster_sizes(model: CLIQUE) -> None:
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


def _plot_top_subspace_grids(X: np.ndarray, model: CLIQUE) -> None:
    names = model.feature_names_ or config.FEATURE_NAMES
    twod = [(s, c) for s, c in model.subspace_coverage_.items() if len(s) == 2]
    twod.sort(key=lambda x: -x[1])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for subspace, _ in twod[:2]:
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
                    (ix / model.xi, iy / model.xi),
                    1 / model.xi,
                    1 / model.xi,
                    facecolor="orange",
                    alpha=0.35,
                    edgecolor="none",
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
                        (ix / model.xi, iy / model.xi),
                        1 / model.xi,
                        1 / model.xi,
                        fill=False,
                        edgecolor=color,
                        lw=2.5,
                    )
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(names[dx])
        ax.set_ylabel(names[dy])
        ax.set_title(f"Subspace: {names[dx]} × {names[dy]}")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"grid_{names[dx]}_{names[dy]}.png", dpi=150)
        plt.close()


def run_retail(model: CLIQUE, cluster_desc: pd.DataFrame) -> pd.DataFrame:
    """Intrinsic baselines on train + test evaluation."""
    config.ensure_dirs()
    _clear_figures()
    X_train = pd.read_csv(config.X_TRAIN_SCALED_CSV)[config.FEATURE_NAMES].values.astype(float)

    print("\n--- BASELINES (train) ---")
    comparison = compare_algorithms(X_train, y_true=None, model=model)
    comparison.to_csv(config.BASELINE_COMPARISON_CSV, index=False)
    print(comparison.to_string(index=False))
    _export_subspace_coverage(model)
    _plot_baseline_intrinsic(comparison)
    _plot_subspace_heatmap(model)
    _plot_cluster_sizes(model)
    _plot_top_subspace_grids(X_train, model)

    evaluate_test_split(model, cluster_desc)
    return comparison
