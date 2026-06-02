"""
CLIQUE training: grid search over (xi, tau) and final model fit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clique.algorithm import CLIQUE
from clique.metrics import compute_silhouette, intrinsic_metrics
from pipelines.data import guess_business_label


def _feature_indices(feature_set: list[str]) -> list[int]:
    return [config.FEATURE_NAMES.index(f) for f in feature_set]


def _feature_set_key(feature_set: list[str]) -> str:
    return "|".join(feature_set)


def _quality_score(row: pd.Series) -> float:
    """Composite score for grid selection (silhouette, coverage, DBI, fragmentation)."""
    sil = row.get("silhouette", np.nan)
    dbi = row.get("davies_bouldin", np.nan)
    n_cl = int(row.get("n_clusters", 0))
    cov = float(row.get("coverage", 0))

    if np.isnan(sil) or n_cl < config.MIN_CLUSTERS or cov < config.MIN_COVERAGE:
        return -1e9

    frag = max(0, n_cl - config.MAX_CLUSTERS)
    under = max(0, config.MIN_CLUSTERS - n_cl)
    score = float(sil) - 0.025 * frag - 0.015 * under - 0.25 * (1.0 - cov)
    if not np.isnan(dbi):
        score -= 0.08 * float(dbi)
    return score


def _point_coverage(labels: np.ndarray) -> float:
    return float((labels >= 0).sum() / len(labels))


def grid_search(
    X: np.ndarray,
    *,
    xi_grid: list[int] | None = None,
    tau_grid: list[float] | None = None,
    feature_sets: list[list[str]] | None = None,
) -> pd.DataFrame:
    """Search (xi, tau) using composite quality score on intrinsic metrics."""
    xi_list = xi_grid or config.XI_GRID
    tau_list = tau_grid or config.TAU_GRID
    fs_list = feature_sets or config.CLIQUE_FEATURE_SETS

    rows: list[dict[str, object]] = []
    total = len(fs_list) * len(xi_list) * len(tau_list)
    run = 0

    for feature_set in fs_list:
        idx = _feature_indices(feature_set)
        X_sub = X[:, idx]
        fs_key = _feature_set_key(feature_set)
        print(f"\n[Feature set] {feature_set}")
        for xi in xi_list:
            for tau in tau_list:
                run += 1
                print(f"\n[{run}/{total}] xi={xi}, tau={tau} ...")
                t0 = time.perf_counter()
                model = CLIQUE(
                    xi=xi,
                    tau=tau,
                    density_decay=config.CLIQUE_DENSITY_DECAY,
                    min_cluster_size_ratio=config.MIN_CLUSTER_SIZE_RATIO,
                    redundant_overlap_threshold=config.REDUNDANT_OVERLAP_THRESHOLD,
                )
                model.fit(X_sub, feature_names=feature_set)
                elapsed = time.perf_counter() - t0

                labels = model.labels_
                assert labels is not None
                coverage = _point_coverage(labels)
                intr = intrinsic_metrics(X_sub, labels)
                row: dict[str, object] = {
                    "feature_set": fs_key,
                    "xi": xi,
                    "tau": tau,
                    "n_clusters": len(set(labels.tolist()) - {-1}),
                    "n_noise": int((labels == -1).sum()),
                    "coverage": round(coverage, 3),
                    "runtime_sec": round(elapsed, 2),
                    **{
                        k: round(v, 4) if k != "calinski_harabasz" else round(v, 1)
                        for k, v in intr.items()
                        if not np.isnan(v)
                    },
                }
                rows.append(row)
                sil = row.get("silhouette", float("nan"))
                sil_s = (
                    f"{sil:.3f}"
                    if isinstance(sil, (int, float)) and not np.isnan(sil)
                    else "nan"
                )
                print(
                    f"  -> n_clusters={row['n_clusters']}, coverage={coverage:.1%}, "
                    f"silhouette={sil_s}"
                )

    df = pd.DataFrame(rows)
    df["quality_score"] = df.apply(_quality_score, axis=1)
    config.ensure_dirs()
    df.to_csv(config.GRID_SEARCH_CSV, index=False)
    print(f"Grid search -> {config.GRID_SEARCH_CSV}")
    return df


def select_best_params(grid: pd.DataFrame) -> tuple[int, float, list[str]]:
    return select_best_params_by_objective(grid, objective=config.SELECTION_OBJECTIVE)


def _normalize(series: pd.Series, *, reverse: bool = False) -> pd.Series:
    s = series.astype(float)
    if reverse:
        s = -s
    denom = s.max() - s.min()
    if denom <= 1e-12:
        return pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    return (s - s.min()) / denom


def select_best_params_by_objective(
    grid: pd.DataFrame,
    *,
    objective: str,
) -> tuple[int, float, list[str]]:
    valid = grid[grid["quality_score"] > -1e8].copy()
    if len(valid) == 0:
        valid = grid.copy()

    within = valid[
        (valid["n_clusters"] >= config.MIN_CLUSTERS)
        & (valid["n_clusters"] <= config.MAX_CLUSTERS)
    ].copy()
    search_space = within if len(within) > 0 else valid

    if objective == "quality":
        positive = search_space[search_space["silhouette"] > 0]
        pool = positive if len(positive) > 0 else search_space
        best = pool.loc[pool["silhouette"].idxmax()]
    elif objective == "coverage":
        positive = search_space[search_space["silhouette"] > 0]
        pool = positive if len(positive) > 0 else search_space
        best = pool.loc[pool["coverage"].idxmax()]
    else:  # balanced
        scored = search_space.copy()
        scored["sil_n"] = _normalize(scored["silhouette"])
        scored["cov_n"] = _normalize(scored["coverage"])
        scored["dbi_n"] = _normalize(scored["davies_bouldin"], reverse=True)
        scored["balance_score"] = (
            0.55 * scored["sil_n"] + 0.30 * scored["cov_n"] + 0.15 * scored["dbi_n"]
        )
        best = scored.loc[scored["balance_score"].idxmax()]
    feature_set = str(best.get("feature_set", _feature_set_key(config.FEATURE_NAMES)))
    return int(best["xi"]), float(best["tau"]), feature_set.split("|")


def _compute_cluster_feature_means(
    labels: np.ndarray,
    *,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mean feature table per cluster id using non-noise points only.

    The input feature_df is expected to be aligned row-wise with training labels.
    """
    work = feature_df.copy()
    work = work.reset_index(drop=True)
    work["cluster_id"] = labels
    work = work[work["cluster_id"] >= 0]
    work = work.dropna(subset=config.FEATURE_NAMES)
    if work.empty:
        return pd.DataFrame(columns=["cluster_id", *config.FEATURE_NAMES])
    means = (
        work.groupby("cluster_id", as_index=False)[config.FEATURE_NAMES]
        .mean()
        .sort_values("cluster_id")
    )
    return means


def _save_best_params(
    grid: pd.DataFrame, xi: int, tau: float, feature_set: list[str]
) -> None:
    """Persist selected params for downstream reports/artifacts."""
    fs_key = _feature_set_key(feature_set)
    selected = grid[
        (grid["xi"] == xi) & (grid["tau"] == tau) & (grid["feature_set"] == fs_key)
    ].head(1).copy()
    if selected.empty:
        selected = pd.DataFrame([{"xi": xi, "tau": tau, "feature_set": fs_key}])
    selected.insert(0, "selection_rule", f"objective={config.SELECTION_OBJECTIVE}")
    config.ensure_dirs()
    selected.to_csv(config.BEST_PARAMS_CSV, index=False)


def _plot_pareto_frontier(
    grid: pd.DataFrame, xi: int, tau: float, feature_set: list[str]
) -> None:
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(
        grid["coverage"],
        grid["silhouette"],
        c=grid["xi"],
        cmap="viridis",
        alpha=0.8,
        s=55,
        edgecolors="white",
        linewidths=0.5,
    )
    fs_key = _feature_set_key(feature_set)
    best = grid[
        (grid["xi"] == xi) & (grid["tau"] == tau) & (grid["feature_set"] == fs_key)
    ].head(1)
    if not best.empty:
        r = best.iloc[0]
        plt.scatter(
            [r["coverage"]],
            [r["silhouette"]],
            color="#d62728",
            s=140,
            marker="*",
            label=f"selected xi={xi}, tau={tau}",
        )
    plt.xlabel("Coverage (non-noise ratio)")
    plt.ylabel("Silhouette")
    plt.title("CLIQUE Grid Pareto View: Coverage vs Silhouette")
    plt.colorbar(label="xi")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(config.PARETO_FRONTIER_PNG, dpi=150)
    plt.close()


def _write_run_summary(
    *,
    grid: pd.DataFrame,
    xi: int,
    tau: float,
    feature_set: list[str],
) -> None:
    fs_key = _feature_set_key(feature_set)
    best = grid[
        (grid["xi"] == xi) & (grid["tau"] == tau) & (grid["feature_set"] == fs_key)
    ].iloc[0]
    top = grid.sort_values("silhouette", ascending=False).head(5)
    lines = [
        "# Retail CLIQUE Run Summary",
        "",
        f"- Selection objective: `{config.SELECTION_OBJECTIVE}`",
        f"- Selected params: `xi={xi}, tau={tau}`",
        f"- Selected features: `{feature_set}`",
        f"- Train silhouette: `{best.get('silhouette', float('nan')):.4f}`",
        f"- Coverage: `{best.get('coverage', float('nan')):.1%}`",
        f"- Non-noise clusters: `{int(best.get('n_clusters', 0))}`",
        "",
        "## Top-5 configs by silhouette",
        "",
        "| xi | tau | silhouette | coverage | n_clusters | davies_bouldin |",
        "|---:|----:|-----------:|---------:|-----------:|---------------:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['xi'])} | {float(row['tau']):.2f} | {float(row['silhouette']):.4f} "
            f"| {float(row['coverage']):.3f} | {int(row['n_clusters'])} | {float(row['davies_bouldin']):.4f} |"
        )
    lines.append("")
    lines.append(f"Pareto chart: `{config.PARETO_FRONTIER_PNG}`")
    config.ensure_dirs()
    config.RUN_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def build_cluster_descriptions(
    model: CLIQUE,
    X_train: np.ndarray,
    *,
    cluster_feature_means: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cluster summary table with business labels."""
    rows: list[dict[str, object]] = []
    feat_names = model.feature_names_ or config.FEATURE_NAMES
    means_lookup: dict[int, np.ndarray] = {}
    if cluster_feature_means is not None and not cluster_feature_means.empty:
        for _, row in cluster_feature_means.iterrows():
            means_lookup[int(row["cluster_id"])] = (
                row[config.FEATURE_NAMES].to_numpy(dtype=float)
            )

    for cluster in model.clusters_:
        dims = [feat_names[d] for d in cluster["subspace"]]
        profile = means_lookup.get(cluster["id"])
        if profile is None:
            profile = model.cluster_profiles_.get(cluster["id"])
        rows.append(
            {
                "cluster_id": cluster["id"],
                "k_dim": cluster.get("k", len(cluster["subspace"])),
                "subspace": str(dims),
                "size": cluster["size"],
                "coverage_pct": round(cluster["size"] / len(X_train) * 100, 1),
                "description": cluster["description"],
                "business_label": (
                    guess_business_label(dims, profile)
                    if profile is not None and len(profile) >= len(config.FEATURE_NAMES)
                    else "General cluster"
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("size", ascending=False)
    if cluster_feature_means is not None and not cluster_feature_means.empty:
        rename_map = {f: f"{f}_mean" for f in config.FEATURE_NAMES}
        merged = out.merge(
            cluster_feature_means.rename(columns=rename_map),
            on="cluster_id",
            how="left",
        )
        return merged
    return out


def run_retail(
    X_train: np.ndarray,
    *,
    train_feature_df: pd.DataFrame | None = None,
    selection_objective: str | None = None,
    do_grid_search: bool = True,
) -> tuple[CLIQUE, pd.DataFrame, pd.DataFrame]:
    """Train CLIQUE on scaled train data."""
    if do_grid_search:
        if selection_objective is not None:
            config.SELECTION_OBJECTIVE = selection_objective
        grid = grid_search(X_train)
        print("\n--- GRID SEARCH ---")
        cols = [
            c
            for c in grid.columns
            if c
            in (
                "xi",
                "tau",
                "n_clusters",
                "coverage",
                "silhouette",
                "davies_bouldin",
                "quality_score",
                "runtime_sec",
            )
        ]
        print(grid[cols].to_string(index=False))
        xi, tau, selected_features = select_best_params_by_objective(
            grid, objective=config.SELECTION_OBJECTIVE
        )
        _save_best_params(grid, xi, tau, selected_features)
        _plot_pareto_frontier(grid, xi, tau, selected_features)
        _write_run_summary(grid=grid, xi=xi, tau=tau, feature_set=selected_features)
        fs_key = _feature_set_key(selected_features)
        best_row = grid[
            (grid["xi"] == xi)
            & (grid["tau"] == tau)
            & (grid["feature_set"] == fs_key)
        ].iloc[0]
        print(
            f"\nBest ({config.SELECTION_OBJECTIVE}): xi={xi}, tau={tau} | sil={best_row.get('silhouette', float('nan')):.4f} "
            f"| clusters={int(best_row['n_clusters'])} | coverage={best_row['coverage']:.1%}"
        )
        print(f"Selected features: {selected_features}")
    else:
        grid = pd.DataFrame()
        xi, tau = config.DEFAULT_XI, config.DEFAULT_TAU
        selected_features = config.FEATURE_NAMES
        print(f"Using defaults: xi={xi}, tau={tau}")
        _save_best_params(
            pd.DataFrame(
                [{"xi": xi, "tau": tau, "feature_set": _feature_set_key(selected_features)}]
            ),
            xi,
            tau,
            selected_features,
        )

    selected_idx = _feature_indices(selected_features)
    X_train_sel = X_train[:, selected_idx]
    model = CLIQUE(
        xi=xi,
        tau=tau,
        density_decay=config.CLIQUE_DENSITY_DECAY,
        min_cluster_size_ratio=config.MIN_CLUSTER_SIZE_RATIO,
        redundant_overlap_threshold=config.REDUNDANT_OVERLAP_THRESHOLD,
    )
    model.fit(X_train_sel, feature_names=selected_features)
    cluster_means = None
    if train_feature_df is not None and model.labels_ is not None:
        cluster_means = _compute_cluster_feature_means(
            model.labels_,
            feature_df=train_feature_df[config.FEATURE_NAMES],
        )
    cluster_desc = build_cluster_descriptions(
        model, X_train_sel, cluster_feature_means=cluster_means
    )
    print("\n--- CLUSTER DESCRIPTIONS ---")
    print(
        cluster_desc[
            ["cluster_id", "k_dim", "size", "business_label", "description"]
        ].to_string(index=False)
    )

    config.ensure_dirs()
    joblib.dump(model, config.MODEL_PKL)
    joblib.dump(cluster_desc, config.PROFILES_PKL)
    cluster_desc.to_csv(config.CLUSTER_DESCRIPTIONS_CSV, index=False)
    print(f"Saved model, profiles, {config.CLUSTER_DESCRIPTIONS_CSV.name}")
    return model, cluster_desc, grid
