"""
CLIQUE training: grid search over (xi, tau) and final model fit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clique.algorithm import CLIQUE
from clique.io import save_model
from clique.metrics import (
    clustering_agreement_metrics,
    compute_silhouette,
    intrinsic_metrics,
    supervised_metrics,
)
from pipelines.data import guess_business_label

XI_GRID = [5, 8, 10]
TAU_GRID = [0.02, 0.05, 0.08]


def _retail_quality_score(row: pd.Series) -> float:
    """
    Composite score for retail grid selection.

    Favors higher silhouette, moderate cluster count, high coverage, lower DBI.
    """
    sil = row.get("silhouette", np.nan)
    dbi = row.get("davies_bouldin", np.nan)
    n_cl = int(row.get("n_clusters", 0))
    cov = float(row.get("coverage", 0))

    if np.isnan(sil) or n_cl < config.RETAIL_MIN_CLUSTERS or cov < config.RETAIL_MIN_COVERAGE:
        return -1e9

    frag = max(0, n_cl - config.RETAIL_MAX_CLUSTERS)
    score = float(sil) - 0.04 * frag - 0.5 * (1.0 - cov)
    if not np.isnan(dbi):
        score -= 0.12 * float(dbi)
    return score


def _point_coverage(labels: np.ndarray) -> float:
    return float((labels >= 0).sum() / len(labels))


def grid_search(
    X: np.ndarray,
    y_true: np.ndarray | None = None,
    *,
    use_silhouette_selection: bool = False,
    xi_grid: list[int] | None = None,
    tau_grid: list[float] | None = None,
) -> pd.DataFrame:
    """
    Search (xi, tau). Synthetic: ARI then silhouette. Retail: composite quality score.
    """
    if use_silhouette_selection:
        xi_list = xi_grid or config.RETAIL_XI_GRID
        tau_list = tau_grid or config.RETAIL_TAU_GRID
    else:
        xi_list = xi_grid or XI_GRID
        tau_list = tau_grid or TAU_GRID

    rows: list[dict[str, object]] = []
    total = len(xi_list) * len(tau_list)
    run = 0

    for xi in xi_list:
        for tau in tau_list:
            run += 1
            print(f"\n[{run}/{total}] xi={xi}, tau={tau} ...")
            t0 = time.perf_counter()
            model = CLIQUE(xi=xi, tau=tau)
            model.fit(X, feature_names=config.FEATURE_NAMES)
            elapsed = time.perf_counter() - t0

            labels = model.labels_
            assert labels is not None
            coverage = _point_coverage(labels)
            intr = intrinsic_metrics(X, labels)
            row: dict[str, object] = {
                "xi": xi,
                "tau": tau,
                "n_clusters": len(set(labels.tolist()) - {-1}),
                "n_noise": int((labels == -1).sum()),
                "coverage": round(coverage, 3),
                "runtime_sec": round(elapsed, 2),
                **{k: round(v, 4) if k != "calinski_harabasz" else round(v, 1)
                   for k, v in intr.items() if not np.isnan(v)},
            }
            if y_true is not None:
                row.update(supervised_metrics(y_true, labels))
                row.update(clustering_agreement_metrics(y_true, labels))
            rows.append(row)
            sil = row.get("silhouette", float("nan"))
            sil_s = f"{sil:.3f}" if isinstance(sil, (int, float)) and not np.isnan(sil) else "nan"
            print(f"  → n_clusters={row['n_clusters']}, coverage={coverage:.1%}, silhouette={sil_s}")

    df = pd.DataFrame(rows)
    if use_silhouette_selection:
        df["quality_score"] = df.apply(_retail_quality_score, axis=1)
    config.ensure_dirs()
    out_csv = config.GRID_SEARCH_CSV if y_true is not None else config.RETAIL_GRID_SEARCH_CSV
    df.to_csv(out_csv, index=False)
    print(f"Grid search -> {out_csv}")
    return df


def select_best_params(
    grid: pd.DataFrame,
    *,
    use_silhouette_selection: bool = False,
) -> tuple[int, float]:
    if use_silhouette_selection:
        if "quality_score" in grid.columns:
            valid = grid[grid["quality_score"] > -1e8]
            if len(valid) == 0:
                valid = grid
            positive = valid[valid["silhouette"] > 0]
            if len(positive) > 0:
                within = positive[
                    (positive["n_clusters"] >= config.RETAIL_MIN_CLUSTERS)
                    & (positive["n_clusters"] <= config.RETAIL_MAX_CLUSTERS)
                    & (positive["coverage"] >= config.RETAIL_MIN_COVERAGE)
                ]
                pick = within if len(within) > 0 else positive
                best = pick.loc[pick["silhouette"].idxmax()]
            else:
                best = valid.loc[valid["quality_score"].idxmax()]
        else:
            valid = grid[
                (grid["n_clusters"] >= config.RETAIL_MIN_CLUSTERS)
                & (grid["n_clusters"] <= config.RETAIL_MAX_CLUSTERS)
                & (grid["coverage"] >= config.RETAIL_MIN_COVERAGE)
            ]
            if len(valid) == 0:
                valid = grid
            best = valid.loc[valid["silhouette"].idxmax()]
        return int(best["xi"]), float(best["tau"])

    ranked = grid.sort_values(["adjusted_rand", "silhouette"], ascending=False)
    best = ranked.iloc[0]
    return int(best["xi"]), float(best["tau"])


def build_cluster_descriptions(model: CLIQUE, X_train: np.ndarray) -> pd.DataFrame:
    """Cluster summary table with business labels (retail / interpretability)."""
    rows: list[dict[str, object]] = []
    for cluster in model.clusters_:
        dims = [config.FEATURE_NAMES[d] for d in cluster["subspace"]]
        profile = model.cluster_profiles_.get(
            cluster["id"], np.zeros(len(config.FEATURE_NAMES))
        )
        rows.append(
            {
                "cluster_id": cluster["id"],
                "k_dim": cluster.get("k", len(cluster["subspace"])),
                "subspace": str(dims),
                "size": cluster["size"],
                "coverage_pct": round(cluster["size"] / len(X_train) * 100, 1),
                "description": cluster["description"],
                "business_label": guess_business_label(dims, profile),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def run_synthetic(*, do_grid_search: bool = True) -> CLIQUE:
    """Train on synthetic scaled train split (ARI-based selection)."""
    config.ensure_dirs()
    X = pd.read_csv(config.X_TRAIN_SCALED_CSV)[config.FEATURE_NAMES].values.astype(float)
    y = pd.read_csv(config.TRAIN_LABELS_CSV)["true_segment"].values.astype(int)

    if do_grid_search:
        grid = grid_search(X, y, use_silhouette_selection=False)
        print(grid.to_string(index=False))
        xi, tau = select_best_params(grid, use_silhouette_selection=False)
    else:
        xi, tau = config.DEFAULT_XI, config.DEFAULT_TAU

    print(f"Selected: xi={xi}, tau={tau}")
    model = CLIQUE(xi=xi, tau=tau)
    model.fit(X, feature_names=config.FEATURE_NAMES)
    save_model(
        model,
        joblib.load(config.SYNTHETIC_MODELS / "scaler.pkl"),
        model.cluster_profiles_,
        model_dir=config.SYNTHETIC_MODELS,
    )
    print(f"Model -> {config.MODEL_PKL} ({len(model.clusters_)} clusters)")
    return model


def run_retail(
    X_train: np.ndarray,
    *,
    do_grid_search: bool = True,
) -> tuple[CLIQUE, pd.DataFrame, pd.DataFrame]:
    """Train on retail scaled data (composite quality selection)."""
    if do_grid_search:
        grid = grid_search(X_train, y_true=None, use_silhouette_selection=True)
        print("\n--- GRID SEARCH ---")
        cols = [c for c in grid.columns if c in (
            "xi", "tau", "n_clusters", "coverage", "silhouette",
            "davies_bouldin", "quality_score", "runtime_sec",
        )]
        print(grid[cols].to_string(index=False))
        xi, tau = select_best_params(grid, use_silhouette_selection=True)
        best_row = grid[(grid["xi"] == xi) & (grid["tau"] == tau)].iloc[0]
        print(
            f"\nBest: xi={xi}, tau={tau} | sil={best_row.get('silhouette', float('nan')):.4f} "
            f"| clusters={int(best_row['n_clusters'])} | coverage={best_row['coverage']:.1%}"
        )
    else:
        grid = pd.DataFrame()
        xi, tau = config.DEFAULT_XI, config.DEFAULT_TAU
        print(f"Using defaults: xi={xi}, tau={tau}")

    model = CLIQUE(xi=xi, tau=tau)
    model.fit(X_train, feature_names=config.FEATURE_NAMES)
    cluster_desc = build_cluster_descriptions(model, X_train)
    print("\n--- CLUSTER DESCRIPTIONS ---")
    print(
        cluster_desc[["cluster_id", "k_dim", "size", "business_label", "description"]].to_string(
            index=False
        )
    )

    import joblib

    config.RETAIL_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.RETAIL_MODEL_PKL)
    joblib.dump(cluster_desc, config.RETAIL_PROFILES_PKL)
    cluster_desc.to_csv(config.CLUSTER_DESCRIPTIONS_CSV, index=False)
    print(f"Saved model, profiles, {config.CLUSTER_DESCRIPTIONS_CSV.name}")
    return model, cluster_desc, grid
