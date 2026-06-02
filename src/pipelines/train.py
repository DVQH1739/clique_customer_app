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


def _point_coverage(labels: np.ndarray) -> float:
    return float((labels >= 0).sum() / len(labels))


def grid_search(
    X: np.ndarray,
    y_true: np.ndarray | None = None,
    *,
    use_silhouette_selection: bool = False,
) -> pd.DataFrame:
    """
    Search (xi, tau). Synthetic mode: rank by ARI then silhouette.
    Retail mode: silhouette with n_clusters/coverage constraints.
    """
    rows: list[dict[str, object]] = []
    total = len(XI_GRID) * len(TAU_GRID)
    run = 0

    for xi in XI_GRID:
        for tau in TAU_GRID:
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
        valid = grid[
            (grid["n_clusters"] >= 3)
            & (grid["n_clusters"] <= 20)
            & (grid["coverage"] >= 0.7)
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


def run_retail(X_train: np.ndarray) -> tuple[CLIQUE, pd.DataFrame]:
    """Train on retail scaled train split (silhouette-based selection)."""
    grid = grid_search(X_train, y_true=None, use_silhouette_selection=True)
    print("\n--- GRID SEARCH ---")
    print(grid.to_string(index=False))
    xi, tau = select_best_params(grid, use_silhouette_selection=True)
    print(f"\nBest: xi={xi}, tau={tau}")

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
    return model, cluster_desc
