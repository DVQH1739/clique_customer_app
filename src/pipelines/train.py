"""
Pipeline step 3 — training.

Optionally runs a grid search over (xi, tau), then fits the final CLIQUE model on
the scaled TRAIN split and persists the model, scaler, and cluster profiles to
``models/``. The grid-search table is written to ``results/metrics/``.

Run:
    python pipelines/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clique.algorithm import CLIQUE
from clique.metrics import (
    clustering_agreement_metrics,
    compute_silhouette,
    supervised_metrics,
)

XI_GRID = [5, 8, 10]
TAU_GRID = [0.02, 0.05, 0.08]


def _load_train() -> tuple[np.ndarray, np.ndarray]:
    if not config.X_TRAIN_SCALED_CSV.exists():
        raise FileNotFoundError(
            f"{config.X_TRAIN_SCALED_CSV} missing. Run pipelines/preprocess.py first."
        )
    X = pd.read_csv(config.X_TRAIN_SCALED_CSV)[config.FEATURE_NAMES].values.astype(float)
    y = pd.read_csv(config.TRAIN_LABELS_CSV)["true_segment"].values.astype(int)
    return X, y


def grid_search(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Search (xi, tau); score by silhouette and F1 against ground truth."""
    rows: list[dict[str, object]] = []
    for xi in XI_GRID:
        for tau in TAU_GRID:
            model = CLIQUE(xi=xi, tau=tau)
            model.fit(X, feature_names=config.FEATURE_NAMES)
            coverage = (
                float(np.mean(list(model.subspace_coverage_.values())))
                if model.subspace_coverage_
                else 0.0
            )
            sil = compute_silhouette(X, model.labels_)
            sup = supervised_metrics(y, model.labels_)
            agree = clustering_agreement_metrics(y, model.labels_)
            rows.append(
                {
                    "xi": xi,
                    "tau": tau,
                    "n_clusters": len(model.clusters_),
                    "n_labels_used": int(len(set(model.labels_.tolist()) - {-1})),
                    "coverage": coverage,
                    "silhouette": sil,
                    "adjusted_rand": agree["adjusted_rand"],
                    "nmi": agree["nmi"],
                    "f1_macro": sup["f1_macro"],
                }
            )
    df = pd.DataFrame(rows)
    config.ensure_dirs()
    df.to_csv(config.GRID_SEARCH_CSV, index=False)
    print(f"Grid search -> {config.GRID_SEARCH_CSV}")
    return df


def select_best(grid: pd.DataFrame) -> tuple[int, float]:
    """
    Pick (xi, tau) maximizing Adjusted Rand Index, then silhouette.

    ARI is used (rather than F1) because the majority-vote F1 trivially rewards
    over-fragmentation: with hundreds of tiny clusters each maps cleanly to a
    segment, inflating F1 to ~1.0. ARI penalizes both over- and under-clustering.
    """
    ranked = grid.sort_values(
        ["adjusted_rand", "silhouette"], ascending=False
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    return int(best["xi"]), float(best["tau"])


def run(do_grid_search: bool = True) -> CLIQUE:
    """Train final CLIQUE and persist artifacts."""
    config.ensure_dirs()
    X, y = _load_train()

    if do_grid_search:
        grid = grid_search(X, y)
        print(grid.to_string(index=False))
        best_xi, best_tau = select_best(grid)
    else:
        best_xi, best_tau = config.DEFAULT_XI, config.DEFAULT_TAU
    print(f"Selected params: xi={best_xi}, tau={best_tau}")

    model = CLIQUE(xi=best_xi, tau=best_tau)
    model.fit(X, feature_names=config.FEATURE_NAMES)

    joblib.dump(model, config.MODEL_PKL)
    joblib.dump(model.cluster_profiles_, config.PROFILES_PKL)
    print(f"Saved model -> {config.MODEL_PKL} ({len(model.clusters_)} clusters)")
    return model


if __name__ == "__main__":
    run()
