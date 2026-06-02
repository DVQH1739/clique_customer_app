"""
Online Retail II pipeline: load xlsx -> profiles -> preprocess -> train -> benchmark.

Primary data source: ``data/raw/online_retail_ii.xlsx`` (not CSV).
Processed CSVs under ``data/processed/`` are optional caches only.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from clique.algorithm import CLIQUE
from pipelines import benchmark, data, preprocess, train, validate


def _project_to_model_features(X: np.ndarray, model: CLIQUE) -> np.ndarray:
    names = model.feature_names_ or config.FEATURE_NAMES
    idx = [config.FEATURE_NAMES.index(f) for f in names]
    return X[:, idx]


def run_from_xlsx(
    xlsx_path: str | Path | None = None,
    *,
    save_artifacts: bool = True,
    do_grid_search: bool = True,
    selection_objective: str | None = None,
) -> dict[str, object]:
    """
    End-to-end retail pipeline from Excel workbook.

    Returns dict with model, cluster_desc, prep bundle, and quality metrics.
    """
    path = data.resolve_retail_xlsx(
        str(xlsx_path) if xlsx_path is not None else None
    )
    print("=" * 70)
    print(f"[Retail] Source: {path}")
    print("=" * 70)

    profiles = data.build_profiles_from_xlsx(path, save_artifacts=save_artifacts)

    print("\n" + "=" * 70)
    print("[Retail] Preprocess")
    print("=" * 70)
    prep = preprocess.run_retail(profiles, winsorize=True)

    print("\n" + "=" * 70)
    print("[Retail] Train CLIQUE")
    print("=" * 70)
    train_feature_df = (
        profiles.assign(CustomerID=profiles["CustomerID"].astype(str))
        .set_index("CustomerID")
        .reindex(prep["ids_train"])
        .reset_index(drop=True)
    )
    model, cluster_desc, grid = train.run_retail(
        prep["X_train"],
        train_feature_df=train_feature_df,
        selection_objective=selection_objective,
        do_grid_search=do_grid_search,
    )

    print("\n" + "=" * 70)
    print("[Retail] Benchmark")
    print("=" * 70)
    comparison = benchmark.run_retail(model, cluster_desc)

    X_test_model = _project_to_model_features(prep["X_test"], model)
    X_train_model = _project_to_model_features(prep["X_train"], model)

    test_labels = model.predict(X_test_model)
    test_sil = float("nan")
    mask = test_labels >= 0
    if mask.sum() >= 2 and len(set(test_labels[mask])) >= 2:
        from sklearn.metrics import silhouette_score

        test_sil = float(
            silhouette_score(
                X_test_model[mask],
                test_labels[mask],
                sample_size=min(500, int(mask.sum())),
                random_state=config.RANDOM_STATE,
            )
        )

    train_sil = float("nan")
    if model.labels_ is not None:
        from clique.metrics import compute_silhouette

        train_sil = compute_silhouette(X_train_model, model.labels_)

    print("\n--- CLUSTER QUALITY SUMMARY ---")
    print(f"  Params:     xi={model.xi}, tau={model.tau}")
    print(f"  Clusters:   {len(model.clusters_)} (assigned labels: {len(set(model.labels_) - {-1}) if model.labels_ is not None else 0})")
    print(f"  Train sil:  {train_sil:.4f}")
    print(f"  Test sil:   {test_sil:.4f}")
    print(f"  Coverage:   {(model.labels_ >= 0).mean():.1%}" if model.labels_ is not None else "")

    print("\n" + "=" * 70)
    print("[Retail] Validation")
    print("=" * 70)
    validate.validate_retail_pipeline(profiles, model, prep["X_train"])

    return {
        "model": model,
        "cluster_desc": cluster_desc,
        "prep": prep,
        "grid": grid,
        "comparison": comparison,
        "train_silhouette": train_sil,
        "test_silhouette": test_sil,
    }
