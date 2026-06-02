"""
Single entry point for CLIQUE pipelines.

Primary workflow (Online Retail II Excel):

    python src/pipelines/run.py retail
    python src/pipelines/run.py retail --xlsx data/raw/online_retail_ii.xlsx

Optional synthetic demo (CSV, for supervised metrics only):

    python src/pipelines/run.py synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clique.algorithm import CLIQUE
from clique.metrics import compute_silhouette, evaluate_labeling
from pipelines import benchmark, data, preprocess, retail, train


def verify_clique() -> None:
    """Smoke test: fit / predict / metrics on random 8D data."""
    import numpy as np

    rng = np.random.default_rng(config.RANDOM_STATE)
    X = rng.uniform(0, 1, (200, 8))
    y = rng.integers(0, 3, size=200)
    X[:60, 0] = rng.uniform(0.6, 0.8, 60)
    X[:60, 1] = rng.uniform(0.7, 0.9, 60)

    model = CLIQUE(xi=config.DEFAULT_XI, tau=config.DEFAULT_TAU)
    model.fit(X, feature_names=config.FEATURE_NAMES)
    assert model.labels_ is not None and len(model.clusters_) >= 1
    assert model.predict(X[:5]).shape == (5,)
    row = evaluate_labeling(X, model.predict(X), y, "CLIQUE")
    sil = compute_silhouette(X, model.labels_)
    print(
        f"OK: {len(model.clusters_)} clusters, silhouette={sil:.3f}, "
        f"f1_macro={row.get('f1_macro', float('nan')):.3f}"
    )


def pipeline_synthetic(steps: set[str] | None = None) -> None:
    all_steps = steps is None
    if all_steps or (steps and "data" in steps):
        print("=" * 70, "\n[1] Synthetic data (CSV demo)\n", "=" * 70)
        data.persist_synthetic()
    if all_steps or (steps and "preprocess" in steps):
        print("=" * 70, "\n[2] Preprocess\n", "=" * 70)
        preprocess.run_synthetic()
    if all_steps or (steps and "train" in steps):
        print("=" * 70, "\n[3] Train CLIQUE\n", "=" * 70)
        train.run_synthetic(do_grid_search=True)
    if all_steps or (steps and "benchmark" in steps):
        print("=" * 70, "\n[4] Benchmark\n", "=" * 70)
        benchmark.run_synthetic()
    print("\nDone (synthetic).")


def pipeline_retail(xlsx: str | None = None, steps: set[str] | None = None) -> None:
    """Retail pipeline from ``online_retail_ii.xlsx`` (in-memory; CSV is cache only)."""
    if steps is None:
        retail.run_from_xlsx(xlsx, save_artifacts=True, do_grid_search=True)
        return

    import joblib
    import pandas as pd

    config.ensure_dirs()
    path = data.resolve_retail_xlsx(xlsx)
    profiles = None
    prep = None
    model = None
    cluster_desc = None

    if "data" in steps:
        profiles = data.build_profiles_from_xlsx(path, save_artifacts=True)
    if "preprocess" in steps:
        if profiles is None:
            profiles = pd.read_csv(config.CUSTOMER_PROFILES_RAW_CSV)
        prep = preprocess.run_retail(profiles, winsorize=True)
    if "train" in steps:
        if prep is None:
            X_train = pd.read_csv(config.RETAIL_X_TRAIN_SCALED_CSV)[
                config.FEATURE_NAMES
            ].values.astype(float)
        else:
            X_train = prep["X_train"]
        model, cluster_desc, _ = train.run_retail(X_train)
    if "benchmark" in steps:
        if model is None:
            model = joblib.load(config.RETAIL_MODEL_PKL)
        if cluster_desc is None:
            cluster_desc = joblib.load(config.RETAIL_PROFILES_PKL)
        benchmark.run_retail(model, cluster_desc)

    print("\nDone (retail).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIQUE clustering — Online Retail II (xlsx) or synthetic demo"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="retail",
        choices=["retail", "synthetic", "verify"],
        help="Default: retail (reads online_retail_ii.xlsx)",
    )
    parser.add_argument(
        "--xlsx",
        type=str,
        default=None,
        help=f"Excel path (default: {config.ONLINE_RETAIL_XLSX})",
    )
    parser.add_argument(
        "--step",
        action="append",
        choices=["data", "preprocess", "train", "benchmark"],
        help="Run selected stages only (default: full pipeline)",
    )
    args = parser.parse_args()
    steps = set(args.step) if args.step else None

    if args.mode == "verify":
        verify_clique()
    elif args.mode == "synthetic":
        pipeline_synthetic(steps)
    else:
        pipeline_retail(args.xlsx, steps)


if __name__ == "__main__":
    main()
