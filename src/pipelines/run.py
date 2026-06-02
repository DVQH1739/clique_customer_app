"""
Single entry point for all pipelines.

Usage (from ``clique_customer_app/``):

    python src/pipelines/run.py synthetic
    python src/pipelines/run.py retail [--xlsx PATH]
    python src/pipelines/run.py verify
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
from pipelines import benchmark, data, preprocess, train


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
        print("=" * 70, "\n[1] Synthetic data\n", "=" * 70)
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
    print("\nDone (synthetic). See results/ and models/.")


def pipeline_retail(xlsx: str | None = None, steps: set[str] | None = None) -> None:
    import joblib
    import pandas as pd

    config.ensure_dirs()
    path = data.resolve_retail_xlsx(xlsx)
    all_steps = steps is None
    profiles: pd.DataFrame | None = None
    cluster_desc: pd.DataFrame | None = None
    model: CLIQUE | None = None

    if all_steps or (steps and "data" in steps):
        print("=" * 70, "\n[1–2] Load, clean, features\n", "=" * 70)
        _, _, profiles = data.load_and_clean_retail(path)
    if all_steps or (steps and "preprocess" in steps):
        if profiles is None:
            if not config.CUSTOMER_PROFILES_RAW_CSV.exists():
                raise FileNotFoundError("Run with --step data first.")
            profiles = pd.read_csv(config.CUSTOMER_PROFILES_RAW_CSV)
        print("=" * 70, "\n[3] Preprocess\n", "=" * 70)
        preprocess.run_retail(profiles)
    if all_steps or (steps and "train" in steps):
        print("=" * 70, "\n[4] Train\n", "=" * 70)
        X_train = pd.read_csv(config.RETAIL_X_TRAIN_SCALED_CSV)[config.FEATURE_NAMES].values.astype(float)
        model, cluster_desc = train.run_retail(X_train)
    if all_steps or (steps and "benchmark" in steps):
        print("=" * 70, "\n[5] Benchmark + test\n", "=" * 70)
        if model is None:
            model = joblib.load(config.RETAIL_MODEL_PKL)
        if cluster_desc is None:
            loaded = joblib.load(config.RETAIL_PROFILES_PKL)
            if isinstance(loaded, pd.DataFrame):
                cluster_desc = loaded
            else:
                X_train = pd.read_csv(config.RETAIL_X_TRAIN_SCALED_CSV)[config.FEATURE_NAMES].values.astype(float)
                cluster_desc = train.build_cluster_descriptions(model, X_train)
        benchmark.run_retail(model, cluster_desc)

    print("\nDone (retail).")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIQUE customer segmentation pipelines")
    parser.add_argument(
        "mode",
        choices=["synthetic", "retail", "verify"],
        help="Pipeline: synthetic (600 labeled), retail (Online Retail II), or verify",
    )
    parser.add_argument("--xlsx", type=str, default=None, help="Path to online_retail_ii.xlsx")
    parser.add_argument(
        "--step",
        action="append",
        choices=["data", "preprocess", "train", "benchmark"],
        help="Run only selected stages (default: all)",
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
