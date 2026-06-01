"""Smoke test for the CLIQUE implementation and metrics (run from project root)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from clique.algorithm import CLIQUE
from clique.metrics import compute_silhouette, evaluate_labeling

rng = np.random.default_rng(config.RANDOM_STATE)
X = rng.uniform(0, 1, (200, 8))
y = rng.integers(0, 3, size=200)
# Inject a dense block to guarantee at least one cluster.
X[:60, 0] = rng.uniform(0.6, 0.8, 60)
X[:60, 1] = rng.uniform(0.7, 0.9, 60)
y[:60] = 0

model = CLIQUE(xi=config.DEFAULT_XI, tau=config.DEFAULT_TAU)
model.fit(X, feature_names=config.FEATURE_NAMES)
assert model.labels_ is not None
assert len(model.clusters_) >= 1
assert model.predict(X[:5]).shape == (5,)
assert not model.get_subspace_heatmap_data().empty or len(model.clusters_) == 0

row = evaluate_labeling(X, model.predict(X), y, "CLIQUE")
sil = compute_silhouette(X, model.labels_)
print(
    f"OK: {len(model.clusters_)} clusters, silhouette={sil:.3f}, "
    f"f1_macro={row['f1_macro']:.3f}, roc_auc={row['roc_auc_ovr']:.3f}"
)
