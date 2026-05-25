"""Smoke test for CLIQUE implementation (run from project root)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clique.algorithm import CLIQUE
from clique.utils import FEATURE_NAMES, compute_silhouette

rng = np.random.default_rng(42)
X = rng.uniform(0, 1, (200, 8))
# Inject dense block
X[:60, 0] = rng.uniform(0.6, 0.8, 60)
X[:60, 1] = rng.uniform(0.7, 0.9, 60)

model = CLIQUE(xi=8, tau=0.05)
model.fit(X, feature_names=FEATURE_NAMES)
assert model.labels_ is not None
assert len(model.clusters_) >= 1
pred = model.predict(X[:5])
assert pred.shape == (5,)
hm = model.get_subspace_heatmap_data()
fig = model.plot_grid_static(X, 0, 1, FEATURE_NAMES)
sil = compute_silhouette(X, model.labels_)
print(f"OK: {len(model.clusters_)} clusters, silhouette={sil:.3f}, heatmap rows={len(hm)}")
