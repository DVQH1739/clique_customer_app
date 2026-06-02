"""
CLIQUE subspace clustering — Agrawal et al., SIGMOD 1998.

Input X should be MinMax-scaled to approximately [0, 1] before fit/predict.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DEFAULT_XI: int = 8
DEFAULT_TAU: float = 0.05
NOISE_LABEL: int = -1


class CLIQUE:
    """CLIQUE subspace clustering with Apriori dense-unit discovery."""

    def __init__(self, xi: int = DEFAULT_XI, tau: float = DEFAULT_TAU) -> None:
        assert xi >= 2
        assert 0 < tau < 1
        self.xi = xi
        self.tau = tau

        self.dense_units_by_dim_: dict[int, list[dict[str, Any]]] = {}
        self.clusters_: list[dict[str, Any]] = []
        self.labels_: np.ndarray | None = None
        self.subspace_coverage_: dict[tuple[int, ...], float] = {}
        self.cluster_profiles_: dict[int, np.ndarray] = {}
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.feature_names_: list[str] | None = None
        self._X_train: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._training_runtime_sec: float | None = None
        self._cell_masks_: dict[tuple[int, int], np.ndarray] = {}

    @property
    def dense_units_(self) -> dict[tuple[int, ...], list[dict[str, Any]]]:
        out: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for units in self.dense_units_by_dim_.values():
            for unit in units:
                key = unit["dims"]
                out.setdefault(key, []).append(
                    {
                        "subspace": key,
                        "intervals": tuple(
                            int(round(unit["intervals"][d][0] * self.xi))
                            for d in key
                        ),
                        "count": unit["count"],
                        "dim_intervals": {
                            d: int(round(unit["intervals"][d][0] * self.xi))
                            for d in key
                        },
                    }
                )
        return out

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "CLIQUE":
        import time

        t0 = time.perf_counter()
        X = np.clip(np.asarray(X, dtype=np.float64), 0.0, 1.0)
        self._X_train = X
        self.n_samples_, self.n_features_ = X.shape
        names = feature_names or [f"f{i}" for i in range(self.n_features_)]
        self.feature_names_ = names
        self._feature_names = names

        self._build_cell_masks(X)
        threshold = self.tau * self.n_samples_

        self.dense_units_by_dim_ = {}
        self._phase1_find_dense_units(threshold)
        self._phase2_find_clusters(X)
        self._phase3_generate_descriptions()
        self._assign_labels()

        self.cluster_profiles_ = {}
        for cluster in self.clusters_:
            idx = cluster["points_idx"]
            if len(idx) > 0:
                self.cluster_profiles_[cluster["id"]] = X[idx].mean(axis=0)

        self._training_runtime_sec = time.perf_counter() - t0
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if not self.clusters_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_new = np.clip(np.asarray(X_new, dtype=np.float64), 0.0, 1.0)
        labels = np.full(len(X_new), NOISE_LABEL, dtype=int)
        sorted_clusters = sorted(
            self.clusters_, key=lambda c: len(c["subspace"]), reverse=True
        )
        for i, point in enumerate(X_new):
            for cluster in sorted_clusters:
                if self._point_in_cluster(point, cluster):
                    labels[i] = cluster["id"]
                    break
        return labels

    def get_subspace_heatmap_data(self) -> pd.DataFrame:
        names = self.feature_names_ or [f"f{i}" for i in range(self.n_features_)]
        rows: list[dict[str, Any]] = []
        for i in range(self.n_features_):
            for j in range(self.n_features_):
                if i == j:
                    rows.append(
                        {"dim_1": names[i], "dim_2": names[j], "n_clusters": 0, "coverage": 0.0}
                    )
                    continue
                sp = (min(i, j), max(i, j))
                n_cl = sum(1 for c in self.clusters_ if c["subspace"] == sp)
                rows.append(
                    {
                        "dim_1": names[i],
                        "dim_2": names[j],
                        "n_clusters": n_cl,
                        "coverage": self.subspace_coverage_.get(sp, 0.0),
                    }
                )
        return pd.DataFrame(rows)

    def plot_grid_static(
        self,
        X: np.ndarray,
        feature_x_idx: int,
        feature_y_idx: int,
        feature_names: list[str] | None = None,
    ) -> go.Figure:
        names = feature_names or self.feature_names_ or [f"f{i}" for i in range(self.n_features_)]
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
        fig = go.Figure()
        labels = self.labels_ if self.labels_ is not None else np.full(len(X), NOISE_LABEL)
        for label in np.unique(labels):
            mask = labels == label
            color = "#AAAAAA" if label == NOISE_LABEL else colors[int(label) % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=X[mask, feature_x_idx],
                    y=X[mask, feature_y_idx],
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.6),
                    name="Noise" if label == NOISE_LABEL else f"Cluster {label}",
                )
            )
        for i in range(self.xi + 1):
            v = i / self.xi
            fig.add_shape(type="line", x0=v, x1=v, y0=0, y1=1, line=dict(color="lightgray", width=0.5))
            fig.add_shape(type="line", x0=0, x1=1, y0=v, y1=v, line=dict(color="lightgray", width=0.5))
        for units in self.dense_units_by_dim_.values():
            for unit in units:
                dims = unit["dims"]
                if feature_x_idx not in dims or feature_y_idx not in dims:
                    continue
                lox, hix = unit["intervals"][feature_x_idx]
                loy, hiy = unit["intervals"][feature_y_idx]
                fig.add_shape(
                    type="rect", x0=lox, x1=hix, y0=loy, y1=hiy,
                    fillcolor="orange", opacity=0.25, line=dict(color="darkorange", width=1),
                )
        fig.update_layout(
            title=f"CLIQUE: {names[feature_x_idx]} × {names[feature_y_idx]}",
            xaxis=dict(range=[-0.02, 1.02]),
            yaxis=dict(range=[-0.02, 1.02]),
            width=700, height=600,
        )
        return fig

    def _build_cell_masks(self, X: np.ndarray) -> None:
        self._cell_masks_ = {}
        for dim in range(self.n_features_):
            for interval_idx in range(self.xi):
                lo = interval_idx / self.xi
                hi = (interval_idx + 1) / self.xi
                if interval_idx == self.xi - 1:
                    self._cell_masks_[(dim, interval_idx)] = (X[:, dim] >= lo) & (X[:, dim] <= hi)
                else:
                    self._cell_masks_[(dim, interval_idx)] = (X[:, dim] >= lo) & (X[:, dim] < hi)

    def _interval_to_idx(self, dim: int, lo: float) -> int:
        return int(round(lo * self.xi))

    def _count_unit(self, unit: dict[str, Any]) -> int:
        mask: np.ndarray | None = None
        for dim in sorted(unit["dims"]):
            idx = self._interval_to_idx(dim, unit["intervals"][dim][0])
            cell = self._cell_masks_[(dim, idx)]
            mask = cell if mask is None else (mask & cell)
        return int(mask.sum()) if mask is not None else 0

    def _unit_key(self, unit: dict[str, Any]) -> tuple[Any, ...]:
        return (
            unit["dims"],
            tuple(sorted((d, unit["intervals"][d]) for d in unit["dims"])),
        )

    def _phase1_find_dense_units(self, threshold: float) -> None:
        dense_1d: list[dict[str, Any]] = []
        for dim in range(self.n_features_):
            for interval_idx in range(self.xi):
                lo, hi = interval_idx / self.xi, (interval_idx + 1) / self.xi
                count = int(self._cell_masks_[(dim, interval_idx)].sum())
                if count >= threshold:
                    dense_1d.append(
                        {
                            "dims": (dim,),
                            "intervals": {dim: (lo, hi)},
                            "count": count,
                            "density": count / self.n_samples_,
                        }
                    )
        self.dense_units_by_dim_[1] = dense_1d

        prev_dense = dense_1d
        prev_keys = {self._unit_key(u): u for u in prev_dense}
        k = 2
        while k <= self.n_features_ and prev_dense:
            candidates = self._candidate_generation(prev_dense, k)
            current_dense: list[dict[str, Any]] = []
            for cand in candidates:
                if not self._all_projections_dense(cand, prev_keys):
                    continue
                count = self._count_unit(cand)
                if count >= threshold:
                    cand["count"] = count
                    cand["density"] = count / self.n_samples_
                    current_dense.append(cand)
            self.dense_units_by_dim_[k] = current_dense
            if not current_dense:
                break
            prev_dense = current_dense
            prev_keys = {self._unit_key(u): u for u in prev_dense}
            k += 1

    def _candidate_generation(self, dense_prev: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        by_prefix: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for unit in dense_prev:
            dims = tuple(sorted(unit["dims"]))
            if len(dims) != k - 1:
                continue
            by_prefix.setdefault(dims[:-1], []).append(unit)

        candidates: list[dict[str, Any]] = []
        for units in by_prefix.values():
            n = len(units)
            for i in range(n):
                for j in range(i + 1, n):
                    u1, u2 = units[i], units[j]
                    dims1 = sorted(u1["dims"])
                    dims2 = sorted(u2["dims"])
                    if dims1[:-1] != dims2[:-1] or dims1[-1] == dims2[-1]:
                        continue
                    new_dims = tuple(sorted(set(dims1) | set(dims2)))
                    if len(new_dims) != k:
                        continue
                    intervals = dict(u1["intervals"])
                    intervals.update(u2["intervals"])
                    candidates.append(
                        {"dims": new_dims, "intervals": intervals, "count": 0, "density": 0.0}
                    )
        return candidates

    def _all_projections_dense(
        self, candidate: dict[str, Any], prev_keys: dict[tuple[Any, ...], dict[str, Any]]
    ) -> bool:
        dims = list(candidate["dims"])
        for i in range(len(dims)):
            proj_dims = tuple(d for d in dims if d != dims[i])
            proj = {
                "dims": proj_dims,
                "intervals": {d: candidate["intervals"][d] for d in proj_dims},
            }
            if self._unit_key(proj) not in prev_keys:
                return False
        return True

    def _phase2_find_clusters(self, X: np.ndarray) -> None:
        self.clusters_ = []
        self.subspace_coverage_ = {}
        cluster_id = 0
        for k, dense_units in self.dense_units_by_dim_.items():
            subspace_map: dict[tuple[int, ...], list[dict[str, Any]]] = {}
            for unit in dense_units:
                subspace_map.setdefault(unit["dims"], []).append(unit)
            for subspace, units in subspace_map.items():
                for component in self._bfs_connected_components(units):
                    point_indices = self._get_points_in_component(component)
                    if len(point_indices) == 0:
                        continue
                    coverage = len(point_indices) / self.n_samples_
                    self.clusters_.append(
                        {
                            "id": cluster_id,
                            "subspace": subspace,
                            "k": k,
                            "units": component,
                            "size": len(point_indices),
                            "points_idx": point_indices,
                            "description": "",
                            "coverage": coverage,
                        }
                    )
                    self.subspace_coverage_[subspace] = max(
                        self.subspace_coverage_.get(subspace, 0.0), coverage
                    )
                    cluster_id += 1

    def _units_are_adjacent(self, u1: dict[str, Any], u2: dict[str, Any]) -> bool:
        if u1["dims"] != u2["dims"]:
            return False
        diff_count = 0
        for dim in u1["dims"]:
            lo1, hi1 = u1["intervals"][dim]
            lo2, hi2 = u2["intervals"][dim]
            if (lo1, hi1) != (lo2, hi2):
                diff_count += 1
                if not (abs(hi1 - lo2) < 1e-9 or abs(hi2 - lo1) < 1e-9):
                    return False
        return diff_count == 1

    def _bfs_connected_components(
        self, units: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        n = len(units)
        visited = [False] * n
        components: list[list[dict[str, Any]]] = []
        for start in range(n):
            if visited[start]:
                continue
            component: list[dict[str, Any]] = []
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                idx = queue.popleft()
                component.append(units[idx])
                for nb in range(n):
                    if not visited[nb] and self._units_are_adjacent(units[idx], units[nb]):
                        visited[nb] = True
                        queue.append(nb)
            components.append(component)
        return components

    def _get_points_in_component(self, component: list[dict[str, Any]]) -> np.ndarray:
        overall = np.zeros(self.n_samples_, dtype=bool)
        for unit in component:
            mask: np.ndarray | None = None
            for dim in sorted(unit["dims"]):
                idx = self._interval_to_idx(dim, unit["intervals"][dim][0])
                cell = self._cell_masks_[(dim, idx)]
                mask = cell if mask is None else (mask & cell)
            if mask is not None:
                overall |= mask
        return np.where(overall)[0]

    def _phase3_generate_descriptions(self) -> None:
        for cluster in self.clusters_:
            parts: list[str] = []
            dim_ranges: dict[int, list[float]] = {}
            for unit in cluster["units"]:
                for dim in cluster["subspace"]:
                    lo, hi = unit["intervals"][dim]
                    if dim not in dim_ranges:
                        dim_ranges[dim] = [lo, hi]
                    else:
                        dim_ranges[dim][0] = min(dim_ranges[dim][0], lo)
                        dim_ranges[dim][1] = max(dim_ranges[dim][1], hi)
            for dim in sorted(cluster["subspace"]):
                lo, hi = dim_ranges[dim]
                fname = self.feature_names_[dim] if self.feature_names_ else f"f{dim}"
                parts.append(f"{fname} ∈ [{lo:.2f}, {hi:.2f}]")
            cluster["description"] = " AND ".join(parts)

    def _assign_labels(self) -> None:
        self.labels_ = np.full(self.n_samples_, NOISE_LABEL, dtype=int)
        for cluster in sorted(self.clusters_, key=lambda c: c["k"], reverse=True):
            for idx in cluster["points_idx"]:
                if self.labels_[idx] == NOISE_LABEL:
                    self.labels_[idx] = cluster["id"]

    def _unit_dims(self, unit: dict[str, Any]) -> tuple[int, ...]:
        dims = unit.get("dims", unit.get("subspace"))
        if dims is None:
            raise KeyError("dense unit missing 'dims' or 'subspace'")
        return tuple(dims)

    def _unit_interval(self, unit: dict[str, Any], dim: int) -> tuple[float, float]:
        """Return (lo, hi) in [0, 1]; supports legacy grid-index units."""
        if isinstance(unit.get("intervals"), dict) and dim in unit["intervals"]:
            lo, hi = unit["intervals"][dim]
            return float(lo), float(hi)
        dim_intervals = unit.get("dim_intervals")
        if isinstance(dim_intervals, dict) and dim in dim_intervals:
            idx = int(dim_intervals[dim])
            return idx / self.xi, (idx + 1) / self.xi
        raw = unit.get("intervals")
        if isinstance(raw, tuple):
            dims = self._unit_dims(unit)
            pos = dims.index(dim)
            idx = int(raw[pos])
            return idx / self.xi, (idx + 1) / self.xi
        raise KeyError(f"Cannot resolve interval for dim {dim}")

    def _point_in_cluster(self, point: np.ndarray, cluster: dict[str, Any]) -> bool:
        for unit in cluster["units"]:
            ok = True
            dims_sorted = sorted(self._unit_dims(unit))
            for dim in dims_sorted:
                lo, hi = self._unit_interval(unit, dim)
                if dim == max(dims_sorted):
                    if not (lo <= point[dim] <= hi):
                        ok = False
                        break
                elif not (lo <= point[dim] < hi):
                    ok = False
                    break
            if ok:
                return True
        return False
