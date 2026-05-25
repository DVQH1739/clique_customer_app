"""
CLIQUE subspace clustering — Agrawal et al., SIGMOD 1998.

Input X must be MinMax-scaled to [0, 1] before fit/predict.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Module-level constants
DEFAULT_XI: int = 8
DEFAULT_TAU: float = 0.05
NOISE_LABEL: int = -1


class CLIQUE:
    """
    CLIQUE (CLustering In QUEst) subspace clustering on [0, 1]-scaled data.

    Parameters
    ----------
    xi : int
        Number of equal-width intervals per dimension.
    tau : float
        Minimum density threshold as a fraction of total points.
    """

    def __init__(self, xi: int = DEFAULT_XI, tau: float = DEFAULT_TAU) -> None:
        self.xi: int = xi
        self.tau: float = tau
        self.dense_units_: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        self.clusters_: list[dict[str, Any]] = []
        self.labels_: np.ndarray | None = None
        self.subspace_coverage_: dict[tuple[int, ...], float] = {}
        self.cluster_profiles_: dict[int, np.ndarray] = {}
        self._X_train: np.ndarray | None = None
        self._n_points: int = 0
        self._feature_names: list[str] | None = None
        self._training_runtime_sec: float | None = None

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "CLIQUE":
        """
        Fit CLIQUE on MinMax-scaled data in [0, 1].

        Phase 1: subspace identification (bottom-up Apriori).
        Phase 2: cluster determination via unit adjacency + BFS.
        Phase 3: minimal DNF descriptions per cluster.
        """
        import time

        t0 = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.min() < 0.0 or X.max() > 1.0:
            raise ValueError("X must be MinMax-scaled to [0, 1]")

        self._X_train = X
        self._n_points = X.shape[0]
        n_dims = X.shape[1]
        self._feature_names = feature_names or [f"dim_{i}" for i in range(n_dims)]

        N = self._n_points
        threshold_count = self.tau * N

        # --- Phase 1: Subspace identification ---
        self.dense_units_ = {}

        # 1D dense units
        for d in range(n_dims):
            units_1d: list[dict[str, Any]] = []
            for interval in range(self.xi):
                count = self._count_points_in_unit(X, {d: interval})
                if count >= threshold_count:
                    units_1d.append(
                        {
                            "subspace": (d,),
                            "intervals": (interval,),
                            "count": count,
                            "dim_intervals": {d: interval},
                        }
                    )
            if units_1d:
                self.dense_units_[(d,)] = units_1d

        # Grow subspaces from k=2 to n_dims
        for k in range(2, n_dims + 1):
            new_dense: dict[tuple[int, ...], list[dict[str, Any]]] = {}
            # Extend each (k-1)-dim dense subspace with a new dimension
            for subspace_k1, units_k1 in list(self.dense_units_.items()):
                if len(subspace_k1) != k - 1:
                    continue
                for d_new in range(n_dims):
                    if d_new in subspace_k1:
                        continue
                    subspace_k = tuple(sorted((*subspace_k1, d_new)))
                    dense_1d_new = self.dense_units_.get((d_new,), [])
                    if not dense_1d_new:
                        continue
                    intervals_1d_new = {u["intervals"][0] for u in dense_1d_new}
                    for unit_k1 in units_k1:
                        for interval_new in intervals_1d_new:
                            dim_intervals = self._unit_to_dim_intervals(
                                subspace_k, unit_k1["intervals"], subspace_k1, d_new, interval_new
                            )
                            if not self._all_projections_dense(
                                subspace_k, dim_intervals, self.dense_units_
                            ):
                                continue
                            count = self._count_points_in_unit(X, dim_intervals)
                            if count >= threshold_count:
                                intervals_k = self._dims_to_intervals_tuple(
                                    subspace_k, dim_intervals
                                )
                                entry = {
                                    "subspace": subspace_k,
                                    "intervals": intervals_k,
                                    "count": count,
                                    "dim_intervals": dim_intervals,
                                }
                                new_dense.setdefault(subspace_k, []).append(entry)

            # Self-join within k-dim candidates from (k-1) — not needed for discovery;
            # extension above covers Apriori growth. Deduplicate by intervals key.
            for subspace_k, units in new_dense.items():
                seen: set[tuple[int, ...]] = set()
                deduped: list[dict[str, Any]] = []
                for u in units:
                    key = u["intervals"]
                    if key not in seen:
                        seen.add(key)
                        deduped.append(u)
                if deduped:
                    self.dense_units_[subspace_k] = deduped

        # --- Phase 2: Cluster determination ---
        self.clusters_ = []
        cluster_id = 0
        point_cluster_map: dict[int, list[tuple[int, int]]] = {
            i: [] for i in range(N)
        }  # point -> [(cluster_id, subspace_dim), ...]

        for subspace, units in self.dense_units_.items():
            if not units:
                continue
            components = self._bfs_connected_components(units)
            covered_points: set[int] = set()
            for component in components:
                dim_intervals_list = [
                    u.get("dim_intervals")
                    or self._intervals_to_dim_intervals(subspace, u["intervals"])
                    for u in component
                ]
                point_indices = self._points_in_units(X, dim_intervals_list)
                if not point_indices:
                    continue
                for idx in point_indices:
                    point_cluster_map[idx].append((cluster_id, len(subspace)))
                covered_points.update(point_indices)
                self.clusters_.append(
                    {
                        "id": cluster_id,
                        "subspace": subspace,
                        "units": component,
                        "size": len(point_indices),
                        "points_idx": np.array(point_indices, dtype=int),
                        "description": "",
                    }
                )
                cluster_id += 1

            coverage = len(covered_points) / N if N > 0 else 0.0
            self.subspace_coverage_[subspace] = coverage

        # Assign labels: highest-dimensional cluster wins
        self.labels_ = np.full(N, NOISE_LABEL, dtype=int)
        for i in range(N):
            assignments = point_cluster_map[i]
            if assignments:
                best = max(assignments, key=lambda x: x[1])
                self.labels_[i] = best[0]

        # --- Phase 3: Minimal descriptions ---
        for cluster in self.clusters_:
            cluster["description"] = self._generate_dnf_description(
                cluster["subspace"], cluster["units"]
            )

        # Cluster profiles (mean in scaled space)
        self.cluster_profiles_ = {}
        for cluster in self.clusters_:
            idx = cluster["points_idx"]
            if len(idx) > 0:
                self.cluster_profiles_[cluster["id"]] = X[idx].mean(axis=0)

        self._training_runtime_sec = time.perf_counter() - t0
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Assign each point to the highest-dimensional cluster whose dense unit it falls into.
        """
        if not self.clusters_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_new = np.asarray(X_new, dtype=np.float64)
        n = X_new.shape[0]
        labels = np.full(n, NOISE_LABEL, dtype=int)

        for i in range(n):
            best_dim = -1
            best_cluster = NOISE_LABEL
            for cluster in self.clusters_:
                subspace = cluster["subspace"]
                for unit in cluster["units"]:
                    dim_intervals = unit.get("dim_intervals") or self._intervals_to_dim_intervals(
                        subspace, unit["intervals"]
                    )
                    if self._point_in_unit(X_new[i], dim_intervals):
                        if len(subspace) > best_dim:
                            best_dim = len(subspace)
                            best_cluster = cluster["id"]
                        break
            labels[i] = best_cluster
        return labels

    def get_subspace_heatmap_data(self) -> pd.DataFrame:
        """DataFrame for n×n subspace heatmap (2D subspaces with clusters)."""
        if self._X_train is None:
            raise RuntimeError("Model has not been fitted.")

        n_dims = self._X_train.shape[1]
        names = self._feature_names or [f"dim_{i}" for i in range(n_dims)]
        rows: list[dict[str, Any]] = []

        for d1 in range(n_dims):
            for d2 in range(d1 + 1, n_dims):
                subspace = tuple(sorted((d1, d2)))
                n_clusters = sum(
                    1 for c in self.clusters_ if c["subspace"] == subspace
                )
                coverage = self.subspace_coverage_.get(subspace, 0.0)
                if n_clusters > 0 or subspace in self.dense_units_:
                    rows.append(
                        {
                            "dim_1": names[d1],
                            "dim_2": names[d2],
                            "n_clusters": n_clusters,
                            "coverage": coverage,
                        }
                    )
        return pd.DataFrame(rows)

    def plot_grid_static(
        self,
        X: np.ndarray,
        feature_x_idx: int,
        feature_y_idx: int,
        feature_names: list[str],
    ) -> go.Figure:
        """
        Plotly figure: scatter, grid, dense units, cluster rectangles.
        """
        X = np.asarray(X, dtype=np.float64)
        subspace = tuple(sorted((feature_x_idx, feature_y_idx)))
        fig = go.Figure()

        # All points (grey)
        fig.add_trace(
            go.Scatter(
                x=X[:, feature_x_idx],
                y=X[:, feature_y_idx],
                mode="markers",
                marker=dict(size=4, color="grey", opacity=0.4),
                name="All points",
                showlegend=True,
            )
        )

        # Grid lines
        for i in range(self.xi + 1):
            v = i / self.xi
            fig.add_vline(x=v, line_width=0.5, line_color="lightgray", line_dash="dot")
            fig.add_hline(y=v, line_width=0.5, line_color="lightgray", line_dash="dot")

        # Dense unit cells (orange)
        dense_units = self.dense_units_.get(subspace, [])
        shapes_dense: list[dict[str, Any]] = []
        for unit in dense_units:
            dx, dy = unit["intervals"]
            x0, x1 = dx / self.xi, (dx + 1) / self.xi
            y0, y1 = dy / self.xi, (dy + 1) / self.xi
            shapes_dense.append(
                dict(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    fillcolor="orange",
                    opacity=0.35,
                    line_width=0,
                    layer="below",
                )
            )

        # Cluster outlines
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
        cluster_shapes: list[dict[str, Any]] = []
        legend_added: set[int] = set()
        for cluster in self.clusters_:
            if cluster["subspace"] != subspace:
                continue
            cid = cluster["id"]
            color = colors[cid % len(colors)]
            xs, ys = [], []
            for unit in cluster["units"]:
                dx, dy = unit["intervals"]
                x0, x1 = dx / self.xi, (dx + 1) / self.xi
                y0, y1 = dy / self.xi, (dy + 1) / self.xi
                cluster_shapes.append(
                    dict(
                        type="rect",
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        line=dict(color=color, width=3),
                        fillcolor="rgba(0,0,0,0)",
                        layer="above",
                    )
                )
                xs.extend([x0, x1])
                ys.extend([y0, y1])
            if cid not in legend_added:
                fig.add_trace(
                    go.Scatter(
                        x=[np.mean(xs)] if xs else [0.5],
                        y=[np.mean(ys)] if ys else [0.5],
                        mode="markers",
                        marker=dict(size=12, color=color, symbol="square"),
                        name=f"Cluster {cid}",
                    )
                )
                legend_added.add(cid)

        fig.update_layout(
            shapes=shapes_dense + cluster_shapes,
            title=f"Subspace: {feature_names[feature_x_idx]} × {feature_names[feature_y_idx]}",
            xaxis_title=feature_names[feature_x_idx],
            yaxis_title=feature_names[feature_y_idx],
            xaxis=dict(range=[0, 1], constrain="domain"),
            yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
            width=700,
            height=650,
        )
        return fig

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_unit_key(intervals: tuple[int, ...]) -> tuple[int, ...]:
        """Hashable unit identifier from interval indices."""
        return tuple(intervals)

    def _count_points_in_unit(self, X: np.ndarray, dim_intervals: dict[int, int]) -> int:
        """Count points falling in the hyper-rectangular unit."""
        mask = self._mask_for_unit(X, dim_intervals)
        return int(mask.sum())

    def _mask_for_unit(self, X: np.ndarray, dim_intervals: dict[int, int]) -> np.ndarray:
        mask = np.ones(X.shape[0], dtype=bool)
        for d, interval in dim_intervals.items():
            low = interval / self.xi
            high = (interval + 1) / self.xi
            if interval == self.xi - 1:
                mask &= (X[:, d] >= low) & (X[:, d] <= high)
            else:
                mask &= (X[:, d] >= low) & (X[:, d] < high)
        return mask

    def _point_in_unit(self, x: np.ndarray, dim_intervals: dict[int, int]) -> bool:
        for d, interval in dim_intervals.items():
            low = interval / self.xi
            high = (interval + 1) / self.xi
            if interval == self.xi - 1:
                if not (low <= x[d] <= high):
                    return False
            else:
                if not (low <= x[d] < high):
                    return False
        return True

    def _candidate_generation(
        self, dense_units_k_minus_1: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apriori self-join: units sharing first (k-2) intervals, differing in one.
        """
        candidates: list[dict[str, Any]] = []
        n_units = len(dense_units_k_minus_1)
        if n_units < 2:
            return candidates

        subspace = dense_units_k_minus_1[0]["subspace"]
        k = len(subspace)
        if k < 2:
            return candidates

        for i in range(n_units):
            for j in range(i + 1, n_units):
                u1 = dense_units_k_minus_1[i]["intervals"]
                u2 = dense_units_k_minus_1[j]["intervals"]
                if len(u1) != k or len(u2) != k:
                    continue
                # Share first k-2 dimensions
                if k >= 2 and u1[: k - 2] != u2[: k - 2]:
                    continue
                # Differ in exactly one position
                diffs = sum(1 for a, b in zip(u1, u2) if a != b)
                if diffs == 1:
                    # Merged candidate is the union envelope (for connectivity, keep both)
                    merged_intervals = tuple(
                        min(a, b) if a != b else a for a, b in zip(u1, u2)
                    )
                    candidates.append(
                        {
                            "subspace": subspace,
                            "intervals": merged_intervals,
                        }
                    )
        return candidates

    def _all_projections_dense(
        self,
        subspace: tuple[int, ...],
        dim_intervals: dict[int, int],
        dense_units: dict[tuple[int, ...], list[dict[str, Any]]],
    ) -> bool:
        """Downward-closure: every (k-1)-dim projection must be dense."""
        k = len(subspace)
        if k <= 1:
            return True
        for proj_dims in combinations(subspace, k - 1):
            proj_dims_sorted = tuple(sorted(proj_dims))
            proj_intervals = tuple(
                dim_intervals[d]
                for d in sorted(proj_dims)
            )
            dense_list = dense_units.get(proj_dims_sorted, [])
            found = any(
                u["intervals"] == proj_intervals for u in dense_list
            )
            if not found:
                return False
        return True

    def _bfs_connected_components(
        self, dense_units: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Connected components of dense units under adjacency."""
        n = len(dense_units)
        if n == 0:
            return []
        visited = [False] * n
        components: list[list[dict[str, Any]]] = []

        for start in range(n):
            if visited[start]:
                continue
            component: list[dict[str, Any]] = []
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                cur = queue.popleft()
                component.append(dense_units[cur])
                for nb in range(n):
                    if not visited[nb] and self._units_are_adjacent(
                        dense_units[cur], dense_units[nb]
                    ):
                        visited[nb] = True
                        queue.append(nb)
            components.append(component)
        return components

    def _units_are_adjacent(
        self, u1: dict[str, Any], u2: dict[str, Any]
    ) -> bool:
        """Adjacent if they differ in exactly one interval in one dimension."""
        i1 = u1["intervals"]
        i2 = u2["intervals"]
        if len(i1) != len(i2):
            return False
        diffs = []
        for a, b in zip(i1, i2):
            if a != b:
                diffs.append(abs(a - b))
        if len(diffs) != 1:
            return False
        return diffs[0] == 1

    def _intervals_to_dim_intervals(
        self, subspace: tuple[int, ...], intervals: tuple[int, ...]
    ) -> dict[int, int]:
        return {d: intervals[i] for i, d in enumerate(subspace)}

    def _unit_to_dim_intervals(
        self,
        subspace_k: tuple[int, ...],
        intervals_k1: tuple[int, ...],
        subspace_k1: tuple[int, ...],
        d_new: int,
        interval_new: int,
    ) -> dict[int, int]:
        dim_intervals: dict[int, int] = {}
        for d in subspace_k1:
            idx = subspace_k1.index(d)
            dim_intervals[d] = intervals_k1[idx]
        dim_intervals[d_new] = interval_new
        return dim_intervals

    def _dims_to_intervals_tuple(
        self, subspace: tuple[int, ...], dim_intervals: dict[int, int]
    ) -> tuple[int, ...]:
        return tuple(dim_intervals[d] for d in subspace)

    def _points_in_units(
        self, X: np.ndarray, dim_intervals_list: list[dict[int, int]]
    ) -> list[int]:
        mask = np.zeros(X.shape[0], dtype=bool)
        for di in dim_intervals_list:
            mask |= self._mask_for_unit(X, di)
        return list(np.where(mask)[0])

    def _generate_dnf_description(
        self,
        subspace: tuple[int, ...],
        units: list[dict[str, Any]],
    ) -> str:
        """
        Greedy cover: merge adjacent units into hyper-rectangles, emit DNF clauses.
        """
        if not units or self._feature_names is None:
            return ""

        names = self._feature_names
        # Build axis-aligned bounding boxes per unit, then greedy merge
        boxes: list[dict[int, tuple[float, float]]] = []
        for unit in units:
            box: dict[int, tuple[float, float]] = {}
            for i, d in enumerate(subspace):
                interval = unit["intervals"][i]
                low = interval / self.xi
                high = (interval + 1) / self.xi
                box[d] = (low, high)
            boxes.append(box)

        merged = self._greedy_cover_boxes(boxes, subspace)
        clauses: list[str] = []
        for box in merged:
            parts = []
            for d in subspace:
                low, high = box[d]
                parts.append(f"{names[d]} ∈ [{low:.2f}, {high:.2f}]")
            clauses.append(" AND ".join(parts))
        return " OR ".join(clauses) if len(clauses) > 1 else (clauses[0] if clauses else "")

    def _greedy_cover_boxes(
        self,
        boxes: list[dict[int, tuple[float, float]]],
        subspace: tuple[int, ...],
    ) -> list[dict[int, tuple[float, float]]]:
        """Greedy merge of overlapping/adjacent boxes along one dimension at a time."""
        if not boxes:
            return []
        remaining = list(boxes)
        merged: list[dict[int, tuple[float, float]]] = []

        while remaining:
            current = remaining.pop(0)
            changed = True
            while changed:
                changed = False
                new_remaining = []
                for other in remaining:
                    if self._boxes_mergeable(current, other, subspace):
                        current = self._merge_boxes(current, other, subspace)
                        changed = True
                    else:
                        new_remaining.append(other)
                remaining = new_remaining
            merged.append(current)
        return merged

    @staticmethod
    def _boxes_mergeable(
        b1: dict[int, tuple[float, float]],
        b2: dict[int, tuple[float, float]],
        subspace: tuple[int, ...],
    ) -> bool:
        """Mergeable if equal on all but one dimension and adjacent/overlapping on that dim."""
        diff_dims = 0
        for d in subspace:
            if b1[d] != b2[d]:
                diff_dims += 1
                if diff_dims > 1:
                    return False
        return diff_dims == 1

    @staticmethod
    def _merge_boxes(
        b1: dict[int, tuple[float, float]],
        b2: dict[int, tuple[float, float]],
        subspace: tuple[int, ...],
    ) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for d in subspace:
            out[d] = (min(b1[d][0], b2[d][0]), max(b1[d][1], b2[d][1]))
        return out
