from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class MicroCluster:
    cluster_id: int
    centroid_raw: NDArray[np.float64]
    weight: float
    last_seen: int
    created_at: int


class OnlineMicroClusterSet:
    def __init__(self, radius: float, max_clusters: int, decay: float, prune_policy: str = "utility") -> None:
        self.radius = radius
        self.max_clusters = max_clusters
        self.decay = decay
        self.prune_policy = prune_policy
        self.clusters: list[MicroCluster] = []
        self.next_id = 0
        self.pruned = 0
        self.merged = 0

    def _adapt_centroids(self, basis: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.clusters:
            return np.empty((0, basis.shape[0]), dtype=np.float64)
        raw = np.vstack([c.centroid_raw for c in self.clusters])
        return (raw @ basis) @ basis.T

    def _merge_closest(self, basis: NDArray[np.float64], now: int) -> None:
        if len(self.clusters) < 2:
            return
        centers = self._adapt_centroids(basis)
        diff = centers[:, None, :] - centers[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        np.fill_diagonal(d2, np.inf)
        a, b = np.unravel_index(np.argmin(d2), d2.shape)
        if b < a:
            a, b = b, a
        ca, cb = self.clusters[a], self.clusters[b]
        total = ca.weight + cb.weight
        ca.centroid_raw = (ca.centroid_raw * ca.weight + cb.centroid_raw * cb.weight) / max(total, 1e-12)
        ca.weight = total
        ca.last_seen = max(ca.last_seen, cb.last_seen, now)
        del self.clusters[b]
        self.merged += 1

    def _make_space(self, basis: NDArray[np.float64], now: int) -> None:
        if len(self.clusters) < self.max_clusters:
            return
        if self.prune_policy == "merge":
            self._merge_closest(basis, now)
            return
        # Low weight and age jointly indicate low current utility.
        scores = np.asarray(
            [c.weight / (1.0 + max(now - c.last_seen, 0)) for c in self.clusters],
            dtype=np.float64,
        )
        del self.clusters[int(np.argmin(scores))]
        self.pruned += 1


    def nearest_id(
        self, z_adapted: NDArray[np.float64], basis: NDArray[np.float64]
    ) -> int:
        """Return the nearest current micro-cluster without mutating state."""
        if not self.clusters:
            return -1
        adapted_centers = self._adapt_centroids(basis)
        return self.clusters[int(np.argmin(np.linalg.norm(adapted_centers - z_adapted, axis=1)))].cluster_id

    def update(
        self,
        z_raw: NDArray[np.float64],
        z_adapted: NDArray[np.float64],
        basis: NDArray[np.float64],
        now: int,
        sample_weight: float = 1.0,
        radius_scale: float = 1.0,
    ) -> int:
        sample_weight = float(max(sample_weight, 1e-12))
        if not self.clusters:
            cid = self.next_id
            self.next_id += 1
            self.clusters.append(MicroCluster(cid, z_raw.copy(), sample_weight, now, now))
            return cid

        adapted_centers = self._adapt_centroids(basis)
        distances = np.linalg.norm(adapted_centers - z_adapted, axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] <= self.radius * max(float(radius_scale), 1e-6):
            cluster = self.clusters[idx]
            old_weight = self.decay * cluster.weight
            new_weight = old_weight + sample_weight
            cluster.centroid_raw = (old_weight * cluster.centroid_raw + sample_weight * z_raw) / max(new_weight, 1e-12)
            cluster.weight = new_weight
            cluster.last_seen = now
            return cluster.cluster_id

        self._make_space(basis, now)
        cid = self.next_id
        self.next_id += 1
        self.clusters.append(MicroCluster(cid, z_raw.copy(), sample_weight, now, now))
        return cid

    def adapted_centers_and_weights(self, basis: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._adapt_centroids(basis), np.asarray([c.weight for c in self.clusters], dtype=np.float64)
