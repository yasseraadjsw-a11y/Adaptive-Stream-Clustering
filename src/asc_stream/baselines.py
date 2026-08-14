from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from numpy.typing import NDArray
from sklearn.cluster import KMeans, kmeans_plusplus

from .microclusters import OnlineMicroClusterSet
from .scaling import OnlineStandardizer


class MicroclusterBaseline:
    """Common study baseline for CluStream/DenStream-style summaries."""

    def __init__(
        self,
        original_dim: int,
        radius: float,
        max_microclusters: int,
        decay: float,
        seed: int,
        prune_policy: str = "utility",
        standardize: bool = True,
    ) -> None:
        self.original_dim = original_dim
        self.seed = seed
        self.scaler = OnlineStandardizer(original_dim) if standardize else None
        self.basis = np.eye(original_dim, dtype=np.float64)
        self.microclusters = OnlineMicroClusterSet(radius, max_microclusters, decay, prune_policy)
        self.time = 0

    def process_batch(self, x) -> NDArray[np.float64]:
        n_rows = int(x.shape[0])
        out = np.empty((n_rows, self.original_dim), dtype=np.float64)
        is_sparse = sparse.issparse(x)
        if is_sparse:
            x = x.tocsr()
        for i in range(n_rows):
            row = x.getrow(i).toarray().reshape(-1) if is_sparse else np.asarray(x[i], dtype=np.float64).reshape(-1)
            self.time += 1
            z = self.scaler.transform_then_update(row) if self.scaler is not None else row
            self.microclusters.update(z, z, self.basis, self.time, 1.0)
            out[i] = z
        return out

    def macro_centers(self, n_clusters: int) -> NDArray[np.float64]:
        centers, weights = self.microclusters.adapted_centers_and_weights(self.basis)
        k = min(n_clusters, len(centers))
        if k <= 1:
            return np.average(centers, axis=0, weights=weights, keepdims=True)
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        km.fit(centers, sample_weight=weights)
        return km.cluster_centers_

    def diagnostics(self) -> dict[str, float | int]:
        state_bytes = sum(c.centroid_raw.nbytes + 32 for c in self.microclusters.clusters)
        return {
            "final_microclusters": len(self.microclusters.clusters),
            "pruned_microclusters": self.microclusters.pruned,
            "merged_microclusters": self.microclusters.merged,
            "model_state_mb": state_bytes / 2**20,
        }


class CluStreamBaseline(MicroclusterBaseline):
    def __init__(self, original_dim: int, radius: float, max_microclusters: int, seed: int, standardize: bool = True) -> None:
        # CluStream maintains bounded micro-cluster summaries; no fading is
        # applied in this study implementation.
        super().__init__(original_dim, radius, max_microclusters, decay=1.0, seed=seed, prune_policy="utility", standardize=standardize)


@dataclass(slots=True)
class _DenMicroCluster:
    linear_sum: NDArray[np.float64]
    squared_sum: NDArray[np.float64]
    weight: float
    last_time: int
    created_at: int

    @property
    def center(self) -> NDArray[np.float64]:
        return self.linear_sum / max(self.weight, 1e-12)

    @property
    def radius(self) -> float:
        mean = self.center
        variance = np.maximum(self.squared_sum / max(self.weight, 1e-12) - mean * mean, 0.0)
        return float(np.sqrt(np.sum(variance)))


class DenStreamBaseline:
    """DenStream-style fading potential/outlier micro-clusters for the shared study evaluator.

    The online state follows the core DenStream design: exponentially faded
    sufficient statistics, potential and outlier micro-clusters, promotion of
    sufficiently weighted outliers, and periodic pruning. The final quality
    evaluator uses weighted K-Means on the maintained potential centers so ARI
    and NMI are computed under the same declared evaluation boundary as the
    other study methods.
    """

    def __init__(
        self,
        original_dim: int,
        radius: float,
        max_microclusters: int,
        seed: int,
        standardize: bool = True,
        beta: float = 0.20,
        mu: float = 6.0,
        fading_lambda: float = 0.01,
    ) -> None:
        if radius <= 0 or max_microclusters < 1 or not (0 < beta <= 1) or mu <= 0 or fading_lambda <= 0:
            raise ValueError("invalid DenStream parameters")
        if beta * mu <= 1.0:
            raise ValueError("DenStream requires beta * mu > 1 for the pruning interval")
        self.original_dim = original_dim
        self.radius = float(radius)
        self.max_microclusters = int(max_microclusters)
        self.seed = int(seed)
        self.beta = float(beta)
        self.mu = float(mu)
        self.fading_lambda = float(fading_lambda)
        self.scaler = OnlineStandardizer(original_dim) if standardize else None
        self.p_micro: list[_DenMicroCluster] = []
        self.o_micro: list[_DenMicroCluster] = []
        self.time = 0
        self.pruned = 0
        self.promoted = 0
        self.tp = max(
            1,
            int(np.ceil((1.0 / self.fading_lambda) * np.log2((self.beta * self.mu) / (self.beta * self.mu - 1.0)))),
        )

    def _decay_to(self, c: _DenMicroCluster, now: int) -> None:
        dt = max(now - c.last_time, 0)
        if dt:
            factor = 2.0 ** (-self.fading_lambda * dt)
            c.linear_sum *= factor
            c.squared_sum *= factor
            c.weight *= factor
            c.last_time = now

    def _candidate(self, c: _DenMicroCluster, x: NDArray[np.float64], now: int) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
        self._decay_to(c, now)
        ls = c.linear_sum + x
        ss = c.squared_sum + x * x
        w = c.weight + 1.0
        mean = ls / w
        var = np.maximum(ss / w - mean * mean, 0.0)
        return float(np.sqrt(np.sum(var))), ls, ss, w

    def _try_merge(self, clusters: list[_DenMicroCluster], x: NDArray[np.float64], now: int) -> int | None:
        if not clusters:
            return None
        for c in clusters:
            self._decay_to(c, now)
        centers = np.vstack([c.center for c in clusters])
        order = np.argsort(np.linalg.norm(centers - x, axis=1))
        for idx in order:
            c = clusters[int(idx)]
            r, ls, ss, w = self._candidate(c, x, now)
            if r <= self.radius:
                c.linear_sum = ls
                c.squared_sum = ss
                c.weight = w
                c.last_time = now
                return int(idx)
        return None

    def _prune(self, now: int) -> None:
        kept_p: list[_DenMicroCluster] = []
        for c in self.p_micro:
            self._decay_to(c, now)
            if c.weight >= self.beta * self.mu:
                kept_p.append(c)
            else:
                self.pruned += 1
        self.p_micro = kept_p

        kept_o: list[_DenMicroCluster] = []
        denom = 2.0 ** (-self.fading_lambda * self.tp) - 1.0
        for c in self.o_micro:
            self._decay_to(c, now)
            age = now - c.created_at
            numer = 2.0 ** (-self.fading_lambda * (age + self.tp)) - 1.0
            xi = numer / denom if abs(denom) > 1e-15 else 0.0
            if c.weight >= xi:
                kept_o.append(c)
            else:
                self.pruned += 1
        self.o_micro = kept_o

        while len(self.p_micro) + len(self.o_micro) > self.max_microclusters:
            pool = [(c.weight, "o", i) for i, c in enumerate(self.o_micro)]
            if not pool:
                pool = [(c.weight, "p", i) for i, c in enumerate(self.p_micro)]
            _, kind, idx = min(pool, key=lambda v: v[0])
            if kind == "o":
                del self.o_micro[idx]
            else:
                del self.p_micro[idx]
            self.pruned += 1

    def process_batch(self, x) -> NDArray[np.float64]:
        n_rows = int(x.shape[0])
        out = np.empty((n_rows, self.original_dim), dtype=np.float64)
        is_sparse = sparse.issparse(x)
        if is_sparse:
            x = x.tocsr()
        for i in range(n_rows):
            row = x.getrow(i).toarray().reshape(-1) if is_sparse else np.asarray(x[i], dtype=np.float64).reshape(-1)
            z = self.scaler.transform_then_update(row) if self.scaler is not None else row
            self.time += 1

            if self._try_merge(self.p_micro, z, self.time) is None:
                o_idx = self._try_merge(self.o_micro, z, self.time)
                if o_idx is not None:
                    c = self.o_micro[o_idx]
                    if c.weight >= self.beta * self.mu:
                        self.p_micro.append(c)
                        del self.o_micro[o_idx]
                        self.promoted += 1
                else:
                    self.o_micro.append(
                        _DenMicroCluster(z.copy(), z * z, 1.0, self.time, self.time)
                    )

            if self.time % self.tp == 0:
                self._prune(self.time)
            elif len(self.p_micro) + len(self.o_micro) > self.max_microclusters:
                self._prune(self.time)
            out[i] = z
        return out

    def macro_centers(self, n_clusters: int) -> NDArray[np.float64]:
        active = self.p_micro if self.p_micro else self.o_micro
        if not active:
            raise RuntimeError("DenStream has no active micro-clusters")
        for c in active:
            self._decay_to(c, self.time)
        centers = np.vstack([c.center for c in active])
        weights = np.asarray([max(c.weight, 1e-12) for c in active], dtype=np.float64)
        k = min(n_clusters, len(centers))
        if k <= 1:
            return np.average(centers, axis=0, weights=weights, keepdims=True)
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        km.fit(centers, sample_weight=weights)
        return km.cluster_centers_

    def diagnostics(self) -> dict[str, float | int]:
        state_bytes = sum(c.linear_sum.nbytes + c.squared_sum.nbytes + 32 for c in self.p_micro + self.o_micro)
        return {
            "potential_microclusters": len(self.p_micro),
            "outlier_microclusters": len(self.o_micro),
            "promoted_microclusters": self.promoted,
            "pruned_microclusters": self.pruned,
            "model_state_mb": state_bytes / 2**20,
        }


@dataclass(slots=True)
class CoresetPoint:
    value: NDArray[np.float64]
    weight: float


class StreamKMPlusPlusBaseline:
    """Bounded streaming coreset with D²-biased non-uniform reduction.

    The implementation keeps the study's declared 1000-point merge buffer and
    500-point coreset. Reduction uses weighted D²-biased representative
    selection followed by exact aggregation of source weights to their nearest
    selected representative. The final macro stage is the common weighted
    K-Means evaluator declared by the manuscript.
    """

    def __init__(
        self,
        original_dim: int,
        coreset_size: int = 500,
        buffer_size: int = 1000,
        seed: int = 7,
        standardize: bool = True,
    ) -> None:
        self.original_dim = int(original_dim)
        self.coreset_size = int(coreset_size)
        self.buffer_size = int(buffer_size)
        self.seed = int(seed)
        self.scaler = OnlineStandardizer(original_dim) if standardize else None
        self.values: list[NDArray[np.float64]] = []
        self.weights: list[float] = []
        self.compressions = 0

    def _compress(self) -> None:
        if len(self.values) <= self.coreset_size:
            return
        x = np.vstack(self.values)
        w = np.asarray(self.weights, dtype=np.float64)
        k = min(self.coreset_size, len(x))
        rng = np.random.default_rng(self.seed + self.compressions)

        # A small weighted k-means++ pilot estimates local representation cost.
        # Sampling the final representatives according to weighted D² cost is
        # non-uniform and data-adaptive, unlike the former MiniBatchKMeans
        # centroid replacement.  A small uniform floor preserves support in
        # already well represented regions.
        pilot_k = min(8, k, len(x))
        pilot, _ = kmeans_plusplus(
            x, n_clusters=pilot_k, sample_weight=np.maximum(w, 1e-12),
            random_state=self.seed + self.compressions,
        )
        min_d2 = np.full(len(x), np.inf, dtype=np.float64)
        for c in pilot:
            d2 = np.sum((x - c) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)
        score = np.maximum(w, 1e-12) * (min_d2 + 0.05 * max(float(np.mean(min_d2)), 1e-12))
        if not np.isfinite(score).all() or float(score.sum()) <= 0:
            score = np.maximum(w, 1e-12)
        prob = score / score.sum()
        chosen = rng.choice(len(x), size=k, replace=False, p=prob)
        reps = x[chosen].copy()

        # Preserve the full summary mass: every source point contributes its
        # current weight to the nearest selected D² representative.
        d2 = np.sum((x[:, None, :] - reps[None, :, :]) ** 2, axis=2)
        owner = np.argmin(d2, axis=1)
        new_w = np.bincount(owner, weights=w, minlength=k).astype(np.float64)
        self.values = [row.copy() for row in reps]
        self.weights = [float(v) for v in new_w]
        self.compressions += 1

    def process_batch(self, x) -> NDArray[np.float64]:
        n_rows = int(x.shape[0])
        out = np.empty((n_rows, self.original_dim), dtype=np.float64)
        is_sparse = sparse.issparse(x)
        if is_sparse:
            x = x.tocsr()
        for i in range(n_rows):
            row = x.getrow(i).toarray().reshape(-1) if is_sparse else np.asarray(x[i], dtype=np.float64).reshape(-1)
            z = self.scaler.transform_then_update(row) if self.scaler is not None else row
            self.values.append(z.copy())
            self.weights.append(1.0)
            if len(self.values) >= self.buffer_size:
                self._compress()
            out[i] = z
        return out

    def macro_centers(self, n_clusters: int) -> NDArray[np.float64]:
        self._compress()
        x = np.vstack(self.values)
        w = np.asarray(self.weights, dtype=np.float64)
        k = min(n_clusters, len(x))
        if k <= 1:
            return np.average(x, axis=0, weights=w, keepdims=True)
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10, init="k-means++")
        km.fit(x, sample_weight=w)
        return km.cluster_centers_

    def diagnostics(self) -> dict[str, float | int | str]:
        state = sum(v.nbytes for v in self.values) + 8 * len(self.weights)
        return {
            "final_coreset_points": len(self.values),
            "compressions": self.compressions,
            "model_state_mb": state / 2**20,
            "reduction": "weighted_D2_nonuniform_representative_sampling",
        }

