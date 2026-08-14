from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns

import numpy as np
from scipy import sparse
from numpy.typing import NDArray

from .config import ASCConfig
from .microclusters import OnlineMicroClusterSet
from .projection import make_sparse_achlioptas_projection
from .scaling import OnlineStandardizer
from .sketch import AdaptiveProjectedSketch, BasisUpdate


@dataclass(slots=True)
class ModelStep:
    microcluster_id: int
    projected: NDArray[np.float64]
    adapted: NDArray[np.float64]
    basis_update: BasisUpdate


@dataclass(slots=True)
class PhaseTimings:
    scaling_ns: int = 0
    projection_ns: int = 0
    sketch_ns: int = 0
    adaptation_ns: int = 0
    microcluster_ns: int = 0

    @property
    def total_ns(self) -> int:
        return self.scaling_ns + self.projection_ns + self.sketch_ns + self.adaptation_ns + self.microcluster_ns

    def to_seconds(self) -> dict[str, float]:
        return {
            "scaling_seconds": self.scaling_ns / 1e9,
            "projection_seconds": self.projection_ns / 1e9,
            "sketch_seconds": self.sketch_ns / 1e9,
            "adaptation_seconds": self.adaptation_ns / 1e9,
            "microcluster_seconds": self.microcluster_ns / 1e9,
            "instrumented_online_seconds": self.total_ns / 1e9,
        }


class AdaptiveSketchStreamClusterer:
    def __init__(self, config: ASCConfig, standardize: bool = True) -> None:
        config.validate()
        self.config = config
        # Separate the projection realization from the online stochastic seed.
        # This permits Proposed/Fixed to share an identical projected stream within
        # each paired repetition while still allowing projection seeds to vary across repetitions.
        projection_seed = config.seed if config.projection_seed is None else config.projection_seed
        projection_rng = np.random.default_rng(projection_seed)
        self.rng = np.random.default_rng(config.seed)
        if config.projection_mode == "identity":
            self.omega = np.eye(config.original_dim, dtype=np.float64)
        elif config.projection_mode == "dense":
            self.omega = projection_rng.normal(
                scale=1.0 / np.sqrt(config.projection_dim),
                size=(config.original_dim, config.projection_dim),
            )
        else:
            self.omega = make_sparse_achlioptas_projection(
                config.original_dim, config.projection_dim, config.projection_sparsity, projection_rng
            )
        self.scaler = OnlineStandardizer(config.original_dim) if standardize else None
        self.sketch = AdaptiveProjectedSketch(
            projection_dim=config.projection_dim,
            window_size=config.window_size,
            initial_rank=config.initial_rank,
            min_rank=config.min_rank,
            max_rank=config.max_rank,
            smoothing=config.threshold_smoothing,
            retained_energy=config.retained_energy,
            ridge_lambda=config.leverage_regularization,
            update_interval=config.basis_update_interval,
            tolerance=config.rank_tolerance,
            stable_before_shrink=config.stable_intervals_before_shrink,
            uniform_floor=config.uniform_floor,
            leverage_mode=config.leverage_mode,
            sampling_rate=config.leverage_sampling_rate,
            min_sampling_probability=config.min_sampling_probability,
            max_sampling_probability=config.max_sampling_probability,
            min_leverage_weight=config.min_leverage_weight,
            max_leverage_weight=config.max_leverage_weight,
            rng=self.rng,
        )
        self.microclusters = OnlineMicroClusterSet(
            config.microcluster_radius, config.max_microclusters, config.decay, config.prune_policy
        )
        self.time = 0
        self.rank_history: list[int] = []
        self.error_history: list[float] = []
        self.threshold_history: list[float] = []
        self.leverage_history: list[float] = []
        self.weight_history: list[float] = []
        self.basis_update_trace: list[dict[str, float | int | bool]] = []
        self.timings = PhaseTimings()

    @property
    def basis(self) -> NDArray[np.float64]:
        return self.sketch.basis[:, : self.sketch.rank]

    @property
    def clustering_basis(self) -> NDArray[np.float64]:
        if self.config.use_adapted_representation_for_clustering:
            return self.basis
        return np.eye(self.config.projection_dim, dtype=np.float64)

    def process_one(self, x) -> ModelStep:
        self.time += 1
        is_sparse = sparse.issparse(x)
        if is_sparse and self.scaler is not None:
            raise ValueError("Sparse input must use stateless/pre-normalized preprocessing (standardize=False).")

        t = perf_counter_ns()
        if is_sparse:
            x_scaled = x.tocsr().astype(np.float64, copy=False)
        else:
            x_dense = np.asarray(x, dtype=np.float64).reshape(-1)
            x_scaled = self.scaler.transform_then_update(x_dense) if self.scaler is not None else x_dense
        self.timings.scaling_ns += perf_counter_ns() - t

        t = perf_counter_ns()
        if sparse.issparse(x_scaled):
            z = np.asarray(x_scaled @ self.omega, dtype=np.float64).reshape(-1)
        else:
            z = np.asarray(x_scaled @ self.omega, dtype=np.float64).reshape(-1)
        self.timings.projection_ns += perf_counter_ns() - t

        t = perf_counter_ns()
        update = self.sketch.update(z)
        self.timings.sketch_ns += perf_counter_ns() - t

        t = perf_counter_ns()
        z_adapted = self.sketch.adapted(z) if self.config.use_adapted_representation_for_clustering else z
        self.timings.adaptation_ns += perf_counter_ns() - t

        sample_weight = update.effective_weight if self.config.leverage_mode == "weight" else 1.0
        radius_scale = (
            update.leverage_weight ** (-self.config.leverage_radius_strength)
            if self.config.leverage_mode == "weight" else 1.0
        )
        t = perf_counter_ns()
        cid = self.microclusters.update(
            z,
            z_adapted,
            self.clustering_basis,
            self.time,
            sample_weight=sample_weight,
            radius_scale=radius_scale,
        )
        self.timings.microcluster_ns += perf_counter_ns() - t

        self.rank_history.append(self.sketch.rank)
        self.error_history.append(update.error)
        self.threshold_history.append(update.threshold)
        self.leverage_history.append(update.leverage_score)
        self.weight_history.append(sample_weight)
        if update.updated:
            self.basis_update_trace.append(
                {
                    "time": self.time,
                    "rank": update.new_rank,
                    "old_rank": update.old_rank,
                    "rank_changed": update.rank_changed,
                    "candidate_rank": update.retained_energy_rank,
                    "reconstruction_error": update.error,
                    "threshold": update.threshold,
                    "sampling_probability": update.inclusion_probability,
                    "leverage_score": update.leverage_score,
                    "accepted": update.accepted,
                }
            )
        return ModelStep(cid, z, z_adapted, update)

    def process_batch(self, x) -> NDArray[np.float64]:
        n_rows = int(x.shape[0])
        projected = np.empty((n_rows, self.config.projection_dim), dtype=np.float64)
        if sparse.issparse(x):
            x = x.tocsr()
            for i in range(n_rows):
                projected[i] = self.process_one(x.getrow(i)).projected
        else:
            for i, row in enumerate(x):
                projected[i] = self.process_one(row).projected
        return projected

    def telemetry(self) -> dict[str, object]:
        """Return manuscript-facing diagnostics from this exact model instance.

        This telemetry is derived from this AdaptiveSketchStreamClusterer
        instance and covers its projection, sketch admission, rank control,
        and micro-clustering state.
        """
        ranks = np.asarray(self.rank_history, dtype=np.int64)
        if ranks.size:
            rank_mean = float(ranks.mean())
            rank_std = float(ranks.std(ddof=0))
            rank_min = int(ranks.min())
            rank_max = int(ranks.max())
            rank_changes = int(np.count_nonzero(np.diff(ranks)))
        else:
            rank_mean = rank_std = float("nan")
            rank_min = rank_max = self.sketch.rank
            rank_changes = 0
        return {
            "observations": int(self.time),
            "rank_mean": rank_mean,
            "rank_std_within_run": rank_std,
            "rank_min_observed": rank_min,
            "rank_max_observed": rank_max,
            "rank_change_count": rank_changes,
            "final_rank": int(self.sketch.rank),
            "sketch_seen": int(self.sketch.seen),
            "sketch_accepted": int(self.sketch.accepted),
            "sampling_acceptance": float(self.sketch.accepted / max(self.sketch.seen, 1)),
            "basis_updates": int(len(self.basis_update_trace)),
            "final_microclusters": int(len(self.microclusters.clusters)),
            "basis_update_interval_semantics": "accepted sketch rows",
            "microcluster_decay": float(self.config.decay),
        }

    def macro_centers(self, n_clusters: int) -> NDArray[np.float64]:
        from sklearn.cluster import KMeans

        centers, weights = self.microclusters.adapted_centers_and_weights(self.clustering_basis)
        if len(centers) == 0:
            raise RuntimeError("model has no microclusters")
        k = min(n_clusters, len(centers))
        if k == 1:
            return np.average(centers, axis=0, weights=weights, keepdims=True)
        km = KMeans(n_clusters=k, random_state=self.config.seed, n_init=10, algorithm="lloyd")
        km.fit(centers, sample_weight=weights)
        return km.cluster_centers_
