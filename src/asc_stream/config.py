from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(slots=True)
class ASCConfig:
    original_dim: int
    projection_dim: int = 32
    window_size: int = 1000
    initial_rank: int = 8
    min_rank: int = 4
    max_rank: int = 32
    microcluster_radius: float = 1.5
    # Exponential decay of micro-cluster weights. Kept as `decay` for
    # compatibility with the study implementation.
    decay: float = 0.95
    threshold_smoothing: float = 0.90
    leverage_regularization: float = 1e-3
    retained_energy: float = 0.95
    projection_sparsity: int = 3
    projection_mode: Literal["sparse", "identity", "dense"] = "sparse"
    max_microclusters: int = 200

    # Fixed operational defaults used by the reported implementation.
    basis_update_interval: int = 100
    rank_tolerance: float = 0.04
    stable_intervals_before_shrink: int = 4
    uniform_floor: float = 0.05
    leverage_mode: Literal["weight", "sample", "uniform", "off"] = "sample"
    leverage_sampling_rate: float = 0.65
    min_sampling_probability: float = 0.15
    max_sampling_probability: float = 1.0
    min_leverage_weight: float = 0.35
    max_leverage_weight: float = 3.0
    leverage_radius_strength: float = 0.25
    use_adapted_representation_for_clustering: bool = True
    prune_policy: Literal["utility", "merge"] = "utility"
    seed: int = 7
    # Keep the projection realization fixed across paired stochastic runs when desired.
    # None uses the model seed to select the projection.
    projection_seed: int | None = None

    def validate(self) -> None:
        if self.original_dim <= 0:
            raise ValueError("original_dim must be positive")
        if not (1 <= self.min_rank <= self.initial_rank <= self.max_rank):
            raise ValueError("Require min_rank <= initial_rank <= max_rank")
        if self.max_rank > self.projection_dim:
            raise ValueError("max_rank cannot exceed projection_dim")
        if self.projection_sparsity < 1:
            raise ValueError("projection_sparsity must be >= 1")
        if self.projection_mode == "identity" and self.projection_dim != self.original_dim:
            raise ValueError("identity projection requires projection_dim == original_dim")
        if self.window_size < self.max_rank:
            raise ValueError("window_size must be at least max_rank")
        if self.basis_update_interval < 1:
            raise ValueError("basis_update_interval must be positive")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError("decay must be in (0,1]")
        if not (0.0 <= self.threshold_smoothing < 1.0):
            raise ValueError("threshold_smoothing must be in [0,1)")
        if not (0.0 < self.retained_energy <= 1.0):
            raise ValueError("retained_energy must be in (0,1]")
        if not (0.0 < self.min_sampling_probability <= self.max_sampling_probability <= 1.0):
            raise ValueError("invalid sampling probability bounds")
        if not (0.0 < self.min_leverage_weight <= self.max_leverage_weight):
            raise ValueError("invalid leverage weight bounds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
