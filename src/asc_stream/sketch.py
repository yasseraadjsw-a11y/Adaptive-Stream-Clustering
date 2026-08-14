from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BasisUpdate:
    updated: bool
    rank_changed: bool
    old_rank: int
    new_rank: int
    error: float
    threshold: float
    retained_energy_rank: int
    accepted: bool
    inclusion_probability: float
    leverage_score: float
    leverage_weight: float
    effective_weight: float


class AdaptiveProjectedSketch:
    """Bounded sketch in projected space.

    The canonical manuscript path uses leverage-based **sketch admission**.
    Equation (7) defines a normalized leverage distribution; the operational
    Bernoulli inclusion probability applies the declared sketch budget and
    probability bounds. Every observation still participates in online
    micro-clustering/evaluation; only sketch admission is sampled. The
    alternative weighting and uniform modes are retained only for ablation.
    """

    def __init__(
        self,
        projection_dim: int,
        window_size: int,
        initial_rank: int,
        min_rank: int,
        max_rank: int,
        smoothing: float,
        retained_energy: float,
        ridge_lambda: float,
        update_interval: int,
        tolerance: float,
        stable_before_shrink: int,
        uniform_floor: float,
        leverage_mode: str,
        sampling_rate: float,
        min_sampling_probability: float,
        max_sampling_probability: float,
        min_leverage_weight: float,
        max_leverage_weight: float,
        rng: np.random.Generator,
    ) -> None:
        self.k = projection_dim
        self.window_size = window_size
        self.rank = initial_rank
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.smoothing = smoothing
        self.retained_energy = retained_energy
        self.ridge_lambda = ridge_lambda
        self.update_interval = update_interval
        self.tolerance = tolerance
        self.stable_before_shrink = stable_before_shrink
        self.uniform_floor = uniform_floor
        self.leverage_mode = leverage_mode
        self.sampling_rate = sampling_rate
        self.min_sampling_probability = min_sampling_probability
        self.max_sampling_probability = max_sampling_probability
        self.min_leverage_weight = min_leverage_weight
        self.max_leverage_weight = max_leverage_weight
        self.rng = rng

        self.rows: deque[NDArray[np.float64]] = deque(maxlen=window_size)
        self.weights: deque[float] = deque(maxlen=window_size)
        # Exact sufficient statistics for the same uncentered weighted
        # covariance used by _weighted_covariance.  Maintaining the rolling
        # numerator avoids rebuilding W^T D W from the full accepted-row
        # window at every basis refresh.
        self._cov_numerator = np.zeros((self.k, self.k), dtype=np.float64)
        self._cov_weight_sum = 0.0
        self.basis = np.eye(self.k, self.rank, dtype=np.float64)
        self.spectral_values = np.ones(self.rank, dtype=np.float64)
        self.threshold: float | None = None
        self.stable_count = 0
        self.seen = 0
        self.accepted = 0
        # Manuscript Eq. (7): keep the current bounded normalization set and
        # its leverage sum so the exact normalization is O(1) per observation.
        self.leverage_scores: deque[float] = deque()
        self.leverage_sum = 0.0
        self.last_error = 0.0
        self.last_candidate_rank = initial_rank

    def _ridge_leverage(self, z: NDArray[np.float64]) -> float:
        h = self.basis[:, : self.rank].T @ z
        denom = np.maximum(self.spectral_values[: self.rank], 0.0) + self.ridge_lambda
        return float(np.sum((h * h) / denom))

    def _equation7_probability(self, leverage: float) -> tuple[float, int]:
        """Normalized leverage probability in manuscript Eq. (7).

        q_i = (l_i + epsilon/n) / sum_j(l_j + epsilon/n)
            = (l_i + epsilon/n) / (sum_j l_j + epsilon).
        The arriving observation is included in the normalization set.
        """
        leverage = max(float(leverage), 0.0)
        n = len(self.leverage_scores) + 1
        denominator = self.leverage_sum + leverage + self.uniform_floor
        if denominator <= 1e-15:
            return 1.0 / n, n
        q = (leverage + self.uniform_floor / n) / denominator
        return float(np.clip(q, 0.0, 1.0)), n

    def _sampling_probability(self, normalized_probability: float, n: int) -> float:
        # Operational Bernoulli budget corresponding to the declared nominal
        # sketch-admission rate rho; Eq. (7) itself remains the normalized q_i.
        return float(np.clip(
            self.sampling_rate * n * normalized_probability,
            self.min_sampling_probability,
            self.max_sampling_probability,
        ))

    def _record_leverage(self, leverage: float) -> None:
        leverage = max(float(leverage), 0.0)
        self.leverage_scores.append(leverage)
        self.leverage_sum += leverage
        if len(self.leverage_scores) > self.window_size:
            self.leverage_sum -= self.leverage_scores.popleft()

    def _leverage_weight(self, influence: float) -> float:
        # Bounded saturating transform. A direct inverse-probability weight can
        # explode during initialization and over-emphasize transient noise.
        # This mapping is monotone in ridge leverage but stays near unit scale.
        weight = 0.5 + max(influence, 0.0) / (1.0 + max(influence, 0.0))
        return float(np.clip(weight, self.min_leverage_weight, self.max_leverage_weight))

    def _energy_rank(self, eigenvalues: NDArray[np.float64]) -> int:
        usable = np.maximum(eigenvalues[: self.max_rank], 0.0)
        total = float(np.sum(usable))
        if total <= 1e-15:
            return self.min_rank
        cumulative = np.cumsum(usable) / total
        rank = int(np.searchsorted(cumulative, self.retained_energy) + 1)
        return int(np.clip(rank, self.min_rank, self.max_rank))

    @staticmethod
    def _weighted_covariance(
        matrix: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        w = np.maximum(weights, 1e-12)
        # The data are already causally standardized; do not recenter the
        # window again because that would use future observations within it.
        return (matrix.T * w) @ matrix / max(float(np.sum(w)), 1e-12)

    @staticmethod
    def _reconstruction_error(
        matrix: NDArray[np.float64], basis: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> float:
        reconstructed = (matrix @ basis) @ basis.T
        residual = matrix - reconstructed
        w = np.maximum(weights, 1e-12)
        numerator = np.sqrt(np.sum(w[:, None] * residual * residual))
        denominator = np.sqrt(np.sum(w[:, None] * matrix * matrix)) + 1e-12
        return float(numerator / denominator)


    def update(self, z: NDArray[np.float64]) -> BasisUpdate:
        self.seen += 1
        leverage = self._ridge_leverage(z)
        normalized_probability, normalization_n = self._equation7_probability(leverage)
        influence = normalization_n * normalized_probability
        p = self._sampling_probability(normalized_probability, normalization_n)
        leverage_weight = 1.0 if self.threshold is None else self._leverage_weight(influence)
        self._record_leverage(leverage)

        if self.leverage_mode == "sample":
            accepted = bool(self.rng.random() <= p)
            point_weight = (1.0 / max(p, 1e-12)) if accepted else 0.0
        elif self.leverage_mode == "uniform":
            # Fair no-leverage ablation: preserve the same expected sketch
            # budget but sample independently of the leverage score.
            p = float(np.clip(self.sampling_rate, self.min_sampling_probability, self.max_sampling_probability))
            accepted = bool(self.rng.random() <= p)
            leverage_weight = 1.0
            point_weight = (1.0 / max(p, 1e-12)) if accepted else 0.0
        elif self.leverage_mode == "weight":
            accepted = True
            p = 1.0
            point_weight = leverage_weight
        elif self.leverage_mode == "off":
            accepted = True
            p = 1.0
            leverage_weight = 1.0
            point_weight = 1.0
        else:
            raise ValueError(f"unknown leverage_mode={self.leverage_mode!r}")

        if accepted:
            # deque(maxlen=...) silently discards its oldest element.  Capture
            # it first so the rolling weighted covariance remains exactly on
            # the same accepted-row window.
            if len(self.rows) == self.window_size:
                old_z = self.rows[0]
                old_w = float(self.weights[0])
                self._cov_numerator -= old_w * np.outer(old_z, old_z)
                self._cov_weight_sum -= old_w
            z_copy = z.copy()
            w_copy = float(point_weight)
            self.rows.append(z_copy)
            self.weights.append(w_copy)
            self._cov_numerator += w_copy * np.outer(z_copy, z_copy)
            self._cov_weight_sum += w_copy
            self.accepted += 1

        old_rank = self.rank
        base = BasisUpdate(
            False,
            False,
            old_rank,
            self.rank,
            self.last_error,
            self.threshold or 0.0,
            self.last_candidate_rank,
            accepted,
            p,
            leverage,
            leverage_weight,
            float(point_weight),
        )
        if (not accepted) or len(self.rows) < max(self.max_rank + 2, 64):
            return base
        if self.accepted % self.update_interval != 0:
            return base

        # Same covariance as _weighted_covariance(rows, weights), evaluated
        # from rolling sufficient statistics.  Symmetrization removes only
        # round-off asymmetry from repeated rank-one add/subtract operations.
        covariance = self._cov_numerator / max(float(self._cov_weight_sum), 1e-12)
        covariance = 0.5 * (covariance + covariance.T)
        # k is bounded (32 for numeric streams, 256 for TweetEval). eigh on
        # the k×k covariance is much faster than repeatedly decomposing W×k.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        candidate_rank = self._energy_rank(eigenvalues)
        self.last_candidate_rank = candidate_rank

        old_basis = self.basis[:, : self.rank]
        # For C = sum_i w_i z_i z_i^T / sum_i w_i, the weighted relative
        # Frobenius reconstruction error satisfies
        #   err^2 = 1 - tr(B^T C B) / tr(C).
        # This is algebraically identical to _reconstruction_error on the
        # accepted window and avoids materializing two W x k matrices.
        total_energy = max(float(np.trace(covariance)), 0.0)
        if total_energy <= 1e-24:
            error = 0.0
        else:
            retained = float(np.trace(old_basis.T @ covariance @ old_basis))
            error = float(np.sqrt(max(total_energy - retained, 0.0) / total_energy))
        previous_threshold = self.threshold
        trigger = previous_threshold is not None and error > previous_threshold * (1.0 + self.tolerance)

        if previous_threshold is None:
            # Keep the declared initial rank until evidence requests change.
            self.stable_count = 0
        elif trigger:
            self.rank = max(self.rank, candidate_rank)
            self.stable_count = 0
        else:
            self.stable_count += 1
            if self.stable_count >= self.stable_before_shrink:
                self.rank = min(self.rank, candidate_rank)
                self.stable_count = 0

        self.rank = int(np.clip(self.rank, self.min_rank, self.max_rank))
        self.basis = eigenvectors[:, : self.rank].copy()
        self.spectral_values = eigenvalues[: self.rank].copy()
        self.threshold = (
            error
            if previous_threshold is None
            else self.smoothing * previous_threshold + (1.0 - self.smoothing) * error
        )
        self.last_error = error
        return BasisUpdate(
            True,
            self.rank != old_rank,
            old_rank,
            self.rank,
            error,
            float(self.threshold),
            candidate_rank,
            accepted,
            p,
            leverage,
            leverage_weight,
            float(point_weight),
        )

    def adapted(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        basis = self.basis[:, : self.rank]
        return (z @ basis) @ basis.T

    def adapt_matrix(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        basis = self.basis[:, : self.rank]
        return (z @ basis) @ basis.T
