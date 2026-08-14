from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_sparse_achlioptas_projection(
    original_dim: int,
    projection_dim: int,
    sparsity: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Create Ω following Eq. (8): ±sqrt(s/k) with probability 1/(2s)."""
    if original_dim <= 0 or projection_dim <= 0 or sparsity <= 0:
        raise ValueError("dimensions and sparsity must be positive")
    u = rng.random((original_dim, projection_dim))
    omega = np.zeros((original_dim, projection_dim), dtype=np.float64)
    scale = np.sqrt(sparsity / projection_dim)
    lower = 1.0 / (2.0 * sparsity)
    upper = 1.0 - lower
    omega[u < lower] = scale
    omega[u > upper] = -scale
    return omega
