from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class OnlineStandardizer:
    """Numerically stable per-feature online standardization without label use."""

    def __init__(self, n_features: int, eps: float = 1e-8) -> None:
        self.n_features = n_features
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def transform_then_update(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n_features,):
            raise ValueError(f"expected shape {(self.n_features,)}, got {x.shape}")
        if self.count < 2:
            z = x - self.mean
        else:
            var = self.m2 / max(self.count - 1, 1)
            z = (x - self.mean) / np.sqrt(var + self.eps)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2
        return np.nan_to_num(z, copy=False)
