from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

PreparedMatrix = Union[NDArray[np.float64], sparse.csr_matrix]


def load_prepared_dataset(processed_root: Path, name: str) -> tuple[PreparedMatrix, NDArray[np.int64]]:
    """Load a prepared public dataset without changing its declared stream order.

    Numeric datasets are stored as compressed NumPy arrays. TweetEval keeps its
    hashed text matrix in CSR form so the full 2,048-dimensional stream is never
    materialized as one dense matrix.
    """
    meta_path = processed_root / f"{name}.npz"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {meta_path}. Run `python main.py setup-data --dataset {name}` first."
        )
    meta = np.load(meta_path, allow_pickle=False)
    y = np.asarray(meta["y"], dtype=np.int64)
    if name == "tweeteval":
        feature_path = processed_root / "tweeteval_features.npz"
        if not feature_path.exists():
            raise FileNotFoundError(
                f"TweetEval sparse features not found: {feature_path}. Re-run data preparation."
            )
        x = sparse.load_npz(feature_path).tocsr().astype(np.float64, copy=False)
        if x.shape[0] != y.shape[0]:
            raise ValueError("TweetEval feature/label row mismatch")
        return x, y
    x = np.asarray(meta["x"], dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{name} feature/label row mismatch")
    return x, y
