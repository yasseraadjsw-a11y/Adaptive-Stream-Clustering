from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ControlledRepresentationStream:
    """Fixed controlled representation-drift stream and offline reference labels."""

    x: NDArray[np.float64]
    y: NDArray[np.int64]
    change_points: tuple[int, ...]


def rebuild_controlled_representation_stream(
    n_samples: int = 9000,
    n_features: int = 256,
    n_clusters: int = 4,
    data_seed: int = 2026,
    separation: float = 3.6,
    informative_noise: float = 0.55,
    nuisance_scale: float = 1.35,
) -> ControlledRepresentationStream:
    """Rebuild the exact Gaussian-mixture representation-drift stream defined by the study protocol.

    The main experiments read the fixed NPZ file committed under ``data/controlled``.
    This function exists only to make the controlled protocol independently reproducible.
    The three equal phases use nuisance ranks 4, 10, and 16, with change points at
    3000 and 6000 for the default 9000-observation experiment. Reference labels are
    used only by offline evaluation and are never supplied to the online controller.
    """
    if n_clusters != 4:
        raise ValueError("the controlled simplex protocol requires four clusters")
    if n_features < 32:
        raise ValueError("n_features must be at least 32")

    rng = np.random.default_rng(data_seed)
    lengths = [n_samples // 3, n_samples // 3, n_samples - 2 * (n_samples // 3)]
    latent_dim = 24

    simplex = np.asarray(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
    ) / np.sqrt(3.0)
    means = separation * simplex

    orientations: list[NDArray[np.float64]] = []
    for _ in range(3):
        q, _ = np.linalg.qr(rng.normal(size=(n_features, latent_dim)))
        orientations.append(q)

    nuisance_ranks = [4, 10, 16]
    xs: list[NDArray[np.float64]] = []
    ys: list[NDArray[np.int64]] = []
    for phase, length in enumerate(lengths):
        probs = np.full(4, 0.25)
        if phase == 2:
            probs = np.asarray([0.30, 0.27, 0.25, 0.18])
        labels = rng.choice(4, size=length, p=probs)

        latent = np.zeros((length, latent_dim), dtype=np.float64)
        latent[:, :3] = means[labels] + rng.normal(
            scale=informative_noise, size=(length, 3)
        )

        rank = nuisance_ranks[phase]
        factors = rng.normal(scale=nuisance_scale, size=(length, rank))
        mixing = rng.normal(size=(rank, latent_dim - 3))
        mixing /= np.linalg.norm(mixing, axis=1, keepdims=True) + 1e-12
        latent[:, 3:] = factors @ mixing + rng.normal(
            scale=0.18, size=(length, latent_dim - 3)
        )
        if phase == 1:
            latent[:, 3:8] += 0.25 * means[labels, :1]
        elif phase == 2:
            latent[labels == 3, 18:22] += 0.55

        x_phase = latent @ orientations[phase].T + rng.normal(
            scale=0.05, size=(length, n_features)
        )
        xs.append(x_phase)
        ys.append(labels.astype(np.int64))

    return ControlledRepresentationStream(
        np.vstack(xs),
        np.concatenate(ys),
        (lengths[0], lengths[0] + lengths[1]),
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
