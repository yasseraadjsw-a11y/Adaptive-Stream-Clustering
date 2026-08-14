from __future__ import annotations

"""Shared complete-window evaluation used by the modern-method extension."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(slots=True)
class WindowMetric:
    start: int
    end: int
    n: int
    ari: float
    nmi: float
    online_seconds: float
    macro_seconds: float


@dataclass(slots=True)
class StreamEvaluation:
    observations_total: int
    observations_evaluated: int
    coverage: float
    n_windows: int
    window_size: int
    warmup_windows: int
    ari_observation_weighted: float
    nmi_observation_weighted: float
    ari_window_mean: float
    nmi_window_mean: float
    ari_window_sd: float
    nmi_window_sd: float
    ari_min_window: float
    ari_max_window: float
    nmi_min_window: float
    nmi_max_window: float
    online_seconds: float
    macro_seconds: float
    runtime_seconds: float
    windows: list[WindowMetric]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def nearest_assign(x: np.ndarray, centers: np.ndarray, block: int = 4096) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) == 0:
        return np.zeros(len(x), dtype=np.int64)
    out = np.empty(len(x), dtype=np.int64)
    c2 = np.sum(centers * centers, axis=1)[None, :]
    for start in range(0, len(x), block):
        xb = x[start:start + block]
        x2 = np.sum(xb * xb, axis=1)[:, None]
        d2 = np.maximum(x2 + c2 - 2.0 * xb @ centers.T, 0.0)
        out[start:start + len(xb)] = np.argmin(d2, axis=1)
    return out


def _slice_rows(x, start: int, end: int):
    return x[start:end] if sparse.issparse(x) else np.asarray(x[start:end])


def evaluate_stream(
    model,
    x,
    y: np.ndarray,
    n_clusters: int,
    *,
    window_size: int,
    warmup_windows: int = 0,
    preprocess_block: Callable[[Any], Any] | None = None,
) -> StreamEvaluation:
    """Evaluate every declared window without silently dropping observations."""
    y = np.asarray(y, dtype=np.int64)
    if int(x.shape[0]) != len(y):
        raise ValueError("feature/label row mismatch")
    if window_size < 1 or warmup_windows < 0:
        raise ValueError("invalid window or warm-up count")

    rows: list[WindowMetric] = []
    online_total = macro_total = 0.0
    n = len(y)
    for widx, start in enumerate(range(0, n, window_size)):
        end = min(start + window_size, n)
        block = _slice_rows(x, start, end)
        if preprocess_block is not None:
            block = preprocess_block(block)
        t0 = perf_counter()
        z = model.process_batch(block)
        online = perf_counter() - t0
        online_total += online
        if widx < warmup_windows:
            continue
        t1 = perf_counter()
        pred = nearest_assign(z, model.macro_centers(int(n_clusters)))
        macro = perf_counter() - t1
        macro_total += macro
        rows.append(WindowMetric(
            start=start,
            end=end,
            n=end - start,
            ari=float(adjusted_rand_score(y[start:end], pred)),
            nmi=float(normalized_mutual_info_score(y[start:end], pred)),
            online_seconds=float(online),
            macro_seconds=float(macro),
        ))

    if not rows:
        raise RuntimeError("No evaluated windows")
    counts = np.asarray([r.n for r in rows], dtype=np.float64)
    aris = np.asarray([r.ari for r in rows], dtype=np.float64)
    nmis = np.asarray([r.nmi for r in rows], dtype=np.float64)
    evaluated = int(counts.sum())
    return StreamEvaluation(
        observations_total=n,
        observations_evaluated=evaluated,
        coverage=float(evaluated / max(n, 1)),
        n_windows=len(rows),
        window_size=int(window_size),
        warmup_windows=int(warmup_windows),
        ari_observation_weighted=float(np.average(aris, weights=counts)),
        nmi_observation_weighted=float(np.average(nmis, weights=counts)),
        ari_window_mean=float(aris.mean()),
        nmi_window_mean=float(nmis.mean()),
        ari_window_sd=float(aris.std(ddof=1)) if len(aris) > 1 else 0.0,
        nmi_window_sd=float(nmis.std(ddof=1)) if len(nmis) > 1 else 0.0,
        ari_min_window=float(aris.min()),
        ari_max_window=float(aris.max()),
        nmi_min_window=float(nmis.min()),
        nmi_max_window=float(nmis.max()),
        online_seconds=float(online_total),
        macro_seconds=float(macro_total),
        runtime_seconds=float(online_total + macro_total),
        windows=rows,
    )
