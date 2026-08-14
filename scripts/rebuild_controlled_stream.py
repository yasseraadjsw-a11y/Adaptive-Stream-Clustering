from __future__ import annotations

"""Optional integrity tool for rebuilding the fixed controlled stream from its declared protocol."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asc_stream.datasets import rebuild_controlled_representation_stream
from asc_stream.scaling import OnlineStandardizer

EXPECTED_RAW_SHA256 = "eda0e8ff77e480a9f6aa30fe252d8b9f1def8295fdcec8a55062195ed5be2f5f"
EXPECTED_STANDARDIZED_SHA256 = "89fe15a7aed55d5fbeda0d2b0838938ba6846e61c34c44fdb6900cfba577a646"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def standardize_causally(x: np.ndarray) -> np.ndarray:
    scaler = OnlineStandardizer(x.shape[1])
    z = np.empty_like(x, dtype=np.float64)
    for i, row in enumerate(x):
        z[i] = scaler.transform_then_update(row)
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "execution_runs" / "controlled_data")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stream = rebuild_controlled_representation_stream()
    raw_path = args.output_dir / "representation_drift_stream.npz"
    std_path = args.output_dir / "representation_drift_standardized.npz"
    np.savez_compressed(raw_path, x=stream.x, y=stream.y, drift_points=np.asarray(stream.change_points, dtype=np.int64))
    rebuilt_z = standardize_causally(stream.x)
    np.savez_compressed(std_path, z=rebuilt_z, y=stream.y)

    fixed_raw = np.load(ROOT / "data" / "controlled" / "representation_drift_stream.npz", allow_pickle=False)
    fixed_std = np.load(ROOT / "data" / "controlled" / "representation_drift_standardized.npz", allow_pickle=False)
    raw_numeric_equivalent = bool(
        np.allclose(stream.x, fixed_raw["x"], rtol=0.0, atol=2e-15)
        and np.array_equal(stream.y, fixed_raw["y"])
        and np.array_equal(np.asarray(stream.change_points, dtype=np.int64), fixed_raw["drift_points"])
    )
    standardized_numeric_equivalent = bool(
        np.allclose(rebuilt_z, fixed_std["z"], rtol=0.0, atol=2e-15)
        and np.array_equal(stream.y, fixed_std["y"])
    )

    result = {
        "raw_sha256": digest(raw_path),
        "standardized_sha256": digest(std_path),
        "raw_matches_fixed_file": digest(raw_path) == EXPECTED_RAW_SHA256,
        "standardized_matches_fixed_file": digest(std_path) == EXPECTED_STANDARDIZED_SHA256,
        "raw_numerically_equivalent": raw_numeric_equivalent,
        "standardized_numerically_equivalent": standardized_numeric_equivalent,
        "shape": list(stream.x.shape),
        "change_points": list(stream.change_points),
    }
    print(json.dumps(result, indent=2))
    if not (raw_numeric_equivalent and standardized_numeric_equivalent):
        raise SystemExit("Rebuilt controlled arrays are not numerically equivalent to the fixed study arrays.")


if __name__ == "__main__":
    main()
