from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from typing import Any

from scipy import sparse
import numpy as np

TWEETEVAL_REPOSITORY = "cardiffnlp/tweeteval"
TWEETEVAL_COMMIT = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
TWEETEVAL_SENTIMENT_SPLITS = {"train": 45615, "val": 2000, "test": 12284}
TWEETEVAL_TOTAL_ROWS = sum(TWEETEVAL_SENTIMENT_SPLITS.values())
TWEETEVAL_VECTORIZER = {
    "method": "HashingVectorizer",
    "n_features": 2048,
    "alternate_sign": False,
    "norm": "l2",
    "lowercase": True,
    "ngram_range": [1, 2],
    "dtype": "float32",
}


@dataclass(frozen=True)
class ProvenanceStatus:
    dataset: str
    canonical: bool
    source: str
    checks: dict[str, bool]
    note: str
    study_asset_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _study_asset_entry(root: Path, key: str) -> dict:
    m = _read_json(root / "data" / "study_assets_manifest.json")
    assets = m.get("assets", {}) if isinstance(m, dict) else {}
    entry = assets.get(key, {}) if isinstance(assets, dict) else {}
    return entry if isinstance(entry, dict) else {}


def validate_study_asset(root: Path, key: str, expected_shape: tuple[int, int]) -> ProvenanceStatus:
    entry = _study_asset_entry(root, key)
    rel = entry.get("path")
    path = root / rel if rel else root / "data" / "rank_validation" / "processed" / f"{key}.npz"
    checks: dict[str, bool] = {
        "manifest_entry_exists": bool(entry),
        "file_exists": path.exists(),
    }
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as d:
                x = d["x"]
                y = d["y"]
                checks["shape_exact"] = tuple(x.shape) == tuple(expected_shape)
                checks["row_match"] = int(len(y)) == int(expected_shape[0])
            checks["size_matches_manifest"] = bool(entry) and int(entry.get("size_bytes", -1)) == path.stat().st_size
            checks["hash_matches_manifest"] = bool(entry.get("sha256")) and entry.get("sha256") == sha256_file(path)
            checks["manifest_shape_matches"] = entry.get("expected_x_shape") == list(expected_shape)
        except Exception:
            checks.update({
                "shape_exact": False,
                "row_match": False,
                "size_matches_manifest": False,
                "hash_matches_manifest": False,
                "manifest_shape_matches": False,
            })
    else:
        checks.update({
            "shape_exact": False,
            "row_match": False,
            "size_matches_manifest": False,
            "hash_matches_manifest": False,
            "manifest_shape_matches": False,
        })
    verified = all(checks.values())
    note = (
        "Packaged study asset verified by shape, size and SHA-256."
        if verified else
        "Packaged study asset is missing or its fingerprint does not match data/study_assets_manifest.json."
    )
    return ProvenanceStatus(key, False, str(path), checks, note, study_asset_verified=verified)


def validate_tweeteval_prepared(root: Path) -> ProvenanceStatus:
    processed = root / "data" / "public" / "processed"
    meta_path = processed / "tweeteval.npz"
    feature_path = processed / "tweeteval_features.npz"
    prep_manifest = _read_json(processed / "preprocessing_manifest.json")
    entry = prep_manifest.get("tweeteval", {}) if isinstance(prep_manifest, dict) else {}

    checks: dict[str, bool] = {
        "metadata_exists": meta_path.exists(),
        "features_exist": feature_path.exists(),
        "manifest_entry_exists": bool(entry),
        "source_revision_pinned": entry.get("source_revision") == TWEETEVAL_COMMIT,
        "vectorizer_exact": entry.get("vectorizer") == TWEETEVAL_VECTORIZER,
        "split_sizes_exact": entry.get("split_sizes") == TWEETEVAL_SENTIMENT_SPLITS,
    }
    if meta_path.exists() and feature_path.exists():
        try:
            meta = np.load(meta_path, allow_pickle=False)
            y = np.asarray(meta["y"], dtype=np.int64)
            x = sparse.load_npz(feature_path)
            checks["row_count_exact"] = len(y) == TWEETEVAL_TOTAL_ROWS and x.shape[0] == TWEETEVAL_TOTAL_ROWS
            checks["feature_dim_exact"] = x.shape[1] == TWEETEVAL_VECTORIZER["n_features"]
            checks["label_domain_exact"] = set(np.unique(y).tolist()) <= {0, 1, 2} and len(np.unique(y)) == 3
            recorded = entry.get("feature_sha256")
            checks["feature_hash_matches_manifest"] = bool(recorded) and recorded == sha256_file(feature_path)
        except Exception:
            checks["row_count_exact"] = False
            checks["feature_dim_exact"] = False
            checks["label_domain_exact"] = False
            checks["feature_hash_matches_manifest"] = False
    else:
        checks["row_count_exact"] = False
        checks["feature_dim_exact"] = False
        checks["label_domain_exact"] = False
        checks["feature_hash_matches_manifest"] = False

    canonical = all(checks.values())
    note = (
        "Official TweetEval sentiment text pinned to commit " + TWEETEVAL_COMMIT +
        ", train->validation->test order, stateless 2048-D HashingVectorizer."
        if canonical else
        "Canonical TweetEval public-source features are not prepared locally. Run `python main.py setup-data --dataset tweeteval` when network access is available."
    )
    study = validate_study_asset(root, "tweeteval_sentiment_projected", (TWEETEVAL_TOTAL_ROWS, 256))
    return ProvenanceStatus("tweeteval", canonical, str(processed), checks, note, study_asset_verified=study.study_asset_verified)


def validate_numeric_bundled(root: Path, dataset: str, shape: tuple[int, int]) -> ProvenanceStatus:
    status = validate_study_asset(root, dataset, shape)
    # CoverType and Electricity are the packaged full numeric study streams.
    # Canonical public-source preparation can additionally be recreated through
    # scripts/acquire_datasets.py + scripts/prepare_datasets.py.
    canonical = status.study_asset_verified and dataset in {"covertype", "electricity"}
    note = (
        "Packaged full numeric study stream verified by SHA-256; public acquisition/preparation scripts are included."
        if status.study_asset_verified else status.note
    )
    return ProvenanceStatus(dataset, canonical, status.source, status.checks, note, status.study_asset_verified)


def dataset_status(root: Path, dataset: str) -> ProvenanceStatus:
    if dataset == "tweeteval":
        return validate_tweeteval_prepared(root)
    if dataset == "covertype":
        return validate_numeric_bundled(root, dataset, (581012, 54))
    if dataset == "electricity":
        return validate_numeric_bundled(root, dataset, (45312, 8))
    if dataset == "synthetic_gmm":
        study = validate_study_asset(root, dataset, (9000, 256))
        return ProvenanceStatus(
            dataset, bool(study.study_asset_verified), study.source, study.checks,
            "Exact realized Synthetic GMM study stream verified by shape, size and SHA-256; the same fixed stream is used by the controlled time-resolved analysis.",
            study_asset_verified=study.study_asset_verified,
        )
    raise ValueError(dataset)
