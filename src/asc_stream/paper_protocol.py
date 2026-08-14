from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json

import numpy as np
from scipy import sparse

from .config import ASCConfig
from .datasets import load_prepared_dataset
from .datasets.provenance import validate_tweeteval_prepared

DatasetId = Literal["covertype", "electricity", "tweeteval", "synthetic_gmm"]

SEEDS = [7, 13, 19, 23, 31, 37, 41, 43, 47, 53]
DISPLAY = {"covertype":"CoverType","electricity":"Electricity","tweeteval":"TweetEval","synthetic_gmm":"Synthetic GMM"}
EXPECTED_ROWS = {"covertype":581_012,"electricity":45_312,"tweeteval":59_899,"synthetic_gmm":9_000}
EXPECTED_DIMS = {"covertype":54,"electricity":8,"tweeteval":2048,"synthetic_gmm":256}
N_CLUSTERS = {"covertype":7,"electricity":2,"tweeteval":3,"synthetic_gmm":4}

@dataclass(frozen=True)
class LoadedPaperDataset:
    x: object
    y: np.ndarray
    dataset: str
    source: str
    canonical: bool
    note: str

def _dataset_cfg_path(root: Path, dataset: str) -> Path:
    return root / "configs" / "datasets" / f"{dataset}.json"

def load_paper_config(root: Path, dataset: str) -> dict:
    """Load the one authoritative study configuration for a primary dataset.

    A compatibility-shaped dictionary is returned so experiment code has one
    source of truth while preserving its existing ``proposed``/``baselines``
    access pattern.
    """
    path=_dataset_cfg_path(root,dataset)
    if not path.exists(): raise FileNotFoundError(f"Missing study config: {path}")
    raw=json.loads(path.read_text(encoding="utf-8"))
    proposed_keys=("projection_dim","window_size","initial_rank","min_rank","max_rank",
        "microcluster_radius","decay","threshold_smoothing","leverage_regularization",
        "retained_energy","projection_sparsity","max_microclusters")
    proposed={k:raw[k] for k in proposed_keys}
    proposed.update({
        "basis_update_interval":100,"rank_tolerance":0.04,"stable_intervals_before_shrink":4,
        "uniform_floor":0.05,"leverage_mode":"sample","leverage_sampling_rate":0.65,
        "min_sampling_probability":0.15,"max_sampling_probability":1.0,
        "min_leverage_weight":0.35,"max_leverage_weight":3.0,
        "leverage_radius_strength":0.0,"use_adapted_representation_for_clustering":True,
        "prune_policy":"utility",
    })
    return {"dataset":dataset,"display":DISPLAY[dataset],"n_clusters":raw["n_clusters"],
            "proposed":proposed,"baselines":raw["baseline_settings"],"raw":raw}

def asc_config_from_paper(root: Path, dataset: str, original_dim: int, seed: int, *, fixed_rank: bool=False, projection_mode: str|None=None) -> ASCConfig:
    values=dict(load_paper_config(root,dataset)["proposed"])
    values.update(original_dim=int(original_dim),seed=int(seed),projection_seed=int(seed))
    if projection_mode is not None: values["projection_mode"]=projection_mode
    if fixed_rank: values.update(initial_rank=8,min_rank=8,max_rank=8)
    return ASCConfig(**values)

def recent_method_settings(root: Path, dataset: str, method: str) -> dict:
    """Return controlled-comparison settings from its single protocol file."""
    cfg=json.loads((root/'configs'/'controlled_representation_drift.json').read_text(encoding='utf-8'))
    key='fra_art' if method in {'fra_art','fraart'} else method
    raw=cfg['recent_methods'][key]
    if key=='twstream':
        return {"radius":float(raw['radius_value']),"k":int(raw['k']),"lambda":float(raw['lambda']),"tau":0.65,
                "structural_max_clusters":200,"structural_max_outliers":200}
    return {"fractional_order":float(raw['fractional_order']),"vigilance":float(raw['vigilance']),
            "choice":float(raw['choice']),"beta":float(raw['learning_rate'])}

def build_recent_comparator(root: Path, dataset: str, method: str, d: int, seed: int, n: int):
    from .comparators import TWStreamComparator, FRAARTComparator
    settings=recent_method_settings(root,dataset,method); key='fra_art' if method in {'fra_art','fraart'} else method
    if key=='twstream':
        return TWStreamComparator(int(d),int(seed),max_clusters=int(settings['structural_max_clusters']),
            max_outliers=int(settings['structural_max_outliers']),radius=float(settings['radius']),k=int(settings['k']),lam=float(settings['lambda']))
    return FRAARTComparator(int(d),int(seed),a=float(settings['fractional_order']),vigilance=float(settings['vigilance']),
        choice=float(settings['choice']),beta=float(settings['beta']),max_stream_points=max(int(n),9000))

def load_full_paper_dataset(root: Path, dataset: DatasetId, *, allow_packaged_tweeteval: bool = False) -> LoadedPaperDataset:
    """Load a full study stream while making the evidence boundary explicit.

    CoverType/Electricity/Synthetic GMM are available as full bundled arrays in
    the practical package. Canonical TweetEval requires the 2,048-dimensional
    CSR matrix prepared from the official train/validation/test text files.
    The bundled 256-dimensional TweetEval array is accepted only when the
    caller explicitly requests a packaged projected study representation.
    """
    processed = root / "data" / "public" / "processed"
    if dataset in {"covertype", "electricity", "tweeteval"}:
        try:
            x, y = load_prepared_dataset(processed, dataset)
            if dataset == "tweeteval":
                status = validate_tweeteval_prepared(root)
                if not status.canonical:
                    raise FileNotFoundError("Prepared TweetEval files exist but fail the pinned canonical provenance checks")
                canonical = True; source = status.source; note = status.note
            else:
                canonical = True
                source = str(processed)
                note = "prepared from the declared public-data preprocessing pipeline"
            _assert_shape(dataset, x, y, canonical=True)
            return LoadedPaperDataset(x, y, dataset, source, canonical, note)
        except FileNotFoundError:
            pass

    if dataset == "covertype":
        path = root / "data" / "rank_validation" / "processed" / "covertype.npz"
        with np.load(path, allow_pickle=False) as d:
            x = np.asarray(d["x"], dtype=np.float64)
            y = np.asarray(d["y"], dtype=np.int64)
        _assert_shape(dataset, x, y, canonical=True)
        return LoadedPaperDataset(x, y, dataset, str(path), True, "full bundled UCI-order numeric stream")

    if dataset == "electricity":
        path = root / "data" / "rank_validation" / "processed" / "electricity.npz"
        with np.load(path, allow_pickle=False) as d:
            x = np.asarray(d["x"], dtype=np.float64)
            y = np.asarray(d["y"], dtype=np.int64)
        _assert_shape(dataset, x, y, canonical=True)
        return LoadedPaperDataset(x, y, dataset, str(path), True, "full bundled Electricity stream")

    if dataset == "synthetic_gmm":
        from .datasets.provenance import dataset_status
        status=dataset_status(root,dataset)
        if status.canonical:
            path=Path(status.source)
            with np.load(path,allow_pickle=False) as d:
                x=np.asarray(d["x"],dtype=np.float64);y=np.asarray(d["y"],dtype=np.int64)
            _assert_shape(dataset,x,y,canonical=True)
            return LoadedPaperDataset(x,y,dataset,str(path),True,status.note)
        path = root / "data" / "rank_validation" / "processed" / "synthetic_gmm.npz"
        with np.load(path, allow_pickle=False) as d:
            x = np.asarray(d["x"], dtype=np.float64)
            y = np.asarray(d["y"], dtype=np.int64)
        _assert_shape(dataset, x, y, canonical=True)
        return LoadedPaperDataset(
            x, y, dataset, str(path), False,
            "full bundled 9,000 x 256 Synthetic GMM stream used by the original benchmark and reused by the reviewer-requested controlled time-resolved analysis",
        )

    if dataset == "tweeteval" and allow_packaged_tweeteval:
        path = root / "data" / "rank_validation" / "processed" / "tweeteval_sentiment.npz"
        with np.load(path, allow_pickle=False) as d:
            x = np.asarray(d["x"], dtype=np.float64)
            y = np.asarray(d["y"], dtype=np.int64)
        if x.shape != (EXPECTED_ROWS[dataset], 256):
            raise ValueError(f"Unexpected bundled TweetEval audit shape: {x.shape}")
        return LoadedPaperDataset(
            x, y, dataset, str(path), False,
            "packaged 256-D TweetEval study representation; public-source reconstruction is supported separately from official text through the declared 2048-D HashingVectorizer pipeline",
        )

    raise FileNotFoundError(
        "Canonical TweetEval features are not bundled. Run `python main.py setup-data --dataset tweeteval` "
        "with the official TweetEval files available, or use the packaged study representation."
    )



def load_packaged_study_dataset(root: Path, dataset: DatasetId) -> LoadedPaperDataset:
    """Load the self-contained packaged study asset for a declared dataset.

    This path is intended for out-of-the-box execution in the release archive.
    It verifies the packaged asset fingerprint before loading. TweetEval uses
    the packaged 256-D study representation; canonical public-source TweetEval
    remains available through ``load_full_paper_dataset`` after ``setup-data``.
    """
    from .datasets.provenance import dataset_status, validate_study_asset
    if dataset == "covertype":
        status = dataset_status(root, dataset)
        path = root / "data" / "rank_validation" / "processed" / "covertype.npz"
    elif dataset == "electricity":
        status = dataset_status(root, dataset)
        path = root / "data" / "rank_validation" / "processed" / "electricity.npz"
    elif dataset == "synthetic_gmm":
        status = dataset_status(root, dataset)
        path = root / "data" / "rank_validation" / "processed" / "synthetic_gmm.npz"
    elif dataset == "tweeteval":
        status = validate_study_asset(root, "tweeteval_sentiment_projected", (EXPECTED_ROWS[dataset], 256))
        path = root / "data" / "rank_validation" / "processed" / "tweeteval_sentiment.npz"
    else:
        raise ValueError(dataset)
    if not status.study_asset_verified:
        raise RuntimeError(f"{dataset}: packaged study asset failed integrity verification")
    with np.load(path, allow_pickle=False) as d:
        x = np.asarray(d["x"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.int64)
    expected_dim = 256 if dataset == "tweeteval" else EXPECTED_DIMS[dataset]
    if x.shape != (EXPECTED_ROWS[dataset], expected_dim) or len(y) != EXPECTED_ROWS[dataset]:
        raise ValueError(f"{dataset}: unexpected packaged study shape X={x.shape}, y={len(y)}")
    return LoadedPaperDataset(
        x=x, y=y, dataset=dataset, source=str(path), canonical=bool(status.canonical),
        note=("packaged study asset verified by SHA-256" if dataset != "tweeteval" else
              "packaged TweetEval study representation verified by SHA-256; canonical 2048-D public-source preparation is supported separately"),
    )

def _assert_shape(dataset: str, x, y: np.ndarray, *, canonical: bool) -> None:
    if int(x.shape[0]) != EXPECTED_ROWS[dataset] or int(y.shape[0]) != EXPECTED_ROWS[dataset]:
        raise ValueError(f"{dataset}: expected {EXPECTED_ROWS[dataset]} rows, got X={x.shape[0]}, y={y.shape[0]}")
    if canonical and int(x.shape[1]) != EXPECTED_DIMS[dataset]:
        raise ValueError(f"{dataset}: expected {EXPECTED_DIMS[dataset]} features, got {x.shape[1]}")
