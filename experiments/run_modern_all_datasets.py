from __future__ import annotations

"""Legacy modern-only runner retained for focused diagnostics.

The manuscript-facing all-seven-method re-execution path is run_primary.py.
This diagnostic runner never reads or combines protected manuscript results.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asc_stream.comparators import FRAARTComparator, TWStreamComparator
from asc_stream.evaluation import evaluate_stream, file_sha256
from asc_stream.method_registry import method_metadata
from asc_stream.release_io import ensure_execution_output
from asc_stream.scaling import OnlineStandardizer

METHODS = ("twstream", "fra_art")
DATASETS = ("covertype", "electricity", "tweeteval", "synthetic_gmm")
DISPLAY = {"covertype": "CoverType", "electricity": "Electricity", "tweeteval": "TweetEval", "synthetic_gmm": "Synthetic GMM"}


class CausalStandardizer:
    def __init__(self, d: int, enabled: bool):
        self.scaler = OnlineStandardizer(d) if enabled else None

    def __call__(self, block):
        x = np.asarray(block, dtype=np.float64)
        if self.scaler is None:
            return x
        z = np.empty_like(x)
        for i, row in enumerate(x):
            z[i] = self.scaler.transform_then_update(row)
        return z


def calibration_view(x: np.ndarray, standardize: bool, n: int) -> np.ndarray:
    rows = np.asarray(x[:min(n, len(x))], dtype=np.float64)
    return CausalStandardizer(rows.shape[1], standardize)(rows)


def calibration_radius(x: np.ndarray) -> float:
    if len(x) < 3:
        return 1.0
    k = min(6, len(x))
    distances, _ = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(x).kneighbors(x)
    return float(max(np.quantile(distances[:, -1], 0.80), 1e-6))


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "modern_methods_all_datasets.json").read_text(encoding="utf-8"))


def validate_config(cfg: dict) -> None:
    if cfg.get("schema") != "modern_methods_all_datasets_v1":
        raise ValueError("unexpected modern-method protocol schema")
    for dataset in DATASETS:
        dcfg = cfg["datasets"][dataset]
        path = ROOT / dcfg["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(dcfg["n_clusters"]) < 2 or int(dcfg["window_size"]) < 1:
            raise ValueError(f"invalid configuration for {dataset}")


def run_one(dataset: str, method: str, seed: int, max_observations: int | None) -> dict:
    cfg = load_config()
    dcfg = cfg["datasets"][dataset]
    path = ROOT / dcfg["path"]
    raw = np.load(path, allow_pickle=False)
    x = np.asarray(raw["x"] if "x" in raw.files else raw["X"], dtype=np.float64)
    y = np.asarray(raw["y"], dtype=np.int64)
    source_n = len(y)
    if max_observations is not None:
        n = min(int(max_observations), source_n)
        x, y = x[:n], y[:n]

    standardize = dataset != "tweeteval"
    cal = calibration_view(x, standardize, int(dcfg.get("calibration_rows", 1000)))
    radius = calibration_radius(cal)
    causal = CausalStandardizer(x.shape[1], standardize)
    mcfg = cfg["methods"][method]
    if method == "twstream":
        model = TWStreamComparator(x.shape[1], seed, max_clusters=200, max_outliers=200,
                                   radius=radius, k=int(mcfg["k_neighbors"]), lam=float(mcfg["lambda"]))
        preprocess = causal
    else:
        model = FRAARTComparator(x.shape[1], seed, a=float(mcfg["a"]),
                                 vigilance=float(mcfg["vigilance"]), choice=float(mcfg["choice"]),
                                 beta=float(mcfg["beta"]), max_stream_points=max(len(y), 1))
        lo, hi = cal.min(axis=0), cal.max(axis=0)

        def preprocess(block):
            return np.clip((causal(block) - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    ev = evaluate_stream(model, x, y, int(dcfg["n_clusters"]),
                         window_size=int(dcfg["window_size"]), warmup_windows=0,
                         preprocess_block=preprocess)
    return {
        "schema": "modern_all_dataset_run_v1",
        "protocol_id": cfg["schema"],
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "source_data": str(path.relative_to(ROOT)),
        "source_sha256": file_sha256(path),
        "source_observations": int(source_n),
        "processed_observations": int(len(y)),
        "features": int(x.shape[1]),
        "preprocessing": dcfg["preprocessing"],
        "calibration_rows": int(dcfg.get("calibration_rows", 1000)),
        "twstream_radius_from_calibration": radius if method == "twstream" else None,
        "method_provenance": method_metadata(method),
        "evaluation": ev.to_dict(),
        "diagnostics": model.diagnostics(),
    }


def write_outputs(root: Path, payloads: list[dict]) -> None:
    rows = []
    for p in payloads:
        e = p["evaluation"]
        rows.append({
            "dataset": p["dataset"], "method": p["method"], "seed": p["seed"],
            "n_observations": p["processed_observations"], "n_windows": e["n_windows"],
            "ari_observation_weighted": e["ari_observation_weighted"],
            "nmi_observation_weighted": e["nmi_observation_weighted"],
            "coverage": e["coverage"], "runtime_s": e["runtime_seconds"],
            "protocol_id": p["protocol_id"], "source_sha256": p["source_sha256"],
            "implementation_fidelity": p["method_provenance"]["fidelity"],
        })
    if not rows:
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "seedwise_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    aggregate = []
    for dataset in DATASETS:
        for method in METHODS:
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if group:
                aggregate.append({
                    "Dataset": DISPLAY[dataset],
                    "Method": "TWStream" if method == "twstream" else "FRA-ART",
                    "ARI": float(np.mean([float(r["ari_observation_weighted"]) for r in group])),
                    "NMI": float(np.mean([float(r["nmi_observation_weighted"]) for r in group])),
                    "Coverage": float(np.mean([float(r["coverage"]) for r in group])),
                    "Runs": len(group),
                })
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate[0])); w.writeheader(); w.writerows(aggregate)

def main() -> None:
    ap = argparse.ArgumentParser(description="Run TWStream and FRA-ART on all four packaged study datasets.")
    ap.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    ap.add_argument("--method", choices=["all", *METHODS], default="all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--max-observations", type=int)
    ap.add_argument("--out-root", type=Path, default=ROOT / "results/execution_runs/modern_all_datasets")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="validate configuration and dataset paths without executing methods")
    args = ap.parse_args()

    cfg = load_config()
    validate_config(cfg)
    if args.dry_run:
        print(json.dumps({"pass": True, "protocol": cfg["schema"], "datasets": list(DATASETS), "methods": list(METHODS)}, indent=2))
        return
    args.out_root = ensure_execution_output(ROOT, args.out_root)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    methods = METHODS if args.method == "all" else (args.method,)
    seeds = [int(s) for s in cfg["seeds"]] if args.all_seeds else [int(args.seed)]
    payloads = []
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                out = args.out_root / dataset / method / f"seed_{seed}.json"
                if out.exists() and not args.force:
                    cached = json.loads(out.read_text(encoding="utf-8"))
                    if cached.get("schema") == "modern_all_dataset_run_v1":
                        payloads.append(cached)
                        continue
                payload = run_one(dataset, method, seed, args.max_observations)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                payloads.append(payload)
                print(json.dumps({"dataset": dataset, "method": method, "seed": seed,
                                  "ari": payload["evaluation"]["ari_observation_weighted"],
                                  "nmi": payload["evaluation"]["nmi_observation_weighted"]}, indent=2))
    write_outputs(args.out_root, payloads)


if __name__ == "__main__":
    main()
