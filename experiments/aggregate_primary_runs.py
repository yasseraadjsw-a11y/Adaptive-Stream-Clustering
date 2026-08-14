from __future__ import annotations

"""Validate and aggregate fresh unified primary executions only."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asc_stream.paper_protocol import DISPLAY, SEEDS
from asc_stream.release_io import ensure_execution_output

DATASETS = ("covertype", "electricity", "tweeteval", "synthetic_gmm")
METHODS = ("proposed", "fixed_rank", "clustream", "denstream", "streamkmpp", "twstream", "fra_art")
METRICS = (
    "online_chunk_mean_ari",
    "online_chunk_mean_nmi",
    "complete_window_final_ari",
    "complete_window_final_nmi",
    "runtime_ms_per_1000",
)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def _finite_metric(payload: dict, name: str) -> bool:
    value = payload.get(name)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _run_path(root: Path, dataset: str, method: str, seed: int) -> Path:
    return root / dataset / method / f"seed_{seed}.json"


def validate_payload(
    payload: dict,
    *,
    dataset: str,
    method: str,
    seed: int,
    require_full_stream: bool,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != "unified_primary_run_v1":
        errors.append("schema")
    if payload.get("dataset") != dataset:
        errors.append("dataset")
    if payload.get("method") != method:
        errors.append("method")
    if int(payload.get("seed", -1)) != int(seed):
        errors.append("seed")
    if require_full_stream and payload.get("full_stream") is not True:
        errors.append("full_stream")
    if float(payload.get("observation_coverage", -1.0)) != 1.0:
        errors.append("observation_coverage")
    for metric in METRICS:
        if not _finite_metric(payload, metric):
            errors.append(metric)
    return errors


def aggregate(
    root: Path,
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    require_full_stream: bool,
) -> dict:
    root = ensure_execution_output(ROOT, root)
    missing: list[str] = []
    invalid: list[dict] = []
    payloads: list[dict] = []

    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                path = _run_path(root, dataset, method, seed)
                if not path.is_file():
                    missing.append(str(path.relative_to(ROOT)))
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    invalid.append({"path": str(path.relative_to(ROOT)), "errors": [type(exc).__name__]})
                    continue
                errors = validate_payload(
                    payload,
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    require_full_stream=require_full_stream,
                )
                if errors:
                    invalid.append({"path": str(path.relative_to(ROOT)), "errors": errors})
                    continue
                payloads.append(payload)

    expected = len(datasets) * len(methods) * len(seeds)
    manifest = {
        "schema": "unified_primary_manifest_v1",
        "root": str(root.relative_to(ROOT)),
        "datasets": list(datasets),
        "methods": list(methods),
        "seeds": list(seeds),
        "require_full_stream": require_full_stream,
        "expected_runs": expected,
        "valid_runs": len(payloads),
        "missing_runs": len(missing),
        "invalid_runs": len(invalid),
        "complete": len(payloads) == expected and not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "execution_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["complete"]:
        return manifest

    seedwise_fields = [
        "dataset", "display", "method", "seed", "full_stream", "source_mode",
        "public_source_verified", "evaluated_observations", "observation_coverage",
        *METRICS,
    ]
    with (root / "seedwise_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=seedwise_fields)
        writer.writeheader()
        for payload in payloads:
            writer.writerow({name: payload.get(name) for name in seedwise_fields})

    dataset_rows: list[dict] = []
    for dataset in datasets:
        for method in methods:
            group = [p for p in payloads if p["dataset"] == dataset and p["method"] == method]
            row = {
                "dataset": dataset,
                "display": DISPLAY[dataset],
                "method": method,
                "runs": len(group),
            }
            for metric in METRICS:
                values = [float(p[metric]) for p in group]
                row[f"{metric}_mean"] = _mean(values)
                row[f"{metric}_sd"] = _sd(values)
            dataset_rows.append(row)

    dataset_fields = list(dataset_rows[0])
    with (root / "dataset_method_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_fields)
        writer.writeheader()
        writer.writerows(dataset_rows)

    overall_rows: list[dict] = []
    for method in methods:
        method_rows = [row for row in dataset_rows if row["method"] == method]
        row = {"method": method, "datasets": len(method_rows), "runs": sum(int(r["runs"]) for r in method_rows)}
        for metric in METRICS:
            dataset_means = [float(r[f"{metric}_mean"]) for r in method_rows]
            row[f"{metric}_equal_dataset_mean"] = _mean(dataset_means)
        overall_rows.append(row)

    with (root / "overall_equal_dataset_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall_rows[0]))
        writer.writeheader()
        writer.writerows(overall_rows)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and aggregate fresh seven-method executions.")
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "execution_runs" / "primary")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--allow-smoke-subsets", action="store_true")
    args = parser.parse_args()
    manifest = aggregate(
        args.root,
        datasets=tuple(args.datasets),
        methods=tuple(args.methods),
        seeds=tuple(int(seed) for seed in args.seeds),
        require_full_stream=not args.allow_smoke_subsets,
    )
    print(json.dumps(manifest, indent=2))
    if not manifest["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
