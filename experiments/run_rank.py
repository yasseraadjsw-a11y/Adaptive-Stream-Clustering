from __future__ import annotations

"""Extract adaptive-rank evidence from the exact main Proposed engine."""

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_primary import _dense_view, build_method, evaluate_full_stream
from asc_stream.optimized import causal_standardize_dense
from asc_stream.paper_protocol import DISPLAY, N_CLUSTERS, SEEDS, load_packaged_study_dataset
from asc_stream.release_io import ensure_execution_output

DATASETS = ("covertype", "electricity", "tweeteval", "synthetic_gmm")

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_cached(payload: dict, dataset: str, seed: int) -> bool:
    return (
        payload.get("schema") == "main_proposed_rank_diagnostic_v1"
        and payload.get("dataset") == dataset
        and int(payload.get("seed", -1)) == int(seed)
        and payload.get("full_stream") is True
        and payload.get("engine") == "main_proposed"
    )


def run_one(dataset: str, seed: int, chunk_size: int, out_root: Path) -> dict:
    loaded = load_packaged_study_dataset(ROOT, dataset)
    x, y = loaded.x, loaded.y
    prestandardized = False
    if dataset != "tweeteval" and not sparse.issparse(x):
        x = causal_standardize_dense(np.asarray(x, dtype=np.float64))
        prestandardized = True
    calibration = _dense_view(x[: min(1000, int(x.shape[0]))])
    model, parameters = build_method(
        "proposed",
        dataset,
        int(x.shape[1]),
        int(seed),
        tweet_preprojected=(dataset == "tweeteval"),
        prestandardized=prestandardized,
        calibration=calibration,
        n_observations=int(x.shape[0]),
    )

    temp_root = ROOT / "results" / "execution_runs" / "_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asc_rank_eval_", dir=str(temp_root)) as directory:
        metrics = evaluate_full_stream(
            model,
            x,
            y,
            N_CLUSTERS[dataset],
            chunk_size=int(chunk_size),
            work_dir=Path(directory),
        )

    ranks = np.asarray(model.rank_history, dtype=np.int64)
    if len(ranks) != int(x.shape[0]):
        raise RuntimeError(f"rank history length mismatch for {dataset}/seed_{seed}: {len(ranks)} != {x.shape[0]}")
    histogram = Counter(int(rank) for rank in ranks)
    chunks = []
    for start in range(0, len(ranks), int(chunk_size)):
        end = min(start + int(chunk_size), len(ranks))
        values = ranks[start:end]
        chunks.append({
            "start": int(start),
            "end": int(end),
            "mean_rank": float(values.mean()),
            "min_rank": int(values.min()),
            "max_rank": int(values.max()),
            "rank_changes": int(np.count_nonzero(np.diff(values))) if len(values) > 1 else 0,
        })

    telemetry = dict(metrics.pop("diagnostics"))
    payload = {
        "schema": "main_proposed_rank_diagnostic_v1",
        "engine": "main_proposed",
        "dataset": dataset,
        "display": DISPLAY[dataset],
        "seed": int(seed),
        "full_stream": True,
        "source_mode": "study",
        "data_source": loaded.source,
        "public_source_verified": bool(loaded.canonical),
        "parameters": parameters,
        "telemetry": telemetry,
        "rank_histogram": {str(rank): int(histogram[rank]) for rank in sorted(histogram)},
        "rank_chunks": chunks,
        **metrics,
    }
    path = out_root / dataset / "proposed" / f"seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_summaries(summary_dir: Path, payloads: list[dict], datasets: tuple[str, ...]) -> None:
    seedwise: list[dict] = []
    summary: list[dict] = []
    distribution: list[dict] = []

    for dataset in datasets:
        rows = [payload for payload in payloads if payload["dataset"] == dataset]
        run_means = np.asarray([row["telemetry"]["rank_mean"] for row in rows], dtype=np.float64)
        acceptances = np.asarray([row["telemetry"]["sampling_acceptance"] for row in rows], dtype=np.float64)
        combined: Counter[int] = Counter()
        for row in rows:
            telemetry = row["telemetry"]
            seedwise.append({
                "dataset": DISPLAY[dataset],
                "dataset_id": dataset,
                "seed": row["seed"],
                "evaluated_observations": row["evaluated_observations"],
                "mean_rank": telemetry["rank_mean"],
                "rank_sd_within_run": telemetry["rank_std_within_run"],
                "minimum_rank_observed": telemetry["rank_min_observed"],
                "maximum_rank_observed": telemetry["rank_max_observed"],
                "rank_change_count": telemetry["rank_change_count"],
                "final_rank": telemetry["final_rank"],
                "basis_updates": telemetry["basis_updates"],
                "sampling_acceptance": telemetry["sampling_acceptance"],
                "ari": row["complete_window_final_ari"],
                "nmi": row["complete_window_final_nmi"],
            })
            combined.update({int(rank): int(count) for rank, count in row["rank_histogram"].items()})

        total = sum(combined.values())
        summary.append({
            "dataset": DISPLAY[dataset],
            "runs": len(rows),
            "mean_rank": float(run_means.mean()),
            "sd_between_run_means": float(run_means.std(ddof=1)) if len(run_means) > 1 else 0.0,
            "min_rank_observed": min(combined),
            "max_rank_observed": max(combined),
            "sampling_acceptance_pct": float(100.0 * acceptances.mean()),
        })
        for rank in sorted(combined):
            distribution.append({
                "dataset": DISPLAY[dataset],
                "rank": rank,
                "observations": combined[rank],
                "pct_observations": float(100.0 * combined[rank] / max(total, 1)),
            })

    _write_csv(summary_dir / "rank_seedwise.csv", seedwise)
    _write_csv(summary_dir / "rank_summary.csv", summary)
    _write_csv(summary_dir / "rank_distribution.csv", distribution)
    metadata = {
        "schema": "main_proposed_rank_summary_v1",
        "engine": "main_proposed",
        "datasets": list(datasets),
        "seeds": list(SEEDS),
        "runs": len(payloads),
        "note": "All rank fields are extracted from the same Proposed implementation used by run-primary.",
    }
    (summary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main-Proposed adaptive-rank diagnostic.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "execution_runs" / "rank_diagnostic" / "raw")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "results" / "execution_runs" / "rank_diagnostic")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_root = ensure_execution_output(ROOT, args.out_root)
    summary_dir = ensure_execution_output(ROOT, args.summary_dir)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)

    payloads: list[dict] = []
    for dataset in datasets:
        for seed in SEEDS:
            path = out_root / dataset / "proposed" / f"seed_{seed}.json"
            if path.exists() and args.resume:
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached = {}
                if _valid_cached(cached, dataset, seed):
                    payloads.append(cached)
                    continue
            if path.exists() and not args.force:
                raise SystemExit(f"fresh rank output already exists: {path}; use --resume or --force")
            payloads.append(run_one(dataset, seed, args.chunk_size, out_root))

    write_summaries(summary_dir, payloads, tuple(datasets))
    print(json.dumps({"engine": "main_proposed", "datasets": list(datasets), "runs": len(payloads)}, indent=2))


if __name__ == "__main__":
    main()
