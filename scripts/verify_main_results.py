"""Verify protected numerical evidence without depending on journal documents."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "main_results"
SEEDS = [7, 13, 19, 23, 31, 37, 41, 43, 47, 53]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    errors: list[str] = []
    checks: list[str] = []

    method_names = {
        "proposed": "Proposed",
        "fixed_rank": "Fixed Rank",
        "twstream": "TWStream",
        "fra_art": "FRA-ART",
    }
    controlled_summary = {r["method"]: r for r in rows(RESULTS / "controlled" / "method_summary.csv")}
    for method, display_name in method_names.items():
        files = [RESULTS / "controlled" / "raw" / method / f"seed_{seed}.json" for seed in SEEDS]
        if not all(path.exists() for path in files):
            errors.append(f"controlled raw incomplete: {method}")
            continue
        values = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        summary = controlled_summary[display_name]
        for metric in ("ari", "nmi"):
            sample = np.array([value[metric] for value in values], dtype=float)
            if abs(sample.mean() - float(summary[f"{metric}_mean"])) > 1e-12:
                errors.append(f"controlled mean mismatch: {method}/{metric}")
            if abs(sample.std(ddof=1) - float(summary[f"{metric}_sd"])) > 1e-12:
                errors.append(f"controlled SD mismatch: {method}/{metric}")
    checks.append("40 controlled seed records reproduce controlled summaries")

    variants = [
        "proposed",
        "fixed_rank",
        "uniform_sampling",
        "keep_all",
        "dense_projection",
        "leverage_weighting",
    ]
    ablation_summary = {r["variant"]: r for r in rows(RESULTS / "ablation" / "summary.csv")}
    for variant in variants:
        files = [RESULTS / "ablation" / "raw" / variant / f"seed_{seed}.json" for seed in SEEDS]
        if not all(path.exists() for path in files):
            errors.append(f"ablation raw incomplete: {variant}")
            continue
        values = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        for metric in ("ari", "nmi"):
            sample = np.array([value[metric] for value in values], dtype=float)
            summary = ablation_summary[variant]
            if abs(sample.mean() - float(summary[f"{metric}_mean"])) > 1e-12:
                errors.append(f"ablation mean mismatch: {variant}/{metric}")
            if abs(sample.std(ddof=1) - float(summary[f"{metric}_sd"])) > 1e-12:
                errors.append(f"ablation SD mismatch: {variant}/{metric}")
    checks.append("60 ablation seed records reproduce ablation summaries")

    rank = rows(RESULTS / "rank_all_datasets" / "rank_seedwise.csv")
    if len(rank) != 40:
        errors.append(f"rank diagnostic rows={len(rank)}, expected 40")
    checks.append("40 dataset-specific rank records are present")

    delay = rows(RESULTS / "drift" / "adaptation_delay.csv")
    recovery_success = {3000: 0, 6000: 0}
    for record in delay:
        if record["recovery_time_points"].strip().lower() not in {"nan", ""}:
            recovery_success[int(record["drift_point"])] += 1
    if recovery_success != {3000: 10, 6000: 6}:
        errors.append(f"drift recovery-success counts mismatch: {recovery_success}")
    checks.append("drift recovery denominators are 10/10 and 6/10")

    primary = rows(RESULTS / "multidataset" / "quality_summary.csv")
    if len(primary) < 2:
        errors.append("primary multidataset summary incomplete")
    direct = rows(RESULTS / "all_dataset_direct_comparison.csv")
    modern = rows(RESULTS / "modern_methods_extension" / "summary.csv")
    if len(direct) != 12:
        errors.append(f"direct all-dataset rows={len(direct)}, expected 12")
    if len(modern) != 8:
        errors.append(f"modern extension rows={len(modern)}, expected 8")

    proposed = {
        record["Dataset"]: (record["ARI"], record["NMI"])
        for record in direct
        if record["Method"] == "Proposed"
    }
    expected_proposed = {
        "CoverType": ("0.680000", "0.610000"),
        "Electricity": ("0.720000", "0.660000"),
        "TweetEval": ("0.390000", "0.420000"),
        "Synthetic GMM": ("0.820000", "0.790000"),
    }
    for dataset, expected in expected_proposed.items():
        if proposed.get(dataset) != expected:
            errors.append(f"protected Proposed value changed: {dataset}")

    direct_modern = {
        (record["Dataset"], record["Method"]): (float(record["ARI"]), float(record["NMI"]))
        for record in direct
        if record["Method"] != "Proposed"
    }
    for record in modern:
        key = (record["Dataset"], record["Method"])
        actual = direct_modern.get(key)
        expected = (float(record["ARI"]), float(record["NMI"]))
        if actual is None or abs(actual[0] - expected[0]) > 5e-7 or abs(actual[1] - expected[1]) > 5e-7:
            errors.append(f"modern extension mapping mismatch: {key[0]}/{key[1]}")
    checks.append("protected original values and modern-method extension mapping are consistent")

    output = {"pass": not errors, "checks": checks, "errors": errors}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
