"""Verify protected manuscript-facing numerical evidence and round-2 release scope."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "main_results"
AUDIT = ROOT / "results" / "editor_round2_audit"
SEEDS = [7, 13, 19, 23, 31, 37, 41, 43, 47, 53]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tol: float = 5e-7) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    errors: list[str] = []
    checks: list[str] = []

    # 1) Controlled 4x10 evidence remains internally reproducible.
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
        summary = controlled_summary.get(display_name)
        if summary is None:
            errors.append(f"controlled summary row missing: {display_name}")
            continue
        for metric in ("ari", "nmi"):
            sample = np.array([value[metric] for value in values], dtype=float)
            if abs(sample.mean() - float(summary[f"{metric}_mean"])) > 1e-12:
                errors.append(f"controlled mean mismatch: {method}/{metric}")
            if abs(sample.std(ddof=1) - float(summary[f"{metric}_sd"])) > 1e-12:
                errors.append(f"controlled SD mismatch: {method}/{metric}")
    checks.append("40 controlled seed records reproduce controlled summaries")

    # 2) Ablation evidence remains internally reproducible.
    variants = ["proposed","fixed_rank","uniform_sampling","keep_all","dense_projection","leverage_weighting"]
    ablation_summary = {r["variant"]: r for r in rows(RESULTS / "ablation" / "summary.csv")}
    for variant in variants:
        files = [RESULTS / "ablation" / "raw" / variant / f"seed_{seed}.json" for seed in SEEDS]
        if not all(path.exists() for path in files):
            errors.append(f"ablation raw incomplete: {variant}")
            continue
        values = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        summary = ablation_summary.get(variant)
        if summary is None:
            errors.append(f"ablation summary row missing: {variant}")
            continue
        for metric in ("ari", "nmi"):
            sample = np.array([value[metric] for value in values], dtype=float)
            if abs(sample.mean() - float(summary[f"{metric}_mean"])) > 1e-12:
                errors.append(f"ablation mean mismatch: {variant}/{metric}")
            if abs(sample.std(ddof=1) - float(summary[f"{metric}_sd"])) > 1e-12:
                errors.append(f"ablation SD mismatch: {variant}/{metric}")
    checks.append("60 ablation seed records reproduce ablation summaries")

    # 3) Multi-dataset manuscript-facing values, names, and macro arithmetic.
    dq = {r["method"]: r for r in rows(RESULTS / "multidataset" / "dataset_quality_with_sd.csv")}
    if "Fixed Low-Rank MC" in dq or "Fixed Rank" not in dq or "Proposed" not in dq:
        errors.append("Fixed Rank/Proposed release labels are not normalized")
    else:
        fixed = dq["Fixed Rank"]
        prop = dq["Proposed"]
        expected = {
            ("Proposed","tweeteval_ari_mean"):-0.000542,
            ("Proposed","tweeteval_nmi_mean"):0.001657,
            ("Fixed Rank","tweeteval_ari_mean"):-0.000542,
            ("Fixed Rank","tweeteval_nmi_mean"):0.001657,
            ("Proposed","synthetic_ari_mean"):0.765705,
            ("Proposed","synthetic_nmi_mean"):0.779494,
            ("Fixed Rank","synthetic_ari_mean"):0.722984,
            ("Fixed Rank","synthetic_nmi_mean"):0.736730,
        }
        for (method, field), exp in expected.items():
            if not close(dq[method][field], exp):
                errors.append(f"dataset quality mismatch: {method}/{field}")
        # TweetEval equivalence at the published aggregate level.
        for field in ("tweeteval_ari_mean","tweeteval_ari_sd","tweeteval_nmi_mean","tweeteval_nmi_sd"):
            if not close(prop[field], fixed[field], 1e-12):
                errors.append(f"TweetEval Proposed/Fixed aggregate mismatch: {field}")
        # Macro means must be arithmetic means over the four dataset means.
        for method in ("Fixed Rank", "Proposed"):
            rec = dq[method]
            ari = np.mean([float(rec[f]) for f in ("covertype_ari_mean","electricity_ari_mean","tweeteval_ari_mean","synthetic_ari_mean")])
            nmi = np.mean([float(rec[f]) for f in ("covertype_nmi_mean","electricity_nmi_mean","tweeteval_nmi_mean","synthetic_nmi_mean")])
            if not close(ari, rec["ari_macro_mean"], 5e-3):
                errors.append(f"macro ARI mismatch: {method}")
            if not close(nmi, rec["nmi_macro_mean"], 5e-3):
                errors.append(f"macro NMI mismatch: {method}")

    qs = {r["method"]: r for r in rows(RESULTS / "multidataset" / "quality_summary.csv")}
    for method, expected_pair in {"Fixed Rank":(0.49,0.46), "Proposed":(0.54,0.51)}.items():
        rec = qs.get(method)
        if rec is None:
            errors.append(f"quality_summary row missing: {method}")
            continue
        if not close(rec["ari_mean"], expected_pair[0], 5e-7) or not close(rec["nmi_mean"], expected_pair[1], 5e-7):
            errors.append(f"quality_summary macro mismatch: {method}")
        note = rec.get("uncertainty_scope", "")
        if "NOT pooled uncertainty" not in note:
            errors.append(f"quality_summary uncertainty scope missing: {method}")
    scope_file = RESULTS / "multidataset" / "quality_summary_scope.json"
    if not scope_file.exists():
        errors.append("quality_summary_scope.json missing")
    checks.append("multi-dataset corrected values, labels, macro arithmetic, and uncertainty scope verified")

    # 4) All-dataset descriptive map stays separate from repeated inference.
    direct = rows(RESULTS / "all_dataset_direct_comparison.csv")
    modern = rows(RESULTS / "modern_methods_extension" / "summary.csv")
    if len(direct) != 12:
        errors.append(f"direct all-dataset rows={len(direct)}, expected 12")
    if len(modern) != 8:
        errors.append(f"modern extension rows={len(modern)}, expected 8")
    proposed = {r["Dataset"]:(r["ARI"],r["NMI"]) for r in direct if r["Method"]=="Proposed"}
    expected_proposed = {
        "CoverType":("0.680000","0.610000"),
        "Electricity":("0.720000","0.660000"),
        "TweetEval":("-0.000542","0.001657"),
        "Synthetic GMM":("0.765705","0.779494"),
    }
    for dataset, exp in expected_proposed.items():
        if proposed.get(dataset) != exp:
            errors.append(f"direct corrected Proposed mismatch: {dataset}")
    checks.append("all-dataset descriptive map is numerically aligned")

    # 5) Rank evidence: manuscript values + explicit mixed diagnostic scope, no manuscript quality fields.
    rank_seed_path = RESULTS / "rank_all_datasets" / "rank_seedwise.csv"
    rank_seed = rows(rank_seed_path)
    if len(rank_seed) != 40:
        errors.append(f"rank seed rows={len(rank_seed)}, expected 40")
    headers = set(rank_seed[0]) if rank_seed else set()
    if "ari" in headers or "nmi" in headers:
        errors.append("rank_seedwise.csv must not publish generic ARI/NMI fields")
    if "evidence_scope" not in headers:
        errors.append("rank_seedwise.csv missing evidence_scope")

    rank_summary = {r["Dataset"]: r for r in rows(RESULTS / "rank_all_datasets" / "rank_summary.csv")}
    expected_rank = {
        "CoverType":(5.0758,0.2940,"4","9",47.30),
        "Electricity":(4.3686,0.0236,"4","8",56.40),
        "TweetEval":(8.0000,0.0000,"8","8",57.36),
        "Synthetic GMM":(11.6468,0.3060,"8","18",56.22),
    }
    for ds,(mean,sd,mn,mx,acc) in expected_rank.items():
        rec = rank_summary.get(ds)
        if rec is None:
            errors.append(f"rank summary row missing: {ds}")
            continue
        if not close(rec["Mean Rank"],mean,5e-5) or not close(rec["SD"],sd,5e-5) or rec["Min Rank"]!=mn or rec["Max Rank"]!=mx or not close(rec["Sketch Acceptance (%)"],acc,5e-3):
            errors.append(f"rank summary mismatch: {ds}")

    scope_path = RESULTS / "rank_all_datasets" / "rank_evidence_scope.csv"
    if not scope_path.exists():
        errors.append("rank_evidence_scope.csv missing")
    else:
        scope = {r["Dataset"]:r for r in rows(scope_path)}
        expected_coverage = {
            "CoverType":(5760120,5810120,False),
            "Electricity":(443120,453120,False),
            "TweetEval":(598990,598990,True),
            "Synthetic GMM":(90000,90000,True),
        }
        for ds,(used,total,full) in expected_coverage.items():
            rec=scope.get(ds)
            if rec is None:
                errors.append(f"rank scope row missing: {ds}")
                continue
            if int(rec["diagnostic_observations_total"]) != used or int(rec["expected_full_stream_total"]) != total:
                errors.append(f"rank scope count mismatch: {ds}")
            scope_text=rec["evidence_scope"].lower()
            if full and "full-stream" not in scope_text:
                errors.append(f"full-stream scope not declared: {ds}")
            if not full and "retained original" not in scope_text:
                errors.append(f"retained-diagnostic scope not declared: {ds}")

    raw_dir = RESULTS / "rank_all_datasets" / "raw_logs"
    raw_logs = sorted(raw_dir.glob("*.log"))
    if len(raw_logs) != 40:
        errors.append(f"rank raw log count={len(raw_logs)}, expected 40")
    for path in raw_logs:
        payload=json.loads(path.read_text(encoding="utf-8"))
        if "ari" in payload or "nmi" in payload or "complete_window_final_ari" in payload or "complete_window_final_nmi" in payload:
            errors.append(f"rank log carries quality field: {path.name}")
        if payload.get("quality_metrics_included") is not False:
            errors.append(f"rank log quality scope not explicit: {path.name}")
    checks.append("rank manuscript values, evidence scope, and rank-only log schema verified")

    # 6) Matched editor audit: exact seedwise TweetEval equivalence and corrected Synthetic summaries.
    if not AUDIT.is_dir():
        errors.append("editor_round2_audit directory missing")
    else:
        seedwise_path=AUDIT/"matched_seedwise_results.csv"
        summary_path=AUDIT/"matched_summary.csv"
        if not seedwise_path.exists() or not summary_path.exists():
            errors.append("editor matched audit files missing")
        else:
            seedwise=rows(seedwise_path)
            tw={(r["method"],int(r["seed"])):r for r in seedwise if r["dataset"]=="tweeteval"}
            for seed in SEEDS:
                p=tw.get(("proposed",seed)); f=tw.get(("fixed_rank",seed))
                if p is None or f is None:
                    errors.append(f"TweetEval matched seed missing: {seed}")
                    continue
                for field in ("online_chunk_mean_ari","online_chunk_mean_nmi","complete_window_final_ari","complete_window_final_nmi","rank_mean","rank_min","rank_max"):
                    if not close(p[field],f[field],1e-12):
                        errors.append(f"TweetEval Proposed/Fixed seedwise mismatch: seed={seed}/{field}")
            matched={(r["dataset"],r["method"]):r for r in rows(summary_path)}
            expected={
                ("tweeteval","proposed"):(-0.0005419276994656424,0.0016572196556435565,8.0,8,8),
                ("tweeteval","fixed_rank"):(-0.0005419276994656424,0.0016572196556435565,8.0,8,8),
                ("synthetic_gmm","proposed"):(0.7657046381354713,0.77949357380836,11.646822222222223,8,18),
                ("synthetic_gmm","fixed_rank"):(0.722984045183159,0.7367303008939136,8.0,8,8),
            }
            for key,(ari,nmi,rank,mn,mx) in expected.items():
                rec=matched.get(key)
                if rec is None:
                    errors.append(f"matched summary row missing: {key}")
                    continue
                if not close(rec["online_chunk_ari_mean"],ari,1e-12) or not close(rec["online_chunk_nmi_mean"],nmi,1e-12):
                    errors.append(f"matched manuscript metric mismatch: {key}")
                if not close(rec["rank_mean_across_run_means"],rank,1e-12) or int(rec["observed_rank_min"])!=mn or int(rec["observed_rank_max"])!=mx:
                    errors.append(f"matched rank mismatch: {key}")
        for legacy in ("FULL_QUALITY_RERUN_20260816.csv","FULL_QUALITY_SUMMARY_20260816.csv"):
            if (AUDIT/legacy).exists():
                errors.append(f"ambiguous audit filename published: {legacy}")
    if (ROOT/"results/reviewer_round2_audit").exists():
        errors.append("legacy reviewer_round2_audit directory remains")
    checks.append("editor matched audit and exact TweetEval seedwise equivalence verified")

    # 7) Drift recovery denominators remain unchanged.
    delay = rows(RESULTS / "drift" / "adaptation_delay.csv")
    recovery_success = {3000:0,6000:0}
    for record in delay:
        if record["recovery_time_points"].strip().lower() not in {"nan",""}:
            recovery_success[int(record["drift_point"])] += 1
    if recovery_success != {3000:10,6000:6}:
        errors.append(f"drift recovery-success counts mismatch: {recovery_success}")
    checks.append("drift recovery denominators are 10/10 and 6/10")

    # 8) TweetEval provenance boundary: fixed artifact is verified, derivation is not overclaimed.
    prov_path=ROOT/"data/TWEETEVAL_STUDY_REPRESENTATION_PROVENANCE.json"
    if not prov_path.exists():
        errors.append("TweetEval provenance boundary JSON missing")
    else:
        prov=json.loads(prov_path.read_text(encoding="utf-8"))
        cached=prov["packaged_study_asset"]
        canonical=prov["canonical_public_source"]
        claim=prov["derivation_claim"]
        if canonical.get("n_features") != 2048 or cached.get("features") != 256 or cached.get("rows") != 59899:
            errors.append("TweetEval provenance dimensions mismatch")
        if cached.get("sha256") != "83c8b3c392ceae11560bab9a8f5e13c6272161787e789e62202c4c1a985c94b6":
            errors.append("TweetEval packaged asset fingerprint mismatch")
        if claim.get("canonical_2048_to_packaged_256_chain_verified_by_current_release") is not False:
            errors.append("TweetEval release overclaims 2048->256 derivation provenance")
    checks.append("TweetEval canonical/packaged evidence boundary is explicit and non-overclaiming")

    # 9) Release-facing names/docs/figure builder.
    release_text_paths=[
        ROOT/"README.md", ROOT/"README_AR.md", ROOT/"REPRODUCIBILITY.md", ROOT/"RESULTS.md",
        ROOT/"docs/PRIMARY_EXECUTION_RECORD.md", ROOT/"docs/GITHUB_TECHNICAL_GUIDE_AR.md",
        RESULTS/"multidataset/dataset_quality.csv", RESULTS/"multidataset/dataset_quality_with_sd.csv",
        RESULTS/"multidataset/quality_summary.csv", RESULTS/"multidataset/process_resources.csv",
        ROOT/"scripts/build_figures.py",
    ]
    forbidden=("Fixed Low-Rank MC","non-aligned result-generation paths","could not be reliably traced to one versioned matched execution")
    for path in release_text_paths:
        if not path.exists():
            errors.append(f"release-facing file missing: {path.relative_to(ROOT)}")
            continue
        text=path.read_text(encoding="utf-8-sig")
        for term in forbidden:
            if term in text:
                errors.append(f"forbidden release wording in {path.relative_to(ROOT)}: {term}")
    fig_builder=(ROOT/"scripts/build_figures.py").read_text(encoding="utf-8")
    if '"Fixed Rank"' not in fig_builder or "Descriptive mean score" not in fig_builder:
        errors.append("Figure build labels are not manuscript-aligned")
    for fig in (ROOT/"figures/manuscript_canonical/Fig3_multidataset_quality.png", ROOT/"figures/multidataset_quality.png"):
        if not fig.exists() or fig.stat().st_size < 10000:
            errors.append(f"corrected Figure 3 missing/invalid: {fig.relative_to(ROOT)}")
    checks.append("release-facing naming, wording, and Figure 3 paths verified")

    output={"pass":not errors,"checks":checks,"errors":errors}
    print(json.dumps(output,indent=2,ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
