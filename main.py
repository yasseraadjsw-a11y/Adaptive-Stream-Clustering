from __future__ import annotations
from pathlib import Path
import argparse, os, subprocess, sys

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 13, 19, 23, 31, 37, 41, 43, 47, 53]
DATASETS = ["covertype", "electricity", "tweeteval", "synthetic_gmm"]
PRIMARY_METHODS = ["proposed", "fixed_rank", "clustream", "denstream", "streamkmpp", "twstream", "fra_art"]
CONTROLLED_METHODS = ["proposed", "fixed_rank", "twstream", "fra_art"]


def controlled_env():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def run(args):
    completed = subprocess.run(args, check=False, cwd=ROOT, env=controlled_env())
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    ap = argparse.ArgumentParser(description="Adaptive sketch-based stream clustering research package")
    ap.add_argument(
        "action",
        choices=[
            "verify", "verify-manifest", "verify-package",
            "show-results", "show-modern-results", "build-figures",
            "download-data", "prepare-data", "setup-data", "data-status", "verify-data", "verify-canonical-data",
            "rebuild-controlled-data",
            "run-controlled", "run-core", "run-primary", "run-modern-all", "run-rank", "run-ablation", "run-drift",
            "run-sensitivity", "profile-resources", "run-study-suite",
            "setup-external", "run-twstream-official",
        ],
    )
    ap.add_argument("--method")
    ap.add_argument("--dataset")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--source", choices=["study", "canonical"], default="study")
    ap.add_argument("--max-observations", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="skip valid-looking outputs already present in the selected fresh-execution root")
    ap.add_argument("--force", action="store_true", help="overwrite fresh-execution outputs already present; never affects protected main_results")
    ap.add_argument("--dry-run", action="store_true", help="print the selected execution matrix without running experiments")
    args = ap.parse_args()

    if args.seed is not None and args.seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}")

    if args.action == "verify":
        # Verify package structure/dependencies first, then data integrity,
        # result aggregation/consistency, immutable file hashes, and tests.
        run([sys.executable, "scripts/verify_package_health.py"])
        run([sys.executable, "scripts/verify_study_protocol.py"])
        run([sys.executable, "scripts/verify_dataset_provenance.py", "--dataset", "all"])
        run([sys.executable, "scripts/verify_main_results.py"])
        run([sys.executable, "scripts/verify_protected_results.py"])
        run([sys.executable, "scripts/verify_manifest.py"])
        run([sys.executable, "-m", "compileall", "-q", "src", "experiments", "scripts", "main.py"])
        run([sys.executable, "scripts/run_unit_tests_if_available.py"])

    elif args.action == "verify-manifest":
        run([sys.executable, "scripts/verify_manifest.py"])

    elif args.action == "verify-package":
        run([sys.executable, "scripts/verify_package_health.py"])
        run([sys.executable, "scripts/verify_dataset_provenance.py", "--dataset", "all"])

    elif args.action == "show-results":
        files = [
            "results/main_results/multidataset/quality_summary.csv",
            "results/main_results/multidataset/dataset_quality.csv",
            "results/main_results/controlled/method_summary.csv",
            "results/main_results/ablation/summary.csv",
            "results/main_results/rank_all_datasets/rank_summary.csv",
            "results/main_results/multidataset/process_resources.csv",
        ]
        for rel in files:
            print(f"\n### {rel}")
            print((ROOT / rel).read_text(encoding="utf-8-sig").strip())

    elif args.action == "show-modern-results":
        rel = "results/main_results/modern_methods_extension/summary.csv"
        print(f"\n### {rel}")
        print((ROOT / rel).read_text(encoding="utf-8-sig").strip())

    elif args.action == "build-figures":
        run([sys.executable, "scripts/build_figures.py"])

    elif args.action == "download-data":
        run([sys.executable, "scripts/acquire_datasets.py", "--dataset", args.dataset or "all"])

    elif args.action == "prepare-data":
        run([sys.executable, "scripts/prepare_datasets.py", "--dataset", args.dataset or "all"])

    elif args.action == "setup-data":
        ds = args.dataset or "all"
        run([sys.executable, "scripts/acquire_datasets.py", "--dataset", ds])
        run([sys.executable, "scripts/prepare_datasets.py", "--dataset", ds])

    elif args.action == "data-status":
        run([sys.executable, "scripts/verify_dataset_provenance.py", "--dataset", args.dataset or "all"])

    elif args.action == "verify-data":
        run([sys.executable, "scripts/verify_dataset_provenance.py", "--dataset", args.dataset or "all"])

    elif args.action == "verify-canonical-data":
        run([sys.executable, "scripts/verify_dataset_provenance.py", "--dataset", args.dataset or "all", "--require-canonical"])

    elif args.action == "rebuild-controlled-data":
        run([sys.executable, "scripts/rebuild_controlled_stream.py"])

    elif args.action == "run-controlled":
        methods = [args.method] if args.method else CONTROLLED_METHODS
        invalid = [m for m in methods if m not in CONTROLLED_METHODS]
        if invalid: raise SystemExit(f"unknown controlled method(s): {invalid}")
        seeds = [args.seed] if args.seed is not None else SEEDS
        for method in methods:
            for seed in seeds:
                out = ROOT / "results" / "execution_runs" / "controlled" / method / f"seed_{seed}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                run([sys.executable, "experiments/run_controlled_method.py", "--method", method, "--seed", str(seed), "--out", str(out.relative_to(ROOT))])

    elif args.action == "run-core":
        methods = [args.method] if args.method else ["proposed", "fixed_rank"]
        seeds = [args.seed] if args.seed is not None else SEEDS
        for method in methods:
            if method not in {"proposed", "fixed_rank"}: raise SystemExit(f"unknown core method: {method}")
            for seed in seeds:
                out = ROOT / "results" / "execution_runs" / "core" / method / f"seed_{seed}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                run([sys.executable, "experiments/run_core_pair.py", "--method", method, "--seed", str(seed), "--out", str(out.relative_to(ROOT))])

    elif args.action == "run-primary":
        datasets = [args.dataset] if args.dataset else DATASETS
        methods = [args.method] if args.method else PRIMARY_METHODS
        seeds = [args.seed] if args.seed is not None else SEEDS
        output_root = (args.output_root or (ROOT / "results" / "execution_runs" / "primary")).resolve()
        execution_root = (ROOT / "results" / "execution_runs").resolve()
        try:
            output_root.relative_to(execution_root)
        except ValueError as exc:
            raise SystemExit(f"fresh output root must be inside {execution_root}: {output_root}") from exc
        planned = []
        for ds in datasets:
            if ds not in DATASETS: raise SystemExit(f"unknown dataset: {ds}")
            for method in methods:
                if method not in PRIMARY_METHODS: raise SystemExit(f"unknown primary method: {method}")
                for seed in seeds:
                    out = output_root / ds / method / f"seed_{seed}.json"
                    planned.append({"dataset":ds,"method":method,"seed":seed,"output":str(out)})
                    if args.dry_run:
                        continue
                    if out.exists() and args.resume:
                        try:
                            cached = __import__('json').loads(out.read_text(encoding='utf-8'))
                        except Exception:
                            cached = {}
                        if cached.get('schema') == 'unified_primary_run_v1' and cached.get('dataset') == ds and cached.get('method') == method and int(cached.get('seed',-1)) == seed:
                            print(f"Skipping completed fresh run: {ds}/{method}/seed_{seed}")
                            continue
                    if out.exists() and not args.force:
                        raise SystemExit(f"fresh output already exists: {out}; use --resume or --force")
                    cmd=[sys.executable,"experiments/run_primary.py","--dataset",ds,"--method",method,"--seed",str(seed),"--source",args.source,"--out",str(out)]
                    if args.max_observations is not None: cmd += ["--max-observations",str(args.max_observations)]
                    if args.chunk_size is not None: cmd += ["--chunk-size",str(args.chunk_size)]
                    run(cmd)
        if args.dry_run:
            import json
            print(json.dumps({"schema":"unified_primary_execution_plan_v1","runs":len(planned),"matrix":planned},indent=2))
            return
        aggregate_cmd=[
            sys.executable,"experiments/aggregate_primary_runs.py",
            "--root",str(output_root),"--datasets",*datasets,"--methods",*methods,"--seeds",*[str(seed) for seed in seeds],
        ]
        if args.max_observations is not None:
            aggregate_cmd.append("--allow-smoke-subsets")
        run(aggregate_cmd)

    elif args.action == "run-modern-all":
        cmd = [sys.executable, "experiments/run_modern_all_datasets.py"]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if args.method:
            cmd += ["--method", args.method]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.max_observations is not None:
            cmd += ["--max-observations", str(args.max_observations)]
        run(cmd)

    elif args.action == "run-rank":
        if args.seed is not None:
            raise SystemExit("run-rank uses the complete declared 10-seed diagnostic; omit --seed")
        run([sys.executable,"experiments/run_rank.py","--dataset",args.dataset or "all"])

    elif args.action == "run-ablation":
        variants = [args.method] if args.method else [
            "proposed", "fixed_rank", "uniform_sampling", "keep_all", "dense_projection", "leverage_weighting"
        ]
        seeds = [args.seed] if args.seed is not None else SEEDS
        for variant in variants:
            for seed in seeds:
                out = ROOT / "results" / "execution_runs" / "ablation" / variant / f"seed_{seed}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                run([sys.executable, "experiments/run_ablation.py", "--variant", variant, "--seed", str(seed), "--out", str(out.relative_to(ROOT))])

    elif args.action == "run-drift":
        run([sys.executable, "experiments/run_drift_trace.py"])

    elif args.action == "run-sensitivity":
        run([sys.executable, "experiments/run_sensitivity_actual.py"])

    elif args.action == "profile-resources":
        run([sys.executable, "experiments/profile_resources.py"])

    elif args.action == "run-study-suite":
        # One-command orchestration of the released execution paths. Fresh outputs
        # remain isolated under results/execution_runs; protected manuscript
        # summaries are never overwritten. This can be computationally heavy.
        sourced = ["--source", args.source]
        run([sys.executable, "main.py", "run-controlled"])
        run([sys.executable, "main.py", "run-primary", *sourced])
        run([sys.executable, "main.py", "run-rank"])
        run([sys.executable, "main.py", "run-ablation"])
        run([sys.executable, "main.py", "run-drift"])
        run([sys.executable, "main.py", "run-sensitivity"])
        run([sys.executable, "main.py", "profile-resources"])
        run([sys.executable, "main.py", "build-figures"])

    elif args.action == "setup-external":
        run([sys.executable, "scripts/setup_external_methods.py", "--method", "twstream"])

    elif args.action == "run-twstream-official":
        if not args.dataset:
            raise SystemExit("run-twstream-official requires --dataset")
        run([sys.executable, "experiments/run_twstream_official.py", "--dataset", args.dataset])


if __name__ == "__main__":
    main()
