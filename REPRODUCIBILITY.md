# Reproducibility and verification

## Fixed study seeds

`7, 13, 19, 23, 31, 37, 41, 43, 47, 53`.

## Verification gate

Run:

```bash
python main.py verify
```

The gate performs source compilation, package-health checks, study-protocol checks, packaged-data SHA-256 checks, result aggregation and consistency checks, protected-result fingerprint checks, release-manifest checks, and unit tests.

## Study configuration

Primary settings have one source of truth: `configs/datasets/*.json`. Controlled-drift settings have one source: `configs/controlled_representation_drift.json`. No alternative protocol profile is used by the release.

The sketch basis refresh interval is **100 accepted sketch rows**. The value `0.95` used for retained-energy rank selection is `retained_energy`; the independent micro-cluster fading factor is also `0.95`. There is no separate basis-decay parameter.

## Primary execution record

The execution index enumerates the original study's 200 primary combinations (4 datasets × 5 methods × 10 seeds). The original multi-dataset result tables remain immutable under `results/main_results/multidataset/`.

Fresh full-stream executions use one unified 280-run matrix (4 datasets × 7 methods × 10 seeds) launched with `python main.py run-primary`; results are isolated under `results/execution_runs/primary/`.

The unified runner creates one JSON record per seed and, only after validating the selected matrix, builds `execution_manifest.json`, `seedwise_results.csv`, `dataset_method_summary.csv`, and `overall_equal_dataset_summary.csv`. These files are derived exclusively from the fresh execution root. The older `run-modern-all` command is retained only for focused modern-method diagnostics and does not read or merge protected Proposed results.

## Seed-level result evidence

The release includes seed-level records for the controlled comparison, ablation, dataset-specific rank diagnostic, and drift analysis. Their protected aggregations are checked by `scripts/verify_main_results.py`. Any fresh rank execution is produced by `python main.py run-rank`, which uses the exact main Proposed constructor and evaluator; no separate rank controller is packaged or used.

## Data verification levels

- `python main.py verify-data`: verifies packaged study assets by shape, size, and SHA-256.
- `python main.py verify-canonical-data`: additional source-provenance check; public TweetEval requires `setup-data`.

The exact realized Synthetic GMM study stream is included as an immutable array. The reviewer-requested controlled representation-drift analysis reuses this same fixed stream and adds its generator configuration, fixed data seed, change points, evaluation seeds, and time-resolved measurements.

## Protected outputs

`results/main_results/` is protected. New runners reject output paths inside that tree and write to `results/execution_runs/` instead.
