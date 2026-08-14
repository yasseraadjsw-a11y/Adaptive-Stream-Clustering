# Adaptive Sketch-Based Stream Clustering

Research code and verified result package for **Adaptive Sketch-Based Stream Clustering for High-Dimensional Data Streams under Representation Drift**.

## Quick verification

Python 3.11 is the supported environment.

```bash
python -m pip install -r requirements.txt
python main.py verify
```

`verify` compiles the Python sources, checks package health, checks the single study protocol, verifies packaged dataset fingerprints, checks result aggregation and consistency, verifies the protected-result fingerprints, verifies the two-way SHA-256 manifest, and runs the complete test suite.

## Result organization

The immutable numerical results used by the manuscript are stored under `results/main_results/`. New executions are written only to `results/execution_runs/`; released runners reject attempts to overwrite the protected study-result tree.

Main result groups include the multi-dataset benchmark, controlled comparison, ablation, rank diagnostic, drift analysis, statistics, sensitivity, and resource profiling. `docs/primary_execution_index.csv` records the original 200-combination five-method study queue. The fresh unified runner covers 280 combinations (4 datasets × 7 methods × 10 seeds).

## One authoritative configuration source

Primary dataset and baseline settings are stored only in `configs/datasets/`. The primary runner reads those files directly. The controlled representation-drift protocol is stored in `configs/controlled_representation_drift.json`.

Examples:

- CoverType / Primary Synthetic GMM: CluStream radius and DenStream epsilon = 1.50.
- Electricity: CluStream radius and DenStream epsilon = 1.00; DenStream beta = 0.20, mu = 6, lambda = 0.01.
- TweetEval: CluStream radius and DenStream epsilon = 0.85.
- StreamKM++: coreset = 500, merge buffer = 1000.

## Main commands

```bash
python main.py verify
python main.py verify-data
python main.py verify-canonical-data   # after public-source preparation
python main.py verify-manifest
python main.py rebuild-controlled-data

python main.py run-controlled
python main.py run-primary
python main.py run-rank
python main.py run-ablation
python main.py run-drift
python main.py run-sensitivity
python main.py profile-resources
```

`run-primary` is the authoritative fresh all-dataset path. It executes Proposed, Fixed Rank, CluStream, DenStream, StreamKM++, TWStream, and FRA-ART under one common loader/evaluator and then creates a completion manifest, seed-wise raw table, dataset-method summary, and equal-dataset overall summary from the fresh outputs only.

`run-modern-all` remains available as a modern-only diagnostic and never merges saved Proposed values.

`run-rank` constructs the same main Proposed model and evaluator and writes its new rank evidence to `results/execution_runs/rank_diagnostic/`; no separate rank controller is used.

Use the following command to inspect the complete 280-run matrix without executing it:

```bash
python main.py run-primary --dry-run
```

Use `--resume` to continue an interrupted fresh run.

```bash
python main.py run-study-suite --source study
```

This command launches the released execution paths sequentially and can be computationally expensive.

## Data

`data/study_assets_manifest.json` records the SHA-256 fingerprint, file size, and expected shape for each packaged study asset.

The package contains:

- The full CoverType study stream.
- The full Electricity study stream.
- The realized 9,000 × 256 Synthetic GMM stream used in the benchmark and controlled time-resolved analysis.
- The packaged TweetEval study representation.

For public-source TweetEval reconstruction, the acquisition and preparation code is pinned to the official repository commit and implements train → validation → test order, a stateless 2048-feature word 1–2 gram `HashingVectorizer`, `alternate_sign=False`, L2 normalization, and sparse CSR storage.

`verify-data` checks all packaged study assets out of the box.

`verify-canonical-data` performs the additional source-provenance check. For public TweetEval, it requires running `setup-data` first.

## Method implementations

CluStream, DenStream, StreamKM++, TWStream, and FRA-ART are the author study implementations described in the manuscript and `METHOD_SOURCES.md`.

A separate optional adapter is included for the pinned official TWStream Java implementation.

## Clean-machine launchers

- Linux/macOS: `./setup_and_verify.sh`
- Windows: `SETUP_AND_VERIFY_WINDOWS.bat`

For a complete repository and result map, see:

- `docs/GITHUB_TECHNICAL_GUIDE_AR.md`
- `REPRODUCIBILITY.md`
- `RESULTS.md`
- `METHOD_SOURCES.md`
- `docs/PRIMARY_EXECUTION_RECORD.md`
- `data/README.md`
