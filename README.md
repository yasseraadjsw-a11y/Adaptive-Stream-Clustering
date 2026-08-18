# Adaptive Sketch-Based Stream Clustering

Research code and verified result package for **Adaptive Sketch-Based Stream
Clustering for High-Dimensional Data Streams under Representation Drift**.

## Quick verification

Python 3.11 is the supported environment.

```bash
python -m pip install -r requirements.txt
python main.py verify
```

`verify` checks package health, the declared study protocol, packaged data
fingerprints, release-result consistency, protected-result fingerprints, the
release manifest, Python compilation, and the test suite.

## Result organization

- `results/main_results/`: protected manuscript-facing release evidence.
- `results/execution_runs/`: destination for any fresh execution.
- `results/editor_round2_audit/`: release-facing audit summaries for the
  second-round consistency verification.

The protected release evidence includes the documented matched
TweetEval/Synthetic GMM update described in
`docs/ROUND2_CORRECTION_20260817.md`.

## One authoritative configuration source

Primary dataset/baseline settings are stored in `configs/datasets/`.
Controlled-drift settings are stored in
`configs/controlled_representation_drift.json`.

## Main commands

```bash
python main.py verify
python main.py verify-data
python main.py verify-canonical-data
python main.py rebuild-controlled-data
python main.py run-controlled
python main.py run-primary
python main.py run-rank
python main.py run-ablation
python main.py run-drift
python main.py run-sensitivity
python main.py profile-resources
python main.py build-figures
```

`run-primary` is the authoritative fresh all-dataset path. It executes Proposed,
Fixed Rank, CluStream, DenStream, StreamKM++, TWStream, and FRA-ART over the
declared datasets/seeds and writes only to `results/execution_runs/`.

## Dataset-level matched metric

For the matched TweetEval/Synthetic GMM entries used by Table 5 / Appendix B,
each seed-level ARI/NMI value is the arithmetic mean over the
post-initialization 1,000-observation evaluation chunks. The complete-window
final partition is a separate audit diagnostic.

## TweetEval evidence levels

The manuscript public-source path builds a stateless 2,048-D sparse hashed text
representation and then applies model projection. The self-contained release
also includes a SHA-256-verified 59,899 × 256 packaged study representation used
by study-mode audit/re-execution tooling.

The 256-D asset is verified as a fixed study artifact. Strict public-source
provenance reproduction uses `setup-data` / `--source canonical`.

## Data

The release includes full packaged CoverType and Electricity study streams, the
fixed 9,000 × 256 Synthetic GMM realization, and the packaged TweetEval study
representation. Asset fingerprints are stored in
`data/study_assets_manifest.json`.

## Method implementations

CluStream, DenStream, StreamKM++, TWStream, and FRA-ART are study
implementations described in `METHOD_SOURCES.md`. A separate optional adapter is
provided for the pinned official TWStream implementation.

## Further documentation

- `RESULTS.md`
- `REPRODUCIBILITY.md`
- `EVIDENCE_SCOPE.md`
- `docs/PRIMARY_EXECUTION_RECORD.md`
- `docs/ROUND2_CORRECTION_20260817.md`
- `docs/GITHUB_TECHNICAL_GUIDE_AR.md`

## Rank-evidence scope

The editor's second-round matched re-verification directly affected TweetEval and Synthetic GMM. Their rank rows are full-stream matched audit results. CoverType and Electricity retain the previously released dataset-specific rank diagnostic because those rows were not implicated by the editor comment. The exact diagnostic coverage and role of every row are recorded in `results/main_results/rank_all_datasets/rank_evidence_scope.csv`; the release does not describe the four rows as one newly re-run full-stream experiment.

Rank-diagnostic files intentionally contain no manuscript ARI/NMI fields. Dataset-level manuscript quality comes from `results/main_results/multidataset/` and the matched audit files under `results/editor_round2_audit/`.

The TweetEval 2,048-D versus packaged 256-D evidence boundary is documented in `data/TWEETEVAL_STUDY_REPRESENTATION_PROVENANCE.md` and `.json` without asserting an unverified derivation chain.
