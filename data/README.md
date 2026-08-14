# Data

## Packaged study assets

The release includes the complete packaged study arrays required for self-contained execution. Their SHA-256 digests, sizes and expected shapes are recorded in:

```text
data/study_assets_manifest.json
```

Verify them with:

```bash
python main.py data-status
```

## Public datasets

The public-source acquisition/preparation workflow is also included:

```bash
python main.py setup-data
```

Declared sources:

- CoverType — UCI Machine Learning Repository, dataset 31.
- Electricity — OpenML dataset 151.
- TweetEval sentiment — official Cardiff NLP repository pinned to commit `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`.

Downloaded files are recorded in `data/public/raw/acquisition_manifest.json`. Prepared files are written under `data/public/processed/` with a preprocessing manifest. TweetEval uses CSR storage for the 2048-dimensional hashed representation.

`python main.py verify-data` verifies the packaged study assets and succeeds out of the box. Use `python main.py verify-canonical-data` after `setup-data` for the additional public-source provenance check.

## Controlled representation-drift stream

`data/controlled/` contains the fixed 9,000×256 representation-drift stream and its causally standardized form. The data seed is 2026 and the declared change points are t=3000 and t=6000. The original Synthetic GMM benchmark uses the same fixed realized stream; the reviewer-requested controlled analysis adds time-resolved measurements and comparison methods.

Rebuild it from the declared generator, report byte identity when the environment reproduces it, and verify machine-precision numerical equivalence across supported numerical-library builds:

```bash
python main.py rebuild-controlled-data
```

The rebuilt files are written under `results/execution_runs/controlled_data/`; the fixed study files are never overwritten.

## Rank-study assets

`data/rank_validation/processed/` contains the full packaged arrays used by the study tooling. Their integrity is checked through `data/study_assets_manifest.json`. New rank-diagnostic executions are written separately under `results/execution_runs/rank_diagnostic/`.


Data checks:
- `python main.py verify-data` — packaged study assets (self-contained).
- `python main.py verify-canonical-data` — strict public-source provenance after `setup-data`.
