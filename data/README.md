# Data

## Packaged study assets

The self-contained release includes the complete packaged study arrays required
by study-mode execution. Their SHA-256 digests, sizes, and expected shapes are
recorded in:

`data/study_assets_manifest.json`

Verify them with:

```bash
python main.py verify-data
```

## Public datasets

The public-source acquisition/preparation workflow is:

```bash
python main.py setup-data
```

Declared sources:
- CoverType — UCI Machine Learning Repository.
- Electricity — OpenML dataset 151.
- TweetEval sentiment — official Cardiff NLP repository pinned by the study
  code to a fixed commit.

TweetEval public-source preparation uses train → validation → test order and a
stateless 2,048-dimensional word 1–2 gram HashingVectorizer stored in CSR form.

## TweetEval packaged study representation

For self-contained study-mode execution, the release also includes:

`data/rank_validation/processed/tweeteval_sentiment.npz`

This fixed asset contains 59,899 × 256 features and is fingerprinted in
`data/study_assets_manifest.json`. Study-mode execution treats it as an already
projected representation at the model projection interface.

The 256-D packaged asset and the canonical 2,048-D public-source representation
are therefore two explicitly distinguished evidence levels. The package
verifies the identity of the 256-D asset itself; strict public-source provenance
reproduction uses `setup-data` followed by the canonical execution path.

## Controlled representation-drift stream

`data/controlled/` contains the fixed 9,000 × 256 representation-drift stream
and its causally standardized form. The data seed is 2026 and the declared
change points are t=3000 and t=6000. The Synthetic GMM benchmark and the
controlled time-resolved analysis use the same fixed realization.

Rebuilt files are written under `results/execution_runs/controlled_data/`; fixed
study files are never overwritten.

## Rank-study assets

`data/rank_validation/processed/` contains the packaged arrays used by the study
tooling. New rank-diagnostic executions are written under
`results/execution_runs/rank_diagnostic/`.

## Provenance boundary file

`TWEETEVAL_STUDY_REPRESENTATION_PROVENANCE.md` and `.json` record the exact evidence boundary between the canonical 2,048-D public-source preparation and the SHA-256-verified 256-D packaged study representation. The current release verifies the cached study artifact itself but does not claim that the distributed files independently prove its derivation from the canonical matrix.
