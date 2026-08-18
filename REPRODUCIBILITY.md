# Reproducibility and verification

## Fixed study seeds

`7, 13, 19, 23, 31, 37, 41, 43, 47, 53`.

## Verification gate

Run:

```bash
python main.py verify
```

The gate performs package-health checks, protocol checks, packaged-data SHA-256
checks, result aggregation/consistency checks, protected-result fingerprint
checks, release-manifest checks, Python compilation, and unit tests.

## Study configuration

Primary settings have one source of truth: `configs/datasets/*.json`.
Controlled-drift settings have one source:
`configs/controlled_representation_drift.json`.

The sketch-basis refresh interval is 100 accepted sketch rows. The retained
energy is 0.95. The micro-cluster fading factor is independently 0.95.

## Protected release results

`results/main_results/` contains the manuscript-facing release evidence. The
matched TweetEval/Synthetic GMM update performed for the second-round
consistency check is documented in `docs/ROUND2_CORRECTION_20260817.md`. After
an intentional release-evidence update, the protected-result registry and the
repository manifest must be regenerated.

Fresh full-stream executions use the unified 280-run matrix
(4 datasets × 7 methods × 10 seeds) and are written only under
`results/execution_runs/primary/`.

## Dataset-level matched metric

For the matched TweetEval/Synthetic GMM entries used in Table 5 / Appendix B,
the seed-level ARI/NMI value is the mean over the post-initialization
1,000-observation evaluation chunks. The first 1,000 observations are
initialization. The complete-window final partition is retained separately only
as an audit diagnostic.

## Seed-level evidence

The release includes seed-level records for the controlled comparison,
ablation, dataset-specific rank diagnostic, and drift analysis. Fresh rank
execution uses the same main Proposed constructor/evaluator as `run-primary`.

## TweetEval data verification levels

- `python main.py verify-data`: verifies packaged study assets by shape, size,
  and SHA-256.
- `python main.py verify-canonical-data`: performs the stricter public-source
  provenance check after `setup-data`.

The manuscript public-source TweetEval path uses a 2,048-D sparse hashed
representation. The self-contained study-mode release uses a SHA-256-verified
59,899 × 256 packaged study representation. These are deliberately identified
as different evidence levels; strict public-source reproduction uses the
canonical path.

## Synthetic GMM

The exact realized 9,000 × 256 Synthetic GMM stream is included as an immutable
study array. The controlled representation-drift analysis reuses that fixed
stream and adds time-resolved measurements and comparison methods.

## Protected outputs

New runners reject output paths inside `results/main_results/` and write to
`results/execution_runs/` instead.

## Rank diagnostic coverage

The rank release combines two clearly labelled evidence roles rather than pretending they are one new full-stream rerun. TweetEval and Synthetic GMM are the full-stream matched re-verification used to answer the editor. CoverType and Electricity retain the earlier dataset-specific rank diagnostic, which was not part of that re-verification. Exact observation totals and coverage are stored in `results/main_results/rank_all_datasets/rank_evidence_scope.csv`.

No ARI/NMI values in the rank folder are used as manuscript quality evidence. The rank logs are rank-only by design.

See `data/TWEETEVAL_STUDY_REPRESENTATION_PROVENANCE.md` for the explicit 2,048-D canonical / 256-D packaged evidence boundary.
