# Round-2 matched verification update — 2026-08-17

## Purpose

In response to the editor’s consistency questions, the affected TweetEval and
Synthetic GMM Proposed/Fixed Rank entries were re-verified under a matched
ten-seed execution. The algorithmic core was not redesigned.

The release update aligns the manuscript-facing quality values, rank evidence,
Figure 3, verification logic, and public documentation with that matched
verification.

## Matched manuscript-facing values

| Dataset | Method | ARI | NMI | Mean rank ± SD | Range | Admission |
|---|---|---:|---:|---:|---:|---:|
| TweetEval | Proposed | -0.000542 ± 0.000424 | 0.001657 ± 0.000523 | 8.0000 ± 0.0000 | 8-8 | 57.36% |
| TweetEval | Fixed Rank | -0.000542 ± 0.000424 | 0.001657 ± 0.000523 | 8.0000 ± 0.0000 | 8-8 | matched control |
| Synthetic GMM | Proposed | 0.765705 ± 0.068506 | 0.779494 ± 0.060014 | 11.6468 ± 0.3060 | 8-18 | 56.22% |
| Synthetic GMM | Fixed Rank | 0.722984 ± 0.059987 | 0.736730 ± 0.056811 | 8.0000 ± 0.0000 | 8-8 | matched control |

The Table 10 SD is the between-run SD of the ten per-seed mean ranks; the rank
range is pooled across the ten runs.

## Quality-metric scope

For the matched TweetEval/Synthetic GMM entries used in Table 5 / Appendix B,
each seed-level ARI/NMI value is the arithmetic mean over the
post-initialization 1,000-observation evaluation chunks.

The audit runner also stores a separate complete-window final-partition
diagnostic. That diagnostic is not the dataset-level metric reported in the
manuscript. Release-facing audit files use explicit field/file names to prevent
the two quantities from being confused.

## Files updated by this patch

- multi-dataset quality tables;
- all-dataset descriptive comparison;
- rank summary/distribution/seed records for the affected datasets;
- Figure 3 and its build-label logic;
- release-facing editor audit summaries;
- result verification;
- result/documentation maps.

No intentional change is made to the ASC model equations, micro-cluster update,
or the already reported controlled 40-run and timing evidence.

## TweetEval representation scope

The public-source manuscript path uses a stateless 2,048-D hashed text
representation followed by model projection. The self-contained study-mode
release uses a fingerprinted 59,899 × 256 packaged study representation and
treats it as already projected at the projection interface.

The cached 256-D asset is verified as a fixed study artifact. Strict
public-source provenance reproduction uses the canonical preparation/execution
path.

## Applying the update safely

Use the package-level `APPLY_TO_REPO.py`. The helper:

1. verifies the update package itself;
2. backs up files that will be changed;
3. applies the overlay/removals;
4. runs semantic result checks **before** accepting new protected hashes;
5. regenerates the protected-result registry and release manifest;
6. runs final verification;
7. rolls back the touched files if a required step fails.

Review `git diff` before committing.

## Rank-scope clarification

Only TweetEval and Synthetic GMM rank rows were re-verified full-stream for the second-round consistency audit. CoverType/Electricity retain their earlier dataset-specific rank diagnostics and are not relabelled as new full-stream reruns. `results/main_results/rank_all_datasets/rank_evidence_scope.csv` records the exact observation totals and role of every row.

Rank release logs are rank-only; their previous generic ARI/NMI fields were removed so they cannot be confused with the manuscript quality metric.

The TweetEval public-source 2,048-D path and fixed packaged 256-D study representation are documented as separate evidence levels. The release verifies the cached 256-D artifact but does not claim an independently proven derivation chain from the canonical matrix.
