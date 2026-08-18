# Editor round-2 audit evidence

This directory contains the release-facing audit summaries used to support the
second-round response.

## Dataset-level matched metric

`matched_seedwise_results.csv` stores two explicitly named quality quantities:

- `online_chunk_mean_ari` / `online_chunk_mean_nmi`: the manuscript-facing
  dataset-level metric for the matched TweetEval/Synthetic GMM re-verification.
  The first 1,000 observations are initialization; the value for each seed is
  the mean over the later 1,000-observation evaluation chunks.
- `complete_window_final_ari` / `complete_window_final_nmi`: a separate final
  complete-window partition diagnostic. It is retained for auditability but is
  not the Table 5 / Appendix B dataset-level metric.

`matched_summary.csv` summarizes both quantities across the ten fixed seeds.

The intentionally ambiguous legacy filenames `FULL_QUALITY_*` are not published
in this release-facing directory; the complete-window diagnostic remains
available inside the separately labelled evidence ZIP
`Editor_Comment1_Matched_Rerun_20260816.zip`.

## Other audit blocks

- `comment3_*`: timing aggregation/boundary verification.
- `comment4_*`: controlled recent-method repeated comparison and paired tests.
- `round2_audit_summary.json`: compact audit map.

## Evidence boundary safeguards

The matched TweetEval/Synthetic GMM quality audit and the rank folder are separate evidence products. Rank release logs are rank-only and must not be used as Table 5 / Appendix B quality values.

The TweetEval matched consistency audit uses the same fixed packaged 256-D study representation for Proposed and Fixed Rank. The repository separately supports canonical 2,048-D public-source preparation. See `data/TWEETEVAL_STUDY_REPRESENTATION_PROVENANCE.md`; no unverified 2,048→256 derivation claim is made.
