# All-dataset adaptive-rank statistics

This folder supplies the rank values shown in manuscript Table 10 while keeping the evidence scope explicit.

## Files

- `rank_summary.csv` — the four manuscript Table 10 rows.
- `rank_seedwise.csv` — rank-only seed-level summaries. Manuscript ARI/NMI are intentionally absent.
- `rank_distribution.csv` — empirical rank occupancy associated with the retained diagnostics.
- `rank_evidence_scope.csv` — observation coverage and evidence role for each dataset row.
- `rank_summary_full_internal.csv` — detailed aggregation metadata with explicit coverage/scope fields.
- `raw_logs/` — release-facing rank-only logs. They do not serve as Table 5/Appendix B quality sources.

## Second-round matched re-verification

The editor comment directly implicated TweetEval and Synthetic GMM. These two rows were re-verified with the current Proposed engine on the full packaged study stream under the ten fixed seeds.

- TweetEval: `8.0000 ± 0.0000`, range `8-8`, admission `57.36%`.
- Synthetic GMM: `11.6468 ± 0.3060`, range `8-18`, admission `56.22%`.

The SD is the between-run SD of the ten per-seed mean ranks; the range is pooled across the observations represented by that diagnostic.

## Unaffected rows

CoverType and Electricity were not implicated by the editor's rank/quality inconsistency. Their Table 10 values are retained from the previously released dataset-specific rank diagnostic rather than being presented as new round-2 full-stream reruns. The exact retained observation totals are recorded in `rank_evidence_scope.csv`.

This scope separation is deliberate: the release does not infer missing rank observations or manufacture a full-stream result for the unaffected rows.
