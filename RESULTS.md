# Final results map

All manuscript-facing numerical values in this release are tied to identifiable
study or verification outputs. In response to the editor’s second-round
consistency questions, TweetEval and Synthetic GMM were re-verified under a
matched ten-seed execution, and the affected manuscript-facing entries were
updated to the matched values.

| Evaluation stage | Numerical source | Role |
|---|---|---|
| Multi-dataset descriptive table | `results/main_results/multidataset/` | Classical rows plus the release Proposed/Fixed Rank dataset-level summaries |
| Modern-method extension | `results/main_results/modern_methods_extension/` | TWStream and FRA-ART complete-window descriptive values |
| All-dataset descriptive comparison | `results/main_results/all_dataset_direct_comparison.csv` | Complementary descriptive context; not the repeated-seed inferential comparison |
| Controlled representation drift | `results/main_results/controlled/` | Proposed, Fixed Rank, TWStream, and FRA-ART over the same ten seeds |
| Ablation | `results/main_results/ablation/` | Sixty seed-level executions |
| Rank analysis | `results/main_results/rank_all_datasets/` | Dataset-specific adaptive-rank evidence over ten seeds |
| Editor round-2 audit | `results/editor_round2_audit/` | Matched dataset-level verification plus timing/statistical checks |
| Drift response | `results/main_results/drift/` | Time-resolved adaptation and recovery evidence |
| Statistics, sensitivity, resources | Corresponding folders under `results/main_results/` | Supporting analyses |

## Manuscript-facing matched values

- TweetEval Proposed = Fixed Rank:
  ARI `-0.000542 ± 0.000424`, NMI `0.001657 ± 0.000523`,
  rank `8.0000 ± 0.0000`, range `8-8`, Proposed admission `57.36%`.
- Synthetic GMM Proposed:
  ARI `0.765705 ± 0.068506`, NMI `0.779494 ± 0.060014`,
  rank `11.6468 ± 0.3060`, range `8-18`, admission `56.22%`.
- Synthetic GMM Fixed Rank:
  ARI `0.722984 ± 0.059987`, NMI `0.736730 ± 0.056811`,
  rank fixed at 8.

For these matched TweetEval/Synthetic GMM dataset-level entries, each seed-level
ARI/NMI value is the arithmetic mean over the post-initialization
1,000-observation evaluation chunks. A separately stored complete-window final
partition is an audit diagnostic and is not the Table 5 / Appendix B metric.

## TweetEval representation boundary

The manuscript’s public-source path constructs a stateless 2,048-dimensional
hashed text representation and then applies the model projection. The
self-contained release also contains a SHA-256-verified 59,899 × 256 packaged
study representation used by study-mode audit/re-execution tooling. The release
does not claim that the cached 256-D asset can be cryptographically regenerated
from the 2,048-D public-source representation using only the cached asset itself;
strict public-source reproduction uses the canonical preparation path.

Fresh executions write to `results/execution_runs/` and do not overwrite the
protected release evidence automatically.

## Release-scope safeguards

- Table 5 / Appendix B quality and rank diagnostics are separate evidence products. Rank files do not carry manuscript ARI/NMI fields.
- TweetEval/Synthetic GMM rank rows are the full-stream matched second-round audit. CoverType/Electricity rank rows are retained from the earlier dataset-specific diagnostic; exact coverage is listed in `rank_evidence_scope.csv`.
- `quality_summary.csv` is a descriptive macro summary. Its compatibility `ari_std`/`nmi_std` fields are arithmetic means of dataset-level SDs and are not pooled uncertainty or inferential statistics; see `quality_summary_scope.json`.
- TweetEval canonical 2,048-D preparation and the fixed 256-D packaged study representation are separate evidence levels; the release does not claim an unverified derivation chain.
