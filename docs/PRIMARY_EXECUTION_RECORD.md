# Primary study execution record

The primary benchmark execution index enumerates four datasets, five methods,
and the ten fixed seeds `{7,13,19,23,31,37,41,43,47,53}`, for 200 declared study
combinations. A broader project validation path covers seven methods and 280
combinations.

The protected release tables are stored under
`results/main_results/multidataset/`. In response to the editor’s second-round
consistency question, the TweetEval and Synthetic GMM Proposed/Fixed Rank
manuscript-facing entries were updated from a matched ten-seed verification and
are documented in `docs/ROUND2_CORRECTION_20260817.md`. After this intentional
release update, the protected-result registry is regenerated and the release
tables are treated as immutable evidence.

The later TWStream/FRA-ART descriptive extension is stored under
`results/main_results/modern_methods_extension/`.

Fresh full-stream executions are generated with `python main.py run-primary`,
written only under `results/execution_runs/`, and never overwrite the protected
release tables automatically.

## Round-2 scope note

The intentional second-round release correction is limited to the affected TweetEval/Synthetic GMM Proposed/Fixed Rank quality summaries and matched rank evidence, plus the documentation/verification required to make those changes internally consistent. It is not a redesign of the algorithm and does not retroactively label unaffected CoverType/Electricity rank diagnostics as newly re-executed full-stream runs.
