# Package contents

This practical release contains the study source code, one authoritative
dataset-specific protocol configuration, protected manuscript-facing numerical
sources, verified study assets, experiment/diagnostic runners, and verification
utilities.

Evidence includes:
- protected multi-dataset ARI/NMI tables and the original 200-combination
  execution index;
- the documented matched TweetEval/Synthetic GMM release update;
- 40 seed-level controlled-comparison records;
- 60 seed-level ablation records;
- 40 seed-level adaptive-rank diagnostic records;
- drift, sensitivity, statistics, timing, and resource evidence;
- SHA-256 fingerprints for packaged study assets and the release manifest.

Fresh executions are isolated under `results/execution_runs/` and cannot
overwrite `results/main_results/`. The unified fresh runner covers all seven
methods and creates seed-level and aggregate outputs solely from its new
execution tree.

The editor-round2 audit uses explicit metric names so that the manuscript
post-initialization chunk-mean metric is not confused with the separate
complete-window final-partition diagnostic.

Journal-submission documents are distributed separately from the practical
repository.

## Round-2 release additions

The package also contains:
- explicit matched TweetEval/Synthetic GMM quality audit summaries;
- an explicit rank-evidence scope table separating the round-2 full-stream rows from unaffected retained diagnostics;
- rank-only release logs with no generic ARI/NMI fields;
- a TweetEval provenance-boundary record for canonical 2,048-D versus packaged 256-D evidence;
- synchronized canonical and legacy Figure 3 outputs.
