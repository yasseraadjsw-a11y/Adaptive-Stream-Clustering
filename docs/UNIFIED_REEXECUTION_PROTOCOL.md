# Unified seven-method re-execution protocol

The authoritative fresh-execution entry point is:

```text
python main.py run-primary
```

It schedules the Cartesian product of:

- four datasets: CoverType, Electricity, TweetEval, and Synthetic GMM;
- seven methods: Proposed, Fixed Rank, CluStream, DenStream, StreamKM++,
  TWStream, and FRA-ART;
- ten fixed seeds: 7, 13, 19, 23, 31, 37, 41, 43, 47, and 53.

This produces 280 fresh executions. Each output is written under
`results/execution_runs/primary/`. The runner cannot write into
`results/main_results/`, and it does not read manuscript result tables when
constructing or evaluating a run.

After the last successful execution, the command validates the complete matrix
and creates, from those fresh JSON files only:

- `execution_manifest.json`;
- `seedwise_results.csv`;
- `dataset_method_summary.csv`;
- `overall_equal_dataset_summary.csv`.

Interrupted work can be continued with `--resume`. Existing fresh outputs are
never overwritten by default; `--force` is required to replace them. The
`--dry-run` option prints the complete matrix without executing an experiment.

All methods receive the same dataset realization, observation order, seed,
common dataset preprocessing, and complete-window evaluator. FRA-ART applies
its required bounded input transformation after common preprocessing; the
bounds are fixed from the unlabeled 1,000-observation calibration prefix.
TWStream's radius is likewise determined from that unlabeled prefix. These
method-required transforms and their resolved values are recorded in every raw
JSON output.

The optional `--max-observations` argument is for smoke testing only. Such an
output is marked `full_stream=false` and cannot be used as manuscript evidence.

## Acceptance boundary

Fresh outputs are independent re-execution evidence and do not alter the protected original study record.
No seed, feature, row, time window, or configuration may be selected because it
is closer to a manuscript value. Numerical agreement is assessed only after the
protocol and code are frozen. A discrepancy must be reported; it must not be
rescaled or hidden by lower-precision truncation.
