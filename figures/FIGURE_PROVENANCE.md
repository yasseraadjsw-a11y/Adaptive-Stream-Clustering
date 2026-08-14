# Figure provenance and reconstruction

- **Fig. 1–2:** original conceptual diagram assets are retained verbatim under `figures/manuscript_canonical/`. They contain no experimental values.
- **Fig. 3:** rebuilt from `results/main_results/multidataset/quality_summary.csv`.
- **Fig. 4:** rebuilt from `results/main_results/controlled/method_summary.csv`, including seed-level standard deviations.
- **Fig. 5:** rebuilt from `results/main_results/ablation/summary.csv`, including seed-level standard deviations.
- **Fig. 6:** rebuilt from the actual seed-7 time trace under `results/main_results/drift/raw_traces/trace_seed_7.npz`. The same directory includes all ten raw traces and the fixed decimated CSV.

Run `python main.py build-figures` to rebuild the data-driven figures. Fresh drift traces can also be re-executed with `python main.py run-drift`; new traces are isolated under `results/execution_runs/drift/`.
