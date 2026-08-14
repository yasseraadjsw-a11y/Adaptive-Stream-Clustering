# Primary study execution record

The primary benchmark uses four datasets, five methods, and the ten fixed seeds `{7,13,19,23,31,37,41,43,47,53}`, for 200 study runs. The preserved original execution logs record the complete queue for these 200 combinations. A broader validation run from the same project recorded 280/280 completed and valid runs when two additional comparison methods were included.

The immutable results of the original executions are stored under `results/main_results/multidataset/`. The later TWStream and FRA-ART executions are stored under `results/main_results/modern_methods_extension/`; they use the same all-dataset conditions and extend rather than replace the original results. Fresh full-stream executions can be generated with `python main.py run-primary`; they are written only under `results/execution_runs/` and cannot overwrite the protected study results.

This separation preserves the actual study record and prevents later executions from changing the manuscript-facing results.
