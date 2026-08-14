# Final results map

All numerical files in this project belong to actual study executions. Some directory labels changed during revision; the map below records their scientific relationship.

| Evaluation stage | Numerical source | Role |
|---|---|---|
| Original all-dataset experiment | `results/main_results/multidataset/` | Unchanged Proposed, Fixed Rank, CluStream, DenStream, and StreamKM++ results on four datasets |
| Modern-method extension | `results/main_results/modern_methods_extension/` | TWStream and FRA-ART executed later on the same datasets, environment, preprocessing, coverage, evaluator, and metrics |
| Unified modern comparison | `results/main_results/all_dataset_direct_comparison.csv` | Original Proposed rows plus the newly added TWStream/FRA-ART rows; no original value is replaced |
| Controlled representation drift | `results/main_results/controlled/` | Proposed, Fixed Rank, TWStream, and FRA-ART on the reviewer-requested matched controlled stream |
| Ablation | `results/main_results/ablation/` | Sixty seed-level executions |
| Rank analysis | `results/main_results/rank_all_datasets/` | Forty dataset-specific rank records requested during review |
| Drift response | `results/main_results/drift/` | Time-resolved adaptation and recovery results |
| Statistics, sensitivity, and resources | Corresponding folders under `results/main_results/` | Additional reviewer-requested analyses |

The original 200-combination execution index is stored in `docs/primary_execution_index.csv`. The modern-method extension adds TWStream and FRA-ART without changing the earlier experiment. Any fresh re-execution uses the separate unified 280-run path and writes only under `results/execution_runs/primary/`; it never replaces these protected values automatically. `python main.py verify` checks integrity and numerical consistency.
