# Fresh execution outputs

All new runs are written under this directory. The protected study-result files under
`results/main_results/` are never overwritten or used as inputs by the unified primary
runner. This directory is ignored by Git except for this README.

A complete `run-primary` execution creates one raw JSON file for each selected
dataset/method/seed combination, then derives its completion manifest and CSV summaries
from those new JSON files only. Do not manually copy fresh outputs into `main_results`.
