# Experimental evidence scope

All reported values are results of actual study or verification executions.
Directory names are organizational and do not, by themselves, define
experimental provenance.

- Proposed, Fixed Rank, CluStream, DenStream, and StreamKM++ are represented in
  the protected multi-dataset release tables.
- TWStream and FRA-ART were additionally evaluated on the four study datasets
  for complementary descriptive context.
- Proposed, Fixed Rank, TWStream, and FRA-ART were evaluated over the same ten
  seeds in the controlled representation-drift comparison.
- Controlled and ablation folders retain seed-level evidence.
- The rank folder records the dataset-specific adaptive-rank analysis requested
  during review.
- The editor-round2 audit directory documents the matched TweetEval/Synthetic
  metric and separates it from the complete-window final-partition diagnostic.

## TweetEval evidence boundary

The public-source preparation produces a 2,048-D sparse hashed representation.
The self-contained package includes a separately fingerprinted 59,899 × 256
study representation used by study-mode execution. The packaged 256-D asset is
verified as a fixed study artifact; this repository does not infer a
cryptographic derivation from the 2,048-D source representation unless that
derivation is explicitly reproduced through the canonical data path.

Verification utilities check integrity and cross-file numerical consistency.

## Second-round scope safeguards

The second-round matched re-verification changes only the affected TweetEval/Synthetic GMM manuscript-facing Proposed/Fixed Rank summaries and matched rank evidence. CoverType/Electricity rank rows retain their previously released dataset-specific diagnostic scope, recorded explicitly in `results/main_results/rank_all_datasets/rank_evidence_scope.csv`. Rank release files do not serve as manuscript ARI/NMI sources.

TweetEval canonical 2,048-D preparation and the fixed SHA-256-verified 256-D packaged study representation are separate evidence levels. The release does not claim a derivation chain that the distributed files do not prove.
