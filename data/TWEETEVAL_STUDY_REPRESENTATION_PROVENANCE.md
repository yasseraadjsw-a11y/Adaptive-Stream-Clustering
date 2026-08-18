# TweetEval study representation provenance boundary

The release intentionally distinguishes two evidence levels.

1. **Canonical public-source path.** Official TweetEval sentiment text is pinned to commit `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`, ordered train -> validation -> test, and converted by the declared stateless 2,048-D word 1-2 gram `HashingVectorizer` before model projection.
2. **Packaged study-mode asset.** `data/rank_validation/processed/tweeteval_sentiment.npz` is a fixed 59,899 x 256 projected study representation with SHA-256 `83c8b3c392ceae11560bab9a8f5e13c6272161787e789e62202c4c1a985c94b6`.

The package verifies the identity of the 256-D asset itself. The files currently distributed do **not** independently prove a byte-for-byte derivation chain from a reconstructed 2,048-D canonical matrix to that cached 256-D asset, and the release therefore makes no such claim.

For strict public-source provenance, use `python main.py setup-data --dataset tweeteval` followed by the canonical execution path. For the second-round matched consistency audit, Proposed and Fixed Rank are compared on the same fixed packaged study representation, so the equivalence check is internally matched even though it is a different evidence level from the public-source reconstruction path.
