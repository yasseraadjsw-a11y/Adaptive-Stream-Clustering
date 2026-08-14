# Method and dataset sources

## Public datasets

- **CoverType** — UCI Machine Learning Repository, dataset 31, DOI `10.24432/C50K5N`.
- **Electricity** — OpenML dataset 151 (`electricity`, version 1).
- **TweetEval sentiment** — official `cardiffnlp/tweeteval` repository; train → validation → test order is retained by the preparation code.

Dataset acquisition/preparation utilities record source information and file fingerprints where available.

## Classical study baselines

The package contains study implementations of:

- **CluStream** — C. C. Aggarwal, J. Han, J. Wang, and P. S. Yu, “A Framework for Clustering Evolving Data Streams,” VLDB, 2003.
- **DenStream** — F. Cao, M. Ester, W. Qian, and A. Zhou, “Density-Based Clustering over an Evolving Data Stream with Noise,” SDM, 2006. DOI `10.1137/1.9781611972764.29`.
- **StreamKM++** — M. R. Ackermann et al., “StreamKM++: A Clustering Algorithm for Data Streams,” ALENEX, 2010. DOI `10.1137/1.9781611972900.16`.

The implementations are evaluated under the study's common quality boundary and use the parameter files distributed in `configs/`.

## Recent comparative methods

- **TWStream** — J. Sun, M. Du, Z. Lew, and Y. Dong, “TWStream: Three-Way Stream Clustering,” *IEEE Transactions on Fuzzy Systems*, 32(9), 4927–4939, 2024. DOI `10.1109/TFUZZ.2024.3369716`.
- **FRA-ART** — Y. Zhu, P. Li, Q. Zhang, Y. Zhu, and J. Yang, “Fractional Adaptive Resonance Theory (FRA-ART): An Extension for a Stream Clustering Method with Enhanced Data Representation,” *Mathematics*, 12(13), 2049, 2024. DOI `10.3390/math12132049`.

The package provides study implementations for the common experimental boundary. A separate optional adapter is included for the pinned official TWStream Java source at commit `4e084d1ce29f116fc9896ffb270640d8fb24348f`.

TweetEval acquisition/preparation is pinned to `cardiffnlp/tweeteval` commit `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`.

## Configuration

Study settings are stored in `configs/`. The code keeps projection, sketch, rank-control, micro-cluster and recent-method parameters explicit rather than relying on hidden constructor defaults. `src/asc_stream/method_registry.py` records implementation identity for executable methods.
