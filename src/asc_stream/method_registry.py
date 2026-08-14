from __future__ import annotations

from copy import deepcopy

# Machine-readable implementation identities. These labels distinguish the
# implementations executed in the common study protocol from the optional
# external official TWStream adapter; they do not define different result protocols.
_METHODS = {
    "proposed": {
        "implementation_id": "asc_project_native_eq7",
        "fidelity": "project_native",
        "native_output": True,
        "scope": "Adaptive sparse projection + Eq.(7) ridge-leverage sketch admission + adaptive-rank controller + bounded micro-clustering.",
        "limitations": [],
    },
    "fixed_rank": {
        "implementation_id": "asc_project_native_fixed_rank8",
        "fidelity": "project_native_ablation",
        "native_output": True,
        "scope": "Same project pipeline as Proposed with rank fixed at 8.",
        "limitations": [],
    },
    "clustream": {
        "implementation_id": "clustream_study_implementation",
        "fidelity": "study_implementation",
        "native_output": False,
        "scope": "Bounded CluStream-style online micro-cluster summaries under the study's common weighted macro evaluator.",
        "limitations": ["Implements the study configuration and common macro-clustering evaluation described in Table 6."],
    },
    "denstream": {
        "implementation_id": "denstream_study_implementation",
        "fidelity": "study_implementation",
        "native_output": False,
        "scope": "Potential/outlier micro-clusters, exponential fading, promotion and pruning; common weighted macro evaluator.",
        "limitations": ["Final labels use the common weighted macro-clustering evaluator declared by the study."],
    },
    "streamkmpp": {
        "implementation_id": "streamkmpp_study_implementation",
        "fidelity": "study_implementation",
        "native_output": False,
        "scope": "Bounded weighted D2-biased representative coreset reduction with the declared buffer/coreset sizes and common weighted macro evaluator.",
        "limitations": ["Uses the declared bounded D2-weighted coreset configuration and common macro evaluator."],
    },
    "twstream": {
        "implementation_id": "twstream_study_implementation",
        "fidelity": "study_implementation",
        "native_output": False,
        "scope": "TWStream-aligned active/outlier summaries, decay, augmented-kNN state and boundary confidence under a common weighted macro evaluator.",
        "limitations": ["The common-protocol study implementation is used for the reported comparison; an optional pinned official Java adapter is provided separately."],
    },
    "twstream_official": {
        "implementation_id": "twstream_official_java_adapter",
        "fidelity": "official_native_adapter",
        "native_output": True,
        "source_repository": "https://github.com/Du-Team/TWStream.git",
        "source_commit": "4e084d1ce29f116fc9896ffb270640d8fb24348f",
        "scope": "Pinned authors' Java TWStream implementation including native three-way clustering; unassigned/outlier points retain label -1 and coverage is reported.",
        "limitations": ["Requires the external JDK/Maven toolchain and the pinned source checkout."],
    },
    "fra_art": {
        "implementation_id": "fraart_study_implementation",
        "fidelity": "study_implementation",
        "native_output": False,
        "scope": "Declared SIBF fractional transform + complement coding + Fuzzy-ART category learning under the study common weighted macro evaluator.",
        "limitations": ["Implements the declared SIBF/Fuzzy-ART equations and common macro evaluator."],
    },
}


def method_metadata(method: str) -> dict:
    key = "fra_art" if method in {"fraart", "fra-art"} else method
    if key not in _METHODS:
        raise KeyError(f"Unknown method registry key: {method!r}")
    return deepcopy(_METHODS[key])


def all_method_metadata() -> dict[str, dict]:
    return {k: deepcopy(v) for k, v in _METHODS.items()}
