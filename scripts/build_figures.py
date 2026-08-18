from __future__ import annotations

"""Rebuild the four data-driven figures and verify the two conceptual assets."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "figures" / "manuscript_canonical"


def build_fig3(out: Path) -> None:
    d = pd.read_csv(ROOT / "results/main_results/multidataset/quality_summary.csv")
    x = np.arange(len(d)); width = 0.36
    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    ax.bar(x - width / 2, d["ari_mean"], width, label="ARI")
    ax.bar(x + width / 2, d["nmi_mean"], width, label="NMI")
    labels = [str(v) for v in d["method"]]
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 0.75); ax.set_ylabel("Descriptive mean score"); ax.legend(loc="upper left")
    fig.tight_layout()
    canonical = out / "Fig3_multidataset_quality.png"
    fig.savefig(canonical, dpi=300)
    # Keep the historical top-level figure path synchronized with the canonical manuscript figure.
    if out.resolve() == DEFAULT_OUT.resolve():
        legacy = ROOT / "figures" / "multidataset_quality.png"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(legacy, dpi=300)
    plt.close(fig)


def build_fig4(out: Path) -> None:
    d = pd.read_csv(ROOT / "results/main_results/controlled/method_summary.csv")
    x = np.arange(len(d)); width = 0.35
    labels = ["Proposed", "Fixed Rank", "TWStream", "FRA-ART"]
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.bar(x - width / 2, d["ari_mean"], width, yerr=d["ari_sd"], capsize=4, label="ARI")
    ax.bar(x + width / 2, d["nmi_mean"], width, yerr=d["nmi_sd"], capsize=4, label="NMI")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0.55, 1.0); ax.set_ylabel("Score"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "Fig4_controlled_recent_comparison.png", dpi=220); plt.close(fig)


def build_fig5(out: Path) -> None:
    d = pd.read_csv(ROOT / "results/main_results/ablation/summary.csv")
    x = np.arange(len(d)); width = 0.36
    labels = ["Proposed", "Fixed Rank", "Uniform", "Keep All", "Dense Projection", "Leverage Weighting"]
    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    ax.bar(x - width / 2, d["ari_mean"], width, yerr=d["ari_sd"], capsize=4, label="ARI")
    ax.bar(x + width / 2, d["nmi_mean"], width, yerr=d["nmi_sd"], capsize=4, label="NMI")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Mean controlled score"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "Fig5_ablation_analysis.png", dpi=300); plt.close(fig)


def build_fig6(out: Path) -> None:
    trace = np.load(ROOT / "results/main_results/drift/raw_traces/trace_seed_7.npz", allow_pickle=False)
    time, error, threshold, rank = (trace[k] for k in ("time", "reconstruction_error", "threshold", "rank"))
    keep = set(range(0, len(time), 25))
    keep.update(int(i) for i in np.flatnonzero(np.r_[True, rank[1:] != rank[:-1]]))
    idx = np.asarray(sorted(keep), dtype=int)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(time[idx], error[idx], label="Reconstruction error")
    ax.plot(time[idx], threshold[idx], label="Smoothed threshold")
    for drift in (3000, 6000):
        ax.axvline(drift, linestyle="--", label="Declared drift" if drift == 3000 else None)
    ax.set_xlabel("Stream time"); ax.set_ylabel("Sketch-space error")
    ax2 = ax.twinx(); ax2.step(time[idx], rank[idx], where="post", linestyle="-.", label="Adaptive rank"); ax2.set_ylabel("Rank")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "Fig6_drift_response.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    # Figures 1 and 2 are original conceptual diagram assets, not numerical plots.
    for name in ("Fig1_proposed_model_diagram.png", "Fig2_adaptive_rank_control.png"):
        source = DEFAULT_OUT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if out != DEFAULT_OUT:
            (out / name).write_bytes(source.read_bytes())
    build_fig3(out); build_fig4(out); build_fig5(out); build_fig6(out)
    required = [f"Fig{i}_" for i in range(1, 7)]
    files = [p.name for p in out.glob("Fig*.png")]
    missing = [prefix for prefix in required if not any(name.startswith(prefix) for name in files)]
    if missing:
        raise FileNotFoundError(f"Missing figure prefixes: {missing}")
    print(f"Six manuscript figures available in {out}")


if __name__ == "__main__":
    main()
