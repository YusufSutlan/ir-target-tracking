"""Sweep target contrast (SNR) and plot detection recall and tracking MOTA.

Usage:
    python -m irtrack.experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import run_scenario
from .sim import Scenario

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"


def plot_example(result: dict) -> None:
    t_show = len(result["frames"]) // 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].imshow(result["frames"][t_show], cmap="inferno")
    axes[0].set_title(f"Raw IR frame #{t_show} (targets are barely visible)")

    axes[1].imshow(result["frames"][t_show], cmap="gray", alpha=0.5)
    paths: dict[int, list] = {}
    for est in result["tracks"][: t_show + 1]:
        for tid, pos in est.items():
            paths.setdefault(tid, []).append(pos)
    for tid, pts in paths.items():
        pts = np.array(pts)
        axes[1].plot(pts[:, 0], pts[:, 1], lw=1.8, label=f"track {tid}")
        axes[1].scatter(*pts[-1], s=40, edgecolor="w")
    for pos in result["truth"][t_show].values():
        axes[1].scatter(*pos, s=160, facecolors="none", edgecolors="lime", lw=1.5)
    axes[1].set_title("Tracks (lines) vs. ground truth (green circles)")
    axes[1].legend(fontsize=7, ncol=3, loc="lower right")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "tracking_example.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    amplitudes = [0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.30]
    seeds = range(5)
    rows = []
    for amp in amplitudes:
        recs = [run_scenario(Scenario(target_amplitude=amp, seed=s)) for s in seeds]
        row = {
            "amplitude": amp,
            "snr_db": 20 * np.log10(amp / 0.02),
            "recall": float(np.mean([r["detection"]["recall"] for r in recs])),
            "precision": float(np.mean([r["detection"]["precision"] for r in recs])),
            "mota": float(np.mean([r["tracking"]["mota"] for r in recs])),
            "id_switches": float(np.mean([r["tracking"]["id_switches"] for r in recs])),
        }
        rows.append(row)
        print(f"contrast {amp:.2f} (peak SNR {row['snr_db']:4.1f} dB): recall {row['recall']:.2f}  "
              f"precision {row['precision']:.2f}  MOTA {row['mota']:.2f}  IDsw {row['id_switches']:.1f}")  # fmt: skip

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    snr = [r["snr_db"] for r in rows]
    ax.plot(snr, [r["recall"] for r in rows], "o-", label="detection recall")
    ax.plot(snr, [r["precision"] for r in rows], "s-", label="detection precision")
    ax.plot(snr, [r["mota"] for r in rows], "^-", label="tracking MOTA")
    ax.set_xlabel("Target peak SNR (dB)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Performance vs. target contrast (5 seeds, 4 targets each)")
    fig.tight_layout()
    fig.savefig(FIGURES / "snr_sweep.png", dpi=130)
    plt.close(fig)

    plot_example(run_scenario(Scenario(target_amplitude=0.14, seed=1)))
    (ROOT / "reports" / "snr_sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
