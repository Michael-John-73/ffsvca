"""
gen_figures.py
Generate Figures 1-3 for PAPER3 (JISA submission).

Figure 1: Fixed-FPR Protocol Diagram (pipeline overview)
Figure 2: z-score distribution shift under attacks (E3 output)
Figure 3: Calibration sample size sensitivity (E4 output)

Output: outputs_p3/figures/fig1_protocol.pdf/png
        outputs_p3/figures/fig2_score_dist.pdf/png
        outputs_p3/figures/fig3_cal_sensitivity.pdf/png

Usage:
    python gen_figures.py --config robin_config.json --figure all
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUTS_ROOT = Path("../PAPER3/outputs_p3")
METRICS_DIR = OUTPUTS_ROOT / "metrics"
FIGURES_DIR = OUTPUTS_ROOT / "figures"

# JISA-compatible style
FIGSIZE_SINGLE = (7, 4.5)
FIGSIZE_WIDE = (10, 4.5)
DPI = 300
FONT_SIZE = 11


# ---------------------------------------------------------------------------
# Figure 2: z-score distribution shift
# ---------------------------------------------------------------------------
def gen_fig2(outputs: Path) -> None:
    """Boxplot of z-score distributions per attack (clean vs watermarked)."""
    df = pd.read_csv(outputs / "metrics" / "e3_score_dist.csv")
    figs_dir = outputs / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    attacks = df["attack"].unique()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

    for ax, label_val, title in zip(axes, [0, 1], ["Clean (label=0)", "Watermarked (label=1)"]):
        sub = df[df["label"] == label_val].set_index("attack")
        data = {a: [sub.loc[a, "q25"], sub.loc[a, "q50"], sub.loc[a, "q75"]] for a in attacks if a in sub.index}
        ax.boxplot(
            [[sub.loc[a, "q05"], sub.loc[a, "q25"], sub.loc[a, "q50"], sub.loc[a, "q75"], sub.loc[a, "q95"]]
             for a in attacks if a in sub.index],
            labels=[a for a in attacks if a in sub.index],
            vert=True,
            patch_artist=True,
        )
        ax.set_title(title, fontsize=FONT_SIZE)
        ax.set_xlabel("Attack", fontsize=FONT_SIZE - 1)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylabel("z-score", fontsize=FONT_SIZE - 1)

    fig.suptitle("Figure 2: z-score Distribution Shift under Attacks", fontsize=FONT_SIZE + 1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figs_dir / f"fig2_score_dist.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[fig2] Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Calibration sample size sensitivity
# ---------------------------------------------------------------------------
def gen_fig3(outputs: Path) -> None:
    """Line plot: threshold tau vs calibration size n_cal, per method."""
    df = pd.read_csv(outputs / "metrics" / "e4_cal_sensitivity.csv")
    figs_dir = outputs / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    primary_alpha = df["alpha"].min()
    df = df[df["alpha"] == primary_alpha]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for method, grp in df.groupby("method"):
        ax.plot(grp["n_cal"], grp["tau"], marker="o", label=method)

    ax.set_xlabel("Calibration Set Size", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$\tau_\alpha$ Threshold", fontsize=FONT_SIZE)
    ax.set_title(f"Figure 3: Threshold Stability vs Calibration Size (α={primary_alpha:.2f})", fontsize=FONT_SIZE + 1)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figs_dir / f"fig3_cal_sensitivity.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[fig3] Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Protocol overview (simple text/box diagram via matplotlib)
# ---------------------------------------------------------------------------
def gen_fig1(outputs: Path) -> None:
    """
    Schematic protocol overview diagram (Figure 1).
    Rendered as a simple flowchart using matplotlib patches.
    """
    figs_dir = outputs / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis("off")

    boxes = [
        (0.05, "Generate\nImages\n(seed=0)"),
        (0.22, "ROBIN\nScore Dump\nz(x)=-d(x)"),
        (0.42, "Calibration\nSplit\n(50%)"),
        (0.60, "Threshold\nτ_α = Q_{1-α}\n(M1/M2/M3)"),
        (0.78, "Evaluate\nFPR / TPR\nCE / Violation"),
    ]
    for x, label in boxes:
        ax.add_patch(
            plt.FancyBboxPatch(
                (x - 0.08, 0.2), 0.15, 0.55,
                boxstyle="round,pad=0.02",
                facecolor="#dce8f7", edgecolor="steelblue", linewidth=1.5,
            )
        )
        ax.text(x, 0.475, label, ha="center", va="center", fontsize=9, wrap=True)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.07
        x1 = boxes[i + 1][0] - 0.09
        ax.annotate(
            "", xy=(x1, 0.475), xytext=(x0, 0.475),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        )

    ax.set_title("Figure 1: Fixed-FPR Security Verification Protocol", fontsize=FONT_SIZE + 1, pad=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figs_dir / f"fig1_protocol.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[fig1] Saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FIGURE_FNS = {"fig1": gen_fig1, "fig2": gen_fig2, "fig3": gen_fig3}


def main():
    parser = argparse.ArgumentParser(description="FFSVCA figure generator")
    parser.add_argument("--config", default="robin_config.json")
    parser.add_argument("--figure", choices=list(FIGURE_FNS.keys()) + ["all"], default="all")
    parser.add_argument("--outputs", default=str(OUTPUTS_ROOT))
    args = parser.parse_args()

    outputs = Path(args.outputs)

    if args.figure == "all":
        for name, fn in FIGURE_FNS.items():
            print(f"\nGenerating {name}...")
            fn(outputs)
    else:
        FIGURE_FNS[args.figure](outputs)


if __name__ == "__main__":
    main()
