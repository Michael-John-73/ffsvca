"""
gen_flow.py
Generate pipeline flow diagram for PAPER3.
Outputs a Graphviz-style diagram saved as PNG/PDF via matplotlib.

Output: outputs_p3/figures/fig_pipeline_flow.png/pdf

Usage:
    python gen_flow.py --outputs ../PAPER3/outputs_p3
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUTS_ROOT = Path("../PAPER3/outputs_p3")
DPI = 300


def gen_pipeline_flow(outputs: Path) -> None:
    figs_dir = outputs / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # Node definitions: (x_center, y_center, label, color)
    nodes = [
        (1.2, 6.2, "Prompt List\n(N=1000)", "#dce8f7"),
        (3.5, 6.2, "ROBIN\nGen + Embed\n(seed=0)", "#dce8f7"),
        (6.0, 6.2, "Attack\nPipeline\n(11 tasks)", "#fff3cd"),
        (8.5, 6.2, "Score Dump\nz(x)=-d(x)", "#dce8f7"),
        (10.8, 6.2, "Raw Scores\nCSV", "#d4edda"),
        (1.2, 3.8, "Clean\nImages\n(no embed)", "#dce8f7"),
        (3.5, 3.8, "Attack\nPipeline\n(11 tasks)", "#fff3cd"),
        (6.0, 3.8, "Score Dump\nz(x)=-d(x)", "#dce8f7"),
        (8.5, 3.8, "Calibration\nSplit (50%)", "#fff3cd"),
        (10.8, 3.8, "τ_α\n(M1/M2/M3)", "#d4edda"),
        (6.0, 1.5, "Evaluate\nFPR / TPR\nCE / Violation", "#f8d7da"),
        (10.8, 1.5, "Tables 2-4\nFigures 2-3", "#d4edda"),
    ]

    NODE_W, NODE_H = 1.8, 0.9
    for (x, y, label, color) in nodes:
        rect = mpatches.FancyBboxPatch(
            (x - NODE_W / 2, y - NODE_H / 2), NODE_W, NODE_H,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="gray", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, linespacing=1.3)

    # Arrows (start, end) by (x_center, y_center)
    arrows = [
        (nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]),
        (nodes[3], nodes[4]),
        (nodes[5], nodes[6]), (nodes[6], nodes[7]), (nodes[7], nodes[8]),
        (nodes[8], nodes[9]),
        (nodes[4], nodes[10]), (nodes[9], nodes[10]),
        (nodes[10], nodes[11]),
    ]
    for (sx, sy, *_), (ex, ey, *_) in arrows:
        ax.annotate(
            "", xy=(ex - NODE_W / 2, ey), xytext=(sx + NODE_W / 2, sy),
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.2),
        )

    # Group labels
    ax.text(6.0, 7.0, "Watermarked Branch", ha="center", fontsize=10, color="steelblue", fontweight="bold")
    ax.text(6.0, 4.6, "Clean Branch", ha="center", fontsize=10, color="steelblue", fontweight="bold")

    ax.set_title("PAPER3 Experiment Pipeline Flow", fontsize=12, pad=5)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = figs_dir / f"fig_pipeline_flow.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[flow] Saved -> {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="FFSVCA pipeline flow diagram")
    parser.add_argument("--outputs", default=str(OUTPUTS_ROOT))
    args = parser.parse_args()
    gen_pipeline_flow(Path(args.outputs))


if __name__ == "__main__":
    main()
