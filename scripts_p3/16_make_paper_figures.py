"""
scripts_p3/16_make_paper_figures.py

Generate the paper figures (PDF + PNG) directly from outputs_p3/metrics/*.csv:

Figure 1: FPR-per-attack bar chart (M1/M2/M3) at primary alpha=0.05.
Figure 2: ROC-like (FPR, TPR) scatter colored by method, faceted by family.
Figure 3: Calibration-size sensitivity (E4) lineplot.
Figure 4: Severity-sweep curves (JPEG quality, additive noise sigma) for M3.
Figure 5: COCO real-image FPR bar chart with Wilson CI (E7, M3).
Figure 6: Diffusion-regeneration TPR vs strength with Wilson CI (E8, M3).
Figure 7: Score-distribution histogram (cal_clean vs test_clean@none vs
          test_watermarked@none). Absorbs the descriptive role of the
          deprecated E3 script.

Uses matplotlib only. All figures written to outputs_p3/figures/.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def fig1(e2: pd.DataFrame, alpha: float, out: Path):
    sub = e2[(e2.alpha == alpha) & (e2.attack_id != "_worst")].copy()
    pivot = sub.pivot_table(index="attack_id", columns="method",
                            values="empirical_fpr").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(pivot.index))
    width = 0.27
    for i, m in enumerate(["M1", "M2", "M3"]):
        if m in pivot.columns:
            ax.bar(x + (i - 1) * width, pivot[m].values, width=width, label=m)
    ax.axhline(alpha, ls="--", color="black", lw=1, label=f"alpha={alpha}")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Empirical FPR")
    ax.set_title(f"Empirical FPR per evaluation condition at alpha={alpha}")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)


"""Manifest 'family' labels encode composition stage count, not an operator taxonomy."""
GROUP_LABEL = {
    "same-family composite": "two-stage composite",
    "cross-family composite": "three-stage composite",
}


def fig2(e2: pd.DataFrame, alpha: float, out: Path):
    sub = e2[(e2.alpha == alpha) & (e2.attack_id != "_worst")].copy()
    fams = sorted(sub.family.unique())
    fig, axes = plt.subplots(1, max(1, len(fams)), figsize=(4 * len(fams), 4),
                             squeeze=False)
    for j, fam in enumerate(fams):
        ax = axes[0, j]
        for m in ["M1", "M2", "M3"]:
            d = sub[(sub.family == fam) & (sub.method == m)]
            ax.scatter(d.empirical_fpr, d.empirical_tpr, label=m, alpha=0.8)
        ax.axvline(alpha, ls="--", color="black", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(GROUP_LABEL.get(fam, fam)); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out)


def fig3(e4: pd.DataFrame, alpha: float, out: Path):
    sub = e4[e4.alpha == alpha].copy()
    is_all = sub.gen_seed.astype(str) == "all"
    per_seed, mean = sub[~is_all], sub[is_all].sort_values("size_effective")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    seeds = sorted(per_seed.gen_seed.astype(str).unique())
    for i, s in enumerate(seeds):
        d = per_seed[per_seed.gen_seed.astype(str) == s].sort_values("size_effective")
        # small multiplicative jitter so the per-seed error bars do not overlap
        x = d.size_effective * (1 + 0.03 * (i - (len(seeds) - 1) / 2))
        ax.errorbar(x, d.fpr_mean, yerr=d.fpr_std, marker="o", ms=3, ls="none",
                    capsize=2, alpha=0.7, label=f"seed {s} (mean $\\pm$ resampling std)")
    ax.plot(mean.size_effective, mean.fpr_mean, color="black", marker="D", lw=1.5,
            label="five-seed mean")
    ax.axhline(alpha, ls="--", color="gray", lw=1, label=f"alpha={alpha}")
    ax.set_xscale("log")
    ax.set_xticks([30, 50, 100, 200, 500])
    ax.set_xticklabels(["30", "50", "100", "200", "500"])
    # log axis adds minor labels (4x10^1, 3x10^2, ...) that overlap the major ones
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Calibration size $n_{cal}$")
    ax.set_ylabel("Empirical FPR (17-condition pool)")
    ax.set_title("E4 calibration-size sensitivity (M3)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out)


def _extract_severity(attack_id: str, family_prefix: str) -> float | None:
    """attack_id like 'jpeg_q50' or 'noise_s003' -> 50.0 or 0.03."""
    m = re.match(rf"^{family_prefix}_([qs])(\d+)$", attack_id)
    if not m:
        return None
    key, num = m.group(1), m.group(2)
    if key == "q":
        return float(num)
    # noise ids encode sigma*100 (s001 -> 0.01, s008 -> 0.08)
    return int(num) / 100.0


def fig4_severity(e2: pd.DataFrame, alpha: float, out: Path):
    sub = e2[(e2.alpha == alpha) & (e2.method == "M3") &
             (e2.attack_id != "_worst")].copy()
    sub_j = sub.copy()
    sub_j["sev"] = sub_j.attack_id.apply(lambda a: _extract_severity(a, "jpeg"))
    sub_j = sub_j.dropna(subset=["sev"]).sort_values("sev")
    sub_n = sub.copy()
    sub_n["sev"] = sub_n.attack_id.apply(lambda a: _extract_severity(a, "noise"))
    sub_n = sub_n.dropna(subset=["sev"]).sort_values("sev")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, xlabel, title in ((axes[0], sub_j, "JPEG quality", "JPEG severity sweep (M3)"),
                                 (axes[1], sub_n, "Noise sigma", "Noise severity sweep (M3)")):
        if d.empty:
            continue
        is_all = d.gen_seed.astype(str) == "all"
        seeds_d, mean_d = d[~is_all], d[is_all].sort_values("sev")
        ax.scatter(seeds_d.sev, seeds_d.empirical_fpr, marker="o", s=12, alpha=0.4,
                   color="tab:blue", label="FPR, per seed")
        ax.plot(mean_d.sev, mean_d.empirical_fpr, marker="o", color="tab:blue",
                label="FPR, five-seed mean")
        ax.scatter(seeds_d.sev, seeds_d.empirical_tpr, marker="s", s=12, alpha=0.4,
                   color="tab:orange", label="TPR, per seed")
        ax.plot(mean_d.sev, mean_d.empirical_tpr, marker="s", color="tab:orange",
                label="TPR, five-seed mean")
        ax.axhline(alpha, ls="--", color="black", lw=1, label=f"alpha={alpha}")
        ax.set_xlabel(xlabel); ax.set_ylabel("rate"); ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out)


def fig5_coco(e7: pd.DataFrame, alpha: float, out: Path):
    sub = e7[(e7.alpha == alpha) & (e7.method == "M3")].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(sub))
    yerr_lo = sub.empirical_fpr - sub.fpr_ci_lo
    yerr_hi = sub.fpr_ci_hi - sub.empirical_fpr
    ax.bar(x, sub.empirical_fpr,
           yerr=[yerr_lo.values, yerr_hi.values], capsize=4)
    ax.axhline(alpha, ls="--", color="black", lw=1, label=f"alpha={alpha}")
    ax.set_xticks(x); ax.set_xticklabels(sub.attack_id, rotation=0)
    ax.set_ylabel("Empirical FPR (COCO real)")
    ax.set_title("E7 COCO real-image FPR with Wilson 95% CI (M3)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)


def fig6_adaptive(e8: pd.DataFrame, alpha: float, out: Path):
    sub = e8[(e8.alpha == alpha) & (e8.method == "M3")].sort_values("strength")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    yerr_lo = sub.tpr - sub.tpr_ci_lo
    yerr_hi = sub.tpr_ci_hi - sub.tpr
    ax.errorbar(sub.strength, sub.tpr,
                yerr=[yerr_lo.values, yerr_hi.values],
                marker="o", capsize=4, label="TPR, regenerated watermarked (95% Wilson CI)")
    if "fpr" in sub.columns and sub.fpr.notna().any():
        ax.errorbar(sub.strength, sub.fpr, marker="s",
                    label="FPR, regenerated non-watermarked control")
    ax.axhline(alpha, ls="--", color="black", lw=1, label=f"alpha={alpha}")
    ax.set_xlabel("Regeneration strength")
    ax.set_ylabel("rate")
    ax.set_title("E8 diffusion-regeneration (M3)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)


def fig7_score_hist(splits_dir: Path, out: Path):
    cal = pd.read_csv(splits_dir / "cal_clean.csv")
    tcl = pd.read_csv(splits_dir / "test_clean.csv")
    twm = pd.read_csv(splits_dir / "test_watermarked.csv")
    cal_n = cal[cal.attack_id == "none"]["score_z"].to_numpy()
    tcl_n = tcl[tcl.attack_id == "none"]["score_z"].to_numpy()
    twm_n = twm[twm.attack_id == "none"]["score_z"].to_numpy()
    if min(len(cal_n), len(tcl_n), len(twm_n)) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 40
    ax.hist(cal_n, bins=bins, alpha=0.45, label="calibration, non-watermarked")
    ax.hist(tcl_n, bins=bins, alpha=0.45, label="test, non-watermarked")
    ax.hist(twm_n, bins=bins, alpha=0.45, label="test, watermarked")
    ax.set_xlabel("detection score $z = -d$")
    ax.set_ylabel("count (five seeds pooled)")
    ax.set_title("Score distributions, unmodified condition")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=Path, default=Path("outputs_p3/metrics"))
    ap.add_argument("--splits_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--out_dir", type=Path, default=Path("outputs_p3/figures"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    e2 = pd.read_csv(args.metrics_dir / "e2_fpr_drift.csv")
    fig1(e2, args.alpha, args.out_dir / f"fig1_fpr_per_attack_alpha{args.alpha}")
    fig2(e2, args.alpha, args.out_dir / f"fig2_fpr_tpr_by_family_alpha{args.alpha}")
    fig4_severity(e2, args.alpha,
                  args.out_dir / f"fig4_severity_sweep_alpha{args.alpha}")

    e4_path = args.metrics_dir / "e4_calibration_size.csv"
    if e4_path.exists():
        fig3(pd.read_csv(e4_path), args.alpha,
             args.out_dir / f"fig3_calibration_size_alpha{args.alpha}")

    e7_path = args.metrics_dir / "e7_coco_real_fpr.csv"
    if e7_path.exists():
        fig5_coco(pd.read_csv(e7_path), args.alpha,
                  args.out_dir / f"fig5_coco_real_fpr_alpha{args.alpha}")

    e8_path = args.metrics_dir / "e8_adaptive_regen.csv"
    if e8_path.exists():
        fig6_adaptive(pd.read_csv(e8_path), args.alpha,
                      args.out_dir / f"fig6_adaptive_regen_alpha{args.alpha}")

    if args.splits_dir.exists():
        fig7_score_hist(args.splits_dir,
                        args.out_dir / "fig7_score_dist_none")

    print(f"[16] figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
