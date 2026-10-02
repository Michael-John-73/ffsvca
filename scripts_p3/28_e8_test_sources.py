"""
scripts_p3/28_e8_test_sources.py

E8 re-aggregation restricted to regeneration sources outside the M3 calibration set (no new data).

regen_scores.csv holds sources i = 0..199 of generation seed 0 (image_id = f"{i:06d}" = prompt_id of
ori-lg7.5-{i}.jpg / wm-lg7.5-{i}.jpg). The seed-0 M3 threshold is calibrated on the non-watermarked
`none` scores of the calibration half of the split_seed=42 partition, so sources in that half are not
independent of the threshold. This script
  (1) checks the id correspondence and the cal/test membership of the 200 sources,
  (2) recomputes the seed-0 M3 threshold from cal_clean.csv and checks it against thresholds_by_method.csv,
  (3) reproduces the published all-200 E8 table,
  (4) recomputes TPR/FPR, counts, denominators and Wilson 95% intervals on the test-half sources only,
      with the same seed-0 threshold, the same selection for both classes and all strengths,
  (5) redraws Figure 6 from the test-half table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import conformal_quantile, wilson_ci  # noqa: E402

ALPHAS = (0.05, 0.10, 0.01)
STRENGTHS = (0.2, 0.3, 0.5)


def table(regen: pd.DataFrame, ids: set, tau: float, alpha: float, subset: str) -> list[dict]:
    rows = []
    for s in STRENGTHS:
        sub = regen[np.isclose(regen.strength, s) & regen.pid.isin(ids)]
        wm = sub[sub.source_label == "watermarked"].score_z.to_numpy()
        cl = sub[sub.source_label == "clean"].score_z.to_numpy()
        kw, kc = int(np.sum(wm > tau)), int(np.sum(cl > tau))
        tlo, thi = wilson_ci(kw, len(wm))
        flo, fhi = wilson_ci(kc, len(cl))
        rows.append(dict(subset=subset, method="M3", alpha=alpha, threshold=tau, strength=s,
                         n_wm=len(wm), k_wm=kw, tpr=kw / len(wm), tpr_ci_lo=tlo, tpr_ci_hi=thi,
                         n_clean=len(cl), k_clean=kc, fpr=kc / len(cl), fpr_ci_lo=flo, fpr_ci_hi=fhi,
                         delta_fpr=kc / len(cl) - alpha))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs_p3"))
    ap.add_argument("--fig_out", type=Path, default=Path("outputs_p3/figures/fig6_adaptive_regen_alpha0"))
    args = ap.parse_args()
    root = args.root
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    regen = pd.read_csv(root / "scores" / "regen_scores.csv", dtype={"image_id": str})
    regen["pid"] = regen.image_id.astype(int)
    cal = pd.read_csv(root / "splits" / "cal_clean.csv")
    test = pd.read_csv(root / "splits" / "test_clean.csv")
    cal0 = cal[(cal.seed == 0) & (cal.attack_id == "none")]
    test0 = test[(test.seed == 0) & (test.attack_id == "none")]
    cal_ids, test_ids = set(cal0.prompt_id.astype(int)), set(test0.prompt_id.astype(int))
    src = set(regen.pid)

    if cal_ids & test_ids or len(cal_ids | test_ids) != 1000:
        raise SystemExit("seed-0 calibration/test halves do not partition 0..999")
    if not regen.image_id.str.fullmatch(r"\d{6}").all():
        raise SystemExit("unexpected image_id format")
    counts = regen.groupby(["strength", "source_label"]).pid.nunique()
    log(f"regen rows={len(regen)} sources={len(src)} id range={min(src)}..{max(src)}; "
        f"sources per (strength,label): {sorted(set(counts.tolist()))}")
    log(f"seed-0 none partition: cal={len(cal_ids)} test={len(test_ids)} (disjoint, cover 0..999)")
    log(f"regen sources in calibration half = {len(src & cal_ids)}; in test half = {len(src & test_ids)}")

    thr_file = root / "thresholds" / "thresholds_by_method.csv"
    stored = pd.read_csv(thr_file) if thr_file.exists() else None
    pub = pd.read_csv(root / "metrics" / "e8_adaptive_regen.csv")
    rows = []
    for al in ALPHAS:
        tau = conformal_quantile(cal0.score_z.to_numpy(), al)
        msg = f"alpha={al}: seed-0 M3 threshold recomputed = {tau!r}"
        if stored is not None:
            st = stored[(stored.method == "M3") & np.isclose(stored.alpha, al) & (stored.gen_seed == 0)].threshold
            msg += f"; stored = {float(st.iloc[0])!r}; equal = {bool(np.isclose(st.iloc[0], tau, rtol=0, atol=0))}"
        log(msg)
        full = table(regen, src, tau, al, "all_200")
        p = pub[np.isclose(pub.alpha, al)].sort_values("strength")
        f = pd.DataFrame(full)
        same = np.allclose(p[["tpr", "fpr", "tpr_ci_lo", "fpr_ci_hi"]].to_numpy(),
                           f[["tpr", "fpr", "tpr_ci_lo", "fpr_ci_hi"]].to_numpy(), rtol=0, atol=1e-12)
        log(f"alpha={al}: all-200 re-aggregation reproduces e8_adaptive_regen.csv: {same}")
        rows += full + table(regen, src & test_ids, tau, al, "test_half")

    df = pd.DataFrame(rows)
    df.to_csv(root / "metrics" / "e8_adaptive_regen_test_sources.csv", index=False)
    show = df[df.subset == "test_half"][["alpha", "strength", "k_wm", "n_wm", "tpr", "tpr_ci_lo", "tpr_ci_hi",
                                          "k_clean", "n_clean", "fpr", "fpr_ci_lo", "fpr_ci_hi"]]
    log("\ntest-half sources (same seed-0 threshold):")
    log(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    (root / "metrics" / "e8_adaptive_regen_test_sources.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sub = df[(df.subset == "test_half") & np.isclose(df.alpha, 0.05)].sort_values("strength")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sub.strength, sub.tpr, yerr=[(sub.tpr - sub.tpr_ci_lo).values, (sub.tpr_ci_hi - sub.tpr).values],
                marker="o", capsize=4, label="TPR (regen wm)")
    ax.errorbar(sub.strength, sub.fpr, yerr=[(sub.fpr - sub.fpr_ci_lo).values, (sub.fpr_ci_hi - sub.fpr).values],
                marker="s", capsize=4, label="FPR (regen clean control)")
    ax.axhline(0.05, ls="--", color="black", lw=1, label="alpha=0.05")
    ax.set_xlabel("Regeneration strength")
    ax.set_ylabel("rate")
    ax.set_title(f"E8 diffusion-regeneration (M3), test-half sources (n={int(sub.n_wm.iloc[0])})")
    ax.legend()
    fig.tight_layout()
    args.fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_out.with_suffix(".pdf"))
    fig.savefig(args.fig_out.with_suffix(".png"), dpi=150)
    print(f"written -> {root / 'metrics' / 'e8_adaptive_regen_test_sources.csv'}, {args.fig_out}.pdf")


if __name__ == "__main__":
    main()
