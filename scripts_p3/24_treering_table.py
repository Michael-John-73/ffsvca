"""Emit LaTeX rows for the Tree-Ring appendix table (alpha=0.05) and check |dFPR| <= KS."""
from pathlib import Path

import pandas as pd

M = Path(__file__).resolve().parents[1] / "sd21_results/outputs_p3/treering/metrics"
f = pd.read_csv(M / "treering_fpr.csv")
r = pd.read_csv(M / "treering_resplit.csv")
out = []
for a in (0.01, 0.05, 0.10):
    for d in ("Tree-Ring", "ROBIN"):
        s = f[(f.detector == d) & (f.alpha == a)].set_index("attack_id")
        shift = (s.fpr - s.loc["none", "fpr"]).abs()
        out.append(f"% check alpha={a} {d}: max(|dFPR|-KS)={float((shift - s.ks_vs_none).max()):.4f} "
                   f"n_exceed={int((s.fpr > a).sum())} min_p={s.p_exceed_one_sided.min():.4f}")
a = 0.05
t = f[(f.detector == "Tree-Ring") & (f.alpha == a)].set_index("attack_id")
b = f[(f.detector == "ROBIN") & (f.alpha == a)].set_index("attack_id")
rt = r[(r.detector == "Tree-Ring") & (r.alpha == a)].set_index("attack_id").exceed_freq
rb = r[(r.detector == "ROBIN") & (r.alpha == a)].set_index("attack_id").exceed_freq
order = ["none", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "noise_s001", "noise_s003",
         "noise_s005", "noise_s008", "blurring", "blurring+noise", "cropping", "rotation",
         "jpeg+cropping", "cropping+jpeg", "rotation+cropping+jpeg", "noise+blurring+jpeg"]
for k in order:
    x, y = t.loc[k], b.loc[k]
    out.append(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {x.fpr:.3f} ({x.wilson_lo:.3f}--{x.wilson_hi:.3f}) & "
               f"{x.ks_vs_none:.3f} & {rt[k]:.2f} & {y.fpr:.3f} ({y.wilson_lo:.3f}--{y.wilson_hi:.3f}) & "
               f"{y.ks_vs_none:.3f} & {rb[k]:.2f} \\\\")
(M / "treering_table_rows.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
