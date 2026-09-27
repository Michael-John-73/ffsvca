"""
scripts_p3/08d_eval_adaptive_regen.py

E8: Adaptive regeneration stress test.

Watermarked images are regenerated with SD 1.5 img2img at strengths
{0.2, 0.3, 0.5}. Regenerated images are scored with ROBIN. We report:
    - empirical_tpr  = fraction of regenerated images still detected as
                       watermarked under M3 thresholds (TPR drop is expected)
    - empirical_fpr_clean = control: FPR on the matched clean regenerated
                            pool (must remain near alpha if calibration holds)

Expected input schema (outputs_p3/scores/regen_scores.csv):
    image_id, strength, source_label, score_z
where source_label in {"watermarked", "clean"}.

Output: outputs_p3/metrics/e8_adaptive_regen.csv
    columns: method, alpha, strength, n_wm, tpr, tpr_ci_lo, tpr_ci_hi,
             n_clean, fpr, fpr_ci_lo, fpr_ci_hi, delta_fpr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (empirical_fpr, empirical_tpr, threshold_for, wilson_ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path,
                    default=Path("outputs_p3/scores/regen_scores.csv"))
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs_p3/thresholds/thresholds_by_method.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/metrics/e8_adaptive_regen.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    ap.add_argument("--methods", nargs="+", type=str, default=["M3"])
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    args = ap.parse_args()

    if not args.scores.exists():
        raise SystemExit(
            f"[08d] {args.scores} not found. Run regen scoring first "
            "(outside this pipeline) and place a CSV with columns "
            "image_id,strength,source_label,score_z there."
        )

    df = pd.read_csv(args.scores)
    thr = pd.read_csv(args.thresholds)

    rows = []
    for alpha in args.alpha:
        for method in args.methods:
            tau = threshold_for(thr, method, alpha, gen_seed=0)  # E8 reuses seed-0 calibration (Appendix A)
            for s in args.strengths:
                sub = df[np.isclose(df.strength, s)]
                wm = sub[sub.source_label == "watermarked"]["score_z"].to_numpy()
                cl = sub[sub.source_label == "clean"]["score_z"].to_numpy()
                n_wm, n_cl = int(len(wm)), int(len(cl))
                tpr = empirical_tpr(wm, tau) if n_wm else float("nan")
                fpr = empirical_fpr(cl, tau) if n_cl else float("nan")
                tpr_lo, tpr_hi = wilson_ci(int(np.sum(wm > tau)), n_wm) \
                    if n_wm else (float("nan"), float("nan"))
                fpr_lo, fpr_hi = wilson_ci(int(np.sum(cl > tau)), n_cl) \
                    if n_cl else (float("nan"), float("nan"))
                rows.append({
                    "method": method, "alpha": alpha, "strength": s,
                    "n_wm": n_wm, "tpr": tpr,
                    "tpr_ci_lo": tpr_lo, "tpr_ci_hi": tpr_hi,
                    "n_clean": n_cl, "fpr": fpr,
                    "fpr_ci_lo": fpr_lo, "fpr_ci_hi": fpr_hi,
                    "delta_fpr": fpr - alpha if not np.isnan(fpr) else float("nan"),
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[08d] E8 adaptive regen -> {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
