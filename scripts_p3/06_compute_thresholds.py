"""
scripts_p3/06_compute_thresholds.py

Phase 2.6 / 4.2: compute thresholds for M1 (Raw ROBIN), M2 (ROC-selected),
M3 (Proposed fixed-FPR clean-quantile) for all alphas, independently per
gen_seed (paper3.md §6.5 — thresholds must never be pooled across seeds).

Paper-canonical method definitions
==================================
- M1 Raw ROBIN
    The reference threshold reported by ROBIN's own evaluation script. Read
    from `--m1_threshold` if provided, else falls back to the midpoint of
    cal_clean/cal_watermarked means on the `none` attack (calibration split
    only — never test data). M1 is alpha-independent.
- M2 ROC-selected
    Threshold maximizing Youden's J on CLEAN-vs-WATERMARKED scores from the
    CALIBRATION pool (cal_clean.csv + cal_watermarked.csv, attack_id=="none"
    portion) — never the test split, to avoid selecting a threshold on the
    same data used to evaluate it (paper3.md §5.2). M2 is alpha-independent.
- M3 Proposed (clean calibration quantile)
    tau_alpha = Q_{1-alpha}(score_z over cal_clean@none), per seed.

Input:
    outputs_p3/splits/{cal_clean.csv, cal_watermarked.csv, test_clean.csv, test_watermarked.csv}
Output:
    outputs_p3/thresholds/thresholds_by_method.csv
        columns: gen_seed, method, alpha, threshold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import conformal_quantile


def roc_youden_threshold(neg: np.ndarray, pos: np.ndarray) -> float:
    """Threshold that maximizes Youden's J = TPR - FPR on raw scores."""
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros_like(neg, dtype=int),
                        np.ones_like(pos, dtype=int)])
    order = np.argsort(-s)  # descending
    s, y = s[order], y[order]
    P = y.sum()
    N = len(y) - P
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(s[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/thresholds/thresholds_by_method.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    ap.add_argument("--m1_threshold", type=float, default=None,
                    help="Optional explicit Raw ROBIN threshold")
    args = ap.parse_args()

    cal_clean = pd.read_csv(args.splits_dir / "cal_clean.csv")
    cal_wm = pd.read_csv(args.splits_dir / "cal_watermarked.csv")

    rows = []
    for seed, cal_c in cal_clean.groupby("seed"):
        cal_w = cal_wm[cal_wm.seed == seed]
        cal_c_none = cal_c[cal_c.attack_id == "none"]["score_z"].to_numpy()
        cal_w_none = cal_w[cal_w.attack_id == "none"]["score_z"].to_numpy()

        # M1 Raw ROBIN: alpha-independent, calibration split only
        if args.m1_threshold is not None:
            m1 = float(args.m1_threshold)
        elif len(cal_c_none) and len(cal_w_none):
            m1 = float((cal_c_none.mean() + cal_w_none.mean()) / 2.0)
        else:
            m1 = float("nan")

        # M2 ROC-selected: alpha-independent, calibration split only (no test leakage)
        m2 = (roc_youden_threshold(neg=cal_c_none, pos=cal_w_none)
              if (len(cal_c_none) and len(cal_w_none)) else float("nan"))

        for a in args.alpha:
            m3 = conformal_quantile(cal_c_none, a) if len(cal_c_none) else float("nan")
            rows.append({"gen_seed": int(seed), "method": "M1", "alpha": a, "threshold": m1})
            rows.append({"gen_seed": int(seed), "method": "M2", "alpha": a, "threshold": m2})
            rows.append({"gen_seed": int(seed), "method": "M3", "alpha": a, "threshold": m3})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    n_seeds = cal_clean["seed"].nunique()
    print(f"[06] thresholds for alphas={args.alpha}, {n_seeds} seed(s) -> {args.out}")


if __name__ == "__main__":
    main()
