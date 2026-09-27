"""
scripts_p3/08c_eval_coco_real_fpr.py

E7: Cross-domain real-image FPR on MS-COCO val 2017.

Reads a precomputed COCO-real score CSV (produced outside this pipeline by
running ROBIN inversion / scoring on non-SD natural photographs) and the
M3 thresholds calibrated on SD-clean. For each (method, alpha, attack_id),
reports the empirical FPR (all COCO images are by construction clean,
label = 0) with Wilson 95% CI and the delta_fpr = empirical_fpr - alpha.

Expected input schema (outputs_p3/scores/coco_real_scores.csv):
    image_id, attack_id, score_z

Output: outputs_p3/metrics/e7_coco_real_fpr.csv
    columns: method, alpha, attack_id, n_eval, n_pos,
             empirical_fpr, fpr_ci_lo, fpr_ci_hi,
             calibration_error, delta_fpr, fpr_violation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import calibration_error, empirical_fpr, threshold_for, wilson_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path,
                    default=Path("outputs_p3/scores/coco_real_scores.csv"))
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs_p3/thresholds/thresholds_by_method.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/metrics/e7_coco_real_fpr.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    ap.add_argument("--methods", nargs="+", type=str, default=["M3"])
    ap.add_argument("--attacks", nargs="+", type=str, default=["none", "jpeg"])
    args = ap.parse_args()

    if not args.scores.exists():
        raise SystemExit(
            f"[08c] {args.scores} not found. Run COCO scoring first "
            "(outside this pipeline) and place a CSV with columns "
            "image_id,attack_id,score_z there."
        )

    df = pd.read_csv(args.scores)
    thr = pd.read_csv(args.thresholds)

    rows = []
    for alpha in args.alpha:
        for method in args.methods:
            tau = threshold_for(thr, method, alpha, gen_seed=0)  # E7 reuses seed-0 calibration (Appendix A)
            for atk in args.attacks:
                neg = df[df.attack_id == atk]["score_z"].to_numpy()
                n_eval = int(len(neg))
                if n_eval == 0:
                    continue
                fpr = empirical_fpr(neg, tau)
                n_pos = int(np.sum(neg > tau))
                lo, hi = wilson_ci(n_pos, n_eval)
                rows.append({
                    "method": method, "alpha": alpha, "attack_id": atk,
                    "n_eval": n_eval, "n_pos": n_pos,
                    "empirical_fpr": fpr,
                    "fpr_ci_lo": lo, "fpr_ci_hi": hi,
                    "calibration_error": calibration_error(fpr, alpha),
                    "delta_fpr": fpr - alpha if not np.isnan(fpr) else float("nan"),
                    "fpr_violation": int(fpr > alpha) if not np.isnan(fpr) else 0,
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[08c] E7 COCO real FPR -> {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
