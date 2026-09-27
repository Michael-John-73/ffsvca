"""
scripts_p3/11_eval_failure_conditions.py

E5: Our failure-condition analysis (Section 5.5).

We compute thresholds per seed and never pool them. We compute the metrics
independently per gen_seed, then append a cross-seed mean/std aggregate row
(gen_seed = "all").

For each (method, alpha, gen_seed) we compute:
    V_t            = total number of violating attacks (empirical_fpr > alpha)
    ViolationRate  = V_t / |attacks|
    delta_fpr_OOD  = mean empirical_fpr - alpha over cross-family composite attacks
    failing_attacks = ";"-joined list of failing attack_ids

We write: outputs_p3/metrics/e5_failure_conditions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import empirical_fpr, threshold_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs_p3/thresholds/thresholds_by_method.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/metrics/e5_failure_conditions.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    args = ap.parse_args()

    tcl = pd.read_csv(args.splits_dir / "test_clean.csv")
    thr = pd.read_csv(args.thresholds)
    attacks = sorted(tcl["attack_id"].unique())
    fam_of = tcl.drop_duplicates("attack_id").set_index("attack_id")["family"].to_dict()
    gen_seeds = sorted(tcl.seed.unique())

    rows = []
    for gen_seed in gen_seeds:
        tcl_s = tcl[tcl.seed == gen_seed]
        for alpha in args.alpha:
            for method in ["M1", "M2", "M3"]:
                tau = threshold_for(thr, method, alpha, gen_seed=gen_seed)
                violating = []
                ood_drift = []
                for atk in attacks:
                    neg = tcl_s[tcl_s.attack_id == atk]["score_z"].to_numpy()
                    fpr = empirical_fpr(neg, tau)
                    if not np.isnan(fpr) and fpr > alpha:
                        violating.append(atk)
                    if fam_of.get(atk) == "cross-family composite" and not np.isnan(fpr):
                        ood_drift.append(fpr - alpha)
                rows.append({
                    "gen_seed": gen_seed, "method": method, "alpha": alpha, "threshold": tau,
                    "V_t": len(violating),
                    "ViolationRate": len(violating) / max(len(attacks), 1),
                    "delta_fpr_OOD_mean": float(np.mean(ood_drift)) if ood_drift else float("nan"),
                    "delta_fpr_OOD_max": float(np.max(ood_drift)) if ood_drift else float("nan"),
                    "failing_attacks": ";".join(violating),
                })

    df = pd.DataFrame(rows)
    agg = (df.groupby(["method", "alpha"])
             .agg(V_t=("V_t", "mean"), V_t_std=("V_t", "std"),
                  ViolationRate=("ViolationRate", "mean"),
                  ViolationRate_std=("ViolationRate", "std"),
                  delta_fpr_OOD_mean=("delta_fpr_OOD_mean", "mean"),
                  delta_fpr_OOD_max=("delta_fpr_OOD_max", "max"))
             .reset_index())
    agg["gen_seed"] = "all"
    agg["threshold"] = float("nan")
    agg["failing_attacks"] = ""

    out_df = pd.concat([df, agg], ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[11] E5 written -> {args.out} ({len(df)} per-seed rows + {len(agg)} aggregate rows)")


if __name__ == "__main__":
    main()
