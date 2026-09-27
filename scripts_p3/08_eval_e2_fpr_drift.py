"""
scripts_p3/08_eval_e2_fpr_drift.py

E2: We measure FPR drift under attack, with worst-case columns.

We compute thresholds per seed and never pool them. We evaluate each
gen_seed's test split against THAT seed's threshold; we write per-seed
rows (gen_seed = 0..4) plus a cross-seed mean/std aggregate row
(gen_seed = "all").

For each (method, alpha, gen_seed) we compute:
    - per-attack empirical_fpr, empirical_tpr, calibration_error
    - delta_fpr     = empirical_fpr - alpha
    - fpr_violation = 1[empirical_fpr > alpha]
    - worst_case_fpr / worst_case_tpr / worst_attack_fpr / worst_attack_tpr
      across attacks within the family (ID-single / same-family
      composite / cross-family composite). We append the worst-case rows
      with attack_id = "_worst".

We write: outputs_p3/metrics/e2_fpr_drift.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (calibration_error, empirical_fpr, empirical_tpr,
                     threshold_for)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs_p3/thresholds/thresholds_by_method.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/metrics/e2_fpr_drift.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    args = ap.parse_args()

    tcl = pd.read_csv(args.splits_dir / "test_clean.csv")
    twm = pd.read_csv(args.splits_dir / "test_watermarked.csv")
    thr = pd.read_csv(args.thresholds)

    fam_of = pd.concat([tcl[["attack_id", "family"]], twm[["attack_id", "family"]]])
    fam_of = fam_of.drop_duplicates("attack_id").set_index("attack_id")["family"].to_dict()

    attack_ids = sorted(fam_of.keys())
    gen_seeds = sorted(set(tcl.seed.unique()) | set(twm.seed.unique()))

    rows = []
    for gen_seed in gen_seeds:
        tcl_s = tcl[tcl.seed == gen_seed]
        twm_s = twm[twm.seed == gen_seed]
        for alpha in args.alpha:
            for method in ["M1", "M2", "M3"]:
                tau = threshold_for(thr, method, alpha, gen_seed=gen_seed)
                per_attack = []
                for atk in attack_ids:
                    neg = tcl_s[tcl_s.attack_id == atk]["score_z"].to_numpy()
                    pos = twm_s[twm_s.attack_id == atk]["score_z"].to_numpy()
                    fpr = empirical_fpr(neg, tau)
                    tpr = empirical_tpr(pos, tau)
                    row = {
                        "gen_seed": gen_seed, "method": method, "alpha": alpha,
                        "threshold": tau,
                        "attack_id": atk, "family": fam_of.get(atk, "unknown"),
                        "empirical_fpr": fpr, "empirical_tpr": tpr,
                        "calibration_error": calibration_error(fpr, alpha),
                        "delta_fpr": fpr - alpha if not np.isnan(fpr) else float("nan"),
                        "fpr_violation": int(fpr > alpha) if not np.isnan(fpr) else 0,
                    }
                    per_attack.append(row)
                    rows.append(row)

                # We take the worst case per family (within this seed)
                df_pa = pd.DataFrame(per_attack)
                for fam, sub in df_pa.groupby("family"):
                    if len(sub) == 0: continue
                    i_fpr = sub["empirical_fpr"].idxmax()
                    i_tpr = sub["empirical_tpr"].idxmin()
                    rows.append({
                        "gen_seed": gen_seed, "method": method, "alpha": alpha,
                        "threshold": tau, "attack_id": "_worst", "family": fam,
                        "empirical_fpr": float(sub.loc[i_fpr, "empirical_fpr"]),
                        "empirical_tpr": float(sub.loc[i_tpr, "empirical_tpr"]),
                        "calibration_error": float(sub.loc[i_fpr, "calibration_error"]),
                        "delta_fpr": float(sub.loc[i_fpr, "delta_fpr"]),
                        "fpr_violation": int(sub["fpr_violation"].sum()),
                        "worst_case_fpr": float(sub.loc[i_fpr, "empirical_fpr"]),
                        "worst_case_tpr": float(sub.loc[i_tpr, "empirical_tpr"]),
                        "worst_attack_fpr": str(sub.loc[i_fpr, "attack_id"]),
                        "worst_attack_tpr": str(sub.loc[i_tpr, "attack_id"]),
                    })

    df = pd.DataFrame(rows)
    agg = (df.groupby(["method", "alpha", "attack_id", "family"], dropna=False)
             .agg(empirical_fpr=("empirical_fpr", "mean"),
                  empirical_fpr_std=("empirical_fpr", "std"),
                  empirical_tpr=("empirical_tpr", "mean"),
                  empirical_tpr_std=("empirical_tpr", "std"),
                  calibration_error=("calibration_error", "mean"),
                  delta_fpr=("delta_fpr", "mean"),
                  worst_case_fpr=("worst_case_fpr", "mean"),
                  worst_case_fpr_std=("worst_case_fpr", "std"),
                  worst_case_tpr=("worst_case_tpr", "mean"),
                  worst_case_tpr_std=("worst_case_tpr", "std"),
                  worst_attack_fpr=("worst_attack_fpr",
                                     lambda s: s.mode().iloc[0] if s.notna().any() else float("nan")),
                  worst_attack_tpr=("worst_attack_tpr",
                                     lambda s: s.mode().iloc[0] if s.notna().any() else float("nan")),
                  fpr_violation=("fpr_violation", "sum"))
             .reset_index())
    agg["gen_seed"] = "all"
    agg["threshold"] = float("nan")

    out_df = pd.concat([df, agg], ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[08] E2 written -> {args.out} ({len(df)} per-seed rows + {len(agg)} aggregate rows)")


if __name__ == "__main__":
    main()
