"""
scripts_p3/07_eval_e1_threshold_comparison.py

E1: We compare threshold methods (M1 vs M2 vs M3) on the CLEAN test pool.

We compute thresholds independently per gen_seed and never pool them. We
therefore evaluate each gen_seed's test split against THAT seed's threshold,
then report both the per-seed rows and the cross-seed aggregate (mean ± std
over the 5 seeds), since we report results as mean ± std across seeds.

For each method, alpha, gen_seed we compute:
    - empirical_fpr on that seed's test_clean.csv (per attack)
    - empirical_tpr on that seed's test_watermarked.csv (per attack)
    - calibration_error = |empirical_fpr - alpha|
    - auc on that seed's test_clean / test_watermarked@none

We write: outputs_p3/metrics/e1_threshold_comparison.csv
    (we store an int in the gen_seed column for per-seed rows, or "all" for the
    cross-seed mean/std aggregate row of the same method/alpha/attack_id)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (calibration_error, empirical_fpr, empirical_tpr,
                     mcnemar_test, roc_auc, threshold_for, wilson_ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs_p3/thresholds/thresholds_by_method.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/metrics/e1_threshold_comparison.csv"))
    ap.add_argument("--mcnemar_out", type=Path,
                    default=Path("outputs_p3/metrics/e1_mcnemar.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    args = ap.parse_args()

    tcl = pd.read_csv(args.splits_dir / "test_clean.csv")
    twm = pd.read_csv(args.splits_dir / "test_watermarked.csv")
    thr = pd.read_csv(args.thresholds)

    gen_seeds = sorted(set(tcl.seed.unique()) | set(twm.seed.unique()))
    attack_ids = sorted(set(tcl.attack_id.unique()) | set(twm.attack_id.unique()))

    rows = []
    mc_rows = []
    for gen_seed in gen_seeds:
        tcl_s = tcl[tcl.seed == gen_seed]
        twm_s = twm[twm.seed == gen_seed]
        tcl_none = tcl_s[tcl_s.attack_id == "none"]["score_z"].to_numpy()
        twm_none = twm_s[twm_s.attack_id == "none"]["score_z"].to_numpy()
        auc_none = roc_auc(tcl_none, twm_none)

        for alpha in args.alpha:
            for method in ["M1", "M2", "M3"]:
                tau = threshold_for(thr, method, alpha, gen_seed=gen_seed)
                for atk in attack_ids:
                    neg = tcl_s[tcl_s.attack_id == atk]["score_z"].to_numpy()
                    pos = twm_s[twm_s.attack_id == atk]["score_z"].to_numpy()
                    fpr = empirical_fpr(neg, tau)
                    tpr = empirical_tpr(pos, tau)
                    ce = calibration_error(fpr, alpha)
                    fpr_lo, fpr_hi = wilson_ci(int(np.sum(neg > tau)), len(neg))
                    rows.append({
                        "gen_seed": gen_seed, "method": method, "alpha": alpha,
                        "threshold": tau, "attack_id": atk,
                        "n_clean": len(neg), "n_wm": len(pos),
                        "empirical_fpr": fpr, "empirical_tpr": tpr,
                        "calibration_error": ce,
                        "fpr_ci_lo": fpr_lo, "fpr_ci_hi": fpr_hi,
                        "auc_none": auc_none,
                    })

        # We run a paired McNemar test on the `none` attack only (M3 vs M1, M3 vs M2), per seed.
        none_scores = np.concatenate([tcl_none, twm_none])
        none_labels = np.concatenate([
            np.zeros_like(tcl_none, dtype=int), np.ones_like(twm_none, dtype=int)])
        for alpha in args.alpha:
            tau_m3 = threshold_for(thr, "M3", alpha, gen_seed=gen_seed)
            for ref in ["M1", "M2"]:
                tau_ref = threshold_for(thr, ref, alpha, gen_seed=gen_seed)
                b, c, chi2, p = mcnemar_test(none_scores, none_labels, tau_m3, tau_ref)
                mc_rows.append({
                    "gen_seed": gen_seed, "alpha": alpha,
                    "method_a": "M3", "method_b": ref, "attack_id": "none",
                    "n_items": int(len(none_scores)),
                    "b_a_correct_b_wrong": b, "c_a_wrong_b_correct": c,
                    "mcnemar_chi2": chi2, "mcnemar_p": p,
                })

    df = pd.DataFrame(rows)
    agg = (df.groupby(["method", "alpha", "attack_id"])
             .agg(n_clean=("n_clean", "sum"), n_wm=("n_wm", "sum"),
                  empirical_fpr=("empirical_fpr", "mean"),
                  empirical_fpr_std=("empirical_fpr", "std"),
                  empirical_tpr=("empirical_tpr", "mean"),
                  empirical_tpr_std=("empirical_tpr", "std"),
                  calibration_error=("calibration_error", "mean"),
                  auc_none=("auc_none", "mean"))
             .reset_index())
    agg["gen_seed"] = "all"
    agg["threshold"] = float("nan")
    agg["fpr_ci_lo"] = float("nan")
    agg["fpr_ci_hi"] = float("nan")

    out_df = pd.concat([df, agg], ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[07] E1 written -> {args.out} ({len(df)} per-seed rows + {len(agg)} aggregate rows)")

    args.mcnemar_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mc_rows).to_csv(args.mcnemar_out, index=False)
    print(f"[07] E1 McNemar written -> {args.mcnemar_out}")


if __name__ == "__main__":
    main()
