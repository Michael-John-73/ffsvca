"""
scripts_p3/10_eval_e4_calibration_size.py

E4: We test calibration sample-size sensitivity.

We keep calibration/test pools per seed and never pool them
across gen_seed (pool_max would otherwise silently become n_cal x 5 seeds).
For each gen_seed, we sweep calibration sizes in {30, 50, 100, 200, 500} (capped
at that seed's own cal_clean@none pool, i.e. 500), draw `--n_repeats` random
subsamples, compute the M3 quantile threshold, and evaluate against THAT
seed's test_clean. We write per-seed rows, plus a cross-seed mean/std
aggregate row (gen_seed = "all").

Our output: outputs_p3/metrics/e4_calibration_size.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import conformal_quantile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/metrics/e4_calibration_size.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    ap.add_argument("--sizes", nargs="+", type=int, default=[30, 50, 100, 200, 500])
    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cal = pd.read_csv(args.splits_dir / "cal_clean.csv")
    tcl = pd.read_csv(args.splits_dir / "test_clean.csv")
    gen_seeds = sorted(set(cal.seed.unique()) | set(tcl.seed.unique()))

    rng = np.random.default_rng(args.seed)
    rows = []
    for gen_seed in gen_seeds:
        cal_none = cal[(cal.seed == gen_seed) & (cal.attack_id == "none")]["score_z"].to_numpy()
        neg = tcl[tcl.seed == gen_seed]["score_z"].to_numpy()
        pool_max = len(cal_none)
        for size in args.sizes:
            eff = min(size, pool_max)
            for alpha in args.alpha:
                fprs_overall = []
                for _ in range(args.n_repeats):
                    samp = rng.choice(cal_none, size=eff, replace=False) \
                           if eff <= pool_max else rng.choice(cal_none, size=eff, replace=True)
                    tau = conformal_quantile(samp, alpha)
                    fprs_overall.append(float(np.mean(neg > tau)))
                rows.append({
                    "gen_seed": gen_seed,
                    "size_requested": size, "size_effective": eff,
                    "alpha": alpha, "n_repeats": args.n_repeats,
                    "fpr_mean": float(np.mean(fprs_overall)),
                    "fpr_std": float(np.std(fprs_overall)),
                    "fpr_min": float(np.min(fprs_overall)),
                    "fpr_max": float(np.max(fprs_overall)),
                    "calibration_error_mean": float(abs(np.mean(fprs_overall) - alpha)),
                })

    df = pd.DataFrame(rows)
    agg = (df.groupby(["size_requested", "alpha"])
             .agg(size_effective=("size_effective", "first"),
                  fpr_mean=("fpr_mean", "mean"), fpr_mean_std=("fpr_mean", "std"),
                  fpr_min=("fpr_min", "min"), fpr_max=("fpr_max", "max"),
                  calibration_error_mean=("calibration_error_mean", "mean"))
             .reset_index())
    agg["gen_seed"] = "all"
    agg["n_repeats"] = args.n_repeats

    out_df = pd.concat([df, agg], ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[10] E4 written -> {args.out} ({len(df)} per-seed rows + {len(agg)} aggregate rows)")


if __name__ == "__main__":
    main()
