"""
scripts_p3/12_compute_ci.py

We compute bootstrap 95% confidence intervals for empirical_fpr / empirical_tpr
per (method, alpha, attack_id). We resample test_clean and test_watermarked
prompt-IDs with replacement, independently within each gen_seed (we keep
thresholds/splits per seed and never pool them), then append
a cross-seed mean/std aggregate row (gen_seed = "all").

We write: outputs_p3/metrics/ci_bootstrap.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import threshold_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--thresholds", type=Path,
                    default=Path("outputs_p3/thresholds/thresholds_by_method.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/metrics/ci_bootstrap.csv"))
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tcl = pd.read_csv(args.splits_dir / "test_clean.csv")
    twm = pd.read_csv(args.splits_dir / "test_watermarked.csv")
    thr = pd.read_csv(args.thresholds)

    rng = np.random.default_rng(args.seed)
    attacks = sorted(set(tcl.attack_id.unique()) | set(twm.attack_id.unique()))
    gen_seeds = sorted(set(tcl.seed.unique()) | set(twm.seed.unique()))

    rows = []
    for gen_seed in gen_seeds:
        tcl_s = tcl[tcl.seed == gen_seed]
        twm_s = twm[twm.seed == gen_seed]
        for alpha in args.alpha:
            for method in ["M1", "M2", "M3"]:
                tau = threshold_for(thr, method, alpha, gen_seed=gen_seed)
                for atk in attacks:
                    neg = tcl_s[tcl_s.attack_id == atk]["score_z"].to_numpy()
                    pos = twm_s[twm_s.attack_id == atk]["score_z"].to_numpy()
                    if len(neg) == 0 and len(pos) == 0:
                        continue
                    fpr_samples = []; tpr_samples = []
                    for _ in range(args.n_bootstrap):
                        if len(neg) > 0:
                            s = rng.choice(neg, size=len(neg), replace=True)
                            fpr_samples.append(float(np.mean(s > tau)))
                        if len(pos) > 0:
                            s = rng.choice(pos, size=len(pos), replace=True)
                            tpr_samples.append(float(np.mean(s > tau)))
                    rows.append({
                        "gen_seed": gen_seed, "method": method, "alpha": alpha, "attack_id": atk,
                        "fpr_mean": float(np.mean(fpr_samples)) if fpr_samples else float("nan"),
                        "fpr_ci_lo": float(np.quantile(fpr_samples, 0.025)) if fpr_samples else float("nan"),
                        "fpr_ci_hi": float(np.quantile(fpr_samples, 0.975)) if fpr_samples else float("nan"),
                        "tpr_mean": float(np.mean(tpr_samples)) if tpr_samples else float("nan"),
                        "tpr_ci_lo": float(np.quantile(tpr_samples, 0.025)) if tpr_samples else float("nan"),
                        "tpr_ci_hi": float(np.quantile(tpr_samples, 0.975)) if tpr_samples else float("nan"),
                        "n_bootstrap": args.n_bootstrap,
                    })

    df = pd.DataFrame(rows)
    agg = (df.groupby(["method", "alpha", "attack_id"])
             .agg(fpr_mean=("fpr_mean", "mean"), fpr_mean_std=("fpr_mean", "std"),
                  fpr_ci_lo=("fpr_ci_lo", "mean"), fpr_ci_hi=("fpr_ci_hi", "mean"),
                  tpr_mean=("tpr_mean", "mean"), tpr_mean_std=("tpr_mean", "std"),
                  tpr_ci_lo=("tpr_ci_lo", "mean"), tpr_ci_hi=("tpr_ci_hi", "mean"))
             .reset_index())
    agg["gen_seed"] = "all"
    agg["n_bootstrap"] = args.n_bootstrap

    out_df = pd.concat([df, agg], ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[12] CI written -> {args.out} ({len(df)} per-seed rows + {len(agg)} aggregate rows)")


if __name__ == "__main__":
    main()
