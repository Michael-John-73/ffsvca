"""
scripts_p3/05_make_splits.py

Phase 2.5 / 4.1: split CLEAN and WATERMARKED scores into calibration / test
pools by a shared prompt_id partition. Uses --split_seed for reproducibility.

Input:  outputs_p3/scores/scores_raw.csv
Output: outputs_p3/splits/{cal_clean.csv, cal_watermarked.csv, test_clean.csv, test_watermarked.csv}

Both cal and test watermarked pools are restricted to the SAME prompt_id
partition as clean (cal_ids / test_ids), so M1/M2 (paper3.md §5.2) can be
computed from cal_clean vs cal_watermarked without any test-split leakage.
Every row keeps its `seed` (gen_seed) column so 06_compute_thresholds.py can
group thresholds independently per seed (paper3.md §6.5).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path,
                    default=Path("outputs_p3/scores/scores_raw.csv"))
    ap.add_argument("--out_dir", type=Path, default=Path("outputs_p3/splits"))
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.scores)

    # Partition prompt_ids using the CLEAN none-attack pool (deterministic, shared
    # by clean and watermarked so cal/test membership is consistent across labels).
    clean_none = df[(df.label == 0) & (df.attack_id == "none")].copy()
    prompt_ids = sorted(clean_none["prompt_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(prompt_ids))
    n_cal = int(len(prompt_ids) * args.cal_ratio)
    cal_ids = set(prompt_ids[i] for i in perm[:n_cal])
    test_ids = set(prompt_ids[i] for i in perm[n_cal:])

    pid = df.prompt_id.astype(str)
    cal_clean = df[(df.label == 0) & pid.isin(cal_ids)]
    cal_wm = df[(df.label == 1) & pid.isin(cal_ids)]
    test_clean = df[(df.label == 0) & pid.isin(test_ids)]
    test_wm = df[(df.label == 1) & pid.isin(test_ids)]

    cal_clean.to_csv(args.out_dir / "cal_clean.csv", index=False)
    cal_wm.to_csv(args.out_dir / "cal_watermarked.csv", index=False)
    test_clean.to_csv(args.out_dir / "test_clean.csv", index=False)
    test_wm.to_csv(args.out_dir / "test_watermarked.csv", index=False)
    print(f"[05] cal_clean={len(cal_clean)}, cal_wm={len(cal_wm)}, "
          f"test_clean={len(test_clean)}, test_wm={len(test_wm)} "
          f"(cal_ratio={args.cal_ratio}, seed={args.split_seed})")


if __name__ == "__main__":
    main()
