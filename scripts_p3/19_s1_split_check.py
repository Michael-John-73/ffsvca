"""
scripts_p3/19_s1_split_check.py

Reviewer-2 S1 check: is M3's unmodified-condition FPR systematically below the
split-conformal expectation 1 - k/(n+1), and is that due to the single prompt
partition (split_seed=42) shared by all generation seeds?

Steps
  A. Confirm the cal/test prompt partition is identical across seeds.
  B. Reproduce per-seed M3 FPR on `none` from the stored splits.
  C. Compare the 5-seed mean with 1 - k/(n+1) using the Beta(n-k+1, k) + Binomial(m, r)
     model of Remark 1 (independent seeds).
  D. Re-split: R random 50/50 prompt partitions (shared across seeds, as in the
     pipeline) -> distribution of the 5-seed mean FPR; locate split_seed=42 in it.
  E. Cross-seed dependence: correlation of non-watermarked `none` scores of the same
     prompt across generation seeds.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from math import ceil, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ALPHAS = (0.01, 0.05, 0.10)


def m3_fpr(cal: np.ndarray, test: np.ndarray, alpha: float) -> float:
    n = len(cal)
    k = ceil((n + 1) * (1 - alpha))
    tau = np.inf if k > n else np.sort(cal)[k - 1]
    return float(np.mean(test > tau))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("sd21_results/outputs_p3"))
    ap.add_argument("--resplits", type=int, default=2000)
    ap.add_argument("--rng_seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "metrics" / "s1_split_check.txt"
    lines: list[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    raw = pd.read_csv(args.root / "scores" / "scores_raw.csv", dtype={"prompt_id": str})
    nw = raw[(raw.label == 0) & (raw.attack_id == "none")].copy()
    nw["pid"] = nw.prompt_id.astype(int)
    cal = pd.read_csv(args.root / "splits" / "cal_clean.csv")
    test = pd.read_csv(args.root / "splits" / "test_clean.csv")
    cal, test = cal[cal.attack_id == "none"], test[test.attack_id == "none"]
    seeds = sorted(nw.seed.unique())

    log("== A. Partition identity across seeds")
    cal_sets = {s: frozenset(cal[cal.seed == s].prompt_id.astype(int)) for s in seeds}
    same = len(set(cal_sets.values())) == 1
    log(f"cal prompt sets identical across seeds: {same}; |cal|={len(cal_sets[seeds[0]])}")

    log("\n== B. Reproduced M3 FPR on none (stored split_seed=42)")
    obs = {a: [] for a in ALPHAS}
    for s in seeds:
        c = cal[cal.seed == s].score_z.to_numpy()
        t = test[test.seed == s].score_z.to_numpy()
        row = [m3_fpr(c, t, a) for a in ALPHAS]
        for a, v in zip(ALPHAS, row):
            obs[a].append(v)
        log(f"seed {s}: n={len(c)} m={len(t)} FPR@.01={row[0]:.4f} @.05={row[1]:.4f} @.10={row[2]:.4f}")
    n, m = len(cal[cal.seed == seeds[0]]), len(test[test.seed == seeds[0]])
    uniq = nw.groupby("seed").score_z.nunique().min()
    log(f"min unique scores per seed (ties check): {uniq} of {n + m}")

    log("\n== C. Expected value under exchangeability (independent seeds)")
    S = len(seeds)
    for a in ALPHAS:
        k = ceil((n + 1) * (1 - a))
        a_, b_ = n - k + 1, k
        mu = a_ / (n + 1)
        var_r = a_ * b_ / ((a_ + b_) ** 2 * (a_ + b_ + 1))
        # law of total variance: Var(r) + E[r(1-r)]/m
        sd_seed = sqrt(var_r + (mu * (1 - mu) - var_r) / m)
        mean_obs = float(np.mean(obs[a]))
        sd_obs = float(np.std(obs[a], ddof=1))
        z = (mean_obs - mu) / (sd_seed / sqrt(S))
        log(f"alpha={a}: k={k} E[FPR]={mu:.5f} sd/seed={sd_seed:.5f} | "
            f"obs mean={mean_obs:.5f} obs sd={sd_obs:.5f} | z(mean)={z:+.2f} | "
            f"sd ratio obs/theory={sd_obs / sd_seed:.2f}")

    log(f"\n== D. Re-split distribution ({args.resplits} shared random partitions)")
    rng = np.random.default_rng(args.rng_seed)
    pids = np.array(sorted(nw.pid.unique()))
    by_seed = {s: nw[nw.seed == s].set_index("pid").score_z.reindex(pids).to_numpy() for s in seeds}
    n_cal = len(pids) // 2
    means = {a: np.empty(args.resplits) for a in ALPHAS}
    per_seed_all = {a: [] for a in ALPHAS}
    for r in range(args.resplits):
        perm = rng.permutation(len(pids))
        ci, ti = perm[:n_cal], perm[n_cal:]
        for a in ALPHAS:
            vals = [m3_fpr(by_seed[s][ci], by_seed[s][ti], a) for s in seeds]
            means[a][r] = np.mean(vals)
            per_seed_all[a].extend(vals)
    for a in ALPHAS:
        mo = float(np.mean(obs[a]))
        pct = float(np.mean(means[a] <= mo)) * 100
        log(f"alpha={a}: re-split mean of 5-seed mean={means[a].mean():.5f} "
            f"(2.5-97.5%: {np.percentile(means[a], 2.5):.5f}-{np.percentile(means[a], 97.5):.5f}) | "
            f"per-seed re-split sd={np.std(per_seed_all[a], ddof=1):.5f} | "
            f"split_seed=42 value={mo:.5f} at percentile {pct:.1f}%")

    log("\n== E. Cross-seed dependence of non-watermarked none scores (same prompt)")
    mat = np.vstack([by_seed[s] for s in seeds])
    rs = [np.corrcoef(mat[i], mat[j])[0, 1] for i, j in combinations(range(S), 2)]
    log(f"Pearson r across seed pairs: mean={np.mean(rs):.3f} min={np.min(rs):.3f} max={np.max(rs):.3f}")
    grand = mat.mean()
    msb = S * np.sum((mat.mean(axis=0) - grand) ** 2) / (mat.shape[1] - 1)
    msw = np.sum((mat - mat.mean(axis=0)) ** 2) / (mat.shape[1] * (S - 1))
    icc = (msb - msw) / (msb + (S - 1) * msw)
    log(f"ICC(1) of prompt effect: {icc:.3f}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
