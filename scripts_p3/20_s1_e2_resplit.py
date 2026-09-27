"""
scripts_p3/20_s1_e2_resplit.py

Reviewer-2 S1 follow-up for E2: the prompt partition (split_seed=42) is shared by all
generation seeds. Re-draw R random shared 50/50 prompt partitions, recompute the M3
threshold on non-watermarked `none` calibration scores per seed, and evaluate the
non-watermarked test FPR under every evaluation condition. Reports, per condition and
alpha, where the split_seed=42 result lies in the re-split distribution, and the
re-split distribution of the within-budget count over the 85 condition-seed observations.
"""
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

ALPHAS = (0.01, 0.05, 0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("sd21_results/outputs_p3"))
    ap.add_argument("--resplits", type=int, default=1000)
    ap.add_argument("--rng_seed", type=int, default=12345)
    ap.add_argument("--split_seed", type=int, default=42)
    args = ap.parse_args()
    out_txt = args.root / "metrics" / "s1_e2_resplit.txt"
    out_csv = args.root / "metrics" / "s1_e2_resplit.csv"
    lines: list[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    raw = pd.read_csv(args.root / "scores" / "scores_raw.csv", dtype={"prompt_id": str})
    nw = raw[raw.label == 0].copy()
    nw["pid"] = nw.prompt_id.astype(int)
    pids = np.array(sorted(nw.pid.unique()))
    seeds = sorted(nw.seed.unique())
    attacks = sorted(nw.attack_id.unique())
    # score[seed][attack] -> array aligned to pids
    score = {s: {a: g.set_index("pid").score_z.reindex(pids).to_numpy()
                 for a, g in nw[nw.seed == s].groupby("attack_id")} for s in seeds}
    n_cal = len(pids) // 2
    ks = {al: ceil((n_cal + 1) * (1 - al)) for al in ALPHAS}

    def fpr_matrix(ci, ti):
        """-> dict alpha -> array [len(attacks), len(seeds)] of test FPR."""
        res = {al: np.empty((len(attacks), len(seeds))) for al in ALPHAS}
        for j, s in enumerate(seeds):
            cal_sorted = np.sort(score[s]["none"][ci])
            for al in ALPHAS:
                k = ks[al]
                tau = np.inf if k > n_cal else cal_sorted[k - 1]
                for i, a in enumerate(attacks):
                    res[al][i, j] = np.mean(score[s][a][ti] > tau)
        return res

    # pipeline partition, read from the stored split so it matches 05_make_splits.py exactly
    cal_split = pd.read_csv(args.root / "splits" / "cal_clean.csv")
    cal_ids = set(cal_split[(cal_split.attack_id == "none") & (cal_split.seed == seeds[0])]
                  .prompt_id.astype(int))
    ci0 = np.array([i for i, p in enumerate(pids) if p in cal_ids])
    ti0 = np.array([i for i, p in enumerate(pids) if p not in cal_ids])
    assert len(ci0) == n_cal, (len(ci0), n_cal)
    obs = fpr_matrix(ci0, ti0)

    rng = np.random.default_rng(args.rng_seed)
    R = args.resplits
    sims = {al: np.empty((R, len(attacks), len(seeds))) for al in ALPHAS}
    for r in range(R):
        p = rng.permutation(len(pids))
        res = fpr_matrix(p[:n_cal], p[n_cal:])
        for al in ALPHAS:
            sims[al][r] = res[al]
        if (r + 1) % 100 == 0:
            print(f"  resplit {r + 1}/{R}", flush=True)

    rows = []
    for al in ALPHAS:
        log(f"\n== alpha={al} (split_seed={args.split_seed} vs {R} re-splits)")
        log(f"{'condition':26s} obs_mean  rs_mean  rs_2.5%  rs_97.5%  pct  P(mean>a)  obs_exc  rs_exc")
        for i, a in enumerate(attacks):
            o = obs[al][i].mean()
            sm = sims[al][:, i, :].mean(axis=1)
            pct = float(np.mean(sm <= o)) * 100
            p_exc = float(np.mean(sm > al))
            o_exc = int(np.sum(obs[al][i] > al))
            rs_exc = float(np.mean(np.sum(sims[al][:, i, :] > al, axis=1)))
            log(f"{a:26s} {o:.4f}   {sm.mean():.4f}   {np.percentile(sm, 2.5):.4f}   "
                f"{np.percentile(sm, 97.5):.4f}   {pct:5.1f}  {p_exc:.3f}      {o_exc}        {rs_exc:.2f}")
            rows.append(dict(alpha=al, attack_id=a, obs_seed_mean=o, resplit_mean=sm.mean(),
                             resplit_p2_5=np.percentile(sm, 2.5), resplit_p97_5=np.percentile(sm, 97.5),
                             obs_percentile=pct, resplit_p_mean_gt_alpha=p_exc,
                             obs_exceed_seeds=o_exc, resplit_mean_exceed_seeds=rs_exc))
        within_obs = int(np.sum(obs[al] <= al))
        within_rs = np.sum(sims[al] <= al, axis=(1, 2))
        cond_obs = int(np.sum(obs[al].mean(axis=1) > al))
        cond_rs = np.sum(sims[al].mean(axis=2) > al, axis=1)
        log(f"within-budget obs (of 85): {within_obs} | re-split mean={within_rs.mean():.1f} "
            f"(2.5-97.5%: {np.percentile(within_rs, 2.5):.0f}-{np.percentile(within_rs, 97.5):.0f}); "
            f"P(re-split <= obs)={np.mean(within_rs <= within_obs):.3f}")
        log(f"conditions with seed-mean FPR > alpha: obs={cond_obs} | re-split mean={cond_rs.mean():.2f} "
            f"(2.5-97.5%: {np.percentile(cond_rs, 2.5):.0f}-{np.percentile(cond_rs, 97.5):.0f})")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwritten -> {out_txt}, {out_csv}")


if __name__ == "__main__":
    main()
