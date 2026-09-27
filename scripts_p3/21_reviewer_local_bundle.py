"""
scripts_p3/21_reviewer_local_bundle.py

Local analyses requested in the reviewer response plan (stored SD2.1 scores, stored split):
  S2  KS distance between the non-watermarked score distributions under `none` and under
      each condition (a lower bound on the total-variation distance of Definition 3), and
      the empirical FPR shift at the M3 threshold.
  S3  Multiplicity: one-sided exact binomial p-values for H0: r_a(c) <= alpha per
      condition-seed (conditional on the realized calibration set, Remark 1), with Holm
      and Benjamini-Hochberg across the 85 observations; pooled-over-seeds tests per
      condition with Holm across the 17 conditions.
  S6  Power of the one-sided exact binomial test (level 0.05) and of the rule
      "empirical FPR > alpha" to detect true FPRs above the budget, for m=500 and m=2500.
  M4  Augmented calibration: conformal threshold on the union of non-watermarked
      calibration scores of all 17 conditions.
  M5  Alternative same-objective rules: (a) calibration-conditional (PAC) threshold with
      P(r(C) <= alpha) >= 1-delta, delta=0.1; (b) condition-wise oracle calibration.
"""
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, ks_2samp

ALPHAS = (0.01, 0.05, 0.10)
KEY_CONDS = ("none", "noise_s005", "blurring+noise", "jpeg_q50", "cropping", "rotation", "jpeg+cropping")


def conformal_tau(cal: np.ndarray, alpha: float) -> float:
    n = len(cal)
    k = ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else float(np.sort(cal)[k - 1])


def pac_tau(cal: np.ndarray, alpha: float, delta: float) -> float:
    """Smallest order statistic z_(k) with P(r(C) <= alpha) >= 1-delta, r ~ Beta(n-k+1, k)."""
    n = len(cal)
    s = np.sort(cal)
    for k in range(1, n + 1):
        if beta.cdf(alpha, n - k + 1, k) >= 1 - delta:
            return float(s[k - 1])
    return np.inf


def holm(p: np.ndarray, level: float = 0.05) -> np.ndarray:
    order = np.argsort(p)
    rej = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] <= level / (len(p) - rank):
            rej[i] = True
        else:
            break
    return rej


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] <= q * (np.arange(1, len(p) + 1) / len(p))
    rej = np.zeros(len(p), bool)
    if ranked.any():
        rej[order[: np.max(np.nonzero(ranked)[0]) + 1]] = True
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("sd21_results/outputs_p3"))
    ap.add_argument("--delta", type=float, default=0.10)
    args = ap.parse_args()
    out_txt = args.root / "metrics" / "reviewer_local_bundle.txt"
    out_csv = args.root / "metrics" / "reviewer_local_bundle_methods.csv"
    lines: list[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    raw = pd.read_csv(args.root / "scores" / "scores_raw.csv", dtype={"prompt_id": str})
    raw["pid"] = raw.prompt_id.astype(int)
    cal_split = pd.read_csv(args.root / "splits" / "cal_clean.csv")
    seeds = sorted(raw.seed.unique())
    cal_ids = set(cal_split[(cal_split.attack_id == "none") & (cal_split.seed == seeds[0])]
                  .prompt_id.astype(int))
    attacks = sorted(raw.attack_id.unique())
    raw["is_cal"] = raw.pid.isin(cal_ids)

    def arr(s, a, label, cal):
        d = raw[(raw.seed == s) & (raw.attack_id == a) & (raw.label == label) & (raw.is_cal == cal)]
        return d.sort_values("pid").score_z.to_numpy()

    D = {(s, a): dict(cal_nw=arr(s, a, 0, True), test_nw=arr(s, a, 0, False),
                      test_wm=arr(s, a, 1, False), all_nw=np.concatenate([arr(s, a, 0, True), arr(s, a, 0, False)]))
         for s in seeds for a in attacks}
    m = len(D[(seeds[0], "none")]["test_nw"])
    n = len(D[(seeds[0], "none")]["cal_nw"])
    log(f"seeds={seeds} conditions={len(attacks)} n_cal={n} m_test={m}")

    # ---------------- S2
    log("\n== S2: KS distance (non-watermarked none vs condition), seed mean over 5 seeds")
    log(f"{'condition':26s} KS_mean  KS_min  KS_max  |dFPR|@M3,a=.05 (mean)")
    s2_rows = []
    for a in attacks:
        ks = [ks_2samp(D[(s, "none")]["all_nw"], D[(s, a)]["all_nw"]).statistic for s in seeds]
        dfpr = []
        for s in seeds:
            tau = conformal_tau(D[(s, "none")]["cal_nw"], 0.05)
            dfpr.append(abs(np.mean(D[(s, a)]["test_nw"] > tau) - np.mean(D[(s, "none")]["test_nw"] > tau)))
        s2_rows.append((a, np.mean(ks), np.min(ks), np.max(ks), np.mean(dfpr)))
        log(f"{a:26s} {np.mean(ks):.3f}    {np.min(ks):.3f}   {np.max(ks):.3f}   {np.mean(dfpr):.4f}")

    # ---------------- S3
    log("\n== S3: multiplicity (one-sided exact binomial, H0: r_a(c) <= alpha, conditional on C)")
    for al in ALPHAS:
        obs_p, obs_lab, pooled = [], [], {}
        for a in attacks:
            xs = []
            for s in seeds:
                tau = conformal_tau(D[(s, "none")]["cal_nw"], al)
                x = int(np.sum(D[(s, a)]["test_nw"] > tau))
                xs.append(x)
                obs_p.append(binom.sf(x - 1, m, al))
                obs_lab.append((a, s, x))
            X = sum(xs)
            pooled[a] = (X, binom.sf(X - 1, m * len(seeds), al))
        obs_p = np.array(obs_p)
        raw_rej = int(np.sum(obs_p <= 0.05))
        h, b = holm(obs_p), bh(obs_p)
        exceed = int(sum(1 for (_, _, x) in obs_lab if x / m > al))
        log(f"alpha={al}: observed exceedances={exceed}/85; unadjusted p<=.05: {raw_rej}; "
            f"Holm: {int(h.sum())}; BH(q=.05): {int(b.sum())}")
        sig = sorted({obs_lab[i][0] for i in np.nonzero(b)[0]})
        log(f"   BH-significant conditions (any seed): {', '.join(sig) if sig else '-'}")
        pa = np.array([pooled[a][1] for a in attacks])
        hp = holm(pa)
        log("   pooled over seeds (N=2500, seeds share one partition -> approximate): Holm-significant: "
            + (", ".join(f"{a} (X={pooled[a][0]}, FPR={pooled[a][0] / (m * len(seeds)):.4f}, p={pooled[a][1]:.2e})"
                         for a, r in zip(attacks, hp) if r) or "-"))

    # ---------------- S6
    log("\n== S6: power to detect true FPR above the budget (one-sided exact binomial, level 0.05)")
    for al in ALPHAS:
        for M in (500, 2500):
            c = int(binom.isf(0.05, M, al)) + 1  # smallest c with P(X>=c | al) <= .05
            while binom.sf(c - 1, M, al) > 0.05:
                c += 1
            parts = []
            for mult in (1.2, 1.4, 1.6, 2.0):
                p = al * mult
                pw = binom.sf(c - 1, M, p)
                pr = binom.sf(int(np.floor(al * M)), M, p)  # P(empirical > alpha)
                parts.append(f"FPR={p:.3f}: power={pw:.2f}, P(emp>a)={pr:.2f}")
            log(f"alpha={al} m={M} (reject if count >= {c}; size={binom.sf(c - 1, M, al):.3f}): " + "; ".join(parts))

    # ---------------- M4 / M5
    log(f"\n== M4/M5: alternative same-objective rules (delta={args.delta} for PAC)")
    rows = []
    for al in ALPHAS:
        for meth in ("M3", "M4-union", "M5a-PAC", "M5b-oracle"):
            fpr = np.empty((len(attacks), len(seeds)))
            tpr = np.empty_like(fpr)
            for j, s in enumerate(seeds):
                if meth == "M3":
                    tau_all = {a: conformal_tau(D[(s, "none")]["cal_nw"], al) for a in attacks}
                elif meth == "M4-union":
                    t = conformal_tau(np.concatenate([D[(s, a)]["cal_nw"] for a in attacks]), al)
                    tau_all = {a: t for a in attacks}
                elif meth == "M5a-PAC":
                    t = pac_tau(D[(s, "none")]["cal_nw"], al, args.delta)
                    tau_all = {a: t for a in attacks}
                else:
                    tau_all = {a: conformal_tau(D[(s, a)]["cal_nw"], al) for a in attacks}
                for i, a in enumerate(attacks):
                    fpr[i, j] = np.mean(D[(s, a)]["test_nw"] > tau_all[a])
                    tpr[i, j] = np.mean(D[(s, a)]["test_wm"] > tau_all[a])
            within = int(np.sum(fpr <= al))
            cond_exc = int(np.sum(fpr.mean(axis=1) > al))
            key = {c: (fpr[attacks.index(c)].mean(), tpr[attacks.index(c)].mean()) for c in KEY_CONDS}
            log(f"alpha={al} {meth:11s}: within-budget {within}/85; conditions seed-mean>a: {cond_exc}; "
                f"mean TPR over conditions={tpr.mean():.4f}; none FPR/TPR={key['none'][0]:.4f}/{key['none'][1]:.4f}")
            log("      " + "; ".join(f"{c}: {key[c][0]:.4f}/{key[c][1]:.4f}" for c in KEY_CONDS[1:]))
            for i, a in enumerate(attacks):
                rows.append(dict(alpha=al, method=meth, attack_id=a, fpr_seed_mean=fpr[i].mean(),
                                 tpr_seed_mean=tpr[i].mean(), exceed_seeds=int(np.sum(fpr[i] > al))))

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nwritten -> {out_txt}, {out_csv}")


if __name__ == "__main__":
    main()
