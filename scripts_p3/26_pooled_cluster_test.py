"""
scripts_p3/26_pooled_cluster_test.py

Pooled-over-seeds FPR test on the stored SD2.1 scores (no new data). Seeds share one prompt partition,
so the per-seed false-positive indicators of a test prompt are not independent.
  (1) Reproduce the manuscript S3 numbers (18/85 exceedances, 7 unadjusted, 0 Holm/BH, pooled Holm).
  (2) Prompt-cluster test of the pooled FPR: Y_i = # seeds in which test prompt i is a false positive;
      H0: E[Y_i]/5 <= alpha; cluster-robust one-sided z over the 500 test prompts and a prompt-level
      cluster bootstrap; Holm over the 17 conditions. Conditional on the realised calibration sets.
  (3) Design effect and cross-seed correlation of the binary false-positive indicator.
  (4) E7 COCO FPR under none and jpeg_q50 at the seed-0 thresholds.
"""
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ALPHAS = (0.01, 0.05, 0.10)
B = 20000


def tau(cal: np.ndarray, a: float) -> float:
    k = ceil((len(cal) + 1) * (1 - a))
    return np.inf if k > len(cal) else float(np.sort(cal)[k - 1])


def holm(p: np.ndarray, level: float = 0.05) -> np.ndarray:
    rej = np.zeros(len(p), bool)
    for rank, i in enumerate(np.argsort(p)):
        if p[i] <= level / (len(p) - rank):
            rej[i] = True
        else:
            break
    return rej


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    rej = np.zeros(len(p), bool)
    if ok.any():
        rej[order[: np.max(np.nonzero(ok)[0]) + 1]] = True
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs_p3"))
    ROOT = ap.parse_args().root
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    raw = pd.read_csv(ROOT / "scores" / "scores_raw.csv", dtype={"prompt_id": str})
    raw["pid"] = raw.prompt_id.astype(int)
    cal_split = pd.read_csv(ROOT / "splits" / "cal_clean.csv")
    seeds = [int(s) for s in sorted(raw.seed.unique())]
    cal_sets = {s: set(cal_split[(cal_split.attack_id == "none") & (cal_split.seed == s)].prompt_id.astype(int))
                for s in seeds}
    if any(cal_sets[s] != cal_sets[seeds[0]] for s in seeds):
        raise SystemExit("calibration prompts differ across seeds")
    cal_ids = cal_sets[seeds[0]]
    raw["is_cal"] = raw.pid.isin(cal_ids)
    attacks = sorted(raw.attack_id.unique())
    nw = raw[raw.label == 0]
    test_pids = np.array(sorted(set(nw.pid) - cal_ids))
    log(f"seeds={seeds} conditions={len(attacks)} cal prompts={len(cal_ids)} test prompts={len(test_pids)} "
        f"(same partition in every seed: True)")

    def scores(s, a, cal):
        d = nw[(nw.seed == s) & (nw.attack_id == a) & (nw.is_cal == cal)].sort_values("pid")
        return d.pid.to_numpy(), d.score_z.to_numpy()

    rng = np.random.default_rng(0)
    for al in ALPHAS:
        th = {s: tau(scores(s, "none", True)[1], al) for s in seeds}
        obs_p, n_exc, pooled_p, rows = [], 0, [], []
        for a in attacks:
            ind = []
            for s in seeds:
                pid, sc = scores(s, a, False)
                if not np.array_equal(pid, test_pids):
                    raise SystemExit("test prompts not aligned")
                ind.append(sc > th[s])
            ind = np.array(ind, dtype=float)  # seeds x prompts
            m = ind.shape[1]
            for x in ind.sum(1):
                obs_p.append(binom.sf(x - 1, m, al))
                n_exc += int(x / m > al)
            X = ind.sum()
            pooled_p.append(binom.sf(X - 1, m * len(seeds), al))
            y = ind.mean(0)  # per-prompt FP share over seeds
            phat, se = y.mean(), y.std(ddof=1) / np.sqrt(m)
            z = (phat - al) / se if se > 0 else np.inf * np.sign(phat - al)
            boot = y[rng.integers(0, m, size=(B, m))].mean(1)
            p_boot = float(np.mean(boot - phat + al >= phat))  # shifted to H0 boundary
            deff = float(np.var(ind.sum(0), ddof=1) / (len(seeds) * phat * (1 - phat))) if 0 < phat < 1 else np.nan
            cc = np.corrcoef(ind)
            rho = float(np.nanmean(cc[np.triu_indices(len(seeds), 1)])) if ind.std(1).min() > 0 else np.nan
            rows.append(dict(alpha=al, condition=a, X=int(X), fpr_pooled=X / (m * len(seeds)),
                             p_indep=pooled_p[-1], p_cluster_z=float(norm.sf(z)), p_cluster_boot=p_boot,
                             deff=deff, rho_fp=rho))
        obs_p = np.array(obs_p)
        log(f"\nalpha={al}: exceedances {n_exc}/85; unadjusted p<=.05: {int((obs_p <= .05).sum())}; "
            f"Holm: {int(holm(obs_p).sum())}; BH: {int(bh(obs_p).sum())}")
        df = pd.DataFrame(rows)
        df["holm_indep"] = holm(df.p_indep.to_numpy())
        df["holm_cluster_z"] = holm(df.p_cluster_z.to_numpy())
        df["holm_cluster_boot"] = holm(df.p_cluster_boot.to_numpy())
        show = df[(df.fpr_pooled > al) | df.holm_indep]
        log(show[["condition", "X", "fpr_pooled", "p_indep", "holm_indep", "p_cluster_z", "holm_cluster_z",
                  "p_cluster_boot", "holm_cluster_boot", "deff", "rho_fp"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        df.to_csv(ROOT / "metrics" / f"pooled_cluster_test_alpha{al}.csv", index=False)

    coco = pd.read_csv(ROOT / "scores" / "coco_real_scores.csv")
    log("\n== E7 COCO (seed-0 thresholds)")
    for al in ALPHAS:
        t0 = tau(scores(0, "none", True)[1], al)
        for a in ("none", "jpeg_q50"):
            v = coco[coco.attack_id == a].score_z.to_numpy()
            log(f"alpha={al} {a:9s}: FPR={np.mean(v > t0):.4f} ({int(np.sum(v > t0))}/{len(v)})")

    out = ROOT / "metrics" / "pooled_cluster_test.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
