"""
scripts_p3/23_treering_fpr.py  (local)

Second detector (Tree-Ring, plan B2, generation seed 0) evaluated with the same
protocol as ROBIN: split-conformal threshold from calibration clean images on the
unmodified condition (06_compute_thresholds.py, M3), FPR on test clean images for
each of the 17 conditions, TPR/AUC on the unmodified condition.
The same prompt partition (splits/cal_clean.csv, split_seed=42) and the same
seed-0 ROBIN scores (scores_raw.csv) are evaluated side by side.

Writes sd21_results/outputs_p3/treering/metrics/
  treering_fpr.csv       per detector x alpha x condition
  treering_resplit.csv   exceedance frequency over re-drawn 500/500 partitions
  treering_summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import conformal_quantile, roc_auc, wilson_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "sd21_results/outputs_p3"
OUT = R / "treering/metrics"
ALPHAS = (0.01, 0.05, 0.10)
N_RESPLIT = 1000
RESPLIT_SEED = 0


def load() -> dict[str, pd.DataFrame]:
    tr = pd.read_csv(R / "treering/scores/seed_0/treering_scores.csv")
    tr = tr.assign(pid=tr.prompt_id.astype(int), score=-tr.tr_distance)
    raw = pd.read_csv(R / "scores/scores_raw.csv")
    raw = raw[raw.seed == 0]
    raw = raw.assign(pid=raw.prompt_id.astype(int), score=raw.score_z)
    return {"Tree-Ring": tr[["pid", "attack_id", "label", "score"]],
            "ROBIN": raw[["pid", "attack_id", "label", "score"]]}


def matrices(df: pd.DataFrame, attacks: list[str], pids: np.ndarray):
    """clean scores as [n_prompts, n_attacks] and watermarked@none as [n_prompts]."""
    c = df[df.label == 0].pivot(index="pid", columns="attack_id", values="score").loc[pids, attacks]
    w = df[(df.label == 1) & (df.attack_id == "none")].set_index("pid").score.loc[pids]
    return c.to_numpy(), w.to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    det = load()
    attacks = sorted(det["Tree-Ring"].attack_id.unique())
    attacks = ["none"] + [a for a in attacks if a != "none"]
    cal_ids = np.array(sorted(pd.read_csv(R / "splits/cal_clean.csv").query("seed == 0").prompt_id.unique()))
    all_ids = np.array(sorted(det["Tree-Ring"].pid.unique()))
    test_ids = np.setdiff1d(all_ids, cal_ids)
    lines = [f"prompts={len(all_ids)} cal={len(cal_ids)} test={len(test_ids)} (split_seed=42 partition, seed 0)"]

    rows = []
    mats = {}
    for name, df in det.items():
        C, W = matrices(df, attacks, all_ids)
        mats[name] = (C, W)
        idx = {p: i for i, p in enumerate(all_ids)}
        ci = np.array([idx[p] for p in cal_ids]); ti = np.array([idx[p] for p in test_ids])
        auc = roc_auc(C[ti, 0], W[ti])
        lines.append(f"\n[{name}] AUC(none, test) = {auc:.4f}")
        for a in ALPHAS:
            tau = conformal_quantile(C[ci, 0], a)
            tpr = float(np.mean(W[ti] > tau))
            within = 0
            for j, atk in enumerate(attacks):
                x = C[ti, j]; m = len(x); k = int(np.sum(x > tau))
                lo, hi = wilson_ci(k, m)
                p = float(binom.sf(k - 1, m, a))
                ks = float(ks_2samp(C[:, 0], C[:, j]).statistic)
                within += int(k / m <= a)
                rows.append({"detector": name, "alpha": a, "attack_id": atk, "tau": tau, "m": m,
                             "fp": k, "fpr": k / m, "wilson_lo": lo, "wilson_hi": hi,
                             "p_exceed_one_sided": p, "ks_vs_none": ks, "tpr_none": tpr})
            sub = [r for r in rows if r["detector"] == name and r["alpha"] == a]
            worst = max(sub, key=lambda r: r["fpr"])
            lines.append(f"  alpha={a:.2f} tau={tau:.4f} TPR(none)={tpr:.3f} "
                         f"FPR<=alpha in {within}/17 conditions; "
                         f"FPR(none)={sub[0]['fpr']:.4f}; max FPR={worst['fpr']:.4f} ({worst['attack_id']})")
    fpr = pd.DataFrame(rows)
    fpr.to_csv(OUT / "treering_fpr.csv", index=False)

    lines.append("\n[per-condition FPR, test split]  alpha: Tree-Ring | ROBIN")
    for atk in attacks:
        s = []
        for a in ALPHAS:
            t = fpr.query("detector=='Tree-Ring' and alpha==@a and attack_id==@atk").iloc[0]
            r = fpr.query("detector=='ROBIN' and alpha==@a and attack_id==@atk").iloc[0]
            s.append(f"{a:.2f}: {t.fpr:.3f} | {r.fpr:.3f}")
        ks_t = fpr.query("detector=='Tree-Ring' and attack_id==@atk").ks_vs_none.iloc[0]
        ks_r = fpr.query("detector=='ROBIN' and attack_id==@atk").ks_vs_none.iloc[0]
        lines.append(f"  {atk:24s} " + "   ".join(s) + f"   KS {ks_t:.3f} | {ks_r:.3f}")

    # re-drawn partitions: how often each condition exceeds alpha
    rng = np.random.default_rng(RESPLIT_SEED)
    n = len(all_ids); n_cal = len(cal_ids)
    exc = {(d, a): np.zeros(len(attacks)) for d in det for a in ALPHAS}
    for b in range(N_RESPLIT):
        perm = rng.permutation(n); ci, ti = perm[:n_cal], perm[n_cal:]
        for d, (C, _) in mats.items():
            for a in ALPHAS:
                tau = conformal_quantile(C[ci, 0], a)
                exc[(d, a)] += (np.mean(C[ti] > tau, axis=0) > a)
        if (b + 1) % 20 == 0 or b + 1 == N_RESPLIT:
            f = (b + 1) * 30 // N_RESPLIT
            print(f"\r[RESPLIT] [{'#' * f}{'.' * (30 - f)}] {b + 1}/{N_RESPLIT}", end="", flush=True)
    print()
    rs = [{"detector": d, "alpha": a, "attack_id": atk, "exceed_freq": exc[(d, a)][j] / N_RESPLIT}
          for (d, a) in exc for j, atk in enumerate(attacks)]
    rs = pd.DataFrame(rs)
    rs.to_csv(OUT / "treering_resplit.csv", index=False)
    lines.append(f"\n[re-split] P(FPR > alpha) over {N_RESPLIT} partitions (rng seed {RESPLIT_SEED})  "
                 "alpha 0.01/0.05/0.10: Tree-Ring | ROBIN")
    for atk in attacks:
        t = [rs.query("detector=='Tree-Ring' and alpha==@a and attack_id==@atk").exceed_freq.iloc[0] for a in ALPHAS]
        r = [rs.query("detector=='ROBIN' and alpha==@a and attack_id==@atk").exceed_freq.iloc[0] for a in ALPHAS]
        lines.append(f"  {atk:24s} " + "/".join(f"{v:.2f}" for v in t) + " | " + "/".join(f"{v:.2f}" for v in r))

    text = "\n".join(lines) + "\n"
    (OUT / "treering_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
