"""
scripts_p3/_common.py
Shared helpers we use in the E1-E5 evaluation scripts (07-11).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """We compute the split-conformal threshold: order statistic z_(k), k = ceil((n+1)(1-alpha)).

    We return +inf when k > n, i.e. when n is too small to certify the level.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(s[k - 1]) if k <= n else float("inf")


def empirical_fpr(scores: np.ndarray, tau: float) -> float:
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores > tau))


def empirical_tpr(scores: np.ndarray, tau: float) -> float:
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores > tau))


def calibration_error(fpr: float, alpha: float) -> float:
    if np.isnan(fpr):
        return float("nan")
    return float(abs(fpr - alpha))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    rad = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(max(0.0, center - rad)), float(min(1.0, center + rad)))


def roc_auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """We compute the Mann-Whitney AUC: P(score(pos) > score(neg))."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros_like(neg, dtype=int),
                        np.ones_like(pos, dtype=int)])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = len(pos)
    n_neg = len(neg)
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def threshold_for(thresholds: pd.DataFrame, method: str, alpha: float, gen_seed=None) -> float:
    sub = thresholds[(thresholds.method == method) & (np.isclose(thresholds.alpha, alpha))]
    if gen_seed is not None:
        sub = sub[sub.gen_seed == gen_seed]
    if len(sub) == 0:
        return float("nan")
    if gen_seed is None and sub["gen_seed"].nunique() > 1:
        raise ValueError(
            f"threshold_for({method}, alpha={alpha}) matched {sub['gen_seed'].nunique()} "
            "distinct gen_seed rows; pass gen_seed= explicitly (we compute thresholds "
            "per seed and never pool them).")
    return float(sub.iloc[0]["threshold"])


def mcnemar_test(
    scores: np.ndarray,
    labels: np.ndarray,
    tau_a: float,
    tau_b: float,
) -> tuple[int, int, float, float]:
    """We run a paired McNemar test on binary decisions of two thresholds A and B over
    the same items. We return (b, c, chi2, p_value) where:
        b = # items where A correct and B wrong
        c = # items where A wrong   and B correct
    We report the continuity-corrected statistic ((|b - c| - 1)^2) / (b + c) and the
    exact two-sided binomial McNemar p-value. If b + c < 1, we return (b, c, nan, nan).
    """
    if len(scores) != len(labels) or len(scores) == 0:
        return (0, 0, float("nan"), float("nan"))
    pred_a = (scores > tau_a).astype(int)
    pred_b = (scores > tau_b).astype(int)
    correct_a = (pred_a == labels)
    correct_b = (pred_b == labels)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if (b + c) < 1:
        return (b, c, float("nan"), float("nan"))
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    from math import comb
    n, k = b + c, min(b, c)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return (b, c, float(chi2), float(p))
