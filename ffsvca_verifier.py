"""
ffsvca_verifier.py
Fixed-FPR threshold computation and decision rule.
Implements M1 (z-score), M2 (normalized rank), M3 (bootstrap-CI) thresholds.

Protocol:
    - Score:     z(x) = -d(x)
    - Threshold: tau_alpha = Q_{1-alpha}(Z_0_cal)  (M1/M2)
    - Decision:  phi_alpha(x) = 1[z(x) > tau_alpha]

Corresponds to MWDRAS/mwdras_meta_learners.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Threshold methods
# ---------------------------------------------------------------------------
MethodName = Literal["M1", "M2", "M3"]


def compute_threshold_m1(
    cal_scores: np.ndarray,
    alpha: float,
) -> float:
    """
    M1: Empirical quantile threshold.
    tau_alpha = Q_{1-alpha}(Z_0_cal)
    Uses the (1-alpha)-quantile of CLEAN calibration scores.
    """
    return float(np.quantile(cal_scores, 1.0 - alpha))


def compute_threshold_m2(
    cal_scores: np.ndarray,
    alpha: float,
) -> float:
    """
    M2: Rank-normalized threshold.
    tau_alpha = Q_{1-alpha}(rank(Z_0_cal) / n)
    Normalized ranks in [0, 1] before quantile.
    """
    n = len(cal_scores)
    ranks = (np.argsort(np.argsort(cal_scores)) + 1) / n
    return float(np.quantile(ranks, 1.0 - alpha))


def compute_threshold_m3(
    cal_scores: np.ndarray,
    alpha: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    M3: Bootstrap upper-CI threshold.
    Bootstrap B times, take mean of (1-alpha)-quantile estimates.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(cal_scores)
    quantiles = []
    for _ in range(n_bootstrap):
        sample = rng.choice(cal_scores, size=n, replace=True)
        quantiles.append(np.quantile(sample, 1.0 - alpha))
    return float(np.mean(quantiles))


def compute_all_thresholds(
    cal_scores: np.ndarray,
    alpha: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute M1/M2/M3 thresholds for a given alpha.
    Returns dict with keys "M1", "M2", "M3".
    """
    rng = np.random.default_rng(seed)
    return {
        "M1": compute_threshold_m1(cal_scores, alpha),
        "M2": compute_threshold_m2(cal_scores, alpha),
        "M3": compute_threshold_m3(cal_scores, alpha, n_bootstrap=n_bootstrap, rng=rng),
    }


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
def phi_alpha(z_score: float, tau: float) -> int:
    """phi_alpha(x) = 1[z(x) > tau_alpha]"""
    return int(z_score > tau)


def apply_decision(scores: np.ndarray, tau: float) -> np.ndarray:
    """Apply phi_alpha to an array of z-scores. Returns binary predictions."""
    return (scores > tau).astype(int)


# ---------------------------------------------------------------------------
# Threshold I/O
# ---------------------------------------------------------------------------
def save_thresholds(thresholds: dict, path: Path, alpha: float) -> None:
    """
    Save thresholds to CSV.
    columns: method, alpha, threshold
    """
    rows = [{"method": m, "alpha": alpha, "threshold": v} for m, v in thresholds.items()]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[verifier] Thresholds saved -> {path}")


def load_thresholds(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Calibration score split
# ---------------------------------------------------------------------------
def split_calibration(
    all_clean_scores: np.ndarray,
    cal_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split clean scores into calibration and test sets.
    Returns (cal_scores, test_scores).
    """
    rng = np.random.default_rng(seed)
    n = len(all_clean_scores)
    idx = rng.permutation(n)
    n_cal = int(n * cal_ratio)
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]
    return all_clean_scores[cal_idx], all_clean_scores[test_idx]
