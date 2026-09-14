"""
ffsvca_result_metrics.py
FPR / TPR / Calibration Error / Confidence Interval / Failure condition metrics.

Key metrics:
    - empirical_fpr(y_true, y_pred, label=0): proportion of clean images detected as watermarked
    - empirical_tpr(y_true, y_pred, label=1): detection rate of watermarked images
    - calibration_error(empirical_fpr, alpha): CE = |FPR - alpha|  (GAP-3)
    - fpr_violation(empirical_fpr, alpha): 1[FPR > alpha]  (failure condition)
    - wilson_ci(k, n, alpha): Wilson score CI for proportion
    - worst_case_fpr/tpr across attack groups  (GAP-2)
    - auc over threshold sweep  (GAP-6, secondary)

Corresponds to MWDRAS/mwdras_result_metrics.py.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------
def empirical_fpr(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Empirical FPR: fraction of clean images (label=0) predicted as watermarked (pred=1).
    """
    clean_mask = labels == 0
    if clean_mask.sum() == 0:
        return float("nan")
    return float(preds[clean_mask].mean())


def empirical_tpr(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Empirical TPR: fraction of watermarked images (label=1) predicted as watermarked.
    """
    wm_mask = labels == 1
    if wm_mask.sum() == 0:
        return float("nan")
    return float(preds[wm_mask].mean())


def calibration_error(emp_fpr: float, alpha: float) -> float:
    """
    Calibration error: CE_t = |FPR_t - alpha|
    Measures deviation of empirical FPR from nominal level.
    """
    return float(abs(emp_fpr - alpha))


def fpr_violation(emp_fpr: float, alpha: float) -> int:
    """
    Violation indicator: 1 if FPR > alpha (security guarantee broken).
    """
    return int(emp_fpr > alpha)


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion k/n.
    Returns (lower, upper).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


# ---------------------------------------------------------------------------
# AUC (secondary metric, GAP-6)
# ---------------------------------------------------------------------------
def roc_auc(z_scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute ROC-AUC using ROBIN z-scores as continuous scores.
    Requires both classes (0 and 1) in labels.
    """
    from sklearn.metrics import roc_auc_score  # noqa: E402
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, z_scores))


# ---------------------------------------------------------------------------
# Per-attack evaluation
# ---------------------------------------------------------------------------
def evaluate_attack(
    scores_df: pd.DataFrame,
    tau: float,
    alpha: float,
    attack: str,
    method: str,
) -> dict:
    """
    Compute full metric set for one (attack, method, alpha) combination.
    Expects scores_df columns: label, z_score
    Returns dict with all metrics for one row of results table.
    """
    z = scores_df["z_score"].values
    labels = scores_df["label"].values
    preds = (z > tau).astype(int)

    fpr = empirical_fpr(labels, preds)
    tpr = empirical_tpr(labels, preds)
    ce = calibration_error(fpr, alpha)
    violated = fpr_violation(fpr, alpha)
    auc = roc_auc(z, labels)

    n_clean = int((labels == 0).sum())
    k_clean_tp = int(preds[labels == 0].sum())
    ci_lo, ci_hi = wilson_ci(k_clean_tp, n_clean)

    return {
        "attack": attack,
        "method": method,
        "alpha": alpha,
        "tau": tau,
        "empirical_fpr": fpr,
        "empirical_tpr": tpr,
        "calibration_error": ce,
        "fpr_violated": violated,
        "fpr_ci_lo": ci_lo,
        "fpr_ci_hi": ci_hi,
        "n_clean": n_clean,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Worst-case aggregation (GAP-2)
# ---------------------------------------------------------------------------
def worst_case_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (method, alpha): compute worst_case_fpr and worst_case_tpr across attacks.
    Returns summary DataFrame.
    """
    rows = []
    for (method, alpha), grp in results_df.groupby(["method", "alpha"]):
        rows.append(
            {
                "method": method,
                "alpha": alpha,
                "worst_case_fpr": grp["empirical_fpr"].max(),
                "worst_case_tpr": grp["empirical_tpr"].min(),
                "n_violations": int(grp["fpr_violated"].sum()),
                "n_attacks": len(grp),
                "mean_ce": grp["calibration_error"].mean(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_metrics(results_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f"[metrics] Saved -> {path}")


def load_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
