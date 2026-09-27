"""
scripts_p3/08e_eval_dwtdctsvd_comparator.py

E9 (Appendix D): Cross-family comparator on DwtDctSvd (invisible-watermark
library, ShieldMnt/invisible-watermark, MIT). The watermark used by the
public Stable Diffusion v1.5 / v2.1 release.

Purpose: demonstrate that the fixed-FPR verification protocol (M3 split
conformal threshold) generalizes to a detector from a *different family*
than ROBIN (which is FFT-ring + DDIM-inversion). DwtDctSvd is a classical
spatial-domain blind watermark with bit-message decoding and no inversion
step, so portability evidence here is genuinely cross-family rather than
an ablation of the same FFT-ring code path.

Score convention for this comparator:
    score_z = bit_accuracy ∈ [0, 1]
        = (# bits matching the embedded key) / key_length
Higher score_z => more confident watermark presence (consistent with the
"higher = positive" convention used elsewhere in this codebase).

Expected input schema (outputs_p3_appd/scores/dwtdctsvd_scores.csv):
    image_id, attack_id, source_label, split, score_z
where:
    attack_id     in {none, jpeg_q50, cropping}
    source_label  in {clean, watermarked}
    split         in {cal, test}    # cal rows used only when source_label=clean

Output: outputs_p3_appd/metrics/e9_dwtdctsvd_comparator.csv
    columns:
        detector, method, alpha, attack_id,
        n_clean, n_wm,
        empirical_fpr, fpr_ci_lo, fpr_ci_hi,
        empirical_tpr, tpr_ci_lo, tpr_ci_hi,
        calibration_error, delta_fpr, fpr_violation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    calibration_error,
    conformal_quantile,
    empirical_fpr,
    empirical_tpr,
    wilson_ci,
)

ATTACKS_DEFAULT = ["none", "jpeg_q50", "cropping"]
DETECTOR_NAME = "dwtdctsvd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores",
        type=Path,
        default=Path("outputs_p3_appd/scores/dwtdctsvd_scores.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs_p3_appd/metrics/e9_dwtdctsvd_comparator.csv"),
    )
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.05, 0.10, 0.01])
    ap.add_argument("--attacks", nargs="+", type=str, default=ATTACKS_DEFAULT)
    args = ap.parse_args()

    if not args.scores.exists():
        raise SystemExit(
            f"[08e] {args.scores} not found. Produce DwtDctSvd scores first "
            "with scripts_p3/score_dwtdctsvd.py (writes columns "
            "image_id,attack_id,source_label,split,score_z)."
        )

    df = pd.read_csv(args.scores)
    required_cols = {"image_id", "attack_id", "source_label", "split", "score_z"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[08e] {args.scores} missing columns: {sorted(missing)}")

    cal = df[
        (df.source_label == "clean")
        & (df.attack_id == "none")
        & (df.split == "cal")
    ]["score_z"].to_numpy()
    if len(cal) == 0:
        raise SystemExit(
            "[08e] no DwtDctSvd calibration rows (source_label=clean, "
            "attack_id=none, split=cal). Cannot compute M3 threshold."
        )

    rows = []
    for alpha in args.alpha:
        tau = conformal_quantile(cal, alpha)
        for atk in args.attacks:
            clean_test = df[
                (df.source_label == "clean")
                & (df.attack_id == atk)
                & (df.split == "test")
            ]["score_z"].to_numpy()
            wm_test = df[
                (df.source_label == "watermarked")
                & (df.attack_id == atk)
                & (df.split == "test")
            ]["score_z"].to_numpy()
            n_clean = int(len(clean_test))
            n_wm = int(len(wm_test))
            if n_clean == 0 and n_wm == 0:
                continue
            fpr = empirical_fpr(clean_test, tau)
            tpr = empirical_tpr(wm_test, tau)
            fpr_lo, fpr_hi = wilson_ci(int(np.sum(clean_test > tau)), n_clean)
            tpr_lo, tpr_hi = wilson_ci(int(np.sum(wm_test > tau)), n_wm)
            rows.append({
                "detector": DETECTOR_NAME,
                "method": "M3",
                "alpha": alpha,
                "attack_id": atk,
                "n_clean": n_clean,
                "n_wm": n_wm,
                "empirical_fpr": fpr,
                "fpr_ci_lo": fpr_lo,
                "fpr_ci_hi": fpr_hi,
                "empirical_tpr": tpr,
                "tpr_ci_lo": tpr_lo,
                "tpr_ci_hi": tpr_hi,
                "calibration_error": calibration_error(fpr, alpha),
                "delta_fpr": (fpr - alpha) if not np.isnan(fpr) else float("nan"),
                "fpr_violation": (
                    int(fpr > alpha) if not np.isnan(fpr) else 0
                ),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[08e] E9 DwtDctSvd comparator -> {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
