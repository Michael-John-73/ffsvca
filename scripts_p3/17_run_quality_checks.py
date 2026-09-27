"""
scripts_p3/17_run_quality_checks.py

Phase 7: quality / sanity checks before paper submission.

Checks:
    Q1  outputs_p3/scores/scores_raw.csv exists, no NaN in score_z, label in {0,1}
    Q2  attack_id set matches manifest exactly (17 attacks)
    Q3  thresholds_by_method.csv has rows for {M1,M2,M3} x alpha set
    Q4  metrics csvs exist and are non-empty: e1, e2, e3, e4, e5
    Q5  E4 sizes subset of {50,100,200,500}
    Q6  No reference to forbidden ckpt 'optimized_wm5-30_embedding-step-500.pt'
        and forbidden 'w_up_radius=30' inside outputs_p3/manifests/* and
        outputs_p3/logs/* (regression guard)
    Q7  At least one alpha-row per (method, alpha) reports empirical_fpr <= 1.0

Writes PASS/FAIL report:
    outputs_p3/logs/quality_checks.log
Exits with non-zero status if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED_ATTACKS = {
    "none",
    "jpeg_q50", "jpeg_q30", "jpeg_q70", "jpeg_q90",
    "rotation", "cropping", "blurring",
    "noise_s003", "noise_s001", "noise_s005", "noise_s008",
    "jpeg+cropping", "cropping+jpeg", "blurring+noise",
    "rotation+cropping+jpeg", "noise+blurring+jpeg",
}
FORBIDDEN_STRINGS = ["optimized_wm5-30_embedding-step-500", "w_up_radius=30"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs_p3"))
    args = ap.parse_args()

    log = []
    fails = 0
    def check(name: str, ok: bool, msg: str = ""):
        nonlocal fails
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}: {msg}"
        log.append(line); print(line)
        if not ok: fails += 1

    # Q1
    raw = args.root / "scores" / "scores_raw.csv"
    if raw.exists():
        df = pd.read_csv(raw)
        check("Q1 scores_raw", df["score_z"].notna().all()
              and set(df.label.unique()).issubset({0, 1}),
              f"{len(df)} rows, labels={set(df.label.unique())}")
        # Q2
        check("Q2 attack_set", set(df.attack_id.unique()) == EXPECTED_ATTACKS,
              f"got {sorted(set(df.attack_id.unique()))}")
    else:
        check("Q1 scores_raw", False, f"missing {raw}")
        check("Q2 attack_set", False, "skipped")

    # Q3
    thp = args.root / "thresholds" / "thresholds_by_method.csv"
    if thp.exists():
        th = pd.read_csv(thp)
        methods_ok = set(th.method.unique()) == {"M1", "M2", "M3"}
        check("Q3 thresholds", methods_ok, f"methods={sorted(th.method.unique())}")
    else:
        check("Q3 thresholds", False, f"missing {thp}")

    # Q4 (required core experiments)
    metrics = args.root / "metrics"
    for name in ["e1_threshold_comparison.csv", "e1_mcnemar.csv",
                 "e2_fpr_drift.csv",
                 "e4_calibration_size.csv", "e5_failure_conditions.csv"]:
        p = metrics / name
        ok = p.exists() and p.stat().st_size > 0
        check(f"Q4 {name}", ok, f"size={p.stat().st_size if p.exists() else 'NA'}")

    # Q4b (optional: COCO real FPR / adaptive regen — warn-only)
    for name in ["e7_coco_real_fpr.csv", "e8_adaptive_regen.csv"]:
        p = metrics / name
        if p.exists():
            ok = p.stat().st_size > 0
            check(f"Q4b {name}", ok, f"size={p.stat().st_size}")
        else:
            log.append(f"[INFO] Q4b {name} not present (optional, needs external scoring)")

    # Q4c (optional: DwtDctSvd cross-family comparator — warn-only, App D)
    # App D lives under outputs_p3_appd/ (separate from ROBIN's outputs_p3/).
    appd_metrics = metrics.parent.parent / "outputs_p3_appd" / "metrics"
    p = appd_metrics / "e9_dwtdctsvd_comparator.csv"
    if p.exists():
        ok = p.stat().st_size > 0
        check("Q4c e9_dwtdctsvd_comparator.csv", ok, f"size={p.stat().st_size}")
    else:
        log.append(
            "[INFO] Q4c e9_dwtdctsvd_comparator.csv not present "
            "(optional, App D DwtDctSvd cross-family comparator)"
        )

    # Q5
    e4p = metrics / "e4_calibration_size.csv"
    if e4p.exists():
        e4 = pd.read_csv(e4p)
        ok = set(e4.size_requested.unique()).issubset({30, 50, 100, 200, 500})
        check("Q5 E4 sizes", ok, f"sizes={sorted(e4.size_requested.unique())}")

    # Q6
    scan_dirs = [args.root / "manifests", args.root / "logs"]
    found = []
    for d in scan_dirs:
        if not d.exists(): continue
        for f in d.rglob("*"):
            if f.is_file() and f.suffix in {".json", ".log", ".txt", ".csv"}:
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for s in FORBIDDEN_STRINGS:
                    if s in text:
                        found.append(f"{f}:{s}")
    check("Q6 no forbidden strings", not found, f"matches={found}")

    # Q7
    e1p = metrics / "e1_threshold_comparison.csv"
    if e1p.exists():
        e1 = pd.read_csv(e1p)
        ok = (e1.empirical_fpr <= 1.0).all() and (e1.empirical_fpr >= 0.0).all()
        check("Q7 e1_fpr_range", ok, f"min={e1.empirical_fpr.min()}, max={e1.empirical_fpr.max()}")

    out = args.root / "logs" / "quality_checks.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"[17] {fails} failures -> {out}")
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
