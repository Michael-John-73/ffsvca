"""
scripts_p3/15_make_paper_tables.py

We emit the canonical paper tables as CSV (machine-readable) for inclusion
in the manuscript. We also absorb our previous 13_build_operating_range_summary
and 14_build_final_summary scripts here.

Tables we produce (default alpha = 0.05):
    Table 1: threshold method comparison (M1/M2/M3) at alpha=0.05
    Table 2: FPR drift summary per family + worst-case (M3 only)
    Table 3: failure-condition summary (V_t, ViolationRate, OOD drift)
    Table 4: calibration-size sensitivity (E4)
    Table 5: COCO real FPR (E7)
    Table 6: adaptive regeneration TPR/FPR vs strength (E8)
    operating_range_summary.csv (per method/alpha, safe vs unsafe families)
    final_summary.csv (merged E1/E2/E5 + CI per (method,alpha,attack_id))

Our output dirs:
    outputs_p3/tables/ (paper tables)
    outputs_p3/metrics/ (summary CSVs)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_operating_range_summary(e2: pd.DataFrame, out: Path) -> None:
    df = e2[e2.attack_id != "_worst"].copy()
    rows = []
    for (alpha, method), sub in df.groupby(["alpha", "method"]):
        safe = sub[sub.empirical_fpr <= alpha]
        unsafe = sub[sub.empirical_fpr > alpha]
        safe_fams = set(safe.family.unique()) - set(unsafe.family.unique())
        unsafe_fams = set(unsafe.family.unique())
        rows.append({
            "alpha": alpha, "method": method,
            "n_attacks": int(len(sub)),
            "n_safe": int(len(safe)),
            "n_unsafe": int(len(unsafe)),
            "safe_families": ";".join(sorted(safe_fams)),
            "unsafe_families": ";".join(sorted(unsafe_fams)),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def write_final_summary(metrics_dir: Path, out: Path) -> None:
    e1 = pd.read_csv(metrics_dir / "e1_threshold_comparison.csv")
    e2 = pd.read_csv(metrics_dir / "e2_fpr_drift.csv")
    e2 = e2[e2.attack_id != "_worst"]
    e5 = pd.read_csv(metrics_dir / "e5_failure_conditions.csv")
    ci_p = metrics_dir / "ci_bootstrap.csv"
    ci = pd.read_csv(ci_p) if ci_p.exists() else pd.DataFrame()

    key = ["method", "alpha", "attack_id"]
    e2_keep = e2[key + ["family", "empirical_fpr", "empirical_tpr",
                        "calibration_error", "delta_fpr", "fpr_violation"]]
    merged = e2_keep.copy()
    if not ci.empty and all(k in ci.columns for k in key):
        merged = merged.merge(ci, on=key, how="left", suffixes=("", "_ci"))
    e5_keep = (e5[["method", "alpha", "violation_rate"]]
               if "violation_rate" in e5.columns
               else e5[["method", "alpha"]].assign(violation_rate=float("nan")))
    merged = merged.merge(e5_keep.drop_duplicates(["method", "alpha"]),
                          on=["method", "alpha"], how="left")
    e1_keep = e1[["method", "alpha", "empirical_fpr", "empirical_tpr",
                  "calibration_error"]].rename(columns={
        "empirical_fpr": "e1_fpr_none",
        "empirical_tpr": "e1_tpr_none",
        "calibration_error": "e1_ce_none",
    })
    merged = merged.merge(e1_keep, on=["method", "alpha"], how="left")

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=Path, default=Path("outputs_p3/metrics"))
    ap.add_argument("--out_dir", type=Path, default=Path("outputs_p3/tables"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1 ---
    e1 = pd.read_csv(args.metrics_dir / "e1_threshold_comparison.csv")
    t1 = e1[e1.alpha == args.alpha].pivot_table(
        index="attack_id", columns="method",
        values=["empirical_fpr", "empirical_tpr", "calibration_error"],
    ).round(4)
    t1.to_csv(args.out_dir / f"table1_threshold_comparison_alpha{args.alpha}.csv")

    # --- Table 2 ---
    e2 = pd.read_csv(args.metrics_dir / "e2_fpr_drift.csv")
    t2 = e2[(e2.alpha == args.alpha) & (e2.method == "M3")]
    t2.to_csv(args.out_dir / f"table2_fpr_drift_m3_alpha{args.alpha}.csv", index=False)

    # --- Table 3 ---
    e5 = pd.read_csv(args.metrics_dir / "e5_failure_conditions.csv")
    t3 = e5[e5.alpha == args.alpha]
    t3.to_csv(args.out_dir / f"table3_failure_conditions_alpha{args.alpha}.csv",
              index=False)

    # --- Table 4 ---
    e4_path = args.metrics_dir / "e4_calibration_size.csv"
    if e4_path.exists():
        e4 = pd.read_csv(e4_path)
        t4 = e4[e4.alpha == args.alpha]
        t4.to_csv(args.out_dir / f"table4_calibration_size_alpha{args.alpha}.csv",
                  index=False)

    # --- Table 5: COCO real FPR (E7) ---
    e7_path = args.metrics_dir / "e7_coco_real_fpr.csv"
    if e7_path.exists():
        e7 = pd.read_csv(e7_path)
        t5 = e7[e7.alpha == args.alpha]
        t5.to_csv(args.out_dir / f"table5_coco_real_fpr_alpha{args.alpha}.csv",
                  index=False)

    # --- Table 6: adaptive regen (E8) ---
    e8_path = args.metrics_dir / "e8_adaptive_regen.csv"
    if e8_path.exists():
        e8 = pd.read_csv(e8_path)
        t6 = e8[e8.alpha == args.alpha]
        t6.to_csv(args.out_dir / f"table6_adaptive_regen_alpha{args.alpha}.csv",
                  index=False)

    # --- Table 7: DwtDctSvd cross-family comparator (E9, App D) ---
    # For App D we use a separate output root (outputs_p3_appd/) so we do not
    # pollute our primary ROBIN outputs (outputs_p3/).
    appd_root = args.metrics_dir.parent.parent / "outputs_p3_appd"
    e9_path = appd_root / "metrics" / "e9_dwtdctsvd_comparator.csv"
    if e9_path.exists():
        e9 = pd.read_csv(e9_path)
        t7 = e9[e9.alpha == args.alpha]
        t7_out_dir = appd_root / "tables"
        t7_out_dir.mkdir(parents=True, exist_ok=True)
        t7.to_csv(t7_out_dir / f"table7_dwtdctsvd_comparator_alpha{args.alpha}.csv",
                  index=False)

    # --- Absorbed: operating-range + final summary ---
    write_operating_range_summary(
        e2, args.metrics_dir / "operating_range_summary.csv")
    write_final_summary(args.metrics_dir,
                        args.metrics_dir / "final_summary.csv")

    print(f"[15] paper tables + summaries -> {args.out_dir} + {args.metrics_dir}")


if __name__ == "__main__":
    main()
