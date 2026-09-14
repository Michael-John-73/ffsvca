"""
ffsvca_runner.py
E1~E4 analysis pipeline runner.
Reads scores CSVs from outputs_p3/scores/, applies M1/M2/M3 thresholds,
computes metrics, and writes results to outputs_p3/metrics/.

Experiments:
    E1 — Threshold comparison (M1 vs M2 vs M3) on none/ID-single attacks
    E2 — FPR drift analysis across all 11 attack tasks
    E3 — Score distribution shift (clean vs attacked z-score distributions)
    E4 — Calibration sample size sensitivity (n=50,100,200,500)

Corresponds to MWDRAS/mwdras_meta_runner.py.

Usage:
    python ffsvca_runner.py --config robin_config.json --experiment E1
    python ffsvca_runner.py --config robin_config.json --experiment all
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ffsvca_verifier import compute_all_thresholds, apply_decision, split_calibration, load_thresholds, save_thresholds
from ffsvca_result_metrics import evaluate_attack, worst_case_summary, save_metrics


OUTPUTS_ROOT = Path("../PAPER3/outputs_p3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_scores(scores_dir: Path, attack: str) -> pd.DataFrame:
    path = scores_dir / f"{attack}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Score CSV not found: {path}")
    return pd.read_csv(path)


def get_alphas(config: dict) -> list[float]:
    vc = config["verification_config"]
    primary = vc["alpha_fpr_primary"]
    secondary = vc["alpha_fpr_secondary"]
    return [primary] + secondary


# ---------------------------------------------------------------------------
# E1: Threshold comparison
# ---------------------------------------------------------------------------
def run_e1(config: dict, outputs: Path) -> None:
    """
    E1: Compare M1/M2/M3 thresholds and their FPR/TPR on ID-single attacks.
    Research question: Does M3 expose hidden FPR risk vs M1/M2?
    """
    scores_dir = outputs / "scores"
    metrics_dir = outputs / "metrics"
    thresholds_dir = outputs / "thresholds"

    vc = config["verification_config"]
    cal_ratio = vc["cal_ratio"]
    split_seed = vc["split_seed"]
    n_bootstrap = vc["n_bootstrap"]
    alphas = get_alphas(config)

    attacks = config["attack_manifest"]

    # Load clean calibration scores (attack=none, label=0)
    none_df = load_scores(scores_dir, "none")
    clean_scores = none_df[none_df["label"] == 0]["z_score"].values
    cal_scores, test_clean_scores = split_calibration(clean_scores, cal_ratio, split_seed)

    rows = []
    for alpha in alphas:
        thresholds = compute_all_thresholds(cal_scores, alpha, n_bootstrap, split_seed)
        save_thresholds(
            thresholds,
            thresholds_dir / f"e1_alpha{alpha:.2f}_thresholds.csv",
            alpha,
        )

        for attack_id in attacks:
            attack_df = load_scores(scores_dir, attack_id)
            # Merge test clean scores with attacked watermarked scores
            wm_scores = attack_df[attack_df["label"] == 1]["z_score"].values
            all_z = np.concatenate([test_clean_scores, wm_scores])
            all_labels = np.concatenate(
                [np.zeros(len(test_clean_scores)), np.ones(len(wm_scores))]
            )
            eval_df = pd.DataFrame({"z_score": all_z, "label": all_labels})

            for method, tau in thresholds.items():
                row = evaluate_attack(eval_df, tau, alpha, attack_id, method)
                rows.append(row)

    result_df = pd.DataFrame(rows)
    save_metrics(result_df, metrics_dir / "e1_threshold_comparison.csv")

    worst = worst_case_summary(result_df)
    save_metrics(worst, metrics_dir / "e1_worst_case.csv")
    print("[E1] Done.")


# ---------------------------------------------------------------------------
# E2: FPR drift across all attacks
# ---------------------------------------------------------------------------
def run_e2(config: dict, outputs: Path) -> None:
    """
    E2: Empirical FPR for all 11 attack tasks, per method, per alpha.
    Research question: Where does fixed-FPR guarantee fail?
    """
    scores_dir = outputs / "scores"
    metrics_dir = outputs / "metrics"
    thresholds_dir = outputs / "thresholds"

    vc = config["verification_config"]
    cal_ratio = vc["cal_ratio"]
    split_seed = vc["split_seed"]
    n_bootstrap = vc["n_bootstrap"]
    alphas = get_alphas(config)
    attacks = config["attack_manifest"]

    none_df = load_scores(scores_dir, "none")
    clean_scores = none_df[none_df["label"] == 0]["z_score"].values
    cal_scores, test_clean_scores = split_calibration(clean_scores, cal_ratio, split_seed)

    rows = []
    for alpha in alphas:
        thresholds = compute_all_thresholds(cal_scores, alpha, n_bootstrap, split_seed)
        for attack_id in attacks:
            attack_df = load_scores(scores_dir, attack_id)
            # Clean test scores under attack (FPR evaluation)
            attacked_clean = attack_df[attack_df["label"] == 0]["z_score"].values
            if len(attacked_clean) == 0:
                # Fallback: use test split from none
                attacked_clean = test_clean_scores
            wm_scores = attack_df[attack_df["label"] == 1]["z_score"].values
            all_z = np.concatenate([attacked_clean, wm_scores])
            all_labels = np.concatenate(
                [np.zeros(len(attacked_clean)), np.ones(len(wm_scores))]
            )
            eval_df = pd.DataFrame({"z_score": all_z, "label": all_labels})

            for method, tau in thresholds.items():
                row = evaluate_attack(eval_df, tau, alpha, attack_id, method)
                rows.append(row)

    result_df = pd.DataFrame(rows)
    save_metrics(result_df, metrics_dir / "e2_fpr_drift.csv")

    worst = worst_case_summary(result_df)
    save_metrics(worst, metrics_dir / "e2_worst_case.csv")
    print("[E2] Done.")


# ---------------------------------------------------------------------------
# E3: Score distribution shift
# ---------------------------------------------------------------------------
def run_e3(config: dict, outputs: Path) -> None:
    """
    E3: z-score distribution statistics per attack.
    Outputs mean/std/quantiles for visualization in Figure 2.
    """
    scores_dir = outputs / "scores"
    metrics_dir = outputs / "metrics"
    attacks = config["attack_manifest"]

    rows = []
    for attack_id in attacks:
        try:
            df = load_scores(scores_dir, attack_id)
        except FileNotFoundError:
            print(f"[E3] Warning: {attack_id}.csv not found, skipping.")
            continue
        for label, grp in df.groupby("label"):
            z = grp["z_score"].values
            rows.append(
                {
                    "attack": attack_id,
                    "label": int(label),
                    "mean": float(np.mean(z)),
                    "std": float(np.std(z)),
                    "q05": float(np.quantile(z, 0.05)),
                    "q25": float(np.quantile(z, 0.25)),
                    "q50": float(np.quantile(z, 0.50)),
                    "q75": float(np.quantile(z, 0.75)),
                    "q95": float(np.quantile(z, 0.95)),
                    "n": len(z),
                }
            )

    result_df = pd.DataFrame(rows)
    save_metrics(result_df, metrics_dir / "e3_score_dist.csv")
    print("[E3] Done.")


# ---------------------------------------------------------------------------
# E4: Calibration sample size sensitivity
# ---------------------------------------------------------------------------
def run_e4(config: dict, outputs: Path) -> None:
    """
    E4: Vary n_cal in {50, 100, 200, 500}, observe threshold stability.
    Research question: How many calibration samples are sufficient?
    """
    scores_dir = outputs / "scores"
    metrics_dir = outputs / "metrics"

    vc = config["verification_config"]
    cal_sizes = vc["e4_calibration_sizes"]
    split_seed = vc["split_seed"]
    alphas = get_alphas(config)

    none_df = load_scores(scores_dir, "none")
    clean_scores = none_df[none_df["label"] == 0]["z_score"].values

    rows = []
    rng = np.random.default_rng(split_seed)
    for n_cal in cal_sizes:
        if n_cal > len(clean_scores):
            print(f"[E4] Warning: n_cal={n_cal} > available {len(clean_scores)}, skipping.")
            continue
        cal_scores = rng.choice(clean_scores, size=n_cal, replace=False)
        for alpha in alphas:
            thresholds = compute_all_thresholds(cal_scores, alpha, n_bootstrap=1000, seed=split_seed)
            for method, tau in thresholds.items():
                rows.append(
                    {
                        "n_cal": n_cal,
                        "method": method,
                        "alpha": alpha,
                        "tau": tau,
                    }
                )

    result_df = pd.DataFrame(rows)
    save_metrics(result_df, metrics_dir / "e4_cal_sensitivity.csv")
    print("[E4] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
EXPERIMENTS = {"E1": run_e1, "E2": run_e2, "E3": run_e3, "E4": run_e4}


def main():
    parser = argparse.ArgumentParser(description="FFSVCA experiment runner")
    parser.add_argument("--config", default="robin_config.json")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--outputs",
        default=str(OUTPUTS_ROOT),
        help="Path to outputs_p3/ directory",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    outputs = Path(args.outputs)

    if args.experiment == "all":
        for name, fn in EXPERIMENTS.items():
            print(f"\n{'='*40}")
            print(f"Running {name}...")
            fn(config, outputs)
    else:
        EXPERIMENTS[args.experiment](config, outputs)


if __name__ == "__main__":
    main()
