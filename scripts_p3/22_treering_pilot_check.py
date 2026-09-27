"""
scripts_p3/22_treering_pilot_check.py  (local)

Checks a Tree-Ring scoring CSV downloaded from the pod:
  1. robin_distance (label 0) vs the main run's scores_raw.csv -> same inversion?
  2. Tree-Ring separation on the unmodified condition (clean vs Tree-Ring image).
Writes sd21_results/outputs_p3/treering/pilot_check.txt (UTF-8).

Usage:
    F:\\RCE\\venv_p3\\Scripts\\python.exe scripts_p3/22_treering_pilot_check.py \\
        --csv sd21_results/outputs_p3/treering/scores/seed_0/treering_scores.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=ROOT / "sd21_results/outputs_p3/scores/scores_raw.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "sd21_results/outputs_p3/treering/pilot_check.txt")
    a = ap.parse_args()

    tr = pd.read_csv(a.csv, dtype={"prompt_id": str})
    raw = pd.read_csv(a.raw, dtype={"prompt_id": str})
    lines = [f"rows={len(tr)} prompts={tr.prompt_id.nunique()} seeds={sorted(tr.seed.unique())}",
             f"rows per (attack,label):\n{tr.groupby(['attack_id', 'label']).size().to_string()}"]

    m = tr[tr.label == 0].merge(raw[raw.label == 0], on=["prompt_id", "seed", "attack_id"], how="left")
    miss = int(m.distance.isna().sum())
    diff = (m.robin_distance - m.distance).abs()
    rel = diff / m.distance.abs()
    lines.append(f"\n[1] ROBIN reproduction on label-0 rows: matched={len(m) - miss} missing={miss}")
    lines.append(f"    |diff| max={diff.max():.6g} mean={diff.mean():.6g}; rel max={rel.max():.3g}")
    lines.append("    per attack max |diff|:\n" + m.assign(d=diff).groupby("attack_id").d.max().to_string())

    c = tr[(tr.label == 0) & (tr.attack_id == "none")].set_index("prompt_id").tr_distance
    w = tr[(tr.label == 1) & (tr.attack_id == "none")].set_index("prompt_id").tr_distance
    lines.append(f"\n[2] Tree-Ring distance, none: clean n={len(c)} mean={c.mean():.3f} "
                 f"[min {c.min():.3f}, max {c.max():.3f}]")
    lines.append(f"    Tree-Ring image n={len(w)} mean={w.mean():.3f} [min {w.min():.3f}, max {w.max():.3f}]")
    # AUC with score = -distance (Mann-Whitney, ties counted 1/2)
    sc, sw = -c.to_numpy()[:, None], -w.to_numpy()[None, :]
    auc = float(np.mean((sw > sc) + 0.5 * (sw == sc)))
    lines.append(f"    AUC(none) = {auc:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
