"""
scripts_p3/04_parse_score_dump.py

We parse the score JSON dumps that 03_score_attacks.py (OpenVINO) /
03_score_attacks_cuda.py (CUDA) produce into the canonical scores_raw.csv
that the rest of our pipeline uses.

Our score convention (as in the paper):
    z = -d   (z = score_z, d = ROBIN inversion distance)
We read higher z as more confident watermark presence.

We expect this input layout (per attack run, configurable via --scores_dir):
    <scores_dir>/<attack_id>/scores.json
        {
          "attack_id": "jpeg+cropping",
          "items": [
            {"prompt_id": "000000", "seed": 0, "label": 0, "distance": 1.234},
            {"prompt_id": "000000", "seed": 0, "label": 1, "distance": 0.456},
            ...
          ]
        }
    We use label = 0 for CLEAN and label = 1 for WATERMARKED.

We write:
    outputs_p3/scores/scores_raw.csv
    columns: prompt_id, seed, attack_id, family, label, distance, score_z

We read `family` from outputs_p3/manifests/attack_manifest.json (our single
source of truth, kept in sync with 02_build_attack_manifest.py) rather than
duplicating a lookup table here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_family_map(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {a["id"]: a["family"] for a in manifest["attacks"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", type=Path, required=True,
                    help="Directory holding <attack_id>/scores.json subfolders")
    ap.add_argument("--manifest", type=Path,
                    default=Path("outputs_p3/manifests/attack_manifest.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3/scores/scores_raw.csv"))
    args = ap.parse_args()

    family_map = load_family_map(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n_files = 0
    for jf in sorted(args.scores_dir.rglob("scores.json")):
        n_files += 1
        data = json.loads(jf.read_text(encoding="utf-8"))
        attack_id = data.get("attack_id") or jf.parent.name
        fam = family_map.get(attack_id, "unknown")
        for item in data.get("items", []):
            d = float(item["distance"])
            rows.append({
                "prompt_id": str(item["prompt_id"]),
                "seed":      int(item.get("seed", 0)),
                "attack_id": attack_id,
                "family":    fam,
                "label":     int(item["label"]),
                "distance":  d,
                "score_z":   -d,
            })

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "seed", "attack_id",
                                          "family", "label", "distance", "score_z"])
        w.writeheader()
        w.writerows(rows)
    print(f"[04] {n_files} json files, {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
