"""
scripts_p3/02_build_attack_manifest.py

We materialize the canonical 17-attack manifest (1 clean + 11 single-stage
including the severity sweep + 5 composite) with FIXED severities. We need
the severity sweep (jpeg quality, noise sigma) for RQ3/E6 (Fig. 4), so we
MUST keep it as first-class attack ids, not merged into a single
"jpeg"/"noise" entry.

We use this script for record keeping only. We apply the actual attacks
inside robin_official/inject_wm_inner_latent_robin.py via --attack_list with
the same identifiers we use here (joined by `+` for composites).

We write: outputs_p3/manifests/attack_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTACK_MANIFEST = {
    "alpha_policy": {"primary": 0.05, "secondary": [0.10, 0.01]},
    "severity": {
        "jpeg": {"primary_quality": 50, "sweep_quality": [30, 50, 70, 90]},
        "rotation": {"degrees": 10},
        "cropping": {"ratio": 0.80},
        "blurring": {"kernel": 5, "sigma": 1.0},
        "noise": {"primary_sigma": 0.03, "sweep_sigma": [0.01, 0.03, 0.05, 0.08]},
    },
    "attacks": [
        # ID-clean
        {"id": "none", "family": "ID-single", "stages": []},
        # ID-single: JPEG severity sweep (primary = q50)
        {"id": "jpeg_q50", "family": "ID-single", "stages": [{"op": "jpeg", "quality": 50}]},
        {"id": "jpeg_q30", "family": "ID-single", "stages": [{"op": "jpeg", "quality": 30}]},
        {"id": "jpeg_q70", "family": "ID-single", "stages": [{"op": "jpeg", "quality": 70}]},
        {"id": "jpeg_q90", "family": "ID-single", "stages": [{"op": "jpeg", "quality": 90}]},
        # ID-single: fixed-severity single attacks
        {"id": "rotation", "family": "ID-single", "stages": [{"op": "rotation", "degrees": 10}]},
        {"id": "cropping", "family": "ID-single", "stages": [{"op": "cropping", "ratio": 0.80}]},
        {"id": "blurring", "family": "ID-single", "stages": [{"op": "blurring", "kernel": 5, "sigma": 1.0}]},
        # ID-single: additive-noise severity sweep (primary = sigma 0.03)
        {"id": "noise_s003", "family": "ID-single", "stages": [{"op": "noise", "sigma": 0.03}]},
        {"id": "noise_s001", "family": "ID-single", "stages": [{"op": "noise", "sigma": 0.01}]},
        {"id": "noise_s005", "family": "ID-single", "stages": [{"op": "noise", "sigma": 0.05}]},
        {"id": "noise_s008", "family": "ID-single", "stages": [{"op": "noise", "sigma": 0.08}]},
        # same-family composite: within-family compositions (primary severities)
        {"id": "jpeg+cropping", "family": "same-family composite", "stages": [
            {"op": "jpeg", "quality": 50}, {"op": "cropping", "ratio": 0.80}]},
        {"id": "cropping+jpeg", "family": "same-family composite", "stages": [
            {"op": "cropping", "ratio": 0.80}, {"op": "jpeg", "quality": 50}]},
        {"id": "blurring+noise", "family": "same-family composite", "stages": [
            {"op": "blurring", "kernel": 5, "sigma": 1.0}, {"op": "noise", "sigma": 0.03}]},
        # cross-family composite: cross-family compositions (primary severities)
        {"id": "rotation+cropping+jpeg", "family": "cross-family composite", "stages": [
            {"op": "rotation", "degrees": 10}, {"op": "cropping", "ratio": 0.80}, {"op": "jpeg", "quality": 50}]},
        {"id": "noise+blurring+jpeg", "family": "cross-family composite", "stages": [
            {"op": "noise", "sigma": 0.03}, {"op": "blurring", "kernel": 5, "sigma": 1.0}, {"op": "jpeg", "quality": 50}]},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs_p3/manifests/attack_manifest.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ATTACK_MANIFEST, indent=2), encoding="utf-8")
    n_total = len(ATTACK_MANIFEST["attacks"])
    assert n_total == 17, f"expected 17 attack_ids in the manifest, got {n_total}"
    print(f"[02] Wrote {n_total} attacks -> {args.out}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
