"""
ffsvca_bridge.py
ROBIN score dump wrapper — calls ROBIN inversion pipeline and exports per-image
scores to CSV for each attack condition.

Corresponds to MWDRAS/mwdras_bridge.py.
Usage:
    python ffsvca_bridge.py --config robin_config.json --phase mini
    python ffsvca_bridge.py --config robin_config.json --phase full
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy ROBIN import (requires robin_official to be on sys.path)
# ---------------------------------------------------------------------------
def _load_robin_root(config: dict) -> Path:
    robin_root = Path(config["robin_source_mapping"]["robin_root"])
    if not robin_root.exists():
        raise FileNotFoundError(f"robin_root not found: {robin_root}")
    if str(robin_root) not in sys.path:
        sys.path.insert(0, str(robin_root))
    return robin_root


# ---------------------------------------------------------------------------
# Score dump helpers
# ---------------------------------------------------------------------------
def compute_score(image_path: Path, config: dict) -> float:
    """
    Compute z(x) = -d(x) for a single image using ROBIN inversion.
    d(x): watermark inversion distance from ROBIN pipeline.
    Returns z score (float).
    """
    from inject_wm_inner_latent_robin import get_score  # noqa: E402
    robin_cfg = config["robin_source_mapping"]
    score = get_score(
        image_path=str(image_path),
        wm_path=robin_cfg["wm_path"],
        model_id=robin_cfg["model_id"],
        guidance_scale=robin_cfg["guidance_scale"],
        num_inference_steps=robin_cfg["num_inference_steps"],
        w_channel=robin_cfg["w_channel"],
        w_pattern=robin_cfg["w_pattern"],
        w_up_radius=robin_cfg["w_up_radius"],
        w_low_radius=robin_cfg["w_low_radius"],
    )
    return float(-score)  # z(x) = -d(x)


def dump_scores(
    image_dir: Path,
    output_csv: Path,
    config: dict,
    label: int,
    attack: str,
) -> None:
    """
    Iterate over all .png images in image_dir, compute z(x), write CSV.
    CSV columns: image_id, image_path, attack, label, z_score
    label: 1 = watermarked, 0 = clean
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No .png images found in {image_dir}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "image_path", "attack", "label", "z_score"]
        )
        writer.writeheader()
        for img in images:
            z = compute_score(img, config)
            writer.writerow(
                {
                    "image_id": img.stem,
                    "image_path": str(img),
                    "attack": attack,
                    "label": label,
                    "z_score": z,
                }
            )
    print(f"[bridge] Wrote {len(images)} scores -> {output_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FFSVCA bridge: ROBIN score dump")
    parser.add_argument(
        "--config", default="robin_config.json", help="Path to robin_config.json"
    )
    parser.add_argument(
        "--phase",
        choices=["mini", "full"],
        default="mini",
        help="mini = 5 prompts / full = 1000 prompts",
    )
    parser.add_argument(
        "--attack",
        default="none",
        help="Attack name (must match attack_manifest key in config)",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        default=1,
        help="0=clean 1=watermarked",
    )
    parser.add_argument("--image_dir", required=True, help="Input image directory")
    parser.add_argument(
        "--output_csv", required=True, help="Output score CSV path"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    _load_robin_root(config)

    dump_scores(
        image_dir=Path(args.image_dir),
        output_csv=Path(args.output_csv),
        config=config,
        label=args.label,
        attack=args.attack,
    )


if __name__ == "__main__":
    main()
