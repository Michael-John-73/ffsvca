"""
scripts_p3/01_build_prompt_list.py

We build the prompt list for ROBIN clean/watermarked image generation.

We source prompts from MS-COCO captions if available
(robin_official/coco/captions_val2017.json); otherwise we fall back to a
combinatorial synthetic prompt generator (see `build_combinatorial_bank`).
gen_clean_image.py / inject_wm_inner_latent_robin.py consume our output CSV
via --start/--end indexing.

Why we use a combinatorial bank (not a small fixed bank): our statistical machinery
(split-conformal calibration, Wilson/bootstrap CIs over n=1000) assumes a
reasonably diverse, roughly-representative sample. If we sampled with replacement
from a ~15-item bank, we would collapse the effective sample size to ~15 x n_seeds
distinct generations (each duplicated ~65x), which would silently invalidate
every downstream CI. Our combinatorial bank below has tens of thousands of
distinct combinations, so our `n=1000` draws without replacement are unique.

Usage:
    python scripts_p3/01_build_prompt_list.py --n 5    --mode debug
    python scripts_p3/01_build_prompt_list.py --n 1000 --mode main
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

SUBJECTS = [
    "a city skyline", "a mountain landscape", "a person in a red coat",
    "a dense forest", "a spaceship", "a vintage car", "a bowl of fruit",
    "an astronaut", "a clockwork bird", "a Japanese garden", "an empty street",
    "a floating castle", "a bustling marketplace", "a vase of flowers",
    "a wandering robot", "a lighthouse", "a coral reef", "a desert caravan",
    "a snow-covered village", "a steam locomotive", "a hot air balloon",
    "a violinist", "a herd of horses", "a glass greenhouse", "a stone bridge",
]
SETTINGS = [
    "at sunset", "under a starry sky", "in heavy rain", "at dawn",
    "in the fog", "during a snowstorm", "at golden hour", "under neon lights",
    "in an empty plaza", "beside a river", "on a cliff edge", "in a quiet alley",
    "surrounded by autumn leaves", "in a bustling square", "at midnight",
    "on the horizon", "in a moonlit field", "near a waterfall",
]
STYLES = [
    "a high-resolution photograph", "an oil painting", "a watercolor illustration",
    "a digital painting", "a charcoal sketch", "a black-and-white photograph",
    "a vintage postcard illustration", "a low-poly 3D render", "an ink wash painting",
    "a pastel drawing", "a studio photograph", "a stained-glass depiction",
]
MODIFIERS = [
    "with dramatic lighting", "with soft focus", "in vivid color",
    "with a shallow depth of field", "in muted tones", "with long shadows",
    "in high contrast", "with a warm color palette", "with a cool color palette",
    "rendered in fine detail",
]


def load_coco_captions(coco_path: Path, n: int, seed: int) -> list[str]:
    with coco_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    captions = [c["caption"].strip() for c in data["annotations"] if c.get("caption")]
    rng = random.Random(seed)
    rng.shuffle(captions)
    return captions[:n]


def build_combinatorial_bank(n: int, seed: int) -> list[str]:
    """We deterministically sample n *distinct* prompts from the SUBJECTS x
    SETTINGS x STYLES x MODIFIERS cartesian product (tens of thousands of
    combinations); we use this only when real COCO captions are unavailable."""
    combos = list(itertools.product(STYLES, SUBJECTS, SETTINGS, MODIFIERS))
    rng = random.Random(seed)
    rng.shuffle(combos)
    if n > len(combos):
        raise ValueError(f"Requested n={n} exceeds combinatorial bank size={len(combos)}")
    chosen = combos[:n]
    return [f"{style} of {subject} {setting}, {modifier}" for style, subject, setting, modifier in chosen]


def build_prompts(n: int, seed: int) -> list[str]:
    coco_candidates = [
        Path("robin_official") / "coco" / "captions_val2017.json",
        Path("..") / "robin_official" / "coco" / "captions_val2017.json",
    ]
    for p in coco_candidates:
        if p.exists():
            try:
                return load_coco_captions(p, n=n, seed=seed)
            except Exception as exc:  # pragma: no cover
                print(f"[01] COCO load failed ({p}): {exc}, falling back to combinatorial bank")
                break
    else:
        print("[01] No COCO captions file found; using combinatorial synthetic prompt bank")
    return build_combinatorial_bank(n, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--mode", choices=["debug", "main"], default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs_p3/prompts"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(args.n, args.seed)
    fname = f"prompt_list_{args.mode}.csv" if args.mode == "debug" else "prompt_list.csv"
    out_path = args.out_dir / fname
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "prompt"])
        for i, p in enumerate(prompts):
            w.writerow([f"{i:06d}", p])
    print(f"[01] Wrote {len(prompts)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
