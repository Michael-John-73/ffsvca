"""
scripts_p3/score_dwtdctsvd.py

Producer of outputs_p3/scores/dwtdctsvd_scores.csv (consumed by 08e).

Detector: DwtDctSvd from the invisible-watermark library
    https://github.com/ShieldMnt/invisible-watermark  (MIT)
    pip install invisible-watermark
The watermark used by the public Stable Diffusion v1.5 / v2.1 release.

This is a *different family* than ROBIN: spatial-domain DWT + block DCT
+ SVD with a fixed-length bit-message key, and no DDIM inversion step.
The intent (App D / E9) is to show that the fixed-FPR verification
protocol generalizes across detector families, not to compare detector
strengths.

Pipeline per clean source image:
    1. Embed a fixed K-bit key into a copy -> "watermarked" image.
    2. For each attack in {none, jpeg_q50, cropping}:
         a. apply attack to clean image    -> decode bits -> bit_acc_clean
         b. apply attack to watermarked    -> decode bits -> bit_acc_wm
       (Bit accuracy under the same fixed key is the natural score; for
        clean images this concentrates near 0.5 by chance, for
        watermarked images it concentrates near 1.0 and degrades with
        attack severity.)
    3. score_z = bit_acc  (higher = more watermark-like, matches the
       convention used elsewhere).

Split assignment (matches §6.5 of 논문3.md):
    cal_ratio of clean@none rows -> split=cal
    everything else              -> split=test

Output columns:
    image_id, attack_id, source_label, split, score_z

CLI:
    python scripts_p3/score_dwtdctsvd.py \
        --clean_dir outputs_p3/clean \
        --out outputs_p3_appd/scores/dwtdctsvd_scores.csv \
        --num_images 1000 --cal_ratio 0.5 --split_seed 42 \
        --key_len 64 --gen_seed 0
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Lazy import: invisible-watermark + cv2 are only required at runtime.
def _load_imwatermark():
    try:
        import cv2  # noqa: F401
        from imwatermark import WatermarkDecoder, WatermarkEncoder
    except ImportError as exc:
        raise SystemExit(
            "[score_dwtdctsvd] Missing dependency. Run:\n"
            "    pip install invisible-watermark opencv-python\n"
            f"(import error: {exc})"
        )
    return WatermarkEncoder, WatermarkDecoder


# ---- attacks ----------------------------------------------------------------

def attack_none(img: Image.Image) -> Image.Image:
    return img.copy()


def attack_jpeg_q50(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=50)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def attack_cropping(img: Image.Image, ratio: float = 0.75) -> Image.Image:
    """Center-crop to `ratio` of each side, then resize back to original size."""
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)


ATTACKS = {
    "none": attack_none,
    "jpeg_q50": attack_jpeg_q50,
    "cropping": attack_cropping,
}


# ---- key + encode/decode helpers --------------------------------------------

def make_bit_key(key_len: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=key_len, dtype=np.int32).tolist()


def encode_image(encoder, bgr_np, key_bits: List[int]) -> np.ndarray:
    encoder.set_watermark("bits", key_bits)
    return encoder.encode(bgr_np, "dwtDctSvd")


def decode_bit_accuracy(decoder, bgr_np, key_bits: List[int]) -> float:
    decoded = decoder.decode(bgr_np, "dwtDctSvd")
    # decoded is a list/array of bits length == key_len
    arr = np.asarray(decoded, dtype=np.int32).reshape(-1)
    key = np.asarray(key_bits, dtype=np.int32).reshape(-1)
    n = min(arr.size, key.size)
    if n == 0:
        return float("nan")
    return float(np.mean(arr[:n] == key[:n]))


# ---- main -------------------------------------------------------------------

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR uint8 ndarray."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return arr[:, :, ::-1].copy()


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr[:, :, ::-1].copy(), mode="RGB")


def list_images(d: Path, n: int) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    return files[:n]


def assign_splits(image_ids: List[str], cal_ratio: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(image_ids))
    rng.shuffle(idx)
    n_cal = int(round(len(image_ids) * cal_ratio))
    cal_set = set(image_ids[i] for i in idx[:n_cal])
    return {iid: ("cal" if iid in cal_set else "test") for iid in image_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", type=Path, required=True,
                    help="Directory of source CLEAN images.")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs_p3_appd/scores/dwtdctsvd_scores.csv"))
    ap.add_argument("--num_images", type=int, default=1000)
    ap.add_argument("--key_len", type=int, default=64,
                    help="Watermark message length in bits.")
    ap.add_argument("--gen_seed", type=int, default=0,
                    help="Seed for the bit key (single seed for App D).")
    ap.add_argument("--cal_ratio", type=float, default=0.5)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS.keys()))
    args = ap.parse_args()

    if not args.clean_dir.is_dir():
        raise SystemExit(f"[score_dwtdctsvd] clean_dir not found: {args.clean_dir}")
    for atk in args.attacks:
        if atk not in ATTACKS:
            raise SystemExit(f"[score_dwtdctsvd] unknown attack: {atk}")

    WatermarkEncoder, WatermarkDecoder = _load_imwatermark()

    image_paths = list_images(args.clean_dir, args.num_images)
    if not image_paths:
        raise SystemExit(f"[score_dwtdctsvd] no images found in {args.clean_dir}")
    image_ids = [p.stem for p in image_paths]
    splits = assign_splits(image_ids, args.cal_ratio, args.split_seed)

    key_bits = make_bit_key(args.key_len, args.gen_seed)
    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder("bits", args.key_len)

    rows: List[dict] = []
    n_total = len(image_paths)
    for i, path in enumerate(image_paths):
        iid = path.stem
        try:
            pil_clean = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[score_dwtdctsvd] skip {path}: {exc}", file=sys.stderr)
            continue
        bgr_clean = pil_to_bgr(pil_clean)
        bgr_wm = encode_image(encoder, bgr_clean, key_bits)
        pil_wm = bgr_to_pil(bgr_wm)

        for atk in args.attacks:
            attacked_clean = ATTACKS[atk](pil_clean)
            attacked_wm = ATTACKS[atk](pil_wm)

            sc_clean = decode_bit_accuracy(decoder, pil_to_bgr(attacked_clean), key_bits)
            sc_wm = decode_bit_accuracy(decoder, pil_to_bgr(attacked_wm), key_bits)

            rows.append({
                "image_id": iid,
                "attack_id": atk,
                "source_label": "clean",
                "split": splits[iid],
                "score_z": sc_clean,
            })
            rows.append({
                "image_id": iid,
                "attack_id": atk,
                "source_label": "watermarked",
                "split": splits[iid],
                "score_z": sc_wm,
            })

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(f"[score_dwtdctsvd] {i + 1}/{n_total} images scored", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[score_dwtdctsvd] wrote {args.out}  ({len(rows)} rows, "
          f"{n_total} images, key_len={args.key_len}, gen_seed={args.gen_seed})", flush=True)


if __name__ == "__main__":
    main()
