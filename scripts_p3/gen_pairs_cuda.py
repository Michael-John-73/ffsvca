"""
scripts_p3/gen_pairs_cuda.py

We generate paired non-watermarked + watermarked images on RunPod/CUDA using
ROBIN's *released, adversarially optimized* SD2.1-base watermark checkpoint
(optimized_r5_15_step10.pt), replacing our earlier SD1.5 no-training variant.

Both images of a pair share the exact same initial latents (set_random_seed(i
+ gen_seed) -> pipe.get_random_latents()), matching ROBIN's own generation
convention (we verified this against robin_official/_gen_clean_pair.py). We inject
the watermark at step 35 of 50 via the ROBINStableDiffusionPipeline 3-way pass.

GPU memory batching (RTX 4090, 24 GB): we generate images in mini-batches of
--batch_size prompts per pipe() call (2 calls per batch: one clean, one
watermarked), instead of one call per image. This amortizes fixed kernel-launch
/ text-encoder overhead across the batch, the same strategy we already use in
03_score_attacks_cuda.py (batch_size=34) for the scoring stage. We start with
--batch_size 16 and reduce (8/4/1) if we hit CUDA OOM; we increase toward 24-32
if headroom remains (watching nvidia-smi).

IMPORTANT: we run a small smoke test first (--start 0 --end 4 --batch_size 4)
and visually check the saved images before launching the full 1000-image run,
since batch>1 requires expanding gt_patch/opt_acond to the batch dimension
(not exercised by ROBIN's original single-image code path).

Usage (per seed, on the pod):
    python scripts_p3/gen_pairs_cuda.py --gen_seed 0 --start 0 --end 1000 \
        --batch_size 16 \
        --out_clean_dir outputs_p3/clean/seed_0 \
        --out_wm_dir outputs_p3/watermarked/seed_0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROBIN_DIR = str(Path(__file__).resolve().parents[2] / "robin_official")
sys.path.insert(0, _ROBIN_DIR)
from inverse_stable_diffusion import InversableStableDiffusionPipeline  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402
from optim_utils import set_random_seed, get_watermarking_mask  # noqa: E402


def load_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["prompt"] for r in rows]


def build_args_ns(a: argparse.Namespace) -> argparse.Namespace:
    """We build the namespace that get_watermarking_mask consumes (mirrors ROBIN's own args)."""
    return argparse.Namespace(
        w_mask_shape="circle", w_channel=a.w_channel,
        w_up_radius=a.w_up_radius, w_low_radius=a.w_low_radius,
        w_injection=a.w_injection, w_measurement=a.w_measurement,
        w_pattern_const=a.w_pattern_const,
    )


def is_black(img: Image.Image, mean_thr: float = 2.0, std_thr: float = 1.0) -> bool:
    """We use the same criterion as robin_official/detect_black_images.py."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return bool(arr.mean() < mean_thr and arr.std() < std_thr)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    # stabilityai/stable-diffusion-2-1-base is gated (HTTP 401); we use this open mirror.
    ap.add_argument("--model_id", default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--wm_path", default=str(
        Path(__file__).resolve().parents[2] / "robin_official" / "ckpts" / "optimized_r5_15_step10.pt"))
    ap.add_argument("--prompts_file", type=Path,
                     default=Path("outputs_p3/prompts/prompt_list.csv"))
    ap.add_argument("--out_clean_dir", type=Path, required=True)
    ap.add_argument("--out_wm_dir", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_length", type=int, default=512)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--watermarking_steps", type=int, default=35)
    ap.add_argument("--lguidance", type=float, default=7.5)
    ap.add_argument("--w_channel", type=int, default=3)
    ap.add_argument("--w_up_radius", type=int, default=15)
    ap.add_argument("--w_low_radius", type=int, default=5)
    ap.add_argument("--w_injection", default="complex")
    ap.add_argument("--w_measurement", default="l1_complex")
    ap.add_argument("--w_pattern_const", type=float, default=0)
    ap.add_argument("--black_retries", type=int, default=3,
                    help="Per-image regeneration attempts for a black frame.")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    args.out_clean_dir.mkdir(parents=True, exist_ok=True)
    args.out_wm_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not visible to torch"
    prompts = load_prompts(args.prompts_file)

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, scheduler=scheduler, torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    # we make the VAE decode one latent at a time; batched 512x512 decode OOMs on 24GB.
    pipe.enable_vae_slicing()
    log(f"model={args.model_id} wm_path={args.wm_path} device={device} "
        f"dtype=fp16 steps={args.num_inference_steps} wm_step={args.watermarking_steps}")

    text_embeddings = pipe.get_text_embedding("")
    ckpt = torch.load(args.wm_path, map_location=device)
    opt_wm = ckpt["opt_wm"].to(device).to(torch.complex64)         # (1,4,64,64)
    opt_acond = ckpt["opt_acond"].to(device).to(text_embeddings.dtype)  # (1,77,dim)
    mask_args = build_args_ns(args)

    def latents_for(idxs: list[int]) -> torch.Tensor:
        lat = []
        for k in idxs:
            set_random_seed(k + args.gen_seed)
            lat.append(pipe.get_random_latents())
        return torch.cat(lat, dim=0)

    def generate(batch_prompts, latents, watermarked: bool):
        kw = {}
        if watermarked:
            b = latents.shape[0]
            kw = dict(
                watermarking_mask=get_watermarking_mask(latents, mask_args, device),
                watermarking_steps=args.watermarking_steps, args=mask_args,
                gt_patch=opt_wm.expand(b, -1, -1, -1).clone(),
                lguidance=args.lguidance,
                opt_acond=opt_acond.expand(b, -1, -1).clone(),
            )
        return pipe(
            batch_prompts, num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length, width=args.image_length,
            latents=latents, **kw,
        ).images

    def repair_black(images, idxs, watermarked: bool, tag: str) -> list[int]:
        """We re-generate any black frame on its own and return indices still black."""
        bad = [n for n, im in enumerate(images) if is_black(im)]
        if not bad:
            return []
        log(f"BLACK {tag}: {len(bad)} frame(s) at idx={[idxs[n] for n in bad]} -> retrying")
        still = []
        for n in bad:
            k = idxs[n]
            for attempt in range(1, args.black_retries + 1):
                im = generate([prompts[k]], latents_for([k]), watermarked)[0]
                if not is_black(im):
                    images[n] = im
                    log(f"BLACK {tag}: idx={k} recovered on attempt {attempt}")
                    break
            else:
                still.append(k)
                log(f"BLACK {tag}: idx={k} STILL BLACK after {args.black_retries} retries")
        return still

    total = args.end - args.start
    unresolved: list[str] = []
    t0 = time.time()
    n_done = 0
    i = args.start
    log(f"start seed={args.gen_seed} range=[{args.start},{args.end}) batch={args.batch_size}")
    pbar = tqdm(total=total, unit="img", desc=f"seed{args.gen_seed}",
                dynamic_ncols=True, mininterval=1.0)
    while i < args.end:
        j = min(i + args.batch_size, args.end)
        idxs = list(range(i, j))
        batch_prompts = [prompts[k] for k in idxs]

        clean_paths = [args.out_clean_dir / f"ori-lg{args.guidance_scale}-{k}.jpg" for k in idxs]
        wm_paths = [args.out_wm_dir / f"wm-lg{args.guidance_scale}-{k}.jpg" for k in idxs]
        if args.skip_existing and all(p.exists() for p in clean_paths + wm_paths):
            n_done += len(idxs)
            pbar.update(len(idxs))
            i = j
            continue

        latents_clean = latents_for(idxs)

        imgs_clean = generate(batch_prompts, latents_clean, watermarked=False)
        unresolved += [f"clean:{k}" for k in repair_black(imgs_clean, idxs, False, "clean")]
        for img, p in zip(imgs_clean, clean_paths):
            img.save(p)

        imgs_wm = generate(batch_prompts, latents_clean.clone(), watermarked=True)
        unresolved += [f"wm:{k}" for k in repair_black(imgs_wm, idxs, True, "wm")]
        for img, p in zip(imgs_wm, wm_paths):
            img.save(p)

        n_done += len(idxs)
        el = time.time() - t0
        rate = n_done / el if el > 0 else 0.0
        eta = (total - n_done) / rate if rate > 0 else 0.0
        pbar.update(len(idxs))
        pbar.set_postfix(black=len(unresolved),
                         vram=f"{torch.cuda.max_memory_allocated()/2**30:.1f}G")
        log(f"seed={args.gen_seed} {n_done}/{total} ({100*n_done/total:5.1f}%) "
            f"batch={i}:{j} {rate:.2f} img/s elapsed={el/60:.1f}m eta={eta/60:.1f}m "
            f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
        i = j

    pbar.close()
    log(f"finished seed={args.gen_seed}: {args.out_clean_dir} + {args.out_wm_dir}")
    if unresolved:
        log(f"FAIL unresolved black frames: {unresolved}")
        sys.exit(1)
    log(f"OK no black frames (mean<2.0 and std<1.0) in {total} pairs")


if __name__ == "__main__":
    main()
