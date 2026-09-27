"""
scripts_p3/gen_treering_cuda.py  (RunPod / CUDA)

Our second-detector experiment (the Tree-Ring comparator (E10)): we generate Tree-Ring
watermarked images on the SAME SD2.1-base pipeline, prompts and per-image initial latents
as our archived non-watermarked images (outputs_p3/clean/seed_N/ori-lg7.5-{i}.jpg,
which gen_pairs_cuda.py produced).

We use the Tree-Ring configuration of the official repo README command
(github.com/YuxinWenRick/tree-ring-watermark):
    --w_channel 3 --w_pattern ring ; argparse defaults w_radius=10,
    w_mask_shape=circle, w_seed=999999, w_injection=complex,
    w_measurement=l1_complex.
Our circle_mask / ring pattern / injection below reproduce the official
optim_utils.py functions (the ROBIN fork changed circle_mask to an annulus
r_min < d <= r_max, which would drop the DC bin, so we do NOT use it here).
We take FFTs in float32 (official code runs them on the fp16 latent).

Initial latents: we use set_random_seed(i + gen_seed) -> pipe.get_random_latents(),
identical to gen_pairs_cuda.py, so each Tree-Ring image shares its initial
noise with the archived clean image before we write the key into it.

Usage (pod, repo root = /workspace/PAPER3):
    python scripts_p3/gen_treering_cuda.py --gen_seed 0 --start 0 --end 20     # pilot
    python scripts_p3/gen_treering_cuda.py --gen_seed 0 --start 0 --end 1000 --skip_existing
"""
from __future__ import annotations

import argparse
import copy
import csv
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
from optim_utils import set_random_seed  # noqa: E402

TR_CHANNEL = 3
TR_RADIUS = 10
TR_SEED = 999999


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [r["prompt"] for r in csv.DictReader(f)]


def is_black(img: Image.Image, mean_thr: float = 2.0, std_thr: float = 1.0) -> bool:
    """We use the same criterion as gen_pairs_cuda.py."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return bool(arr.mean() < mean_thr and arr.std() < std_thr)


def tr_circle_mask(size: int = 64, r: int = 10) -> np.ndarray:
    """We reproduce the official Tree-Ring circle_mask (disk d <= r, DC bin included)."""
    x0 = y0 = size // 2
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def tr_watermarking_mask(shape, device) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[:, TR_CHANNEL] = torch.tensor(tr_circle_mask(shape[-1], r=TR_RADIUS), device=device)
    return mask


def tr_ring_pattern(pipe, device) -> torch.Tensor:
    """We reproduce the official get_watermarking_pattern, w_pattern='ring'."""
    set_random_seed(TR_SEED)
    gt_init = pipe.get_random_latents().to(torch.float32)
    gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
    gt_patch_tmp = copy.deepcopy(gt_patch)
    for i in range(TR_RADIUS, 0, -1):
        tmp_mask = torch.tensor(tr_circle_mask(gt_init.shape[-1], r=i), device=device)
        for j in range(gt_patch.shape[1]):
            gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    return gt_patch.to(torch.complex64)


def tr_inject(latents: torch.Tensor, mask: torch.Tensor, gt_patch: torch.Tensor) -> torch.Tensor:
    """We reproduce the official inject_watermark, w_injection='complex' (batched)."""
    dtype = latents.dtype
    fft = torch.fft.fftshift(torch.fft.fft2(latents.to(torch.float32)), dim=(-1, -2))
    b = latents.shape[0]
    m = mask.expand(b, -1, -1, -1)
    fft[m] = gt_patch.expand(b, -1, -1, -1)[m].clone()
    return torch.fft.ifft2(torch.fft.ifftshift(fft, dim=(-1, -2))).real.to(dtype)


def load_key(path: Path, pipe, device):
    """We build the key once and persist it; our later runs (and the scorer) reuse the file."""
    if path.exists():
        k = torch.load(path, map_location=device)
        return k["gt_patch"].to(device), k["mask"].to(device)
    gt_patch = tr_ring_pattern(pipe, device)
    mask = tr_watermarking_mask(gt_patch.shape, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gt_patch": gt_patch.cpu(), "mask": mask.cpu(),
                "config": {"w_channel": TR_CHANNEL, "w_radius": TR_RADIUS, "w_seed": TR_SEED,
                           "w_pattern": "ring", "w_mask_shape": "circle",
                           "w_injection": "complex", "w_measurement": "l1_complex"}}, path)
    log(f"key written: {path}")
    return gt_patch, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--prompts_file", type=Path, default=Path("outputs_p3/prompts/prompt_list.csv"))
    ap.add_argument("--key_path", type=Path, default=Path("outputs_p3/treering/treering_key.pt"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--image_length", type=int, default=512)
    ap.add_argument("--black_retries", type=int, default=3)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not visible to torch"
    device = "cuda"
    out_dir = args.out_dir or Path(f"outputs_p3/treering/watermarked/seed_{args.gen_seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts_file)

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, scheduler=scheduler, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.enable_vae_slicing()
    gt_patch, mask = load_key(args.key_path, pipe, device)
    log(f"model={args.model_id} steps={args.num_inference_steps} cfg={args.guidance_scale} "
        f"TR channel={TR_CHANNEL} radius={TR_RADIUS} w_seed={TR_SEED} mask_bins={int(mask.sum())}")

    def latents_for(idxs):
        lat = []
        for k in idxs:
            set_random_seed(k + args.gen_seed)
            lat.append(pipe.get_random_latents())
        return tr_inject(torch.cat(lat, dim=0), mask, gt_patch)

    def generate(idxs):
        return pipe([prompts[k] for k in idxs], num_images_per_prompt=1,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length, width=args.image_length,
                    latents=latents_for(idxs)).images

    total = args.end - args.start
    unresolved, n_done, t0 = [], 0, time.time()
    log(f"start seed={args.gen_seed} range=[{args.start},{args.end}) batch={args.batch_size} -> {out_dir}")
    pbar = tqdm(total=total, unit="img", desc=f"TR seed{args.gen_seed}", dynamic_ncols=True, mininterval=1.0)
    i = args.start
    while i < args.end:
        j = min(i + args.batch_size, args.end)
        idxs = list(range(i, j))
        paths = [out_dir / f"trwm-lg{args.guidance_scale}-{k}.jpg" for k in idxs]
        if not (args.skip_existing and all(p.exists() for p in paths)):
            imgs = generate(idxs)
            for n, im in enumerate(imgs):
                if is_black(im):
                    k = idxs[n]
                    log(f"BLACK idx={k} -> retrying")
                    for _ in range(args.black_retries):
                        im = generate([k])[0]
                        if not is_black(im):
                            break
                    else:
                        unresolved.append(k)
                    imgs[n] = im
            for im, p in zip(imgs, paths):
                im.save(p)
        n_done += len(idxs)
        el = time.time() - t0
        rate = n_done / el if el > 0 else 0.0
        pbar.update(len(idxs))
        log(f"seed={args.gen_seed} {n_done}/{total} ({100 * n_done / total:5.1f}%) "
            f"{rate:.2f} img/s elapsed={el / 60:.1f}m eta={(total - n_done) / rate / 60 if rate else 0:.1f}m "
            f"vram={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB")
        i = j
    pbar.close()
    if unresolved:
        log(f"FAIL unresolved black frames: {unresolved}")
        sys.exit(1)
    log(f"OK {total} Tree-Ring images, no black frames -> {out_dir}")


if __name__ == "__main__":
    main()
