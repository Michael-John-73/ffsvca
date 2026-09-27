"""
scripts_p3/score_adaptive_regen.py

Producer of outputs_p3/scores/regen_scores.csv (consumed by 08d_eval_adaptive_regen.py, E8).

paper3.md §6.4(d): SD 1.5 img2img regeneration applied to watermarked images
at strength in {0.2, 0.3, 0.5}, with the SAME prompt as the original
generation, plus a matched clean control regenerated at the same strengths.
Single seed (gen_seed=0, per Appendix A).

Sample composition (per paper3_flow_실험정합성_검토.md §6.1, already-approved
plan): 200 source pairs (clean_i, watermarked_i) x 2 classes x 3 strengths
= 1200 regenerated images total.

img2img is implemented manually (partial re-noising + truncated denoising
loop over the pipeline's own unet/scheduler) since InversableStableDiffusionPipeline
has no built-in strength-based img2img call; this does not modify
robin_official/ (substrate untouched).

Usage:
    python scripts_p3/score_adaptive_regen.py --num_images 200 --gen_seed 0

Output columns: image_id, strength, source_label, score_z
    source_label in {"watermarked", "clean"}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

_ROBIN_DIR = str(Path(__file__).resolve().parents[2] / "robin_official")
sys.path.insert(0, _ROBIN_DIR)
from inverse_stable_diffusion import InversableStableDiffusionPipeline  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402
from optim_utils import get_watermarking_mask, get_dataset, transform_img  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _attack_scoring_common import batched_ring_distance, WmArgs  # noqa: E402


@torch.no_grad()
def img2img_regenerate(pipe, image_latents, prompt, strength, num_inference_steps, guidance_scale, device):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    noise = torch.randn(image_latents.shape, device=device, dtype=image_latents.dtype)
    latents = scheduler.add_noise(image_latents, noise, timesteps[:1])

    uncond = pipe.get_text_embedding("")
    cond = pipe.get_text_embedding(prompt)
    text_embeddings = torch.cat([uncond, cond])
    do_cfg = guidance_scale > 1.0

    for t in timesteps:
        lm = torch.cat([latents] * 2) if do_cfg else latents
        lm = scheduler.scale_model_input(lm, t)
        noise_pred = pipe.unet(lm, t, encoder_hidden_states=text_embeddings).sample
        if do_cfg:
            u, c = noise_pred.chunk(2)
            noise_pred = u + guidance_scale * (c - u)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    image = pipe.decode_image(latents)
    image = pipe.torch_to_numpy(image)
    return pipe.numpy_to_pil(image)[0]


@torch.no_grad()
def score_one(pipe, img, null_emb, opt_wm, watermarking_mask, steps, num_inference_steps, device):
    t = transform_img(img).unsqueeze(0).to(torch.float32).to(device=device, dtype=pipe.text_encoder.dtype)
    lat = pipe.get_image_latents(t, sample=False)
    buf = [lat]
    _, buf, _ = pipe.forward_diffusion(
        latents=lat, text_embeddings=null_emb, guidance_scale=1.0,
        num_inference_steps=num_inference_steps, latents_b=buf)
    d = batched_ring_distance(buf[steps], opt_wm, watermarking_mask)
    return float(d[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--clean_dir", type=Path, default=None)
    ap.add_argument("--wm_dir", type=Path, default=None)
    ap.add_argument("--wm_ckpt", default=str(Path(__file__).resolve().parents[2] / "robin_official" / "ckpts" / "optimized_r5_15_step10.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs_p3/scores/regen_scores.csv"))
    # stabilityai/stable-diffusion-2-1-base is gated (HTTP 401); this is the open mirror.
    ap.add_argument("--model_id", default="sd2-community/stable-diffusion-2-1-base")
    # the mirror has no fp16 branch; fp16 comes from torch_dtype, not from this.
    ap.add_argument("--variant", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    ap.add_argument("--checkpoint_every", type=int, default=20)
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    gen_seed = args.gen_seed
    clean_dir = args.clean_dir or Path(f"outputs_p3/clean/seed_{gen_seed}")
    wm_dir = args.wm_dir or Path(f"outputs_p3/watermarked/seed_{gen_seed}")

    class _DsArgs:
        dataset = "Gustavosta/Stable-Diffusion-Prompts"
    dataset, prompt_key = get_dataset(_DsArgs())

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    if not hasattr(scheduler, "final_alpha_cumprod"):
        scheduler.final_alpha_cumprod = torch.tensor(1.0)
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, scheduler=scheduler, safety_checker=None,
        torch_dtype=torch.float16, variant=args.variant,
    ).to(args.device)

    null_emb = pipe.get_text_embedding("")
    ckpt = torch.load(args.wm_ckpt, map_location="cpu")
    opt_wm = ckpt["opt_wm"].to(torch.complex64).to(args.device)
    ref = pipe.get_random_latents()
    watermarking_mask = get_watermarking_mask(ref, WmArgs(), args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = {}
    if args.out.exists():
        import csv as _csv
        with args.out.open(newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                items[(row["image_id"], row["strength"], row["source_label"])] = row

    def checkpoint():
        import csv as _csv
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["image_id", "strength", "source_label", "score_z"])
            w.writeheader()
            w.writerows(items.values())

    for i in range(args.num_images):
        image_id = f"{i:06d}"
        c_path = clean_dir / f"ori-lg7.5-{i}.jpg"
        w_path = wm_dir / f"wm-lg7.5-{i}.jpg"
        if not (c_path.exists() and w_path.exists()):
            print(f"[regen] skip idx {i}: missing image(s)")
            continue
        prompt = dataset[i][prompt_key]
        img_clean = Image.open(c_path).convert("RGB")
        img_wm = Image.open(w_path).convert("RGB")

        t_clean = transform_img(img_clean).unsqueeze(0).to(torch.float32).to(device=args.device, dtype=pipe.text_encoder.dtype)
        t_wm = transform_img(img_wm).unsqueeze(0).to(torch.float32).to(device=args.device, dtype=pipe.text_encoder.dtype)
        lat_clean = pipe.get_image_latents(t_clean, sample=False)
        lat_wm = pipe.get_image_latents(t_wm, sample=False)

        for strength in args.strengths:
            key_c = (image_id, str(strength), "clean")
            key_w = (image_id, str(strength), "watermarked")
            if key_c in items and key_w in items:
                continue
            regen_clean = img2img_regenerate(pipe, lat_clean, prompt, strength,
                                              args.num_inference_steps, args.guidance_scale, args.device)
            regen_wm = img2img_regenerate(pipe, lat_wm, prompt, strength,
                                           args.num_inference_steps, args.guidance_scale, args.device)
            d_clean = score_one(pipe, regen_clean, null_emb, opt_wm, watermarking_mask,
                                 args.steps, args.num_inference_steps, args.device)
            d_wm = score_one(pipe, regen_wm, null_emb, opt_wm, watermarking_mask,
                              args.steps, args.num_inference_steps, args.device)
            items[key_c] = {"image_id": image_id, "strength": strength, "source_label": "clean", "score_z": -d_clean}
            items[key_w] = {"image_id": image_id, "strength": strength, "source_label": "watermarked", "score_z": -d_wm}

        if i % args.checkpoint_every == 0:
            checkpoint()
            print(f"[regen] idx={i} done, {len(items)} rows so far")

    checkpoint()
    print(f"[regen] done: {len(items)} rows -> {args.out}")


if __name__ == "__main__":
    main()
