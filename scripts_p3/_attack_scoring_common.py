"""
scripts_p3/_attack_scoring_common.py

Shared attack-application + resumable scoring loop used by both backends:
- 03_score_attacks.py       (OpenVINO / Intel Arc GPU, local machine)
- 03_score_attacks_cuda.py  (CUDA, external GPU)

Kept identical across backends so scores are comparable regardless of which
GPU actually ran Phase 2. Any backend-specific pipeline just needs to expose:
    pipe.get_text_embedding(prompt) -> tensor
    pipe.get_image_latents(image_tensor, sample=False) -> tensor
    pipe.forward_diffusion(latents=..., text_embeddings=..., guidance_scale=1.0,
                           num_inference_steps=50, latents_b=[...]) -> (final, latents_b, noise_b)
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from optim_utils import set_random_seed, eval_watermark, transform_img


class WmArgs:
    """Minimal args namespace for optim_utils.get_watermarking_mask/eval_watermark."""
    w_channel = 3
    w_pattern = "ring"
    w_mask_shape = "circle"
    w_up_radius = 15
    w_low_radius = 5
    w_measurement = "l1_complex"
    w_injection = "complex"
    w_pattern_const = 0
    w_seed = 999999


def apply_stage(img: Image.Image, stage: dict, rng_seed: int) -> Image.Image:
    op = stage["op"]
    if op == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(stage["quality"]))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "rotation":
        return transforms.RandomRotation((stage["degrees"], stage["degrees"]))(img)
    if op == "cropping":
        set_random_seed(rng_seed)
        ratio = stage["ratio"]
        return transforms.RandomResizedCrop(img.size, scale=(ratio, ratio), ratio=(ratio, ratio))(img)
    if op == "blurring":
        return img.filter(ImageFilter.GaussianBlur(radius=stage["sigma"]))
    if op == "noise":
        arr = np.array(img).astype(np.float32)
        set_random_seed(rng_seed)
        noise = np.random.normal(0, stage["sigma"], arr.shape) * 255
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    raise NotImplementedError(f"unknown attack op: {op}")


def apply_attack(img: Image.Image, stages: list, rng_seed: int) -> Image.Image:
    for stage in stages:
        img = apply_stage(img, stage, rng_seed)
    return img


def batched_ring_distance(reversed_latents: torch.Tensor, gt_patch: torch.Tensor,
                           watermarking_mask: torch.Tensor) -> torch.Tensor:
    """Vectorized equivalent of optim_utils.eval_watermark's l1_complex distance,
    computed independently per batch item. reversed_latents: [B,C,H,W] real.
    Returns a 1-D tensor of B distances (same formula/scale as eval_watermark).
    """
    fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents.to(torch.float32)), dim=(-1, -2))
    mask = watermarking_mask[0] if watermarking_mask.dim() == 4 else watermarking_mask
    target = gt_patch.to(torch.complex64)
    target = target[0] if target.dim() == 4 else target
    diffs = torch.abs(fft[:, mask] - target[mask].unsqueeze(0))  # [B, n_selected]
    return diffs.mean(dim=1)


def run_scoring_loop(pipe, opt_wm, watermarking_mask, null_emb, attacks,
                      gen_seed: int, start: int, end: int,
                      clean_dir: Path, wm_dir: Path, out_dir: Path,
                      steps: int, num_inference_steps: int,
                      checkpoint_every: int = 10):
    """Backend-agnostic scoring loop. `pipe` must expose the 3 methods
    documented at module level. Writes/resumes outputs_p3/scores/seed_N/<attack_id>/scores.json.
    """
    wargs = WmArgs()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_attack_items = {}
    for atk in attacks:
        f = out_dir / atk["id"] / "scores.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            existing = {(it["prompt_id"], it["label"]): it for it in data.get("items", [])}
        else:
            existing = {}
        per_attack_items[atk["id"]] = existing

    def score_one(img: Image.Image) -> float:
        t = transform_img(img).unsqueeze(0).to(torch.float32)
        lat = pipe.get_image_latents(t, sample=False)
        buf = [lat]
        _, buf, _ = pipe.forward_diffusion(
            latents=lat, text_embeddings=null_emb, guidance_scale=1.0,
            num_inference_steps=num_inference_steps, latents_b=buf)
        d, _ = eval_watermark(buf[steps], buf[steps], watermarking_mask, opt_wm, wargs)
        return float(d)

    def checkpoint():
        for atk in attacks:
            f = out_dir / atk["id"] / "scores.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            items = per_attack_items[atk["id"]]
            f.write_text(json.dumps(
                {"attack_id": atk["id"], "items": list(items.values())}, indent=0),
                encoding="utf-8")

    for i in range(start, end):
        prompt_id = f"{i:06d}"
        c_path = clean_dir / f"ori-lg7.5-{i}.jpg"
        w_path = wm_dir / f"wm-lg7.5-{i}.jpg"
        if not (c_path.exists() and w_path.exists()):
            print(f"[03] skip idx {i}: missing image(s)")
            continue
        img_clean_base = Image.open(c_path).convert("RGB")
        img_wm_base = Image.open(w_path).convert("RGB")

        for atk in attacks:
            key0 = (prompt_id, 0)
            key1 = (prompt_id, 1)
            items = per_attack_items[atk["id"]]
            if key0 in items and key1 in items:
                continue  # already scored (resume)
            rng_seed = gen_seed * 1_000_000 + i
            img_c = apply_attack(img_clean_base.copy(), atk["stages"], rng_seed)
            img_w = apply_attack(img_wm_base.copy(), atk["stages"], rng_seed)
            d_c = score_one(img_c)
            d_w = score_one(img_w)
            items[key0] = {"prompt_id": prompt_id, "seed": gen_seed, "label": 0, "distance": d_c}
            items[key1] = {"prompt_id": prompt_id, "seed": gen_seed, "label": 1, "distance": d_w}

        if (i - start) % checkpoint_every == 0 or i == end - 1:
            checkpoint()
            print(f"[03] gen_seed={gen_seed} checkpoint at idx={i}")

    checkpoint()
    print(f"[03] gen_seed={gen_seed} done: {end - start} images x {len(attacks)} attacks -> {out_dir}")


def run_scoring_loop_batched(pipe, opt_wm, watermarking_mask, null_emb, attacks,
                              gen_seed: int, start: int, end: int,
                              clean_dir: Path, wm_dir: Path, out_dir: Path,
                              steps: int, num_inference_steps: int,
                              checkpoint_every: int = 10, batch_size: int = 34):
    """CUDA-only variant of run_scoring_loop: batches all pending (attack, label)
    variants of an image together into one UNet forward pass per DDIM step,
    instead of 34 sequential batch=1 calls. Same output contract/format as
    run_scoring_loop. Not used by the OpenVINO backend (static-shape IR models
    cannot change batch size without recompilation).
    """
    wargs = WmArgs()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_attack_items = {}
    for atk in attacks:
        f = out_dir / atk["id"] / "scores.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            existing = {(it["prompt_id"], it["label"]): it for it in data.get("items", [])}
        else:
            existing = {}
        per_attack_items[atk["id"]] = existing

    def score_batch(imgs: list) -> list:
        b = len(imgs)
        t = torch.cat([transform_img(im).unsqueeze(0) for im in imgs], dim=0).to(torch.float32)
        lat = pipe.get_image_latents(t, sample=False)
        text_emb = null_emb.expand(b, -1, -1)
        buf = [lat]
        _, buf, _ = pipe.forward_diffusion(
            latents=lat, text_embeddings=text_emb, guidance_scale=1.0,
            num_inference_steps=num_inference_steps, latents_b=buf)
        dists = batched_ring_distance(buf[steps], opt_wm, watermarking_mask)
        return [float(d) for d in dists]

    def checkpoint():
        for atk in attacks:
            f = out_dir / atk["id"] / "scores.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            items = per_attack_items[atk["id"]]
            f.write_text(json.dumps(
                {"attack_id": atk["id"], "items": list(items.values())}, indent=0),
                encoding="utf-8")

    for i in range(start, end):
        prompt_id = f"{i:06d}"
        c_path = clean_dir / f"ori-lg7.5-{i}.jpg"
        w_path = wm_dir / f"wm-lg7.5-{i}.jpg"
        if not (c_path.exists() and w_path.exists()):
            print(f"[03] skip idx {i}: missing image(s)")
            continue
        img_clean_base = Image.open(c_path).convert("RGB")
        img_wm_base = Image.open(w_path).convert("RGB")
        rng_seed = gen_seed * 1_000_000 + i

        pending_imgs, pending_keys = [], []
        for atk in attacks:
            key0, key1 = (prompt_id, 0), (prompt_id, 1)
            items = per_attack_items[atk["id"]]
            if key0 in items and key1 in items:
                continue  # already scored (resume)
            pending_imgs.append(apply_attack(img_clean_base.copy(), atk["stages"], rng_seed))
            pending_keys.append((atk["id"], 0))
            pending_imgs.append(apply_attack(img_wm_base.copy(), atk["stages"], rng_seed))
            pending_keys.append((atk["id"], 1))

        for lo in range(0, len(pending_imgs), batch_size):
            chunk_imgs = pending_imgs[lo:lo + batch_size]
            chunk_keys = pending_keys[lo:lo + batch_size]
            dists = score_batch(chunk_imgs)
            for (atk_id, label), d in zip(chunk_keys, dists):
                per_attack_items[atk_id][(prompt_id, label)] = {
                    "prompt_id": prompt_id, "seed": gen_seed, "label": label, "distance": d}

        if (i - start) % checkpoint_every == 0 or i == end - 1:
            checkpoint()
            print(f"[03] gen_seed={gen_seed} checkpoint at idx={i}")

    checkpoint()
    print(f"[03] gen_seed={gen_seed} done: {end - start} images x {len(attacks)} attacks (batched) -> {out_dir}")
