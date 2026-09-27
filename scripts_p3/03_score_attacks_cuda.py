"""
scripts_p3/03_score_attacks_cuda.py  (CUDA backend, external GPU)

We score with the identical contract as 03_score_attacks.py (OpenVINO backend),
for use on a machine with a real CUDA GPU (much faster than Intel Arc/OpenVINO
for the 170,000 DDIM-inversion workload of our full 5-seed x 1000-image x
17-attack scoring pass).

Our target environment: RunPod (external GPU hosting), CUDA 12.1. No CUDA-version
record exists for the original robin_official/package.txt stack (torch==
1.13.0+cu117 / diffusers==0.11.1); we do NOT require that stack here. We
previously ran this exact codebase (inverse_stable_diffusion.py, unmodified) on a
newer stack — Python 3.12 + a post-0.11.1 diffusers — and the run completed
successfully, only emitting harmless FutureWarnings (`unet.in_channels` and
`_encode_prompt()` deprecations). So we expect a modern CUDA-12.1 torch/diffusers
pair to work without code changes.

Our RunPod setup:
    1. We launch a pod: a "CUDA 12.1"-tagged PyTorch template, RTX 4090 GPU
       (widely available; RTX 5090 may not yet be a listed RunPod SKU —
       we check availability), >=20GB container disk.
    2. We transfer data with runpodctl (we need no network volume for a
       one-off run):
           local: runpodctl send outputs_p3/clean outputs_p3/watermarked \
                      outputs_p3/manifests robin_official scripts_p3
           pod:   runpodctl receive <code>
    3. On the pod we install requirements.txt (torch 2.3.1, CUDA 12.1 wheels).
    4. python scripts_p3/03_score_attacks_cuda.py --gen_seed 0 --start 0 --end 1000
    5. We copy results back:
           pod:   runpodctl send outputs_p3/scores
           local: runpodctl receive <code>
       then we run 04_parse_score_dump.py locally.

We read:  outputs_p3/clean/seed_{N}/ori-lg7.5-{i}.jpg
          outputs_p3/watermarked/seed_{N}/wm-lg7.5-{i}.jpg
          outputs_p3/manifests/attack_manifest.json
We write: outputs_p3/scores/seed_{N}/<attack_id>/scores.json
          (04_parse_score_dump.py consumes it; the format is identical regardless of backend)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# We keep robin_official as a sibling of the PAPER3 repo root on both Windows (F:/RCE/robin_official)
# and the RunPod pod (/workspace/robin_official); we resolve it relative to this file so our
# script works unmodified on either machine.
_ROBIN_DIR = str(Path(__file__).resolve().parents[2] / "robin_official")
sys.path.insert(0, _ROBIN_DIR)
from inverse_stable_diffusion import InversableStableDiffusionPipeline  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402
from optim_utils import get_watermarking_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _attack_scoring_common import run_scoring_loop, WmArgs  # noqa: E402


class CudaRobinPipe:
    """We wrap InversableStableDiffusionPipeline in this thin adapter to match the
    OVRobinPipe surface that _attack_scoring_common.run_scoring_loop uses.
    """

    def __init__(self, model_id: str, device: str = "cuda", revision: str | None = None,
                 variant: str | None = "fp16"):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        if not hasattr(scheduler, "final_alpha_cumprod"):
            scheduler.final_alpha_cumprod = torch.tensor(1.0)
        kwargs = dict(scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
        if revision:
            kwargs["revision"] = revision
        if variant:
            kwargs["variant"] = variant
        try:
            self.pipe = InversableStableDiffusionPipeline.from_pretrained(model_id, **kwargs).to(device)
        except Exception:
            kwargs.pop("variant", None)
            self.pipe = InversableStableDiffusionPipeline.from_pretrained(model_id, **kwargs).to(device)
        self.device = device

    def get_text_embedding(self, prompt):
        return self.pipe.get_text_embedding(prompt)

    def get_random_latents(self):
        return self.pipe.get_random_latents()

    def get_image_latents(self, image, sample=False):
        image = image.to(device=self.device, dtype=self.pipe.text_encoder.dtype)
        return self.pipe.get_image_latents(image, sample=sample)

    def forward_diffusion(self, latents, text_embeddings, guidance_scale=1.0,
                          num_inference_steps=50, latents_b=None):
        return self.pipe.forward_diffusion(
            latents=latents, text_embeddings=text_embeddings,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
            latents_b=latents_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_seed", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--clean_dir", type=Path, default=None)
    ap.add_argument("--wm_dir", type=Path, default=None)
    ap.add_argument("--wm_ckpt", default=str(Path(__file__).resolve().parents[2] / "robin_official" / "ckpts" / "optimized_r5_15_step10.pt"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("outputs_p3/manifests/attack_manifest.json"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    # stabilityai/stable-diffusion-2-1-base is gated (HTTP 401); we use this open mirror.
    ap.add_argument("--model_id", default="sd2-community/stable-diffusion-2-1-base")
    ap.add_argument("--revision", default=None)
    # the mirror has no fp16 branch; we get fp16 from torch_dtype, not from this.
    ap.add_argument("--variant", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint_every", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=34,
                    help="Batch all pending (attack,label) variants of an image into groups of"
                         " this size per UNet forward pass (1 = old sequential behaviour).")
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        raise SystemExit("[03-cuda] CUDA not available on this machine; pass --device cpu to force CPU (slow).")

    # We always keep cudnn.deterministic=True, cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    gen_seed = args.gen_seed
    clean_dir = args.clean_dir or Path(f"outputs_p3/clean/seed_{gen_seed}")
    wm_dir = args.wm_dir or Path(f"outputs_p3/watermarked/seed_{gen_seed}")
    out_dir = args.out_dir or Path(f"outputs_p3/scores/seed_{gen_seed}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    attacks = manifest["attacks"]

    pipe = CudaRobinPipe(args.model_id, device=args.device, revision=args.revision or None, variant=args.variant or None)
    ckpt = torch.load(args.wm_ckpt, map_location="cpu")
    opt_wm = ckpt["opt_wm"].to(torch.complex64).to(args.device)
    null_emb = pipe.get_text_embedding("")

    ref = pipe.get_random_latents()
    watermarking_mask = get_watermarking_mask(ref, WmArgs(), args.device)

    if args.batch_size > 1:
        from _attack_scoring_common import run_scoring_loop_batched
        run_scoring_loop_batched(
            pipe, opt_wm, watermarking_mask, null_emb, attacks,
            gen_seed=gen_seed, start=args.start, end=args.end,
            clean_dir=clean_dir, wm_dir=wm_dir, out_dir=out_dir,
            steps=args.steps, num_inference_steps=args.num_inference_steps,
            checkpoint_every=args.checkpoint_every, batch_size=args.batch_size,
        )
        return

    run_scoring_loop(
        pipe, opt_wm, watermarking_mask, null_emb, attacks,
        gen_seed=gen_seed, start=args.start, end=args.end,
        clean_dir=clean_dir, wm_dir=wm_dir, out_dir=out_dir,
        steps=args.steps, num_inference_steps=args.num_inference_steps,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
