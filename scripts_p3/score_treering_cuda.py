"""
scripts_p3/score_treering_cuda.py  (RunPod / CUDA)

Our second-detector experiment (the Tree-Ring comparator (E10)), one generation seed:
  label 0: archived clean image  x 17 attack conditions (attack_manifest.json)
  label 1: Tree-Ring image       x "none" only
= 18 inversions per prompt, which we batch in one UNet pass per DDIM step.

Our inversion is the same as 03_score_attacks_cuda.py (same CudaRobinPipe adapter,
null prompt, guidance 1.0, 50 steps, same attack code and rng_seed). From the
single pass we read two distances:
  tr_distance    : Tree-Ring l1_complex distance on the fully inverted latent x_T
                   (official detection point)
  robin_distance : ROBIN distance on buf[--steps], exactly as in our main run;
                   we use it only to check that this run reproduces scores_raw.csv.

We write (resumable, appended per prompt):
  outputs_p3/treering/scores/seed_N/treering_scores.csv
  columns: prompt_id,seed,attack_id,label,tr_distance,robin_distance

Usage (pod, repo root = /workspace/PAPER3):
    python scripts_p3/score_treering_cuda.py --gen_seed 0 --start 0 --end 20     # pilot
    python scripts_p3/score_treering_cuda.py --gen_seed 0 --start 0 --end 1000
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROBIN_DIR = str(_HERE.parents[1] / "robin_official")
sys.path.insert(0, _ROBIN_DIR)
sys.path.insert(0, str(_HERE))
from optim_utils import get_watermarking_mask, transform_img  # noqa: E402
from _attack_scoring_common import WmArgs, apply_attack, batched_ring_distance  # noqa: E402

_spec = importlib.util.spec_from_file_location("score03", _HERE / "03_score_attacks_cuda.py")
_score03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_score03)
CudaRobinPipe = _score03.CudaRobinPipe

FIELDS = ["prompt_id", "seed", "attack_id", "label", "tr_distance", "robin_distance"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tr_distance(x_T: torch.Tensor, gt_patch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """We reproduce the official eval_watermark (l1_complex), per batch item."""
    fft = torch.fft.fftshift(torch.fft.fft2(x_T.to(torch.float32)), dim=(-1, -2))
    m = mask[0]
    return torch.abs(fft[:, m] - gt_patch[0][m].unsqueeze(0)).mean(dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--clean_dir", type=Path, default=None)
    ap.add_argument("--tr_dir", type=Path, default=None)
    ap.add_argument("--key_path", type=Path, default=Path("outputs_p3/treering/treering_key.pt"))
    ap.add_argument("--manifest", type=Path, default=Path("outputs_p3/manifests/attack_manifest.json"))
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--wm_ckpt", default=str(_HERE.parents[1] / "robin_official" / "ckpts" / "optimized_r5_15_step10.pt"))
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--model_id", default="sd2-community/stable-diffusion-2-1-base")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not visible to torch"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    s = args.gen_seed
    clean_dir = args.clean_dir or Path(f"outputs_p3/clean/seed_{s}")
    tr_dir = args.tr_dir or Path(f"outputs_p3/treering/watermarked/seed_{s}")
    out_csv = args.out_csv or Path(f"outputs_p3/treering/scores/seed_{s}/treering_scores.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    attacks = json.loads(args.manifest.read_text(encoding="utf-8"))["attacks"]
    pipe = CudaRobinPipe(args.model_id, device=device, revision=None, variant=None)
    null_emb = pipe.get_text_embedding("")

    key = torch.load(args.key_path, map_location=device)
    gt_patch, tr_mask = key["gt_patch"].to(device), key["mask"].to(device)
    opt_wm = torch.load(args.wm_ckpt, map_location="cpu")["opt_wm"].to(torch.complex64).to(device)
    robin_mask = get_watermarking_mask(pipe.get_random_latents(), WmArgs(), device)

    done = set()
    if out_csv.exists():
        with out_csv.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["prompt_id"])
    new_file = not out_csv.exists()
    fh = out_csv.open("a", encoding="utf-8", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        wr.writeheader()

    log(f"seed={s} range=[{args.start},{args.end}) attacks={len(attacks)} "
        f"TR bins={int(tr_mask.sum())} resume_done={len(done)} -> {out_csv}")
    todo = [i for i in range(args.start, args.end) if f"{i:06d}" not in done]
    total, n_done, t0 = len(todo), 0, time.time()
    for i in todo:
        pid = f"{i:06d}"
        c_path = clean_dir / f"ori-lg7.5-{i}.jpg"
        t_path = tr_dir / f"trwm-lg7.5-{i}.jpg"
        if not (c_path.exists() and t_path.exists()):
            log(f"skip idx {i}: missing image(s)")
            continue
        rng_seed = s * 1_000_000 + i
        base = Image.open(c_path).convert("RGB")
        imgs = [apply_attack(base.copy(), atk["stages"], rng_seed) for atk in attacks]
        keys = [(atk["id"], 0) for atk in attacks]
        imgs.append(Image.open(t_path).convert("RGB"))
        keys.append(("none", 1))

        t = torch.cat([transform_img(im).unsqueeze(0) for im in imgs], dim=0).to(torch.float32)
        lat = pipe.get_image_latents(t, sample=False)
        buf = [lat]
        x_T, buf, _ = pipe.forward_diffusion(
            latents=lat, text_embeddings=null_emb.expand(len(imgs), -1, -1), guidance_scale=1.0,
            num_inference_steps=args.num_inference_steps, latents_b=buf)
        d_tr = tr_distance(x_T, gt_patch, tr_mask).tolist()
        d_rb = batched_ring_distance(buf[args.steps], opt_wm, robin_mask).tolist()
        for (aid, lab), a, b in zip(keys, d_tr, d_rb):
            wr.writerow({"prompt_id": pid, "seed": s, "attack_id": aid, "label": lab,
                         "tr_distance": a, "robin_distance": b})
        fh.flush()

        n_done += 1
        el = time.time() - t0
        if n_done % 10 == 0 or n_done == total:
            rate = n_done / el
            log(f"seed={s} {n_done}/{total} ({100 * n_done / total:5.1f}%) "
                f"{el / n_done:.2f} s/prompt ({el / n_done / len(imgs):.3f} s/item) "
                f"eta={(total - n_done) / rate / 60:.1f}m "
                f"vram={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB")
    fh.close()
    log(f"done seed={s} -> {out_csv}")


if __name__ == "__main__":
    main()
