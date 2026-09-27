"""
scripts_p3/25_make_flow_figure.py  (local)

We draw our pipeline figure with real SD2.1 artefacts: one test-split prompt (the first one, seed 0),
its non-watermarked and ROBIN images, the ROBIN Fourier key, four evaluation conditions
applied with the same attack code as our scoring run, the stored detection scores of
exactly these images, the calibration threshold, and the resulting FPR/TPR map.

We write outputs_p3/figures/fig0_flow.{pdf,png} and print every number drawn.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageFilter
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "archives_extracted/sd21_images/outputs_p3"
R = ROOT / "sd21_results/outputs_p3"
CKPT = ROOT.parent / "robin_official/ckpts/optimized_r5_15_step10.pt"
OUT = ROOT / "outputs_p3/figures/fig0_flow"
SEED, ALPHA = 0, 0.05
SHOW = ["none", "jpeg_q50", "noise_s005", "rotation"]


def set_random_seed(seed: int) -> None:
    """We reproduce robin_official/optim_utils.set_random_seed on CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed + 3)
    random.seed(seed + 5)


def apply_stage(img: Image.Image, st: dict, rng_seed: int) -> Image.Image:
    """We copy _attack_scoring_common.apply_stage here."""
    op = st["op"]
    if op == "jpeg":
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=int(st["quality"])); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "rotation":
        return transforms.RandomRotation((st["degrees"], st["degrees"]))(img)
    if op == "cropping":
        set_random_seed(rng_seed); r = st["ratio"]
        return transforms.RandomResizedCrop(img.size, scale=(r, r), ratio=(r, r))(img)
    if op == "blurring":
        return img.filter(ImageFilter.GaussianBlur(radius=st["sigma"]))
    if op == "noise":
        arr = np.array(img).astype(np.float32); set_random_seed(rng_seed)
        noise = np.random.normal(0, st["sigma"], arr.shape) * 255
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    raise ValueError(op)


def attack(img, stages, rng_seed):
    for st in stages:
        img = apply_stage(img, st, rng_seed)
    return img


def ring_mask(size=64, r_max=15, r_min=5):
    """We reproduce robin_official circle_mask (annulus r_min < d <= r_max)."""
    x0 = y0 = size // 2
    y, x = np.ogrid[:size, :size]; y = y[::-1]
    d2 = (x - x0) ** 2 + (y - y0) ** 2
    return (d2 <= r_max ** 2) & (d2 > r_min ** 2)


def arrow(fig, a, b, text=None):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=12, lw=1.2, color="0.25"))
    if text:
        fig.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.008, text, ha="center", va="bottom",
                 fontsize=6.5, color="0.25")


def main() -> None:
    manifest = {a["id"]: a for a in json.loads((R / "manifests/attack_manifest.json").read_text())["attacks"]}
    test_ids = sorted(pd.read_csv(R / "splits/test_clean.csv").query("seed == @SEED").prompt_id.unique())
    pid = int(test_ids[0])
    prompt = pd.read_csv(R / "prompts/prompt_list.csv", dtype={"prompt_id": str}).set_index("prompt_id").loc[f"{pid:06d}", "prompt"]
    raw = pd.read_csv(R / "scores/scores_raw.csv", dtype={"prompt_id": str}).query("seed == @SEED")
    th = pd.read_csv(R / "thresholds/thresholds_by_method.csv")
    tau = float(th.query("gen_seed == @SEED and method == 'M3' and alpha == @ALPHA").threshold.iloc[0])
    e2 = pd.read_csv(R / "metrics/e2_fpr_drift.csv")
    e2 = e2[(e2.method == "M3") & (e2.alpha == ALPHA) & e2.gen_seed.astype(str).isin([str(s) for s in range(5)])
            & e2.attack_id.isin(list(manifest))]
    fpr_mean = e2.groupby("attack_id").empirical_fpr.mean()
    cal = pd.read_csv(R / "splits/cal_clean.csv").query("seed == @SEED and attack_id == 'none'").score_z.to_numpy()
    tst = pd.read_csv(R / "splits/test_clean.csv").query("seed == @SEED")
    twm = pd.read_csv(R / "splits/test_watermarked.csv").query("seed == @SEED")

    x_c = Image.open(IMG / f"clean/seed_{SEED}/ori-lg7.5-{pid}.jpg").convert("RGB")
    x_w = Image.open(IMG / f"watermarked/seed_{SEED}/wm-lg7.5-{pid}.jpg").convert("RGB")
    key = torch.load(CKPT, map_location="cpu")["opt_wm"][0, 3]
    key_img = np.where(ring_mask(), np.log1p(np.abs(key.numpy())), np.nan)
    rng_seed = SEED * 1_000_000 + pid

    def z(aid, lab):
        return float(raw.query("prompt_id == @f and attack_id == @aid and label == @lab",
                               local_dict={"f": f"{pid:06d}", "aid": aid, "lab": lab}).score_z.iloc[0])

    print(f"prompt {pid}: {prompt}\ntau(M3, alpha={ALPHA}, seed {SEED}) = {tau:.3f}")
    fig = plt.figure(figsize=(7.2, 8.6))
    fs = 7

    # A. generation
    fig.text(0.01, 0.975, "A  Generation (same prompt, same initial latent, seed 0)", fontsize=8, weight="bold")
    axp = fig.add_axes([0.01, 0.745, 0.20, 0.20]); axp.axis("off")
    axp.text(0.5, 0.5, f"prompt {pid}\n\n\u201c{prompt}\u201d", ha="center", va="center", fontsize=6.5, wrap=True,
             bbox=dict(boxstyle="round", fc="0.95", ec="0.6"))
    for k, (im, t) in enumerate([(x_c, "non-watermarked $x_0$\n(SD 2.1-base, 50 steps)"),
                                 (x_w, "ROBIN watermarked $x_0^*$\n(key injected at step 35)")]):
        ax = fig.add_axes([0.26 + 0.215 * k, 0.745, 0.20, 0.20]); ax.imshow(im); ax.axis("off")
        ax.set_title(t, fontsize=fs)
    axk = fig.add_axes([0.74, 0.745, 0.20, 0.20])
    axk.imshow(key_img, cmap="viridis"); axk.set_xticks([]); axk.set_yticks([])
    axk.set_title("ROBIN key: $\\log(1+|w|)$,\nFourier ring $5<r\\leq15$, ch. 3", fontsize=fs)
    arrow(fig, (0.215, 0.845), (0.255, 0.845))

    # B. post-processing + inversion score
    fig.text(0.01, 0.705, "B  Post-processing, DDIM inversion to step 35, score $z=-d$ (flagged if $z>\\tau$)",
             fontsize=8, weight="bold")
    for j, aid in enumerate(SHOW):
        st = manifest[aid]["stages"]
        for r_, (base, lab, name) in enumerate([(x_c, 0, "non-wm"), (x_w, 1, "ROBIN")]):
            im = attack(base.copy(), st, rng_seed)
            ax = fig.add_axes([0.03 + 0.245 * j, 0.53 - 0.155 * r_, 0.14, 0.14]); ax.imshow(im); ax.axis("off")
            zz = z(aid, lab); flag = zz > tau
            print(f"  {aid:10s} {name:6s} z={zz:.2f} flagged={flag}")
            ax.text(1.04, 0.5, f"{name}\nz={zz:.1f}\n{'flagged' if flag else 'not flagged'}",
                    transform=ax.transAxes, fontsize=6, va="center",
                    color=("C3" if flag != bool(lab) else "0.1"))
            if r_ == 0:
                ax.set_title(aid, fontsize=fs)
    arrow(fig, (0.5, 0.36), (0.5, 0.325))
    fig.text(0.52, 0.338, "same threshold for every condition", fontsize=6.5, color="0.25", va="center")

    # C. calibration and evaluation
    fig.text(0.01, 0.315, "C  Calibration on non-watermarked none (n=500) and evaluation per condition (m=500)",
             fontsize=8, weight="bold")
    ax1 = fig.add_axes([0.07, 0.05, 0.25, 0.22])
    ax1.hist(cal, bins=30, color="0.6"); ax1.axvline(tau, color="C3", ls="--", lw=1)
    ax1.set_title(f"calibration scores, $\\tau_{{0.05}}$={tau:.2f}", fontsize=fs)
    ax1.set_xlabel("score z", fontsize=fs); ax1.tick_params(labelsize=6)
    ax2 = fig.add_axes([0.40, 0.05, 0.25, 0.22])
    rows = []
    for aid, c in [("none", "0.4"), ("noise_s005", "C1")]:
        s = tst[tst.attack_id == aid].score_z.to_numpy(); f = float(np.mean(s > tau)); rows.append((aid, f))
        ax2.hist(s, bins=30, histtype="step", color=c, label=f"non-wm {aid}: FPR {f:.3f}")
    w = twm[twm.attack_id == "none"].score_z.to_numpy(); tpr = float(np.mean(w > tau))
    ax2.hist(w, bins=30, histtype="step", color="C0", label=f"ROBIN none: TPR {tpr:.3f}")
    ax2.axvline(tau, color="C3", ls="--", lw=1); ax2.legend(fontsize=5.5, loc="upper left")
    ax2.set_title("test scores (seed 0)", fontsize=fs); ax2.set_xlabel("score z", fontsize=fs)
    ax2.tick_params(labelsize=6)
    print(f"  seed 0 test: FPR none={rows[0][1]:.3f} FPR noise_s005={rows[1][1]:.3f} TPR none={tpr:.3f}")
    ax3 = fig.add_axes([0.80, 0.05, 0.19, 0.22])
    fm = fpr_mean.sort_values()
    ax3.barh(range(len(fm)), fm.values, color=["C3" if v > ALPHA else "0.6" for v in fm.values])
    ax3.axvline(ALPHA, color="C3", ls="--", lw=1)
    ax3.set_yticks(range(len(fm))); ax3.set_yticklabels(fm.index, fontsize=4.8)
    ax3.set_title("operating-range map: FPR\nper condition (5-seed mean)", fontsize=fs)
    ax3.set_xlabel("empirical FPR", fontsize=fs); ax3.tick_params(axis="x", labelsize=6)
    print("  5-seed mean FPR > alpha:", {k: round(v, 4) for k, v in fm.items() if v > ALPHA})
    arrow(fig, (0.33, 0.16), (0.385, 0.16), "$\\tau$")
    arrow(fig, (0.66, 0.16), (0.69, 0.16))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"-> {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
