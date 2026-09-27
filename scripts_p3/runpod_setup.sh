#!/bin/bash
# scripts_p3/runpod_setup.sh
#
# We run this ONCE on a fresh RunPod pod (after copying the repo over) to prepare
# our scoring on an external CUDA GPU (see 03_score_attacks_cuda.py).
#
# Pod selection (we do this in the RunPod web UI before connecting):
#   - Template: any "RunPod PyTorch" base image with CUDA 12.1 (e.g.
#     "RunPod Pytorch 2.1" / "cuda12.1"-tagged template). We do NOT pick a
#     CUDA<12.0 template; we install our own torch/diffusers here on
#     top regardless, but the pod's driver must support >=12.1.
#   - GPU: RTX 4090 (widely available on RunPod, well-supported). RTX 5090
#     may not yet be listed as a RunPod SKU — we check availability first.
#   - Disk: >= 20GB container disk (10,000 source images ~1-2GB + SD 2.1-base
#     weights ~5GB + venv ~6GB).
#
# Data transfer (repo -> pod, and results pod -> repo) — we use runpodctl:
#   Local:  runpodctl send outputs_p3/clean outputs_p3/watermarked \
#               outputs_p3/manifests robin_official scripts_p3
#           (prints a one-time receive code)
#   Pod:    runpodctl receive <code>
#   ... after our scoring finishes on the pod ...
#   Pod:    runpodctl send outputs_p3/scores
#   Local:  runpodctl receive <code>
set -euo pipefail

pip install --no-cache-dir \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir \
    diffusers==0.27.2 transformers==4.40.0 accelerate==0.30.1 \
    pytorch-msssim==1.0.0 scikit-learn scipy datasets Pillow

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('CUDA OK:', torch.cuda.get_device_name(0), torch.version.cuda)"

echo "[runpod_setup] done. Smoke test:"
echo "  python scripts_p3/03_score_attacks_cuda.py --gen_seed 0 --start 0 --end 1"
