#!/bin/bash
# scripts_p3/resume_treering_b2.sh — continue the Tree-Ring B2 run on a fresh pod.
# 1) local:  runpodctl send treering_b2_resume.tar      (prints a code)
# 2) pod:    cd /workspace && runpodctl receive <code> && tar -xf treering_b2_resume.tar
# 3) pod:    bash /workspace/PAPER3/scripts_p3/resume_treering_b2.sh
# Generation skips existing images and scoring skips prompts already in treering_scores.csv.
set -euo pipefail
cd /workspace/PAPER3
export HF_HOME=/workspace/hf_cache
grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

echo "[STEP] setup $(date)"
# 7GB root disk cannot hold torch 2.3.1: use a venv on /workspace (see paper3_status memory)
[ -n "${VIRTUAL_ENV:-}" ] || bash scripts_p3/runpod_setup.sh
pip install --no-cache-dir -q "huggingface_hub==0.23.4"
python -c "import torch,diffusers,huggingface_hub as h;print('[STEP] versions torch',torch.__version__,'cuda',torch.cuda.is_available(),'diffusers',diffusers.__version__,'hf_hub',h.__version__)"
mkdir -p logs outputs_p3/treering/logs

D=outputs_p3/treering
CSV=$D/scores/seed_0/treering_scores.csv
echo "[STEP] resume from: TR_images=$(ls $D/watermarked/seed_0 | wc -l)/1000 score_rows=$(($(wc -l < $CSV)-1))/18000 $(date)"
nohup bash scripts_p3/run_treering_b2.sh full > $D/logs/full_all.log 2>&1 &
JOB=$!
echo "[STEP] job PID=$JOB log=$D/logs/full_all.log"

bar(){ local n=$1 t=$2 w=30 f; f=$((n*w/t)); printf '%s%s' "$(printf '%*s' $f '' | tr ' ' '#')" "$(printf '%*s' $((w-f)) '' | tr ' ' '.')"; }
s0=$(($(wc -l < $CSV)-1)); t0=$(date +%s)
while kill -0 $JOB 2>/dev/null; do
  g=$(ls $D/watermarked/seed_0 | wc -l); s=$(($(wc -l < $CSV)-1)); el=$(( $(date +%s)-t0 ))
  r=$(( s>s0 && el>0 ? (s-s0)*1000/el : 0 )); eta=$(( r>0 ? (18000-s)*1000/r/60 : -1 ))
  printf '\r[%s] GEN [%s] %4d/1000 | SCORE [%s] %5d/18000 %3d%% | eta %s min   ' \
    "$(date +%T)" "$(bar $g 1000)" $g "$(bar $s 18000)" $s $((s*100/18000)) "$eta"
  sleep 5
done
echo; echo "[DONE] $(date)"; tail -n 30 $D/logs/full_all.log
