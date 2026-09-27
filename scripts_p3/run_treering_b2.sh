#!/bin/bash
# scripts_p3/run_treering_b2.sh  — Tree-Ring second detector, plan B2 (seed 0)
# Run from /workspace/PAPER3 after scripts_p3/runpod_setup.sh.
#   bash scripts_p3/run_treering_b2.sh pilot   # 20 prompts: generate, score, check
#   bash scripts_p3/run_treering_b2.sh full    # 1000 prompts (resumes; pilot rows kept)
set -euo pipefail
STAGE=${1:?usage: run_treering_b2.sh pilot|full}
mkdir -p outputs_p3/treering/logs
pip install --no-cache-dir -q pandas

if [ "$STAGE" = "pilot" ]; then
  python scripts_p3/gen_treering_cuda.py --gen_seed 0 --start 0 --end 20 --batch_size 16 \
    2>&1 | tee outputs_p3/treering/logs/gen_pilot.log
  python scripts_p3/score_treering_cuda.py --gen_seed 0 --start 0 --end 20 \
    2>&1 | tee outputs_p3/treering/logs/score_pilot.log
  python scripts_p3/22_treering_pilot_check.py \
    --csv outputs_p3/treering/scores/seed_0/treering_scores.csv \
    --raw outputs_p3/scores_raw_seed0.csv \
    --out outputs_p3/treering/pilot_check.txt
elif [ "$STAGE" = "full" ]; then
  python scripts_p3/gen_treering_cuda.py --gen_seed 0 --start 0 --end 1000 --batch_size 16 --skip_existing \
    2>&1 | tee outputs_p3/treering/logs/gen_full.log
  python scripts_p3/score_treering_cuda.py --gen_seed 0 --start 0 --end 1000 \
    2>&1 | tee outputs_p3/treering/logs/score_full.log
  python scripts_p3/22_treering_pilot_check.py \
    --csv outputs_p3/treering/scores/seed_0/treering_scores.csv \
    --raw outputs_p3/scores_raw_seed0.csv \
    --out outputs_p3/treering/full_check.txt
  tar -cf treering_results.tar outputs_p3/treering/scores outputs_p3/treering/logs \
    outputs_p3/treering/treering_key.pt outputs_p3/treering/*.txt
  echo "[b2] results -> /workspace/PAPER3/treering_results.tar  (runpodctl send treering_results.tar)"
else
  echo "unknown stage: $STAGE"; exit 1
fi
