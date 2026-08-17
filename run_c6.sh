#!/bin/bash
# C6 -- the equal-inference-budget baseline for Paper 1.
# C7 measured 5x ViT-B/16 = 0.2381s vs 1x ViT-L/16 = 0.0724s for one batch, i.e. the 5-member
# ensemble costs 3.3x a single ViT-L at inference. So "why not just use a bigger backbone?" is
# the sharpest cost objection to the paper, and this answers it like-for-like: ONE first-session
# LoRA on ViT-L, same recipe, same PILOT class order, scored through the same RanPAC read-out.
# If this ties or beats the 5-member ensemble (IMAGENETR T=10: ens_ranpac 81.78, FE 82.58),
# the paper's gain is capacity rather than diversity and the framing has to change.
set -x
source ~/venvs/ml_env/bin/activate
cd /home/khiemphi/Conic-Hull-Continual-Learning
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot
# See run_c1.sh: fwdproxy is required and the Xet backend does not work through it.
export http_proxy=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080
export HF_HUB_DISABLE_XET=1

# Wait for C1 to finish -- ViT-L at BS=128 plus C1's ViT-B training would contend for the GPU,
# and exp49 measured concurrency on this box as strictly worse than sequential.
while pgrep -f "run_c1.sh" > /dev/null 2>&1; do sleep 60; done
echo "[chain] C1 finished, starting C6"

# Retry, not set -e: this box intermittently drops its filesystem (it killed C1 once with a
# libtorch dlopen error). Cells are written as they finish and existing keys skip, so retrying
# only redoes the interrupted seed.
for i in 1 2 3 4; do
  echo "[chain] C6 attempt $i/4"
  MODE=scale MODEL=vit_large_patch16_224.augreg_in21k DS=IMAGENETR T=10 SEED=0,1,2 \
    python -u exp67_cost_and_scale.py && { echo "[chain] C6 OK"; break; }
  echo "[chain] C6 failed attempt $i; sleeping 120s"; sleep 120
done
echo C6_DONE
