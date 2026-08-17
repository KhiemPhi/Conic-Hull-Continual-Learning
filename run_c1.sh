#!/bin/bash
# C1 -- the seed-only diversity control. Members differ ONLY by LoRA init (q32v1..v4 are
# architecturally identical to q32), so this isolates whether the paper's gain comes from the
# STRUCTURAL diversity design (mlp/all targets, class bagging, rank 64) or from plain
# ensembling. Run on IMAGENETR because that is where the method's claim actually lives.
set -x
# NOT set -e. This box has an INTERMITTENT filesystem: on 2026-08-15 it transiently lost
# ~/.cache/huggingface, the whole repo view, and libtorch_cuda_linalg.so (which killed the
# first C1 attempt mid-run with a dlopen error), then all three came back. Every step below is
# RESUMABLE -- exp55 skips existing feature caches, exp66 skips existing cells -- so retrying
# is cheap and is the only thing that makes a multi-hour run survive this box.
retry() {  # retry <n> <label> <cmd...>
  local n=$1 label=$2; shift 2
  for i in $(seq 1 "$n"); do
    echo "[chain] $label attempt $i/$n"
    "$@" && { echo "[chain] $label OK"; return 0; }
    echo "[chain] $label FAILED on attempt $i; sleeping 120s"; sleep 120
  done
  echo "[chain] $label GAVE UP after $n attempts"; return 1
}
source ~/venvs/ml_env/bin/activate
cd /home/khiemphi/Conic-Hull-Continual-Learning
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot
# HF hub access, two devserver-specific facts:
#   1. external fetches need fwdproxy;
#   2. huggingface_hub's default Xet backend does NOT work through it -- it fails in the CAS
#      client AFTER the proxy is set, which reads like a proxy problem and is not one.
# NOT offline: ~/.cache/huggingface no longer exists on this box, so the ViT weights must be
# re-fetched (the DATASETS are safe in the repo's data/hf). With HF_HUB_OFFLINE=1 this dies with
# LocalEntryNotFoundError; without the proxy it dies with "client has been closed".
export http_proxy=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080
export HF_HUB_DISABLE_XET=1
M=q32,q32v1,q32v2,q32v3,q32v4

# 1) train the 4 seed-variant members (q32 itself is exp16's cache; never retrained)
retry 4 train env DS=IMAGENETR T=10 SEED=0,1,2 MEMBERS=$M python -u exp55_lora_diversity_pilot.py || exit 1

# 2) score them through the SAME read-out as the structural ensemble. VERIFY=0 because the
#    exp56 anchor only exists for the structural member set; exp66 logs the skip either way.
retry 3 score env DS=IMAGENETR T=10 SEED=0,1,2 MEMBERS=$M SUFFIX=_c1 VERIFY=0 python -u exp66_controls.py || exit 1
echo C1_DONE
