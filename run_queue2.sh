#!/bin/bash
# Queue: C1 @ T=20,50  ->  C6 @ T=20,50  ->  ce_conic on shifted data.
# Sequential by design: exp49 measured concurrency on this box as strictly worse, and these are
# all GPU training jobs. Retry rather than set -e -- this box intermittently drops its
# filesystem (it has killed a run with a libtorch dlopen error). Every step is resumable:
# exp55 skips existing feature caches, exp66/exp67/exp48 skip existing result keys.
set -x
source ~/venvs/ml_env/bin/activate
cd /home/khiemphi/Conic-Hull-Continual-Learning
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot
# fwdproxy is required for HF, and the Xet backend does NOT work through it.
export http_proxy=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 HF_HUB_DISABLE_XET=1
M=q32,q32v1,q32v2,q32v3,q32v4

retry() { local n=$1 label=$2; shift 2
  for i in $(seq 1 "$n"); do
    echo "[queue] $label attempt $i/$n"
    "$@" && { echo "[queue] $label OK"; return 0; }
    echo "[queue] $label FAILED attempt $i; sleep 120"; sleep 120
  done
  echo "[queue] $label GAVE UP"; return 1; }

# don't contend with the cone-only sweep still filling in exp61
while pgrep -f "exp61_cone_only" > /dev/null 2>&1; do sleep 60; done
echo "[queue] cone_only finished; starting"

# ---- C1 at T=20 and T=50: does the diversity design matter at longer sequences?
for T in 20 50; do
  retry 3 "c1_train_T$T"  env DS=IMAGENETR T=$T SEED=0,1,2 MEMBERS=$M python -u exp55_lora_diversity_pilot.py
  retry 2 "c1_score_T$T"  env DS=IMAGENETR T=$T SEED=0,1,2 MEMBERS=$M SUFFIX=_c1 VERIFY=0 python -u exp66_controls.py
done

# ---- C6 at T=20 and T=50: does the ViT-L dominance hold as the sequence lengthens?
for T in 20 50; do
  retry 3 "c6_T$T" env MODE=scale MODEL=vit_large_patch16_224.augreg_in21k DS=IMAGENETR T=$T SEED=0,1,2 \
    python -u exp67_cost_and_scale.py
done

# ---- ce_conic: a conic AUXILIARY on top of CE, on SHIFTED data for the first time.
# exp48 only ever ran on CUB200, which C2 shows is cone-neutral. Success metric is the RANPAC
# read-out (absolute accuracy), NOT cone-minus-sub -- if the features genuinely improve, the
# cone's read-out edge should shrink.
for s in 0 1 2; do
  retry 2 "ceconic_s$s" env DS=IMAGENETR T=10 SEED=$s ARMS=ce,ce_conic python -u exp48_conic_feature_loss.py
done
echo QUEUE2_DONE
