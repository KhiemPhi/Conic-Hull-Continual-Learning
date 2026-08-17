#!/bin/bash
# Cone-only across the full grid, R=64 fixed (the same arm as the headline f64 column, so this
# is free of per-cell selection). NOTE R=64 CLIPS on CUB200P (median 43 rows/class) and
# IMAGENETAP (median 19), and the sweeps showed IMAGENETAP prefers R=8 by ~1.8 A-Last, so its
# cone-only row here is a LOWER bound. Retry, not set -e: this box drops its filesystem.
set -x
source ~/venvs/ml_env/bin/activate
cd /home/khiemphi/Conic-Hull-Continual-Learning
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot
export http_proxy=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 HF_HUB_DISABLE_XET=1
for T in 10 20 50; do
  for i in 1 2 3; do
    echo "[chain] cone_only T=$T attempt $i"
    DS=CIFAR100,IMAGENETR,CUB200P,IMAGENETAP T=$T SEED=0,1,2 R=64 \
      MEMBERS=q32,m32,a16,q32b70,q64 VERIFY=1 python -u exp61_cone_only.py && break
    echo "[chain] T=$T failed attempt $i; sleep 120"; sleep 120
  done
done
echo CONE_ONLY_DONE
