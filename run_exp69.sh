#!/bin/bash
# exp69 pilot. Retries because this box intermittently drops shared libraries off its
# filesystem -- it has already killed runs with a libtorch_cuda_linalg.so dlopen error and a
# libcudnn_graph.so load failure, both of which were fine seconds later. exp69 skips existing
# result keys, so a retry only redoes the arm that was interrupted.
set -x
source ~/venvs/ml_env/bin/activate
cd /home/khiemphi/Conic-Hull-Continual-Learning
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot
export http_proxy=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 HF_HUB_DISABLE_XET=1
# cheap arms first so a harness bug surfaces before hours of training are spent
for A in frozen,fs cont0 cont50 cont50_accum; do
  for i in 1 2 3 4; do
    echo "[exp69] arms=$A attempt $i/4"
    DS=IMAGENETR T=10 SEED=0 ARMS=$A python -u exp69_gram_ranpac.py && { echo "[exp69] $A OK"; break; }
    echo "[exp69] $A failed attempt $i; sleep 90"; sleep 90
  done
done
echo EXP69_PILOT_DONE
