#!/usr/bin/env bash
set -u
cd /home/khiemphi/Conic-Hull-Continual-Learning
source ~/venvs/ml_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
DS=CUB200 T=10 SEED=0 ARMS=base,d1,d2,d4,d8,l5,l10,l25,z \
  python -u exp50_crowding.py > logs/exp50_sweep.log 2>&1
echo "SWEEP DONE $(date +%H:%M)"
