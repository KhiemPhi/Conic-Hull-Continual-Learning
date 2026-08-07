#!/usr/bin/env bash
# Is the 79.78 / 80.05 disagreement on IMAGENETR|10|0|k5m24 nondeterminism or a bug?
#
# Both numbers came from the SAME nominal cell: 79.78 with ARMS=k5m24 (logs/exp49_verify.log)
# and 80.05 with ARMS=k5m24,k5m8 (logs/exp49_grid.log). Stage 0 agrees exactly (90.50) and
# they separate at s1 (90.36 vs 90.45), so it is not the arm list feeding different
# negatives -- the rng is keyed on the arm NAME (exp49_seed_grid.py:193) and `past` is the
# arm's own rays (:197), both arm-list-independent. The remaining suspect is thread-count
# nondeterminism in eigh / KMeans inside b_opca, which would perturb k-means basins.
#
# Those two runs are the "default threads, disagrees" arm of the test. This is the other
# arm: two reps pinned to ONE thread. Agreement to the digit implicates threading and makes
# every future cell reproducible by pinning. Disagreement means a real bug and every delta
# in exp49 -- and in the 10-stage grid now running -- is unreadable.
#
# One thread on purpose: the exp48 grid owns the rest of the box.
set -u
cd /home/khiemphi/Conic-Hull-Continual-Learning
source ~/venvs/ml_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for R in 1 2; do
  echo "=== $(date +%H:%M:%S) rep $R"
  DS=IMAGENETR T=10 SEED=0 ARMS=k5m24 SUFFIX="_det_rep${R}" \
    python -u exp49_seed_grid.py > "logs/exp49_det_rep${R}.log" 2>&1
  echo "=== $(date +%H:%M:%S) rep $R exit $?"
done
