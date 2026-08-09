#!/usr/bin/env bash
# run_exp48_fsa.sh -- does the conic loss improve the FIRST SESSION, not the sequence?
#
# EPOCHS_T=0 makes the epoch loop `for e in range(0)` a no-op for tasks 1..9, so the LoRA
# weights are frozen after task 0 and the later tasks cost feature extraction only. That is
# the first-session-adaptation regime -- exp16's A_plus, the bar this project keeps losing
# to -- with the conic loss substituted into the one session that trains.
#
# WHY THE `ce` ARM IS NOT OPTIONAL
#   exp49's 1-stage numbers come from the exp16 A_plus cache under a different recipe, so
#   they are NOT a matched control for these. `ce` at e40-0 in this same pipeline is. Three
#   arms or the ce_conic2 number means nothing.
#
# PROTOCOL=final ONLY, DELIBERATELY
#   With the backbone frozen after task 0, ON_tr (birth-stage features) is IDENTICALLY Ftr
#   (final-model features) for every row, so `drift` would recompute `final` and report the
#   same number. That also means the +5.80 drift effect -- the only real ce_conic result to
#   date -- CANNOT appear here. This run measures the other question.
#
# WHAT IT UNIQUELY TESTS: TRANSFER
#   The loss only ever sees task-0 classes (20 of 200 on CUB) but the features must serve
#   all 200. Does conic structure imposed on 20 classes transfer to the 180 unseen ones?
#   Unmeasurable in the 10-stage runs, where the loss keeps seeing new classes. Read
#   erank_t0 vs erank_rest in the output: if rank drops only for task-0 classes, the loss
#   overfitted the first session and cannot help.
#
# HONEST PRIOR: small. v1 gave +0.32 in the `final` bracket over 10 stages and 1050 steps;
# here the loss gets 240 steps over 20 classes, and the stability mechanism it has actually
# demonstrated is definitionally absent. Expect ~0 unless the transfer hypothesis holds.
#
# COST  ~36 min/cell (task 0 874s measured + 9 extraction-only passes ~45s + read-out 875s)
#       9 cells ~= 5.4h.  Seed-outer / arm-inner, so all three arms at seed 0 land in ~1.8h
#       and a partial run is always a complete comparison at some seed.
# KEYS  e40-0, so nothing collides with the e40-15 cells or their feature caches.
set -u
cd /home/khiemphi/Conic-Hull-Continual-Learning
source ~/venvs/ml_env/bin/activate

run () {                                     # $1 seed  $2 arm
  local LOG="logs/exp48_fsa_CUB200_s${1}_${2}.log"
  echo "=== $(date +%F\ %H:%M:%S) START s$1 $2"
  DS=CUB200 T=10 SEED=$1 ARMS=$2 PROTOCOL=final EPOCHS_T0=40 EPOCHS_T=0 \
    python -u exp48_conic_feature_loss.py > "$LOG" 2>&1
  local rc=$?                                # capture BEFORE any $(date): it resets $?
  echo "=== $(date +%F\ %H:%M:%S) END   s$1 $2 exit $rc"
  [ $rc -ne 0 ] && echo "!!! FAILED s$1 $2 -- see $LOG"
  return 0
}

for S in 0 1 2; do
  for A in ce ce_conic ce_conic2; do run "$S" "$A"; done
done
echo "=== ALL DONE $(date +%F\ %H:%M:%S)"
