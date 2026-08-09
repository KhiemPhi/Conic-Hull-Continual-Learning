#!/usr/bin/env bash
# run_exp48_v12.sh -- the regime map: ce_conic (v1) vs ce_conic2 (v2), 3 seeds, 4 datasets.
#
# WHAT THE TWO ARMS ARE
#   ce_conic   CE + LAM * episodic conic CE, exactly as it has run to date.
#   ce_conic2  the same objective with three defects of the v1 EPISODE repaired -- one-draw
#              support/query split (v1 leaked ~67% of query rows back into their own
#              support), b_opca-matched generators instead of arbitrary interleaved group
#              means, and a cumulative rather than EMA whitener. Batch geometry, ray count
#              and step count are unchanged, so v2 - v1 attributes to the episode and not
#              to a change of budget. See the ce_conic2 block in the exp48 docstring.
#   v1 is bit-reproducible against the committed version (verified on logits, query mask
#   and gmap for both `cone` and `sub`), so every cached v1 feature file stays valid and
#   CUB200 s0 ce_conic|final is skipped from the JSON rather than retrained.
#
# SEQUENTIAL, AND THAT IS MEASURED, NOT ASSUMED
#   One cell already runs at ~1546% CPU across 27 threads with the GPU near idle. Four
#   concurrent streams drove load average to 50 and DOUBLED IMAGENETA's per-task time from
#   206s to 414s with no cell completing in nine minutes. Throughput is decode-bound on a
#   saturated box. Leave this sequential. See run_exp48_grid.sh for the full measurement.
#
# ORDER
#   Datasets cheapest-and-most-informative first. Seed-outer / arm-inner, so each v1/v2
#   PAIR completes together and a partial run is always balanced rather than all-v1.
#
# MEASURED PER-CELL COST (this box, uncontended, T=10, train + final + drift read-out)
#   CUB200      1.27h v1 / 1.34h v2     (train 3692s measured, read-out 873s measured)
#   IMAGENETR   4.96h / 5.25h           (train 16682s measured, read-out 1156s measured)
#   IMAGENETA   2.11h / 2.22h           (ratio-scaled from CUB; read-out 1350s measured)
#   CIFAR100    ~8.7h / ~9.2h           (RATIO-SCALED, never trained -- treat as a guess)
#   v2 costs ~7% more than v1 (smoke: 290.4s vs 271.6s train, CUB200 T=2 e1-1).
#   Cumulative: CUB ~6.8h | +IN-R ~37h | +IN-A ~50h | +CIFAR ~104h (~4.3 days).
#   Do NOT trust the estimates in run_exp48_grid.sh; they ran 2.7-3.0x optimistic on every
#   cell that has since been measured.
#
# KILL SWITCHES
#   ~6.8h   CUB200 complete. `ce` on CUB200 has seed sd 0.18, so a paired v2-v1 under
#           ~0.3 across three seeds means the episode fixes did not move it and the 30h of
#           ImageNet-R behind it is a hard sell.
#   ~50h    CIFAR100 starts. It is half the total budget and exp49 measured the cone at a
#           dead tie there (92.62 vs 92.55) with no headroom. Ctrl-C is safe.
#   Results are written to the JSON after EVERY cell, so killing this loses at most the
#   cell in flight.
set -u
cd /home/khiemphi/Conic-Hull-Continual-Learning
source ~/venvs/ml_env/bin/activate

run () {                                     # $1 dataset  $2 seed  $3 arm
  local LOG="logs/exp48_v12_${1}_s${2}_${3}.log"
  echo "=== $(date +%F\ %H:%M:%S) START $1 s$2 $3"
  DS=$1 T=10 SEED=$2 ARMS=$3 PROTOCOL=final,drift \
    python -u exp48_conic_feature_loss.py > "$LOG" 2>&1
  # Capture BEFORE the echo: a $(date) inside the string runs first and resets $?, which is
  # how three crashed IMAGENETA cells once got logged as "exit 0".
  local rc=$?
  echo "=== $(date +%F\ %H:%M:%S) END   $1 s$2 $3 exit $rc"
  [ $rc -ne 0 ] && echo "!!! FAILED $1 s$2 $3 -- see $LOG"
  return 0                                   # never let one bad cell kill the sweep
}

for DS in CUB200 IMAGENETR IMAGENETA CIFAR100; do
  for S in 0 1 2; do
    for A in ce_conic ce_conic2; do run "$DS" "$S" "$A"; done
  done
done
echo "=== ALL DONE $(date +%F\ %H:%M:%S)"
