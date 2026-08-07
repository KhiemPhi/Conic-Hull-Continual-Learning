#!/usr/bin/env bash
# Put the exp49 grid into the 10-STAGE regime.
#
# exp49's 25 cells all read exp16's A_plus cache: LoRA adapted on task 0, then FROZEN.
# exp48's `ce` arm is the same read-out (oPCA g=0.5, k5m8, self-consistent negatives) on
# features trained through all T stages, so running ARMS=ce per (dataset, seed) produces
# the 10-stage twin of every exp49 cell. `drift` (stale rays, current queries) reuses the
# birth-stage train features already in the same cache, so it costs evaluation only.
# NOT `online` -- it picks each test row's extractor by that row's true label. See the
# exp48 docstring.
#
# SEQUENTIAL, AND THAT IS MEASURED, NOT ASSUMED
#     The loop is CPU-bound: one cell runs at ~1546% CPU across 27 threads with the GPU at
#     0% util and 10.5/80 GB. The idle GPU invites running the four datasets concurrently.
#     That was tried and it is strictly worse: one process already occupies 15.5 of 22
#     cores and background daemons hold ~8, so there is no spare capacity to parallelise
#     into. Four streams at OMP_NUM_THREADS=5 drove load average to 50, doubled
#     IMAGENETA's per-task time from 206s to 414s, and produced no completed task in nine
#     minutes. Throughput here is decode-bound on a saturated box; concurrency only adds
#     scheduling loss. Leave this sequential.
#
# MEASURED PER-CELL COST (uncontended, `ce` arm, T=10)
#     CUB200      1389s   = 415s task 0 (40 ep) + ~107s x 9 (15 ep)
#     IMAGENETA   2346s   = 492s + ~206s x 9
#     IMAGENETR   ~4x CUB       (24k train rows -> 25 batches/epoch vs 6)
#     CIFAR100    ~8x CUB       (50k train rows -> 52 batches/epoch vs 6)
#     plus ~15 min/cell for the two read-out protocols.
set -u
cd /home/khiemphi/Conic-Hull-Continual-Learning
source ~/venvs/ml_env/bin/activate

# IMAGENETA s0 was launched separately and is nearly done; gate on its feature cache so a
# rerun of this script does not duplicate it.
INA0=exp48_feats_IMAGENETA_T10_s0_ce_e40-15_lr0.0003_lam1_P8K12s8R4_t0.1_w20_augreg_in21k.npz
while [ ! -f "$INA0" ]; do sleep 60; done

run () {                                     # $1 dataset, $2 seed
  local LOG="logs/exp48_grid_${1}_s${2}.log"
  echo "=== $(date +%H:%M:%S) START $1 s$2"
  DS=$1 T=10 SEED=$2 ARMS=ce PROTOCOL=final,drift \
    python -u exp48_conic_feature_loss.py > "$LOG" 2>&1
  # capture BEFORE the echo: $(date) inside the string runs first and resets $?, which is
  # why three crashed IMAGENETA cells were logged as "exit 0".
  local rc=$?
  echo "=== $(date +%H:%M:%S) END   $1 s$2 exit $rc"
  [ $rc -ne 0 ] && echo "!!! FAILED $1 s$2 -- see $LOG"
}

run IMAGENETA 0                              # eval only; training cache already written
for S in 1 2;   do run IMAGENETA $S; done
for S in 0 1 2; do run CUB200    $S; done    # CUB s0 `ce` features are already cached
for S in 0 1 2; do run IMAGENETR $S; done
for S in 0 1 2; do run CIFAR100  $S; done
echo "=== ALL DONE $(date +%H:%M:%S)"
