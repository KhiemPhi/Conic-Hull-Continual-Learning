#!/usr/bin/env bash
# Resume the 10-STAGE grid after the IMAGENETA eval crash. See run_exp48_grid.sh for what
# the grid is and why it stays sequential.
#
# WHY THIS IS A SECOND FILE, not an edit of the first: bash reads a running script by byte
# offset, so editing the file it is currently executing makes it resume mid-token. The
# first driver was retired and its python child left to finish IMAGENETR s0 on its own.
#
# STATE AT RESUME
#   done      CUB200 s0/s1/s2          (final + drift, in the json)
#   trained   IMAGENETA s0/s1/s2       feature caches written, then evaluate() raised
#                                      KeyError on a class with <2 fit rows -- now fixed,
#                                      so these three are EVAL-ONLY and cheap
#   running   IMAGENETR s0             separate python, writes its own cache and results
#   todo      IMAGENETR s1/s2, CIFAR100 s0/s1/s2
set -u
cd /home/khiemphi/Conic-Hull-Continual-Learning
source ~/venvs/ml_env/bin/activate

# don't start until the orphaned IMAGENETR s0 process is done -- the box has no spare cores
while pgrep -f "[e]xp48_conic_feature_loss.py" >/dev/null; do sleep 60; done

run () {
  local LOG="logs/exp48_grid_${1}_s${2}.log"
  echo "=== $(date +%H:%M:%S) START $1 s$2"
  DS=$1 T=10 SEED=$2 ARMS=ce PROTOCOL=final,drift \
    python -u exp48_conic_feature_loss.py > "$LOG" 2>&1
  local rc=$?                       # BEFORE the echo: $(date) would reset $?
  echo "=== $(date +%H:%M:%S) END   $1 s$2 exit $rc"
  [ $rc -ne 0 ] && echo "!!! FAILED $1 s$2 -- see $LOG"
  return 0
}

for S in 0 1 2; do run IMAGENETA $S; done     # eval only, caches already on disk
for S in 1 2;   do run IMAGENETR $S; done
for S in 0 1 2; do run CIFAR100  $S; done
echo "=== ALL DONE $(date +%H:%M:%S)"
