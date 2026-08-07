#!/usr/bin/env bash
# run_final_table.sh — build the full paper table: 4 datasets x T in {10,20,50} x 3 seeds.
#
# THE R RECIPE IS A ROWS-PER-CENTROID RULE, NOT A FIXED R.
#   The conic rule earns its keep only once centroids become individually noisy -- that is
#   the pathology it repairs. Measured on ImageNet-R (median 100 fit rows/class):
#       R=4  -> 25 rows/centroid -> fused rule gap +0.08
#       R=16 -> 6.3              -> +0.45
#       R=32 -> 3.1              -> +0.40
#       R=64 -> 1.6              -> +0.75   (but 24.5% of classes CLAMP, see below)
#   The gap switches on between 25 and 6 rows/centroid. Target 3-7, subject to R <= min
#   fit rows, because km() silently does k=min(k,len(X)) and a class with 35 rows given
#   64 centroids becomes pure 1-NN.
#
#   dataset     classes  fit/class min/med/max   clamp-free R   CHOSEN R   rows/centroid
#   CIFAR100      100      450 / 450 / 450          128+           64          7.0
#   ImageNet-R    200       35 / 100 / 308           32            32          3.1  <- measured
#   CUB200        200       26 /  27 /  27           16             8          3.4
#   ImageNet-A    200        0 /  18 /  79           <4             4          4.5
#
#   R is T-INDEPENDENT: the replay splits by class, so fit rows per class do not change
#   with the number of tasks. One R per dataset.
#
#   NE=0 everywhere. The R=32 control showed eigen atoms contribute +0.00 to the fused
#   result once R is in the right regime -- they and the conic rule are substitutes, both
#   repairing the same overfitting, and stacking them is antagonistic (pm_eig 79.67 beats
#   cone_eig 77.92 at R=64). Plain k-means centroids is the simplest config that wins.
#
# RESUME: exp35 keys every cell and skips completed ones, so re-running this script after
# an interruption costs only the unfinished cells. Safe to Ctrl-C.
#
# RUNTIME: ~11 h total on one GPU. Phases are ordered by information-per-minute, so the
# decisive numbers land first and you can stop after any phase.
set -u
cd "$(dirname "$0")"
mkdir -p logs

EXP=exp35_fused_eigencone.py
SEEDS=0,1,2
TS=10,20,50

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: activate the venv first --  source ~/venvs/ml_env/bin/activate" >&2
  exit 1
fi

run () {  # run <dataset> <R> <phase-label>
  local ds=$1 r=$2 label=$3
  local log="logs/final_${ds}_R${r}.log"
  echo ""
  echo "=============================================================="
  echo " ${label}   DS=${ds}  R=${r}  T=${TS}  SEED=${SEEDS}"
  echo " log -> ${log}"
  echo " started $(date '+%H:%M:%S')"
  echo "=============================================================="
  DS="$ds" T="$TS" SEED="$SEEDS" R="$r" NK="$r" NE=0 ALPHAS=0.5 \
    python -u "$EXP" 2>&1 | tee "$log"
  echo " finished $(date '+%H:%M:%S')"
}

# Phase 1 — the anchor. 81.02 is currently ONE SEED on ONE CELL; every claim in the
# paper rests on it. If this regresses toward 80.6, stop and rethink before spending
# the other 9 hours.                                                          (~4.5 h)
run IMAGENETR 32 "PHASE 1/4  anchor + the cell we lose hardest"

# Phase 2 — cheapest transfer test. 5994 train rows and R=8 make this fast, and CUB200
# at 27 rows/class sits deepest in the noisy-centroid regime, so it is where the
# mechanism should show most clearly.                                         (~0.8 h)
run CUB200 8 "PHASE 2/4  transfer, smallest rows/class"

# Phase 3 — does the mechanism engage when rows are ABUNDANT? 450 rows/class means the
# centroids are well estimated; at R=32 that is 14 rows/centroid, the R=4-equivalent
# regime where the gap was only +0.08. R=64 buys 7.0. This is the prediction most
# likely to fail.                                                             (~5.3 h)
run CIFAR100 64 "PHASE 3/4  abundant rows -- does the mechanism engage?"

# Phase 4 — degenerate by construction: 1 class has 0 fit rows, 6 have <4, 85 of 200
# have <16. Needs the miss-class fusion fallback added to exp35 on 2026-08-05; without
# it the fused score suppresses every class the cone could not model.         (~0.4 h)
run IMAGENETA 4 "PHASE 4/4  degenerate rows -- expect the largest error pool"

# ---------------------------------------------------------------- final paper table
python - <<'PY'
import json, os, re
import numpy as np

OUT = "exp35_fused_eigencone_augreg_in21k.json"
if not os.path.exists(OUT):
    raise SystemExit("no results json yet")
res = json.load(open(OUT))

# external bars, A-Last / A-Avg, as reported in exp16_full_table
BARS = {
    ("CIFAR100", 10):  (91.97, 94.65, 91.86, 94.44),
    ("CIFAR100", 20):  (91.46, 94.41, 90.31, 93.47),
    ("CIFAR100", 50):  (90.03, 93.38, 85.09, 90.89),
    ("IMAGENETR", 10): (82.09, 86.20, 81.82, 85.76),
    ("IMAGENETR", 20): (80.23, 85.05, 79.46, 84.25),
    ("IMAGENETR", 50): (76.74, 82.64, 70.10, 77.47),
    ("CUB200", 10):    (89.91, 93.85, 90.23, 93.78),
    ("CUB200", 20):    (89.76, 94.08, 88.63, 93.52),
    ("CUB200", 50):    (89.68, 93.94, 82.06, 91.04),
}
R_OF = {"CIFAR100": 64, "IMAGENETR": 32, "CUB200": 8, "IMAGENETA": 4}

cells = {}
for key, blob in res.items():
    m = re.match(r"([A-Z0-9]+)\|(\d+)\|(\d+)\|R(\d+)_k\d+e(\d+)_", key)
    if not m:
        continue
    ds, T, seed, R, ne = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    if ne != 0 or R != R_OF.get(ds):          # only the chosen recipe
        continue
    cells.setdefault((ds, T), {})[seed] = blob

def ms(v):
    return (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0)

W = 118
print("\n" + "=" * W)
print("FINAL TABLE — conic fusion (fuse_km, NE=0) vs the RanPAC/A+ bar and external methods")
print("=" * W)
print(f"{'dataset':<11}{'T':>3}{'R':>4}{'n':>3} | {'OURS A-Last':>16}{'OURS A-Avg':>16} | "
      f"{'A+ bar':>14} | {'GR-LoRA':>15}{'MACIL':>15} | {'dLast':>7}")
print("-" * W)
for ds in ("CIFAR100", "IMAGENETR", "CUB200", "IMAGENETA"):
    for T in (10, 20, 50):
        got = cells.get((ds, T))
        if not got:
            print(f"{ds:<11}{T:>3}{R_OF[ds]:>4}{0:>3} |  (not run)")
            continue
        sd = sorted(got)
        ours_l, ours_a = ms([got[s]["fuse_km"]["A_last"] * 100 for s in sd]), \
                         ms([got[s]["fuse_km"]["A_avg"] * 100 for s in sd])
        bar_l = ms([got[s]["ranpac"]["A_last"] * 100 for s in sd])[0]
        bar_a = ms([got[s]["ranpac"]["A_avg"] * 100 for s in sd])[0]
        b = BARS.get((ds, T))
        ext = (f"{b[0]:>8.2f}/{b[1]:<6.2f}{b[2]:>8.2f}/{b[3]:<6.2f}" if b
               else f"{'--':>15}{'--':>15}")
        dl = f"{ours_l[0]-b[0]:>+7.2f}" if b else f"{'--':>7}"
        star = " *" if b and (ours_l[0] > b[0] or ours_a[0] > b[1]) else "  "
        print(f"{ds:<11}{T:>3}{R_OF[ds]:>4}{len(sd):>3} | "
              f"{ours_l[0]:>10.2f}+-{ours_l[1]:<4.2f}{ours_a[0]:>10.2f}+-{ours_a[1]:<4.2f} | "
              f"{bar_l:>6.2f}/{bar_a:<7.2f} | {ext} | {dl}{star}")
print("-" * W)
print("OURS = fuse_km (conic rule over R k-means centroids, z-fused with the RanPAC head).")
print("A+ bar = the ranpac arm recomputed inside the SAME replay -- it must match exp16;")
print("   if it does not, the cell is broken and the row means nothing.")
print("'*' = beats GR-LoRA on either metric.  dLast = ours - GR-LoRA A-Last.")
print("ImageNet-A has no external bar in exp16_full_table and 85/200 classes have <16 fit")
print("   rows, so read it as a robustness datapoint, not a comparison.")
print("=" * W)
PY
