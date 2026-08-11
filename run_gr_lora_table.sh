#!/usr/bin/env bash
# run_gr_lora_table.sh -- build the full GR-LoRA comparison table under PILOT's class order.
#
# THE GRID
#   4 datasets x 3 task counts x 3 seeds = 36 cells, matching the cells GR-LoRA reports
#   (exp16_full_table.py:99, ICML'26 Tables 1,2,6, ViT-B/16-IN21k, mean of 3 seeds).
#   IMAGENETAP and CUB200P are the PILOT-CORRECTED splits -- the published IMAGENETA/CUB200
#   numbers correspond to those, not to our original variants. See splits.py.
#
# THREE STAGES PER CELL, IN DEPENDENCY ORDER
#   exp16  trains member q32 and writes the bar that exp55 asserts against
#   exp55  trains m32/a16/q32b70/q64 and writes the ensemble anchor that exp56 asserts against
#   exp56  the method: per-member cone fused in matched geometry (FE), R=64, uniform average
#   Each stage skips keys it has already written, so this script is RESUMABLE and safe to
#   kill. Re-running it costs one JSON read per completed cell.
#
# WHY T IS THE OUTER LOOP
#   Read-out cost scales as sum(t^2)/T^2: T=20 is 1.86x T=10 and T=50 is 4.46x. Looping T
#   outermost finishes every dataset at T=10 (~12 h) before spending ~38 h on T=50, so the
#   biggest open question -- does this generalise past IMAGENETR, given CUB200P historically
#   had ZERO read-out headroom and IMAGENETAP was the worst base deficit -- gets answered
#   first and cheaply. Do not reorder these loops without a reason.
#
# FAILURE POLICY
#   A failing cell is RECORDED AND SKIPPED, not fatal: one bad dataset must not kill a
#   three-day run. Every failure is re-listed in the final summary with its log path, and the
#   exit status is non-zero if anything failed, so this is still safe to chain.
#
# USAGE
#   ./run_gr_lora_table.sh                      # everything, phased
#   TCOUNTS=10 ./run_gr_lora_table.sh           # phase 1 only (recommended first)
#   DSETS=IMAGENETR TCOUNTS=10 SEEDS=1,2 ./run_gr_lora_table.sh
#   DRY_RUN=1 ./run_gr_lora_table.sh            # print the plan, run nothing
#   SKIP_TRAIN=1 ./run_gr_lora_table.sh         # read-out only, assumes features cached
set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 1

# ------------------------------------------------------------------ config
DSETS="${DSETS:-IMAGENETR CUB200P IMAGENETAP CIFAR100}"
TCOUNTS="${TCOUNTS:-10 20 50}"
SEEDS="${SEEDS:-0,1,2}"
MEMBERS="${MEMBERS:-q32,m32,a16,q32b70,q64}"
R="${R:-64}"
ORDER="${ORDER:-pilot}"
SUFFIX="${SUFFIX:-_table}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
VENV="${VENV:-$HOME/venvs/ml_env}"
LOGDIR="$REPO/logs/table"

# exp49 measured the unpinned noise floor at 0.27, larger than most effects in this table,
# and unpinned runs break the exp55/exp56 repro asserts. Non-negotiable.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ORDER

# ------------------------------------------------------------------ preflight
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "FATAL: no venv at $VENV (override with VENV=...)" >&2; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
for f in exp16_full_table.py exp55_lora_diversity_pilot.py exp56_ray_ensemble.py \
         class_order.py; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done
python3 -c "import os;os.environ['ORDER']='$ORDER';import class_order as C;C.mode()" \
  || { echo "FATAL: ORDER=$ORDER is not a valid class_order mode" >&2; exit 1; }
mkdir -p "$LOGDIR"

NCELLS=0
for _t in $TCOUNTS; do for _d in $DSETS; do NCELLS=$((NCELLS + 1)); done; done
echo "=================================================================="
echo " GR-LoRA table build   ORDER=$ORDER  R=$R  seeds=$SEEDS"
echo " datasets : $DSETS"
echo " T counts : $TCOUNTS   ->  $NCELLS (dataset,T) groups"
echo " members  : $MEMBERS"
echo " logs     : $LOGDIR"
[[ "$SKIP_TRAIN" == "1" ]] && echo " SKIP_TRAIN=1 -- read-out only, features must be cached"
[[ "$DRY_RUN" == "1" ]]   && echo " DRY_RUN=1 -- nothing will be executed"
echo "=================================================================="

# ------------------------------------------------------------------ runner
declare -a FAILED=()
declare -a DONE=()
T_START=$SECONDS

run_stage () {           # run_stage <label> <logfile> <env assignments...> -- <script>
  local label="$1"; shift
  local logf="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '    [dry] %-34s %s\n' "$label" "$*"
    return 0
  fi
  local t0=$SECONDS
  printf '    %-34s ' "$label"
  if env "$@" > "$logf" 2>&1; then
    printf 'ok   %5ds\n' $((SECONDS - t0))
    return 0
  fi
  printf 'FAIL %5ds  -> %s\n' $((SECONDS - t0)) "$logf"
  tail -n 15 "$logf" | sed 's/^/        | /'
  return 1
}

for TT in $TCOUNTS; do
  echo ""
  echo "###### PHASE T=$TT ################################################"
  for DSET in $DSETS; do
    echo ""
    echo "  --- $DSET  T=$TT  seeds=$SEEDS ---"
    STAMP="${DSET}_T${TT}_${ORDER}"
    CELL_OK=1

    if [[ "$SKIP_TRAIN" != "1" ]]; then
      run_stage "exp16  (member q32)" "$LOGDIR/exp16_$STAMP.log" \
        "ORDER=$ORDER" "DATASETS=$DSET" "TASKS=$TT" "SEEDS=$SEEDS" \
        "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1" \
        python3 -u exp16_full_table.py || CELL_OK=0

      if [[ $CELL_OK == 1 ]]; then
        run_stage "exp55  (members 2-5 + anchor)" "$LOGDIR/exp55_$STAMP.log" \
          "ORDER=$ORDER" "DS=$DSET" "T=$TT" "SEED=$SEEDS" "MEMBERS=$MEMBERS" "VERIFY=1" \
          "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1" \
          python3 -u exp55_lora_diversity_pilot.py || CELL_OK=0
      fi
    fi

    if [[ $CELL_OK == 1 ]]; then
      run_stage "exp56  (FE cone, R=$R)" "$LOGDIR/exp56_$STAMP.log" \
        "ORDER=$ORDER" "DS=$DSET" "T=$TT" "SEED=$SEEDS" "MEMBERS=$MEMBERS" \
        "ARMS=f$R" "RULES=cone" "ORDERS=FE" "RAYSETS=all" "VERIFY=1" "SUFFIX=$SUFFIX" \
        "OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1" \
        python3 -u exp56_ray_ensemble.py || CELL_OK=0
    fi

    if [[ $CELL_OK == 1 ]]; then
      DONE+=("$DSET T=$TT")
    else
      FAILED+=("$DSET T=$TT  (see $LOGDIR/*_$STAMP.log)")
      echo "    !! skipping the rest of this group; continuing with the next one"
    fi
  done
done

# ------------------------------------------------------------------ summary
echo ""
echo "=================================================================="
printf ' finished in %dh%02dm   %d ok   %d failed\n' \
  $(((SECONDS - T_START) / 3600)) $((((SECONDS - T_START) % 3600) / 60)) \
  "${#DONE[@]}" "${#FAILED[@]}"
if ((${#FAILED[@]})); then
  echo " FAILED GROUPS:"
  printf '   - %s\n' "${FAILED[@]}"
fi
echo "=================================================================="
if [[ "$DRY_RUN" != "1" ]]; then
  echo ""
  echo " Final table (re-runs the exp56 summary over everything cached; computes nothing):"
  echo "   ORDER=$ORDER DS=\"\$(echo $DSETS | tr ' ' ,)\" T=\"\$(echo $TCOUNTS | tr ' ' ,)\" \\"
  echo "     SEED=$SEEDS MEMBERS=$MEMBERS ARMS=f$R RULES=cone ORDERS=FE RAYSETS=all \\"
  echo "     SUFFIX=$SUFFIX python3 -u exp56_ray_ensemble.py"
fi
((${#FAILED[@]} == 0))
