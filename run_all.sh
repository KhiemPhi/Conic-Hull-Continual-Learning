#!/bin/bash
# run_all.sh — every experiment still worth running, in dependency order.
# Each writes logs/<name>.txt and appends a one-line status to logs/SUMMARY.txt.
#
# Ordered so the DECISION-RELEVANT runs come first: bars before methods (a method number
# is meaningless without the bar it is measured against), ceiling before targets (no CIL
# method can exceed the joint/offline bound).
#
#   ./run_all.sh            # run everything not already done (~7 h)
#   ./run_all.sh bars       # one stage only: bars|methods|ceiling|cone|extras
#   FORCE=1 ./run_all.sh    # re-run even if a log already exists
#   DRY=1 ./run_all.sh      # print the plan and exit
#
# Safe to Ctrl-C and re-run: completed steps are skipped unless FORCE=1.

set -u
cd "$(dirname "$0")"

VENV="${VENV:-$HOME/venvs/ml_env}"
# Standard PTM-CIL backbone (ImageNet-21k only). Override to compare, e.g.
#   MODEL=vit_base_patch16_224.augreg2_in21k_ft_in1k ./run_all.sh
export MODEL="${MODEL:-vit_base_patch16_224.augreg_in21k}"
export HF_HUB_OFFLINE=1
LOGS="logs/${MODEL##*.}"
mkdir -p "$LOGS"
SUMMARY="$LOGS/SUMMARY.txt"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
DRY="${DRY:-0}"

# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "no venv at $VENV"; exit 1; }

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# run <name> <est-minutes> <command...>
run() {
  local name="$1" mins="$2"; shift 2
  local log="$LOGS/$name.txt"
  if [[ "$DRY" == "1" ]]; then printf '  %-26s ~%3s min  %s\n' "$name" "$mins" "$*"; return; fi
  if [[ -s "$log" && "$FORCE" != "1" ]]; then
    echo "  [skip] $name (log exists; FORCE=1 to redo)"; return
  fi
  banner "$name  (~$mins min)"
  local t0=$SECONDS
  {
    echo "### $name"
    echo "### model=$MODEL"
    echo "### cmd: $*"
    echo "### started: $(date -Is)"
    echo
  } > "$log"
  # stdbuf keeps the log live so you can tail it while it runs
  if stdbuf -oL -eL "$@" >> "$log" 2>&1; then
    local st="OK"
  else
    local st="FAIL(rc=$?)"
  fi
  local el=$(( (SECONDS - t0) / 60 ))
  echo "### finished: $(date -Is)  [$st]" >> "$log"
  printf '%-26s %-10s %4d min  %s\n' "$name" "$st" "$el" "$(date -Is)" >> "$SUMMARY"
  echo "  -> $st in ${el} min   ($log)"
}

want() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

echo "backbone : $MODEL"
echo "logs     : $LOGS/"
[[ "$DRY" == "1" ]] && echo "(dry run — nothing will execute)"

# ---------------------------------------------------------------- 0. preflight
if [[ "$DRY" != "1" ]]; then
  banner "preflight: is the backbone cached?"
  python - <<PY || { echo "
Backbone not cached. Download once (needs the proxy), then re-run:

  export http_proxy=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 HF_HUB_DISABLE_XET=1
  python -c \"import timm; timm.create_model('$MODEL', pretrained=True, num_classes=0)\"
  unset http_proxy https_proxy
"; exit 1; }
import os, timm
os.environ["HF_HUB_OFFLINE"] = "1"
m = timm.create_model("$MODEL", pretrained=True, num_classes=0)
print(f"  OK  {m.num_features}-d features")
PY
fi

# ---------------------------------------------------------------- 1. bars
# Nothing below is interpretable without these. A_plus (adapt task 0, then freeze) is THE
# bar; A_frozen (never adapt) is the floor. Both measured with the same head/protocol.
if want bars; then
  run bars_A_frozen_A_plus 40 \
      env VARIANTS=A_frozen,A_plus python -u exp8_combined.py
fi
[[ "$STAGE" == "bars" ]] && { echo; echo "bars done -> $LOGS"; exit 0; }

# ---------------------------------------------------------------- 2. ceiling
# The joint/offline bound. No CIL method can exceed it, so it decides which targets are
# even reachable. (On augreg2 it was 0.8355; on IN21k 0.8332 -> A-last 0.84 is impossible.)
if want ceiling; then
  run ceiling_joint 40 \
      env DATASET=IMAGENETR python -u crux_headroom.py
fi

# --- 2b. RECIPE SWEEP: is the ceiling underfit?  The original number trained with the EVAL
#     transform (no crop/flip/augmentation at all) for 10 epochs on 24k images. The ceiling
#     bounds every target, so probe it before chasing a better CIL method.
#     DECISION: ceiling >=0.87 -> tune the recipe everywhere (~6 h to re-validate);
#               ceiling ~0.84  -> recipe is not the bottleneck, switch backbone (CLIP/DINOv2).
if want recipe; then
  run recipe_sweep 150 \
      env DATASET=IMAGENETR SWEEP=1 python -u crux_headroom.py
fi

# --- STAGE 0: is 0.8498 even the right ceiling? ----------------------------------------
# Our bound used a SINGLE r32 LoRA; SOTA aggregates ten task-specific adapters
# (W0 + sum Bi Ai). Jointly trained, that sum has rank <= T*r, so it is EXACTLY as
# expressive as one rank-320 LoRA -- and this is the BEST case, since their adapters each
# see only one task. A rank sweep therefore answers the ceiling question directly.
#   r320 ~ r32  -> capacity is not the constraint, targets unchanged
#   r320 >> r32 -> we were measuring against the wrong bound
if want ranks; then
  run ceiling_ranks 90 \
      env DATASET=IMAGENETR RANKS=1 python -u crux_headroom.py
fi

# ---------------------------------------------------------------- 3. methods
# The contenders. full_accum is the untested configuration with the best shot at the
# bar's A-avg; cov_maha_distill is the current best A-last (0.7457 on augreg2).
if want methods; then
  run method_full_accum 25 \
      env VARIANTS=full_accum python -u exp8_combined.py
  run method_cov_maha_distill 25 \
      env VARIANTS=cov_maha_distill python -u exp8_combined.py
  run method_null_baseline 25 \
      env VARIANTS=null_baseline python -u exp8_combined.py
fi

# ---------------------------------------------------------------- 4. cone ablations
# cone_only is the one component whose structure-vs-trivial control PASSED (+21 vs L2).
# lam1=10/100 were never reachable before grad clipping; M_EIG=256 targets the 22.7%
# unprotected energy (low expectation: the diagonal fix on the same 22.7% gave +0.0025).
if want cone; then
  run cone_accum 25 \
      env VARIANTS=cone_accum python -u exp8_combined.py
  run cone_lam10 25 \
      env VARIANTS=cone_accum LAM1=10 GRAD_CLIP=1.0 python -u exp8_combined.py
  run cone_lam100 25 \
      env VARIANTS=cone_accum LAM1=100 GRAD_CLIP=1.0 python -u exp8_combined.py
  run cone_lam0.1 25 \
      env VARIANTS=cone_accum LAM1=0.1 python -u exp8_combined.py
  run cone_meig256 25 \
      env VARIANTS=cone_accum M_EIG=256 python -u exp8_combined.py
fi

# --- 4c. RECIPE PROPAGATION -----------------------------------------------------------
# The joint sweep raised the ceiling 0.8332 -> 0.8510 with AUG=1 + 40 epochs (augmentation
# alone at 10 ep was worth +0.0008 — the schedule is what pays). The CIL runs still use the
# OLD recipe, so they are undertrained relative to the ceiling they are measured against.
# Both arms must move together: a tuned method vs an untuned bar is not a comparison.
if want recipe2; then
  run r2_A_plus     70 env AUG=1 EPOCHS=40 TAG=aug40 VARIANTS=A_plus     python -u exp8_combined.py
  run r2_full_accum 60 env AUG=1 EPOCHS=40 TAG=aug40 VARIANTS=full_accum python -u exp8_combined.py
fi

# --- 4d. HEADROOM DECOMPOSITION -------------------------------------------------------
# 61% of the headroom (~10 acc points) is uncaptured. Is it approximate STATISTICS or worse
# FEATURES? They need opposite fixes. ORACLE_STATS rebuilds the head from all real seen data
# in the current frame (cheating, diagnostic only) -- no retraining, ~5 min extra.
if want split; then
  run split_full_accum 30 \
      env ORACLE_STATS=1 TAG=orc VARIANTS=full_accum python -u exp8_combined.py
fi

# ---------------------------------------------------------------- 4b. SEEDS
# full_accum beat the bar by +0.0063 A-last / +0.0007 A-avg on seed 0. Those margins are thin
# enough that they could be noise, so the paired comparison must be repeated. Both arms use the
# SAME seed each round (same class order), which is what makes the pairing meaningful.
# Seeds must run at the WINNING recipe (aug40) -- that is where the headline claim lives.
# Both arms share the seed each round so the comparison is PAIRED (class order is the dominant
# variance source in CIL, and pairing removes it).
# Tests maha_distill_accum -- the CURRENT best (0.7910/0.8421) after the cone was ablated out.
# full_accum (0.7903, with the cone) is superseded; running seeds on it would waste 2 h.
if want seeds; then
  for s in 1 2; do
    run "seed${s}_A_plus"       10 env SEED=$s AUG=1 EPOCHS=40 TAG=aug40 \
                                   VARIANTS=A_plus python -u exp8_combined.py
    run "seed${s}_maha_distill" 55 env SEED=$s AUG=1 EPOCHS=40 TAG=aug40 \
                                   VARIANTS=maha_distill_accum python -u exp8_combined.py
  done
fi

# --- 4e. NEW PENALTY FORMULATIONS ------------------------------------------------------
# Three orthogonal extensions, each ablated against the current penalty (full_accum 0.7527).
#   kfac      protect the FUNCTION (E||J dW x||^2) instead of the LAYER output (E||dW x||^2).
#             Adds K-FAC's G factor; A (x) G IS the Fisher, we currently use only A.
#             CAUTION: two-sided truncation compounds -- the log prints the retained energy
#             product; if it is small, raise M_EIG/M_OUT before judging the idea.
#   transport Procrustes-align the accumulated bank into the current frame before adding.
#             Fixes summing covariances measured in DIFFERENT frames.
#   between   protect class-mean scatter (where discriminative info lives) rather than total.
if want newpen; then
  run np_kfac        35 env PEN_MODE=kfac  TAG=kfac  VARIANTS=full_accum python -u exp8_combined.py
  run np_kfac_m256   40 env PEN_MODE=kfac  M_OUT=256 M_EIG=256 TAG=kfac256 \
                         VARIANTS=full_accum python -u exp8_combined.py
  run np_transport   35 env BANK_MODE=transport TAG=trans VARIANTS=full_accum python -u exp8_combined.py
  run np_between     30 env COV_MODE=between TAG=btw   VARIANTS=full_accum python -u exp8_combined.py
  run np_within      30 env COV_MODE=within  TAG=wth   VARIANTS=full_accum python -u exp8_combined.py
  run np_kfac_trans  40 env PEN_MODE=kfac BANK_MODE=transport TAG=kfactrans \
                         VARIANTS=full_accum python -u exp8_combined.py
fi

# --- 4f. THE MISSING ABLATION + recipe-matched comparison ------------------------------
# cone_accum ALONE is below A_frozen at every lambda tried (0.6557/0.5788/0.6510 vs 0.6867),
# yet full_accum = 0.7527. Does the cone contribute at all, or is maha+distill carrying it?
# Run at the WINNING recipe so the comparison is against A_plus_aug40 (0.7897), not the
# stale 10ep bar.
if want ablate; then
  run ab_maha_distill      35 env VARIANTS=maha_distill_accum python -u exp8_combined.py
  run ab_maha_distill_aug40 60 env AUG=1 EPOCHS=40 TAG=aug40 \
                              VARIANTS=maha_distill_accum python -u exp8_combined.py
  run ab_cone_accum_aug40   60 env AUG=1 EPOCHS=40 TAG=aug40 \
                              VARIANTS=cone_accum python -u exp8_combined.py
fi

# --- 4g. DIVERSITY vs VOLUME -----------------------------------------------------------
# `joint 0.8498 > A_plus 0.7897` confounds 200 CLASSES with 24000 IMAGES. This control
# separates them and decides whether ANY prototype-free loss can work:
#   div_only (200 cls, 12 img/cls) ~ joint     -> DIVERSITY matters, a cross-task loss can help
#   div_only ~ vol_only (20 cls)              -> VOLUME matters, only replay closes it
if want control; then
  run diversity_vs_volume 60 \
      env DATASET=IMAGENETR CONTROL=1 python -u crux_headroom.py
fi

# --- 4h. PROTOTYPE-FREE cross-task pressure --------------------------------------------
# No stored per-class information anywhere.
#   anticollapse : keep the representation full-volume so later classes still have dimensions
#                  to occupy. Task-wise CE is otherwise free to collapse onto its 20 classes.
#                  (VERIFIED: VICReg's variance term alone is BLIND to rank collapse; logdet
#                   and the decorrelation term are what see it.)
#   maha_base    : distil relational geometry from the frozen pretrained phi_0 rather than
#                  phi^{t-1}. phi_0 is free and its structure reflects ALL 200 classes.
if want protofree; then
  run pf_anticollapse   60 env AUG=1 EPOCHS=40 LAM4=1.0 TAG=ac \
                            VARIANTS=full_accum python -u exp8_combined.py
  run pf_anticollapse_10 60 env AUG=1 EPOCHS=40 LAM4=10 TAG=ac10 \
                            VARIANTS=full_accum python -u exp8_combined.py
  run pf_maha_base      60 env AUG=1 EPOCHS=40 MAHA_TEACHER=base TAG=mbase \
                            VARIANTS=full_accum python -u exp8_combined.py
  run pf_both           60 env AUG=1 EPOCHS=40 LAM4=1.0 MAHA_TEACHER=base TAG=pfboth \
                            VARIANTS=full_accum python -u exp8_combined.py
fi

# --- 4i. HEAD COMPARISON ----------------------------------------------------------------
# Head quality is separable from feature quality. Both our method AND the ceiling use the
# same RanPAC head, so a better head raises BOTH -- it moves the absolute number the 0.82/0.84
# targets are stated in. Cheapest measurement left; we have never varied this component.
if want heads; then
  run heads_frozen 8  env FEATS=frozen python -u exp9_heads.py
  run heads_joint  50 env FEATS=joint  python -u exp9_heads.py
fi

# ---------------------------------------------------------------- 5. extras
# Cheap, cached-feature experiments unrelated to the CIL stack.
# exp3b decides whether the semantic-order positive is cone-specific or just "regions".
if want extras; then
  run exp3b_order_baselines 5 \
      python -u exp3b_order_baselines.py
  # (11) conic gain as an INSTRUMENT rather than a classifier -- the one use where the cone's
  #      known behaviour (sensitivity to intra-class multimodality) is the SIGNAL, not a
  #      liability. Standalone: cached CIFAR-100 ViT + Waterbirds CLIP features, no backbone.
  run exp2_cone_diagnostic 12 \
      python -u exp2_cone_diagnostic.py
  # (10) conic (non-negative) vs signed combination of task adapters. NOTE: runs on the
  #      DEPRECATED augreg2_in21k_ft_in1k backbone, because that is what the cached adapters
  #      were trained on. Mechanism question only -- not comparable to current IN21k numbers.
  run exp4_nonneg_merge 45 \
      python -u exp4_nonneg_merge.py
  # the last untested CONE-SPECIFIC claim: does the literal Dirichlet conic-mixture beat
  # Gaussian-around-prototype for virtual features? (exp6 has both; exp8 only has gauss)
  run exp6_dirichlet_vs_gauss 50 \
      env VARIANTS=virt_gauss,virt_dirichlet python -u exp6_covcone_virtual.py
fi

# ---------------------------------------------------------------- report
banner "SUMMARY"
[[ -s "$SUMMARY" ]] && cat "$SUMMARY"
echo
echo "logs: $LOGS/"
echo
# Print the FINAL results table from each log (the per-variant rows, not just headers).
# Rows look like:  <name>   0.7527   0.8049 | +0.0063 +0.0007 | ...
echo "all measured variants (deduplicated, best A-last first):"
grep -hE '^ *[A-Za-z_0-9]+ +0\.[0-9]{4} +0\.[0-9]{4}' "$LOGS"/*.txt 2>/dev/null \
  | sed 's/  */ /g' | sort -u -k1,1 | sort -t' ' -k3,3gr
echo
echo "ceiling / headroom:"
grep -hE "RanPAC headroom|^ *(frozen|lora|fullft) " "$LOGS"/*.txt 2>/dev/null | sort -u
echo
echo "wins:"
grep -hE "BEATS BAR" "$LOGS"/*.txt 2>/dev/null | sed 's/  */ /g' | sort -u
