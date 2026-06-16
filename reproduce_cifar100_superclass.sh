#!/bin/bash
# reproduce_cifar100_superclass.sh
# ------------------------------------------------------------------------------
# REAL multimodality: CIFAR-100's 20 official semantic superclasses (each = 5
# RELATED fine classes), vs random merge-5 (strong synthetic multimodality) vs
# 20 unimodal fine classes — all at 20 classes, transform=pca, disc rays.
# Plus sample-matched arms (500 train/class) so the multimodal win can't be
# attributed to more data.  Reuses the CIFAR-100 feature cache (instant).
#
#   bash reproduce_cifar100_superclass.sh
# ------------------------------------------------------------------------------
set -u
source "$HOME/venvs/ml_env/bin/activate"
OUT=results_superclass; mkdir -p "$OUT"

run () {  # run <label> <python knob statements>
  local label="$1"; shift
  echo "=== $label ==="
  python -u -c "import demo_joint_floor as J; J.DATASET='CIFAR100'; \
J.RAY_METHOD='disc'; J.TRANSFORM='pca'; $*; J.main()" 2>&1 | tee "$OUT/$label.log" \
    | grep -E "cone − NCM|best cone AUROC|\[data|\[coarse|\[merge|\[limit|\[subsample" || true
  echo
}

# ── 20 classes, NOT sample-matched (semantic/random have 2500/cls, unimodal 500) ─
run uni20            "J.MERGE_K=1; J.CLASS_LIMIT=20"      # 20 unimodal fine classes
run semantic20       "J.SEMANTIC_COARSE=True"            # 20 REAL superclasses (5 related modes)
run random20         "J.MERGE_K=5"                       # 20 random-merged (5 dissimilar modes)

# ── 20 classes, SAMPLE-MATCHED to 500 train/class (airtight: same data volume) ──
run uni20_m          "J.MERGE_K=1; J.CLASS_LIMIT=20;  J.SAMPLES_PER_CLASS=500"
run semantic20_m     "J.SEMANTIC_COARSE=True;          J.SAMPLES_PER_CLASS=500"
run random20_m       "J.MERGE_K=5;                     J.SAMPLES_PER_CLASS=500"

echo "===================== SUMMARY (cone − NCM) ====================="
echo " expectation: uni20 ≤ 0 ; semantic20 > 0 (real, milder) ; random20 ≫ 0 (strong)"
for f in uni20 semantic20 random20 uni20_m semantic20_m random20_m; do
  printf "%-16s  %s\n" "$f" "$(grep -h 'cone − NCM' "$OUT/$f.log" 2>/dev/null | tail -1)"
done
