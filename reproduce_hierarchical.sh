#!/bin/bash
# reproduce_hierarchical.sh
# ------------------------------------------------------------------------------
# Verifies the coarse(cone)→fine(centroid) hierarchy on CIFAR-100's 20 official
# superclasses (using the cached fine-label features — instant).
#
# Each run reports, for end-to-end 100-way fine accuracy:
#   flat 100-way NCM (prototype)         — baseline
#   hier: NCM-coarse  → centroid         — coarse routed by prototype
#   hier: CONE-coarse → centroid         — coarse routed by cone (the proposed method)
#   hier: ORACLE-coarse → centroid       — ceiling (perfect routing)
# plus coarse routing accuracy (cone vs NCM).
#
# Verify:
#   - cone routes coarse BETTER than NCM   (coarse classes are multimodal → cone wins)
#   - hier(cone) ≥ flat NCM                (hierarchy helps end-to-end)
#   - gap to ORACLE = cost of routing errors
#
#   bash reproduce_hierarchical.sh
# ------------------------------------------------------------------------------
set -u
source "$HOME/venvs/ml_env/bin/activate"
OUT=results_hier; mkdir -p "$OUT"

GREP='HIERARCHICAL|flat|hier:|top-|α=|cone vs NCM|best soft|--'

run () {  # run <label> <python knob statements>
  local label="$1"; shift
  echo "=== $label ==="
  python -u -c "import demo_joint_floor as J; J.DATASET='CIFAR100'; \
J.RUN_HIERARCHICAL=True; $*; J.main()" 2>&1 | tee "$OUT/$label.log" \
    | grep -E "$GREP" || true
  echo
}

# transform controls whether the modes survive (pca/none preserve; lda collapses)
run hier_pca   "J.TRANSFORM='pca'"
run hier_none  "J.TRANSFORM='none'"
run hier_lda   "J.TRANSFORM='lda'"

echo "================= SUMMARY (vs flat NCM) ================="
for f in hier_pca hier_none hier_lda; do
  echo "--- $f ---"; grep -E "flat 100|hier: CONE|top-|α=|best soft|cone vs NCM" "$OUT/$f.log" 2>/dev/null
done
