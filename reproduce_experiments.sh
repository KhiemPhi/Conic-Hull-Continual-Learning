#!/bin/bash
# reproduce_experiments.sh
# ------------------------------------------------------------------------------
# Reproduces the joint-floor conic-hull study (closed-set + open-set OOD) on
# frozen ViT-B/16 IN21k features, plus the incremental forgetting-free boundary.
#
# Each run drives demo_joint_floor.py by overriding module-level knobs, so no
# edits are needed.  Features are cached per (dataset, model) in feats_*.npz —
# the first run of each dataset extracts once; reruns are instant.
#
# Usage:
#   bash reproduce_experiments.sh            # core (CIFAR-100, fast on cache)
#   HEAVY=1 bash reproduce_experiments.sh    # + extra datasets (downloads ~GBs)
#   BOUNDARY=1 bash reproduce_experiments.sh # + incremental cone_boundary run
# ------------------------------------------------------------------------------
set -u
source "$HOME/venvs/ml_env/bin/activate"
export http_proxy=${http_proxy:-http://fwdproxy:8080}
export https_proxy=${https_proxy:-http://fwdproxy:8080}

OUT=results
mkdir -p "$OUT"

# run <label> <python knob-setting statements>  → logs to results/<label>.log
run () {
  local label="$1"; shift
  echo "=== $label ==="
  python -u -c "import demo_joint_floor as J; $*; J.main()" 2>&1 | tee "$OUT/$label.log" \
    | grep -E "cone − NCM|best cone AUROC|NCM \(cosine|health|\[merge|\[limit|\[data" || true
  echo
}

C="J.DATASET='CIFAR100'"          # CIFAR-100 base (cached)

# ── 1. Transform comparison (CIFAR-100, disc rays) ────────────────────────────
#    Shows the denoise↔decorrelate axis: whiten→prototype, pca/lda→cone.
for T in none whiten partial_whiten pca lda; do
  run "01_transform_$T"   "$C; J.RAY_METHOD='disc'; J.TRANSFORM='$T'; J.MERGE_K=1"
done

# ── 2. Ray-construction comparison (CIFAR-100, lda) ───────────────────────────
#    spa (reconstruction) vs kmeans (multi-proto) vs disc (discriminative core).
for R in spa kmeans disc; do
  run "02_rays_$R"        "$C; J.RAY_METHOD='$R'; J.TRANSFORM='lda'; J.MERGE_K=1"
done

# ── 3. Multimodal test — merge-5 under mode-collapsing vs mode-preserving xform
#    lda collapses modes (tie); pca/none preserve them (cone wins).
run "03_merge5_lda"       "$C; J.RAY_METHOD='disc'; J.TRANSFORM='lda';  J.MERGE_K=5"
run "03_merge5_none"      "$C; J.RAY_METHOD='disc'; J.TRANSFORM='none'; J.MERGE_K=5"
run "03_merge5_pca"       "$C; J.RAY_METHOD='disc'; J.TRANSFORM='pca';  J.MERGE_K=5"

# ── 4. CONFOUND CONTROL — few classes vs multimodal (the open question) ────────
#    Same class count (20), vary modality.  If 20-unimodal also shows a big
#    cone−NCM, the merge-5 win was "few classes", not multimodality.
run "04_ctrl_20_unimodal" "$C; J.RAY_METHOD='disc'; J.TRANSFORM='pca'; J.MERGE_K=1; J.CLASS_LIMIT=20"
run "04_ctrl_20_multimodal" "$C; J.RAY_METHOD='disc'; J.TRANSFORM='pca'; J.MERGE_K=5"   # =20 merged
#    Class-count sweep (unimodal) — does cone−NCM grow as #classes shrinks?
for L in 10 20 50 100; do
  run "04_ctrl_unimodal_C$L" "$C; J.RAY_METHOD='disc'; J.TRANSFORM='pca'; J.MERGE_K=1; J.CLASS_LIMIT=$L"
done
#    Multimodal at MATCHED few-class counts via MERGE_K (100→ {50,33,20,10})
for K in 2 3 5 10; do
  run "04_merge_K$K"      "$C; J.RAY_METHOD='disc'; J.TRANSFORM='pca'; J.MERGE_K=$K"
done

# ── 5. Per-dataset floor + normalization health check (MERGE_K=1) ─────────────
run "05_data_CIFAR100"    "$C; J.RAY_METHOD='disc'; J.TRANSFORM='lda'; J.MERGE_K=1"
if [ "${HEAVY:-0}" = "1" ]; then
  for D in FGVCAircraft Flowers102 OxfordIIITPet Food101; do
    run "05_data_$D"      "J.DATASET='$D'; J.RAY_METHOD='disc'; J.TRANSFORM='lda'; J.MERGE_K=1"
  done
  # HuggingFace sets (need: pip install datasets)
  for D in CUB200 StanfordCars; do
    run "05_data_$D"      "J.DATASET='$D'; J.RAY_METHOD='disc'; J.TRANSFORM='lda'; J.MERGE_K=1"
  done
fi

# ── 6. Incremental forgetting-free boundary (separate driver) ─────────────────
if [ "${BOUNDARY:-0}" = "1" ]; then
  echo "=== 06_cone_boundary ==="
  python -u demo_cone_boundary.py 2>&1 | tee "$OUT/06_cone_boundary.log" \
    | grep -E "final_avg_acc|eps=|boundary_held" || true
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo "================= SUMMARY (cone − NCM / OOD Δ) ================="
for f in "$OUT"/*.log; do
  cs=$(grep -h "cone − NCM" "$f" | tail -1)
  od=$(grep -h "best cone AUROC" "$f" | tail -1)
  printf "%-26s  %s  %s\n" "$(basename "$f" .log)" "${cs:-—}" "${od:-—}"
done
echo "Logs in $OUT/.  Findings: FINDINGS.md"
