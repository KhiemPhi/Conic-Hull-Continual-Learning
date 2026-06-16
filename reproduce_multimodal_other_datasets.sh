#!/bin/bash
# reproduce_multimodal_other_datasets.sh
# ------------------------------------------------------------------------------
# Tests whether "cones beat prototypes on MULTIMODAL classes" generalizes to other
# datasets — which have no semantic superclasses, so we use RANDOM merge-K.
#
# Per dataset, MATCHED control (same class count, optionally same samples/class):
#   unimodal  : first  C/K fine classes (MERGE_K=1, CLASS_LIMIT=C/K)
#   multimodal: random merge-K          (MERGE_K=K → ~C/K classes)
# Both use transform=pca, disc rays.  If cone−NCM is ≫ larger for the multimodal
# arm, the effect holds for that dataset.
#
# Usage:
#   bash reproduce_multimodal_other_datasets.sh                    # light tv sets
#   K=5 S=30 bash reproduce_multimodal_other_datasets.sh          # merge-5, 30 samples/class
#   DATASETS="FGVCAircraft Food101 CUB200 StanfordCars" bash reproduce_multimodal_other_datasets.sh
#
# Notes: CUB200/StanfordCars need `pip install datasets`; first run of each set
# downloads + extracts features (cached afterward).  Proxy may be needed.
# ------------------------------------------------------------------------------
set -u
source "$HOME/venvs/ml_env/bin/activate"
export http_proxy=${http_proxy:-http://fwdproxy:8080}
export https_proxy=${https_proxy:-http://fwdproxy:8080}

K=${K:-5}                     # merge factor
S=${S:-0}                     # samples/class cap (0 = off; set e.g. 30 for airtight match)
DATASETS=${DATASETS:-"FGVCAircraft Flowers102 OxfordIIITPet"}
OUT=results_multimodal; mkdir -p "$OUT"

# known class counts (for the matched unimodal class limit C/K)
declare -A NCLS=( [CIFAR100]=100 [FGVCAircraft]=100 [Flowers102]=102 \
                  [OxfordIIITPet]=37 [Food101]=101 [CUB200]=200 [StanfordCars]=196 )

run () {  # run <label> <python knob statements>
  local label="$1"; shift
  echo "=== $label ==="
  python -u -c "import demo_joint_floor as J; J.RAY_METHOD='disc'; J.TRANSFORM='pca'; \
$*; J.main()" 2>&1 | tee "$OUT/$label.log" \
    | grep -E "cone − NCM|best cone AUROC|\[data|\[merge|\[limit|\[subsample|health" || true
  echo
}

for D in $DATASETS; do
  C=${NCLS[$D]:-100}
  LIM=$(( C / K ))
  echo "##################### $D  (C=$C, K=$K → ~$LIM classes, S=$S) #####################"
  run "${D}_unimodal"   "J.DATASET='$D'; J.MERGE_K=1; J.CLASS_LIMIT=$LIM; J.SAMPLES_PER_CLASS=$S"
  run "${D}_multimodal" "J.DATASET='$D'; J.MERGE_K=$K;                    J.SAMPLES_PER_CLASS=$S"
done

echo "===================== SUMMARY (cone − NCM, matched class count) ====================="
echo " holds for a dataset iff: multimodal cone−NCM  ≫  unimodal cone−NCM"
for D in $DATASETS; do
  u=$(grep -h 'cone − NCM' "$OUT/${D}_unimodal.log"   2>/dev/null | tail -1)
  m=$(grep -h 'cone − NCM' "$OUT/${D}_multimodal.log" 2>/dev/null | tail -1)
  printf "%-16s  unimodal: %-22s  multimodal: %s\n" "$D" "${u:-—}" "${m:-—}"
done
