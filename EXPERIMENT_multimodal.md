# Experiment: Multimodal class test (random-5 + superclass-5)

Tests the hypothesis **"a conic hull beats a prototype (NCM) in proportion to
intra-class multimodality"** on frozen ViT-B/16 IN21k features, with controls that
rule out the two confounds (class count, data volume). Driver:
`demo_joint_floor.py`; scripts: `reproduce_cifar100_superclass.sh`,
`reproduce_multimodal_other_datasets.sh`.

## Setup (shared)

```
backbone  = frozen ViT-B/16 (timm IN21k, model's own normalization)   # never trained
Ftr, ytr  = features(backbone, train)   # cached to feats_<dataset>_*.npz
Fte, yte  = features(backbone, test)
NCM(F, y) : per-class mean direction; classify test by max cosine to means
CONE(F,y) : per class, DISC rays (most-distinctive samples → k-means centroids);
            classify test by max NNLS membership  (best of cosine/blended/...)
metric    : cone_minus_ncm = acc(CONE) − acc(NCM)        # >0 ⇒ cone wins
```

## How a single run is conducted

```
function run(dataset, label_mode, transform, class_limit=0, samples_per_class=0):
    Ftr, ytr, Fte, yte = features(dataset)          # cached; labels are fine (0..C-1)
    assert_healthy(Ftr, ytr)                         # train-NCM ≫ chance ⇒ norm OK

    # ---- impose class structure (the independent variable) ----
    if   label_mode == "unimodal":      pass                         # fine labels, no change
    elif label_mode == "random5":       ytr,yte = MERGE_RANDOM(ytr,yte, k=5)
    elif label_mode == "superclass5":   ytr,yte = CIFAR100_SUPERCLASS(ytr,yte)  # 100→20 real

    # ---- controls (hold confounds fixed) ----
    if class_limit:        keep classes [0..class_limit)             # fix #classes
    if samples_per_class:  subsample each class to S train samples   # fix data volume

    # ---- representation + classifiers (identical for both) ----
    T   = fit_transform(Ftr, ytr, transform)         # pca = denoise (mode-preserving)
    Ftr, Fte = T(Ftr), T(Fte)
    return acc(CONE(Ftr,ytr) on Fte) − acc(NCM(Ftr,ytr) on Fte)     # cone_minus_ncm
```

### MERGE_RANDOM (random-5): synthetic, *dissimilar* modes
```
function MERGE_RANDOM(ytr, yte, k=5, seed=0):
    perm = shuffle([0..C-1], seed)                   # random order of fine classes
    for group_id, chunk in enumerate(chunks(perm, k)):   # 5 fine classes per group
        fine2coarse[chunk] = group_id                # → C/k coarse labels
    return fine2coarse[ytr], fine2coarse[yte]        # FEATURES UNCHANGED, labels only
# each coarse class = 5 unrelated fine classes ⇒ 5 distinct feature clusters (modes)
```

### CIFAR100_SUPERCLASS (superclass-5): real, *related* modes
```
function CIFAR100_SUPERCLASS(ytr, yte):
    fine2coarse = official CIFAR-100 map (100 fine → 20 superclasses, 5 each)
    return fine2coarse[ytr], fine2coarse[yte]        # FEATURES UNCHANGED, labels only
# each superclass = 5 RELATED fine classes (e.g. trees) ⇒ milder multimodality
```

Both touch **only the labels** — the cached features are identical to the unimodal
run, so any change in `cone_minus_ncm` is attributable to the imposed multimodality,
not to different inputs.

## How it verifies the results

The claim is verified by a set of **controlled contrasts**, each isolating one
variable (all with `transform=pca`, `rays=disc`):

```
# A. CORE: same #classes (20), vary modality          → isolates multimodality
run("CIFAR100","unimodal",   pca, class_limit=20)   ->  -0.005   (NCM wins)
run("CIFAR100","random5",    pca)                   ->  +0.061   (cone wins)
   ⇒ at fixed class count, modality flips the winner. "few classes" REFUTED.

# B. DOSE-RESPONSE: vary DEGREE of multimodality      → mechanism is multimodality
unimodal  -0.005   <   superclass5 +0.006   <   random5 +0.061
   ⇒ cone gain is MONOTONE in mode dissimilarity (none < related < dissimilar).

# C. DATA-VOLUME control: cap S=500 train/class        → isolates from data amount
random5 +0.061  ≈  random5(S=500) +0.063
   ⇒ win survives matched data volume. "more data" REFUTED.

# D. GENERALIZATION: random5, matched class count, per dataset
unimodal vs multimodal cone_minus_ncm:
   FGVCAircraft +0.021→+0.097 ; Pet -0.001→+0.057 ;
   CUB +0.000→+0.116 ; Cars +0.002→+0.139           (4/5 hold; larger than CIFAR)

# E. BOUNDARY: severe scarcity                          → defines when it fails
Flowers102 (~10 train/fine ⇒ ~10/mode): multimodal -0.085 (cone overfits, NCM wins)
   ⇒ effect needs ≳30 samples/mode.
```

**Verdict logic:** the hypothesis holds iff, *at fixed class count and data volume*,
`cone_minus_ncm(multimodal) ≫ cone_minus_ncm(unimodal)`. Contrasts A–D confirm it
(causal, dose-dependent, dataset-general); E bounds it (samples-per-mode floor).
NCM is the control classifier throughout — the prototype (1-ray) special case of the
cone — so the comparison isolates the value of the *region* representation.

## Reproduce

```bash
bash reproduce_cifar100_superclass.sh                 # A, B, C on CIFAR-100 (cached)
K=5 S=30 bash reproduce_multimodal_other_datasets.sh  # D, E on other datasets
```
