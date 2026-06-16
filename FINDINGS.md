# Conic-Hull Continual Learning — Findings

Frozen ViT-B/16 (timm `vit_base_patch16_224.orig_in21k`) features; per-class
**conic hulls** (extreme rays + NNLS membership) vs **NCM** (nearest-class-mean,
the frozen-prototype baseline). Joint (offline, all-classes-at-once) "floor" in
`demo_joint_floor.py`; incremental forgetting study in `demo_cone_boundary.py`.
Reproduce everything with `bash reproduce_experiments.sh`.

---

## 0. Critical bug found & fixed: input normalization

The shared loader normalized CIFAR-100 with **CIFAR channel stats**, but the IN21k
ViT expects its own (**mean/std = 0.5**). A frozen backbone is extremely sensitive
to this.

- Joint floor (cone, CIFAR-100): **0.38 → 0.82** after using the model's own
  `timm.data.resolve_model_data_config` transform.
- Incremental `cone_frozen_rp`: **0.51 → 0.83**.

⇒ All earlier numbers on the buggy norm are void. A `check_feature_health` step
now prints feature-norm stats + train-NCM after extraction to catch this on any
new dataset (train-NCM ≈ chance ⇒ suspect normalization).

---

## 1. Closed-set: cone ≈ NCM on standard data

On the corrected frozen features, cone and NCM are within ~±1% closed-set:

| dataset | kind | NCM | best cone | cone − NCM |
|---|---|---|---|---|
| CIFAR-100 (100, fine) | unimodal | 0.818 | 0.826 (none) / 0.841 (lda) | +0.008 / +0.007 |
| FGVCAircraft (100) | fine-grained | 0.362 | 0.345 | **−0.017** |
| CUB-200 (200) | fine-grained | 0.843 | 0.843 | +0.000 |
| StanfordCars (196) | fine-grained | 0.436 | 0.406 | **−0.030** |

**Conclusion:** the cone does NOT beat a frozen prototype on standard (unimodal /
fine-grained) closed-set classification. Fine-grained = tight, *crowded* classes,
which favor a single prototype and punish the over-covering cone. A *decorrelated*
prototype (RanPAC, ~0.88–0.90 on CIFAR-100) beats the cone outright on closed-set.

---

## 2. Feature transforms are primitive-dependent

Optimal transform differs for points vs cones (fit on train; ID-only for OOD):

| transform | what it does | helps |
|---|---|---|
| **whiten** (ZCA, Σ^−1/2) | decorrelate; **amplifies low-variance noise** | **prototype** (Mahalanobis/RanPAC); HURTS cones |
| **lda** | discriminant subspace; whitens within-class scatter | prototype on multimodal (collapses modes); best for unimodal closed-set |
| **pca / none** | denoise / preserve structure | **cone** |

Evidence (CIFAR-100): whiten lifts NCM 0.818→0.834 but drops cone OOD 0.86→0.73.
PCA drops NCM OOD but holds cone OOD ⇒ cone overtakes. See `whitening-hurts-cones`.

---

## 3. Open-set OOD: cone consistently ≥ NCM (the universal edge)

80/20 ID/unseen split; AUROC + FPR@95; cone score = max in-cone membership or
raw NNLS residual (`geo_residual`):

| dataset | NCM AUROC | best cone AUROC | Δ |
|---|---|---|---|
| CIFAR-100 (pca) | 0.849 | 0.860 | **+0.011** |
| FGVCAircraft (lda) | 0.650 | 0.659 | +0.009 |
| CUB-200 (lda) | 0.964 | 0.964 | +0.000 (saturated) |
| StanfordCars (lda) | 0.645 | 0.661 | **+0.017** |

**Conclusion:** across every dataset the cone is ≥ NCM on OOD, with better FPR@95;
the edge is largest where the problem is hardest. The cone's calibrated
"outside-the-cone" residual is a structural advantage a point classifier lacks.

---

## 4. THE result: cones beat prototypes on multimodal classes

CIFAR-100, randomly merge K=5 fine classes → 1 (20 classes, each with 5 dissimilar
sub-modes). Transform decides whether the modes survive:

| transform | NCM | best cone | cone − NCM | OOD Δ |
|---|---|---|---|---|
| **lda** (collapses modes) | 0.823 | 0.823 | **+0.000 (tied)** | +0.004 |
| **pca** (preserves modes) | 0.742 | **0.803** | **+0.061** | **+0.058** |

A single mean cannot represent 5 modes; the cone places rays on each and scores a
query as their **non-negative combination**. Under PCA the cone wins closed-set by
+6% and OOD by +6%. Under LDA it ties — LDA's within-class whitening collapses the
inter-mode spread, erasing exactly the structure the cone exploits.

Bonus: on multimodal data the NNLS scores (`blended`/`cosine`) BEAT `max_ray_sim`
(nearest-prototype) — i.e. the actual *conic* geometry adds value beyond
multi-prototype. (On unimodal data `max_ray_sim` was best ⇒ cone reduced to
multi-prototype there.) See `cones-win-on-multimodal`.

---

## 5. OPEN CONFOUND (top priority): "few classes" vs "multimodal"

Merge-5 changed **two** things at once: classes became multimodal **and** dropped
100 → 20. The +6% could be from *fewer classes* (less crowding), not modes.

**Control (now runnable via `CLASS_LIMIT`):** fix the class count, vary modality.
- `CLASS_LIMIT=20, MERGE_K=1` → 20 **unimodal** classes.
- `MERGE_K=5` → 20 **multimodal** classes.
Same count; if the unimodal-20 also shows a large cone−NCM, the win was class
count, not multimodality. Also: unimodal class-count sweep {10,20,50,100}.
(Section 4 of `reproduce_experiments.sh`.) **This must be resolved before claiming
the multimodal result.**

---

## 6. Incremental: forgetting-free boundary (ε < γ/2)

`demo_cone_boundary.py`: a class is a cone; forgetting splits into **drift** (ε,
angular movement of features from where the cone was frozen) and **crowding**
(γ/2, min inter-class ray angle shrinking as classes accumulate).

- **`cone_frozen_rp`** (cone-anchor first-session shaping + RP(10000)+ReLU, then
  FREEZE): ε ≡ 0 by construction ⇒ drift-free. Final avg acc **0.833** on
  CIFAR-100, forgetting +0.058 (= the irreducible crowding floor).
- Adapting the backbone grows ε past γ/2 → forgetting; freezing removes the drift
  term entirely.
- Note: the γ metric is a `max`-cosine (fragile to one degenerate ray pair); it can
  read 0 spuriously — a robust percentile/zero-ray-filtered version is a TODO.

---

## 7. Net thesis

**A class is a cone, not a point.**
1. On **multimodal/coarse** classes the conic hull beats prototypes on closed-set
   (+6%) AND OOD (+6%) — *provided a structure-preserving transform (pca/none);
   mode-collapsing transforms (lda/whiten) hide it.* (Pending the §5 confound.)
2. On **unimodal/fine-grained** data the cone matches the prototype on closed-set
   and consistently edges it on **OOD**.
3. The optimal transform is **primitive-dependent** (decorrelate→points,
   denoise→cones) — don't bolt RanPAC's RP+Gram onto cones.
4. Incrementally, the cone is **forgetting-free by construction** (ε≡0 when frozen),
   with forgetting reducing to a measurable crowding floor.

## 8. TODO
- [ ] §5 confound control (few classes vs multimodal) — **decisive**.
- [ ] MERGE_K sweep {2,3,5,10} + `none`≈`pca` check (mechanism curve).
- [ ] Semantic CIFAR-100 superclasses (real, not synthetic, multimodality).
- [ ] Robust γ metric in `cone_boundary.measure_gamma`.
- [ ] Discriminative-cone vs decorrelated-prototype head-to-head on multimodal.
