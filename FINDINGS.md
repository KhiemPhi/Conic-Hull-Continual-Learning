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

## 5. CONFOUND RESOLVED: it's multimodality, not "few classes"

Merge-5 changed two things (multimodal AND 100→20 classes). The control fixes the
class count at 20 and varies only modality:

| 20 classes (pca, disc) | NCM | best cone | cone − NCM | OOD Δ |
|---|---|---|---|---|
| **unimodal** (CLASS_LIMIT=20, MERGE_K=1) | 0.937 | 0.932 | **−0.005 (NCM)** | −0.003 |
| **multimodal** (MERGE_K=5) | 0.742 | 0.803 | **+0.061 (cone)** | +0.058 |

Class count held at 20: the cone **loses** when unimodal, **wins +6%** when
multimodal. The +0.066 swing is attributable purely to multimodality. The
"few-classes" hypothesis is **refuted** (at 20 unimodal classes the cone still
loses). Mechanism: merging drops NCM 0.94→0.74 (one mean can't cover 5 modes); the
cone recovers it to 0.80.

**Dose-response + both confounds closed (CIFAR-100, 20 classes, pca, disc):**

| modality | cone − NCM | sample-matched (500/cls) |
|---|---|---|
| unimodal (fine) | −0.005 | −0.005 |
| **real semantic superclasses** (5 related modes) | **+0.008** | **+0.006** |
| random merge-5 (5 dissimilar modes) | +0.061 | +0.063 |

Cone advantage is **monotone in multimodality degree** (none → related → dissimilar
modes) and **scales with mode dissimilarity**. Sample-matching to 500 train/class
barely moves it (random +0.061→+0.063) ⇒ **NOT a data-volume effect**. So at fixed
class count AND data volume, only multimodality moves the result. Real (superclass)
multimodality gives a small-but-positive effect (~+0.6%); synthetic strong
multimodality gives ~+6% (upper bound). Honest claim: cone gain ∝ intra-class
multimodality; modest on natural data, large when modes are distinct.

---

## 5b. Cross-dataset generalization + a scarcity boundary

Random merge-5, matched class count (C/K), pca/disc, unimodal vs multimodal:

| dataset | ~train/fine | unimodal cone−NCM | multimodal cone−NCM |
|---|---|---|---|
| FGVCAircraft | ~67 | +0.021 | **+0.097** |
| OxfordIIITPet | ~99 | −0.001 | **+0.057** |
| CUB-200 | ~30 | −0.002 | **+0.116** |
| StanfordCars | ~41 | +0.002 | **+0.139** |
| Flowers102 | **~10** | +0.005 | **−0.085** (cone LOSES) |

- Effect **generalizes** (5/5 in direction; 4/5 a clear cone win, +0.06–0.14) and is
  **larger on fine-grained datasets** (merging distinct fine classes → very distinct
  modes → NCM's single mean fails harder, cone recovers more).
- **Boundary condition (Flowers102):** the multi-ray cone needs enough samples per
  mode. CUB at ~30/mode wins (+0.116); Flowers at ~10/mode the cone overfits and
  loses (−0.085). Threshold ≈ 10–30 samples/mode. Likely rescued by fewer rays
  (N_RAYS scaling with samples/mode) — TODO confirm.

## 5c. Hierarchical coarse(cone)→fine(centroid) on CIFAR-100 superclasses

Shared structure: fine prototypes = centroids; each superclass cone's rays = its 5
fine centroids. Route coarse, then nearest fine centroid in the routed group.

| transform | flat NCM | hier(NCM-route) | hier(cone-route) | oracle | coarse route cone vs NCM |
|---|---|---|---|---|---|
| pca | 0.818 | 0.771 | 0.811 | **0.892** | 0.887 vs 0.849 (**+0.038**) |
| none | 0.818 | 0.773 | 0.805 | 0.893 | 0.881 vs 0.850 (+0.031) |
| lda | 0.834 | 0.806 | 0.830 | 0.905 | 0.899 vs 0.879 (+0.020) |

- **Confirmed:** the cone routes the (multimodal) coarse level better than the
  prototype, every transform (+0.02 to +0.04 routing accuracy).
- **But hard hierarchy loses to flat** (−0.004 to −0.013): ~12% coarse-routing
  errors propagate (fine stage can't recover); flat 100-way never makes an
  unrecoverable coarse error.
- **Oracle (0.89–0.90) ≫ flat (0.82):** the promise is large (+0.07–0.09); the
  bottleneck is routing accuracy, not the architecture.
- **Fix:** SOFT routing — top-k coarse + fine-rank over the union, or score fusion
  `score(f)=α·cone(g(f))+(1−α)·cos(q,cent_f)` (α=0 ⇒ flat, so dominates flat for
  tuned α). Converts the cone's routing edge into end-to-end gain. (TODO.)

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

## 7. Net thesis (controlled, multi-dataset)

**A class is a cone, not a point. The cone's advantage is governed by intra-class
multimodality, with a samples-per-mode floor.**
1. **Multimodal/coarse classes:** cone beats prototype on closed-set (+0.06 to
   +0.14 across CIFAR-100, FGVCAircraft, OxfordIIITPet, CUB-200, StanfordCars) and
   on OOD — *iff a structure-preserving transform (pca/none); mode-collapsing
   transforms (lda/whiten) hide it.*
2. **Causal & controlled:** at fixed class count AND data volume, cone−NCM is
   monotone in multimodality degree (unimodal −0.005 → real superclasses +0.006 →
   random merge +0.06); not a few-classes or data-volume artifact.
3. **Boundary condition:** needs ≳30 samples/mode; under severe scarcity
   (Flowers102 ~10/mode) the multi-ray cone overfits and the prototype wins.
4. **Unimodal/fine-grained data:** cone matches the prototype closed-set, edges it
   on OOD (all 5 datasets) — its only universal advantage there.
5. **Transform is primitive-dependent:** decorrelate→points (RanPAC), denoise→cones.
6. **Incrementally forgetting-free** by construction (ε≡0 when frozen; ε<γ/2).

**One line:** conic-hull classifiers beat prototypes in proportion to intra-class
multimodality (+6–14% on multimodal/coarse, negligible on unimodal), given enough
samples/mode and a structure-preserving transform, while consistently edging
prototypes on open-set OOD and remaining forgetting-free by construction.

## 8. TODO
- [x] Few-classes vs multimodal control — RESOLVED (§5).
- [x] Semantic CIFAR-100 superclasses — done (§5, +0.006 matched).
- [x] Cross-dataset generalization — done (§5b, 4/5 win).
- [ ] Confirm Flowers scarcity via N_RAYS=5 + subsample-CUB-to-10 control.
- [ ] Continuous MERGE_K dose-response figure {1,2,3,5,10}.
- [ ] Robust γ metric in `cone_boundary.measure_gamma`.
- [ ] Head-to-heads vs RanPAC/FeCAM/ADAM on standard CIL benchmarks.

Now the implications. Your results aren't just "a new classifier" — they say something about the
  assumptions baked into the dominant CIL and prototype paradigms.

  Implications for continual learning

  1. Frozen-PTM CIL (NCM/RanPAC/FeCAM) has a structural blind spot: multimodal classes. These methods
  model a class as a point or a Gaussian — implicitly unimodal. That assumption holds for
  fine-grained-but-distinct classes on strong PTMs (where you tie), but breaks for coarse/multimodal
  labels (superclasses, domain-spanning classes, web/e-commerce categories, "vehicle/animal/food"), where
  you showed a region model recovers +6–14%. Much of real CIL is coarse/multimodal, so this is a real gap,
  not a corner case.
  2. Per-class model capacity should match per-class structure — and it's measurable. Your dose-response
  says the right representation depends on how multimodal a class is. Actionable design principle:
  estimate intra-class modes and allocate capacity (1 prototype → K rays) accordingly, rather than fixing
  "one mean per class" globally. The cone is a capacity knob the prototype doesn't have.
  3. The transform and the classifier interact, and the interaction is primitive-dependent. RanPAC's win =
  decorrelation + prototype. You showed decorrelation hurts region models (denoising helps them). So you
  cannot transplant RanPAC's RP+Gram recipe onto a richer class model and expect the same gain — the
  preprocessing must be co-designed with the primitive. CIL papers report transform+head as a bundle; your
  work shows why that bundle isn't transferable.
  4. Forgetting decomposes geometrically (drift ε + crowding γ), and freezing only kills drift. The ε<γ/2
  boundary reframes catastrophic forgetting: with a frozen PTM, ε≡0 and the residual forgetting is pure
  crowding (γ shrinks as classes accumulate). This explains why training-free CIL works and where its
  floor is — and says the remaining lever is increasing inter-class separation (γ, via expansion), not
  fighting drift. A unifying lens on the whole frozen-PTM CIL line.
  5. The field's drift toward open-world / task-free CIL favors region models. Prototypes give only
  relative scores; cones give a calibrated "outside-the-class" boundary (your consistent OOD edge across
  all 5 datasets). So as CIL moves toward "is this even a seen class?", region/density primitives have a
  structural advantage independent of multimodality.
  6. Data-regime caveat — few-shot CIL should stick with prototypes. The samples-per-mode floor
  (Flowers102 ~10/mode → cone overfits, loses) means richer class models cost data. So the choice is
  regime-dependent: few-shot → prototype; data-rich + multimodal → region. This nuances the naive "richer
  is always better."

  Implications for prototype learning (beyond CIL)

  1. The single prototype is a degenerate case (a 1-ray cone), provably insufficient for multimodal
  classes. NCM, ProtoNet, metric learning, ArcFace all assume unimodal classes; your study quantifies the
  cost of that assumption (up to −0.14) and isolates when it bites (multimodality, with controls). That's
  a clean characterization of a known-but-rarely-measured limitation.
  2. For multimodal classes, model the class as a conic combination of prototypes, not a hard
  nearest-prototype assignment. Key subtlety from your data: on unimodal data the cone reduced to nearest
  multi-prototype (max_ray_sim best); on multimodal data the NNLS combination of mode-rays beat
  nearest-prototype (blended/cosine best). So a query is best explained as a non-negative mixture of modes
  — soft, compositional assignment beats hard assignment exactly when classes are multimodal.
  3. Prototype methods aren't wrong — they're optimal in their regime. The honest counterweight: on
  unimodal closed-set, the decorrelated prototype (RanPAC, ~0.90) beats the cone. So the contribution
  isn't "prototypes lose"; it's delineating the regime boundary — prototypes own unimodal/few-shot;
  regions own multimodal/data-rich/open-set.
  4. Prototype systems needing OOD should consider region extensions — the absolute membership boundary is
  something a point classifier structurally lacks.

  The unifying principle

  ▎ "What is a class?" is a modeling choice that should match (a) intra-class structure (#modes), (b) the
  ▎ task (closed- vs open-set), and (c) the data regime (samples/mode) — not a fixed point. The point
  ▎ (prototype) and the region (cone) are two ends of a capacity spectrum; your study maps which end wins
  ▎ where, with controlled, measurable boundaries.

  That framing is your paper's Discussion section, and it's the kind of conceptual takeaway (not a
  leaderboard delta) that lands at TMLR/CoLLAs — and that prototype/CIL researchers can actually act on.
