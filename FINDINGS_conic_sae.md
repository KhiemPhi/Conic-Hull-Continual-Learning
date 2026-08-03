# Conic-hull / non-negativity in vision — complete findings

**One-line thesis.** Non-negativity (conic structure) is an **interpretability** property, not a
downstream-task property. It causally makes concept dictionaries *monosemantic*; it ties or
loses on every predictive task, including the ones designed to favor it.

## The predictive rule (earned from the failures)

A conic representation helps **iff** (1) the target is a non-negative *superposition* (not a
contrast), (2) atoms are near-pure / supervised (separability), and (3) the *code is the output*
or non-negativity is a hard constraint. Downstream tasks violate ≥1 clause → a simpler method
(linear / dense / prototype / signed) wins. Interpretability satisfies all three.

---

## Part A — cone as a discriminative scorer / downstream learner: TIES or LOSES

| # | line | result | winner |
|---|---|---|---|
| A1 | CLIP cross-dataset OOD (`min_c r_c` vs NCM) | tie, mean AUROC .976 vs .975 | NCM (cone wins only ImageNet-A +.008) |
| A2 | Synthetic ⊕-of-parts vs matched vMF | tie even with **oracle** generators (+.003±.002); SPA recovers parts at 0.205 cos | vMF |
| A3 | CUB fine-tuned OOD (vs NCM/Maha/KNN) | loses on ViT (.975 vs 1.0), ties on ResNet18 (.993) | NCM/Maha/KNN |
| A4 | Cone vs **multi-prototype** NCM (multimodal ID) | loses every budget (CIFAR100 Δ −.043→−.014) | multi-prototype |
| A5 | Train the space conic (cone-margin loss) | makes cone **worse** (Δ −.057 vs raw −.011) | multi-prototype |
| A6 | Multi-label mAP (NNLS vs linear) | mirage: +.19 vs weak probe → tie vs tuned; NNLS≈cos-kNN | tuned linear |
| A7 | Counting (NNLS vs ridge) | COCO +.031 (5 seeds) but **flips on CPPE-5** (−.08); ranking-only | ridge (dataset-fragile) |
| A8 | USS clustering, STEGO protocol (COCO-Stuff-27 style, ADE) | raw DINOv2 **23.6** > conic 17.2 > signed 16.3 mIoU | raw features |
| A9 | Debiasing, Waterbirds worst-group | ERM 61.2, **INLP 73.2**, conic 58.9, signed 56.9 | linear concept removal (INLP) |
| A10 | CZSL, UT-Zappos (best-case arena) | **signed** unseen 56.2 / AUC 26.7 vs conic 34.6 / 17.3 — non-neg **HURTS** composition | signed |

**Mechanism of failure:** the cone's "fill between atoms" and additive superposition is a
*liability* for OOD/clustering/debiasing (fill = where negatives live) and irrelevant for
discrimination; a matched signed code reconstructs slightly better and transfers better.

---

## Part B — cone as a non-negative sparse CODE for interpretability: WINS (on the right metric)

Setup: Top-K SAE dictionary on frozen ViT patch tokens; conic (ReLU) vs signed (no ReLU),
matched dict size, matched L0, matched reconstruction — a clean causal isolation of non-negativity.

**B1. Code quality (CLIP / COCO, 3 seeds):**
| method | nMSE↓ | coherence↑ | purity↑ | probe mAP↑ |
|---|--:|--:|--:|--:|
| conic-SAE | 0.409 | **0.571±.000** | **0.763** | **0.677** |
| signed-SAE | 0.418 | 0.417±.006 | 0.658 | 0.631 |
| PCA | 0.636 | 0.240 | 0.590 | 0.457 |
| kmeans (L0=1) | 0.647 | 0.525 | 0.807 | 0.575 |

**B2. Monosemanticity generalizes (conic−signed coherence / purity / probe-mAP):**
CLIP/COCO +.154/+.105/+.046 · DINOv2/COCO +.177/+.098/+.050 · CLIP/CUB +.102/+.161/+.073 ·
DINOv2/CUB +.138/+.089/−.019. Coherence + purity conic>signed in **all 4 cells** (probe-mAP
3/4). Holds across contrastive & self-supervised backbones and two datasets.

**B3. Ground-truth localization (Network-Dissection style, best-atom):**
| GT source | CLIP conic / signed | DINOv2 conic / signed |
|---|--:|--:|
| COCO boxes (loc-AP) | 0.294 / 0.198 | 0.366 / 0.245 |
| COCO boxes (IoU) | 0.228 / 0.159 | 0.292 / 0.192 |
| ADE dense masks, 329 cls (loc-AP/IoU) | 0.168 / 0.135 (vs signed 0.125/0.101) | — |
PCA ≈ random on all.

**B4. Validated with the repo's ConicHull (SPA extreme rays + NNLS), not just the SAE:**
ConicHull coherence **0.590 = conic-SAE 0.590 ≫ signed 0.418** → monosemanticity is a *conic*
property, reproduced by the actual extreme-ray hull. But ConicHull *localizes worse*
(loc-AP 0.179 vs learned 0.300) — SPA picks extreme/atypical exemplars: coherent but off-centre.

---

## Part C — the important refinements (what NOT to claim)

- **NOT a free lunch.** On CIFAR-100 SpLiCE-style faithfulness (zero-shot acc vs sparsity,
  ceiling 77.3): learned dicts ≫ text-dict (conic 76.3 / signed 76.5 vs SpLiCE 62.2 @k=32 — but
  SpLiCE handicapped to 362 concepts offline). **conic ≤ signed on faithfulness** (k=8: 73.8 vs
  74.9). Non-negativity costs ~1pt accuracy; its benefit is monosemanticity only.
- **The trade-off is real, not a null:** non-negativity → monosemantic atoms **but** worse
  compositional transfer (A10) and slightly worse fidelity (C). Monosemantic ⇏ better learner.
- **Downstream thesis FALSIFIED** (A10 was the pre-registered decider — every failure condition
  inverted, still lost).

---

## Conclusion & positioning

The conic/non-negativity contribution is **interpretability-only and real**: at matched cost it
causally improves the *monosemanticity and localizability* of ViT concept dictionaries, robust
across backbones, datasets, and two conic instantiations (learned SAE + SPA extreme-ray hull).
Compete on **Network Dissection / Broden** (best-atom concept-IoU) and **SAEBench** axes
(auto-interp, probing) — where the metric *is* the property you win. Do **not** enter task
leaderboards (OV-seg, USS, debiasing, OOD, CZSL) — a simpler method wins every one.

Closest priors to differentiate from: **CRAFT / Fel et al.** (NMF concept extraction — you add
causal isolation of non-negativity on dense ViT features) and **SpLiCE** (global text-dictionary
— you add dense/spatial + learned dictionary).

*Files: cone_geometry.py, cone_ood.py, compositional/, cub_ood_benchmark.py, cone_vs_multiproto_ood.py,
conic_train_ood.py, coco_multilabel_law.py, coco_counting.py, additive_counting.py, vit_sae_conic.py,
sae_segmentation_iou.py, stego_eval.py, waterbirds_debias.py, czsl_conic.py, splice_cifar.py,
conichull_interp.py.*
