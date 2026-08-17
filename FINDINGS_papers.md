# FINDINGS — claim-to-evidence map for both papers

Snapshot: 2026-08-16. **Not a source of truth for numbers** — `results.txt` (Paper 1) and
`results_paper2.txt` (Paper 2) are generated from the experiment JSONs by `make_results.py` and
`make_paper2_results.py` and supersede anything here if they disagree. This file exists to record
which experiment backs which *claim*, which controls have actually been run, and what a referee
will still be able to attack.

Every number below is a mean over 3 seeds unless the row says otherwise.

---

## PAPER 2 — NeurReps extended abstract (deadline 2026-08-24), then ICML

**Central claim.** Representation drift under parameter-efficient continual adaptation is
approximately confined to the gauge orbit of the prototype read-out — a change of frame, not a
loss of content. Reading it that way identifies the Gram matrix as the invariant to preserve, and
a penalty built on that invariant eliminates representational forgetting.

Setup for all rows: CIFAR-100, `vit_base_patch16_224.augreg2_in21k_ft_in1k`, shared LoRA r32.

| # | Claim | Experiment | Key numbers |
|---|---|---|---|
| 1 | The drift is **affine** | `crux_transport` ×3 seeds | MLP recon cos 0.8678 vs linear 0.8637 (**+0.004**) and MLP is *worse* on accuracy (0.8997 vs 0.9077) |
| 2 | The drift is **bounded**, not accumulating | `crux_sweep` ×3 | eps saturates 25.5deg -> 57.3deg over 10 tasks |
| 3 | **One Procrustes map suffices, at every depth** | `crux_sweep` ×3 | \|transport - oracle\| **<= 0.004 at all 10 stages**; stale prototypes decay to 0.873 |
| 4 | The **residual is itself a rotation** | `crux_recovered` ×3 | Gram **0.964** with 12.7deg displacement; rec/rec 0.739 vs oracle 0.750 (recovered features retain all class information) |
| 5 | **The invariant prescribes the loss** | `crux_relational` ×3 | cosine-Gram penalty: oracle 0.749 -> **0.816**; rigidity 80.6% -> 91.8%; Gram-corr 0.688 -> 0.908; eps 57.5 -> 52.7deg; new-task accuracy unchanged (0.967 -> 0.968) |
| 6 | eps is **cross-validated across protocols** | `crux_transport` + `crux_drift` | identity recon cos 0.6554 = **cos(49.0deg)**, independently reproducing eps measured on the separate single-step protocol |
| 7 | The penalty also **removes seed variance** | `crux_relational` ×3 | oracle sd **±0.032 -> ±0.001** — what you expect if a degree of freedom is genuinely being removed |

### Honest limits (stated in the paper, not hidden)
- Oracle 0.816 vs frozen 0.815 is a **TIE** (+0.002). CIFAR-100 on this backbone has no
  adaptation headroom to recover; a win requires a benchmark that does, which is NOT run.
- The gauge account explains prototype **staleness**, not **representational degradation**: at
  lambda=0 the SEEN-way oracle itself falls below frozen from t2 on and the gap widens
  (t9: 0.747 vs 0.815).
- One dataset, one backbone. "MLP ~ linear" is underpowered (~1e4 pairs vs a 5.7M-param MLP), so
  the honest claim is *no detectable nonlinear gain at this budget*.
- `crux_recovered` accuracy columns build prototypes from test features (~1% self-inclusion) and
  are mildly optimistic; the structural metrics that carry the argument (angles, %systematic, CKA,
  Gram) involve no classification and are unaffected.

### Bug fixed en route
`crux_transport`'s lambda=50 arm previously used a scrambled Gram reference (983/1000 rows wrong,
from indexing a class-blocked array with `searchsorted`). The rerun after the fix gives the
**first valid numbers** for that arm. No lambda=50 transport number was ever reported before it.

**Status: experimentally complete.** Remaining work is prose, two figures, related work.

---

## PAPER 1 — CVPR (~Nov)

**Central claim.** In PTM-CIL the gains attributed to ensemble *design* are largely not where the
literature assumes. A controlled decomposition shows which component does the work, when, and why
the rest cannot.

The original framing ("our 5-member conic ensemble beats SOTA") was **retired by our own
controls** — C1 showed the diversity design is worth ~0 on A-Last, C6 showed a single larger
backbone dominates the whole comparison. What replaced it is stronger.

| # | Claim | Experiment | Key numbers |
|---|---|---|---|
| 1 | **Non-negativity is a real, separable effect — but only under distribution shift, and it grows with T** | `exp66` C2, **36/36 cells ×3 seeds**, one fixed arm (f64), no per-cell selection | cone-minus-sub at *matched storage*, PAIRED per seed: ImageNet-R **+0.27 / +0.39 / +0.58** at T=10/20/50 (**9/9 seeds positive**, paired sd ~0.1); ImageNet-A +0.58/+0.53/+0.53; CIFAR100 ~0; CUB200P ~0 |
| 2 | **Member diversity is created and then structurally discarded** | `exp55` + `exp66_c1` (C1), `exp68` on both pools | Structural vs seed-only pool: disagreement 0.141 vs 0.111, error-corr 0.779 vs 0.831, ORACLE **86.81 vs 86.02 (+0.79)** — yet ensembles land at **81.78 vs 81.71 (+0.07)**. `exp68`: **78% vs 72%** of the oracle headroom is minority-correct (k<3 of 5); only **~14%** captured; **13.2%** of samples no member gets right; averaging breaks **0.00%** of unanimous-correct |
| 3 | **Backbone scaling dominates the ViT-B literature, ours included** | `exp67` C6/C7 ×3 seeds | single ViT-L **84.84 / 88.92** vs the 5-member ensemble 82.58 / 87.01 = **+2.26 / +1.91**, at **1/3.3** the inference cost (measured: 5x ViT-B 0.2381s vs 1x ViT-L 0.0724s per batch) |
| 4 | The ensemble **saturates at M=3** | `exp66` C5, 12 cells | M3 is within 0.03–0.29 of the full 5; *better* than full on ImageNet-A |
| 5 | **Ray budget must match the row-count SPREAD, not the median** | `exp56` sweeps, 6 cells ×3 seeds | sweeping bought CUB **+0.02 / +0.08** (nothing; rows 26–52) vs ImageNet-A **+1.26 / +1.75 / +1.76** (rows 1–75). Flipped ImageNet-A T=50 from -1.69 to **+0.07** vs GR-LoRA |
| 6 | The cone **never substitutes** for the ridge read-out | `exp61` (29/36 cells at time of writing) | FE > cone_ens > cone_q32 on *every* cell measured; cone-only clears GR-LoRA on CIFAR100 only |
| 7 | **Protocol corrections that affect published numbers** | `splits.py`, `class_order.py` | CUB's official split is 57% of the data PILOT-protocol papers use (5994/5794 vs 9430/2358); ImageNet-A under a global 80/20 leaves **4 classes with zero test images**; the PILOT class order is MT19937(1993+s), not `default_rng(seed)` — and that correction **flipped a pre-registered verdict** in our own work |
| 8 | Headline comparison | `exp56` table, 12 cells ×3 seeds | vs GR-LoRA: wins both metrics on CIFAR100 (all T), ImageNet-R (all T), ImageNet-A T=10; A-Avg wins on ImageNet-A T=20/50; **CUB200P is a clean null**, which the thesis predicts for fine-grained in-domain data |

### The mechanism behind claim 2 (this is the paper's best negative result)
Uniform averaging of z-scored member scores behaves like a **vote** near the decision boundary, so
it can only recover samples a *majority* of members already get right. The vote structure on
ImageNet-R T=10 (`exp68`, per-member accuracies asserted against `exp56` to 1e-9):

```
 k correct    mass   ens acc here   best-single acc here
        0    13.2%           0.5%                  0.0%
        1     3.5%          11.6%                 17.8%
        2     3.0%          45.3%                 41.3%
        3     3.2%          88.6%                 72.6%
        4     5.1%          99.6%                 92.4%
        5    72.1%         100.0%                100.0%
headroom mass by k:  k=1 49%   k=2 30%   k=3 15%   k=4 7%
```

85% of samples are unanimous (72.1% right + 13.2% wrong) so there is nothing to combine; the
headroom is concentrated at k=1. At k=1 the **best single member beats the ensemble** (17.8% vs
11.6%) — four confident wrong votes drown one right one. Recovering it needs per-sample routing,
and per-*class* routing is already closed: `oracle_class_cv` 0.803 < `best_single` 0.807, a
NEGATIVE share of the headroom.

### Caveat that must travel with claim 1
The cone's advantage is largest where the features are *worst* (~0 in-domain on CIFAR100/CUB200P,
largest on shifted ImageNet-R/-A). That is the signature of the cone **compensating for feature
deficiency**, not exploiting a fundamental geometry. Consequence for any future feature-side
experiment: if features improve, the cone's read-out edge should SHRINK even as absolute accuracy
rises, so absolute accuracy — not cone-minus-sub — is the success metric.

### Known gaps a referee will find
1. **RanPAC has NOT been run on our protocol.** On CIFAR our single-member base already beats
   GR-LoRA (92.78 vs 91.97 at T=10), so a referee will ask whether the gain is a stronger RanPAC
   implementation rather than the method. Highest-priority missing baseline.
2. C1 and C6 were IMAGENETR **T=10 only** when this file was written (T=20/T=50 now queued). T=50
   is where both the cone effect and the margin vs GR-LoRA are largest.
3. CIFAR100 T=10 is **2 seeds**; the ImageNet-R f4..f128 ray sweep is **seed 0 only**.
4. A conic *training* objective has never been tested on shifted data — see below.

---

## PAPER 3 — parked

`exp65`, **144/144 cells** (6 pretraining objectives x 4 adaptation budgets x 3 lrs x 2 datasets).
The pre-registered premise test came back **ALIVE at reorder_rate 0.25** (>= 0.20), but the
companion metric kills the framing: **regret is 0.13 pts** — choosing the backbone with a cheap
frozen linear probe costs **0.00 pts in 5 of 6 cells**. That makes the pre-registered P2 criterion
(a predictor must beat LogME and a K-shot probe by >= 1.0 pt on regret@1) **unreachable on these
datasets**, since total headroom is 0.13. No predictor was ever built.

Also: the three "reorders" are the same event (X -> full fine-tuning on ImageNet-R) counted three
times, so 0.25 overstates the evidence. Both datasets are ImageNet-adjacent natural-image sets —
the axis the premise is *about* (distance from the pretraining distribution) was never tested.
RESISC45 / SVHN are already in `download_datasets.py` and reserved for exactly this.

A prediction I got wrong and should record: I expected DINOv2 to degrade under adaptation. It
improves monotonically (CUB 89.31 -> 89.65, ImageNet-R 84.50 -> 90.67) and wins ImageNet-R at full
fine-tuning. **MAE** is the budget-sensitive backbone (29.39 -> 75.61 on CUB, +46 pts).

---

## Now queued

- **ce_conic on shifted data.** `exp48_conic_feature_loss` already tested conic *training*
  objectives, on CUB200 T=10 seed 0: `ce_conic` beat plain `ce` on all three read-outs
  (cone 89.45 vs 89.23, RanPAC **90.33 vs 89.96**, sub 89.30 vs 89.04), while `conic` ALONE was
  worse than `ce` (88.78) — a conic auxiliary supplements linear separability, it cannot replace
  it. But CUB200 is precisely the dataset where C2 says the cone read-out is worth ~0, so the
  objective was tested where its geometry does not matter. It has never been run on ImageNet-R/-A.
  Pre-register: >= +0.5 A-Last on the **RanPAC** read-out, paired, all 3 seeds positive. Prior ~30%
  (the one supporting number is +0.37 on a single seed, against an unpinned noise floor of 0.27).
- **C1 and C6 at T=20 and T=50.**

### Distinguish two kinds of conic training objective — one is closed for a structural reason
- **Preservation / anti-drift** (`exp6_covcone_virtual`, `exp8_combined`): 0.7065/0.8232 and
  0.7283/0.8016 against the A+ bar 0.7858/0.8313. Four independent regularizers all converged to
  0.71–0.74. The reason is structural and worth stating: a preservation penalty's **lambda -> inf
  limit IS the frozen backbone**, i.e. the first-session bar, so it cannot exceed what it
  converges to. This family is capped by construction.
- **Prospective / discriminative** (`exp48`): trains the reader's own decision rule (build each
  class cone from support rows, CE over `||Pi_C q||^2` for queries). Not capped the same way, and
  it is the family the queued run belongs to.
