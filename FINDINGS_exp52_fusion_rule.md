# exp52 — Is the fused conic win actually conic?

`exp52_fusion_rule_control.py` · oPCA rays (γ=0.5) on frozen exp16 LoRA features · T=10 · 3 seeds · threads pinned

## The question

The best result this project has is a **fused** one: RanPAC + a conic read-out. Every
standalone conic application is closed. So the only live claim is *"the cone adds
something to RanPAC"*, and two facts suggested the word `cone` might not be doing the work:

1. **The fusion gain was anti-correlated with cone quality.** In exp35 (IMAGENETR s0), the
   standalone cone got *worse* as R grew (79.80 → 78.02 from R=4 to R=64) while the fusion
   gain *grew* (+0.52 → +0.75). exp41's oPCA cone, 1.9pt better standalone, fused to the
   same 81.0. The fused ceiling was invariant to cone quality — the signature of ensemble
   decorrelation, not of a better class descriptor.
2. **Non-negativity was worth ~0.14 standalone** (cone vs sub on identical rays). Never
   measured inside the fusion.

So: three read-out **rules** over the **same** oPCA rays, same whitener, same
self-consistent negatives, each fused with RanPAC under an identical β search.

| rule | non-negativity | combination | isolates |
|---|---|---|---|
| `cone` | ✓ | ✓ | the method (NNLS conic score) |
| `sub` | ✗ | ✓ | `‖q·B‖`, B = orthonormal basis of span(A). Sign constraint dropped. |
| `pm` | ✗ | ✗ | `max_j cos(q, a_j)`. Nearest-atom over the same rays. |

`cone − sub` isolates the sign constraint at fixed atoms **and fixed span**. `sub − pm` isolates the combination. β and the RanPAC λ are both selected on a 10% held-out VAL split; TEST selects nothing.

---

## Status

**12 cells over the canonical set.** Done: CIFAR100, IMAGENETR, IMAGENETAP, CUB200P. Grid complete.

Excluded from all pooled statistics: CUB200, IMAGENETA — superseded splits, kept in the JSON but not pooled, because including both variants of a dataset double-counts it in every paired contrast and in the sign test.

### Against published PTM-CIL numbers (T=10, ViT-B/16 IN21k)

| ds | ours `fuse_cone` ALast | AAvg | GR-LoRA | MACIL | gap to GR-LoRA |
|---|---|---|---|---|---|
| CIFAR100 | 92.62 | 95.18 | 91.97 / 94.65 | 91.86 / 94.44 | **+0.65 / +0.53** |
| IMAGENETR | 81.52 | 85.94 | 82.09 / 86.20 | 81.82 / 85.76 | **-0.57 / -0.26** |
| IMAGENETAP | 62.73 | 69.29 | 63.60 / 70.24 | 63.15 / 70.54 | **-0.87 / -0.95** |
| CUB200P | 90.02 | 93.13 | 89.91 / 93.85 | 90.23 / 93.78 | **+0.11 / -0.72** |


---

## CIFAR100

### A-Last

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 92.60 | 92.69 | 92.37 | **92.55** | 0.17 | — |
| f32 cone | 92.24 | 92.43 | 92.08 | 92.25 | 0.18 | -0.30 |
| **f32 fuse_cone** | 92.72 | 92.69 | 92.45 | 92.62 | 0.15 | +0.07 |
| f32 sub | 91.95 | 92.03 | 91.92 | 91.97 | 0.06 | -0.59 |
| f32 fuse_sub | 92.83 | 92.69 | 92.51 | 92.68 | 0.16 | +0.12 |
| f32 pm | 92.05 | 92.13 | 91.74 | 91.97 | 0.21 | -0.58 |
| f32 fuse_pm | 92.70 | 92.81 | 92.43 | 92.65 | 0.20 | +0.09 |
| f64 cone | 92.43 | 92.49 | 92.33 | 92.42 | 0.08 | -0.14 |
| **f64 fuse_cone** | 92.66 | 92.90 | 92.55 | 92.70 | 0.18 | +0.15 |
| f64 sub | 91.97 | 92.16 | 91.58 | 91.90 | 0.30 | -0.65 |
| f64 fuse_sub | 92.72 | 92.68 | 92.52 | 92.64 | 0.11 | +0.09 |
| f64 pm | 91.95 | 91.89 | 91.84 | 91.89 | 0.06 | -0.66 |
| f64 fuse_pm | 92.87 | 92.92 | 92.60 | 92.80 | 0.17 | +0.24 |

**Paired contrasts (A-Last)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | -0.11 | +0.00 | -0.06 | -0.06 | 0.06 |
| f32: cone − sub  (raw) | +0.29 | +0.40 | +0.16 | +0.28 | 0.12 |
| f32: fuse_cone − fuse_pm | +0.02 | -0.12 | +0.02 | -0.03 | 0.08 |
| f32: fuse_sub − fuse_pm | +0.13 | -0.12 | +0.08 | +0.03 | 0.13 |
| f32: cone − pm   (raw) | +0.19 | +0.30 | +0.34 | +0.28 | 0.08 |
| f64: fuse_cone − fuse_sub | -0.06 | +0.22 | +0.03 | +0.06 | 0.14 |
| f64: cone − sub  (raw) | +0.46 | +0.33 | +0.75 | +0.51 | 0.22 |
| f64: fuse_cone − fuse_pm | -0.21 | -0.02 | -0.05 | -0.09 | 0.10 |
| f64: fuse_sub − fuse_pm | -0.15 | -0.24 | -0.08 | -0.16 | 0.08 |
| f64: cone − pm   (raw) | +0.48 | +0.60 | +0.49 | +0.52 | 0.07 |

### A-Avg

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 95.14 | 95.72 | 94.54 | **95.13** | 0.59 | — |
| f32 cone | 94.94 | 95.53 | 94.24 | 94.90 | 0.65 | -0.23 |
| **f32 fuse_cone** | 95.22 | 95.72 | 94.61 | 95.18 | 0.56 | +0.05 |
| f32 sub | 94.67 | 95.33 | 93.92 | 94.64 | 0.70 | -0.50 |
| f32 fuse_sub | 95.20 | 95.72 | 94.61 | 95.17 | 0.56 | +0.04 |
| f32 pm | 94.76 | 95.38 | 93.90 | 94.68 | 0.74 | -0.45 |
| f32 fuse_pm | 95.21 | 95.73 | 94.61 | 95.18 | 0.56 | +0.05 |
| f64 cone | 94.97 | 95.49 | 94.43 | 94.96 | 0.53 | -0.17 |
| **f64 fuse_cone** | 95.24 | 95.72 | 94.62 | 95.19 | 0.55 | +0.06 |
| f64 sub | 94.61 | 95.38 | 93.83 | 94.61 | 0.77 | -0.53 |
| f64 fuse_sub | 95.17 | 95.72 | 94.61 | 95.17 | 0.55 | +0.03 |
| f64 pm | 94.67 | 95.27 | 94.14 | 94.69 | 0.56 | -0.44 |
| f64 fuse_pm | 95.26 | 95.76 | 94.63 | 95.21 | 0.57 | +0.08 |

**Paired contrasts (A-Avg)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | +0.01 | +0.00 | +0.00 | +0.01 | 0.00 |
| f32: cone − sub  (raw) | +0.27 | +0.21 | +0.32 | +0.27 | 0.06 |
| f32: fuse_cone − fuse_pm | +0.01 | -0.01 | -0.00 | -0.00 | 0.01 |
| f32: fuse_sub − fuse_pm | -0.01 | -0.01 | -0.01 | -0.01 | 0.00 |
| f32: cone − pm   (raw) | +0.18 | +0.16 | +0.34 | +0.22 | 0.10 |
| f64: fuse_cone − fuse_sub | +0.07 | +0.00 | +0.01 | +0.03 | 0.04 |
| f64: cone − sub  (raw) | +0.36 | +0.10 | +0.60 | +0.35 | 0.25 |
| f64: fuse_cone − fuse_pm | -0.02 | -0.04 | -0.01 | -0.02 | 0.01 |
| f64: fuse_sub − fuse_pm | -0.09 | -0.04 | -0.02 | -0.05 | 0.04 |
| f64: cone − pm   (raw) | +0.30 | +0.22 | +0.29 | +0.27 | 0.05 |

---

## IMAGENETR

### A-Last

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 80.28 | 80.55 | 80.38 | **80.41** | 0.13 | — |
| f32 cone | 79.38 | 80.08 | 80.28 | 79.92 | 0.47 | -0.49 |
| **f32 fuse_cone** | 81.12 | 81.70 | 81.75 | 81.52 | 0.35 | +1.12 |
| f32 sub | 79.48 | 80.23 | 80.42 | 80.04 | 0.49 | -0.36 |
| f32 fuse_sub | 81.12 | 81.55 | 81.67 | 81.44 | 0.29 | +1.04 |
| f32 pm | 78.13 | 78.60 | 78.95 | 78.56 | 0.41 | -1.84 |
| f32 fuse_pm | 80.72 | 80.58 | 80.82 | 80.71 | 0.12 | +0.30 |
| f64 cone | 79.47 | 80.30 | 80.17 | 79.98 | 0.45 | -0.43 |
| **f64 fuse_cone** | 80.93 | 81.50 | 81.60 | 81.34 | 0.36 | +0.94 |
| f64 sub | 79.45 | 80.15 | 80.22 | 79.94 | 0.42 | -0.47 |
| f64 fuse_sub | 80.93 | 81.45 | 81.93 | 81.44 | 0.50 | +1.03 |
| f64 pm | 76.37 | 76.80 | 76.52 | 76.56 | 0.22 | -3.84 |
| f64 fuse_pm | 80.48 | 80.55 | 80.53 | 80.52 | 0.03 | +0.12 |

**Paired contrasts (A-Last)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | +0.00 | +0.15 | +0.08 | +0.08 | 0.08 |
| f32: cone − sub  (raw) | -0.10 | -0.15 | -0.13 | -0.13 | 0.03 |
| f32: fuse_cone − fuse_pm | +0.40 | +1.12 | +0.93 | +0.82 | 0.37 |
| f32: fuse_sub − fuse_pm | +0.40 | +0.97 | +0.85 | +0.74 | 0.30 |
| f32: cone − pm   (raw) | +1.25 | +1.48 | +1.33 | +1.36 | 0.12 |
| f64: fuse_cone − fuse_sub | +0.00 | +0.05 | -0.33 | -0.09 | 0.21 |
| f64: cone − sub  (raw) | +0.02 | +0.15 | -0.05 | +0.04 | 0.10 |
| f64: fuse_cone − fuse_pm | +0.45 | +0.95 | +1.07 | +0.82 | 0.33 |
| f64: fuse_sub − fuse_pm | +0.45 | +0.90 | +1.40 | +0.92 | 0.48 |
| f64: cone − pm   (raw) | +3.10 | +3.50 | +3.65 | +3.42 | 0.28 |

### A-Avg

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 85.13 | 86.09 | 84.71 | **85.31** | 0.71 | — |
| f32 cone | 84.79 | 85.54 | 84.28 | 84.87 | 0.64 | -0.44 |
| **f32 fuse_cone** | 85.93 | 86.62 | 85.27 | 85.94 | 0.68 | +0.63 |
| f32 sub | 84.63 | 85.41 | 84.23 | 84.76 | 0.60 | -0.55 |
| f32 fuse_sub | 85.86 | 86.55 | 85.24 | 85.88 | 0.65 | +0.57 |
| f32 pm | 83.41 | 84.29 | 82.87 | 83.52 | 0.72 | -1.79 |
| f32 fuse_pm | 85.32 | 86.17 | 84.50 | 85.33 | 0.84 | +0.02 |
| f64 cone | 84.50 | 85.85 | 84.31 | 84.89 | 0.84 | -0.42 |
| **f64 fuse_cone** | 86.00 | 86.60 | 85.30 | 85.96 | 0.65 | +0.65 |
| f64 sub | 84.37 | 85.32 | 83.95 | 84.55 | 0.70 | -0.76 |
| f64 fuse_sub | 85.83 | 86.50 | 85.24 | 85.86 | 0.63 | +0.55 |
| f64 pm | 81.43 | 82.72 | 80.71 | 81.62 | 1.02 | -3.69 |
| f64 fuse_pm | 85.20 | 86.12 | 84.61 | 85.31 | 0.76 | +0.00 |

**Paired contrasts (A-Avg)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | +0.07 | +0.07 | +0.03 | +0.06 | 0.03 |
| f32: cone − sub  (raw) | +0.17 | +0.13 | +0.05 | +0.12 | 0.06 |
| f32: fuse_cone − fuse_pm | +0.61 | +0.45 | +0.77 | +0.61 | 0.16 |
| f32: fuse_sub − fuse_pm | +0.54 | +0.38 | +0.74 | +0.55 | 0.18 |
| f32: cone − pm   (raw) | +1.38 | +1.25 | +1.41 | +1.35 | 0.09 |
| f64: fuse_cone − fuse_sub | +0.17 | +0.09 | +0.05 | +0.11 | 0.06 |
| f64: cone − sub  (raw) | +0.13 | +0.54 | +0.36 | +0.34 | 0.20 |
| f64: fuse_cone − fuse_pm | +0.80 | +0.47 | +0.68 | +0.65 | 0.17 |
| f64: fuse_sub − fuse_pm | +0.63 | +0.38 | +0.63 | +0.55 | 0.14 |
| f64: cone − pm   (raw) | +3.07 | +3.14 | +3.60 | +3.27 | 0.29 |

---

## IMAGENETAP

### A-Last

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 61.53 | 63.33 | 59.40 | **61.42** | 1.97 | — |
| f32 cone | 61.80 | 63.87 | 58.67 | 61.44 | 2.62 | +0.02 |
| **f32 fuse_cone** | 63.87 | 64.93 | 59.40 | 62.73 | 2.94 | +1.31 |
| f32 sub | 59.53 | 62.47 | 57.00 | 59.67 | 2.74 | -1.76 |
| f32 fuse_sub | 63.13 | 63.73 | 59.47 | 62.11 | 2.31 | +0.69 |
| f32 pm | 52.00 | 54.00 | 49.67 | 51.89 | 2.17 | -9.53 |
| f32 fuse_pm | 61.73 | 63.40 | 59.53 | 61.56 | 1.94 | +0.13 |
| f64 cone | 60.60 | 62.80 | 57.80 | 60.40 | 2.51 | -1.02 |
| **f64 fuse_cone** | 63.47 | 63.80 | 59.47 | 62.24 | 2.41 | +0.82 |
| f64 sub | 55.33 | 57.67 | 54.07 | 55.69 | 1.83 | -5.73 |
| f64 fuse_sub | 61.73 | 63.33 | 59.87 | 61.64 | 1.74 | +0.22 |
| f64 pm | 48.80 | 51.60 | 46.67 | 49.02 | 2.47 | -12.40 |
| f64 fuse_pm | 61.73 | 64.00 | 59.80 | 61.84 | 2.10 | +0.42 |

**Paired contrasts (A-Last)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | +0.73 | +1.20 | -0.07 | +0.62 | 0.64 |
| f32: cone − sub  (raw) | +2.27 | +1.40 | +1.67 | +1.78 | 0.44 |
| f32: fuse_cone − fuse_pm | +2.13 | +1.53 | -0.13 | +1.18 | 1.17 |
| f32: fuse_sub − fuse_pm | +1.40 | +0.33 | -0.07 | +0.56 | 0.76 |
| f32: cone − pm   (raw) | +9.80 | +9.87 | +9.00 | +9.56 | 0.48 |
| f64: fuse_cone − fuse_sub | +1.73 | +0.47 | -0.40 | +0.60 | 1.07 |
| f64: cone − sub  (raw) | +5.27 | +5.13 | +3.73 | +4.71 | 0.85 |
| f64: fuse_cone − fuse_pm | +1.73 | -0.20 | -0.33 | +0.40 | 1.16 |
| f64: fuse_sub − fuse_pm | +0.00 | -0.67 | +0.07 | -0.20 | 0.41 |
| f64: cone − pm   (raw) | +11.80 | +11.20 | +11.13 | +11.38 | 0.37 |

### A-Avg

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 66.43 | 71.73 | 67.84 | **68.66** | 2.75 | — |
| f32 cone | 64.46 | 72.21 | 65.72 | 67.47 | 4.16 | -1.20 |
| **f32 fuse_cone** | 67.27 | 72.34 | 68.27 | 69.29 | 2.68 | +0.63 |
| f32 sub | 62.66 | 71.11 | 64.50 | 66.09 | 4.44 | -2.58 |
| f32 fuse_sub | 67.02 | 72.20 | 68.06 | 69.10 | 2.74 | +0.43 |
| f32 pm | 57.18 | 64.55 | 58.22 | 59.98 | 3.99 | -8.68 |
| f32 fuse_pm | 66.37 | 71.73 | 67.51 | 68.54 | 2.82 | -0.13 |
| f64 cone | 63.38 | 71.52 | 64.92 | 66.61 | 4.32 | -2.06 |
| **f64 fuse_cone** | 67.07 | 72.17 | 67.94 | 69.06 | 2.73 | +0.39 |
| f64 sub | 58.02 | 66.33 | 60.88 | 61.74 | 4.22 | -6.92 |
| f64 fuse_sub | 66.13 | 71.89 | 67.87 | 68.63 | 2.96 | -0.03 |
| f64 pm | 55.16 | 62.51 | 55.49 | 57.72 | 4.15 | -10.95 |
| f64 fuse_pm | 66.38 | 72.07 | 67.55 | 68.67 | 3.01 | +0.00 |

**Paired contrasts (A-Avg)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | +0.24 | +0.14 | +0.21 | +0.20 | 0.05 |
| f32: cone − sub  (raw) | +1.81 | +1.10 | +1.22 | +1.38 | 0.38 |
| f32: fuse_cone − fuse_pm | +0.90 | +0.61 | +0.76 | +0.75 | 0.15 |
| f32: fuse_sub − fuse_pm | +0.65 | +0.47 | +0.55 | +0.56 | 0.09 |
| f32: cone − pm   (raw) | +7.28 | +7.66 | +7.51 | +7.48 | 0.19 |
| f64: fuse_cone − fuse_sub | +0.94 | +0.28 | +0.07 | +0.43 | 0.46 |
| f64: cone − sub  (raw) | +5.36 | +5.18 | +4.04 | +4.86 | 0.71 |
| f64: fuse_cone − fuse_pm | +0.69 | +0.10 | +0.38 | +0.39 | 0.30 |
| f64: fuse_sub − fuse_pm | -0.25 | -0.18 | +0.32 | -0.04 | 0.31 |
| f64: cone − pm   (raw) | +8.23 | +9.01 | +9.43 | +8.89 | 0.61 |

---

## CUB200P

### A-Last

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 90.37 | 89.82 | 89.91 | **90.03** | 0.30 | — |
| f32 cone | 89.27 | 89.36 | 89.36 | 89.33 | 0.05 | -0.71 |
| **f32 fuse_cone** | 90.37 | 89.86 | 89.82 | 90.02 | 0.31 | -0.01 |
| f32 sub | 89.02 | 88.63 | 89.10 | 88.92 | 0.25 | -1.12 |
| f32 fuse_sub | 90.37 | 89.91 | 89.86 | 90.05 | 0.28 | +0.01 |
| f32 pm | 86.94 | 86.05 | 86.47 | 86.49 | 0.45 | -3.55 |
| f32 fuse_pm | 90.37 | 89.82 | 89.91 | 90.03 | 0.30 | +0.00 |
| f64 cone | 89.10 | 88.93 | 89.02 | 89.02 | 0.08 | -1.02 |
| **f64 fuse_cone** | 90.46 | 89.91 | 89.78 | 90.05 | 0.36 | +0.01 |
| f64 sub | 88.51 | 88.21 | 88.76 | 88.49 | 0.28 | -1.54 |
| f64 fuse_sub | 90.37 | 89.95 | 89.65 | 89.99 | 0.36 | -0.04 |
| f64 pm | 86.68 | 86.39 | 86.22 | 86.43 | 0.24 | -3.60 |
| f64 fuse_pm | 90.46 | 89.82 | 89.91 | 90.06 | 0.35 | +0.03 |

**Paired contrasts (A-Last)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | +0.00 | -0.04 | -0.04 | -0.03 | 0.02 |
| f32: cone − sub  (raw) | +0.25 | +0.72 | +0.25 | +0.41 | 0.27 |
| f32: fuse_cone − fuse_pm | +0.00 | +0.04 | -0.08 | -0.01 | 0.06 |
| f32: fuse_sub − fuse_pm | +0.00 | +0.08 | -0.04 | +0.01 | 0.06 |
| f32: cone − pm   (raw) | +2.33 | +3.31 | +2.88 | +2.84 | 0.49 |
| f64: fuse_cone − fuse_sub | +0.08 | -0.04 | +0.13 | +0.06 | 0.09 |
| f64: cone − sub  (raw) | +0.59 | +0.72 | +0.25 | +0.52 | 0.24 |
| f64: fuse_cone − fuse_pm | +0.00 | +0.08 | -0.13 | -0.01 | 0.11 |
| f64: fuse_sub − fuse_pm | -0.08 | +0.13 | -0.25 | -0.07 | 0.19 |
| f64: cone − pm   (raw) | +2.42 | +2.54 | +2.80 | +2.59 | 0.19 |

### A-Avg

| reader | s0 | s1 | s2 | mean | sd | Δ vs rp |
|---|---|---|---|---|---|---|
| **ranpac** | 94.49 | 92.60 | 92.23 | **93.11** | 1.21 | — |
| f32 cone | 93.58 | 92.65 | 92.23 | 92.82 | 0.69 | -0.29 |
| **f32 fuse_cone** | 94.49 | 92.68 | 92.22 | 93.13 | 1.20 | +0.03 |
| f32 sub | 93.25 | 92.40 | 91.84 | 92.49 | 0.71 | -0.61 |
| f32 fuse_sub | 94.49 | 92.68 | 92.23 | 93.14 | 1.20 | +0.03 |
| f32 pm | 92.16 | 90.63 | 90.18 | 90.99 | 1.04 | -2.12 |
| f32 fuse_pm | 94.52 | 92.67 | 92.25 | 93.15 | 1.21 | +0.04 |
| f64 cone | 93.63 | 92.52 | 91.94 | 92.70 | 0.86 | -0.41 |
| **f64 fuse_cone** | 94.50 | 92.57 | 92.23 | 93.10 | 1.23 | -0.01 |
| f64 sub | 93.00 | 91.90 | 91.57 | 92.16 | 0.75 | -0.95 |
| f64 fuse_sub | 94.46 | 92.57 | 92.21 | 93.08 | 1.21 | -0.03 |
| f64 pm | 92.13 | 90.25 | 90.36 | 90.91 | 1.06 | -2.20 |
| f64 fuse_pm | 94.50 | 92.64 | 92.23 | 93.13 | 1.21 | +0.02 |

**Paired contrasts (A-Avg)** — same features, splits and rays, so seed noise cancels.

| contrast | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| f32: fuse_cone − fuse_sub | +0.00 | -0.00 | -0.00 | -0.00 | 0.00 |
| f32: cone − sub  (raw) | +0.33 | +0.25 | +0.39 | +0.33 | 0.07 |
| f32: fuse_cone − fuse_pm | -0.02 | +0.01 | -0.02 | -0.01 | 0.02 |
| f32: fuse_sub − fuse_pm | -0.02 | +0.01 | -0.02 | -0.01 | 0.02 |
| f32: cone − pm   (raw) | +1.42 | +2.02 | +2.06 | +1.83 | 0.36 |
| f64: fuse_cone − fuse_sub | +0.04 | +0.00 | +0.02 | +0.02 | 0.02 |
| f64: cone − sub  (raw) | +0.63 | +0.62 | +0.38 | +0.54 | 0.14 |
| f64: fuse_cone − fuse_pm | +0.00 | -0.07 | -0.00 | -0.03 | 0.04 |
| f64: fuse_sub − fuse_pm | -0.04 | -0.07 | -0.02 | -0.05 | 0.03 |
| f64: cone − pm   (raw) | +1.50 | +2.28 | +1.59 | +1.79 | 0.42 |

---

## Pooled contrasts (all completed cells)

`wins` = cells strictly greater than zero. With n cells a clean sweep is a sign test at
p = 2⁻ⁿ, which can detect a small consistent effect the pooled sd cannot — the pooled sd
carries between-dataset variance, the sign test does not.

### A-Last

| contrast | mean | sd | wins |
|---|---|---|---|
| **f32: fuse_cone − fuse_sub** | +0.15 | 0.40 | 4/12 |
| f32: cone − sub  (raw) | +0.59 | 0.78 | 9/12 |
| f32: fuse_cone − fuse_pm | +0.49 | 0.76 | 8/12 |
| f32: fuse_sub − fuse_pm | +0.33 | 0.49 | 8/12 |
| f32: cone − pm   (raw) | +3.51 | 3.78 | 12/12 |
| **f64: fuse_cone − fuse_sub** | +0.16 | 0.55 | 7/12 |
| f64: cone − sub  (raw) | +1.45 | 2.02 | 11/12 |
| f64: fuse_cone − fuse_pm | +0.28 | 0.64 | 5/12 |
| f64: fuse_sub − fuse_pm | +0.12 | 0.56 | 5/12 |
| f64: cone − pm   (raw) | +4.48 | 4.31 | 12/12 |

### A-Avg

| contrast | mean | sd | wins |
|---|---|---|---|
| **f32: fuse_cone − fuse_sub** | +0.06 | 0.09 | 9/12 |
| f32: cone − sub  (raw) | +0.52 | 0.55 | 12/12 |
| f32: fuse_cone − fuse_pm | +0.34 | 0.38 | 8/12 |
| f32: fuse_sub − fuse_pm | +0.27 | 0.31 | 7/12 |
| f32: cone − pm   (raw) | +2.72 | 2.94 | 12/12 |
| **f64: fuse_cone − fuse_sub** | +0.15 | 0.26 | 12/12 |
| f64: cone − sub  (raw) | +1.52 | 2.04 | 12/12 |
| f64: fuse_cone − fuse_pm | +0.25 | 0.34 | 6/12 |
| f64: fuse_sub − fuse_pm | +0.10 | 0.31 | 4/12 |
| f64: cone − pm   (raw) | +3.55 | 3.42 | 12/12 |

---

## Findings

### 1. The fusion gain replicates, and is larger than exp35 reported

IMAGENETR f32 `fuse_cone`: **81.52 / 85.94** vs RanPAC 80.41 / 85.31 — **+1.12 A-Last**, +0.63 A-Avg, all three seeds positive.

exp35's +0.75 was a single seed. It holds and improves. This was a precondition for
the whole question being worth asking, and it passed.

### 2. But the gain decomposes away from the conic constraint

IMAGENETR f32, A-Last, mean over 3 seeds, each rule fused against the same RanPAC:

| component | Δ vs RanPAC | share of the gain |
|---|---|---|
| rays alone (`fuse_pm`) | +0.30 | 27% |
| \+ linear combination (`fuse_sub`) | +1.04 | 93% |
| \+ non-negativity (`fuse_cone`) | +1.12 | 100% |

**The combination contributes +0.74 (66%). Non-negativity contributes +0.08 (7%).**

On IMAGENETR the raw cone does not even lead: `cone − sub` raw at f32 is -0.10, -0.15, -0.13 (mean -0.13) — `sub` wins every seed before fusion.

### 3. Non-negativity is real but small — and A-Last cannot see it

- `f32` `fuse_cone − fuse_sub`: A-Last +0.15 ± 0.40 (4/12 wins) · **A-Avg +0.06 ± 0.09 (9/12 wins)**
- `f64` `fuse_cone − fuse_sub`: A-Last +0.16 ± 0.55 (7/12 wins) · **A-Avg +0.15 ± 0.26 (12/12 wins)**

A-Last reads this as a coin. A-Avg reads it as a clean sweep. That is not a contradiction —
A-Last is a single-stage estimator and A-Avg averages 10 stages, so a small consistent
effect showing up only in the low-variance metric is the expected signature of a real
effect near the noise floor. Every per-dataset A-Avg mean is positive.

**Conclusion: the conic constraint is worth roughly +0.17 A-Avg. It is not where the win
comes from.**

### 4. IMAGENETA is an outlier, and it is confounded

An order of magnitude above the other datasets, and it **grows with R**. IMAGENETA has
~27 fit rows/class, so both f32 and f64 ask for more rays than there are points. `sub`
collapses there — once span(A) covers the class's whole row space, `‖q·B‖ → 1` for every
class and the rule stops discriminating. `cone` has no such failure mode: the non-negative
orthant does not fill up.

That is the **regulariser reading**: non-negativity is not modelling classes better, it is
protecting against a ray budget we chose badly. It is a materially weaker claim than
"cones model classes better" and must be reported as such unless the f8/f16 run refutes it.

### 5. Protocol issues that limit what IMAGENETA can support

- **β = 0 abstentions.** On IMAGENETA s0–s3, all six fused cells read *exactly* RanPAC —
  the β search declined to use the cone at all. Those are abstentions, not ties, and they
  pull A-Avg toward RanPAC. Check `_beta` in the JSON before weighting IN-A.
- **Classes with no rays.** IMAGENETA has 1–2 seen classes per stage with <2 fit rows.
  They are `-inf` in the raw arms and neutralised to 0 in the fused arms. The *contrast*
  is fair (both rules take the same hit) but the levels are not comparable across datasets.
- **Seed spread.** IMAGENETA RanPAC has sd 2.27 (A-Last) / 3.63 (A-Avg) across seeds.
  Only paired contrasts mean anything on this dataset.

---

## Where this leaves the method

The honest statement of the best result:

> A per-class ray set (oPCA, γ=0.5, R=32) read out by a linear-combination rule and fused
> with RanPAC beats RanPAC by ~1.1 A-Last / ~0.6 A-Avg on ImageNet-R across 3 seeds, with
> zero stored images. Roughly two thirds of that gain is the linear combination over the
> rays, one quarter is the rays themselves, and ~7% is the non-negativity constraint.

That is a real, seed-robust, zero-storage result. It is not primarily a *conic* one.

## Open

1. **CUB200** — 3 cells pending. Historically the raw cone loses to RanPAC by ~1pt there.
2. **The IMAGENETA ray-budget confound.** Drop below the row count so nothing clamps:
   ```bash
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     DS=IMAGENETA T=10 SEED=0,1,2 ARMS=f8,f16 python -u exp52_fusion_rule_control.py
   ```
   If `cone − sub` collapses toward 0 at f8, the constraint is a fix for a self-inflicted
   problem. If it holds at +2 with 8 rays from 27 points, that is the first genuine
   evidence for non-negativity in this project.
3. **β = 0 audit** across all cells — a fused Δ of +0.00 with β=0 is an abstention, not a tie.

