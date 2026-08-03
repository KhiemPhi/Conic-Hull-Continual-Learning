"""
exp2_cone_diagnostic.py — CONE GAIN AS A SCIENTIFIC INSTRUMENT, NOT A CLASSIFIER.

Your most robust, replicated positive was never an accuracy claim. It was the dose-response:
    cone_acc - ncm_acc  grows monotonically with intra-class MULTIMODALITY.
So stop using the cone to classify and start using it to MEASURE. A class whose cone beats
its prototype has hidden modes => hidden subpopulations, label noise, or annotation drift.

"Conic gain" as a per-class statistic:
    g(c) = mean in-class membership under a K-ray hull  -  membership under a 1-ray hull
           (both fit on the SAME data; the only difference is capacity for multiple modes)
Computed WITHOUT any subgroup labels.

Three arms:
  A. CALIBRATION  — inject KNOWN multimodality (merge k fine classes into one label,
                    k = 1,2,3,5,10) and check g tracks k monotonically. Establishes the
                    instrument reads what it claims.
  B. LABEL NOISE  — inject p% wrong-label samples into a class (p = 0,5,10,20,40) and check
                    g rises. Detecting mislabeled subgroups is a real, useful application.
  C. REAL SUBGROUPS — Waterbirds: every class genuinely splits by background attribute a
                    (landbird/waterbird x land/water background). Does g rank the classes and
                    the SUBGROUP-IMBALANCED ones correctly, with no access to a?
                    Report AUROC of g for detecting "this class has a hidden subgroup".

Run:  python -u exp2_cone_diagnostic.py
"""
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from conic_hull import ConicHull

SEED = 0
np.random.seed(SEED)
K_RAYS = int(os.environ.get("K_RAYS", 10))
N_SUB = int(os.environ.get("N_SUB", 250))       # samples per synthetic class


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def conic_gain(X, K=K_RAYS, holdout=0.3, seed=0):
    """g = held-out membership under a K-ray hull minus under a 1-ray hull (the prototype).
    Positive => the class needs more than one direction to explain itself => multimodal."""
    X = un(X); n = len(X)
    if n < 8:
        return np.nan
    r = np.random.default_rng(seed).permutation(n)
    nh = max(int(holdout * n), 2)
    te, tr = X[r[:nh]], X[r[nh:]]
    k = int(min(K, max(len(tr) - 1, 1)))
    h1 = ConicHull(n_rays=1, use_pca=False).fit(tr)
    hK = ConicHull(n_rays=k, use_pca=True,
                   pca_dim=int(min(64, max(len(tr) - 1, 2)))).fit(tr)
    return float(hK.score_all(te)["geo_residual"].mean()
                 - h1.score_all(te)["geo_residual"].mean())


def multi_gain(X, K=K_RAYS, holdout=0.3, seed=0):
    """Control: same capacity increase but with k-means centroids instead of rays.
    If multi_gain tracks structure just as well, the CONE is not doing the work."""
    X = un(X); n = len(X)
    if n < 8:
        return np.nan
    r = np.random.default_rng(seed).permutation(n)
    nh = max(int(holdout * n), 2)
    te, tr = X[r[:nh]], X[r[nh:]]
    k = int(min(K, max(len(tr) // 4, 1)))
    mu = un(tr.mean(0, keepdims=True))
    cc = un(KMeans(n_clusters=k, n_init=4, random_state=0).fit(tr).cluster_centers_)
    return float(np.max(te @ cc.T, 1).mean() - (te @ mu.T).max(1).mean())


# ============================ A. calibration on known multimodality ============================
z = np.load("ranpac_out/cifar100_feats.npz")
X, y = z["ftr"], z["ytr"]
print("=== A. CALIBRATION: does conic gain track injected multimodality? ===")
rng = np.random.default_rng(SEED)
rowsA = []
for k in [1, 2, 3, 5, 10]:
    gs, ms = [], []
    for rep in range(12):
        cs = rng.choice(100, k, replace=False)
        pool = np.concatenate([X[y == c][:max(N_SUB // k, 5)] for c in cs])
        gs.append(conic_gain(pool, seed=rep)); ms.append(multi_gain(pool, seed=rep))
    rowsA.append((k, float(np.nanmean(gs)), float(np.nanstd(gs)), float(np.nanmean(ms))))
    print(f"  merge k={k:>2}  conic_gain {np.nanmean(gs):+.4f} +- {np.nanstd(gs):.4f}  |  "
          f"multi_gain {np.nanmean(ms):+.4f}", flush=True)

# ============================ B. label noise ============================
print("\n=== B. LABEL NOISE: does conic gain rise with contamination? ===")
rowsB = []
for p in [0.0, 0.05, 0.10, 0.20, 0.40]:
    gs, ms = [], []
    for rep in range(12):
        c = int(rng.integers(0, 100))
        clean = X[y == c][:N_SUB]
        nn_ = int(p * len(clean))
        other = X[y != c]
        pool = np.concatenate([clean[:len(clean) - nn_],
                               other[rng.choice(len(other), nn_, replace=False)]]) \
            if nn_ else clean
        gs.append(conic_gain(pool, seed=rep)); ms.append(multi_gain(pool, seed=rep))
    rowsB.append((p, float(np.nanmean(gs)), float(np.nanmean(ms))))
    print(f"  noise {p:>4.0%}  conic_gain {np.nanmean(gs):+.4f}  |  "
          f"multi_gain {np.nanmean(ms):+.4f}", flush=True)

# ============================ C. real hidden subgroups (Waterbirds) ============================
print("\n=== C. REAL SUBGROUPS: Waterbirds (attribute a = background) ===")
w = np.load("waterbirds_out/waterbirds_clip.npz")
Xw, yw, aw = w["train_f"], w["train_y"], w["train_a"]
print(f"  {Xw.shape} | classes {np.unique(yw)} | attributes {np.unique(aw)}")
# build many pseudo-classes: for each (class, mixing ratio) draw a pool whose hidden subgroup
# balance varies. Ground truth "has hidden subgroup" = the pool mixes both attributes.
pools, labels, ratios = [], [], []
for c in np.unique(yw):
    A = Xw[(yw == c) & (aw == 0)]
    B = Xw[(yw == c) & (aw == 1)]
    if len(A) < 40 or len(B) < 40:
        print(f"  class {c}: too few in one subgroup ({len(A)},{len(B)}) — skipping")
        continue
    for ratio in [0.0, 0.1, 0.25, 0.5]:
        for rep in range(8):
            n = min(len(A), len(B), N_SUB)
            nb = int(ratio * n)
            pool = np.concatenate([A[rng.choice(len(A), n - nb, replace=False)],
                                   B[rng.choice(len(B), nb, replace=False)]]) if nb else \
                A[rng.choice(len(A), n, replace=False)]
            pools.append(pool); labels.append(1 if ratio >= 0.25 else 0); ratios.append(ratio)
gC = np.array([conic_gain(p, seed=i) for i, p in enumerate(pools)])
mC = np.array([multi_gain(p, seed=i) for i, p in enumerate(pools)])
labels = np.array(labels); ratios = np.array(ratios)
ok = ~np.isnan(gC)
auc_cone = roc_auc_score(labels[ok], gC[ok])
auc_multi = roc_auc_score(labels[ok], mC[ok])
print(f"  pools {len(pools)} | AUROC detect-hidden-subgroup: "
      f"CONE {auc_cone:.4f} vs MULTIPROTO {auc_multi:.4f}")
for r in sorted(set(ratios.tolist())):
    m = ratios == r
    print(f"    minority ratio {r:>4.0%}: conic_gain {np.nanmean(gC[m]):+.4f} | "
          f"multi_gain {np.nanmean(mC[m]):+.4f}")

np.save("exp2_results.npy", dict(A=rowsA, B=rowsB,
                                 C=dict(auc_cone=auc_cone, auc_multi=auc_multi,
                                        g=gC, m=mC, ratio=ratios, lab=labels)),
        allow_pickle=True)
print("\n" + "=" * 92)
print("EXP2 — conic gain as a diagnostic instrument")
print("=" * 92)
print("A. monotone in injected multimodality?  ", [f"k={k}:{g:+.3f}" for k, g, _, _ in rowsA])
print("B. rises with label noise?              ", [f"{p:.0%}:{g:+.3f}" for p, g, _ in rowsB])
print(f"C. AUROC hidden-subgroup detection: cone {auc_cone:.4f} | multiproto {auc_multi:.4f}")
print("-" * 92)
print("WIN CONDITION: conic_gain is monotone in A and B, AND beats multi_gain in C.")
print("If multi_gain matches it everywhere, the INSTRUMENT works but is not cone-specific")
print("(same control that falsified the RanPAC-projection win) — report it as such.")
print("=" * 92)
