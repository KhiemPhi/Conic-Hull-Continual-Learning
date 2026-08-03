"""
exp3b_order_baselines.py — IS THE SEMANTIC PARTIAL ORDER CONE-SPECIFIC, OR JUST "REGIONS"?

exp3 found an unsupervised subsumption order in CLIP space read by conic containment:
    fine -> OWN coarse  0.9017  (at the p10 ceiling)
    fine -> UNRELATED   0.2116  (size/capacity/direction matched)
    => +0.69 semantic gap.
But exp3 had NO non-conic baseline — and that is precisely the control that falsified every
previous cone positive in this project (random datasamples tied extreme rays as a RanPAC
projection; k-means centroids beat rays as classifiers; herding beat rays as a coreset).

This script holds EVERYTHING fixed — same hierarchy, same held-out split, same tau = p10 of
each region's OWN members, same containment definition — and swaps ONLY the region primitive:

  ball        centroid + calibrated cosine radius        (1 direction)
  multiproto  K k-means centroids, max cosine            (K directions)   <-- key control
  subspace    class PCA to K dims, reconstruction cosine (K dims)
  cone        conic hull, K rays, NNLS geo_residual      (K rays)         <-- the claim

Two metrics, one thresholded and one threshold-free:

  1. SEMANTIC GAP = containment(fine -> own coarse) - containment(fine -> unrelated coarse)
     Both targets are built identically, so this is size- and capacity-matched.
  2. PARENT AUROC (threshold-free, removes the tau choice entirely): over all 100x20
     (fine, coarse) pairs, can the raw containment score pick out the TRUE parent?
     100 positives vs 1900 negatives. This is the cleanest single number.

VERDICT: cone >> ball/multiproto/subspace  => a genuine cone-specific finding (the first).
         all tie                            => "region models reveal an unsupervised semantic
                                                partial order" — still novel, not cone-specific.

Run:  python -u exp3b_order_baselines.py
      K_RAYS=40 PCT=5 python -u exp3b_order_baselines.py
"""
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from conic_hull import ConicHull

SEED = 0
np.random.seed(SEED)
FEATS = os.environ.get("FEATS", "clip")
K = int(os.environ.get("K_RAYS", 20))
PCT = float(os.environ.get("PCT", 10))
HOLD = 0.3
ONLY = [p for p in os.environ.get("ONLY", "").split(",") if p]

T0 = time.time()


def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


src = ("splice_out/cifar100_clip.npz" if FEATS == "clip" else "ranpac_out/cifar100_feats.npz")
z = np.load(src)
X, y = z["ftr"], z["ytr"]
log(f"[{FEATS}] {src}  {X.shape}  K={K}  tau=p{PCT:g}")

COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13])


def un(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


# ----------------------- region primitives (identical interface) -----------------------
class Ball:
    """Centroid + calibrated cosine radius. The 1-direction control."""
    def __init__(self, Xtr, k):
        self.mu = un(un(Xtr).mean(0, keepdims=True))[0]

    def score(self, Q):
        return un(Q) @ self.mu


class MultiProto:
    """K k-means centroids, max cosine. Equal-budget control -- the primitive that has beaten
    the cone in every previous head-to-head."""
    def __init__(self, Xtr, k):
        Xn = un(Xtr)
        kk = int(min(k, len(Xn)))
        self.C = un(KMeans(n_clusters=kk, n_init=4, random_state=0).fit(Xn).cluster_centers_)

    def score(self, Q):
        return (un(Q) @ self.C.T).max(1)


class Subspace:
    """Class PCA to K dims; score = cosine between q and its projection. A linear-subspace
    region rather than a conic one."""
    def __init__(self, Xtr, k):
        Xn = un(Xtr)
        d = int(min(k, max(len(Xn) - 1, 2), Xn.shape[1]))
        self.p = PCA(n_components=d).fit(Xn)

    def score(self, Q):
        Qn = un(Q)
        rec = self.p.inverse_transform(self.p.transform(Qn))
        return np.clip((un(rec) * Qn).sum(1), -1, 1)


class Cone:
    """Conic hull, K rays, NNLS residual -- the claim under test."""
    def __init__(self, Xtr, k):
        Xn = un(Xtr)
        self.h = ConicHull(n_rays=int(min(k, len(Xn))), use_pca=True,
                           pca_dim=int(min(64, max(len(Xn) - 1, 2)))).fit(Xn)

    def score(self, Q):
        return self.h.score_all(un(Q))["geo_residual"]


PRIMS = {"ball": Ball, "multiproto": MultiProto, "subspace": Subspace, "cone": Cone}
WANT = ONLY or list(PRIMS)


def build(Prim, Xc, k=K, seed=0):
    """Fit on 70%, calibrate tau on the held-out 30%. Same protocol for every primitive."""
    Xc = un(Xc)
    r = np.random.default_rng(seed).permutation(len(Xc))
    nh = max(int(HOLD * len(Xc)), 3)
    te, tr = Xc[r[:nh]], Xc[r[nh:]]
    reg = Prim(tr, k)
    tau = float(np.percentile(reg.score(te), PCT))
    return reg, tau, te


results = {}
for pname in WANT:
    Prim = PRIMS[pname]
    log(f"=== primitive: {pname} ===")
    t0 = time.time()
    fine = {c: build(Prim, X[y == c], seed=c) for c in range(100)}
    log(f"  {pname}: 100 fine regions built ({time.time()-t0:.1f}s)")
    coarse = {g: build(Prim, X[np.isin(y, np.where(COARSE == g)[0])], seed=100 + g)
              for g in range(20)}
    log(f"  {pname}: 20 coarse regions built ({time.time()-t0:.1f}s)")

    # --- full 100 x 20 containment matrix (fraction inside, and raw mean score) ---
    FR = np.zeros((100, 20))          # thresholded containment
    RAW = np.zeros((100, 20))         # raw mean score (threshold-free)
    for c in range(100):
        teF = fine[c][2]
        for g in range(20):
            regC, tauC, _ = coarse[g]
            s = regC.score(teF)
            FR[c, g] = float((s >= tauC).mean())
            RAW[c, g] = float(s.mean())
        if (c + 1) % 25 == 0:
            log(f"  {pname}: containment {c+1}/100 fine classes ({time.time()-t0:.1f}s)")

    parent = COARSE                                  # true parent of each fine class
    own = FR[np.arange(100), parent]
    unrel_mask = np.ones((100, 20), bool)
    unrel_mask[np.arange(100), parent] = False
    unrel = FR[unrel_mask].reshape(100, 19).mean(1)

    # --- reverse direction: coarse -> its own fine children ---
    rev = []
    for c in range(100):
        regF, tauF, _ = fine[c]
        teC = coarse[int(parent[c])][2]
        rev.append(float((regF.score(teC) >= tauF).mean()))
    rev = np.array(rev)

    # --- siblings (fine -> fine within the same superclass) ---
    sib = []
    for g in range(20):
        mem = np.where(COARSE == g)[0]
        for i in mem:
            regA, tauA, _ = fine[i]
            for j in mem:
                if i != j:
                    sib.append(float((regA.score(fine[j][2]) >= tauA).mean()))
    sib = float(np.mean(sib))

    # --- threshold-free parent AUROC over all 100x20 pairs ---
    lab = np.zeros((100, 20), int)
    lab[np.arange(100), parent] = 1
    auroc = roc_auc_score(lab.ravel(), RAW.ravel())
    # rank of the true parent among the 20 coarse regions (1 = best)
    rank = np.array([int(1 + (RAW[c] > RAW[c, parent[c]]).sum()) for c in range(100)])
    top1 = float((rank == 1).mean())

    results[pname] = dict(own=float(own.mean()), unrel=float(unrel.mean()),
                          gap=float(own.mean() - unrel.mean()), rev=float(rev.mean()),
                          asym=float(own.mean() - rev.mean()), sib=sib,
                          auroc=float(auroc), top1=top1, rank=float(rank.mean()))
    r = results[pname]
    log(f"  {pname}: own {r['own']:.4f} unrel {r['unrel']:.4f} GAP {r['gap']:+.4f} | "
        f"AUROC {r['auroc']:.4f} top1 {r['top1']:.4f} | rev {r['rev']:.4f} sib {r['sib']:.4f}")

np.save("exp3b_results.npy", results, allow_pickle=True)

print("\n" + "=" * 104)
print(f"EXP3b — is the semantic order CONE-SPECIFIC?  ({FEATS}, CIFAR-100, K={K}, tau=p{PCT:g})")
print("=" * 104)
print(f"{'primitive':>11} | {'fine->own':>9} {'fine->unrel':>11} {'SEM GAP':>9} | "
      f"{'AUROC':>7} {'top-1':>7} {'meanRank':>9} | {'coarse->fine':>12} {'sibling':>8}")
print("-" * 104)
for p in WANT:
    r = results[p]
    star = "  <- claim" if p == "cone" else ""
    print(f"{p:>11} | {r['own']:>9.4f} {r['unrel']:>11.4f} {r['gap']:>+9.4f} | "
          f"{r['auroc']:>7.4f} {r['top1']:>7.4f} {r['rank']:>9.2f} | "
          f"{r['rev']:>12.4f} {r['sib']:>8.4f}{star}")
print("-" * 104)
if "cone" in results and len(results) > 1:
    best_other = max((v["auroc"], k) for k, v in results.items() if k != "cone")
    bg = max((v["gap"], k) for k, v in results.items() if k != "cone")
    c = results["cone"]
    print(f"cone AUROC   {c['auroc']:.4f}  vs best non-conic ({best_other[1]}) "
          f"{best_other[0]:.4f}   delta {c['auroc']-best_other[0]:+.4f}")
    print(f"cone SEM GAP {c['gap']:+.4f}  vs best non-conic ({bg[1]}) "
          f"{bg[0]:+.4f}   delta {c['gap']-bg[0]:+.4f}")
    print()
    if c["auroc"] > best_other[0] + 0.02:
        print("=> CONE-SPECIFIC. First genuine cone-specific positive in the project.")
    elif abs(c["auroc"] - best_other[0]) <= 0.02:
        print("=> TIE. The finding is 'REGION models reveal an unsupervised semantic partial")
        print("   order in pretrained space' — novel and publishable, but NOT about cones.")
        print("   (Same outcome as the RanPAC-projection control: the structure is real, the")
        print("    conic parameterization is not what surfaces it.)")
    else:
        print("=> CONE LOSES to a simpler region primitive. Report the winner instead.")
print("=" * 104)
print("NOTE: AUROC/top-1 are threshold-free (no tau), so they are the primary numbers;")
print("SEM GAP depends on the p%g calibration. All primitives share fit/holdout/tau protocol." % PCT)
