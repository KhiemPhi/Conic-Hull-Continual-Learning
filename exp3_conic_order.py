"""
exp3_conic_order.py — IS THERE A CONIC PARTIAL ORDER IN PRETRAINED FEATURE SPACE?

Completely unexploited property: a convex cone K induces a PARTIAL ORDER
        x  <=_K  y     iff     y - x  in  K
So a cone is not only a region — it is an ORDERING. Applied to concepts rather than to
samples, this gives a geometric test for ENTAILMENT / subsumption:

        cone(fine class)  SUBSET-OF  cone(its superclass)          e.g. cone(oak) ⊆ cone(tree)

That is a claim about RELATIONS BETWEEN CONCEPTS, not about discriminating samples — which
is exactly the regime where every previous cone use failed. Hyperbolic entailment cones
(Ganea'18) and order embeddings (Vendrov'16) TRAIN such structure; the question here is
whether it is ALREADY PRESENT, unsupervised, in a pretrained VLM/ViT space.

Measured, using the official CIFAR-100 fine->coarse hierarchy (100 fine, 20 superclasses):
  containment(A -> B) = fraction of class-A samples that the cone of B explains as well as
                        B explains its own held-out samples (calibrated, so it is a real
                        in/out test rather than a similarity score)
  ASYMMETRY = containment(fine -> coarse) - containment(coarse -> fine)
      a genuine order needs fine ⊆ coarse and NOT coarse ⊆ fine  => asymmetry >> 0
  ANTISYMMETRY / TRANSITIVITY spot-checks
  CONTROL: random unrelated class pairs must show NO asymmetry (else it is an artifact
           of cone size / sample count rather than semantics)

Run:  python -u exp3_conic_order.py
      FEATS=vit python -u exp3_conic_order.py
"""
import os
import numpy as np
from conic_hull import ConicHull

SEED = 0
np.random.seed(SEED)
FEATS = os.environ.get("FEATS", "clip")        # clip | vit
K_RAYS = int(os.environ.get("K_RAYS", 20))
PCT = float(os.environ.get("PCT", 10))         # calibration percentile for "inside"

src = ("splice_out/cifar100_clip.npz" if FEATS == "clip" else "ranpac_out/cifar100_feats.npz")
z = np.load(src)
X, y = z["ftr"], z["ytr"]
print(f"[{FEATS}] {src}  {X.shape}")

# official CIFAR-100 fine -> coarse (20 superclasses)
COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13])


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def build(Xc, K=K_RAYS, holdout=0.3, seed=0):
    """Fit a hull on 70% and calibrate an 'inside' threshold on the held-out 30%."""
    Xc = un(Xc); n = len(Xc)
    r = np.random.default_rng(seed).permutation(n)
    nh = max(int(holdout * n), 3)
    te, tr = Xc[r[:nh]], Xc[r[nh:]]
    h = ConicHull(n_rays=int(min(K, len(tr))), use_pca=True,
                  pca_dim=int(min(64, max(len(tr) - 1, 2)))).fit(tr)
    tau = float(np.percentile(h.score_all(te)["geo_residual"], PCT))
    return h, tau, te


def contains(h, tau, Q):
    """Fraction of Q that the hull explains at least as well as it explains its own
    held-out members (calibrated in/out test, not a raw similarity)."""
    return float((h.score_all(un(Q))["geo_residual"] >= tau).mean())


print(f"\n=== building hulls: 100 fine + 20 coarse (K={K_RAYS}, tau=p{PCT:g}) ===")
fine, coarse = {}, {}
for c in range(100):
    fine[c] = build(X[y == c], seed=c)
for g in range(20):
    coarse[g] = build(X[np.isin(y, np.where(COARSE == g)[0])], seed=100 + g)
print("  done")

# ---------------- fine -> its own coarse (should be contained) ----------------
print("\n=== A. does cone(fine) sit inside cone(its superclass)? ===")
up, down = [], []
for c in range(100):
    g = int(COARSE[c])
    hF, tF, teF = fine[c]
    hC, tC, teC = coarse[g]
    up.append(contains(hC, tC, teF))        # fine samples inside coarse cone
    down.append(contains(hF, tF, teC))      # coarse samples inside fine cone
up, down = np.array(up), np.array(down)
print(f"  containment fine->coarse  {up.mean():.4f}")
print(f"  containment coarse->fine  {down.mean():.4f}")
print(f"  ASYMMETRY (order signal)  {up.mean()-down.mean():+.4f}")

# ---------------- control: unrelated pairs ----------------
print("\n=== B. CONTROL — unrelated (fine, coarse) pairs ===")
rng = np.random.default_rng(SEED)
cu, cd = [], []
for c in range(100):
    g = int(COARSE[c])
    other = rng.choice([k for k in range(20) if k != g])
    hF, tF, teF = fine[c]
    hC, tC, teC = coarse[other]
    cu.append(contains(hC, tC, teF)); cd.append(contains(hF, tF, teC))
cu, cd = np.array(cu), np.array(cd)
print(f"  containment fine->UNRELATED coarse {cu.mean():.4f}")
print(f"  containment UNRELATED coarse->fine {cd.mean():.4f}")
print(f"  ASYMMETRY (should be ~0)           {cu.mean()-cd.mean():+.4f}")

# ---------------- fine vs fine within a superclass (antisymmetry) ----------------
print("\n=== C. ANTISYMMETRY — sibling fine classes (neither should contain the other) ===")
sib = []
for g in range(20):
    mem = np.where(COARSE == g)[0]
    for i in range(len(mem)):
        for j in range(len(mem)):
            if i == j:
                continue
            hA, tA, _ = fine[mem[i]]
            _, _, teB = fine[mem[j]]
            sib.append(contains(hA, tA, teB))
print(f"  mean sibling containment {np.mean(sib):.4f}  (want LOW; high => cones over-cover)")

# ---------------- transitivity spot-check ----------------
print("\n=== D. TRANSITIVITY — fine ⊆ coarse and coarse ⊆ ALL implies fine ⊆ ALL ===")
hALL, tALL, teALL = build(X[rng.choice(len(X), 5000, replace=False)], seed=999)
f_all = np.mean([contains(hALL, tALL, fine[c][2]) for c in range(100)])
c_all = np.mean([contains(hALL, tALL, coarse[g][2]) for g in range(20)])
print(f"  fine  -> ALL  {f_all:.4f}")
print(f"  coarse-> ALL  {c_all:.4f}   (both should be HIGH if the order is coherent)")

np.save(f"exp3_results_{FEATS}.npy",
        dict(up=up, down=down, ctrl_up=cu, ctrl_down=cd, sib=np.array(sib),
             f_all=f_all, c_all=c_all), allow_pickle=True)
print("\n" + "=" * 92)
print(f"EXP3 — conic order / entailment structure ({FEATS}, CIFAR-100 hierarchy)")
print("=" * 92)
print(f"  real   pairs: fine->coarse {up.mean():.4f} | coarse->fine {down.mean():.4f} "
      f"| asym {up.mean()-down.mean():+.4f}")
print(f"  control pairs: fine->coarse {cu.mean():.4f} | coarse->fine {cd.mean():.4f} "
      f"| asym {cu.mean()-cd.mean():+.4f}")
print(f"  sibling containment {np.mean(sib):.4f} | fine->ALL {f_all:.4f} coarse->ALL {c_all:.4f}")
print("-" * 92)
print("WIN CONDITION: real asymmetry >> control asymmetry (~0), sibling containment LOW,")
print("and transitivity holds. That would mean a semantic PARTIAL ORDER already exists,")
print("unsupervised, in pretrained feature space — readable with conic geometry alone.")
print("FAILURE MODE to watch: if containment is ~1.0 everywhere the cones over-cover")
print("(K rays spanning <=K of D dims) — lower PCT or raise K_RAYS and re-run.")
print("=" * 92)
