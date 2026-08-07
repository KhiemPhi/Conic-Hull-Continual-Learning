#!/usr/bin/env python3
"""exp21_cone_vs_multiproto.py — cone vs multi-prototype AS A CLASSIFIER, matched budget.

THE QUESTION
    Per class, store exactly R vectors. Does a conic hull over them beat taking the max
    cosine to them? exp20 compares the hull to NCM and RanPAC but never to multi-proto at
    the same per-class budget, which is the comparison that actually decides whether the
    conic score earns its keep.

WHY R MUST BE SWEPT
    The historical config (R=24) is DEGENERATE on CUB: ~27 images/class span 26 dimensions,
    and the hybrid path asks for 3*24=72 SPA candidates, so 46 of them come from a
    numerically-zero residual. Any R near the class size makes "extreme ray" meaningless --
    every sample is already extreme. Sweeping R from 1 is the only way to see the trend.

ARMS (identical atoms budget R per class; the ONLY differences are atom choice and rule)
    ncm          1 mean                                      (floor; R=1 for every arm)
    multiproto   R k-means sub-centroids, score = max cos
    cone_km      the same R k-means centroids as GENERATORS, score = cos(q, NNLS(q))
    cone_spa     R SPA extremal rays as generators,  score = cos(q, NNLS(q))
    protomaha    R k-means centroids, max cos in a TIED-WHITENED space   (the unexplored
                 cell: multiproto's atoms with Mahalanobis's metric)

    cone_km vs multiproto isolates the SCORING RULE at identical atoms.
    cone_spa vs cone_km isolates the ATOMS at an identical rule.
    spa_oversample is reduced per class so the candidate pool never exceeds the rank --
    otherwise cone_spa is scored in the degenerate regime and the comparison is rigged.

METRIC
    A-Last equivalent: argmax over all classes on the full test set, all classes seen.
    (A_plus freezes the backbone, so the final stage is just the head fitted on everything.)

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=CUB200 python -u exp21_cone_vs_multiproto.py
    DS=CUB200,IMAGENETR RS=1,2,4,8,16,32 python -u exp21_cone_vs_multiproto.py
"""
import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DSETS = os.environ.get("DS", "CUB200").split(",")
RS = [int(x) for x in os.environ.get("RS", "1,2,4,8,16").split(",")]
ARMS = os.environ.get("ARMS", "ncm,multiproto,protomaha,cone_km,cone_w,cone_wsx,cone_wl1").split(",")
ITERS = int(os.environ.get("ITERS", 500))
SHRINK = float(os.environ.get("SHRINK", 1e-2))
L1 = float(os.environ.get("L1", 0.01))
OUT = os.path.join(REPO, "exp21_cone_vs_multiproto.json")


def centroids(X, R):
    R = int(min(R, len(X)))
    if R <= 1:
        return E.un(X.mean(0, keepdims=True))
    return E.un(KMeans(R, n_init=4, random_state=0).fit(X).cluster_centers_)


def spa_rays(X, R):
    """spa_oversample shrunk to fit the class rank: 3R candidates in an (n-1)-dim span is
    the degenerate regime the library warns about, and it is the historical default."""
    n = len(X)
    R = int(min(R, max(n - 2, 1)))
    if R <= 1:
        return E.un(X.mean(0, keepdims=True))
    ov = max(1, min(3, (n - 2) // R))
    return ConicHull(n_rays=R, spa_oversample=ov, center=True,
                     ray_diversity="hybrid" if ov > 1 else "spa").fit(X).extreme_rays_


def cone(X, R, **kw):
    """New-API hull: k-means generators by default, optional tied whitening applied
    identically at fit and score time, optional regularisation / simplex constraint."""
    return ConicHull(n_rays=int(min(R, len(X))), ray_init="kmeans",
                     nnls_iters=ITERS, **kw).fit(X)


def cone_scorer(A):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = A

    def f(Q):
        rec = h._reconstruct_norm(Q) @ A
        return (Q * (rec / (np.linalg.norm(rec, axis=1, keepdims=True) + 1e-12))).sum(1)
    return f


def run(ds):
    Ztr, Zte = E.adapted_features(ds)
    ytr, yte, ncls = E.get_labels(ds)
    Q = E.un(Zte)
    sizes = [int((ytr == c).sum()) for c in range(ncls)]
    log(f"{ds}: {ncls} classes, {len(Q)} test, imgs/class min {min(sizes)} "
        f"med {int(np.median(sizes))}")

    # tied within-class covariance for `protomaha` (estimable when per-class is not)
    Xc = np.concatenate([Ztr[ytr == c] - Ztr[ytr == c].mean(0) for c in range(ncls)])
    S = (Xc.T @ Xc) / len(Xc)
    S += SHRINK * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)
    Qw = E.un(Q @ Wh)

    out = {}
    for R in RS:
        acc = {a: np.full((len(Q), ncls), -np.inf, np.float32) for a in ARMS}
        for c in range(ncls):
            X = Ztr[ytr == c]
            if len(X) < 2:
                continue
            km = centroids(X, R)
            if "ncm" in ARMS:
                acc["ncm"][:, c] = Q @ E.un(X.mean(0, keepdims=True)).T[:, 0]
            if "multiproto" in ARMS:
                acc["multiproto"][:, c] = (Q @ km.T).max(1)
            if "cone_km" in ARMS:
                acc["cone_km"][:, c] = cone(X, R).score(Q)
            if "cone_spa" in ARMS:
                acc["cone_spa"][:, c] = cone_scorer(spa_rays(X, R))(Q)
            if "protomaha" in ARMS:
                acc["protomaha"][:, c] = (Qw @ E.un(centroids(E.un(X @ Wh), R)).T).max(1)
            # --- the composed objects: cone INSIDE the whitened metric ---------------
            if "cone_w" in ARMS:
                acc["cone_w"][:, c] = cone(X, R, whiten=Wh).score(Q)
            if "cone_wsx" in ARMS:
                acc["cone_wsx"][:, c] = cone(X, R, whiten=Wh,
                                             constraint="simplex").score(Q)
            if "cone_wl1" in ARMS:
                acc["cone_wl1"][:, c] = cone(X, R, whiten=Wh, nnls_l1=L1).score(Q)
        for a in ARMS:
            out[f"{a}|{R}"] = float((acc[a].argmax(1) == yte).mean())
        log(f"  R={R:<3d} " + "  ".join(f"{a} {out[f'{a}|{R}']*100:.2f}" for a in ARMS))
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    allres.setdefault(ds, {}).update(run(ds))
    json.dump(allres, open(OUT, "w"), indent=2)

W = 88
print("\n" + "=" * W)
print("EXP21 — classifier accuracy at a matched per-class budget of R vectors")
print("=" * W)
for ds, res in allres.items():
    print(f"\n--- {ds}")
    print(f"{'R':>4}" + "".join(f"{a:>13}" for a in ARMS) + f"{'bestcone-pmaha':>16}")
    for R in RS:
        if f"{ARMS[0]}|{R}" not in res:
            continue
        row = "".join(f"{res.get(f'{a}|{R}', float('nan'))*100:>13.2f}" for a in ARMS)
        best_cone = max((res.get(f"{a}|{R}", float("-inf"))
                         for a in ARMS if a.startswith("cone")), default=float("nan"))
        d = best_cone - res.get(f"protomaha|{R}", float("nan"))
        print(f"{R:>4}{row}{d*100:>+16.2f}")
print("\n" + "-" * W)
print("cone_km vs multiproto = the scoring rule at IDENTICAL atoms (the decisive column).")
print("cone_spa vs cone_km   = the atoms at an identical rule.")
print("protomaha             = multiproto's atoms under a tied-whitened metric.")
print("=" * W)
print(f"wrote {OUT}")
