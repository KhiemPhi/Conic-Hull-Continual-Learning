#!/usr/bin/env python3
"""exp20_cone_revisit.py — re-run EVERY cone idea in this repo against the fixed solver.

WHY THIS EXISTS
    conic_hull.py's FISTA loop tested convergence by comparing a tensor with itself
    (`W = W_next` executed before `norm(W_next - W)`), so it always broke at iteration 0.
    Every GPU hull score this repo ever produced was the one-step approximation
        w = lr * relu(R^T q)
    not an NNLS conic projection. Three further defects compounded it: fit() was
    non-deterministic (unseeded randomized PCA), the candidate pool exceeded the working
    rank so a majority of SPA picks came from a numerically-zero residual, and PCA CENTRED
    the data even though a cone is anchored at the origin.

    Measured on one cell before the fix: CUB200 near-OOD cone AUROC 0.626 -> 0.865.
    So the sharpest claim on record -- "the hull is barely better than a single mean vector"
    -- is an artifact. Every conic conclusion in this repo has to be re-taken.

WHAT THIS DOES
    Re-runs each cone idea under BOTH solvers on the same features, same rays, same splits:
        legacy  nnls_iters=1    bit-for-bit the old behaviour, so old numbers reproduce
        fixed   nnls_iters=500  a real NNLS solution (verified == scipy.optimize.nnls)
    and prints, per idea: legacy score, fixed score, the non-cone baseline it has to beat,
    and whether the historical verdict FLIPS.

    A conclusion only counts as re-established if it holds under `fixed`. A conclusion that
    flips was never a fact about cones -- it was a fact about a broken solver.

IDEAS (env IDEAS=..., default all)
    descriptor  hull as a set descriptor, near-OOD AUROC at matched budget   (exp19)
    classifier  per-class hulls as the classifier, A-Last                    (exp17)
    reranker    hull re-ranks RanPAC's top-2                                 (exp14)
    klocal      shrink-wrapped local hull (k_local nearest rays)             (exp17)
    multicone   N_SUB k-means sub-cones per class, max over sub-cones        (exp17)
    sweep       nnls_iters in {1,2,5,25,100,500} -- how much of any result is the solver
    centring    center=True vs False, the other conceptual fix

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=CUB200 python -u exp20_cone_revisit.py
    DS=CUB200,IMAGENETR,CIFAR100,IMAGENETA IDEAS=descriptor,sweep python -u exp20_cone_revisit.py
    DS=CUB200 IDEAS=classifier R=24 python -u exp20_cone_revisit.py
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DSETS = os.environ.get("DS", "CUB200").split(",")
IDEAS = os.environ.get("IDEAS", "descriptor,classifier,reranker,klocal,multicone,"
                                "sweep,centring").split(",")
R = int(os.environ.get("R", 24))            # per-class budget (exp17's default)
R_SET = int(os.environ.get("R_SET", 64))    # whole-set budget (exp19's informative budget)
K_LOCAL = int(os.environ.get("K_LOCAL", 8))
N_SUB = int(os.environ.get("N_SUB", 4))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
MIN_N = 8
SWEEP = [1, 2, 5, 25, 100, 500]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(REPO, "exp20_cone_revisit.json")
SOLVERS = {"legacy": 1, "fixed": 500}


def hull(X, n_rays, k_local=None, iters=500, center=False):
    """One place that builds hulls, so every arm shares ray selection and differs ONLY in
    the solver. pca_dim is left at the default and auto-raised by fit() to cover the
    candidate pool -- the old call sites passed pca_dim=64 with 3*n_rays candidates."""
    n = len(X)
    return ConicHull(n_rays=int(np.clip(n_rays, 2, max(n - 2, 2))),
                     k_local=k_local, ray_diversity="hybrid",
                     nnls_iters=iters, center=center).fit(X)


# ------------------------------------------------------------------ data
def cell(ds):
    Ztr, Zte = E.adapted_features(ds)
    ytr, yte, ncls = E.get_labels(ds)
    rng = np.random.default_rng(0)
    p = rng.permutation(len(Ztr))
    ncal = max(int(0.1 * len(p)), 50)
    return dict(fit=Ztr[p[ncal:]], yfit=ytr[p[ncal:]], cal=Ztr[p[:ncal]],
                te=Zte, yte=yte, ncls=ncls)


def near_ood(d):
    """exp19's P3: fit on the first half of the classes, OOD = test rows of the unseen half."""
    seen = np.arange(d["ncls"] // 2)
    m = np.isin(d["yfit"], seen)
    return (d["fit"][m][:6000], d["yfit"][m][:6000],
            d["te"][np.isin(d["yte"], seen)], d["te"][~np.isin(d["yte"], seen)])


def auc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


# ------------------------------------------------------------------ ideas
def idea_descriptor(d, res):
    X, y, ID, OOD = near_ood(d)
    for name, it in SOLVERS.items():
        h = hull(X, R_SET, iters=it)
        res[f"descriptor|cone|{name}"] = auc(h.score(ID), h.score(OOD))
    for v in ["subspace", "kmeans", "multiproto", "maha", "mean"]:
        f = E.FIT[v](X, y, R_SET, np.random.default_rng(0))
        res[f"descriptor|{v}|baseline"] = auc(f(ID), f(OOD))


def _ranpac(d, seen_rows):
    """Final-stage RanPAC head (statistics are exactly additive, so the last stage is just
    the head fitted on everything seen)."""
    P = torch.randn(d["fit"].shape[1], M_RP,
                    generator=torch.Generator().manual_seed(0)).to(DEV)

    def H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.tensor(X[i:i + bs], device=DEV,
                                             dtype=torch.float32) @ P).double()
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, d["ncls"], device=DEV, dtype=torch.float64)
    Xf, yf = d["fit"][seen_rows], d["yfit"][seen_rows]
    for i, h in H(Xf):
        Y = torch.zeros(h.shape[0], d["ncls"], device=DEV, dtype=torch.float64)
        Y[torch.arange(h.shape[0]), torch.tensor(yf[i:i + h.shape[0]], device=DEV)] = 1.0
        G += h.T @ h
        C += h.T @ Y
    W = torch.linalg.solve(G + LAMBDAS[1] * torch.eye(M_RP, device=DEV,
                                                      dtype=torch.float64), C)
    L = torch.cat([h @ W for _, h in H(d["te"])]).cpu().numpy()
    del G, C, P
    torch.cuda.empty_cache()
    return L


def _class_scores(d, sizes, iters, k_local=None, multi=False):
    S = np.full((len(d["te"]), d["ncls"]), -np.inf, np.float32)
    for c in range(d["ncls"]):
        if sizes[c] < MIN_N:
            continue
        Xc = d["fit"][d["yfit"] == c]
        if multi:
            from sklearn.cluster import KMeans
            k = min(N_SUB, max(1, len(Xc) // MIN_N))
            lab = (KMeans(n_clusters=k, n_init=4, random_state=c).fit_predict(Xc)
                   if k > 1 else np.zeros(len(Xc), int))
            subs = [Xc[lab == j] for j in range(k) if (lab == j).sum() >= MIN_N] or [Xc]
            S[:, c] = np.stack([hull(s, R, iters=iters).score(d["te"]) for s in subs]).max(0)
        else:
            S[:, c] = hull(Xc, R, k_local=k_local, iters=iters).score(d["te"])
    return S


def idea_classifier(d, res, klocal=False, multi=False):
    sizes = [int((d["yfit"] == c).sum()) for c in range(d["ncls"])]
    tag = "klocal" if klocal else ("multicone" if multi else "classifier")
    for name, it in SOLVERS.items():
        S = _class_scores(d, sizes, it, K_LOCAL if klocal else None, multi)
        res[f"{tag}|cone|{name}"] = float((S.argmax(1) == d["yte"]).mean())
    if tag == "classifier":                                   # baselines once
        mu = np.stack([d["fit"][d["yfit"] == c].mean(0) if sizes[c] else
                       np.zeros(d["fit"].shape[1]) for c in range(d["ncls"])])
        mu = E.un(mu)
        res["classifier|ncm|baseline"] = float(((d["te"] @ mu.T).argmax(1) == d["yte"]).mean())
        L = _ranpac(d, np.arange(len(d["fit"])))
        res["classifier|ranpac|baseline"] = float((L.argmax(1) == d["yte"]).mean())


def idea_reranker(d, res):
    """exp14: let the hull arbitrate RanPAC's top-2. Can only help if the hull knows
    something RanPAC does not."""
    sizes = [int((d["yfit"] == c).sum()) for c in range(d["ncls"])]
    L = _ranpac(d, np.arange(len(d["fit"])))
    res["reranker|ranpac|baseline"] = float((L.argmax(1) == d["yte"]).mean())
    top2 = np.argsort(-L, axis=1)[:, :2]
    for name, it in SOLVERS.items():
        S = _class_scores(d, sizes, it)
        pick = np.where(S[np.arange(len(S)), top2[:, 0]] >= S[np.arange(len(S)), top2[:, 1]],
                        top2[:, 0], top2[:, 1])
        res[f"reranker|cone|{name}"] = float((pick == d["yte"]).mean())


def idea_sweep(d, res):
    X, y, ID, OOD = near_ood(d)
    for it in SWEEP:
        h = hull(X, R_SET, iters=it)
        res[f"sweep|cone|iters{it}"] = auc(h.score(ID), h.score(OOD))


def idea_centring(d, res):
    X, y, ID, OOD = near_ood(d)
    for cen in (True, False):
        h = hull(X, R_SET, iters=500, center=cen)
        res[f"centring|cone|{'centred' if cen else 'uncentred'}"] = auc(h.score(ID),
                                                                        h.score(OOD))


RUN = {"descriptor": idea_descriptor,
       "classifier": lambda d, r: idea_classifier(d, r),
       "klocal": lambda d, r: idea_classifier(d, r, klocal=True),
       "multicone": lambda d, r: idea_classifier(d, r, multi=True),
       "reranker": idea_reranker, "sweep": idea_sweep, "centring": idea_centring}

# ------------------------------------------------------------------ main
allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    d = cell(ds)
    log(f"=== {ds}: {d['ncls']} classes, fit {len(d['fit'])}, test {len(d['te'])}")
    res = allres.setdefault(ds, {})
    for idea in IDEAS:
        if idea not in RUN:
            continue
        if any(k.startswith(idea + "|") for k in res):
            log(f"  skip {idea} (done)")
            continue
        t = time.time()
        RUN[idea](d, res)
        json.dump(allres, open(OUT, "w"), indent=2)
        log(f"  {idea:11s} done [{time.time()-t:.0f}s]  " +
            "  ".join(f"{k.split('|',1)[1]} {v:.4f}"
                      for k, v in res.items() if k.startswith(idea + "|")))

# ------------------------------------------------------------------ verdicts
W = 98
print("\n" + "=" * W)
print("EXP20 — every cone idea, legacy 1-iteration solver vs fixed NNLS")
print("=" * W)
for ds, res in allres.items():
    print(f"\n--- {ds}")
    print(f"{'idea':<13}{'legacy':>9}{'fixed':>9}{'Δ':>9}   {'best non-cone baseline':<28}"
          f"{'verdict':>10}")
    for idea in ["descriptor", "classifier", "klocal", "multicone", "reranker"]:
        lo, fx = res.get(f"{idea}|cone|legacy"), res.get(f"{idea}|cone|fixed")
        if lo is None or fx is None:
            continue
        base = {k.split("|")[1]: v for k, v in res.items()
                if k.startswith(idea + "|") and k.endswith("|baseline")}
        if not base and idea in ("klocal", "multicone"):
            base = {k.split("|")[1]: v for k, v in res.items()
                    if k.startswith("classifier|") and k.endswith("|baseline")}
        bn, bv = max(base.items(), key=lambda kv: kv[1]) if base else ("-", float("nan"))
        verdict = ("n/a" if not base else
                   "BEATS" if fx > bv else
                   "still loses" if lo <= bv else "flips")
        print(f"{idea:<13}{lo:>9.4f}{fx:>9.4f}{fx-lo:>+9.4f}   "
              f"{f'{bn} {bv:.4f}':<28}{verdict:>10}")
    sw = {int(k.split('iters')[1]): v for k, v in res.items() if k.startswith("sweep|")}
    if sw:
        print("  solver sensitivity: " +
              "  ".join(f"{i}it {sw[i]:.4f}" for i in sorted(sw)))
    ce = {k.split("|")[2]: v for k, v in res.items() if k.startswith("centring|")}
    if ce:
        print(f"  centring: centred {ce.get('centred', float('nan')):.4f}  "
              f"uncentred {ce.get('uncentred', float('nan')):.4f}")
print("\n" + "-" * W)
print("'flips'       = the cone lost under the broken solver and BEATS the baseline once fixed")
print("'still loses' = conclusion re-established; the negative was real, not a solver artifact")
print("Only `fixed` numbers may be quoted. `legacy` exists to reproduce the old record.")
print("=" * W)
print(f"wrote {OUT}")
