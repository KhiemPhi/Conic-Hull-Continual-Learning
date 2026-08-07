#!/usr/bin/env python3
"""exp26_learned_dictionary.py — can a LEARNED conic dictionary beat k-means centroids?

WHERE THIS COMES FROM
    Generators are the largest lever we have measured (SPA -> k-means was +62 points at R=2
    on ImageNet-R). k-means is not optimal for anything conic though -- it minimises
    quantisation error of the OWN class, not reconstruction by a NON-NEGATIVE combination
    and certainly not discrimination. Both endpoints of the design space are now known:

        SPA      dictionary atoms are DATA POINTS            (separable NMF)   -- extremal
        k-means  atoms are convex combinations of data       (~convex NMF)     -- central

    This script fills in the middle by actually optimising the dictionary:

        cnmf   min ||X - A B X||^2   s.t. A >= 0, B >= 0            (convex NMF, Ding et al.)
        arch   same, with A and B rows on the SIMPLEX               (archetypal analysis)

    Archetypal analysis is the controlled interpolation: atoms stay convex combinations of
    data (denoised, like centroids) but the objective pushes them toward the hull boundary
    (extremal, like SPA). Both are initialised FROM k-means, so they can only improve on it
    in-objective -- if they lose on the task, the task and the reconstruction objective
    disagree, which is itself the finding.

    `random` (R random data rows) is the control: if it ties the learned dictionaries then
    nothing about the dictionary matters and only R does.

TWO TASKS, because the cone behaves oppositely on them
    clf   per-class dictionaries; argmax cos(q, NNLS(q)) over all classes (A-Last-equivalent)
    ood   one dictionary on the seen half of the classes; ID = seen test rows, OOD = unseen
          (exp19's P3). NOTE exp19's stored cone cells are STALE -- 1-iteration solver plus
          SPA -- so this re-measures that comparison for the first time correctly.

    For every dictionary we also score `max cosine` over the SAME atoms, which isolates the
    dictionary from the scoring rule.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR RS=4,16 python -u exp26_learned_dictionary.py
    DS=CUB200 RS=4 DICTS=kmeans,arch TASKS=clf python -u exp26_learned_dictionary.py
    SELFCHECK_ONLY=1 python -u exp26_learned_dictionary.py
"""
import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
RS = [int(x) for x in os.environ.get("RS", "4,16").split(",")]
DICTS = os.environ.get("DICTS", "random,spa,kmeans,cnmf,arch").split(",")
TASKS = os.environ.get("TASKS", "clf,ood").split(",")
OUTER = int(os.environ.get("OUTER", 40))
INNER = int(os.environ.get("INNER", 25))
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, "exp26_learned_dictionary.json")


def simplex_rows(Z):
    """Row-wise Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al. 2008)."""
    K = Z.shape[1]
    U = -np.sort(-Z, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, K + 1)
    rho = (U - css / ind > 0).cumsum(1).argmax(1)
    theta = css[np.arange(len(Z)), rho] / (rho + 1)
    return np.maximum(Z - theta[:, None], 0.0)


def learn_dict(X, R, mode, seed=0, outer=OUTER, inner=INNER):
    """Alternating projected gradient for  min ||X - A(BX)||^2 .

    mode 'cnmf': A >= 0, B >= 0                (convex NMF)
    mode 'arch': A, B rows on the simplex      (archetypal analysis)
    Initialised from the k-means solution, so obj[0] is the k-means objective and any
    decrease is a strict improvement over the current default. Returns (D, objective trace).
    """
    n = len(X)
    R = int(min(R, n))
    K = X @ X.T
    lam_K = float(np.linalg.eigvalsh(K)[-1])
    lab = (np.zeros(n, int) if R == 1 else
           KMeans(R, n_init=4, random_state=seed).fit(X).labels_)
    B = np.zeros((R, n))
    for r in range(R):
        m = lab == r
        if not m.any():
            m = np.zeros(n, bool); m[r % n] = True
        B[r, m] = 1.0 / m.sum()
    A = np.zeros((n, R)); A[np.arange(n), lab] = 1.0
    proj = simplex_rows if mode == "arch" else (lambda Z: np.maximum(Z, 0.0))

    obj = [float(np.linalg.norm(X - A @ (B @ X)) ** 2)]
    for _ in range(outer):
        D = B @ X                                     # (R, d)
        G = D @ D.T
        c = X @ D.T
        L = max(float(np.linalg.eigvalsh(G)[-1]), 1e-12)
        for _ in range(inner):                        # codes
            A = proj(A - (A @ G - c) / L)
        AtA = A.T @ A
        L2 = max(float(np.linalg.eigvalsh(AtA)[-1]) * lam_K, 1e-12)
        for _ in range(inner):                        # dictionary
            B = proj(B - (AtA @ B @ K - A.T @ K) / L2)
        obj.append(float(np.linalg.norm(X - A @ (B @ X)) ** 2))
    return E.un(B @ X), obj


def make_dict(X, R, kind, seed=0):
    n = len(X)
    R = int(min(R, n))
    if R <= 1:
        return E.un(X.mean(0, keepdims=True)), None
    if kind == "random":
        return E.un(X[np.random.default_rng(seed).choice(n, R, replace=False)]), None
    if kind == "kmeans":
        return E.un(KMeans(R, n_init=4, random_state=seed).fit(X).cluster_centers_), None
    if kind == "spa":
        ov = max(1, min(3, (n - 2) // R))
        return ConicHull(n_rays=R, ray_init="spa", spa_oversample=ov,
                         ray_diversity="hybrid" if ov > 1 else "spa").fit(X).extreme_rays_, None
    if kind in ("cnmf", "arch"):
        return learn_dict(X, R, kind, seed)
    raise ValueError(kind)


def cone_score(A, Q):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = E.un(A)
    return h.score(Q)


def max_score(A, Q):
    return (E.un(Q) @ E.un(A).T).max(1)


# ------------------------------------------------------------------ self-checks
def selfcheck():
    rng = np.random.default_rng(0)
    X = E.un(np.abs(rng.normal(size=(120, 32))) + 0.3 * rng.normal(size=(120, 32)))
    bad = []

    def chk(name, ok, detail=""):
        print(f"   {'PASS' if ok else '**FAIL**'}  {name}  {detail}")
        if not ok:
            bad.append(name)

    for mode in ("cnmf", "arch"):
        D, obj = learn_dict(X, 6, mode)
        d = np.diff(obj)
        chk(f"{mode}: objective non-increasing", bool((d <= 1e-8).all()),
            f"worst step {d.max():+.2e}, {obj[0]:.4f} -> {obj[-1]:.4f}")
        chk(f"{mode}: improves on its k-means init", obj[-1] <= obj[0] + 1e-9,
            f"{(obj[0]-obj[-1])/max(obj[0],1e-12)*100:.2f}% lower")
        chk(f"{mode}: atoms are unit rows", np.allclose(np.linalg.norm(D, axis=1), 1, atol=1e-5))
    D0, o0 = learn_dict(X, 6, "cnmf", outer=0)
    Dk, _ = make_dict(X, 6, "kmeans")
    chk("outer=0 returns exactly the k-means dictionary",
        np.allclose(np.sort(D0, 0), np.sort(Dk, 0), atol=1e-5))
    Z = simplex_rows(rng.normal(size=(50, 7)))
    chk("simplex projection: rows >= 0 and sum to 1",
        bool((Z >= -1e-9).all() and np.allclose(Z.sum(1), 1, atol=1e-9)))
    for k in DICTS:
        A, _ = make_dict(X, 1, k)
        chk(f"{k}: R=1 collapses to the class mean",
            np.allclose(A, E.un(X.mean(0, keepdims=True)), atol=1e-5))
    A, _ = make_dict(X, 5, "kmeans")
    chk("cone score of an atom is 1", abs(cone_score(A, A).min() - 1.0) < 1e-3,
        f"min {cone_score(A, A).min():.6f}")
    return bad


print("SELF-CHECKS")
_bad = selfcheck()
if _bad:
    raise SystemExit(f"self-checks failed: {_bad} -- refusing to produce numbers")
if int(os.environ.get("SELFCHECK_ONLY", 0)):
    raise SystemExit(0)


# ------------------------------------------------------------------ tasks
def run(ds):
    Ztr, Zte = E.adapted_features(ds)
    ytr, yte, ncls = E.get_labels(ds)
    Q = E.un(Zte)
    out = {}
    for R in RS:
        for kind in DICTS:
            t = time.time()
            if "clf" in TASKS:
                Sc = np.full((len(Q), ncls), -np.inf, np.float32)
                Sm = np.full((len(Q), ncls), -np.inf, np.float32)
                for c in range(ncls):
                    X = Ztr[ytr == c]
                    if len(X) < 2:
                        continue
                    A, _ = make_dict(X, R, kind, seed=c)
                    Sc[:, c] = cone_score(A, Q)
                    Sm[:, c] = max_score(A, Q)
                out[f"clf|{kind}|cone|{R}"] = float((Sc.argmax(1) == yte).mean())
                out[f"clf|{kind}|max|{R}"] = float((Sm.argmax(1) == yte).mean())
            if "ood" in TASKS:
                seen = np.arange(ncls // 2)
                Xs = Ztr[np.isin(ytr, seen)][:6000]
                ID, OOD = Q[np.isin(yte, seen)], Q[~np.isin(yte, seen)]
                A, _ = make_dict(Xs, R, kind, seed=0)
                lab = np.r_[np.ones(len(ID)), np.zeros(len(OOD))]
                for nm, f in (("cone", cone_score), ("max", max_score)):
                    out[f"ood|{kind}|{nm}|{R}"] = float(
                        roc_auc_score(lab, np.r_[f(A, ID), f(A, OOD)]))
            log(f"  R={R:<3d} {kind:<8s} " + "  ".join(
                f"{k.split('|',1)[1]} {v:.4f}" for k, v in out.items()
                if k.endswith(f"|{R}") and f"|{kind}|" in k) + f"  [{time.time()-t:.0f}s]")
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    allres.setdefault(ds, {}).update(run(ds))
    json.dump(allres, open(OUT, "w"), indent=2)

W = 86
print("\n" + "=" * W)
print("EXP26 — learned conic dictionaries vs k-means, matched budget")
print("=" * W)
for ds, res in allres.items():
    for task, lab in (("clf", "classifier accuracy (all classes)"),
                      ("ood", "near-OOD AUROC (seen vs unseen classes)")):
        if not any(k.startswith(task) for k in res):
            continue
        print(f"\n--- {ds}   {lab}")
        print(f"{'R':>4}{'dictionary':>12}{'cone (NNLS)':>14}{'max cosine':>13}{'Δ rule':>9}")
        for R in RS:
            for kind in DICTS:
                a, b = res.get(f"{task}|{kind}|cone|{R}"), res.get(f"{task}|{kind}|max|{R}")
                if a is None:
                    continue
                print(f"{R:>4}{kind:>12}{a:>14.4f}{b:>13.4f}{a-b:>+9.4f}")
print("\n" + "-" * W)
print("cnmf/arch are initialised FROM k-means, so a loss means the reconstruction objective")
print("   and the task disagree -- not that the optimiser failed (the objective is asserted")
print("   non-increasing in the self-checks).")
print("`random` is the control: if it ties the rest, only R matters, not the dictionary.")
print("=" * W)
print(f"wrote {OUT}")
