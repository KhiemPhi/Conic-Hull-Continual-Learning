#!/usr/bin/env python3
"""exp33_lifted_cone.py — move the cone ABOVE RanPAC's random-feature lift.

THE ARGUMENT
    Decomposing the cone's +8.4 over NCM on ImageNet-R (exp25, R=4):
        NCM 71.02  ->  whitened prototypes 78.15  (+7.13, the METRIC)
                   ->  conic mixture      79.43  (+1.28, the CONE)
        RanPAC                            80.28
    Meanwhile RanPAC's random ReLU lift alone is worth +9.26 over NCM -- larger than
    everything else in the pipeline combined -- and EVERY cone we have built operates on the
    raw 768-d phi, UNDERNEATH it. At R=4 the cone reads energy in a 4-dimensional subspace of
    768; RanPAC reads a ridge in 10,000. That capacity gap is a complete candidate
    explanation for the constant -0.85 deficit, with no appeal to conic overlap.

    The lifted space is also NON-NEGATIVE by construction (post-ReLU), so non-negativity is
    native rather than imposed on a signed space -- the condition exp1 was built to test.

    CAVEAT ON exp1: `exp1_nonneg_native` found ReLU(XW) made cones WORSE. But it compared
    cone-vs-multiproto WITHIN such a space; it never compared cone-in-lifted against
    cone-in-raw, and the expansion factor is the whole point of random features. This is a
    different question, and the M sweep here is what settles it.

THE 2x2 THAT ISOLATES IT
                       raw phi          lifted relu(phi P)
        max cosine     pm_raw           pm_lift
        conic NNLS     cone_raw         cone_lift
    lift effect  = lifted - raw at fixed rule.  rule effect = cone - pm at fixed space.
    If only `pm_lift` gains, the lift helps everything and nothing is conic about it.

SIMPLIFICATION, STATED UP FRONT
    Generators are fitted ONCE at each class's birth and never re-metricated, so these arms
    use NO accumulated whitener. exp25 measured that accumulation is worth +1.9 over a frozen
    task-0 whitener, so `cone_w0` (task-0 whitener, raw space) is the fair raw counterpart
    and every lifted arm is handicapped by the same missing +1.9. Read lift effects, not
    absolute A-Last against the exp25 numbers.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 R=4 MS=2000,10000 python -u exp33_lifted_cone.py
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
RS = [int(x) for x in os.environ.get("R", "4").split(",")]
MS = [int(x) for x in os.environ.get("MS", "2000,10000").split(",")]
M_RP = int(os.environ.get("MRP", 10000))          # RanPAC's own lift, for the bar
LAMBDAS = [1e2, 1e3, 1e4]
SHRINK = float(os.environ.get("SHRINK", 1e-2))
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, f"exp33_lifted_cone_{TAG}.json")
EPS = 1e-12


def proj(d, M, seed=0):
    """RanPAC's projection, same generator and seed, so the lifted space the cone lives in
    IS the space RanPAC operates in."""
    return torch.randn(d, M, generator=torch.Generator().manual_seed(seed)).to(DEV)


def lift(X, P, bs=2048):
    """relu(X P), unit-normalised. Non-negative by construction, so every row lands in the
    non-negative orthant of the sphere -- native conic geometry, not an imposed constraint."""
    out = []
    for i in range(0, len(X), bs):
        h = torch.relu(torch.as_tensor(X[i:i + bs], device=DEV, dtype=torch.float32) @ P)
        out.append(h / (h.norm(dim=1, keepdim=True) + EPS))
    return torch.cat(out)


def gens_t(Ht, R, seed=0):
    """R k-means centroids of a (torch, GPU) block of unit rows; returned unit-normalised."""
    X = Ht.cpu().numpy()
    R = int(min(R, len(X)))
    A = (X.mean(0, keepdims=True) if R <= 1 else
         KMeans(R, n_init=4, random_state=seed).fit(X).cluster_centers_)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    return torch.as_tensor(A, device=DEV, dtype=torch.float32)


def cone_score_t(A, Q, iters=ITERS):
    """cos(q, Pi_C q) with FISTA on GPU, all tensors resident -- the raw-space helper would
    round-trip 10,000-d matrices through numpy for every class."""
    G = A @ A.T
    L = float(torch.linalg.eigvalsh(G.double())[-1].clamp(min=1e-12))
    C = Q @ A.T
    W = torch.zeros_like(C)
    Y = torch.zeros_like(C)
    t = 1.0
    for i in range(iters):
        Wp = W
        W = (Y - (Y @ G - C) / L).clamp_(min=0)
        tn = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        Y = W + ((t - 1.0) / tn) * (W - Wp)
        t = tn
        if i and i % 25 == 0 and torch.norm(W - Wp) / (torch.norm(Wp) + 1e-8) < 1e-7:
            break
    rec = W @ A
    return ((Q * rec).sum(1) / (rec.norm(dim=1) + EPS)).cpu().numpy()


def max_score_t(A, Q):
    return (Q @ A.T).max(1).values.cpu().numpy()


def run_cell(ds, T, seed, R):
    E.T, E.SEED = T, seed
    F = E.adapted_features(ds)
    assert F is not None, f"no exp16 cache for {ds} T={T} s={seed}"
    Ztr, Zte = F
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    Qv, Qt = Ztr[VAL_ALL], Zte

    # task-0 whitener, frozen (the raw-space counterpart; see the header simplification note)
    t0 = FIT[0]
    Xc = np.concatenate([Ztr[t0][ytr[t0] == c] - Ztr[t0][ytr[t0] == c].mean(0)
                         for c in tasks[0] if (ytr[t0] == c).sum() >= 2])
    S = (Xc.T @ Xc) / len(Xc)
    S += SHRINK * np.trace(S) / d * np.eye(d)
    Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)

    spaces = {"raw": (Ztr, Qv, Qt), "w0": (E.un(Ztr @ Wh), E.un(Qv @ Wh), E.un(Qt @ Wh))}
    Ps = {M: proj(d, M) for M in MS}
    for M in MS:
        P = Ps[M]
        spaces[f"lift{M}"] = ("LIFT", M)          # lifted lazily, see below
    log(f"  {ds} T={T} s={seed} R={R}: spaces {list(spaces)}")

    # lifted query matrices, computed once and kept on GPU
    LQ = {}
    for M in MS:
        LQ[M] = (lift(Qv, Ps[M]), lift(Qt, Ps[M]))
        log(f"    lifted queries M={M}: val {tuple(LQ[M][0].shape)} test {tuple(LQ[M][1].shape)}")
        assert (LQ[M][1] >= -1e-6).all(), "lifted features must be non-negative"

    arms = ["pm_raw", "cone_raw", "pm_w0", "cone_w0"] + \
           [f"{a}_lift{M}" for M in MS for a in ("pm", "cone")]
    Sv = {a: np.full((len(Qv), n_cls), -np.inf, np.float32) for a in arms}
    St = {a: np.full((len(Qt), n_cls), -np.inf, np.float32) for a in arms}

    for c in range(n_cls):
        rows = np.concatenate([FIT[t][ytr[FIT[t]] == c] for t in range(T)
                               if (ytr[FIT[t]] == c).any()])
        if len(rows) < 2:
            continue
        for nm, key in (("raw", "raw"), ("w0", "w0")):
            Xs, qv, qt = spaces[key]
            A = gens_t(torch.as_tensor(Xs[rows], device=DEV, dtype=torch.float32), R, c)
            QV = torch.as_tensor(qv, device=DEV, dtype=torch.float32)
            QT = torch.as_tensor(qt, device=DEV, dtype=torch.float32)
            Sv[f"cone_{nm}"][:, c] = cone_score_t(A, QV)
            St[f"cone_{nm}"][:, c] = cone_score_t(A, QT)
            Sv[f"pm_{nm}"][:, c] = max_score_t(A, QV)
            St[f"pm_{nm}"][:, c] = max_score_t(A, QT)
        for M in MS:
            Hc = lift(Ztr[rows], Ps[M])
            A = gens_t(Hc, R, c)
            Sv[f"cone_lift{M}"][:, c] = cone_score_t(A, LQ[M][0])
            St[f"cone_lift{M}"][:, c] = cone_score_t(A, LQ[M][1])
            Sv[f"pm_lift{M}"][:, c] = max_score_t(A, LQ[M][0])
            St[f"pm_lift{M}"][:, c] = max_score_t(A, LQ[M][1])
        if c % 50 == 0:
            log(f"    fitted class {c}/{n_cls}")

    # ---- RanPAC bar, exp16's head ----
    P = proj(d, M_RP)
    Zn, Ztn = E.un(Ztr), E.un(Zte)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.as_tensor(X[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    res = {a: [] for a in arms + ["ranpac"]}
    nval = 0
    for t in range(T):
        for i, h in _H(Zn[FIT[t]]):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval += len(VAL[t])
        yv = ytr[VAL_ALL[:nval]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(Z, y, rows=None):
            Zs = Z if rows is None else Z[rows]
            return float((np.asarray(seen)[Zs[:, seen].argmax(1)] == y).mean())

        best, ba = -1.0, -1.0
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            lv = torch.cat([(h.double() @ Wm) for _, h in _H(Zn[VAL_ALL[:nval]])]).cpu().numpy()
            a = acc(lv, yv)
            if a > best:
                lt = torch.cat([(h.double() @ Wm) for _, h in _H(Ztn[tei])]).cpu().numpy()
                best, ba = a, float((np.asarray(seen)[lt[:, seen].argmax(1)] == yt).mean())
        res["ranpac"].append(ba)
        for a in arms:
            res[a].append(acc(St[a], yt, tei))
        log(f"    s{t}: " + "  ".join(f"{a} {res[a][-1]*100:.2f}"
                                      for a in ["ranpac", "cone_w0"] +
                                      [f"cone_lift{M}" for M in MS]))
    del G, C, P, eye
    torch.cuda.empty_cache()
    return {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
            for a, v in res.items()}


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            for R in RS:
                key = f"{ds}|{T}|{seed}|R{R}|MS{'-'.join(map(str, MS))}"
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, R)
                json.dump(allres, open(OUT, "w"), indent=2)

W = 84
print("\n" + "=" * W)
print("EXP33 — the cone above vs below RanPAC's random-feature lift")
print("=" * W)
for key, r in allres.items():
    rp = r["ranpac"]["A_last"]
    print(f"\n--- {key}")
    print(f"{'arm':<16}{'A-Last':>9}{'A-Avg':>9}{'vs ranpac':>11}")
    for a in sorted(r, key=lambda k: -r[k]["A_last"]):
        print(f"{a:<16}{r[a]['A_last']*100:>9.2f}{r[a]['A_avg']*100:>9.2f}"
              f"{(r[a]['A_last']-rp)*100:>+11.2f}")
    print("  lift effect (cone):  " + "  ".join(
        f"M={M}: {(r[f'cone_lift{M}']['A_last']-r['cone_raw']['A_last'])*100:+.2f}"
        for M in MS if f"cone_lift{M}" in r))
    print("  lift effect (max):   " + "  ".join(
        f"M={M}: {(r[f'pm_lift{M}']['A_last']-r['pm_raw']['A_last'])*100:+.2f}"
        for M in MS if f"pm_lift{M}" in r))
    print("  rule effect (cone-max): raw "
          f"{(r['cone_raw']['A_last']-r['pm_raw']['A_last'])*100:+.2f}  " + "  ".join(
              f"M={M}: {(r[f'cone_lift{M}']['A_last']-r[f'pm_lift{M}']['A_last'])*100:+.2f}"
              for M in MS if f"cone_lift{M}" in r))
print("\n" + "-" * W)
print("If only `pm_lift` gains, the lift helps every reader and nothing is conic about it.")
print("If `cone_lift` gains MORE than `pm_lift`, non-negativity is doing work in a space")
print("   where it is native rather than imposed -- the exp1 hypothesis, retested properly.")
print("No accumulated whitener in any arm (generators fitted once at birth), so compare")
print("   lift EFFECTS, not absolute A-Last against exp25's numbers.")
print("=" * W)
print(f"wrote {OUT}")
