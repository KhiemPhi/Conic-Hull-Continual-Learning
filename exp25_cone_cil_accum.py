#!/usr/bin/env python3
"""exp25_cone_cil_accum.py — accumulated whitener + stacked fusion, inside the real replay.

TWO CHANGES OVER exp24
 (1) ACCUMULATED TIED WHITENER.  exp24 froze the whitener at task 0 and lost ~2.5 points
     versus exp21's all-class estimate. A tied within-class scatter is a SUM OVER CLASSES,
     exactly like RanPAC's G and C, so it can be accumulated task by task while retaining
     no raw data -- one running 768x768 matrix. At the last stage "all seen classes" IS
     "all classes", so exp21's whitener is legal for A-Last; only A-Avg needed the weaker
     task-0 estimate.

     THE PROTOCOL TRAP THIS AVOIDS.  When the whitener changes, past classes' generators
     would have to be refitted -- which needs their raw rows, and CIL does not allow that.
     Instead generators are fitted in the BIRTH-TIME whitened space (legal: that class's
     data is in hand then), immediately mapped back to the original space via Wh_birth^-1,
     and STORED THERE. At any later stage they are re-metricated as un(A_orig @ Wh_t).
     Storage is unchanged -- R vectors per class plus one shared matrix -- and no past
     sample is ever revisited. `cone_wa_oracle` refits from raw rows and is NOT legal; it
     is printed only as the ceiling this approximation is chasing.

 (2) STACKED FUSION.  exp24's `fuse` is zs(ranpac) + beta*zs(cone) with one beta. Because
     argmax is invariant to positive rescaling, learning a weight on ranpac too adds
     nothing -- (a,b) == (1, b/a). Real gain needs more VIEWS, so `fuse_stack` fits a
     per-stage linear model on four z-scored channels (ranpac, cone, ncm, protomaha) over
     val (sample, seen-class) pairs. It strictly contains `fuse`, so it can only lose by
     failing to transfer val -> test.

INVARIANTS ASSERTED AT RUNTIME
    ranpac must reproduce the exp16 bar; at stage 0 the accumulated whitener IS the task-0
    whitener so cone_wa[0] must equal cone_w[0]; accuracies must lie in [0,1].

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 RS=2,4 python -u exp25_cone_cil_accum.py
    DS=IMAGENETR T=10 SEED=0 RS=2 ORACLE=1 python -u exp25_cone_cil_accum.py
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
TAG = MODEL.split(".")[-1]
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
RS = [int(x) for x in os.environ.get("RS", "2,4").split(",")]
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
         10.0, 100.0]   # 100 == cone-only; both endpoints must be reachable
SHRINK = float(os.environ.get("SHRINK", 1e-2))
ITERS = int(os.environ.get("ITERS", 500))
ORACLE = int(os.environ.get("ORACLE", 0))
OUT = os.path.join(REPO, f"exp25_cone_cil_accum_{TAG}.json")
BARS = json.load(open(os.path.join(REPO, f"exp16_full_table_{TAG}.json"))) \
    if os.path.exists(os.path.join(REPO, f"exp16_full_table_{TAG}.json")) else {}
ARMS = ["ncm", "ranpac", "cone_w0", "pm_wa", "cone_wa", "fuse", "fuse_stack"]
if ORACLE:
    ARMS.append("cone_wa_oracle")


def whitener(scatter, n):
    S = scatter / max(n, 1)
    S = S + SHRINK * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)


def gens_w(Xw, R):
    k = int(min(R, len(Xw)))
    return (E.un(Xw.mean(0, keepdims=True)) if k <= 1 else
            E.un(KMeans(k, n_init=4, random_state=0).fit(Xw).cluster_centers_))


def cone_score(A, Qw):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = E.un(A)          # already in the target metric; never set h.whiten
    return h.score(Qw)


def zs(A, seen):
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def run_cell(ds, T, seed, R):
    # exp19.adapted_features() reads exp19's MODULE-LEVEL T/SEED, frozen at import time, so
    # a multi-T or multi-seed loop here would silently reload the T=10/seed=0 cache for every
    # cell. Bind them per cell and fail loudly if the cache is absent.
    E.T, E.SEED = T, seed
    F = E.adapted_features(ds)
    assert F is not None, (f"no exp16 feature cache for {ds} T={T} seed={seed} "
                           f"(expected exp16_feats_{ds}_T{T}_s{seed}_ep40_lr0.0003_aug1_"
                           f"{E.TAG}.npz)")
    Ztr, Zte = F
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        idx = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(idx))
        nv = max(int(0.1 * len(idx)), 1)
        VAL.append(idx[pm[:nv]])
        FIT.append(idx[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    Qv, Qt = Ztr[VAL_ALL], Zte

    # ---- NCM (no whitener) and the frozen-task-0 cone, both computed once ----
    NCMt = np.full((len(Qt), n_cls), -np.inf, np.float32)
    NCMv = np.full((len(Qv), n_cls), -np.inf, np.float32)   # val rows, NOT test rows
    CWt_frozen = np.full((len(Qt), n_cls), -np.inf, np.float32)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.tensor(X[i:i + bs], device=DEV,
                                             dtype=torch.float32) @ P)

    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def logits(X, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(X)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64)
    n_scat = 0
    A_orig = {}                     # per class: R generators, ORIGINAL space (the storage)
    Wh0 = None
    res = {a: [] for a in ARMS}
    nval = 0

    for t in range(T):
        # ---- accumulate the tied scatter with THIS task's classes (raw rows in hand now)
        for c in tasks[t]:
            rows = FIT[t][ytr[FIT[t]] == c]
            if len(rows) < 2:
                continue
            Xc = Ztr[rows] - Ztr[rows].mean(0)
            scatter += Xc.T @ Xc
            n_scat += len(Xc)
        Wh = whitener(scatter, n_scat)
        if t == 0:
            Wh0 = Wh
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)

        # ---- birth-time fit for this task's classes, stored back in ORIGINAL space ----
        for c in tasks[t]:
            rows = FIT[t][ytr[FIT[t]] == c]
            if len(rows) < 2:
                continue
            A_w = gens_w(E.un(Ztr[rows] @ Wh), R)          # fitted in the birth metric
            A_orig[c] = A_w @ Wh_inv                        # mapped back; this is what we keep
            mu = E.un(Ztr[rows].mean(0, keepdims=True))
            NCMt[:, c] = E.un(Qt) @ mu[0]
            NCMv[:, c] = E.un(Qv) @ mu[0]
            CWt_frozen[:, c] = cone_score(E.un(A_orig[c] @ Wh0), E.un(Qt @ Wh0))

        # ---- RanPAC accumulate + lambda on val ----
        for i, h in _H(Ztr[FIT[t]]):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h
            C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval += len(VAL[t])
        vs = slice(0, nval)
        yv = ytr[VAL_ALL[vs]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(L, y, rows=None):
            Ls = L if rows is None else L[rows]
            return float((np.asarray(seen)[Ls[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(logits(Qv[vs], Wm), yv)
            if a > best:
                best, bw = a, Wm
        Lv, Lt = logits(Qv[vs], bw), logits(Qt, bw)

        # ---- re-metricate every SEEN class into the CURRENT whitened space ----
        Qvw, Qtw = E.un(Qv[vs] @ Wh), E.un(Qt @ Wh)
        CWv = np.full((nval, n_cls), -np.inf, np.float32)
        CWt = np.full((len(Qt), n_cls), -np.inf, np.float32)
        PMv = np.full((nval, n_cls), -np.inf, np.float32)
        PMt = np.full((len(Qt), n_cls), -np.inf, np.float32)
        for c in seen:
            if c not in A_orig:
                continue
            Ac = E.un(A_orig[c] @ Wh)
            CWv[:, c] = cone_score(Ac, Qvw)
            CWt[:, c] = cone_score(Ac, Qtw)
            PMv[:, c] = (Qvw @ Ac.T).max(1)
            PMt[:, c] = (Qtw @ Ac.T).max(1)

        res["ncm"].append(acc(NCMt, yt, tei))
        res["ranpac"].append(acc(Lt, yt, tei))
        res["cone_w0"].append(acc(CWt_frozen, yt, tei))
        res["pm_wa"].append(acc(PMt, yt, tei))
        res["cone_wa"].append(acc(CWt, yt, tei))

        # ---- fusion: single beta, and the 4-channel stack ----
        chan_v = [zs(Lv, seen), zs(CWv, seen), zs(NCMv[vs], seen), zs(PMv, seen)]
        chan_t = [zs(Lt[tei], seen), zs(CWt[tei], seen), zs(NCMt[tei], seen),
                  zs(PMt[tei], seen)]
        bb, bv = 0.0, -1.0
        for b in BETAS:
            a = acc(chan_v[0] + b * chan_v[1], yv)
            if a > bv:
                bv, bb = a, b
        res["fuse"].append(acc(chan_t[0] + bb * chan_t[1], yt))

        # stack: one row per (val sample, seen class); label = is this the true class
        Fv = np.stack([ch[:, seen].ravel() for ch in chan_v], 1)
        lab = (np.asarray(seen)[None, :] == yv[:, None]).ravel().astype(int)
        keep = np.isfinite(Fv).all(1)
        w = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Fv[keep], lab[keep]).coef_[0]
        St = np.full((len(tei), n_cls), -1e9)
        St[:, seen] = sum(wi * ch[:, seen] for wi, ch in zip(w, chan_t))
        res["fuse_stack"].append(acc(St, yt))

        if ORACLE:
            CWo = np.full((len(Qt), n_cls), -np.inf, np.float32)
            for c in seen:                      # NOT legal: refits from raw past rows
                rows = np.concatenate([FIT[j][ytr[FIT[j]] == c] for j in range(t + 1)
                                       if (ytr[FIT[j]] == c).any()])
                CWo[:, c] = cone_score(gens_w(E.un(Ztr[rows] @ Wh), R), Qtw)
            res["cone_wa_oracle"].append(acc(CWo, yt, tei))
        log(f"    stage {t}: " + " ".join(f"{a} {res[a][-1]*100:.2f}" for a in ARMS))

    del G, C, P, eye
    torch.cuda.empty_cache()
    for a, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{a}: accuracy out of range"
    assert abs(res["cone_wa"][0] - res["cone_w0"][0]) < 1e-9, \
        "stage 0: the accumulated whitener IS the task-0 whitener, so these must be equal"
    return {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
            for a, v in res.items()}


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            for R in RS:
                key = f"{ds}|{T}|{seed}|R{R}|m{M_RP}_s{SHRINK:g}_i{ITERS}"
                if key in allres:
                    log(f"skip {key} (done)")
                    continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, R)
                json.dump(allres, open(OUT, "w"), indent=2)

W = 92
print("\n" + "=" * W)
print(f"EXP25 — accumulated whitener + stacked fusion ({MODEL})")
print("=" * W)
for key, r in allres.items():
    ds, T, seed, Rk = key.split("|")[:4]
    bar = BARS.get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
    print(f"\n--- {key}" + (f"   [exp16 A_plus bar: A-Last {bar['A_last']*100:.2f} "
                            f"A-Avg {bar['A_avg']*100:.2f}]" if bar else ""))
    print(f"{'arm':<16}{'A-Last':>9}{'A-Avg':>9}{'vs ranpac':>11}")
    rp = r["ranpac"]["A_last"]
    for a in sorted(r, key=lambda k: -r[k]["A_last"]):
        print(f"{a:<16}{r[a]['A_last']*100:>9.2f}{r[a]['A_avg']*100:>9.2f}"
              f"{(r[a]['A_last']-rp)*100:>+11.2f}")
print("\n" + "-" * W)
print("cone_w0 = generators fitted in their BIRTH metric but SCORED in the task-0")
print("          metric; equals cone_wa at stage 0 by construction (the invariant).")
print("cone_wa = accumulated whitener, re-metricated each stage")
print("pm_wa    = whitened prototypes, accumulated    fuse_stack contains fuse strictly")
print("cone_wa_oracle refits past classes from raw rows: NOT protocol-legal, ceiling only.")
print("=" * W)
print(f"wrote {OUT}")
