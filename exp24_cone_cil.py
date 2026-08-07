#!/usr/bin/env python3
"""exp24_cone_cil.py — does cone_w survive the REAL CIL protocol, against the RanPAC bar?

WHAT exp21 DID NOT SHOW
    exp21 scored the final stage only, with a prototype head, and reported 79.77 on
    ImageNet-R. The actual bar is A_plus + RanPAC at 80.41 (exp16, T=10, mean of 3 seeds).
    Those are not comparable: different head, no staged replay, no validation protocol.
    This script runs cone_w inside exp16/exp17's exact replay so the comparison is paired.

PROTOCOL (identical to exp16/exp17 in every respect that matters)
    class order = rng(SEED).permutation(n_cls); T tasks of cpt classes; per-task 10% val
    carve-out at rng(t); RanPAC accumulated with lambda chosen on the val carve-out from
    {1e2,1e3,1e4}; A_t measured over the classes seen so far. Because A_plus freezes the
    backbone, per-class objects are fitted ONCE at their birth stage and the
    (n_query x n_cls) score matrices are computed once and sliced per stage.

THE WHITENER IS PROTOCOL-LEGAL
    The tied within-class covariance is estimated from TASK 0 FIT ROWS ONLY and then frozen,
    exactly mirroring A_plus's "adapt on the first session, then freeze". Using all classes
    would leak future tasks. It is ONE shared 768x768 matrix (2.4 MB), not per-class state.

ARMS
    ncm         nearest class mean                                   floor
    ranpac      the bar, recomputed in-run so every number is paired
    protomaha   R whitened k-means prototypes per class, max cosine
    cone_w      R whitened k-means GENERATORS per class, cos(q, NNLS(q))   <- exp21's winner
    fuse        zs(ranpac) + beta*zs(cone_w), beta picked on val (beta=0 is in the grid, so
                this arm can only lose by failing to transfer val -> test)

STORAGE (the claim this method has to make)
    cone_w / protomaha: R vectors per class + one shared 768x768 whitener.
    At R=2 on 200 classes that is 200*2*768 + 768^2 floats = 2.5 MB, against the 473 MB of
    per-class covariance the compared methods store. RanPAC stores no per-class state but a
    10000x10000 Gram.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 R=2 python -u exp24_cone_cil.py
    DS=IMAGENETR T=10 SEED=0,1,2 RS=2,4 python -u exp24_cone_cil.py
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
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
TAG = MODEL.split(".")[-1]
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
RS = [int(x) for x in os.environ.get("RS", os.environ.get("R", "2,4")).split(",")]
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
SHRINK = float(os.environ.get("SHRINK", 1e-2))
ITERS = int(os.environ.get("ITERS", 500))
MIN_N = 2
OUT = os.path.join(REPO, f"exp24_cone_cil_{TAG}.json")
BARS = {}
_bp = os.path.join(REPO, f"exp16_full_table_{TAG}.json")
if os.path.exists(_bp):
    BARS = json.load(open(_bp))
ARMS = ["ncm", "ranpac", "protomaha", "cone_w", "fuse"]


def zs(A, seen):
    """Row z-score over SEEN columns only; unseen get a finite floor."""
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
    assert len(Ztr) == len(ytr) and len(Zte) == len(yte), "feature/label mismatch"
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
    FIT_ALL, VAL_ALL = np.concatenate(FIT), np.concatenate(VAL)

    # ---- tied whitener from TASK 0 FIT ROWS ONLY, then frozen (protocol-legal) ----
    t0 = FIT[0]
    Xc = np.concatenate([Ztr[t0][ytr[t0] == c] - Ztr[t0][ytr[t0] == c].mean(0)
                         for c in tasks[0] if (ytr[t0] == c).sum() >= 2])
    S = (Xc.T @ Xc) / len(Xc)
    S += SHRINK * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)
    log(f"  whitener from task 0 only: {len(Xc)} rows, {len(tasks[0])} classes")

    Qv, Qt = Ztr[VAL_ALL], Zte
    Qvw, Qtw = E.un(Qv @ Wh), E.un(Qt @ Wh)
    sizes = [int((ytr[FIT_ALL] == c).sum()) for c in range(n_cls)]

    # ---- per-class objects, fitted once at birth (backbone is frozen) ----
    NCMv, NCMt = [np.full((len(q), n_cls), -np.inf, np.float32) for q in (Qv, Qt)]
    PMv, PMt = [np.full((len(q), n_cls), -np.inf, np.float32) for q in (Qv, Qt)]
    CWv, CWt = [np.full((len(q), n_cls), -np.inf, np.float32) for q in (Qv, Qt)]
    for c in range(n_cls):
        rows = FIT_ALL[ytr[FIT_ALL] == c]
        if len(rows) < MIN_N:
            continue
        X = Ztr[rows]
        mu = E.un(X.mean(0, keepdims=True))
        NCMv[:, c] = E.un(Qv) @ mu[0]
        NCMt[:, c] = E.un(Qt) @ mu[0]
        Xw = E.un(X @ Wh)
        k = int(min(R, len(Xw)))
        A = (E.un(Xw.mean(0, keepdims=True)) if k <= 1 else
             E.un(KMeans(k, n_init=4, random_state=0).fit(Xw).cluster_centers_))
        PMv[:, c] = (Qvw @ A.T).max(1)
        PMt[:, c] = (Qtw @ A.T).max(1)
        h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
        h.extreme_rays_ = A                       # already whitened; do NOT set h.whiten
        CWv[:, c] = h.score(Qvw)
        CWt[:, c] = h.score(Qtw)
    log(f"  fitted {n_cls} per-class objects at R={R}")

    # ---- RanPAC, accumulated, lambda on val (exp16's head) ----
    P = torch.randn(Ztr.shape[1], M_RP,
                    generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.tensor(X[i:i + bs], device=DEV,
                                             dtype=torch.float32) @ P)

    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def logits(X, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(X)]).cpu().numpy()

    res = {a: [] for a in ARMS}
    nval = 0
    for t in range(T):
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

        res["ncm"].append(acc(NCMt, yt, tei))
        res["ranpac"].append(acc(Lt, yt, tei))
        res["protomaha"].append(acc(PMt, yt, tei))
        res["cone_w"].append(acc(CWt, yt, tei))
        zLv, zSv = zs(Lv, seen), zs(CWv[vs], seen)
        zLt, zSt = zs(Lt[tei], seen), zs(CWt[tei], seen)
        bb, bv = 0.0, -1.0
        for b in BETAS:
            a = acc(zLv + b * zSv, yv)
            if a > bv:
                bv, bb = a, b
        res["fuse"].append(acc(zLt + bb * zSt, yt))
    del G, C, P, eye
    torch.cuda.empty_cache()

    # ---- invariants ----
    for a, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{a}: accuracy out of range"
    if R == 1:
        assert abs(res["cone_w"][-1] - res["protomaha"][-1]) < 1e-9, \
            "R=1: a 1-generator cone must equal the whitened prototype exactly"
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
                r = allres[key]
                log("  " + "  ".join(f"{a} {r[a]['A_last']*100:.2f}" for a in ARMS))

W = 92
print("\n" + "=" * W)
print(f"EXP24 — cone_w inside the real CIL replay ({MODEL})")
print("=" * W)
for key, r in allres.items():
    ds, T, seed, Rk = key.split("|")[:4]
    bar = BARS.get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
    print(f"\n--- {key}" + (f"   [exp16 A_plus bar: A-Last {bar['A_last']*100:.2f} "
                            f"A-Avg {bar['A_avg']*100:.2f}]" if bar else ""))
    print(f"{'arm':<12}{'A-Last':>9}{'A-Avg':>9}{'vs ranpac':>11}")
    rp = r["ranpac"]["A_last"]
    for a in sorted(r, key=lambda k: -r[k]["A_last"]):
        print(f"{a:<12}{r[a]['A_last']*100:>9.2f}{r[a]['A_avg']*100:>9.2f}"
              f"{(r[a]['A_last']-rp)*100:>+11.2f}")
print("\n" + "-" * W)
print("'ranpac' is recomputed in-run and should match the exp16 bar; if it does not, the")
print("   replay is broken and nothing else on that line means anything.")
print("'fuse' has beta=0 in its grid, so it can only lose by failing to transfer val->test.")
print("=" * W)
print(f"wrote {OUT}")
