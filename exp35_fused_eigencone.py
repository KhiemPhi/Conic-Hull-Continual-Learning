#!/usr/bin/env python3
"""exp35_fused_eigencone.py — the eigen-augmented cone fused with RanPAC, in the real replay.

THE GAP THIS CLOSES
    Neither existing script has both improvements:
        exp34  eigen-augmented generators (+0.28 matched budget, 79.77 -> 80.05) but NO fusion
        exp25  fusion with RanPAC (+0.70, 80.28 -> 80.98) but PLAIN k-means generators
    This runs both together inside exp16/exp25's staged replay, so the number is comparable
    to the 80.28 bar rather than to a simplified protocol.

WHAT THE GENERATORS ARE
    Per class, budget R held fixed, split between centroids and eigen-displacements:
        A_c = un([ m_1 .. m_nk ,  mu_c +- alpha*sqrt(lam_j) v_j  for j < ne ])   (nk + 2*ne = R)
    v_j / lam_j are the top eigenvectors of the WITHIN-class scatter in the whitened space.
    exp29 measured that the binding constraint is own-subspace coverage (rho -0.585), not
    between-class overlap (+0.037); centroids sit in the middle of the mass and span badly,
    while +-eigen displacements extend the cone along the directions the class varies in.
    exp34 found the gain is inverted-U in alpha (peak 0.5-1), so this is not coverage
    inflation -- there is a genuine optimum.

WHY GENERATORS ARE STORED IN THE ORIGINAL SPACE
    The tied whitener accumulates over tasks, so it changes every stage; refitting past
    classes would need their raw rows, which CIL forbids. Generators are built in the
    BIRTH-time whitened space, mapped back via Wh_birth^-1, stored there, and re-metricated
    as un(A_orig @ Wh_t). exp25 measured that approximation as lossless (oracle gap +0.05).

HONEST FORECAST
    exp25 showed the fusion gain did NOT improve when the cone got 1.9 points better via the
    accumulated whitener -- it stayed pinned at +0.2/+0.4. So these are probably NOT additive.
    Expect 81.0-81.3, not 81.7. And every number here is SEED 0; exp16's three-seed spread on
    this cell is 80.28/80.55/80.38 for RanPAC alone, so 81 on one seed is not 81.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 R=4 NK=2 NE=1 ALPHAS=0.5,1 python -u exp35_fused_eigencone.py
    DS=IMAGENETR T=10 SEED=0,1,2 R=4 NK=2 NE=1 ALPHAS=0.5 python -u exp35_fused_eigencone.py
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
R = int(os.environ.get("R", 4))
NK = int(os.environ.get("NK", 2))
NE = int(os.environ.get("NE", 1))
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0.5,1.0").split(",")]
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
SHRINK = float(os.environ.get("SHRINK", 3e-2))     # exp34's cone-optimal delta
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, f"exp35_fused_eigencone_{TAG}.json")
EPS = 1e-12
assert NK + 2 * NE == R, f"budget mismatch: nk={NK} + 2*ne={NE} != R={R}"


def zs(A, seen):
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def km(X, k, seed):
    k = int(min(k, len(X)))
    return E.un(X.mean(0, keepdims=True) if k <= 1 else
                KMeans(k, n_init=4, random_state=seed).fit(X).cluster_centers_)


def build_gens(Xw, nk, ne, alpha, c):
    """nk centroids + ne eigen-displaced pairs, total nk + 2*ne generators, all unit."""
    parts = [km(Xw, nk, c)] if nk > 0 else []
    if ne > 0:
        mu = Xw.mean(0)
        Y = Xw - mu
        sv, Vt = np.linalg.svd(Y, full_matrices=False)[1:]
        lam = (sv ** 2) / max(len(Y) - 1, 1)
        for j in range(min(ne, len(lam))):
            step = alpha * np.sqrt(max(lam[j], 0.0)) * Vt[j]
            parts.append(np.stack([mu + step, mu - step]))
    return E.un(np.concatenate(parts, 0))


def max_score(A, Q):
    """Naive multi-prototype: max cosine over the SAME atoms. Without this the conic rule
    is unattributable -- cone_km vs pm_km isolates the RULE at fixed atoms, exactly as
    eig vs km isolates the ATOMS at a fixed rule."""
    return (E.un(Q) @ E.un(A).T).max(1)


def cone_score(A, Q):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = E.un(A)
    return h.score(Q)


def run_cell(ds, T, seed, alpha):
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

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.as_tensor(X[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def logits(X, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(X)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    Aeig, Akm = {}, {}
    arms = ["ranpac", "ncm", "pm_km", "pm_eig", "cone_km", "cone_eig",
            "fuse_pm_km", "fuse_pm_eig", "fuse_km", "fuse_eig"]
    res = {a: [] for a in arms}

    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        S = scatter / max(n_scat, 1)
        S = S + SHRINK * np.trace(S) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = E.un(Ztr[r] @ Wh)
            Aeig[c] = build_gens(Xw, NK, NE, alpha, c) @ Wh_inv
            Akm[c] = km(Xw, R, c) @ Wh_inv

        for i, h in _H(E.un(Ztr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        yv = ytr[VAL_ALL[:nval]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(Z, y):
            return float((np.asarray(seen)[Z[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(logits(E.un(Qv[:nval]), Wm), yv)
            if a > best:
                best, bw = a, Wm
        Lv = logits(E.un(Qv[:nval]), bw)
        Lt = logits(E.un(Qt), bw)[tei]

        Qvw, Qtw = E.un(Qv[:nval] @ Wh), E.un(Qt[tei] @ Wh)
        keys = ("km", "eig", "pm_km", "pm_eig")
        Cv = {k: np.full((nval, n_cls), -np.inf, np.float32) for k in keys}
        Ct = {k: np.full((len(tei), n_cls), -np.inf, np.float32) for k in keys}
        NCv = np.full((nval, n_cls), -np.inf, np.float32)
        NCt = np.full((len(tei), n_cls), -np.inf, np.float32)
        for c in seen:
            if c not in Aeig:
                continue
            for k, store in (("km", Akm), ("eig", Aeig)):
                Ac = E.un(store[c] @ Wh)
                Cv[k][:, c] = cone_score(Ac, Qvw)          # conic NNLS
                Ct[k][:, c] = cone_score(Ac, Qtw)
                Cv["pm_" + k][:, c] = max_score(Ac, Qvw)   # naive max-cosine, same atoms
                Ct["pm_" + k][:, c] = max_score(Ac, Qtw)
            mu = E.un((Akm[c] @ Wh).mean(0, keepdims=True))[0]
            NCv[:, c] = Qvw @ mu
            NCt[:, c] = Qtw @ mu

        # Classes too small to yield generators (len(r) < 2) keep a -inf score column.
        # zs() maps -inf to the ROW MINIMUM, so in the fusion zL + b*zS those classes get
        # actively SUPPRESSED -- including ones RanPAC would have called correctly. On
        # ImageNet-A that is real damage (1 class has 0 fit rows, 6 have <4, 30 have <8);
        # it is a protocol artifact, not a property of the method. Neutralise them in the
        # FUSED score only (0 == the mean of the z-distribution, so the class falls back to
        # RanPAC alone). The RAW cone arms keep -inf, which is correct: the cone genuinely
        # cannot model a class it has no generators for.
        # No-op for CIFAR100 / ImageNet-R / CUB200 (min fit rows 450 / 35 / 26), so cached
        # v2 cells for those datasets remain valid; the log below fires if that is ever false.
        miss = [c for c in seen if c not in Aeig]
        if miss:
            log(f"      s{t}: {len(miss)} seen classes have NO generators -> "
                f"neutralised in fusion, -inf in raw arms")

        res["ranpac"].append(acc(zs(Lt, seen), yt))
        res["ncm"].append(acc(zs(NCt, seen), yt))
        for k, nm in (("km", "cone_km"), ("eig", "cone_eig"),
                      ("pm_km", "pm_km"), ("pm_eig", "pm_eig")):
            res[nm].append(acc(zs(Ct[k], seen), yt))
            zLv, zSv = zs(Lv, seen), zs(Cv[k], seen)
            zLt, zSt = zs(Lt, seen), zs(Ct[k], seen)
            if miss:
                zSv[:, miss] = 0.0
                zSt[:, miss] = 0.0
            b = max(BETAS, key=lambda bb: acc(zLv + bb * zSv, yv))
            res[f"fuse_{k}" if k.startswith("pm_") else f"fuse_{k}"].append(
                acc(zLt + b * zSt, yt))
        log(f"    s{t}: " + "  ".join(f"{a} {res[a][-1]*100:.2f}" for a in
                                       ("ranpac", "pm_km", "cone_km", "pm_eig",
                                        "cone_eig", "fuse_eig")))
    del G, C, P, eye
    torch.cuda.empty_cache()
    for a, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{a} out of range"
    return {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
            for a, v in res.items()}


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            for alpha in ALPHAS:
                key = (f"{ds}|{T}|{seed}|R{R}_k{NK}e{NE}_a{alpha:g}"
                       f"|m{M_RP}_s{SHRINK:g}_i{ITERS}|v2")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, alpha)
                json.dump(allres, open(OUT, "w"), indent=2)

W = 86
print("\n" + "=" * W)
print("EXP35 — eigen-augmented cone, fused with RanPAC, in the staged replay")
print("=" * W)
for key, r in allres.items():
    rp = r["ranpac"]["A_last"]
    print(f"\n--- {key}")
    print(f"{'arm':<12}{'A-Last':>9}{'A-Avg':>9}{'vs ranpac':>11}")
    for a in ["ncm", "pm_km", "pm_eig", "cone_km", "cone_eig",
              "fuse_pm_km", "fuse_pm_eig", "fuse_km", "fuse_eig", "ranpac"]:
        if a not in r:
            continue
        print(f"{a:<12}{r[a]['A_last']*100:>9.2f}{r[a]['A_avg']*100:>9.2f}"
              f"{(r[a]['A_last']-rp)*100:>+11.2f}")
    if "pm_km" in r:
        g = lambda k: r[k]["A_last"] * 100
        print(f"\n  RULE  (cone - max, same atoms):  km {g('cone_km')-g('pm_km'):+.2f}"
              f"   eig {g('cone_eig')-g('pm_eig'):+.2f}")
        print(f"  ATOMS (eig - km, same rule):     max {g('pm_eig')-g('pm_km'):+.2f}"
              f"   cone {g('cone_eig')-g('cone_km'):+.2f}")
        print(f"  fused:  rule {g('fuse_eig')-g('fuse_pm_eig'):+.2f}"
              f"   atoms {g('fuse_eig')-g('fuse_km'):+.2f}")
print("\n" + "-" * W)
print("`ranpac` must reproduce the exp16 bar (ImageNet-R T=10 s0: 80.28). If it does not,")
print("   the replay is broken and nothing else on the line means anything.")
print("`cone_km` is exp25's plain arm at the same R, so cone_eig - cone_km is the eigen")
print("   contribution measured INSIDE the replay, paired.")
print("exp25 found the fusion gain did not track cone quality, so if fuse_eig - fuse_km is")
print("   ~0 while cone_eig - cone_km is positive, the two improvements are not additive.")
print("=" * W)
print(f"wrote {OUT}")
