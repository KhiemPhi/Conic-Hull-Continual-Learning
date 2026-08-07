#!/usr/bin/env python3
"""exp28_gated_fusion.py — can PLASTIC's mechanisms capture the headroom exp27 exposed?

WHAT exp27 ESTABLISHED (ImageNet-R, T=10, R=16)
    The two channels are NOT redundant -- only_c grows 0.79 -> 3.75 and their correlation
    FALLS 0.808 -> 0.500. The per-sample oracle ceiling at the last stage is 84.03 against
    RanPAC's 80.28, i.e. 3.75 points of headroom, and `fuse` captures 8.4% of it. It also
    showed that 74% of the last-stage collapse is beta failing to transfer from val
    (b_val=5 vs b_test=1.5, worth 0.48).

    So the bottleneck is the COMBINER, not the channels. A single global beta cannot express
    "trust the cone on THIS query", which is exactly what a growing only_c demands.

THREE MECHANISMS BORROWED FROM PLASTIC (Marouf et al., CoLLAs 2025)
  (1) GATE   PLASTIC minimises the entropy of an augmentation-marginal prediction. We do not
             want their gradient step -- we want the observation that per-sample confidence
             is a free reliability signal. beta becomes per-sample:
                 beta_i = beta0 * exp(g * d_i),   d_i = zscore(margin_cone - margin_ranpac)
             g=0 is in the grid, so this strictly contains `fuse` and cannot lose on val.
  (2) EMA    Their Table 4 (no reset 43.31 / reset 59.15 / reset+KL 91.57) says: adapt, but
             anchor to a stable reference. Our beta is re-picked from scratch every stage on
             a noisy val set. Anchor it instead:  beta_t <- alpha*beta_{t-1} + (1-alpha)*beta_t^val.
             alpha=0 reproduces the current behaviour exactly.
  (3) AUG    Their Eq. 1 marginalises predictions over M augmentations. We only have CACHED
             FEATURES, not images, and exp16 saves no checkpoints -- so the real mechanism
             (image transforms through the backbone) is NOT reproducible here. What is
             implemented is a FEATURE-SPACE SURROGATE: q_m = normalize(q + sigma*eps_m),
             scores averaged over M draws. It tests the variance-reduction principle only,
             NOT PLASTIC's invariance-to-input-transforms. Do not report it as PLASTIC's AUG.
             AUG_M=1 disables it and reproduces the un-augmented arm exactly.

ARMS
    ranpac cone_wa                      the two channels
    fuse                                current single global beta, re-picked per stage
    fuse_ema fuse_gate fuse_aug         one mechanism each
    fuse_all                            (1)+(2)+(3)
    oracle_sel                          per-sample best-of-two: the hard ceiling
    oracle_beta                         best global beta chosen ON TEST: the transfer bound

THE NUMBER THAT MATTERS is `captured` = (arm - ranpac) / (oracle_sel - ranpac). `fuse` sits
at 8.4% at the last stage. Anything that does not move that has not addressed the finding.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 RS=16 python -u exp28_gated_fusion.py
    DS=IMAGENETR T=10 SEED=0 RS=16 AUG_M=1 python -u exp28_gated_fusion.py   # skip (3)
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
RS = [int(x) for x in os.environ.get("RS", "16").split(",")]
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
GAINS = [0.0, 0.25, 0.5, 1.0, 2.0]          # gate slope; 0.0 == the global-beta arm
ALPHA = float(os.environ.get("ALPHA", 0.5))  # EMA anchor on beta across stages
AUG_M = int(os.environ.get("AUG_M", 4))      # feature-space draws; 1 disables
AUG_SIG = float(os.environ.get("AUG_SIG", 0.05))
SHRINK = float(os.environ.get("SHRINK", 1e-2))
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, f"exp28_gated_fusion_{TAG}.json")
EPS = 1e-12
ARMS = ["ranpac", "cone_wa", "fuse", "fuse_ema", "fuse_gate", "fuse_aug", "fuse_all",
        "oracle_beta", "oracle_sel"]


def zs(A, seen):
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def margin(Z, seen):
    """top1 - top2 over the seen columns: a per-query confidence for one channel."""
    s = np.sort(Z[:, seen], axis=1)
    return s[:, -1] - s[:, -2] if s.shape[1] > 1 else s[:, -1]


def cone_scores(A, Q):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = E.un(A)
    return h.score(Q)


def run_cell(ds, T, seed, R):
    E.T, E.SEED = T, seed
    F = E.adapted_features(ds)
    assert F is not None, f"no exp16 cache for {ds} T={T} seed={seed}"
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
        VAL.append(idx[pm[:nv]]); FIT.append(idx[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    Qv, Qt = Ztr[VAL_ALL], Zte

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

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A_orig = {}
    prev = None                      # (beta0, gain) carried across stages for the EMA arms
    res = {a: [] for a in ARMS}
    rows = []

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
            k = int(min(R, len(Xw)))
            Aw = (E.un(Xw.mean(0, keepdims=True)) if k <= 1 else
                  E.un(KMeans(k, n_init=4, random_state=0).fit(Xw).cluster_centers_))
            A_orig[c] = Aw @ Wh_inv

        for i, h in _H(Ztr[FIT[t]]):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vs = slice(0, nval)
        yv = ytr[VAL_ALL[vs]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(Z, y):
            return float((np.asarray(seen)[Z[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(logits(Qv[vs], Wm), yv)
            if a > best:
                best, bw = a, Wm

        # ---- channel score matrices, plain and feature-space-marginalised ----
        rng = np.random.default_rng(1234 + t)

        def channels(Xq, m):
            """(zs(ranpac), zs(cone)) for query block Xq, averaged over m feature draws.
            m=1 is the plain path, bit-for-bit."""
            Ls, Cs = [], []
            for j in range(m):
                Xj = Xq if j == 0 and m == 1 else E.un(
                    Xq + AUG_SIG * rng.normal(size=Xq.shape).astype(np.float32))
                Ls.append(logits(Xj, bw))
                Xjw = E.un(Xj @ Wh)
                Ct = np.full((len(Xj), n_cls), -np.inf, np.float32)
                for c in seen:
                    if c in A_orig:
                        Ct[:, c] = cone_scores(E.un(A_orig[c] @ Wh), Xjw)
                Cs.append(Ct)
            return np.mean(Ls, 0), np.mean(Cs, 0)

        Lv1, Cv1 = channels(Qv[vs], 1)
        Lt1, Ct1 = channels(Qt[tei], 1)
        if AUG_M > 1:
            LvA, CvA = channels(Qv[vs], AUG_M)
            LtA, CtA = channels(Qt[tei], AUG_M)
        else:
            LvA, CvA, LtA, CtA = Lv1, Cv1, Lt1, Ct1

        def fit_gate(Lv, Cv, Lt, Ct):
            """Grid-search (beta0, gain) on val; return the applied test score matrix.
            d_i is standardised with VAL statistics and that transform is reused on test --
            using test statistics would leak."""
            zLv, zSv = zs(Lv, seen), zs(Cv, seen)
            zLt, zSt = zs(Lt, seen), zs(Ct, seen)
            dv = margin(zSv, seen) - margin(zLv, seen)
            mu, sd = float(dv.mean()), float(dv.std() + 1e-8)
            dv = (dv - mu) / sd
            dt = ((margin(zSt, seen) - margin(zLt, seen)) - mu) / sd
            bb, gg, bv = 0.0, 0.0, -1.0
            for b0 in BETAS:
                for g in GAINS:
                    a = acc(zLv + (b0 * np.exp(g * dv))[:, None] * zSv, yv)
                    if a > bv:
                        bv, bb, gg = a, b0, g
            return (bb, gg), (zLv, zSv, dv), (zLt, zSt, dt)

        (b_g, g_g), (zLv, zSv, dv), (zLt, zSt, dt) = fit_gate(Lv1, Cv1, Lt1, Ct1)
        b_plain = max(BETAS, key=lambda b: acc(zLv + b * zSv, yv))       # gain forced to 0
        b_or = max(BETAS, key=lambda b: acc(zLt + b * zSt, yt))          # oracle, diagnostic

        # ---- EMA anchoring across stages ----
        if prev is None:
            b_ema, g_ema = b_plain, g_g
        else:
            b_ema = ALPHA * prev[0] + (1 - ALPHA) * b_plain
            g_ema = ALPHA * prev[1] + (1 - ALPHA) * g_g
        prev = (b_ema, g_ema)

        res["ranpac"].append(acc(zLt, yt))
        res["cone_wa"].append(acc(zSt, yt))
        res["fuse"].append(acc(zLt + b_plain * zSt, yt))
        res["fuse_ema"].append(acc(zLt + b_ema * zSt, yt))
        res["fuse_gate"].append(acc(zLt + (b_g * np.exp(g_g * dt))[:, None] * zSt, yt))
        res["oracle_beta"].append(acc(zLt + b_or * zSt, yt))

        if AUG_M > 1:
            (b_a, g_a), (zLvA, zSvA, dvA), (zLtA, zStA, dtA) = fit_gate(LvA, CvA, LtA, CtA)
            b_pa = max(BETAS, key=lambda b: acc(zLvA + b * zSvA, yv))
            res["fuse_aug"].append(acc(zLtA + b_pa * zStA, yt))
            res["fuse_all"].append(
                acc(zLtA + (b_ema * np.exp(g_ema * dtA))[:, None] * zStA, yt))
        else:
            res["fuse_aug"].append(res["fuse"][-1])
            res["fuse_all"].append(res["fuse_ema"][-1])

        okr = np.asarray(seen)[zLt[:, seen].argmax(1)] == yt
        okc = np.asarray(seen)[zSt[:, seen].argmax(1)] == yt
        res["oracle_sel"].append(float((okr | okc).mean()))

        assert res["oracle_sel"][-1] >= max(res["ranpac"][-1], res["cone_wa"][-1]) - 1e-9
        assert res["oracle_beta"][-1] >= res["fuse"][-1] - 1e-9
        rows.append(dict(stage=t, beta=b_plain, beta_ema=b_ema, beta_gate=b_g, gain=g_g))
        head = res["oracle_sel"][-1] - res["ranpac"][-1]
        log(f"    s{t}: " + " ".join(f"{a} {res[a][-1]*100:.2f}" for a in
                                     ("ranpac", "fuse", "fuse_ema", "fuse_gate",
                                      "fuse_aug", "fuse_all", "oracle_sel")) +
            f"  | b {b_plain:g} b_ema {b_ema:.2f} gate({b_g:g},{g_g:g})"
            f"  cap {100*(res['fuse_all'][-1]-res['ranpac'][-1])/max(head,1e-9):.1f}%")
    del G, C, P, eye
    torch.cuda.empty_cache()
    return {"arms": {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
                     for a, v in res.items()}, "betas": rows}


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            for R in RS:
                key = f"{ds}|{T}|{seed}|R{R}|m{M_RP}_a{ALPHA:g}_M{AUG_M}"
                if key in allres:
                    log(f"skip {key} (done)"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, R)
                json.dump(allres, open(OUT, "w"), indent=2)

W = 92
print("\n" + "=" * W)
print("EXP28 — gated / anchored / marginalised fusion vs the oracle ceiling")
print("=" * W)
for key, blob in allres.items():
    r = blob["arms"]
    rp, ceil = r["ranpac"]["A_last"], r["oracle_sel"]["A_last"]
    head = ceil - rp
    print(f"\n--- {key}   headroom (ceiling - ranpac) = {head*100:.2f}")
    print(f"{'arm':<14}{'A-Last':>9}{'A-Avg':>9}{'vs ranpac':>11}{'captured':>11}")
    for a in ARMS:
        g = r[a]["A_last"] - rp
        cap = f"{100*g/head:>10.1f}%" if head > 1e-9 else " " * 11
        print(f"{a:<14}{r[a]['A_last']*100:>9.2f}{r[a]['A_avg']*100:>9.2f}"
              f"{g*100:>+11.2f}{cap}")
print("\n" + "-" * W)
print("captured = fraction of (oracle_sel - ranpac) recovered. exp27 measured `fuse` at 8.4%")
print("   on the final stage; an arm that does not move that has not addressed the finding.")
print("fuse_gate contains fuse (gain=0 is in the grid); fuse_ema == fuse at ALPHA=0;")
print("   fuse_aug == fuse at AUG_M=1. So each mechanism can only lose by failing val->test.")
print("AUG is a FEATURE-SPACE surrogate, not PLASTIC's image-augmentation marginal -- exp16")
print("   saves features, not checkpoints, so the real mechanism is not reproducible here.")
print("=" * W)
print(f"wrote {OUT}")
