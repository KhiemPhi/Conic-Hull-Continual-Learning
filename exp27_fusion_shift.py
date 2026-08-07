#!/usr/bin/env python3
"""exp27_fusion_shift.py — why does the cone/RanPAC fusion gain collapse at the last stage?

THE OBSERVATION THIS EXISTS TO EXPLAIN
    exp25, `fuse - ranpac` per stage at R=16:
        0.00  0.00  0.19  0.87  0.70  0.94  0.97  0.90  0.46  0.32
    An inverted U at every R. A-Last -- the headline metric -- is the WORST stage.
    Channel quality only explains half of it: cone_wa - ranpac widens from -1.22 (s6) to
    -1.55 (s9), a 0.33 degradation against a 0.65 collapse in the gain.

    Two candidate causes, and they demand opposite responses:
      (a) VAL TRANSFER   beta is chosen on the val carve-out and stops generalising late.
                         -> fixable: shrink beta toward the previous stage, or regularise.
      (b) REDUNDANCY     the two channels converge as RanPAC sees more data, so there is
                         genuinely nothing left to fuse.  -> no combiner can help; stop.

WHAT IS MEASURED, PER STAGE
    beta_val / beta_test    the chosen beta and the beta that WOULD have been best on test.
                            acc(beta_test) - acc(beta_val) is the val-transfer cost, i.e. (a).
    oracle_sel              per sample, credit if EITHER channel is right. This is the hard
                            ceiling on any fusion whatsoever; oracle_sel - ranpac is the
                            total headroom and `fuse` captures some fraction of it.
    both / only_r / only_c / neither
                            the confusion decomposition. `only_c` IS the headroom -- if it
                            shrinks late, that is (b) and it is decisive.
    corr                    mean per-query correlation between the two z-scored channels
                            across seen classes. The direct redundancy measure.
    conc                    concentration of the NNLS weights, max(w)/sum(w), as a THIRD
                            channel. exp25 throws the coefficient vector away and keeps only
                            ||Pi_C q||; this tests whether the discarded part carries
                            anything (problem 2), at the cost of one extra matrix.

READING IT
    only_c stays flat and the transfer cost grows  -> (a), fix the beta schedule.
    only_c shrinks toward zero                     -> (b), the ceiling is real, stop tuning.
    oracle_sel - ranpac stays large while fuse does not  -> the combiner is the problem, not
                                                            the channels.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 RS=16 python -u exp27_fusion_shift.py
    DS=IMAGENETR T=20 SEED=0 RS=16 python -u exp27_fusion_shift.py   # more stages
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
SHRINK = float(os.environ.get("SHRINK", 1e-2))
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, f"exp27_fusion_shift_{TAG}.json")
EPS = 1e-12


def zs(A, seen):
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def cone_score_conc(A, Q):
    """One NNLS solve, two outputs: cos(q, Pi_C q) and the weight concentration
    max(w)/sum(w). The second is the part exp25 discards."""
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = E.un(A)
    W = h._reconstruct_norm(E.un(Q))
    rec = W @ h.extreme_rays_
    s = (E.un(Q) * (rec / (np.linalg.norm(rec, axis=1, keepdims=True) + EPS))).sum(1)
    return s, W.max(1) / (W.sum(1) + EPS)


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

        def acc(L, y):
            return float((np.asarray(seen)[L[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(logits(Qv[vs], Wm), yv)
            if a > best:
                best, bw = a, Wm
        Lv, Lt = logits(Qv[vs], bw), logits(Qt, bw)[tei]

        Qvw, Qtw = E.un(Qv[vs] @ Wh), E.un(Qt[tei] @ Wh)
        CWv = np.full((nval, n_cls), -np.inf, np.float32)
        CWt = np.full((len(tei), n_cls), -np.inf, np.float32)
        CCv = np.full((nval, n_cls), -np.inf, np.float32)
        CCt = np.full((len(tei), n_cls), -np.inf, np.float32)
        for c in seen:
            if c not in A_orig:
                continue
            Ac = E.un(A_orig[c] @ Wh)
            CWv[:, c], CCv[:, c] = cone_score_conc(Ac, Qvw)
            CWt[:, c], CCt[:, c] = cone_score_conc(Ac, Qtw)

        zLv, zSv, zKv = zs(Lv, seen), zs(CWv, seen), zs(CCv, seen)
        zLt, zSt, zKt = zs(Lt, seen), zs(CWt, seen), zs(CCt, seen)

        a_r, a_c = acc(zLt, yt), acc(zSt, yt)
        pr = np.asarray(seen)[zLt[:, seen].argmax(1)]
        pc = np.asarray(seen)[zSt[:, seen].argmax(1)]
        ok_r, ok_c = pr == yt, pc == yt

        bv = max(BETAS, key=lambda b: acc(zLv + b * zSv, yv))
        bt = max(BETAS, key=lambda b: acc(zLt + b * zSt, yt))     # oracle beta (diagnostic)
        f_val, f_test = acc(zLt + bv * zSt, yt), acc(zLt + bt * zSt, yt)
        bk = max(BETAS, key=lambda b: acc(zLv + bv * zSv + b * zKv, yv))
        f_conc = acc(zLt + bv * zSt + bk * zKt, yt)

        a, b_ = zLt[:, seen], zSt[:, seen]
        corr = float((((a - a.mean(1, keepdims=True)) * (b_ - b_.mean(1, keepdims=True))).sum(1)
                      / (np.linalg.norm(a - a.mean(1, keepdims=True), axis=1)
                         * np.linalg.norm(b_ - b_.mean(1, keepdims=True), axis=1) + EPS)).mean())

        row = dict(stage=t, n_seen=int(len(seen)), ranpac=a_r, cone=a_c,
                   beta_val=bv, beta_test=bt, fuse=f_val, fuse_oracle_beta=f_test,
                   transfer_cost=f_test - f_val, fuse_conc=f_conc, beta_conc=bk,
                   oracle_sel=float((ok_r | ok_c).mean()),
                   both=float((ok_r & ok_c).mean()), only_r=float((ok_r & ~ok_c).mean()),
                   only_c=float((~ok_r & ok_c).mean()),
                   neither=float((~ok_r & ~ok_c).mean()),
                   agree=float((pr == pc).mean()), corr=corr)
        assert abs(row["both"] + row["only_r"] + row["only_c"] + row["neither"] - 1) < 1e-9
        assert row["oracle_sel"] >= max(a_r, a_c) - 1e-9
        assert row["fuse_oracle_beta"] >= row["fuse"] - 1e-9
        rows.append(row)
        log(f"    s{t}: ranpac {a_r*100:.2f} cone {a_c*100:.2f} fuse {f_val*100:.2f} "
            f"(+{(f_val-a_r)*100:.2f})  b_val {bv:g} b_test {bt:g} "
            f"transfer {row['transfer_cost']*100:+.2f}  only_c {row['only_c']*100:.2f} "
            f"ceil {row['oracle_sel']*100:.2f}  corr {corr:.3f}")
    del G, C, P, eye
    torch.cuda.empty_cache()
    return rows


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            for R in RS:
                key = f"{ds}|{T}|{seed}|R{R}|m{M_RP}"
                if key in allres:
                    log(f"skip {key} (done)"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, R)
                json.dump(allres, open(OUT, "w"), indent=2)

W = 116
print("\n" + "=" * W)
print("EXP27 — where the fusion gain goes, stage by stage")
print("=" * W)
for key, rows in allres.items():
    print(f"\n--- {key}")
    print(f"{'s':>3}{'#cls':>6}{'ranpac':>8}{'cone':>8}{'fuse':>8}{'gain':>7}"
          f"{'b_val':>7}{'b_test':>7}{'transf':>8}{'only_c':>8}{'ceiling':>9}"
          f"{'captured':>10}{'corr':>7}{'+conc':>7}")
    for r in rows:
        head = r["oracle_sel"] - r["ranpac"]
        cap = (r["fuse"] - r["ranpac"]) / head * 100 if head > 1e-9 else float("nan")
        print(f"{r['stage']:>3}{r['n_seen']:>6}{r['ranpac']*100:>8.2f}{r['cone']*100:>8.2f}"
              f"{r['fuse']*100:>8.2f}{(r['fuse']-r['ranpac'])*100:>+7.2f}"
              f"{r['beta_val']:>7g}{r['beta_test']:>7g}{r['transfer_cost']*100:>+8.2f}"
              f"{r['only_c']*100:>8.2f}{r['oracle_sel']*100:>9.2f}{cap:>9.1f}%"
              f"{r['corr']:>7.3f}{(r['fuse_conc']-r['fuse'])*100:>+7.2f}")
print("\n" + "-" * W)
print("only_c  = fraction the cone gets right and RanPAC does not -- the actual headroom.")
print("ceiling = per-sample oracle over the two channels; NO fusion can exceed it.")
print("captured= what `fuse` recovers of (ceiling - ranpac).")
print("transf  = acc(oracle beta) - acc(val beta): the cost of choosing beta on val.")
print("+conc   = gain from adding the NNLS weight concentration as a third channel,")
print("          i.e. whether the coefficient vector exp25 discards carries anything.")
print("\nonly_c flat + transf growing -> val transfer; fix the beta schedule.")
print("only_c shrinking             -> the channels went redundant; no combiner helps.")
print("=" * W)
print(f"wrote {OUT}")
