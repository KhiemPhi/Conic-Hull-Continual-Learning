#!/usr/bin/env python3
"""exp43_hard_negatives.py — weight S_F toward the classes that actually confuse c.

THE ARGUMENT
    Since E[||Pq||^2] = tr(P^T S P), maximising tr(P^T (S_c - gamma S_F) P) is PROVABLY the
    optimal rank-R subspace for the gap between own-class and *AVERAGE* foreign squared
    score. It is not a heuristic. But a classification error does not happen when the
    average foreign class scores high -- it happens when ONE foreign class beats the true
    one. The criterion optimises the mean; the errors come from the max.

    exp40 (on the current k5 model) measured that the max is extremely concentrated:
        share of a class's errors going to its TOP-1 confuser :  29.3%
        share going to its TOP-3 confusers                    :  61.3%
        distinct wrong classes per true class: median 4   (chance: 199)
    A median of FOUR confusers out of 199. Pooling S_F uniformly gives each of them
    ~1/200 of the weight, diluting the only mechanism that has ever worked by ~50x.

THE THREE ARMS, and why the middle one has to exist
    base    S_F is an item-proportional subsample of the foreign pool -- EXACTLY the
            current method. Must reproduce k5m24's 80.07.
    ucls    equal total weight per foreign CLASS.
    conf<t> weight per class by softmax(o_cc' / t), o = confusability.

    base and ucls differ because the pool is INHOMOGENEOUS: current-task classes
    contribute ~2160 raw ROWS while each past class contributes ~26 RAYS, so an
    item-proportional draw is dominated by the current task, and that mix drifts every
    stage (~100% rows at task 0, ~27% at task 9). That is a real confound inside gamma,
    separate from confusability. Without `ucls` in between, conf - base would conflate
    "weight the hard negatives" with "stop letting the current task dominate".
        conf - ucls  isolates hard negatives.
        ucls - base  isolates homogenisation.

CONFUSABILITY
    o(c,c') = mean over c's fit rows x of  max over c's material f of (x . f)
    i.e. how well class c' already covers class c's data. Uses stored rays for past
    classes and fit rows for current-task ones -- no past images. Computed at birth
    against every class seen so far, which is all that is legal.

THE FAILURE MODE TO WATCH
    Hard-negative mining collapses in metric learning: weight too sharply and the subspace
    specialises against two or three classes and loses to the other 197. That is what the
    TAU sweep is for, and `ucls` (tau -> inf) is the control that catches it. If every
    finite TAU is below ucls, the mechanism is real but the weighting is too sharp; if
    conf ~ ucls at every TAU, concentration did not translate.

RAW CONE ONLY -- no fusion, no calibration, no beta. Allocation is k5m24, exp41's best
(80.07 A-Last, 26.5 rays/class, 0.84x storage).

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp43_hard_negatives.py
    DS=IMAGENETR T=10 SEED=0 ARMS=base,ucls,conf0.05 python -u exp43_hard_negatives.py
"""
import json
import os
import re
import time

import numpy as np
import torch

import exp19_dataset_hull as E
import exp39_cone_construction as X

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
ARMS = os.environ.get("ARMS", "base,ucls,conf0.1,conf0.05,conf0.02").split(",")
METHOD = os.environ.get("METHOD", "opca")
GAMMA = float(os.environ.get("GAMMA", 0.5))
KDIV = float(os.environ.get("KDIV", 5))
RMIN = int(os.environ.get("RMIN", 24))         # k5m24 = exp41's best allocation
RMAX = int(os.environ.get("RMAX", 128))
O_ROWS = int(os.environ.get("O_ROWS", 50))     # rows used to estimate confusability
F_MAX = int(os.environ.get("F_MAX", 2000))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
OUT = os.path.join(REPO, f"exp43_hard_negatives_{TAG}.json")

un = X.un


def parse(a):
    if a in ("base", "ucls"):
        return a, 0.0
    m = re.match(r"conf([\d.]+)$", a)
    assert m, f"bad arm {a!r}; want base | ucls | conf<tau>"
    return "conf", float(m.group(1))


def run_cell(ds, T, seed):
    E.T, E.SEED = T, seed
    assert (E.T, E.SEED) == (T, seed)
    Ztr, Zte = E.adapted_features(ds)
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

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def project(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    RAY = {a: {} for a in ARMS}                  # arm -> class -> rays (ORIGINAL space)
    res = {a: [] for a in ["ranpac"] + ARMS}
    diag = {a: [] for a in ARMS}

    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        S_ = scatter / max(n_scat, 1)
        S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)
        rng = np.random.default_rng(1234 + t)

        for arm in ARMS:
            kind, tau = parse(arm)
            # per-class foreign material available RIGHT NOW: stored rays for past
            # classes of THIS arm, fit rows for the other classes of the current task.
            mat = {o: un(RAY[arm][o] @ Wh) for o in RAY[arm] if o not in tasks[t]}
            for c in tasks[t]:
                rc = FIT[t][ytr[FIT[t]] == c]
                if len(rc) < 2:
                    continue
                Xw = un(Ztr[rc] @ Wh)
                Rc = int(np.clip(len(rc) / KDIV, RMIN, RMAX))
                pool = dict(mat)
                for o in tasks[t]:
                    if o == c:
                        continue
                    ro = FIT[t][ytr[FIT[t]] == o]
                    if len(ro) >= 2:
                        pool[int(o)] = un(Ztr[ro] @ Wh)
                Fw = np.zeros((0, d), np.float32)
                if pool and GAMMA > 0:
                    keys = sorted(pool)
                    if kind == "base":                      # item-proportional == current
                        w = np.array([len(pool[k]) for k in keys], np.float64)
                    elif kind == "ucls":                    # equal per class
                        w = np.ones(len(keys))
                    else:                                   # confusability-weighted
                        Xs = Xw[:O_ROWS]
                        o = np.array([float((Xs @ pool[k].T).max(1).mean())
                                      for k in keys])
                        o = (o - o.max()) / max(tau, 1e-6)
                        w = np.exp(o)
                        diag[arm].append(float(np.sort(w / w.sum())[::-1][:3].sum()))
                    w = w / w.sum()
                    # draw F_MAX items, class chosen by w, item uniform within class.
                    # Same budget for every arm, so only the MIX differs. Flattened into
                    # one gather -- the per-item python loop this replaces cost ~20 s/run.
                    sz = np.array([len(pool[k]) for k in keys])
                    allm = np.concatenate([pool[k] for k in keys], 0)
                    off = np.concatenate([[0], np.cumsum(sz)[:-1]])
                    ci = rng.choice(len(keys), F_MAX, p=w)
                    Fw = allm[off[ci] + (rng.random(F_MAX) * sz[ci]).astype(int)]
                RAY[arm][c] = X.BUILD[METHOD](Xw, Fw, Rc, int(c), GAMMA) @ Wh_inv

        for i, h in _H(un(Ztr[FIT[t]])):
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
        sa = np.asarray(seen)

        def acc(Z, y):
            return float((sa[Z[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(project(un(Ztr[VAL_ALL[:nval]]), Wm), yv)
            if a > best:
                best, bw = a, Wm
        res["ranpac"].append(acc(project(un(Zte[tei]), bw), yt))

        Qw = un(Zte[tei] @ Wh)
        for arm in ARMS:
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            for c in seen:
                if c in RAY[arm]:
                    St[:, c] = X.cone_score(un(RAY[arm][c] @ Wh), Qw)
            res[arm].append(acc(St, yt))
        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}" + "".join(
            f" | {a} {res[a][-1]*100:.2f}" for a in ARMS))

    del G, C, P, eye
    torch.cuda.empty_cache()
    out = {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for a, v in res.items()}
    for a in ARMS:
        if diag[a]:
            out[a]["top3_weight_share"] = float(np.mean(diag[a]))
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            key = (f"{ds}|{T}|{seed}|{METHOD}g{GAMMA:g}k{KDIV:g}m{RMIN}|{'+'.join(ARMS)}"
                   f"|f{F_MAX}_o{O_ROWS}|m{M_RP}_s{SHRINK:g}|v1")
            if key in allres:
                log(f"skip {key}"); continue
            log(f"=== {key}")
            allres[key] = run_cell(ds, T, seed)
            json.dump(allres, open(OUT, "w"), indent=2)

W = 86
print("\n" + "=" * W)
print("EXP43 — confusability-weighted S_F (hard negatives), raw cone")
print("=" * W)
for key, r in sorted(allres.items()):
    print(f"\n--- {key}")
    print(f"  {'arm':<12}{'A-Last':>9}{'A-Avg':>9}{'top3 w':>9}{'vs base':>10}")
    b = r["base"]["A_last"] * 100 if "base" in r else None
    for a in ARMS:
        v = r[a]
        dl = f"{v['A_last']*100-b:>+10.2f}" if b is not None else f"{'--':>10}"
        print(f"  {a:<12}{v['A_last']*100:>9.2f}{v['A_avg']*100:>9.2f}"
              f"{v.get('top3_weight_share', float('nan')):>9.2f}{dl}")
    print(f"  {'[ranpac]':<12}{r['ranpac']['A_last']*100:>9.2f}"
          f"{r['ranpac']['A_avg']*100:>9.2f}")
    g = {a: r[a]["A_last"] * 100 for a in ARMS if a in r}
    if "ucls" in g:
        for a in ARMS:
            if a.startswith("conf"):
                print(f"\n  HARD NEGATIVES   {a} - ucls = {g[a]-g['ucls']:+.2f}"
                      f"   <- isolates confusability")
        if "base" in g:
            print(f"  HOMOGENISATION   ucls - base = {g['ucls']-g['base']:+.2f}"
                  f"   <- isolates the row/ray mix")
print("\n" + "-" * W)
print("ranpac must be 80.28. `base` should land NEAR exp41's k5m24 (80.07) but will not")
print("   match exactly: it draws the same item-proportional distribution through a")
print("   different RNG path, so S_F differs by sampling noise. A gap beyond ~0.2 means")
print("   the reimplementation is wrong, not that the sampling changed anything.")
print("conf - ucls IS THE RESULT. conf - base would conflate hard negatives with")
print("   homogenising an S_F whose row/ray mix drifts from ~100% rows at task 0 to")
print("   ~27% at task 9 -- a confound sitting inside gamma, independent of this idea.")
print("top3 w = mean share of the S_F weight on a class's three heaviest negatives.")
print("   Near 1.0 means the softmax has collapsed onto a handful of classes, which is")
print("   the known failure mode of hard-negative mining; expect conf to fall below ucls")
print("   at the sharpest TAU even if the mechanism is real.")
print("=" * W)
print(f"wrote {OUT}")
