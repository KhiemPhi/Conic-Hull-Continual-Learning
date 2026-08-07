#!/usr/bin/env python3
"""exp42_old_class_prune.py — let OLD classes adapt to classes that arrive later.

THE ASYMMETRY
    oPCA picks class c's directions to maximise  own energy - gamma * foreign energy,
    using the foreign material available WHEN c IS BORN. Class 1 is therefore chosen in
    ignorance of classes 21..200, and never updated again. Later classes get to avoid
    earlier ones; earlier classes never get to avoid later ones. That is the last
    structural asymmetry in the method, and gamma -- the only intervention that has ever
    paid (+0.52) -- is exactly the mechanism it breaks.

WHAT IS LEGAL AFTER BIRTH
    Class c's images are gone. Its rays and their cluster weights are all that remain, so
    the ENTIRE edit space is:
        rotate within span(A_c)  -> a subspace score is invariant to it, and cone ~ sub
                                    was measured (-0.10), so this is a null
        rescale a ray            -> a cone is scale-invariant in each ray: exact null
        DROP a ray               -> the only edit that changes anything
    So old-class adaptation can only be PRUNING. That is not a design choice, it is what
    the geometry permits.

THE RULE
    Each time a task arrives, score every ray of every OLD class by how much it overlaps
    the new arrivals, and accumulate:
        leak_j += mean over new rays f of (a_j . f)_+
    Rays are kept by  z(w_j) - GP * z(leak_j)  with within-class z-scores (w_j = the
    cluster size behind that ray, so the dominant modes are protected and the two terms
    are on one scale). The budget shrinks geometrically with age:
        K_c(t) = max(RMIN_P, round(R_c * RHO^(t - birth_t)))

THE CONTROL THAT DECIDES IT
    exp41 measured that FEWER RAYS IS ALREADY BETTER: k12 at 9.7 rays/class scores 79.70
    against fixed R=32's 79.50. So any gain from pruning could be nothing but "smaller is
    better". `pr` prunes RANDOMLY to the IDENTICAL K_c(t) on every class at every stage.
        pd - pr  is the only number that means anything here.
    Conflating those two is how eigen-augmentation looked like a win for two rounds.

WHY THE FOREIGN-RAY PROXY SHOULD BE OK HERE
    cal_bg used foreign rays as a background estimate and failed badly (-1.57): rays are
    denoised centroids, not typical queries, so the offset was biased -- and it was applied
    as a fine-grained numerical correction where a small bias destroys the argmax.
    Pruning uses the same proxy for a COARSE, DISCRETE keep/drop decision, which is far
    more robust to a biased estimate. And oPCA's gamma already uses stored foreign rays for
    past classes and is the one thing that worked, so the proxy is proven in exactly this
    (selection) role.

RAW CONE ONLY. No fusion, no calibration -- both measured null or harmful.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp42_old_class_prune.py
    DS=IMAGENETR T=10 SEED=0 MODES=np,pd0.8,pr0.8 python -u exp42_old_class_prune.py
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
MODES = os.environ.get("MODES", "np,pd0.9,pr0.9,pd0.8,pr0.8").split(",")
METHOD = os.environ.get("METHOD", "opca")
GAMMA = float(os.environ.get("GAMMA", 0.5))
KDIV = float(os.environ.get("KDIV", 5))        # R_c = n_c / KDIV -- exp41's best (k5)
RMIN = int(os.environ.get("RMIN", 8))
RMAX = int(os.environ.get("RMAX", 128))
RMIN_P = int(os.environ.get("RMIN_P", 4))      # floor a pruned class may never go below
GP = float(os.environ.get("GP", 1.0))          # weight on z(leak) vs z(cluster size)
F_MAX = int(os.environ.get("F_MAX", 2000))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
OUT = os.path.join(REPO, f"exp42_old_class_prune_{TAG}.json")

un = X.un


def parse(mode):
    if mode == "np":
        return "np", 1.0
    m = re.match(r"(pd|pr)([\d.]+)$", mode)
    assert m, f"bad mode {mode!r}; want np | pd<rho> | pr<rho>"
    return m.group(1), float(m.group(2))


def zc(v):
    v = np.asarray(v, np.float64)
    return (v - v.mean()) / (v.std() + 1e-9)


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
    RAY, WGT, BORN = {}, {}, {}          # shared across modes: the birth state is identical
    LEAK = {}                            # per class, accumulated overlap with LATER classes
    res = {a: [] for a in ["ranpac"] + MODES}
    nray = {m: [] for m in MODES}
    rng_g = np.random.default_rng(777)

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

        # ---- birth: build rays exactly as exp41's k5 arm does
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = un(Ztr[r] @ Wh)
            Rc = int(np.clip(len(r) / KDIV, RMIN, RMAX))
            Fw = np.zeros((0, d), np.float32)
            if METHOD in X.DISCRIM and GAMMA > 0:
                oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                past = [RAY[o] for o in RAY if o not in tasks[t]]
                Fr = np.concatenate([Ztr[oth]] + past, 0)
                if len(Fr) > F_MAX:
                    Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                Fw = un(Fr @ Wh)
            Aw = X.BUILD[METHOD](Xw, Fw, Rc, int(c), GAMMA)
            # cluster weight per ray = share of this class's rows nearest to it. Cheap,
            # stored once, and it is what protects the dominant modes from being pruned.
            asg = (Xw @ Aw.T).argmax(1)
            WGT[c] = np.bincount(asg, minlength=len(Aw)).astype(np.float64) + 1e-9
            RAY[c] = Aw @ Wh_inv
            BORN[c] = t
            LEAK[c] = np.zeros(len(Aw))

        # ---- old classes see the new arrivals and accumulate leak
        new = np.concatenate([un(RAY[c] @ Wh) for c in tasks[t] if c in RAY]) \
            if any(c in RAY for c in tasks[t]) else None
        if new is not None and len(new):
            for c in RAY:
                if BORN[c] == t:
                    continue
                LEAK[c] += np.maximum(un(RAY[c] @ Wh) @ new.T, 0).mean(1)

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
        for mode in MODES:
            kind, rho = parse(mode)
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            tot = 0
            for c in seen:
                if c not in RAY:
                    continue
                Rc = len(RAY[c])
                K = max(RMIN_P, int(round(Rc * rho ** max(0, t - BORN[c]))))
                if kind == "np" or K >= Rc:
                    keep = np.arange(Rc)
                elif kind == "pd":
                    keep = np.argsort(-(zc(WGT[c]) - GP * zc(LEAK[c])))[:K]
                else:                                   # pr: identical K, chosen at random
                    keep = rng_g.choice(Rc, K, replace=False)
                Ac = un(RAY[c][keep] @ Wh)
                St[:, c] = X.cone_score(Ac, Qw)
                tot += len(Ac)
            res[mode].append(acc(St, yt))
            nray[mode].append(tot / max(len(seen), 1))
        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}" + "".join(
            f" | {m} {res[m][-1]*100:.2f} (r{nray[m][-1]:.1f})" for m in MODES))

    del G, C, P, eye
    torch.cuda.empty_cache()
    out = {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for a, v in res.items()}
    for m in MODES:
        out[m]["mean_rays"] = float(np.mean(nray[m]))
        out[m]["final_rays"] = float(nray[m][-1])
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            key = (f"{ds}|{T}|{seed}|{METHOD}g{GAMMA:g}k{KDIV:g}|{'+'.join(MODES)}"
                   f"|p{RMIN_P}_g{GP:g}_r{RMIN}-{RMAX}|m{M_RP}_s{SHRINK:g}|v1")
            if key in allres:
                log(f"skip {key}"); continue
            log(f"=== {key}")
            allres[key] = run_cell(ds, T, seed)
            json.dump(allres, open(OUT, "w"), indent=2)

W = 88
print("\n" + "=" * W)
print("EXP42 — do old classes benefit from adapting (pruning) to later arrivals?")
print("=" * W)
for key, r in sorted(allres.items()):
    print(f"\n--- {key}")
    print(f"  {'mode':<10}{'A-Last':>9}{'A-Avg':>9}{'mean r':>9}{'final r':>9}{'vs np':>8}")
    base = r["np"]["A_last"] * 100 if "np" in r else None
    for m in MODES:
        v = r[m]
        dl = f"{v['A_last']*100-base:>+8.2f}" if base is not None else f"{'--':>8}"
        print(f"  {m:<10}{v['A_last']*100:>9.2f}{v['A_avg']*100:>9.2f}"
              f"{v['mean_rays']:>9.1f}{v['final_rays']:>9.1f}{dl}")
    print(f"  {'[ranpac]':<10}{r['ranpac']['A_last']*100:>9.2f}"
          f"{r['ranpac']['A_avg']*100:>9.2f}")
    g = {m: r[m]["A_last"] * 100 for m in MODES if m in r}
    for rho in sorted({parse(m)[1] for m in MODES if parse(m)[0] == "pd"}):
        a, b = f"pd{rho:g}", f"pr{rho:g}"
        if a in g and b in g:
            print(f"\n  DISCRIMINATIVE PRUNING   {a} - {b} = {g[a]-g[b]:+.2f}"
                  f"   (identical ray budget)")
            print(f"  SIZE EFFECT ALONE        {b} - np = {g[b]-g.get('np',0):+.2f}")
print("\n" + "-" * W)
print("ranpac must be the exp16 bar (IN-R T=10 s0: 80.28) and `np` must be ~79.92, exp41's")
print("   k5 value. Both are recomputations of known cells; check them first.")
print("pd - pr IS THE RESULT. exp41 showed fewer rays is already better on its own")
print("   (k12, 9.7 rays: 79.70 vs fixed R=32's 79.50), so `pd - np` conflates the")
print("   discriminative criterion with the plain size effect. `pr` holds the ray count")
print("   identical class-by-class and stage-by-stage, so only pd - pr isolates the rule.")
print("Pruning is the ONLY legal post-birth edit: rotation inside span(A_c) is a null for a")
print("   subspace score (and cone ~ sub was measured at -0.10), and a cone is exactly")
print("   scale-invariant per ray. So a negative result here closes old-class adaptation.")
print("=" * W)
print(f"wrote {OUT}")
