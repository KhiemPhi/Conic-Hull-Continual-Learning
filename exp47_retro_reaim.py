#!/usr/bin/env python3
"""exp47_retro_reaim.py — GATE 1. Let OLD cones re-aim against classes born after them.

THE DEFECT
    exp39's discriminative construction is ASYMMETRIC and says so in its own docstring:
    "later classes avoid earlier ones, earlier ones never adapt." Class c's generators
    maximise ||Pi_C x||^2 - gamma ||Pi_C a||^2 over the foreign material available AT
    BIRTH. At stage 0 that is 19 classes; class 0's cone never learns to avoid the 180
    classes that arrive later. Since oPCA is worth +1.58 and is the largest read-out lever
    in the project, running it on 10% of the negatives for the earliest classes is a
    straightforward loss with no mechanism attached.

    This is a READ-OUT-ONLY fix: no backbone change, no feature training, no new storage.
    It is the cheapest item on the plan and it gates the expensive ones, because if
    re-aiming is worth nothing then the asymmetry was never costing anything and exp48's
    L_out term (which attacks the same defect from the feature side) loses its motivation.

WHAT IS LEGAL, AND THE HARD LIMIT IT IMPLIES
    At stage t we hold, for a past class o: its R_o stored rays, and nothing else. No
    images, no scatter, no mean beyond what the rays encode. So the re-aim can only use
    the rank-R surrogate
        S_hat_o = V_o^T diag(pi) V_o
    with pi the stored ray masses (uniform unless RECORD_MASS). Re-solving oPCA with
    S_hat_o in place of S_o produces new generators INSIDE span(V_o):

        THE RE-AIM CAN ROTATE AND REWEIGHT WITHIN THE STORED SPAN. IT CANNOT RECOVER A
        DIRECTION THAT WAS NEVER STORED.

    That is a real ceiling and it is why `oracle` below is mandatory. Note this is exactly
    the operation exp46 predicts is cheap for a cone (movement inside its own span), so the
    two files are testing complementary halves of the same invariance: exp46 asks whether
    in-span movement is HARMLESS, this asks whether in-span movement can be USEFUL.

ARMS
    base      no re-aim. Must reproduce exp41's k5m24 (80.07 A-Last, 26.5 rays). If it does
              not, the replay is broken and nothing else on the page means anything.
    reaim     at every stage, for every PAST class, re-solve top-R eigvecs of
              S_hat_o - gamma * S_F(current) and re-project the stored rays into it.
    reaim_l   the same but only for classes whose birth task is >= LAG stages old, so a
              class is not re-aimed against negatives it already saw. Isolates "genuinely
              new negatives" from "re-running the same solve with more samples".
    oracle    ILLEGAL. Refits past classes from their RAW FIT ROWS against the current
              foreign set. Uses past images and is not a method -- it is the ceiling on
              what any re-aiming scheme can deliver. If oracle - base is small the whole
              idea is dead regardless of how clever the legal version is. READ IT FIRST.

WHAT WOULD MAKE THIS A FALSE POSITIVE
    Re-solving an eigenproblem and re-running k-means perturbs the rays even at gamma=0,
    and exp41 measured the read-out is nearly flat in ray count over a 3.5x range, so a
    small change could be reshuffling noise. `reaim_g0` (gamma=0, same code path, no
    foreign term) is the control that catches it: it does everything reaim does EXCEPT
    look at other classes. reaim - reaim_g0 is the honest effect size.

RAW CONE ONLY -- no fusion, no calibration.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp47_retro_reaim.py
    DS=IMAGENETR T=10 SEED=0 ARMS=base,reaim,oracle GAMMAS=0.5,1 python -u exp47_retro_reaim.py
"""
import json
import os
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
ARMS = os.environ.get("ARMS", "base,reaim_g0,reaim,reaim_l,oracle").split(",")
GAMMA = float(os.environ.get("GAMMA", 0.5))
RGAMMA = float(os.environ.get("RGAMMA", GAMMA))   # gamma used by the re-aim solve
KVAL = float(os.environ.get("KVAL", 5))
RMIN = int(os.environ.get("RMIN", 24))
RMAX = int(os.environ.get("RMAX", 128))
LAG = int(os.environ.get("LAG", 2))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
OUT = os.path.join(REPO, f"exp47_retro_reaim_{TAG}.json")
ALL = ("base", "reaim_g0", "reaim", "reaim_l", "oracle")
assert all(a in ALL for a in ARMS), f"unknown arm; pick from {ALL}"

un = X.un


def rays_for(n):
    return int(np.clip(n / KVAL, RMIN, RMAX))


def reaim(Vw, Fw, R_, seed, g):
    """Re-solve oPCA for a past class using ONLY its own stored rays as the class evidence.

    Vw : (R, d) the class's stored rays, mapped into the CURRENT whitened space
    Fw : (F, d) current foreign material
    The class second moment is the rank-R surrogate Vw^T Vw / R. Everything else is exactly
    b_opca, so `reaim` at g=0 differs from `base` only by the surrogate, which is what
    reaim_g0 measures."""
    d_ = Vw.shape[1]
    M = (Vw.T.astype(np.float64) @ Vw) / len(Vw)
    if g > 0 and len(Fw):
        M = M - g * (Fw.T.astype(np.float64) @ Fw) / len(Fw)
    k = int(min(R_, len(Vw), d_))
    U = np.linalg.eigh(M)[1][:, ::-1][:, :k].astype(np.float32)
    # k-means over the rays themselves inside the new subspace: the rays ARE the only
    # samples we have. With k == len(Vw) this is an identity up to the projection, which is
    # the correct degenerate behaviour -- re-aiming a cone that has no spare rays to merge
    # can only rotate it.
    return un(X.b_kmeans(Vw @ U, None, R_, seed, 0) @ U.T)


def run_cell(ds, T, seed):
    E.T, E.SEED = T, seed
    assert (E.T, E.SEED) == (T, seed)
    F_ = E.adapted_features(ds)
    assert F_ is not None, f"no exp16 cache for {ds} T={T} s={seed}"
    Ztr, Zte = F_
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    task_of = {int(c): t for t in range(T) for c in tasks[t]}
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
    A = {a: {} for a in ARMS}                 # arm -> class -> rays (ORIGINAL space)
    nfit = {}
    res = {a: [] for a in ARMS + ["ranpac"]}
    nray = {a: [] for a in ARMS}

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

        # ---- birth: identical for every arm, so arms differ ONLY in what happens later
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = un(Ztr[r] @ Wh)
            oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
            base_arm = ARMS[0]
            past = [A[base_arm][o] for o in A[base_arm] if o not in tasks[t]]
            Fr = np.concatenate([Ztr[oth]] + past, 0)
            if len(Fr) > F_MAX:
                Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
            V = X.b_opca(Xw, un(Fr @ Wh), rays_for(len(r)), int(c), GAMMA) @ Wh_inv
            for a in ARMS:
                A[a][c] = V.copy()
            nfit[int(c)] = len(r)

        seen = np.concatenate(tasks[:t + 1])

        # ---- re-aim: every PAST class, against the foreign material available NOW
        if t > 0:
            for a in ARMS:
                if a == "base":
                    continue
                g = 0.0 if a == "reaim_g0" else RGAMMA
                for o in [int(c) for c in np.concatenate(tasks[:t])]:
                    if o not in A[a]:
                        continue
                    if a == "reaim_l" and (t - task_of[o]) < LAG:
                        continue
                    # foreign = every OTHER seen class's current material for this arm.
                    # Rays for past classes, fit rows for the current task -- same mixture
                    # rule as birth, so the only difference from birth is WHICH classes
                    # exist, which is the variable under test.
                    cur = FIT[t][~np.isin(ytr[FIT[t]], [o])]
                    pastr = [A[a][p] for p in A[a] if p != o and task_of[p] < t]
                    Fr = np.concatenate([Ztr[cur]] + pastr, 0) if g > 0 else np.zeros((0, d), np.float32)
                    if len(Fr) > F_MAX:
                        Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                    Fw = un(Fr @ Wh) if len(Fr) else np.zeros((0, d), np.float32)
                    if a == "oracle":
                        # ILLEGAL: re-reads the class's raw fit rows.
                        rr = FIT[task_of[o]][ytr[FIT[task_of[o]]] == o]
                        A[a][o] = X.b_opca(un(Ztr[rr] @ Wh), Fw, rays_for(len(rr)),
                                           o, g) @ Wh_inv
                    else:
                        Vw = un(A[a][o] @ Wh)
                        A[a][o] = reaim(Vw, Fw, len(Vw), o, g) @ Wh_inv

        # ---- RanPAC bar
        for i, h in _H(un(Ztr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        nval = sum(len(v) for v in VAL[:t + 1])
        yv = ytr[VAL_ALL[:nval]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        Qt = Zte[tei]

        def acc(Z, y):
            return float((np.asarray(seen)[Z[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            aa = acc(project(un(Ztr[VAL_ALL[:nval]]), Wm), yv)
            if aa > best:
                best, bw = aa, Wm
        res["ranpac"].append(acc(project(un(Qt), bw), yt))

        Qw = un(Qt @ Wh)
        for a in ARMS:
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            tot = 0
            for c in seen:
                if c not in A[a]:
                    continue
                Ac = un(A[a][c] @ Wh)
                St[:, c] = X.cone_score(Ac, Qw)
                tot += len(Ac)
            res[a].append(acc(St, yt))
            nray[a].append(tot / max(len(seen), 1))

        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}"
            + "".join(f"  {a} {res[a][-1]*100:.2f}" for a in ARMS))

    del G, C, P, eye
    torch.cuda.empty_cache()
    out = {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for a, v in res.items()}
    for a in ARMS:
        out[a]["mean_rays"] = float(np.mean(nray[a]))
    # per-birth-task final accuracy: the re-aim should help the EARLIEST classes most,
    # which is a sharper prediction than the aggregate and costs nothing to check.
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(ARMS)}|g{GAMMA:g}_rg{RGAMMA:g}"
                       f"|k{KVAL:g}m{RMIN}_L{LAG}|f{F_MAX}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed)
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 84
    print("\n" + "=" * W)
    print("EXP47 — retroactive re-aim: can an old cone learn to avoid a new class?")
    print("=" * W)
    for key, r in sorted(allres.items()):
        print(f"\n--- {key}")
        print(f"  {'arm':<12}{'A-Last':>9}{'A-Avg':>9}{'rays':>8}{'vs base':>10}")
        b = r.get("base", {}).get("A_last")
        for a in [x for x in ALL if x in r]:
            v = r[a]
            dl = f"{(v['A_last']-b)*100:>+10.2f}" if b is not None else f"{'--':>10}"
            print(f"  {a:<12}{v['A_last']*100:>9.2f}{v['A_avg']*100:>9.2f}"
                  f"{v.get('mean_rays', 0):>8.1f}{dl}")
        if "ranpac" in r:
            print(f"  {'[ranpac]':<12}{r['ranpac']['A_last']*100:>9.2f}"
                  f"{r['ranpac']['A_avg']*100:>9.2f}")
        g = {a: r[a]["A_last"] * 100 for a in r if a != "ranpac"}
        if "oracle" in g and "base" in g:
            print(f"\n  CEILING      oracle - base   = {g['oracle']-g['base']:+.2f}")
        if "reaim" in g and "reaim_g0" in g:
            print(f"  TRUE EFFECT  reaim - reaim_g0 = {g['reaim']-g['reaim_g0']:+.2f}")
        if "reaim" in g and "base" in g:
            print(f"  RAW          reaim - base     = {g['reaim']-g['base']:+.2f}")
    print("\n" + "-" * W)
    print("READ `oracle - base` FIRST. It uses past images and is not a method; it is the")
    print("   ceiling. The legal arms are restricted to span(V_o) and cannot exceed it.")
    print("`reaim - reaim_g0` is the only honest effect size. reaim_g0 re-solves and")
    print("   re-clusters WITHOUT looking at other classes, so it absorbs the reshuffling")
    print("   noise that re-aiming would otherwise get credit for.")
    print("base MUST reproduce exp41's k5m24 (IMAGENETR T=10 s0: 80.07, 26.5 rays).")
    print("=" * W)
    print(f"wrote {OUT}")
