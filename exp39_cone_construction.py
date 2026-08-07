#!/usr/bin/env python3
"""exp39_cone_construction.py — sweep how the generating set is CHOSEN.

WHY SELECTION AND NOT ANYTHING ELSE
    Ray selection is the largest measured lever in this line of work: at R=2 on
    ImageNet-R, SPA generators score 11.60 and k-means centroids 73.43. Nothing else --
    metric, rule, atom count, lift -- has ever moved a number that far. Yet both
    incumbents optimise something unrelated to the cone:
        SPA      solves NMF separability -> simplex VERTICES, deliberately the most
                 atypical samples, i.e. the worst possible summary of where mass lives
        k-means  solves quantisation, min sum_k min ||x - m_k||^2, in Euclidean space
    and, decisively, BOTH ARE BLIND TO OTHER CLASSES. A per-class generative fit cannot
    make its score class-selective, which is exactly the deficit measured elsewhere:
    cone_km falls from 79.50 (R=4) to 77.92 (R=32) and below a plain class mean (78.57),
    because adding rays raises s_c for foreign queries as fast as for own-class ones.

    This file is the first thing in the project to put foreign information into the
    construction itself.

THE OBJECTIVE the discriminative methods approximate
        max_V  sum_{x in X_c} ||Pi_C(V) x||^2  -  gamma * sum_{a in F} ||Pi_C(V) a||^2
    Cover your own class; do not cover anyone else's. F is the foreign material legally
    available when class c is born: FIT ROWS of the other classes in the current task
    (they arrive together) plus the STORED RAYS of every past class. No past images.

METHODS
    mean     one ray at the class mean. Degenerate floor; the cone becomes NCM.
    random   R random fit rows. The TRUE NULL -- if a method cannot beat this, selection
             does not matter at this R and the rest of the table is noise.
    spa      successive projection; the historical baseline, expected to lose badly.
    kmeans   the incumbent.
    opca     oriented PCA: top-R eigenvectors of S_c - gamma*S_F, k-means inside that
             subspace, lift back. Closed form. Strictly sharper than a global LDA metric:
             LDA finds directions separating all classes ON AVERAGE, this finds directions
             where class c specifically differs from everyone else.
    dkm      discriminative k-means: ordinary k-means run in the metric (I + gamma*S_F)^-1/2,
             which shrinks directions foreign material occupies. Rays are stored in the
             ORIGINAL space and scored in the shared whitener, so selection is
             discriminative while scoring stays comparable across classes.
    mp0      conic matching pursuit, gamma=0.
    mp       discriminative conic matching pursuit.

MATCHING PURSUIT, and why it is not an eigenproblem
    Adding a unit ray v to a cone with residual r changes the score by exactly the
    column-generation reduced cost, delta ||Pi||^2 = (v^T r)_+^2. So the greedy step
    maximises  sum_x (v^T r_x)_+^2  -  gamma * sum_a (v^T r_a)_+^2.
    Dropping the ()_+ turns this into the top eigenvector of a difference of second
    moments -- which is WRONG for a cone twice over: eigenvectors are sign-ambiguous, and
    they are two-sided, so half the data can have negative projection on the chosen ray.
    Instead this iterates the one-sided fixed point
        S  = {x : v.r_x > 0},   S' = {a : v.r_a > 0}
        v <- un( sum_{S} r_x  -  gamma * sum_{S'} r_a )
    whose solution is a non-negative combination of ACTIVE residuals, so rays stay inside
    the conic span of the data. That is precisely why k-means beats SPA, kept by design.

HONEST PRIOR
    mp0 is close to PCA in disguise: with V empty the residual IS the data, so the first
    direction is ~the class mean and later ones approximate principal directions. Those
    are already measured neutral for the cone (eigen-augmentation: +0.30 at R=4, -0.10 at
    R=64, +0.00 fused). Expect mp0 ~ kmeans. GAMMA IS THE LOAD-BEARING PART; mp0 and the
    gamma=0 columns exist to prove that, not to win.
    The SPA-vs-kmeans swing overstates the headroom -- that was pathological vs reasonable,
    and this is reasonable vs optimal. Expect +0.3 to +0.8, not +60.

SYNTHETIC SANITY CHECK (7 isotropic Gaussian clusters, d=96, R=8; NOT a result -- the
metric is the own-minus-foreign score gap, not accuracy, and real features are not
isotropic). Reported because it fixes the expected shape of each dial:
        opca g=2   gap .2875      mp   g=0.5  gap .2303
        dkm  g=2   gap .2742      mp0         gap .1888
        kmeans     gap .2649      spa         gap .1259
                                  random      gap .1095
    - kmeans >> random confirms selection matters at all; spa ~ random reproduces the
      known SPA pathology.
    - opca and dkm both beat kmeans, and both are MONOTONE in gamma up to 2.
    - mp is inverted-U with a peak near gamma 0.25-0.5 and does not reach kmeans. mp0 has
      the HIGHEST own-class coverage (.4469) and the worst gap of any sensible method --
      greedy coverage maximisation produces broad directions that cover everything, which
      is exactly the "PCA in disguise" failure predicted above.
    So sweep gamma in {0.25,0.5} for mp and {1,2,4} for opca/dkm; a single gamma=1
    underserves both. The closed-form methods look better than the iterative one, and
    they are also the cheap ones.

NO FUSION ANYWHERE. Raw cone accuracy only. beta selection was measured to swing the
    headline by 0.77 on a 0.09 feature perturbation, larger than any effect here.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp39_cone_construction.py
    DS=IMAGENETR T=10 SEED=0 METHODS=kmeans,dkm,mp GAMMAS=0.5,1,2 python -u exp39...
    DS=IMAGENETR T=10 SEED=0 METHODS=kmeans,random R=8 python -u exp39...
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
METHODS = os.environ.get(
    "METHODS", "mean,random,spa,kmeans,opca,dkm,mp0,mp").split(",")
GAMMAS = [float(x) for x in os.environ.get("GAMMAS", "1.0").split(",")]

R = int(os.environ.get("R", 32))
F_MAX = int(os.environ.get("F_MAX", 2000))         # cap on foreign material per class
MP_IRLS = int(os.environ.get("MP_IRLS", 8))        # inner one-sided fixed-point steps
MP_ITERS = int(os.environ.get("MP_ITERS", 100))    # NNLS iters during construction only
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
ITERS = int(os.environ.get("ITERS", 500))          # NNLS iters at SCORING time
OUT = os.path.join(REPO, f"exp39_cone_construction_{TAG}.json")

DISCRIM = {"opca", "dkm", "mp"}                    # gamma actually does something
ALLM = ("mean", "random", "spa", "kmeans", "opca", "dkm", "mp0", "mp")
assert all(m in ALLM for m in METHODS), f"unknown method; pick from {ALLM}"


def un(A):
    A = np.atleast_2d(np.asarray(A, np.float32))
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def cone_score(A, Q, iters=None):
    h = ConicHull(n_rays=len(A), nnls_iters=iters or ITERS)
    h.extreme_rays_ = un(A)
    return h.score(Q)


def cone_resid(A, Q, iters):
    """Q - Pi_C(A) Q. The NNLS residual, i.e. the KKT certificate: A r <= 0, so r is
    exactly the direction the current cone cannot reach."""
    An = un(A)
    h = ConicHull(n_rays=len(An), nnls_iters=iters)
    h.extreme_rays_ = An
    return np.asarray(Q, np.float32) - h.reconstruct(Q).astype(np.float32) @ An


# ---------------------------------------------------------------- constructions
# every builder takes  (Xw, Fw, R, seed, gamma)  with Xw/Fw unit rows in the CURRENT
# whitened space, and returns <= R unit rays in that same space.

def b_mean(Xw, Fw, R_, seed, g):
    return un(Xw.mean(0, keepdims=True))


def b_random(Xw, Fw, R_, seed, g):
    k = min(R_, len(Xw))
    return un(Xw[np.random.default_rng(seed).choice(len(Xw), k, replace=False)])


def b_kmeans(Xw, Fw, R_, seed, g):
    k = min(R_, len(Xw))
    return un(Xw.mean(0, keepdims=True) if k <= 1 else
              KMeans(k, n_init=4, random_state=seed).fit(Xw).cluster_centers_)


def b_spa(Xw, Fw, R_, seed, g):
    """Successive projection: repeatedly take the row of largest residual norm and
    project it out. Returns simplex vertices -- the most atypical samples."""
    Xr = np.array(Xw, np.float64, copy=True)
    idx = []
    for _ in range(min(R_, len(Xw))):
        n2 = (Xr ** 2).sum(1)
        j = int(n2.argmax())
        if n2[j] <= 1e-12:
            break
        idx.append(j)
        u = Xr[j] / np.linalg.norm(Xr[j])
        Xr -= np.outer(Xr @ u, u)
    return un(Xw[idx]) if idx else un(Xw.mean(0, keepdims=True))


def _sf(Fw, d):
    return (Fw.T.astype(np.float64) @ Fw) / max(len(Fw), 1) if len(Fw) else np.zeros((d, d))


def b_dkm(Xw, Fw, R_, seed, g):
    """k-means in the metric (I + g*S_F)^-1/2, which shrinks directions foreign material
    occupies. Rays come back to the shared space, so scoring stays comparable."""
    d = Xw.shape[1]
    S = _sf(Fw, d)
    S = S / max(np.trace(S) / d, 1e-12) if np.trace(S) > 0 else S
    Wd = np.linalg.cholesky(np.linalg.inv(np.eye(d) + g * S)).astype(np.float32)
    cent = b_kmeans(un(Xw @ Wd), None, R_, seed, 0)
    return un(cent @ np.linalg.inv(Wd).astype(np.float32))


def b_opca(Xw, Fw, R_, seed, g):
    """Oriented PCA: top-R eigenvectors of S_c - g*S_F, k-means inside, lift back."""
    d = Xw.shape[1]
    M = (Xw.T.astype(np.float64) @ Xw) / len(Xw) - g * _sf(Fw, d)
    k = int(min(R_, d))
    V = np.linalg.eigh(M)[1][:, ::-1][:, :k].astype(np.float32)
    return un(b_kmeans(Xw @ V, None, R_, seed, 0) @ V.T)


def b_mp(Xw, Fw, R_, seed, g):
    """Discriminative conic matching pursuit -- see the module docstring."""
    rx = np.array(Xw, np.float32, copy=True)
    ra = np.array(Fw, np.float32, copy=True) if (g > 0 and len(Fw)) else None
    rays = []
    for _ in range(min(R_, len(Xw))):
        v = rx.sum(0)
        if np.linalg.norm(v) < 1e-8:
            v = rx[int((rx ** 2).sum(1).argmax())]
        v = v / (np.linalg.norm(v) + 1e-12)
        for _ in range(MP_IRLS):                     # one-sided fixed point
            # MEANS, not sums. The two active sets have wildly different cardinality --
            # ~100 own rows against up to F_MAX=2000 foreign items -- so summing makes
            # gamma=1 weight foreign ~20x and pushes the ray off the data entirely
            # (measured: gap +0.19 at g=0 collapsing to +0.08 at g=4, i.e. backwards).
            # With means, gamma=1 means "one foreign sample counts as much as one own
            # sample", which is interpretable and stable across stages and datasets.
            ax = rx[(rx @ v) > 0]
            num = ax.mean(0) if len(ax) else np.zeros_like(v)
            if ra is not None:
                aa = ra[(ra @ v) > 0]
                if len(aa):
                    num = num - g * aa.mean(0)
            n = np.linalg.norm(num)
            if n < 1e-8:
                break
            v = num / n
        rays.append(v.astype(np.float32))
        A = un(np.stack(rays))
        rx = cone_resid(A, Xw, MP_ITERS)             # recompute against the FULL cone
        if ra is not None:
            ra = cone_resid(A, Fw, MP_ITERS)
        if np.linalg.norm(rx) < 1e-6:
            break
    return un(np.stack(rays))


BUILD = {"mean": b_mean, "random": b_random, "spa": b_spa, "kmeans": b_kmeans,
         "opca": b_opca, "dkm": b_dkm, "mp0": lambda *a: b_mp(*a[:4], 0.0), "mp": b_mp}


def arm_name(m, g):
    return f"{m}_g{g:g}" if m in DISCRIM else m


# ---------------------------------------------------------------- replay
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
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.as_tensor(X[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def project(X, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(X)]).cpu().numpy()

    ARMS = [arm_name(m, g) for m in METHODS for g in (GAMMAS if m in DISCRIM else [0.0])]
    ARMS = list(dict.fromkeys(ARMS))
    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A = {a: {} for a in ARMS}                       # arm -> class -> rays (ORIGINAL space)
    MU = {}                                         # class -> TRUE mean (ORIGINAL space)
    res = {a: [] for a in ARMS + ["ranpac", "ncm"]}
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
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = un(Ztr[r] @ Wh)
            MU[c] = un(Xw.mean(0, keepdims=True)) @ Wh_inv
            for a in ARMS:
                m = a.split("_g")[0]
                g = float(a.split("_g")[1]) if "_g" in a else 0.0
                # FOREIGN MATERIAL, legal at this instant: fit rows of the OTHER classes
                # in this task (they arrive together, so no ordering asymmetry within a
                # task) plus the stored rays of every past class of THIS arm. Never a
                # past image. Capped at F_MAX by random subsample -- uncapped it is
                # ~24k rows at the last task and dominates construction cost.
                Fw = np.zeros((0, d), np.float32)
                if m in DISCRIM and g > 0:
                    oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                    # PAST tasks only for the ray part. A[a] already holds the earlier
                    # classes of THIS task, and their fit rows are in `oth` too, so
                    # including them would count them twice and make the negative set
                    # depend on position within the task.
                    past = [A[a][o] for o in A[a] if o not in tasks[t]]
                    Fr = np.concatenate([Ztr[oth]] + past, 0)
                    if len(Fr) > F_MAX:
                        Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                    Fw = un(Fr @ Wh) if len(Fr) else Fw
                A[a][c] = BUILD[m](Xw, Fw, R, int(c), g) @ Wh_inv

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
        NC = np.full((len(tei), n_cls), -np.inf, np.float32)
        for a in ARMS:
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            tot = 0
            for c in seen:
                if c not in A[a]:
                    continue
                Ac = un(A[a][c] @ Wh)
                St[:, c] = cone_score(Ac, Qw)
                tot += len(Ac)
            res[a].append(acc(St, yt))
            nray[a].append(tot / max(len(seen), 1))
        for c in seen:                                  # TRUE class-mean reference. Must
            if c in MU:                                 # NOT be read off ARMS[0]'s rays --
                NC[:, c] = Qw @ un(MU[c] @ Wh)[0]       # that made `ncm` change with the
        res["ncm"].append(acc(NC, yt))                  # order of METHODS.

        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}  ncm {res['ncm'][-1]*100:.2f}"
            + "".join(f"  {a} {res[a][-1]*100:.2f}" for a in ARMS))

    del G, C, P, eye
    torch.cuda.empty_cache()
    out = {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for a, v in res.items()}
    for a in ARMS:
        out[a]["mean_rays"] = float(np.mean(nray[a]))
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(METHODS)}|g{'+'.join(f'{x:g}' for x in GAMMAS)}"
                       f"|R{R}_f{F_MAX}_i{MP_IRLS}"
                       f"|m{M_RP}_s{SHRINK:g}_i{ITERS}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed)
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 88
    print("\n" + "=" * W)
    print("EXP39 — how the generating set is CHOSEN (raw cone accuracy, no fusion)")
    print("=" * W)
    for key, r in sorted(allres.items()):
        print(f"\n--- {key}")
        print(f"  {'method':<16}{'A-Last':>9}{'A-Avg':>9}{'rays/cls':>10}{'vs kmeans':>11}")
        base = r.get("kmeans", {}).get("A_last")
        rows = [(a, v) for a, v in r.items() if a not in ("ranpac", "ncm")]
        for a, v in sorted(rows, key=lambda kv: -kv[1]["A_last"]):
            dl = f"{(v['A_last']-base)*100:>+11.2f}" if base is not None else f"{'--':>11}"
            print(f"  {a:<16}{v['A_last']*100:>9.2f}{v['A_avg']*100:>9.2f}"
                  f"{v.get('mean_rays', 0):>10.1f}{dl}")
        for a in ("ncm", "ranpac"):
            if a in r:
                print(f"  {'['+a+']':<16}{r[a]['A_last']*100:>9.2f}{r[a]['A_avg']*100:>9.2f}")
        g = {a: v["A_last"] * 100 for a, v in rows}
        if "kmeans" in g and "random" in g:
            print(f"\n  DOES SELECTION MATTER   kmeans - random = {g['kmeans']-g['random']:+.2f}")
        if "mp0" in g and "kmeans" in g:
            print(f"  PURSUIT vs QUANTISATION mp0 - kmeans    = {g['mp0']-g['kmeans']:+.2f}")
        mps = [a for a in g if a.startswith("mp_g")]
        if mps and "mp0" in g:
            bm = max(mps, key=lambda a: g[a])
            print(f"  DISCRIMINATION          {bm} - mp0 = {g[bm]-g['mp0']:+.2f}")
    print("\n" + "-" * W)
    print("ranpac must reproduce the exp16 bar (ImageNet-R T=10 s0: 80.28); if not, the")
    print("   replay is broken and no row means anything.")
    print("READ `random` FIRST. If kmeans - random is ~0, ray selection does not matter at")
    print("   this R and every other comparison on the page is noise. It is the true null.")
    print("mp0 - kmeans isolates PURSUIT vs QUANTISATION with no discrimination; mp_g* - mp0")
    print("   isolates DISCRIMINATION at a fixed selector. Conflating them is how eigen-")
    print("   augmentation looked like a win for two rounds.")
    print("EXPECT mp0 ~ kmeans: with V empty the residual is the data, so mp0's first ray is")
    print("   ~the class mean and later ones approximate principal directions, which are")
    print("   already measured neutral for the cone (+0.30 R=4, -0.10 R=64, +0.00 fused).")
    print("   gamma is the load-bearing part -- mp0 exists to prove that, not to win.")
    print("rays/cls flags clamping: km and friends cap at min(R, n_rows) and 24.5% of")
    print("   ImageNet-R classes have <64 fit rows, so at R=64 the arms are not budget-matched.")
    print("The discriminative arms are ASYMMETRIC: later classes avoid earlier ones, earlier")
    print("   ones never adapt. Within-task negatives make it work from task 0, but the gain")
    print("   should grow with task index -- check A-Last, not just A-Avg.")
    print("=" * W)
    print(f"wrote {OUT}")
