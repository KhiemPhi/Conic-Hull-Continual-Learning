#!/usr/bin/env python3
"""exp38_fair_cone.py — the fair-comparison suite for the conic rule.

WHY THIS FILE EXISTS
    Every cone-vs-RanPAC number so far compares a GENERATIVE per-class model against a
    DISCRIMINATIVE jointly-fitted one. That is naive-Bayes vs LDA; the cone was never in
    a fair fight. It is also not a capacity problem -- at R=32 the generators are
    32*768*200 = 4.9M parameters (20 MB) against RanPAC's 10000*200 = 2.0M (8 MB), so the
    cone has 2.5x MORE parameters and still loses by 0.78. The deficit is entirely in what
    those parameters are fitted to.

    Five suites, each removing one specific unfairness. Every suite compares RAW arms --
    no RanPAC fusion anywhere in this file -- because beta selection was measured to swing
    the headline by 0.77 on a 0.09 feature perturbation, which is larger than any effect
    here. Fusion belongs in exp35; attribution belongs here.

SUITE=signal      what is RanPAC's discriminative fit actually worth?
    Decomposes the head at a FIXED lifted feature map h = relu(W_r phi(x)):
        rp_ncm    cos(h, m_c)                 generative: class means only, no between-class
        rp_diag   C / (diag(G) + lam)         per-feature scaling, no cross-feature term
        rp_ridge  (G + lam I)^-1 C            the real RanPAC
    rp_ncm is the fair generative counterpart of the cone -- same information budget, same
    lift. cone - rp_ncm is the honest rule comparison; rp_ridge - rp_ncm is the price of
    being generative. Nothing else in this file is interpretable without those two numbers.

SUITE=constraint  is "conic" even the right region primitive, at matched storage?
    Identical atoms, identical metric, identical everything -- only the constraint on w:
        sub       unconstrained          -> linear subspace, TWO-sided (span)
        cone      w >= 0                 -> conic hull, ONE-sided
        simplex   w >= 0, sum w = 1      -> convex hull
        sparse    w >= 0 + L1            -> sparse conic hull
    `sub` is the comparator that matters: a rank-R subspace costs exactly the same R*d
    floats, so this isolates NON-NEGATIVITY and nothing else. exp3b's old "cone worst of 4
    primitives" was AUROC on an OOD task, never CIL accuracy at matched budget.

SUITE=metric      the other half of LDA
    We whiten by the tied within-class scatter S_w and never touch the between-class
    scatter S_b -- yet S_b = sum_c n_c (mu_c - mubar)(mu_c - mubar)^T is computable from
    the per-class means we already store. Zero images, recomputed every stage, additive.
    This is the ONLY way to inject between-class information into a per-class generative
    model without a class-count-dependent feature map (which CIL forbids: old queries were
    never scored against classes that did not exist yet).
    NOTE rank(S_b) <= n_seen - 1, so at stage 0 with 20 classes LDA offers at most 19
    dimensions. Early stages are structurally starved; that is a property of LDA in CIL,
    not a bug, and it is logged.

SUITE=calib       a bias RanPAC does not have
    ||Pi_C q|| grows with a cone's solid angle, so a class with more spread data scores
    higher on EVERY query. RanPAC's least-squares fit calibrates this away for free; the
    cone has no such mechanism and it has never been tested.
        cal_bg     s - mu_bg, mu from OTHER classes' stored rays. Drift-free (recomputed
                   each stage) and it targets the bias directly: a wide cone scores high on
                   foreign rays and is shifted down by exactly that amount.
        cal_bgz    additionally / sig_bg. Separate arm because sigma changes the relative
                   scaling as well as removing the bias -- here it is safe, being estimated
                   over BG_MAX foreign rays.
        cal_birth  s - mu, mu from the class's own held-out VAL rows at birth. MEAN ONLY:
                   VAL holds 4..12 rows per class, so a per-class sigma from it is noise
                   (dividing by it measured 90.9 -> 59.8). Frozen in the birth metric, so
                   it drifts as the whitener accumulates -- that drift is the thing this
                   arm measures against cal_bg.

SUITE=openset     evaluate each method on what it is designed for
    At stage t, queries from classes in stages t+1..T are true unknowns. RanPAC
    STRUCTURALLY cannot reject them -- its logits are relative, with no representation for
    "outside all classes". The cone's ||Pi_C q|| is absolute and Moreau gives outside-ness
    free. Reports AUROC of max_c score for seen vs unseen. Note exp35's zs() destroys
    exactly this signal by normalising per query across classes.

NROWS  (the regime axis, orthogonal to SUITE)
    Subsample every class to NROWS fit rows before anything else -- generators, whitener
    and RanPAC all get less data. The cone's non-negativity is a regulariser and should
    hold up where the ridge cannot estimate a 10000x10000 Gram. Sweep NROWS=5,10,20,50,0
    (0 = all). This is not academic: CUB200 has 27 rows/class and ImageNet-A has 18, so if
    the cone wins below ~20 it predicts the two hardest datasets in the table.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 SUITE=signal     python -u exp38_fair_cone.py
    DS=IMAGENETR T=10 SEED=0 SUITE=constraint python -u exp38_fair_cone.py
    DS=IMAGENETR T=10 SEED=0 SUITE=signal NROWS=10 python -u exp38_fair_cone.py
    DS=IMAGENETR T=10 SEED=0 SUITE=signal,constraint,calib,metric,openset python -u ...
"""
import json
import os
import time

import numpy as np
import torch
from scipy.stats import rankdata
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
SUITES = os.environ.get("SUITE", "signal").split(",")

R = int(os.environ.get("R", 32))
NROWS = int(os.environ.get("NROWS", 0))            # 0 = use all fit rows
LDA_DIM = int(os.environ.get("LDA_DIM", 192))      # clamped to min(d, n_seen-1)
L1 = float(os.environ.get("L1", 0.02))             # for the `sparse` arm
BG_MAX = int(os.environ.get("BG_MAX", 2000))       # foreign rays for cal_bg
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, f"exp38_fair_cone_{TAG}.json")

ALL = ("signal", "constraint", "metric", "calib", "openset")
assert all(s in ALL for s in SUITES), f"unknown suite; pick from {ALL}"


def un(A):
    return np.asarray(A, np.float32) / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def km(X, k, seed):
    k = int(min(k, len(X)))
    return un(X.mean(0, keepdims=True) if k <= 1 else
              KMeans(k, n_init=4, random_state=seed).fit(X).cluster_centers_)


def auroc(pos, neg):
    """P(score(pos) > score(neg)); tie-corrected via average ranks."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# ---------------------------------------------------------------- region primitives
def s_sub(A, Q):
    """Unconstrained least squares == projection onto span(A). The budget-matched
    comparator: a rank-R subspace costs the same R*d floats as R rays.

    SVD rather than QR for the basis. numpy's reduced QR returns a full set of (d, R)
    orthonormal columns even when A is rank-deficient, padding with arbitrary directions
    OUTSIDE span(A) -- which would hand this arm free dimensions the cone does not get and
    silently break the budget match. Columns below the numerical rank tolerance are cut.
    """
    U, s, _ = np.linalg.svd(un(A).T, full_matrices=False)
    B = U[:, s > max(s[0], 1e-12) * 1e-6] if len(s) else U
    return np.linalg.norm(np.asarray(Q, np.float32) @ B, axis=1)


def s_hull(A, Q, constraint="cone", l1=0.0):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS, constraint=constraint, nnls_l1=l1)
    h.extreme_rays_ = un(A)
    return h.score(Q)


PRIMITIVE = {
    "sub":     lambda A, Q: s_sub(A, Q),
    "cone":    lambda A, Q: s_hull(A, Q, "cone"),
    "simplex": lambda A, Q: s_hull(A, Q, "simplex"),
    "sparse":  lambda A, Q: s_hull(A, Q, "cone", L1),
}


# ---------------------------------------------------------------- driver
def run_cell(ds, T, seed, suite):
    E.T, E.SEED = T, seed
    assert (E.T, E.SEED) == (T, seed)
    F = E.adapted_features(ds)
    assert F is not None, f"no exp16 cache for {ds} T={T} s={seed}"
    Ztr, Zte = F
    ytr, yte, n_cls = E.get_labels(ds)
    d0 = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]])
        fit = ix[pm[nv:]]
        if NROWS > 0:                                   # regime axis: starve every class
            keep = []
            for c in tasks[t]:
                rc = fit[ytr[fit] == c]
                keep.append(rc[:NROWS])
            fit = np.concatenate(keep) if keep else fit
        FIT.append(fit)
    VAL_ALL = np.concatenate(VAL)
    if NROWS > 0:
        per = [int((ytr[FIT[t]] == c).sum()) for t in range(T) for c in tasks[t]]
        log(f"  NROWS={NROWS}: {sum(len(f) for f in FIT)} fit rows, "
            f"per-class min {min(per)} med {int(np.median(per))} max {max(per)}")

    P = torch.randn(d0, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.as_tensor(X[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    ncnt = torch.zeros(n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def project(X, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(X)]).cpu().numpy()

    scatter = np.zeros((d0, d0), np.float64); n_scat = 0
    Aorig, Morig, Ncls, CAL = {}, {}, {}, {}
    res, extra = {}, {}

    for t in range(T):
        # ---- tied within-class scatter (accumulated, protocol-legal)
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        S = scatter / max(n_scat, 1)
        S = S + SHRINK * np.trace(S) / d0 * np.eye(d0)
        Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)

        # ---- generators + class means, born in the current metric, stored in ORIGINAL space
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Aorig[c] = km(un(Ztr[r] @ Wh), R, c) @ Wh_inv
            Morig[c] = Ztr[r].mean(0)
            Ncls[c] = len(r)
            if "calib" in suite:                        # birth-metric self-statistics
                v = VAL[t][ytr[VAL[t]] == c]
                if len(v) >= 2:
                    # MEAN ONLY. VAL holds 4..12 rows per class on ImageNet-R, so a
                    # per-class sigma from it is noise, and dividing by it lets whichever
                    # class drew the smallest sigma win spuriously (measured: 90.9 -> 59.8).
                    # Cone-size bias lives in the MEAN level of s_c anyway, which is what
                    # this arm is meant to remove.
                    CAL[c] = float(s_hull(un(Aorig[c] @ Wh), un(Ztr[v] @ Wh)).mean())

        # ---- RanPAC sufficient statistics (exact, additive)
        for i, h in _H(un(Ztr[FIT[t]])):
            h = h.double()
            lab = torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]), lab] = 1.0
            G += h.T @ h; C += h.T @ Y
            ncnt += Y.sum(0)

        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        yv = ytr[VAL_ALL[:nval]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        uei = np.where(~np.isin(yte, seen))[0]          # true unknowns, for openset
        have = [c for c in seen if c in Aorig]

        def acc(Zt, y):
            return float((np.asarray(seen)[Zt[:, seen].argmax(1)] == y).mean())

        def put(name, v):
            res.setdefault(name, []).append(v)

        # ---- the metric actually used for the geometric arms
        if "metric" in suite and have:
            mw = np.stack([Morig[c] @ Wh for c in have])
            w_ = np.array([Ncls[c] for c in have], np.float64)[:, None]
            dev = mw - (w_ * mw).sum(0) / w_.sum()
            Sb = (dev * w_).T @ dev
            k = int(min(LDA_DIM, len(have) - 1, d0))
            if k >= 1:
                V = np.linalg.eigh(Sb)[1][:, ::-1][:, :k].astype(np.float32)
                Wlda = (Wh @ V).astype(np.float32)
            else:
                Wlda = Wh
            # rank(S_b) <= n_seen-1, so early stages are structurally starved. Worse, if
            # k < R the R rays span the WHOLE k-dim space and the cone degenerates to
            # score ~1 everywhere -- cone_lda is meaningless at those stages, not merely
            # weak. Read A-Last, and treat early A-Avg contributions with suspicion.
            if k < R:
                log(f"    s{t} LDA DEGENERATE: k={k} < R={R}; the cone spans the whole "
                    f"LDA subspace at this stage")
            elif t == 0:
                log(f"    LDA: rank(S_b) <= {len(have)-1}, using k={k} of {d0}")

        Qt_ = Zte[tei]
        Qu_ = Zte[uei]

        # ------------------------------------------------ SUITE: signal
        if "signal" in suite:
            best, bw = -1.0, None
            for lam in LAMBDAS:
                W_ = torch.linalg.solve(G + lam * eye, C)
                a = acc(project(un(Ztr[VAL_ALL[:nval]]), W_), yv)
                if a > best:
                    best, bw = a, W_
            put("rp_ridge", acc(project(un(Qt_), bw), yt))

            # Diagonal-Mahalanobis template matcher: per-feature variance scaling, no
            # cross-feature term. Sits exactly between rp_ncm (no scaling) and rp_ridge
            # (full decorrelation).
            # C MUST be divided by the class counts. ImageNet-R fit rows run 35..308 --
            # 8.8x imbalance -- so raw column sums weight frequent classes ~9x higher and
            # the arm collapses to ~20% (measured). rp_ridge is immune because (G+lam I)^-1
            # corrects for it; a diagonal approximation is not.
            Mraw = C / ncnt.clamp(min=1)[None, :]
            dgn = torch.diagonal(G) / max(float(ncnt.sum()), 1.0)   # per-feature 2nd moment
            dgn = dgn / dgn.mean().clamp(min=1e-12)                 # ~1 on average, so the
            bestd, bwd = -1.0, None                                 # lambda grid is O(1)
            for lam in (1e-2, 1e-1, 1.0, 1e1):
                Wd = Mraw / (dgn[:, None] + lam)
                a = acc(project(un(Ztr[VAL_ALL[:nval]]), Wd), yv)
                if a > bestd:
                    bestd, bwd = a, Wd
            put("rp_diag", acc(project(un(Qt_), bwd), yt))

            # lifted-space NCM: class means read straight off C, no G at all. Column
            # normalisation makes this cos(h, m_c) up to the per-row ||h||, which is
            # constant across classes and so cannot change the argmax.
            Mn = C / ncnt.clamp(min=1)[None, :]
            Mn = Mn / (Mn.norm(dim=0, keepdim=True) + 1e-12)
            put("rp_ncm", acc(project(un(Qt_), Mn), yt))

        # ------------------------------------------------ geometric arms
        # A dict keyed by ARM NAME, so repeated requests for "cone" across suites collapse
        # to one computation. (A list of (name, transform) pairs cannot be de-duplicated
        # with `in`: the transform is an ndarray and the comparison raises.)
        plan = {}
        if {"signal", "calib", "openset"} & set(suite):
            plan["cone"] = ("cone", Wh)
        if "constraint" in suite:
            for p in PRIMITIVE:
                plan[p] = (p, Wh)
        if "metric" in suite:
            plan["cone"] = ("cone", Wh)
            plan["cone_lda"] = ("cone", Wlda)

        for name, (base, Wx) in plan.items():
            Qw = un(Qt_ @ Wx)
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            for c in have:
                St[:, c] = PRIMITIVE[base](un(Aorig[c] @ Wx), Qw)
            put(name, acc(St, yt))

            if name == "cone" and "calib" in suite:
                # Background statistics from FOREIGN generators: a wide cone scores high
                # on other classes' rays and is penalised, which is exactly the cone-size
                # bias. Subsampled to BG_MAX rays -- at 200 classes the full foreign set is
                # 6368 rays per class and would double the cost of the whole run.
                pool = np.concatenate([Aorig[o] for o in have]) if have else np.zeros((0, d0))
                owner = (np.concatenate([np.full(len(Aorig[o]), o) for o in have])
                         if have else np.zeros(0, int))
                rng = np.random.default_rng(0)
                Sm_, Sz_ = St.copy(), St.copy()
                for c in have:
                    fo = np.where(owner != c)[0]
                    if len(fo) == 0:
                        continue
                    if len(fo) > BG_MAX:
                        fo = rng.choice(fo, BG_MAX, replace=False)
                    sb = PRIMITIVE["cone"](un(Aorig[c] @ Wx), un(pool[fo] @ Wx))
                    Sm_[:, c] = St[:, c] - sb.mean()                  # bias removal only
                    Sz_[:, c] = (St[:, c] - sb.mean()) / (sb.std() + 1e-6)
                put("cone_cal_bg", acc(Sm_, yt))    # mean-centred: targets the bias itself
                put("cone_cal_bgz", acc(Sz_, yt))   # full z; sigma over BG_MAX foreign rays
                Sc_ = St.copy()                     # is well estimated, unlike VAL's 4..12
                for c in have:
                    if c in CAL:
                        Sc_[:, c] = St[:, c] - CAL[c]
                put("cone_cal_birth", acc(Sc_, yt))

            if name == "cone" and "openset" in suite:
                # The final stage has no unseen classes; append nan rather than skipping,
                # so every extra[] list stays the same length as res[] and the per-stage
                # log cannot silently reprint the previous stage's value.
                if len(uei) == 0:
                    extra.setdefault("os_cone", []).append(float("nan"))
                    if "signal" in suite:
                        extra.setdefault("os_ranpac", []).append(float("nan"))
                else:
                    Su = np.full((len(uei), n_cls), -np.inf, np.float32)
                    Quw = un(Qu_ @ Wx)
                    for c in have:
                        Su[:, c] = PRIMITIVE["cone"](un(Aorig[c] @ Wx), Quw)
                    extra.setdefault("os_cone", []).append(
                        auroc(St[:, seen].max(1), Su[:, seen].max(1)))
                    if "signal" in suite:
                        Lt_ = project(un(Qt_), bw)
                        extra.setdefault("os_ranpac", []).append(
                            auroc(Lt_[:, seen].max(1), project(un(Qu_), bw)[:, seen].max(1)))

        log(f"    s{t}: " + "  ".join(f"{a} {res[a][-1]*100:.2f}" for a in sorted(res))
            + ("  |  " + "  ".join(f"{a} {extra[a][-1]:.3f}" for a in sorted(extra))
               if extra else ""))

    del G, C, P, eye
    torch.cuda.empty_cache()
    out = {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for a, v in res.items()}
    out["_auroc"] = {a: {"mean": float(np.nanmean(v)), "per_stage": v}
                     for a, v in extra.items()}
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            key = (f"{ds}|{T}|{seed}|{'+'.join(sorted(SUITES))}"
                   f"|R{R}_n{NROWS}_l{LDA_DIM}_L{L1:g}"
                   f"|m{M_RP}_s{SHRINK:g}_i{ITERS}|v1")
            if key in allres:
                log(f"skip {key}"); continue
            log(f"=== {key}")
            allres[key] = run_cell(ds, T, seed, SUITES)
            json.dump(allres, open(OUT, "w"), indent=2)

W = 92
print("\n" + "=" * W)
print("EXP38 — fair-comparison suite (RAW arms only; no fusion, so beta never enters)")
print("=" * W)
for key, r in sorted(allres.items()):
    print(f"\n--- {key}")
    arms = [(a, v) for a, v in r.items() if a != "_auroc"]
    for a, v in sorted(arms, key=lambda kv: -kv[1]["A_last"]):
        print(f"  {a:<16}{v['A_last']*100:>8.2f}{v['A_avg']*100:>8.2f}")
    if r.get("_auroc"):
        print("  open-set AUROC (seen vs true unknowns, mean over stages):")
        for a, v in sorted(r["_auroc"].items()):
            print(f"    {a:<14}{v['mean']:>8.3f}")
    g = {a: v["A_last"] * 100 for a, v in arms}
    if "rp_ncm" in g and "cone" in g:
        print(f"\n  FAIR RULE GAP   cone - rp_ncm  = {g['cone']-g['rp_ncm']:+.2f}"
              "   (same budget, same information)")
        print(f"  PRICE OF GENERATIVE  rp_ridge - rp_ncm = "
              f"{g.get('rp_ridge',0)-g['rp_ncm']:+.2f}")
    if "sub" in g and "cone" in g:
        print(f"  NON-NEGATIVITY  cone - sub    = {g['cone']-g['sub']:+.2f}"
              "   (identical R*d storage)")
    if "cone_cal_bg" in g:
        print(f"  CALIBRATION     bg {g['cone_cal_bg']-g['cone']:+.2f}"
              f"   bgz {g.get('cone_cal_bgz',0)-g['cone']:+.2f}"
              f"   birth {g.get('cone_cal_birth',0)-g['cone']:+.2f}   (vs cone)")
    if "cone_lda" in g:
        print(f"  LDA METRIC      cone_lda - cone = {g['cone_lda']-g['cone']:+.2f}")
print("\n" + "-" * W)
print("rp_ridge must reproduce the exp16 bar (ImageNet-R T=10 s0 R=32: 80.28) whenever")
print("   NROWS=0; if it does not, the replay is broken and no row means anything.")
print("FAIR RULE GAP is the headline: cone vs a generative head on the SAME lifted")
print("   features with the SAME per-class-only information. The old cone-vs-rp_ridge")
print("   comparison was naive-Bayes vs LDA and always flattered RanPAC.")
print("NON-NEGATIVITY is the foundational test: `sub` is a rank-R subspace at IDENTICAL")
print("   R*d storage, so cone - sub isolates w>=0 and nothing else. <= 0 means the")
print("   conic parameterisation earns nothing over a plain span.")
print("LDA is rank-limited to n_seen-1, so early stages are structurally starved. That is")
print("   a real property of LDA in CIL, not an artifact -- read A-Avg, not just A-Last.")
print("Sweep NROWS=5,10,20,50,0 for the regime axis. CUB200 has 27 rows/class and")
print("   ImageNet-A 18, so the low-NROWS rows predict those datasets directly.")
print("=" * W)
print(f"wrote {OUT}")
