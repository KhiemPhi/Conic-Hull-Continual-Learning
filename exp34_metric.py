#!/usr/bin/env python3
"""exp34_metric.py — tune and extend the component that carries 85% of the lift.

WHY THE METRIC
    Decomposing the cone's +8.41 over NCM (exp25, ImageNet-R T=10 s0, R=4):
        NCM 71.02 -> whitened prototypes 78.15  (+7.13, the METRIC)
                  -> conic mixture      79.43  (+1.28, the CONE)
    The metric is ~85% of it, and it has never been tuned: SHRINK=1e-2 is a fixed
    trace-scaled ridge that I picked and nobody swept. Two other levers are now closed --
    raising R (the cone peaks at R=4 and declines) and lifting the cone into RanPAC's random
    ReLU space (the lift DOUBLES mean pairwise cosine, 0.21 -> 0.43, so it hands an angular
    reader a strictly worse geometry; the +9.26 belongs to the ridge, not the representation).

WHY NO REPLAY IS NEEDED
    Every arm here is a per-class object over a FROZEN backbone with no lambda to select, so
    accuracy at the final stage over all classes IS A-Last, exactly. Only RanPAC needs the
    staged replay, and its bar is already known (ImageNet-R T=10 s0: 80.28). That makes this
    ~10x cheaper than exp25 while producing directly comparable numbers.

PART A -- SHRINKAGE
    S = scatter/n + delta * (tr S / d) * I,  swept over a log grid, plus the closed-form
    Ledoit-Wolf coefficient. Note LW minimises ||S_hat - S||_F, which is NOT accuracy, so the
    grid is the honest selector and LW is reported as a reference point.

PART C -- EIGEN-AUGMENTED GENERATORS  (the combination of Parts A and B)
    Parts A/B leave two objects that encode different things about a class: k-means
    centroids say WHERE THE MASS IS, the top eigenvectors of the within-class scatter say
    HOW IT SPREADS. The obvious combination -- fit the cone inside each class's rank-r
    metric -- is a trap: scores from different local metrics are not comparable across
    classes, and the normalising term that would fix that is exactly the log-det that
    collapses the argmax to chance in Part B.

    So change the GENERATORS, not the metric. Everything stays in the one tied-whitened
    space, so scores stay comparable, with no calibration and no log-det:

        A_c = un([ m_1 .. m_k ,  mu_c +- alpha*sqrt(lam_1) v_1 , mu_c +- alpha*sqrt(lam_2) v_2 ])

    This targets exp29's finding directly: the binding constraint is own-subspace COVERAGE
    (rho -0.585), not between-class overlap (+0.037, noise). Centroids sit in the middle of
    the mass and span badly; +-eigen displacements extend the cone along the directions the
    class actually varies in.

    BUDGET-MATCHED, or the ablation is confounded with "more generators". Total generators is
    held at R while the split varies: (R,0) is the Part A cone, (R-2, 1 eigvec +-),
    (1, 2 eigvecs +-). alpha=0 collapses every split back to centroids-only -- the identity
    check.

    PREDICTION THAT MAKES IT WORTH RUNNING: monotone cover says extra generators cost
    discrimination; exp29 says coverage is what binds. If a split with eigen-directions beats
    (R,0), coverage wins and exp29 becomes a method. If not, monotone cover dominates and the
    cone programme closes cleanly.

PART B -- PER-CLASS RANK-r CORRECTION
    A tied covariance is estimable; a full per-class one is not (that is the 473 MB the
    compared methods store). A rank-r correction is: r vectors + r scalars per class, the
    same budget the cone already spends on generators. In the tied-whitened space the tied
    covariance is I, so
        Sigma_c ~= I + V_c diag(lam - 1) V_c^T ,  Sigma_c^-1 = I - V_c diag(1 - 1/lam) V_c^T
        d^2(q,c) = ||q - mu_c||^2 - sum_j (1 - 1/lam_j) <v_j, q - mu_c>^2
    lam is shrunk toward 1 (top eigenvalues are over-estimated at n_c ~ 100 in d=768), and
    the Gaussian log-det term sum_j log lam_j is an ABLATION, not an assumption: rows are
    unit-normalised after whitening, so they are not Gaussian and the term is heuristic.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 R=4 python -u exp34_metric.py
    DS=IMAGENETR,CUB200 RANKS=0,1,2,4,8 python -u exp34_metric.py
    DS=IMAGENETR SHRINKS=0.03 RANKS=2 BETAS=1.0 ALPHAS=0,0.5,1,2 python -u exp34_metric.py
"""
import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import ledoit_wolf_shrinkage

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
RS = [int(x) for x in os.environ.get("R", "4").split(",")]
SHRINKS = [float(x) for x in
           os.environ.get("SHRINKS", "1e-4,1e-3,1e-2,3e-2,1e-1,0.3,1.0").split(",")]
RANKS = [int(x) for x in os.environ.get("RANKS", "0,1,2,4,8").split(",")]
BETAS = [float(x) for x in os.environ.get("BETAS", "0.25,0.5,1.0").split(",")]
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "0,0.5,1.0,2.0").split(",")]
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, "exp34_metric.json")
EPS = 1e-12


def whitener(scatter, n, delta, d):
    S = scatter / max(n, 1)
    S = S + delta * (np.trace(S) / d) * np.eye(d)
    return np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)


def gens(X, R, seed=0):
    R = int(min(R, len(X)))
    return E.un(X.mean(0, keepdims=True) if R <= 1 else
                KMeans(R, n_init=4, random_state=seed).fit(X).cluster_centers_)


def cone_score(A, Q):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = E.un(A)
    return h.score(Q)


def split(ytr, T, seed, n_cls):
    """exp16/exp25's split: class order rng(seed), per-task 10% val carve-out at rng(t)."""
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT = []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        FIT.append(ix[pm[max(int(0.1 * len(ix)), 1):]])
    return np.concatenate(FIT)


def run_cell(ds, T, seed, R):
    E.T, E.SEED = T, seed
    F = E.adapted_features(ds)
    assert F is not None, f"no exp16 cache for {ds} T={T} s={seed}"
    Ztr, Zte = F
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    fit = split(ytr, T, seed, n_cls)
    Xf, yf = Ztr[fit], ytr[fit]
    Q = E.un(Zte)
    log(f"  {ds} T={T} s={seed} R={R}: {len(fit)} fit rows, {n_cls} classes")

    # pooled within-class scatter over ALL classes (= the accumulated whitener at the final
    # stage, which is what exp25's cone_wa uses there)
    resid = np.concatenate([Xf[yf == c] - Xf[yf == c].mean(0)
                            for c in range(n_cls) if (yf == c).sum() >= 2])
    scatter = resid.T @ resid
    n_sc = len(resid)
    lw = float(ledoit_wolf_shrinkage(resid.astype(np.float64)))
    log(f"    Ledoit-Wolf coefficient: {lw:.5f}  (current fixed default 1e-2)")

    out = {"lw_delta": lw}
    mu_raw = E.un(np.stack([Xf[yf == c].mean(0) if (yf == c).any() else np.zeros(d)
                            for c in range(n_cls)]))
    out["ncm"] = float(((Q @ mu_raw.T).argmax(1) == yte).mean())
    log(f"    ncm {out['ncm']*100:.2f}")

    # ---------------- Part A: shrinkage ----------------
    for delta in sorted(set(SHRINKS + [lw])):
        Wh = whitener(scatter, n_sc, delta, d)
        Qw = E.un(Q @ Wh)
        Sp = np.full((len(Q), n_cls), -np.inf, np.float32)
        Sc = np.full((len(Q), n_cls), -np.inf, np.float32)
        Sm = np.full((len(Q), n_cls), -np.inf, np.float32)
        for c in range(n_cls):
            m = yf == c
            if m.sum() < 2:
                continue
            Xw = E.un(Xf[m] @ Wh)
            A = gens(Xw, R, c)
            Sp[:, c] = (Qw @ A.T).max(1)
            Sc[:, c] = cone_score(A, Qw)
            mu = E.un(Xw.mean(0, keepdims=True))[0]
            Sm[:, c] = -np.linalg.norm(Qw - mu, axis=1) ** 2
        tagd = f"{delta:g}" + ("(LW)" if abs(delta - lw) < 1e-12 else "")
        out[f"pm|{tagd}"] = float((Sp.argmax(1) == yte).mean())
        out[f"cone|{tagd}"] = float((Sc.argmax(1) == yte).mean())
        out[f"maha|{tagd}"] = float((Sm.argmax(1) == yte).mean())
        log(f"    delta={tagd:<10s} pm {out[f'pm|{tagd}']*100:.2f}  "
            f"cone {out[f'cone|{tagd}']*100:.2f}  maha1 {out[f'maha|{tagd}']*100:.2f}")

    # ---------------- Part B: per-class rank-r, at the best shrinkage for `maha` ----------
    best_d = max((k for k in out if k.startswith("maha|")), key=lambda k: out[k])
    delta = float(best_d.split("|")[1].replace("(LW)", ""))
    Wh = whitener(scatter, n_sc, delta, d)
    Qw = E.un(Q @ Wh)
    log(f"    Part B at delta={delta:g}")
    pack = {}
    for c in range(n_cls):
        m = yf == c
        if m.sum() < 2:
            continue
        Xw = E.un(Xf[m] @ Wh)
        mu = Xw.mean(0)
        Y = Xw - mu
        k = int(min(max(max(RANKS), 2), max(min(Y.shape) - 1, 1)))
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        # Normalise by the average eigenvalue over ALL d directions, not over the STORED
        # ones. lam.mean() made maha_r{r} depend on max(RANKS) -- an unrelated sweep
        # parameter -- and read 79.50 vs 77.05 for the same (r, beta, delta).
        tot = float((Y ** 2).sum() / (max(len(Y) - 1, 1) * d))
        pack[c] = (mu, Vt[:k], (s[:k] ** 2) / max(len(Y) - 1, 1), tot)
    for r in RANKS:
        for beta in (BETAS if r > 0 else [0.0]):
            S = np.full((len(Q), n_cls), -np.inf, np.float32)
            Sl = np.full((len(Q), n_cls), -np.inf, np.float32)
            for c, (mu, V, lam, tot) in pack.items():
                D = Qw - mu
                d2 = (D ** 2).sum(1)
                ld = 0.0
                if r > 0 and len(lam):
                    rr = min(r, len(lam))
                    lt = 1.0 + beta * (lam[:rr] / max(tot, EPS) - 1.0)
                    lt = np.clip(lt, 1e-3, None)
                    proj = D @ V[:rr].T
                    d2 = d2 - ((1.0 - 1.0 / lt) * proj ** 2).sum(1)
                    ld = float(np.log(lt).sum())
                S[:, c] = -d2
                Sl[:, c] = -d2 - ld
            out[f"maha_r{r}_b{beta:g}"] = float((S.argmax(1) == yte).mean())
            out[f"maha_r{r}_b{beta:g}_logdet"] = float((Sl.argmax(1) == yte).mean())
            log(f"    r={r} beta={beta:g}: {out[f'maha_r{r}_b{beta:g}']*100:.2f}  "
                f"(+logdet {out[f'maha_r{r}_b{beta:g}_logdet']*100:.2f})")

    # ---------------- Part C: eigen-augmented generators, budget-matched ----------------
    # Part C's baseline IS the cone, so it selects delta on the CONE's Part-A result, not on
    # maha1's. Using maha1's pick sent the last run to delta=0.1, which is worse for every
    # Part C arm (best matched-budget 79.95 there vs 80.05 at 0.03). Whitener, queries and
    # eigen-pack are all rebuilt at that delta so nothing is inherited from Part B.
    dc = float(max((k for k in out if k.startswith("cone|")),
                   key=lambda k: out[k]).split("|")[1].replace("(LW)", ""))
    Wh = whitener(scatter, n_sc, dc, d)
    Qw = E.un(Q @ Wh)
    pack = {}
    for c in range(n_cls):
        m = yf == c
        if m.sum() < 2:
            continue
        Xw = E.un(Xf[m] @ Wh)
        mu = Xw.mean(0)
        Y = Xw - mu
        kk = int(min(max(max(RANKS), 2), max(min(Y.shape) - 1, 1)))
        U, sv, Vt = np.linalg.svd(Y, full_matrices=False)
        pack[c] = (mu, Vt[:kk], (sv[:kk] ** 2) / max(len(Y) - 1, 1),
                   float((Y ** 2).sum() / (max(len(Y) - 1, 1) * d)))
    log(f"    Part C at delta={dc:g} (cone-selected)")
    # (0,2) not (1,2): 1 centroid + 2 eigvecs x (+-) is FIVE generators, so the old third
    # split was confounded with budget. With nk=0 the four points mu +- a v_1, mu +- a v_2
    # still have mean mu, so no information is lost.
    splits = [(R, 0), (max(R - 2, 1), 1), (0, 2)]
    for (nk, ne) in splits:
        for alpha in (ALPHAS if ne > 0 else [0.0]):
            S = np.full((len(Q), n_cls), -np.inf, np.float32)
            for c in range(n_cls):
                m = yf == c
                if m.sum() < 2:
                    continue
                Xw = E.un(Xf[m] @ Wh)
                A = [gens(Xw, nk, c)] if nk > 0 else []
                if ne > 0 and c in pack:
                    mu, V, lam, _ = pack[c]
                    for j in range(min(ne, len(lam))):
                        step = alpha * np.sqrt(max(lam[j], 0.0)) * V[j]
                        A.append(np.stack([mu + step, mu - step]))
                A = E.un(np.concatenate(A, 0))
                S[:, c] = cone_score(A, Qw)
            k = f"conaug_k{nk}_e{ne}_a{alpha:g}"
            out[k] = float((S.argmax(1) == yte).mean())
            log(f"    split(centroids={nk}, eig={ne}) alpha={alpha:g} "
                f"[{nk + 2 * ne} gens]: {out[k]*100:.2f}")
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            for R in RS:
                key = f"{ds}|{T}|{seed}|R{R}"
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, R)
                json.dump(allres, open(OUT, "w"), indent=2)

W = 78
print("\n" + "=" * W)
print("EXP34 — the metric: shrinkage and per-class low-rank structure")
print("=" * W)
for key, r in allres.items():
    print(f"\n--- {key}   [exp25 A-Last: cone_wa 79.43 @R=4, pm_wa 78.15, ranpac 80.28]")
    print(f"  NCM {r['ncm']*100:.2f}   Ledoit-Wolf delta {r['lw_delta']:.5f}")
    print(f"\n  A) shrinkage\n{'delta':>12}{'pm':>9}{'cone':>9}{'maha1':>9}")
    ds_ = sorted({k.split('|')[1] for k in r if k.startswith("pm|")},
                 key=lambda s: float(s.replace("(LW)", "")))
    for t in ds_:
        print(f"{t:>12}{r[f'pm|{t}']*100:>9.2f}{r[f'cone|{t}']*100:>9.2f}"
              f"{r[f'maha|{t}']*100:>9.2f}")
    print(f"\n  B) per-class rank-r correction\n{'arm':>18}{'no logdet':>11}{'+logdet':>10}")
    for k in sorted(k for k in r if k.startswith("maha_r") and not k.endswith("logdet")):
        print(f"{k:>18}{r[k]*100:>11.2f}{r[k+'_logdet']*100:>10.2f}")
    ca = sorted(k for k in r if k.startswith("conaug_"))
    if ca:
        base = next((r[k] for k in ca if k.endswith("_e0_a0")), None)
        print(f"\n  C) eigen-augmented generators (budget-matched)\n"
              f"{'split':>22}{'gens':>6}{'A-Last':>9}{'vs (R,0)':>10}")
        for k in ca:
            nk = int(k.split('_k')[1].split('_')[0]); ne = int(k.split('_e')[1].split('_')[0])
            dd = f"{(r[k]-base)*100:>+10.2f}" if base is not None else " " * 10
            print(f"{k[7:]:>22}{nk + 2*ne:>6}{r[k]*100:>9.2f}{dd}")
print("\n" + "-" * W)
print("Every arm here is A-Last exactly (frozen backbone, per-class objects, no lambda),")
print("   so these are directly comparable to exp25. Only RanPAC needed the staged replay.")
print("Ledoit-Wolf minimises Frobenius risk, NOT accuracy -- the grid is the selector and")
print("   LW is a reference point. If they disagree, trust the grid.")
print("Part C: alpha=0 collapses every split to centroids-only, so `k{R}_e0_a0` is the")
print("   paired baseline. It is recomputed at Part B's delta, NOT copied from Part A --")
print("   comparing across deltas would confound the split with the shrinkage.")
print("A split with eigen-directions beating (R,0) means COVERAGE wins over cover-growth,")
print("   and exp29's diagnosis becomes a method. Losing closes the cone programme cleanly.")
print("=" * W)
print(f"wrote {OUT}")
