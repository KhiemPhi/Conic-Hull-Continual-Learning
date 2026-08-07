#!/usr/bin/env python3
"""exp22_containment.py — test THE CONE THAT CONTAINS THE CLASS, not distance-to-a-cone.

WHY THIS IS A DIFFERENT EXPERIMENT
    Every cone test in this repo scores queries by cos(q, projection onto the cone) and
    grades it with AUROC. AUROC integrates over all thresholds and therefore never looks at
    the boundary -- the cone's only distinctive feature. SPA's guarantee, meanwhile, is
    about CONTAINMENT (under separability, every data point is a non-negative combination
    of the selected vertices), so SPA has never once been evaluated on what it is for.

THE TRAP THIS DESIGN AVOIDS
    Literal containment is vacuous here. R generators span an R-dim subspace of R^768, so a
    held-out point lands in it with probability zero: the NNLS residual is > 0 for EVERY
    query and containment is 0% for ID and OOD alike. Membership therefore has to be either
    decomposed or margined. This script does both.

MEASUREMENT 1 -- SPAN / ORTHANT DECOMPOSITION  (the sharp test)
    Membership in cone(V) factors into two independent questions:
        span    is q near span(V)?      r_perp = ||q - P_span q||   (the RADIAL coordinate)
        orthant are the unconstrained least-squares coefficients non-negative?
                nu = ||min(w,0)|| / ||w||  with  w = argmin ||V^T w - q||
    `nu` IS the conic hypothesis, isolated from everything else -- no solver, no budget, no
    scoring rule. If nu separates ID from OOD, classes really are cone-shaped. If its AUROC
    is ~0.5, non-negativity carries no information and the premise is dead outright.
    Only computed where rank(V) == R: with rank-deficient V the least-squares solution is
    non-unique and the SIGN PATTERN of the min-norm solution is arbitrary, which would make
    nu meaningless. Classes failing that test are counted and excluded, not silently kept.

MEASUREMENT 2 -- COVERAGE / EXCLUSION AT AN OPERATING POINT
    C_eps = {q : angle(q, cone(V)) <= eps}. Per class, eps* is set to the 95th percentile of
    the HELD-OUT same-class distances (so coverage is 0.95 by construction) and we report
    the fraction of other-class points excluded at that eps*. This is volume-where-it-
    matters: two cones can both contain the class; the better one is smaller.

MEASUREMENT 3 -- GENERATORS ON THEIR HOME TURF
    spa vs kmeans vs the exact hull (every training row a generator). If SPA gives tighter
    containment at equal coverage it has a real niche and we have been judging it by the
    wrong criterion; if it loses here too it is dead unambiguously.

BASELINES (matched storage, same coverage/exclusion protocol)
    cap    1 vector, angle to the class mean
    wcap   1 vector, angle to the class mean in a TIED-whitened space
    caps   R vectors, min angle to R k-means centroids (multi-prototype)

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR python -u exp22_containment.py            # ~112 img/class: best sweep range
    DS=CUB200 NCLS=50 RS=1,2,4,8 python -u exp22_containment.py
    SELFCHECK_ONLY=1 python -u exp22_containment.py        # just run the invariants
"""
import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
RS = [int(x) for x in os.environ.get("RS", "1,2,4,8,16").split(",")]
NCLS = int(os.environ.get("NCLS", 50))        # classes to build cones for (evaluated vs ALL test rows)
COV = float(os.environ.get("COV", 0.95))      # target ID coverage defining eps*
SHRINK = float(os.environ.get("SHRINK", 1e-2))
ITERS = int(os.environ.get("ITERS", 500))
GENS = os.environ.get("GENS", "kmeans,spa,exact").split(",")
OUT = os.path.join(REPO, "exp22_containment.json")
EPS = 1e-9


# ------------------------------------------------------------------ core geometry
def hull_from(A, iters=ITERS):
    """A hull with EXPLICIT generators. whiten is deliberately never used here: score()
    would then re-whiten the queries while these rays were assigned already-whitened, so
    any whitening in this script is done to the features up front instead."""
    A = E.un(np.asarray(A, np.float32))
    h = ConicHull(n_rays=len(A), nnls_iters=iters)
    h.extreme_rays_ = A
    return h


def cone_angle(A, Q):
    """Angular distance from each row of Q to cone(A), in radians. 0 == inside."""
    return np.arccos(np.clip(hull_from(A).score(Q), -1.0, 1.0))


def cap_angle(mu, Q):
    return np.arccos(np.clip(E.un(Q) @ E.un(mu.reshape(1, -1))[0], -1.0, 1.0))


def caps_angle(A, Q):
    return np.arccos(np.clip((E.un(Q) @ E.un(A).T).max(1), -1.0, 1.0))


def gens(X, R, kind, rs=0):
    """R generators of the requested kind. Never returns more rows than X has."""
    R = int(min(R, len(X)))
    if kind == "exact":
        return E.un(X)
    if R <= 1:
        return E.un(X.mean(0, keepdims=True))
    if kind == "kmeans":
        return E.un(KMeans(R, n_init=4, random_state=rs).fit(X).cluster_centers_)
    if kind == "spa":
        # oversample shrunk to the class rank -- 3R candidates in an (n-1)-dim span is the
        # degenerate regime, and it is what every historical call site used.
        ov = max(1, min(3, (len(X) - 2) // R))
        return ConicHull(n_rays=R, ray_init="spa", spa_oversample=ov,
                         ray_diversity="hybrid" if ov > 1 else "spa").fit(X).extreme_rays_
    raise ValueError(kind)


def span_orthant(V, Q):
    """(r_perp, nu, ok) -- the two halves of membership, plus whether nu is meaningful.

    r_perp : ||q - P_span(V) q||          out-of-span residual  (radial part)
    nu     : ||min(w,0)|| / ||w||, w = argmin ||V^T w - q||   (orthant part == the cone)
    ok     : False when rank(V) < len(V); the min-norm least-squares solution is then one of
             infinitely many and its sign pattern is arbitrary, so nu would be noise.
    """
    V = np.asarray(V, np.float64)
    Q = np.asarray(E.un(Q), np.float64)
    s = np.linalg.svd(V, compute_uv=False)
    rank = int((s > s[0] * 1e-8).sum()) if len(s) and s[0] > 0 else 0
    B = np.linalg.svd(V, full_matrices=False)[2][:rank]          # (rank, d) orthonormal rows
    # Direct residual, NOT sqrt(1 - ||P q||^2): the trig identity loses all precision near
    # the span (cos error 1e-8 -> r_perp error 1.4e-4, verified) and generators would then
    # fail the "r_perp == 0" invariant for purely numerical reasons.
    r_perp = np.linalg.norm(Q - (Q @ B.T) @ B, axis=1)
    W = np.linalg.lstsq(V.T, Q.T, rcond=None)[0].T               # (n, R)
    nu = (np.linalg.norm(np.minimum(W, 0.0), axis=1)
          / (np.linalg.norm(W, axis=1) + EPS))
    return r_perp, nu, rank == len(V)


def excl_at_cov(d_id, d_ood, cov=COV):
    """Exclusion of out-of-class points at the smallest eps* with in-class coverage >= cov.

    With ~30 held-out rows per class the achievable coverage is quantised to multiples of
    1/30, so the realised coverage is >= cov but not exactly cov. It is identical across
    methods for a given class, which is what makes the comparison fair."""
    eps = float(np.quantile(d_id, cov, method="higher"))
    return float((d_ood > eps).mean()), eps


def auc(d_id, d_ood):
    """AUROC with SMALLER distance = more in-distribution."""
    return float(roc_auc_score(np.r_[np.ones(len(d_id)), np.zeros(len(d_ood))],
                               np.r_[-np.asarray(d_id), -np.asarray(d_ood)]))


# ------------------------------------------------------------------ self-checks
def selfcheck():
    rng = np.random.default_rng(0)
    X = E.un(np.abs(rng.normal(size=(200, 64))) + 0.3 * rng.normal(size=(200, 64)))
    Q = E.un(np.abs(rng.normal(size=(50, 64))))
    fails = []

    def chk(name, cond, detail=""):
        print(f"   {'PASS' if cond else '**FAIL**'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    A = gens(X, 8, "kmeans")
    d0 = cone_angle(A, A)                       # a generator must lie IN its own cone
    chk("generators are inside their own cone", np.abs(d0).max() < 1e-3,
        f"max angle {np.abs(d0).max():.2e}")
    d = cone_angle(A, Q)
    chk("angles in [0, pi]", bool((d >= -1e-9).all() and (d <= np.pi + 1e-9).all()),
        f"[{d.min():.3f}, {d.max():.3f}]")
    mu = X.mean(0)
    chk("R=1 cone == cap exactly",
        np.allclose(cone_angle(gens(X, 1, "kmeans"), Q), cap_angle(mu, Q), atol=1e-5),
        f"max|d| {np.abs(cone_angle(gens(X,1,'kmeans'),Q)-cap_angle(mu,Q)).max():.2e}")
    chk("R=1 caps == cap exactly",
        np.allclose(caps_angle(gens(X, 1, "kmeans"), Q), cap_angle(mu, Q), atol=1e-5))
    rp, nu, ok = span_orthant(A, Q)
    pr = np.linalg.norm(np.asarray(E.un(Q), np.float64)
                        @ np.linalg.svd(np.asarray(A, np.float64),
                                        full_matrices=False)[2][:len(A)].T, axis=1)
    chk("||P q||^2 + r_perp^2 == 1", np.allclose(pr ** 2 + rp ** 2, 1.0, atol=1e-8))
    chk("nu in [0,1]", bool((nu >= -1e-12).all() and (nu <= 1 + 1e-12).all()),
        f"[{nu.min():.3f}, {nu.max():.3f}]")
    chk("rank(V)==R flag true for kmeans generators", ok)
    rpx, nux, _ = span_orthant(A, A)            # a generator: in-span, non-negative coeffs
    chk("generators: r_perp==0 and nu==0",
        bool(np.abs(rpx).max() < 1e-6 and np.abs(nux).max() < 1e-6),
        f"r_perp {np.abs(rpx).max():.1e}  nu {np.abs(nux).max():.1e}")
    e, eps = excl_at_cov(d, d + 1.0)
    chk("coverage at eps* is >= COV by construction", (d <= eps).mean() >= COV - 1e-9,
        f"cov {(d <= eps).mean():.4f}")
    chk("exclusion of a strictly-farther set is 1.0", e == 1.0)
    return fails


print("SELF-CHECKS")
_f = selfcheck()
if _f:
    raise SystemExit(f"self-checks failed: {_f} -- refusing to produce numbers")
if int(os.environ.get("SELFCHECK_ONLY", 0)):
    raise SystemExit(0)


# ------------------------------------------------------------------ one dataset
def run(ds):
    Ztr, Zte = E.adapted_features(ds)
    ytr, yte, ncls = E.get_labels(ds)
    Q = E.un(Zte)
    rng = np.random.default_rng(0)
    sizes = np.array([(ytr == c).sum() for c in range(ncls)])
    elig = np.where(sizes >= 8)[0]
    cls = np.sort(rng.permutation(elig)[:min(NCLS, len(elig))])
    log(f"{ds}: {ncls} classes ({len(cls)} evaluated), {len(Q)} test rows, "
        f"img/class med {int(np.median(sizes))}")

    # tied within-class whitener for the `wcap` baseline (estimable when per-class is not)
    Xc = np.concatenate([Ztr[ytr == c] - Ztr[ytr == c].mean(0) for c in range(ncls)])
    S = (Xc.T @ Xc) / len(Xc) + SHRINK * np.eye(Ztr.shape[1]) * np.trace(Xc.T @ Xc) \
        / len(Xc) / Ztr.shape[1]
    Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)
    Qw = E.un(Q @ Wh)

    out = {}
    for R in RS:
        methods = {f"cone_{g}": g for g in GENS if g != "exact" or R == RS[-1]}
        rows = {m: [] for m in methods}
        rows.update({"cap": [], "wcap": [], "caps": []})
        dec = {"r_perp": [], "nu": []}
        bad_rank = 0
        for c in cls:
            X = Ztr[ytr == c]
            idm, oodm = (yte == c), (yte != c)
            if idm.sum() < 10:
                continue
            for m, kind in methods.items():
                d = cone_angle(gens(X, R, kind), Q)
                rows[m].append(excl_at_cov(d[idm], d[oodm])[0])
            dc = cap_angle(X.mean(0), Q)
            rows["cap"].append(excl_at_cov(dc[idm], dc[oodm])[0])
            dw = np.arccos(np.clip(Qw @ E.un((E.un(X @ Wh)).mean(0).reshape(1, -1))[0],
                                   -1.0, 1.0))
            rows["wcap"].append(excl_at_cov(dw[idm], dw[oodm])[0])
            dk = caps_angle(gens(X, R, "kmeans"), Q)
            rows["caps"].append(excl_at_cov(dk[idm], dk[oodm])[0])
            # span/orthant on the kmeans generators
            rp, nu, ok = span_orthant(gens(X, R, "kmeans"), Q)
            if ok:
                dec["r_perp"].append(auc(rp[idm], rp[oodm]))
                dec["nu"].append(auc(nu[idm], nu[oodm]))
            else:
                bad_rank += 1
        for m, v in rows.items():
            if v:
                out[f"excl@{int(COV*100)}|{m}|{R}"] = float(np.mean(v))
        for k, v in dec.items():
            if v:
                out[f"auroc|{k}|{R}"] = float(np.mean(v))
        out[f"rankdef|{R}"] = bad_rank
        log(f"  R={R:<3d} excl@{int(COV*100)}: " +
            "  ".join(f"{m} {out[f'excl@{int(COV*100)}|{m}|{R}']:.4f}"
                      for m in rows if f"excl@{int(COV*100)}|{m}|{R}" in out) +
            f"   | AUROC r_perp {out.get(f'auroc|r_perp|{R}', float('nan')):.4f} "
            f"nu {out.get(f'auroc|nu|{R}', float('nan')):.4f}"
            + (f"  [{bad_rank} rank-deficient]" if bad_rank else ""))
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    allres.setdefault(ds, {}).update(run(ds))
    json.dump(allres, open(OUT, "w"), indent=2)

# ------------------------------------------------------------------ report
W = 96
print("\n" + "=" * W)
print(f"EXP22 — containment: exclusion of other classes at {int(COV*100)}% in-class coverage")
print("=" * W)
for ds, res in allres.items():
    ms = ["cone_kmeans", "cone_spa", "cone_exact", "caps", "wcap", "cap"]
    print(f"\n--- {ds}")
    print(f"{'R':>4}" + "".join(f"{m:>13}" for m in ms) +
          f"{'AUROC r_perp':>14}{'AUROC nu':>10}")
    for R in RS:
        cells = "".join(
            f"{res.get(f'excl@{int(COV*100)}|{m}|{R}', float('nan')):>13.4f}" for m in ms)
        print(f"{R:>4}{cells}{res.get(f'auroc|r_perp|{R}', float('nan')):>14.4f}"
              f"{res.get(f'auroc|nu|{R}', float('nan')):>10.4f}")
print("\n" + "-" * W)
print("AUROC nu is THE test: nu is the non-negativity of the least-squares coefficients,")
print("   i.e. the conic hypothesis with the solver, budget and scoring rule stripped out.")
print("   ~0.50 => classes are not cone-shaped and no conic method can recover it.")
print("AUROC r_perp is the radial/subspace half of membership, shown for contrast.")
print("cone_spa is SPA judged on containment -- the criterion it was actually designed for.")
print("=" * W)
print(f"wrote {OUT}")
