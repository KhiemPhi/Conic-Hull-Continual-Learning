#!/usr/bin/env python3
"""exp23_nu_independence.py — is `nu` independent of `r_perp`, or just a proxy for it?

exp22 found nu (negativity of the unconstrained least-squares coefficients = the conic
hypothesis in isolation) reaches AUROC 0.877 at R=16, next to r_perp's 0.964. If nu is
largely a restatement of the radial coordinate it adds nothing and "classes are cone-shaped"
is not supported. Three tests, none of which involve fitting anything to the test labels:

  1 CONDITIONAL AUROC (primary).  Bin queries by r_perp decile, computed on the POOLED
    ID+OOD distribution so the binning is label-blind, then take AUROC of -nu WITHIN each
    bin and average weighted by bin size. Inside a bin r_perp is ~constant, so any residual
    separation is signal nu carries that r_perp does not. ~0.50 => redundant.
  2 RANK CORRELATION.  Spearman(nu, r_perp) pooled. A blunt but assumption-free check.
  3 INCREMENTAL AUROC (secondary).  5-fold cross-validated logistic regression on {r_perp}
    vs {r_perp, nu}. Cross-validated because fitting on the evaluation rows would manufacture
    an improvement from nothing. Reports the lift from adding nu.

A bin with only one class present has undefined AUROC and is skipped, not counted as 0.5.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR python -u exp23_nu_independence.py
    DS=IMAGENETR,CUB200 RS=8,16,32 NCLS=50 python -u exp23_nu_independence.py
"""
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import exp19_dataset_hull as E

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
RS = [int(x) for x in os.environ.get("RS", "4,8,16").split(",")]
NCLS = int(os.environ.get("NCLS", 50))
NBINS = int(os.environ.get("NBINS", 10))
N_OOD = int(os.environ.get("N_OOD", 2000))     # OOD rows sampled per class (speed)
OUT = os.path.join(REPO, "exp23_nu_independence.json")
EPS = 1e-9


def span_orthant(V, Q):
    """Identical to exp22's, kept local so the two scripts cannot silently diverge."""
    V = np.asarray(V, np.float64)
    Q = np.asarray(E.un(Q), np.float64)
    s = np.linalg.svd(V, compute_uv=False)
    rank = int((s > s[0] * 1e-8).sum()) if len(s) and s[0] > 0 else 0
    B = np.linalg.svd(V, full_matrices=False)[2][:rank]
    r_perp = np.linalg.norm(Q - (Q @ B.T) @ B, axis=1)          # direct, not sqrt(1-cos^2)
    W = np.linalg.lstsq(V.T, Q.T, rcond=None)[0].T
    nu = np.linalg.norm(np.minimum(W, 0.0), axis=1) / (np.linalg.norm(W, axis=1) + EPS)
    return r_perp, nu, rank == len(V)


def cond_auc(r, nu, y, nbins=NBINS):
    """AUROC of -nu within bins of r_perp. Bins are quantiles of the POOLED r_perp, so the
    binning never sees a label."""
    edges = np.quantile(r, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    num = den = 0.0
    for i in range(nbins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue                                   # undefined, skip (never scored 0.5)
        num += m.sum() * roc_auc_score(y[m], -nu[m])
        den += m.sum()
    return float(num / den) if den else float("nan")


def cv_auc(Xf, y, folds=5):
    """Cross-validated logistic AUROC; fitting on the rows being scored would invent lift."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(Xf, y):
        sc = StandardScaler().fit(Xf[tr])
        p[te] = LogisticRegression(max_iter=2000).fit(sc.transform(Xf[tr]), y[tr]) \
            .predict_proba(sc.transform(Xf[te]))[:, 1]
    return float(roc_auc_score(y, p))


def run(ds):
    Ztr, Zte = E.adapted_features(ds)
    ytr, yte, ncls = E.get_labels(ds)
    Q = E.un(Zte)
    rng = np.random.default_rng(0)
    sizes = np.array([(ytr == c).sum() for c in range(ncls)])
    cls = np.sort(rng.permutation(np.where(sizes >= 8)[0])[:NCLS])
    log(f"{ds}: {len(cls)} classes evaluated, {len(Q)} test rows")
    out = {}
    for R in RS:
        a_r, a_nu, a_cond, a_sp, a_base, a_both = [], [], [], [], [], []
        skipped = 0
        for c in cls:
            X = Ztr[ytr == c]
            if len(X) < max(R, 10):
                skipped += 1
                continue
            V = E.un(KMeans(int(min(R, len(X))), n_init=4,
                            random_state=0).fit(X).cluster_centers_)
            idi = np.where(yte == c)[0]
            oodi = np.where(yte != c)[0]
            oodi = rng.permutation(oodi)[:N_OOD]
            if len(idi) < 10:
                skipped += 1
                continue
            sel = np.r_[idi, oodi]
            r, nu, ok = span_orthant(V, Q[sel])
            if not ok:
                skipped += 1
                continue
            y = np.r_[np.ones(len(idi)), np.zeros(len(oodi))]
            a_r.append(roc_auc_score(y, -r))
            a_nu.append(roc_auc_score(y, -nu))
            a_cond.append(cond_auc(r, nu, y))
            a_sp.append(spearmanr(r, nu).statistic)
            a_base.append(cv_auc(r.reshape(-1, 1), y))
            a_both.append(cv_auc(np.c_[r, nu], y))
        f = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
        out[f"R{R}"] = dict(auc_r_perp=f(a_r), auc_nu=f(a_nu), auc_nu_given_r=f(a_cond),
                            spearman=f(a_sp), cv_r=f(a_base), cv_both=f(a_both),
                            lift=f(a_both) - f(a_base), n=len(a_r), skipped=skipped)
        o = out[f"R{R}"]
        log(f"  R={R:<3d} r_perp {o['auc_r_perp']:.4f}  nu {o['auc_nu']:.4f}  "
            f"nu|r_perp {o['auc_nu_given_r']:.4f}  rho {o['spearman']:+.3f}  "
            f"cv r {o['cv_r']:.4f} -> r+nu {o['cv_both']:.4f} (lift {o['lift']:+.4f})"
            + (f"  [{o['skipped']} skipped]" if o['skipped'] else ""))
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    allres.setdefault(ds, {}).update(run(ds))
    json.dump(allres, open(OUT, "w"), indent=2)

W = 96
print("\n" + "=" * W)
print("EXP23 — does the conic coordinate `nu` carry signal beyond the radial one?")
print("=" * W)
for ds, res in allres.items():
    print(f"\n--- {ds}")
    print(f"{'R':>4}{'AUROC r_perp':>14}{'AUROC nu':>10}{'AUROC nu | r_perp':>19}"
          f"{'rho(r,nu)':>11}{'cv r':>8}{'cv r+nu':>9}{'lift':>8}")
    for R in RS:
        o = res.get(f"R{R}")
        if not o:
            continue
        print(f"{R:>4}{o['auc_r_perp']:>14.4f}{o['auc_nu']:>10.4f}"
              f"{o['auc_nu_given_r']:>19.4f}{o['spearman']:>+11.3f}"
              f"{o['cv_r']:>8.4f}{o['cv_both']:>9.4f}{o['lift']:>+8.4f}")
print("\n" + "-" * W)
print("AUROC nu | r_perp ~ 0.50  =>  nu is a restatement of the radial coordinate and the")
print("   'classes are cone-shaped' reading of exp22 is not supported.")
print("lift is the cross-validated gain from adding nu to r_perp: the operational version")
print("   of the same question.")
print("=" * W)
print(f"wrote {OUT}")
