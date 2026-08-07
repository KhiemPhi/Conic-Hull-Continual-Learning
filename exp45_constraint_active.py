#!/usr/bin/env python3
"""exp45_constraint_active.py — does w >= 0 BIND, and does it bind DIFFERENTIALLY?

THE CLAIM UNDER ATTACK
    exp38 read sub 78.02 vs cone 77.92 at R=32 k-means atoms and the writeup concluded
    "non-negativity is worth nothing". Two things are wrong with that inference.

    (1) IT IS ONE STAGE. Over the 10 stages of the same run the cone beats the subspace on
        7 of them (mean +0.18, sign test p=0.17, mid-run stages 2-8 mean +0.37). A-Last is
        the single noisiest statistic in the file and it is the only one that favours sub.

    (2) IT IS THE WRONG ATOMS. exp38's constraint suite ran on k-means centroids at a
        FIXED R=32 -- the generator set exp39 subsequently showed to be strictly dominated
        (oPCA g=0.5 is +1.58) and the allocation exp41 showed to be wrong (k5m24 is +0.57).
        The constraint was never tested at the atoms the method actually uses. If the
        constraint's value depends on the atom set -- and it must, because a cone over
        centroids that all sit near the class mean is nearly a ray, where w>=0 cannot bind
        -- then exp38 measured the constraint in the cell least able to show it.

WHAT DECIDES IT, and it is not another accuracy number
    A constraint that never activates cannot contribute. Define, for a query q and a class
    model V:
        gap(q, V) = ||P_span(V) q||  -  ||Pi_cone(V) q||   >= 0  ALWAYS (Moreau)
    gap == 0 exactly means the unconstrained least-squares solution was already
    non-negative, i.e. the cone and the subspace return the same number and no experiment
    comparing them can measure anything.

    But gap > 0 on its own is not enough either. The score is only used through an argmax
    over classes, so a constraint that shaves the same amount off every class is a constant
    and cannot change a decision. The quantity that matters is the DIFFERENTIAL:
        Delta = E[gap | c is FOREIGN]  -  E[gap | c is TRUE]
    Delta > 0 means non-negativity penalises wrong classes more than the right one, which
    is exactly a discriminative mechanism, and it should be read against the decision
    margin (top1 - top2) that it has to move to matter.

ARMS
    km32        k-means, fixed R=32              -- exp38's cell, reproduced
    opca        oPCA g=0.5, R_c=clip(n_c/5,24,128) -- the method's actual atoms (k5m24)
    Each scored with sub / cone / simplex, plus the gap diagnostics.

NO RANPAC. This file compares primitives at fixed atoms; the 10000-dim Gram is the
    expensive part of a replay and contributes nothing to the question. The bar is in
    exp39/exp41 and does not need re-measuring.

FINAL STAGE ONLY, exp40-style: the whitener and the generators still accumulate over all
    T stages (required for the state to be correct), but nothing is scored until the last.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp45_constraint_active.py
"""
import json
import os
import time

import numpy as np
import torch

import exp19_dataset_hull as E
import exp39_cone_construction as X
from exp38_fair_cone import s_sub, s_hull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
TAG = "augreg_in21k"
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
ARMS = os.environ.get("ARMS", "km32,opca").split(",")
GAMMA = float(os.environ.get("GAMMA", 0.5))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
NFOREIGN = int(os.environ.get("NFOREIGN", 8))    # foreign classes sampled per query
OUT = os.path.join(REPO, f"exp45_constraint_active_{TAG}.json")

un = X.un
E.T, E.SEED = T, SEED
assert (E.T, E.SEED) == (T, SEED)
Ztr, Zte = E.adapted_features(DS)
ytr, yte, n_cls = E.get_labels(DS)
d = Ztr.shape[1]
cpt = n_cls // T
order = np.random.default_rng(SEED).permutation(n_cls)
tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

FIT, VAL = [], []
for t in range(T):
    ix = np.where(np.isin(ytr, tasks[t]))[0]
    pm = np.random.default_rng(t).permutation(len(ix))
    nv = max(int(0.1 * len(ix)), 1)
    VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])


def rays_for(arm, n):
    return 32 if arm == "km32" else int(np.clip(n / 5.0, 24, 128))


def method_for(arm):
    return ("kmeans", 0.0) if arm == "km32" else ("opca", GAMMA)


# ---------------------------------------------------------------- accumulate
def build(arm):
    """Full staged accumulation of the tied whitener and the per-class generators.
    Returns the final whitener and {class: rays in ORIGINAL space}."""
    meth, g = method_for(arm)
    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A, nfit = {}, {}
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
            Fw = np.zeros((0, d), np.float32)
            if meth in X.DISCRIM and g > 0:
                oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                past = [A[o] for o in A if o not in tasks[t]]
                Fr = np.concatenate([Ztr[oth]] + past, 0)
                if len(Fr) > F_MAX:
                    Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                Fw = un(Fr @ Wh)
            A[c] = X.BUILD[meth](Xw, Fw, rays_for(arm, len(r)), int(c), g) @ Wh_inv
            nfit[int(c)] = len(r)
        log(f"  [{arm}] stage {t} accumulated")
    return Wh, A, nfit


def run(arm):
    Wh, A, nfit = build(arm)
    Qw = un(Zte @ Wh)
    seen = np.asarray(sorted(A))
    n = len(yte)

    S_sub = np.full((n, n_cls), -np.inf, np.float32)
    S_con = np.full((n, n_cls), -np.inf, np.float32)
    S_sim = np.full((n, n_cls), -np.inf, np.float32)
    negfrac, nray = {}, {}
    for c in seen:
        Ac = un(A[c] @ Wh)
        S_sub[:, c] = s_sub(Ac, Qw)
        S_con[:, c] = s_hull(Ac, Qw, "cone")
        S_sim[:, c] = s_hull(Ac, Qw, "simplex")
        nray[int(c)] = len(Ac)
        # unconstrained LSQ coefficients -> how much mass the projection has to kill
        Wls = np.linalg.lstsq(Ac.T, Qw.T, rcond=None)[0].T          # (n, R)
        neg = np.clip(-Wls, 0, None).sum(1)
        negfrac[int(c)] = (neg / (np.abs(Wls).sum(1) + 1e-12)).astype(np.float32)
    log(f"  [{arm}] scored, mean rays {np.mean(list(nray.values())):.1f}")

    def acc(S):
        return float((seen[S[:, seen].argmax(1)] == yte).mean())

    # ---------------------------------------------------------- gap diagnostics
    GAP = S_sub[:, seen] - S_con[:, seen]                            # >= 0 by Moreau
    assert GAP.min() > -1e-4, f"Moreau violated: {GAP.min()}"
    GAP = np.clip(GAP, 0, None)
    col = {int(c): j for j, c in enumerate(seen)}
    tcol = np.array([col[int(y)] for y in yte])
    rows = np.arange(n)
    gap_own = GAP[rows, tcol]
    NF = np.asarray(negfrac[int(seen[0])]).shape  # noqa: F841  (shape sanity)
    nf_own = np.array([negfrac[int(y)][i] for i, y in enumerate(yte)])

    rng = np.random.default_rng(0)
    fc = rng.integers(0, len(seen), size=(n, NFOREIGN))
    fix = fc == tcol[:, None]
    fc[fix] = (fc[fix] + 1) % len(seen)
    gap_for = GAP[rows[:, None], fc].mean(1)
    nf_for = np.stack([negfrac[int(seen[j])] for j in range(len(seen))], 1)[rows[:, None], fc].mean(1)

    # the margin the differential has to move
    part = np.partition(S_con[:, seen], -2, axis=1)
    margin = part[:, -1] - part[:, -2]

    # CONTESTED differential. The mean over 8 random foreign classes is dominated by
    # classes that were never in the running; a decision is made against the ONE foreign
    # class that scores highest. Restricting to that pair is the statistic that can move
    # an argmax. Computed under the SUBSPACE ranking, so the comparator is chosen without
    # reference to the constraint being tested.
    Ssub_s = S_sub[:, seen].copy()
    Ssub_s[rows, tcol] = -np.inf
    rcol = Ssub_s.argmax(1)                                # top foreign under `sub`
    gap_run = GAP[rows, rcol]
    beat_sub = S_sub[rows, seen[rcol]] > S_sub[rows, tcol]  # sub gets it wrong
    beat_con = S_con[rows, seen[rcol]] > S_con[rows, tcol]  # cone gets it wrong

    dis = seen[S_sub[:, seen].argmax(1)] != seen[S_con[:, seen].argmax(1)]
    cone_r = seen[S_con[:, seen].argmax(1)] == yte
    sub_r = seen[S_sub[:, seen].argmax(1)] == yte

    return {
        "mean_rays": float(np.mean(list(nray.values()))),
        "acc_sub": acc(S_sub), "acc_cone": acc(S_con), "acc_simplex": acc(S_sim),
        "gap_own_mean": float(gap_own.mean()),
        "gap_foreign_mean": float(gap_for.mean()),
        "gap_differential": float(gap_for.mean() - gap_own.mean()),
        "frac_inactive_own": float((gap_own < 1e-6).mean()),
        "frac_inactive_foreign": float((GAP[rows[:, None], fc] < 1e-6).mean()),
        "negmass_own": float(nf_own.mean()),
        "negmass_foreign": float(nf_for.mean()),
        "margin_mean": float(margin.mean()),
        "differential_over_margin": float((gap_for.mean() - gap_own.mean()) / margin.mean()),
        "gap_runnerup_mean": float(gap_run.mean()),
        "differential_contested": float(gap_run.mean() - gap_own.mean()),
        "contested_over_margin": float((gap_run.mean() - gap_own.mean()) / margin.mean()),
        "rescued": int((beat_sub & ~beat_con).sum()),
        "broken": int((~beat_sub & beat_con).sum()),
        "disagree_rate": float(dis.mean()),
        "disagree_cone_right": int((dis & cone_r).sum()),
        "disagree_sub_right": int((dis & sub_r).sum()),
        "n_test": int(n),
    }


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm in ARMS:
        key = f"{DS}|{T}|{SEED}|{arm}|g{GAMMA:g}|s{SHRINK:g}|v1"
        if key in allres:
            log(f"skip {key}"); continue
        log(f"=== {key}")
        allres[key] = run(arm)
        json.dump(allres, open(OUT, "w"), indent=2)

    W = 78
    print("\n" + "=" * W)
    print("EXP45 — is the non-negativity constraint ACTIVE, and is it DIFFERENTIAL?")
    print("=" * W)
    for k, r in sorted(allres.items()):
        print(f"\n--- {k}   ({r['mean_rays']:.1f} rays/class)")
        print(f"  accuracy    sub {r['acc_sub']*100:.2f}   cone {r['acc_cone']*100:.2f}"
              f"   simplex {r['acc_simplex']*100:.2f}"
              f"   |  cone-sub {(r['acc_cone']-r['acc_sub'])*100:+.2f}")
        print(f"  gap         own {r['gap_own_mean']:.5f}   foreign {r['gap_foreign_mean']:.5f}"
              f"   |  differential {r['gap_differential']:+.5f}")
        print(f"  inactive    own {r['frac_inactive_own']*100:.1f}%"
              f"   foreign {r['frac_inactive_foreign']*100:.1f}%")
        print(f"  neg mass    own {r['negmass_own']*100:.1f}%"
              f"   foreign {r['negmass_foreign']*100:.1f}%")
        print(f"  margin      {r['margin_mean']:.5f}"
              f"   |  differential/margin {r['differential_over_margin']:.3f}")
        if "gap_runnerup_mean" in r:
            print(f"  CONTESTED   own {r['gap_own_mean']:.5f}"
                  f"   runner-up {r['gap_runnerup_mean']:.5f}"
                  f"   |  differential {r['differential_contested']:+.5f}"
                  f"  ({r['contested_over_margin']:+.3f} margins)")
            print(f"  top-1 pair  rescued {r['rescued']}   broken {r['broken']}"
                  f"   |  net {r['rescued']-r['broken']:+d}")
        print(f"  argmax      disagree {r['disagree_rate']*100:.2f}%"
              f"   cone right {r['disagree_cone_right']}"
              f"   sub right {r['disagree_sub_right']}")
    print("\n" + "-" * W)
    print("READ `inactive` FIRST. If the constraint is inactive on most (q,c) pairs the")
    print("   cone IS the subspace there and no accuracy comparison between them measures")
    print("   anything -- exp38's -0.10 would be a null result, not a negative one.")
    print("`differential` is the only quantity that can change an argmax: a constraint that")
    print("   shaves the same amount off every class is a constant. Read it against")
    print("   `margin`; differential/margin is the fraction of a decision it can move.")
    print("`disagree` bounds the whole question empirically: sub and cone can differ on at")
    print("   most that many queries, so |cone-sub| accuracy is capped by it.")
    print("=" * W)
    print(f"wrote {OUT}")
