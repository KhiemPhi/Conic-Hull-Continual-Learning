#!/usr/bin/env python3
"""exp40_error_analysis.py — WHERE does the cone lose to RanPAC?

WHY
    opca_g0.5 reads 79.77 A-Last against RanPAC's 80.28: a 0.51 gap. Every attempt to
    close it so far has come from inventing a mechanism and testing it (1 win in 6).
    This file inverts that: characterise the errors first, let the diagnosis pick the fix.

    It scores ONLY the final stage. The whitener and the generators still accumulate over
    all T stages -- that is required for the state to be correct -- but nothing is scored
    until the last one, which is where A-Last lives. That makes the run ~5x cheaper than
    exp39 and is the only reason a full-replay diagnostic is affordable.

THE FIVE DIAGNOSTICS, and what each would tell you to build

  A  COMPLEMENTARITY.  The 2x2 of (cone right/wrong) x (ranpac right/wrong).
       oracle = both_right + cone_only + ranpac_only is the ceiling on ANY combination
       rule. If oracle - ranpac is small the two make the same mistakes and no gating,
       fusion or re-ranking can help -- the cone must be made better on its own terms.
       If it is large, the errors are complementary and a selector is worth building.

  B  CLASS SIZE.  Accuracy binned by fit rows/class (ImageNet-R runs 35..308, 8.8x).
       The cone fits each class in isolation from its own rows; RanPAC pools every class
       through one shared Gram. So the cone should be hurt more by small classes. If the
       gap concentrates in the small-class bins, the fix is shrinkage/pooling of the
       per-class geometry toward a shared prior, not a better ray-selection rule.

  C  RECENCY.  Accuracy by the task a class was learned in.
       Generators are built in the BIRTH metric and re-metricated as the tied whitener
       accumulates; RanPAC's head is re-solved from scratch every stage. If the cone's
       deficit concentrates on task-0 classes, the birth-metric approximation is the
       problem (exp25 measured it as lossless at +0.05 -- this re-tests that at R=32).

  D  PREDICTION-FREQUENCY BIAS.  For each class, (times predicted)/(times true).
       ||Pi_C q|| grows with a cone's solid angle, so wide cones should be over-predicted.
       Correlating that ratio against each class's mean score on FOREIGN queries measures
       cone-size bias directly. A strong positive correlation says per-class calibration
       (exp38's cal_bg) is the fix and it is nearly free.

  E  ERROR SEVERITY.  Rank of the true class among all seen classes, on errors.
       Rank 2 errors are near-misses a better tie-break can fix. Rank 50 errors mean the
       class model is simply wrong and no re-ranking will save it. This decides whether
       margin-level fixes are worth anything at all.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp40_error_analysis.py
    DS=IMAGENETR T=10 SEED=0 METHOD=kmeans python -u exp40_error_analysis.py
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
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
METHOD = os.environ.get("METHOD", "opca")
GAMMA = float(os.environ.get("GAMMA", 0.5))
RMODE = os.environ.get("RMODE", "f32")     # f<R> fixed, or k<K> adaptive (exp41 syntax)
RMIN = int(os.environ.get("RMIN", 8))
RMAX = int(os.environ.get("RMAX", 128))


def rays_for(mode, n):
    """Same allocation rule as exp41 so the breakdown describes the model we actually have.
    exp40's original run used f32; the current best is k5, which hands large classes up to
    61 rays and small ones 8 -- i.e. it targets exactly the deficit this file diagnosed."""
    if mode.startswith("f"):
        return int(mode[1:])
    m = re.match(r"k([\d.]+)$", mode)
    assert m, f"bad RMODE {mode!r}"
    return int(np.clip(n / float(m.group(1)), RMIN, RMAX))
F_MAX = int(os.environ.get("F_MAX", 2000))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
OUT = os.path.join(REPO, f"exp40_error_analysis_{TAG}.json")

un = X.un
E.T, E.SEED = T, SEED
assert (E.T, E.SEED) == (T, SEED)
Ztr, Zte = E.adapted_features(DS)
ytr, yte, n_cls = E.get_labels(DS)
d = Ztr.shape[1]
cpt = n_cls // T
order = np.random.default_rng(SEED).permutation(n_cls)
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
A, nfit = {}, {}
log(f"accumulating {T} stages ({METHOD} g={GAMMA}, RMODE={RMODE}) -- last stage only")
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
        if METHOD in X.DISCRIM and GAMMA > 0:
            oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
            past = [A[o] for o in A if o not in tasks[t]]
            Fr = np.concatenate([Ztr[oth]] + past, 0)
            if len(Fr) > F_MAX:
                Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
            Fw = un(Fr @ Wh)
        A[c] = X.BUILD[METHOD](Xw, Fw, rays_for(RMODE, len(r)), int(c), GAMMA) @ Wh_inv
        nfit[int(c)] = len(r)
    for i, h in _H(un(Ztr[FIT[t]])):
        h = h.double()
        Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
        Y[torch.arange(h.shape[0]),
          torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
        G += h.T @ h; C += h.T @ Y
    log(f"  stage {t} accumulated")

seen = np.concatenate(tasks)
nval = len(VAL_ALL)
yv = ytr[VAL_ALL]
tei = np.arange(len(yte))
yt = yte
Qt = Zte


def acc_of(pred):
    return float((pred == yt).mean())


best, bw = -1.0, None
for lam in LAMBDAS:
    Wm = torch.linalg.solve(G + lam * eye, C)
    a = float((np.asarray(seen)[project(un(Ztr[VAL_ALL]), Wm)[:, seen].argmax(1)] == yv).mean())
    if a > best:
        best, bw = a, Wm
L = project(un(Qt), bw)
log("ranpac scored")

Qw = un(Qt @ Wh)
S = np.full((len(yt), n_cls), -np.inf, np.float32)
for c in seen:
    if c in A:
        S[:, c] = X.cone_score(un(A[c] @ Wh), Qw)
log("cone scored")

sa = np.asarray(seen)
p_cone = sa[S[:, seen].argmax(1)]
p_rp = sa[L[:, seen].argmax(1)]
ok_c, ok_r = p_cone == yt, p_rp == yt

W_ = 84
print("\n" + "=" * W_)
print(f"EXP40 — where does the cone lose?   {DS} T{T} s{SEED}  {METHOD} g={GAMMA} {RMODE}")
print("=" * W_)
print(f"cone   {ok_c.mean()*100:.2f}      ranpac {ok_r.mean()*100:.2f}"
      f"      gap {(ok_c.mean()-ok_r.mean())*100:+.2f}")

# ---- A complementarity
bb = int((ok_c & ok_r).sum()); co = int((ok_c & ~ok_r).sum())
ro = int((~ok_c & ok_r).sum()); nn = int((~ok_c & ~ok_r).sum())
N = len(yt)
print(f"\nA  COMPLEMENTARITY")
print(f"   both right {bb/N*100:6.2f}   cone only {co/N*100:6.2f}"
      f"   ranpac only {ro/N*100:6.2f}   both wrong {nn/N*100:6.2f}")
print(f"   ORACLE combination ceiling {(bb+co+ro)/N*100:6.2f}"
      f"   (+{((bb+co+ro)/N-ok_r.mean())*100:.2f} over ranpac)")
print(f"   -> small ceiling means the two share their errors and NO selector helps;")
print(f"      large means a gate/re-rank is worth building.")

# ---- B class size
print(f"\nB  CLASS SIZE   (cone fits each class alone; ranpac pools via one shared Gram)")
sz = np.array([nfit.get(int(c), 0) for c in yt])
print(f"   {'fit rows':<14}{'n test':>8}{'cone':>9}{'ranpac':>9}{'gap':>9}")
for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 10 ** 9)):
    m = (sz >= lo) & (sz < hi)
    if m.sum() == 0:
        continue
    print(f"   {f'{lo}-{hi if hi<10**9 else 'inf'}':<14}{int(m.sum()):>8}"
          f"{ok_c[m].mean()*100:>9.2f}{ok_r[m].mean()*100:>9.2f}"
          f"{(ok_c[m].mean()-ok_r[m].mean())*100:>+9.2f}")

# ---- C recency
print(f"\nC  RECENCY   (generators live in the BIRTH metric; ranpac is re-solved each stage)")
tk = np.array([task_of[int(c)] for c in yt])
print(f"   {'task':<14}{'n test':>8}{'cone':>9}{'ranpac':>9}{'gap':>9}")
for t in range(T):
    m = tk == t
    if m.sum() == 0:
        continue
    print(f"   {t:<14}{int(m.sum()):>8}{ok_c[m].mean()*100:>9.2f}"
          f"{ok_r[m].mean()*100:>9.2f}{(ok_c[m].mean()-ok_r[m].mean())*100:>+9.2f}")

# ---- D prediction-frequency bias vs cone size
print(f"\nD  PREDICTION-FREQUENCY BIAS   (does a wide cone get over-predicted?)")
ratio, width, ntrue = [], [], []
for c in seen:
    if c not in A:
        continue
    nt = int((yt == c).sum())
    if nt == 0:
        continue
    ratio.append(int((p_cone == c).sum()) / nt)
    width.append(float(S[yt != c, c].mean()))     # mean score on FOREIGN queries
    ntrue.append(nt)
ratio, width = np.array(ratio), np.array(width)
rho = float(np.corrcoef(width, ratio)[0, 1])
print(f"   over-prediction ratio: median {np.median(ratio):.2f}"
      f"   p10 {np.percentile(ratio,10):.2f}   p90 {np.percentile(ratio,90):.2f}")
print(f"   corr(mean foreign score, over-prediction ratio) = {rho:+.3f}")
print(f"   -> strongly positive means wide cones swallow queries and per-class")
print(f"      calibration (exp38 cal_bg) is the fix, and it is nearly free.")

# ---- E error severity
print(f"\nE  ERROR SEVERITY   (rank of the TRUE class among {len(seen)} seen, on errors)")
sub = S[:, seen]
tr_col = np.array([int(np.where(sa == c)[0][0]) for c in yt])
rk = (sub > sub[np.arange(len(yt)), tr_col][:, None]).sum(1) + 1
er = rk[~ok_c]
for q in (2, 3, 5, 10, 25):
    print(f"   true class in top-{q:<3}: {float((er <= q).mean())*100:6.2f}% of cone errors")
print(f"   median rank of the true class on errors: {int(np.median(er))}")
print(f"   -> mass at rank 2-3 means near-misses a better tie-break can fix;")
print(f"      a long tail means the class model itself is wrong.")
# ---- F confusion concentration
print(f"\nF  CONFUSION CONCENTRATION   (does hard-negative weighting have a basis?)")
tot_e = t1 = t3 = 0
npair = []
for c in seen:
    m = (yt == c) & ~ok_c
    if m.sum() == 0:
        continue
    cnt = np.bincount(p_cone[m], minlength=n_cls)
    srt = np.sort(cnt)[::-1]
    tot_e += int(m.sum()); t1 += int(srt[0]); t3 += int(srt[:3].sum())
    npair.append(int((cnt > 0).sum()))
print(f"   share of a class's errors going to its TOP-1 confuser : {t1/max(tot_e,1)*100:5.1f}%")
print(f"   share going to its TOP-3 confusers                    : {t3/max(tot_e,1)*100:5.1f}%")
print(f"   distinct wrong classes per true class: median {int(np.median(npair))}"
      f"  (chance would spread over {len(seen)-1})")
print(f"   -> concentrated (top-1 >~30%) means S_F should be weighted toward the few")
print(f"      classes that actually confuse c; diffuse means uniform S_F is already right")
print("=" * W_)

json.dump({"method": METHOD, "gamma": GAMMA, "rmode": RMODE,
           "rays_mean": float(np.mean([len(A[c]) for c in A])),
           "conf_top1": t1 / max(tot_e, 1), "conf_top3": t3 / max(tot_e, 1),
           "cone": float(ok_c.mean()), "ranpac": float(ok_r.mean()),
           "oracle": (bb + co + ro) / N,
           "cell": {"both": bb, "cone_only": co, "ranpac_only": ro, "neither": nn},
           "corr_width_overpred": rho,
           "err_rank_median": float(np.median(er)),
           "err_top2": float((er <= 2).mean()), "err_top5": float((er <= 5).mean())},
          open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
