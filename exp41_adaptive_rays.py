#!/usr/bin/env python3
"""exp41_adaptive_rays.py — the three fixes exp40's error analysis pointed at.

WHAT exp40 FOUND (ImageNet-R T=10 s0, opca g=0.5, R=32, final stage)
        fit rows    n test    cone   ranpac     gap
        0-50           239   71.97    57.74   +14.23
        50-100        1559   78.58    75.63    +2.95
        100-200       3433   80.31    82.61    -2.30
        200+           769   80.10    86.35    -6.24
    Both extremes ~4.5 sigma. The cone CRUSHES RanPAC on small classes (its ridge column
    is starved and underfits) and LOSES badly on large ones (R=32 saturates while RanPAC's
    shared Gram keeps exploiting samples). The 0.78 deficit is a CAPACITY MISMATCH, not
    geometry. Recency was eliminated (no trend across tasks); cone-size bias is real but
    modest (corr +0.385); errors are near-misses (median true rank 4, 33% at rank 2).

FIX 1 — ADAPTIVE PER-CLASS R.   R_c = clip(n_c / K, RMIN, RMAX)
    Every R sweep so far was GLOBAL, which conflates the two regimes: R=64 looked bad
    because a quarter of classes clamped, but the large classes wanted it and never got
    to say so. This is removing an arbitrary constant, not adding a mechanism.
    Roughly STORAGE-NEUTRAL: at K=3 ImageNet-R's 21600 fit rows give ~7200 rays against
    6400 at fixed R=32. `mean_rays` is reported so the comparison stays honest.

FIX 2 — SIZE-KEYED BETA.   beta_c = beta0 * (n_ref / n_c)^p,  n_ref = median seen n_c
    An oracle using the cone below 100 rows and RanPAC above scores 81.62 (+1.34 over
    RanPAC). That oracle is not applicable per query -- you do not know the true class --
    but weighting each CLASS by its stored count is, and it needs no val-time per-query
    decision. p=0 recovers the global beta, and is reported as the control.

FIX 3 — cal_bg.   s_c <- s_c - mean(s_c on foreign stored rays)
    exp40 measured corr(cone width, over-prediction ratio) = +0.385, so wide cones do
    swallow queries. Mean-centring targets that directly. MEAN ONLY: exp38 measured that
    dividing by a per-class sigma estimated from few samples is catastrophic (90.9 -> 59.8);
    here sigma would come from BG_MAX foreign rays and is better conditioned, but the bias
    lives in the mean level so the sigma adds variance without addressing the hypothesis.

BETA SELECTION.  beta0 and p are picked jointly on val. That selection was measured to
    swing a headline by 0.77 on a 0.09 feature perturbation, so a FIXED-beta column
    (BETA_FIX) is reported alongside every selected one. Trust the fixed column when they
    disagree; the raw cone arms have no beta at all and are the cleanest comparison.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp41_adaptive_rays.py
    DS=IMAGENETR T=10 SEED=0 RMODES=f32,k2,k3,k5 python -u exp41_adaptive_rays.py
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
RMODES = os.environ.get("RMODES", "f32,k3,k5").split(",")
METHOD = os.environ.get("METHOD", "opca")
GAMMA = float(os.environ.get("GAMMA", 0.5))
RMIN = int(os.environ.get("RMIN", 8))
RMAX = int(os.environ.get("RMAX", 128))
BG_MAX = int(os.environ.get("BG_MAX", 2000))
F_MAX = int(os.environ.get("F_MAX", 2000))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
PS = [0.0, 0.25, 0.5, 1.0]
BETA_FIX = float(os.environ.get("BETA_FIX", 0.5))
CALIB = int(os.environ.get("CALIB", 1))     # cal_bg measured -1.6..-2.9; off = faster
FUSE = int(os.environ.get("FUSE", 1))       # 0 = raw arms only, skips ALL val scoring
SHRINK = float(os.environ.get("SHRINK", 3e-2))
OUT = os.path.join(REPO, f"exp41_adaptive_rays_{TAG}.json")

un = X.un


def rays_for(mode, n):
    """f<R> = fixed. k<K>[m<RMIN>][M<RMAX>] = one ray per K fit rows, clipped.

    R_c = n_c / K is per-class RANK SELECTION: "estimate one direction per K samples".
    That is the standard statistical rule, and it is what makes the adaptive arms work --
    a 35-row class cannot support 32 reliable directions and a 300-row class is wasted on
    them. m/M override the global floor/ceiling per mode so RMIN can be swept: exp40 found
    the cone WINS on small classes (+14.23), so cutting them to 8 rays may be discarding
    the one regime the cone owns.
    """
    if mode.startswith("f"):
        return int(mode[1:])
    m = re.match(r"k([\d.]+)(?:m(\d+))?(?:M(\d+))?$", mode)
    assert m, f"bad rmode {mode!r}; want f<R> or k<K>[m<RMIN>][M<RMAX>]"
    lo = int(m.group(2)) if m.group(2) else RMIN
    hi = int(m.group(3)) if m.group(3) else RMAX
    return int(np.clip(n / float(m.group(1)), lo, hi))


def zs(A, seen):
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


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
    A = {m: {} for m in RMODES}
    nfit = {}
    sfxs = ["", "_cal"] + (["_fuse", "_fuseP", "_fuseFix"] if FUSE else [])
    arms = ["ranpac"] + [f"{m}{sfx}" for m in RMODES for sfx in sfxs]
    res = {a: [] for a in arms}
    meta = {m: [] for m in RMODES}

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
            nfit[int(c)] = len(r)
            Xw = un(Ztr[r] @ Wh)
            Fw = np.zeros((0, d), np.float32)
            if METHOD in X.DISCRIM and GAMMA > 0:
                oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                past = [A[RMODES[0]][o] for o in A[RMODES[0]] if o not in tasks[t]]
                Fr = np.concatenate([Ztr[oth]] + past, 0)
                if len(Fr) > F_MAX:
                    Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                Fw = un(Fr @ Wh)
            for m in RMODES:                      # SAME foreign set across R modes, so the
                Rc = rays_for(m, len(r))          # only thing varying is the ray budget
                A[m][c] = X.BUILD[METHOD](Xw, Fw, Rc, int(c), GAMMA) @ Wh_inv

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
        zLv = zs(project(un(Ztr[VAL_ALL[:nval]]), bw), seen)
        zLt = zs(project(un(Zte[tei]), bw), seen)
        res["ranpac"].append(acc(zLt, yt))

        Qvw = un(Ztr[VAL_ALL[:nval]] @ Wh)
        Qtw = un(Zte[tei] @ Wh)
        nref = float(np.median([nfit[int(c)] for c in seen if int(c) in nfit]))
        # per-class beta multiplier from STORED COUNTS only -- no val-time per-query
        # decision, no extra storage
        mult = {p: np.ones(n_cls) for p in PS}
        for p in PS:
            for c in seen:
                if int(c) in nfit:
                    mult[p][c] = (nref / max(nfit[int(c)], 1)) ** p

        for m in RMODES:
            have = [c for c in seen if c in A[m]]
            Sv = np.full((nval, n_cls), -np.inf, np.float32)
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            Sv_c, St_c = Sv.copy(), St.copy()
            pool = np.concatenate([A[m][o] for o in have])
            owner = np.concatenate([np.full(len(A[m][o]), o) for o in have])
            for c in have:
                Ac = un(A[m][c] @ Wh)
                if FUSE:
                    Sv[:, c] = X.cone_score(Ac, Qvw)
                St[:, c] = X.cone_score(Ac, Qtw)
                bg = 0.0
                if CALIB:
                    fo = np.where(owner != c)[0]
                    if len(fo) > BG_MAX:
                        fo = rng.choice(fo, BG_MAX, replace=False)
                    bg = X.cone_score(Ac, un(pool[fo] @ Wh)).mean() if len(fo) else 0.0
                Sv_c[:, c] = Sv[:, c] - bg
                St_c[:, c] = St[:, c] - bg
            meta[m].append(float(np.mean([len(A[m][c]) for c in have])))
            res[m].append(acc(zs(St, seen), yt))
            res[f"{m}_cal"].append(acc(zs(St_c, seen), yt))

            if not FUSE:
                continue
            zSv, zSt = zs(Sv_c, seen), zs(St_c, seen)     # calibrated scores into fusion
            b0 = max(BETAS, key=lambda bb: acc(zLv + bb * zSv, yv))
            res[f"{m}_fuse"].append(acc(zLt + b0 * zSt, yt))
            bp = max(((bb, p) for bb in BETAS for p in PS),
                     key=lambda z: acc(zLv + z[0] * mult[z[1]][None, :] * zSv, yv))
            res[f"{m}_fuseP"].append(acc(zLt + bp[0] * mult[bp[1]][None, :] * zSt, yt))
            res[f"{m}_fuseFix"].append(acc(zLt + BETA_FIX * zSt, yt))

        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}" + "".join(
            f" | {m} raw {res[m][-1]*100:.2f} cal {res[f'{m}_cal'][-1]*100:.2f}"
            f" (r{meta[m][-1]:.0f})" for m in RMODES))

    del G, C, P, eye
    torch.cuda.empty_cache()
    out = {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for a, v in res.items() if v}
    for m in RMODES:
        out[m]["mean_rays"] = float(np.mean(meta[m]))
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            key = (f"{ds}|{T}|{seed}|{METHOD}g{GAMMA:g}|{'+'.join(RMODES)}"
                   f"|r{RMIN}-{RMAX}_b{BETA_FIX:g}_f{F_MAX}|m{M_RP}_s{SHRINK:g}|v1")
            if key in allres:
                log(f"skip {key}"); continue
            log(f"=== {key}")
            allres[key] = run_cell(ds, T, seed)
            json.dump(allres, open(OUT, "w"), indent=2)

W = 92
print("\n" + "=" * W)
print("EXP41 — adaptive per-class rays + size-keyed beta + cal_bg")
print("=" * W)
for key, r in sorted(allres.items()):
    print(f"\n--- {key}")
    rp = r["ranpac"]["A_last"] * 100
    print(f"  {'arm':<14}{'A-Last':>9}{'A-Avg':>9}{'rays/cls':>10}{'vs ranpac':>11}")
    print(f"  {'ranpac':<14}{rp:>9.2f}{r['ranpac']['A_avg']*100:>9.2f}{'--':>10}{'--':>11}")
    for a, v in sorted(((a, v) for a, v in r.items() if a != "ranpac"),
                       key=lambda kv: -kv[1]["A_last"]):
        print(f"  {a:<14}{v['A_last']*100:>9.2f}{v['A_avg']*100:>9.2f}"
              f"{v.get('mean_rays', float('nan')):>10.1f}"
              f"{v['A_last']*100-rp:>+11.2f}")
    g = {a: v["A_last"] * 100 for a, v in r.items()}
    if "f32" in g and "f32_fuse" in g:
        for m in [x for x in r if x in RMODES and x != "f32"]:
            print(f"\n  ADAPTIVE R   {m} - f32 = {g[m]-g['f32']:+.2f} raw"
                  f"   ({r[m]['mean_rays']:.1f} vs {r['f32']['mean_rays']:.1f} rays/cls)")
        print(f"  CALIBRATION  f32_cal - f32 = {g['f32_cal']-g['f32']:+.2f}")
        print(f"  SIZE-KEYED   f32_fuseP - f32_fuse = {g['f32_fuseP']-g['f32_fuse']:+.2f}")
        print(f"  BETA STABILITY  f32_fuse {g['f32_fuse']:.2f} vs fixed"
              f" {g['f32_fuseFix']:.2f}  (diff {g['f32_fuse']-g['f32_fuseFix']:+.2f})")
print("\n" + "-" * W)
print("ranpac must be the exp16 bar (IN-R T=10 s0: 80.28); f32 raw must be ~79.50, the")
print("   exp40 value for this exact configuration. Check both before reading anything.")
print("rays/cls is the storage check: adaptive R is only a fair comparison against f32")
print("   if the mean is close to 32. K=3 should land near 36 on ImageNet-R.")
print("Read the RAW arms first -- they have no beta and so no selection noise. If")
print("   adaptive R does not move the raw number, the capacity story is wrong and the")
print("   fused columns are just beta fitting.")
print("BETA STABILITY: if selected and fixed beta disagree by more than ~0.3, the val set")
print("   cannot resolve beta at this stage and the selected column is not trustworthy.")
print("=" * W)
print(f"wrote {OUT}")
