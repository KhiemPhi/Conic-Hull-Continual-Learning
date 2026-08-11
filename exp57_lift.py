#!/usr/bin/env python3
"""exp57_lift.py -- three cached-feature levers against the 4.4pt of headroom exp56 left.

WHY THIS FILE EXISTS
    exp56 landed 82.52/87.39 on IMAGENETR T=10 PILOT seed 0 against GR-LoRA's 82.09/86.20.
    The number that says how much is still on the table is exp55's oracle:

        best single member 80.73    ORACLE 86.90    headroom 6.17
        sel_FE 82.52  ->  captured 1.79 of 6.17 = 29%.  4.38pt unclaimed.

    Nothing here trains anything. Every member's features are already cached by exp16/exp55,
    so all three levers are read-out work on top of them. That is the entire point: if a lever
    needs a training run it belongs in a different file.

THE THREE LEVERS, AND THE MEASUREMENT THAT MOTIVATES EACH

  1. RAYS ARE CAPPED BY DATA, NOT BY THE MODEL.  exp56's FE gain is MONOTONE in R:
        f4 +0.32   f8 +0.32   f16 +0.40   f32 +0.49   f64 +0.61      (A-Avg over ens)
     and then f128 DROPS to +0.47 -- but f128 is the clipped arm. `b_kmeans` does
     `k = min(R_, len(Xw))` and IMAGENETR has a median of 100 fit rows per class, so f128 is
     not a 128-ray cone at all, it is one ray per training point. The curve is still rising
     where the data runs out. SYNTH backfills each class with Gaussian draws in the oPCA
     subspace so k-means can actually produce R centroids at R > n_rows.
        PRE-REGISTERED: if R=128 WITH backfill does not beat R=64 by at least +0.10 A-Avg,
        the monotonicity was a proxy for "more rays ~ more training points" and the ray
        ceiling is real, not an artefact. Lever closed.

  2. THE HEADROOM IS ROW-RANDOM, AND EVERY FITTED PARAMETER HAS LOST.  exp55 measured
     `oracle_class_cv` BELOW best_single (cv share -6%), so a per-CLASS member rule captures
     nothing -- that door is shut. But the ORACLE selects per ROW, and nothing has tested
     that. The trap is equally well measured: EF, JT and FEW all added fitted parameters and
     all lost, with val-test gaps 1.98/2.37/2.46 against FE's 0.86.
     So the per-row rules here are PARAMETER-FREE by construction. Each member gets a
     confidence on each row (margin / negentropy / top-z), weights are softmax(conf/tau) over
     members, and tau comes from a fixed grid whose endpoints are the two things we already
     understand: tau=inf IS uniform averaging (exactly FE), tau=0 is hard per-row selection
     (the routing analogue GR-LoRA's interface() implements, and which our own routing
     experiment found capped). One global tau, chosen on val, is the ONLY fitted quantity.
        Because tau=inf reproduces FE exactly, this family cannot lose to FE on VAL. Any
        test-side loss is therefore pure val overfit and is reported as such -- same
        construction as FEW, which is how we caught FEW.
        PRE-REGISTERED: a combiner is interesting only if it beats `uniform` by >= +0.30
        A-Avg AND its val-test gap is no worse than uniform's. exp49's unpinned noise floor
        is 0.27; anything under that is not a result.

  3. MEMBER COUNT WAS NEVER SWEPT.  K=5 was assumed from exp55 and never justified. The
     inference-cost argument against GR-LoRA is "5 forward passes, constant in T, versus
     their T passes" -- if K=3 keeps most of the gain that argument gets materially stronger,
     and if q32b70 (weakest member, 21 per-class wins of 200) is dead weight we should know.
     All 2^K-1 subsets are evaluated under the uniform combiner. This is free: the cones and
     the per-member fusion are already computed, subsets are just different averages.

CONTROLS -- three, all asserted under VERIFY=1
    a) `ens_ranpac` per stage == exp55's `ensemble` for the same members/order  (to 1e-9)
    b) `uniform` (all members, tau=inf) per stage == exp56's `FE|f{R}` for the same R,
       when that cell exists  (to 1e-9). This is the one that matters: it proves the fusion
       here is exp56's fusion and every combiner delta is measured against the real FE.
    c) with SYNTH=0 the cone builder calls X.BUILD['opca'] itself, so it is exp56's cone by
       construction rather than by inspection.
    If (b) is unavailable the run still proceeds but records verified.uniform_vs_exp56=false,
    so a cell that never checked cannot be mistaken for one that passed.

COST -- no training. 5 members x 1 ray budget, i.e. exp56's f32-control shape: ~35-45 min per
    cell at T=10. The combiner and subset sweeps run over cached per-member scores and are
    negligible next to the cone build.

PIN YOUR THREADS.

USAGE
    source ~/venvs/ml_env/bin/activate

    # baseline lever 2+3 at the winning budget, reproduces exp56's FE|f64 as a control
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot DS=IMAGENETR T=10 SEED=0 \
      MEMBERS=q32,m32,a16,q32b70,q64 R=64 VERIFY=1 python -u exp57_lift.py

    # lever 1: break the clip
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot DS=IMAGENETR T=10 SEED=0 \
      MEMBERS=q32,m32,a16,q32b70,q64 R=128 SYNTH=3 python -u exp57_lift.py
"""
import itertools
import json
import os
import time
import warnings
import zlib

import numpy as np
import torch

_DS = os.environ.get("DS", "IMAGENETR").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
_R = int(os.environ.get("R", 64))
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])
os.environ["ARMS"], os.environ["RULES"] = f"f{_R}", "cone"

warnings.filterwarnings("ignore", message=r"Number of distinct clusters.*",
                        category=UserWarning, module=r"sklearn\..*")

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402
import exp54_stack as S54                   # noqa: E402
import class_order as CO                    # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = S54.TAG
DSETS, TS, SEEDS, R = _DS, _TS, _SEEDS, _R
MEMBERS = os.environ.get("MEMBERS", "q32,m32,a16,q32b70,q64").split(",")
ARM = f"f{R}"
SYNTH = int(os.environ.get("SYNTH", 0))          # 0 = exp56's builder verbatim
SYNTH_POOL = float(os.environ.get("SYNTH_POOL", 3.0))   # draw SYNTH_POOL*R points total
EPOCHS, LR = int(os.environ.get("EPOCHS", 40)), float(os.environ.get("LR", 3e-4))
METHOD, GAMMA, F_MAX = S54.METHOD, S54.GAMMA, S54.F_MAX
SHRINK, M_RP, N_PASS = S54.SHRINK, S54.M_RP, S54.N_PASS
LAMBDAS, BETAS_V2 = S54.LAMBDAS, S54.BETAS_V2
VERIFY = int(os.environ.get("VERIFY", 0))
SUBSETS = int(os.environ.get("SUBSETS", 1))

# tau=inf IS uniform averaging and tau=0 is hard per-row argmax. Both endpoints are in the
# grid on purpose: the first makes the family contain FE exactly, the second makes it contain
# the routing rule we already believe is capped.
TAUS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
CONFS = os.environ.get("CONFS", "margin,negent,topz").split(",")

OUT = os.path.join(REPO, f"exp57_lift{os.environ.get('SUFFIX', '')}_{TAG}.json")
EXP55 = os.path.join(REPO, f"exp55_lora_diversity_{TAG}.json")
EXP56 = os.path.join(REPO, f"exp56_ray_ensemble_rays_{TAG}.json")

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
acc_margin, pick_beta_v2, score = S54.acc_margin, S54.pick_beta_v2, S54.score

assert MEMBERS[0] == "q32", f"member 0 must be q32 (the exp16 control). Got {MEMBERS[0]!r}."
assert set(CONFS) <= {"margin", "negent", "topz"}, f"unknown confidence in {CONFS}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}); exp49's unpinned noise floor is "
        f"0.27, larger than the +0.30 bar this file pre-registers, and it breaks the exp56 "
        f"control. Prefix with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 or set ALLOW_UNPINNED=1.")


# ------------------------------------------------------------------ features (cached only)
def member_features(ds, T, seed, spec):
    ot = CO.order_tag()
    if spec == "q32":
        f = os.path.join(
            REPO, f"exp16_feats_{ds}_T{T}_s{seed}_ep40_lr0.0003_aug1{ot}_{TAG}.npz")
        hint = (f"ORDER={CO.mode()} DATASETS={ds} SEEDS={seed} TASKS={T} python -u "
                f"exp16_full_table.py")
    else:
        f = os.path.join(
            REPO, f"exp55_feats_{ds}_T{T}_s{seed}_{spec}_ep{EPOCHS}_lr{LR:g}{ot}_{TAG}.npz")
        hint = (f"ORDER={CO.mode()} DS={ds} T={T} SEED={seed} MEMBERS=q32,{spec} python -u "
                f"exp55_lora_diversity_pilot.py")
    assert os.path.exists(f), (
        f"feature cache missing: {f}\nexp57 NEVER trains. Produce it with:\n  {hint}")
    z = np.load(f)
    return un(z["Ftr"]), un(z["Fte"])


# ------------------------------------------------------------------ lever 1: the ray ceiling
def build_cone(Xw, Fw, R_, cseed, g):
    """oPCA cone. With SYNTH=0 this IS X.BUILD['opca'] -- called, not reimplemented, so the
    exp56 control cannot drift. With SYNTH=1 the class is backfilled with Gaussian draws
    inside the oPCA subspace so k-means can emit R centroids when the class has fewer than R
    training rows.

    The backfill happens in the k-dim oPCA subspace rather than the 768-dim feature space
    because that is where k-means runs and where a covariance estimated from ~100 rows is
    actually conditioned; drawing in 768-d from a rank-deficient covariance would synthesise
    noise, not class structure."""
    if not SYNTH:
        return X.BUILD[METHOD](Xw, Fw, R_, cseed, g)
    d = Xw.shape[1]
    M = (Xw.T.astype(np.float64) @ Xw) / len(Xw) - g * X._sf(Fw, d)
    k = int(min(R_, d))
    V = np.linalg.eigh(M)[1][:, ::-1][:, :k].astype(np.float32)
    Y = np.asarray(Xw @ V, np.float64)
    want = int(SYNTH_POOL * R_)
    if len(Y) < want and len(Y) >= 2:
        rng = np.random.default_rng(90_000 + cseed)
        mu = Y.mean(0)
        S = np.cov(Y.T)
        S = np.atleast_2d(S)
        S = S + 1e-3 * (np.trace(S) / max(len(S), 1)) * np.eye(len(S))
        Y = np.vstack([Y, rng.multivariate_normal(mu, S, size=want - len(Y))])
    return un(X.BUILD["kmeans"](Y.astype(np.float32), None, R_, cseed, 0) @ V.T)


# ------------------------------------------------------------------ lever 2: per-row combiners
def confidence(Sz, seen, kind):
    """Per-row confidence of one member's fused score. PARAMETER-FREE. Returns (n_rows,)."""
    A = np.asarray(Sz[:, seen], np.float64)
    if kind == "topz":
        return A.max(1)
    if kind == "margin":
        part = np.partition(A, -2, axis=1)
        return part[:, -1] - part[:, -2]
    p = np.exp(A - A.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return (p * np.log(p + 1e-12)).sum(1)      # negative entropy


def combine(Ss, confs, tau):
    """Weighted average of member score matrices. tau=inf -> uniform (EXACTLY FE), tau=0 ->
    hard per-row argmax over members. Weights are softmax over MEMBERS of conf/tau."""
    M = len(Ss)
    if not np.isfinite(tau):
        return sum(Ss) / M
    C = np.stack(confs, 1)                                  # (n_rows, M)
    C = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-8)
    if tau <= 0:
        W = np.zeros_like(C)
        W[np.arange(len(C)), C.argmax(1)] = 1.0
    else:
        Z = C / tau
        Z -= Z.max(1, keepdims=True)
        W = np.exp(Z)
        W /= W.sum(1, keepdims=True)
    return sum(W[:, m:m + 1] * Ss[m] for m in range(M))


def run_cell(ds, T, seed, verify):
    E.T, E.SEED = T, seed
    ytr, yte, n_cls = E.get_labels(ds)
    Z = {m: member_features(ds, T, seed, m) for m in MEMBERS}
    d = Z[MEMBERS[0]][0].shape[1]
    cpt = n_cls // T
    order = CO.class_order(n_cls, seed)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm_ = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm_[:nv]]); FIT.append(ix[pm_[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    G = {m: torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64) for m in MEMBERS}
    C_ = {m: torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64) for m in MEMBERS}
    scat = {m: (np.zeros((d, d), np.float64), 0) for m in MEMBERS}
    A = {m: {} for m in MEMBERS}

    def _H(Zm, bs=4096):
        for i in range(0, len(Zm), bs):
            yield i, torch.relu(torch.as_tensor(Zm[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Zm, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Zm)]).cpu().numpy()

    # ---- references
    ref55 = ref56 = None
    if verify:
        k55 = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}"
               f"|ep{EPOCHS}_lr{LR:g}_a4{CO.order_tag()}|m{M_RP}|v1")
        if os.path.exists(EXP55):
            d55 = json.load(open(EXP55))
            ref55 = d55[k55]["ensemble"]["accs"] if k55 in d55 else None
        log(f"    VERIFY: exp55 ensemble {'loaded' if ref55 else 'NOT FOUND (skip)'}")
        if os.path.exists(EXP56) and not SYNTH:
            d56 = json.load(open(EXP56))
            for k, v in d56.items():
                p = k.split("|")
                if (p[0] == ds and int(p[1]) == T and int(p[2]) == seed
                        and p[3] == "+".join(MEMBERS) and f"FE|{ARM}|cone" in v):
                    ref56 = v[f"FE|{ARM}|cone"]["accs"]
                    break
        log(f"    VERIFY: exp56 FE|{ARM}|cone {'loaded' if ref56 else 'NOT FOUND (skip)'}"
            + ("  [SYNTH on -> control intentionally N/A]" if SYNTH else ""))

    SUBS = ([tuple(c) for k in range(1, len(MEMBERS) + 1)
             for c in itertools.combinations(MEMBERS, k)] if SUBSETS
            else [tuple(MEMBERS)])
    COMBOS = [(c, tau) for c in CONFS for tau in TAUS if np.isfinite(tau)]
    log(f"    {len(MEMBERS)} members  R={R}  SYNTH={SYNTH}  "
        f"{len(SUBS)} subsets  {len(COMBOS)} conf x tau combiners")

    res, val = {}, {}

    def put(k, a, v=None):
        res.setdefault(k, []).append(a)
        if v is not None:
            val.setdefault(k, []).append(v)

    for t in range(T):
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vix = VAL_ALL[:nval]
        yv = ytr[vix]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        col = {int(c): j for j, c in enumerate(seen)}
        tcv = np.array([col[int(v)] for v in yv])

        FUv, FUt, zLv, zLt = {}, {}, {}, {}
        for m in MEMBERS:
            Ztr_m, Zte_m = Z[m]
            sc, ns = scat[m]
            for c in tasks[t]:
                r = FIT[t][ytr[FIT[t]] == c]
                if len(r) < 2:
                    continue
                Xc = Ztr_m[r] - Ztr_m[r].mean(0)
                sc += Xc.T @ Xc; ns += len(Xc)
            scat[m] = (sc, ns)
            S_ = sc / max(ns, 1)
            S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
            Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
            Wh_inv = np.linalg.inv(Wh).astype(np.float32)

            for c in tasks[t]:
                r = FIT[t][ytr[FIT[t]] == c]
                if len(r) < 2:
                    continue
                Xw = un(Ztr_m[r] @ Wh)
                rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(ARM.encode()) % 1000)
                Fw = np.zeros((0, d), np.float32)
                if METHOD in X.DISCRIM and GAMMA > 0:
                    oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                    past = [A[m][o] for o in A[m] if o not in tasks[t]]
                    Fr = np.concatenate([Ztr_m[oth]] + past, 0)
                    if len(Fr) > F_MAX:
                        Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                    Fw = un(Fr @ Wh)
                A[m][int(c)] = build_cone(Xw, Fw, R, int(c), GAMMA) @ Wh_inv

            for i, h in _H(un(Ztr_m[FIT[t]])):
                h = h.double()
                Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
                Y[torch.arange(h.shape[0]),
                  torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
                G[m] += h.T @ h; C_[m] += h.T @ Y
            best, bw = -1.0, None
            for lam in LAMBDAS:
                Wm = torch.linalg.solve(G[m] + lam * eye, C_[m])
                a_ = acc_v1(logits(un(Ztr_m[vix]), Wm), seen, yv)
                if a_ > best:
                    best, bw = a_, Wm
            zLv[m] = zs(logits(un(Ztr_m[vix]), bw), seen)
            zLt[m] = zs(logits(un(Zte_m), bw)[tei], seen)

            Qvw, Qtw = un(Ztr_m[vix] @ Wh), un(Zte_m[tei] @ Wh)
            miss = [c for c in seen if c not in A[m]]
            Sv = np.full((nval, n_cls), -np.inf, np.float32)
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            for c in seen:
                if c not in A[m]:
                    continue
                Ac = un(A[m][c] @ Wh)
                Sv[:, c] = score("cone", Ac, Qvw)
                St[:, c] = score("cone", Ac, Qtw)
            cv_, ct_ = zs(Sv, seen), zs(St, seen)
            if miss:
                cv_[:, miss] = 0.0
                ct_[:, miss] = 0.0
            # exp56's FE inner loop, verbatim: per-member beta on that member's own base.
            b, _ = pick_beta_v2(zLv[m], cv_, seen, tcv, BETAS_V2)
            FUv[m], FUt[m] = zs(zLv[m] + b * cv_, seen), zs(zLt[m] + b * ct_, seen)

        eLt = sum(zLt[m] for m in MEMBERS) / len(MEMBERS)
        put("ens_ranpac", acc_v1(eLt, seen, yt))
        if ref55 is not None:
            assert abs(res["ens_ranpac"][-1] - ref55[t]) < 1e-9, (
                f"s{t}: ens_ranpac {res['ens_ranpac'][-1]:.10f} != exp55 {ref55[t]:.10f}")

        # ---- lever 3: member subsets, uniform combiner
        for sub in SUBS:
            nm = "uniform" if len(sub) == len(MEMBERS) else f"sub:{'+'.join(sub)}"
            St_ = sum(FUt[m] for m in sub) / len(sub)
            Sv_ = sum(FUv[m] for m in sub) / len(sub)
            put(nm, acc_v1(St_, seen, yt), acc_margin(Sv_, seen, tcv)[0])
        if ref56 is not None:
            assert abs(res["uniform"][-1] - ref56[t]) < 1e-9, (
                f"s{t}: uniform {res['uniform'][-1]:.10f} != exp56 FE|{ARM} {ref56[t]:.10f}. "
                f"The fusion here is not exp56's, so no combiner delta is interpretable.")
        if t == 0 and (ref55 is not None or ref56 is not None):
            log(f"    VERIFY ok at s0: "
                + (f"exp55 {res['ens_ranpac'][0]:.6f}  " if ref55 is not None else "")
                + (f"exp56 {res['uniform'][0]:.6f}" if ref56 is not None else ""))

        # ---- lever 2: per-row confidence weighting over ALL members
        for kind in CONFS:
            cfv = [confidence(FUv[m], seen, kind) for m in MEMBERS]
            cft = [confidence(FUt[m], seen, kind) for m in MEMBERS]
            Ssv = [FUv[m] for m in MEMBERS]
            Sst = [FUt[m] for m in MEMBERS]
            for tau in TAUS:
                if not np.isfinite(tau):
                    continue
                nm = f"{kind}@t{tau:g}"
                put(nm, acc_v1(combine(Sst, cft, tau), seen, yt),
                    acc_margin(combine(Ssv, cfv, tau), seen, tcv)[0])

        log(f"    s{t}: ens {res['ens_ranpac'][-1]*100:.2f}  uniform "
            f"{res['uniform'][-1]*100:.2f}  best-combiner "
            f"{max(res[f'{k}@t{tau:g}'][-1] for k in CONFS for tau in TAUS if np.isfinite(tau))*100:.2f}"
            f"  best-subset "
            f"{max([res[k][-1] for k in res if k.startswith('sub:')] or [0])*100:.2f}")

    del G, C_, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for k, v in res.items():
        out[k] = {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
        if k in val:
            out[f"{k}|_val"] = {"A_last": val[k][-1], "A_avg": float(np.mean(val[k]))}
    out["_meta"] = {"members": MEMBERS, "R": R, "synth": SYNTH, "synth_pool": SYNTH_POOL,
                    "order": CO.mode(), "cpt": cpt, "taus": [str(x) for x in TAUS],
                    "verified": {"exp55_ensemble": ref55 is not None,
                                 "uniform_vs_exp56": ref56 is not None}}
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|R{R}|sy{SYNTH}"
                       f"|{'+'.join(CONFS)}|{METHOD}g{GAMMA:g}|m{M_RP}{CO.order_tag()}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                t_ = time.time()
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                log(f"    cell took {time.time()-t_:.0f}s")
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 96
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if len(p) < 6 or p[3] != "+".join(MEMBERS) or p[4] != f"R{R}" or p[5] != f"sy{SYNTH}":
            continue
        cells[(p[0], int(p[1]), int(p[2]))] = v
    dts = sorted({(a, b) for a, b, _ in cells})

    def sds(ds, T):
        return sorted(s for (a, b, s) in cells if a == ds and b == T)

    def gm(ds, T, n, f):
        xs = [cells[(ds, T, s)][n][f] * 100 for s in sds(ds, T) if n in cells[(ds, T, s)]]
        return float(np.mean(xs)) if xs else float("nan")

    print("\n" + "=" * W)
    print(f"EXP57 — cached-feature levers   R={R}  SYNTH={SYNTH}  ORDER={CO.mode()}")
    print("=" * W)
    for ds, T in dts:
        u_a, u_l = gm(ds, T, "uniform", "A_avg"), gm(ds, T, "uniform", "A_last")
        uv = gm(ds, T, "uniform|_val", "A_avg")
        print(f"\n{ds} T={T}  seeds {sds(ds, T)}")
        print(f"  ens_ranpac        {gm(ds,T,'ens_ranpac','A_avg'):>7.2f} A-Avg"
              f"  {gm(ds,T,'ens_ranpac','A_last'):>7.2f} A-Last")
        print(f"  uniform (= FE)    {u_a:>7.2f}        {u_l:>7.2f}"
              f"        val-test {uv-u_a:+.2f}")

        print(f"\n  LEVER 2 — per-row confidence weighting (tau=inf IS uniform; "
              f"bar is +0.30 A-Avg)")
        print(f"    {'combiner':<18}{'A-Avg':>8}{'A-Last':>8}{'dAvg':>7}{'val-test':>10}")
        rows = []
        for kind in CONFS:
            for tau in TAUS:
                if not np.isfinite(tau):
                    continue
                n = f"{kind}@t{tau:g}"
                if n not in cells[(ds, T, sds(ds, T)[0])]:
                    continue
                rows.append((n, gm(ds, T, n, "A_avg"), gm(ds, T, n, "A_last"),
                             gm(ds, T, f"{n}|_val", "A_avg")))
        for n, a, l, v in sorted(rows, key=lambda r: -r[1])[:6]:
            print(f"    {n:<18}{a:>8.2f}{l:>8.2f}{a-u_a:>+7.2f}{v-a:>10.2f}")
        # tau=0 is hard per-row selection. Always print it, win or lose: it is the routing
        # rule GR-LoRA's interface() implements, and our own routing result says it is capped.
        for n, a, l, v in [r for r in rows if r[0].endswith("@t0")]:
            print(f"    {n+' (hard sel)':<18}{a:>8.2f}{l:>8.2f}{a-u_a:>+7.2f}{v-a:>10.2f}")
        if rows:
            bn, ba = max(rows, key=lambda r: r[3])[0], 0.0
            ba = [r for r in rows if r[0] == bn][0][1]
            print(f"    val-selected: {bn}  A-Avg {ba:.2f}  ({ba-u_a:+.2f} vs uniform)"
                  f"  -> {'LIFT' if ba-u_a >= 0.30 else 'no lift (under the +0.30 bar)'}")

        if SUBSETS:
            print(f"\n  LEVER 3 — member subsets (uniform combiner). `val-sel` is chosen on "
                  f"VAL and\n  is the only obtainable column; `oracle` is the best on TEST "
                  f"among the subsets of that K.")
            print(f"    {'K':<3}{'val-selected subset':<30}{'A-Avg':>8}{'A-Last':>8}"
                  f"{'vsK5':>7}{'oracle':>8}{'shop':>7}")
            for k in range(1, len(MEMBERS) + 1):
                # `sub:...|_val` ALSO startswith("sub:") -- excluding it is not cosmetic: the
                # val entries score higher, so they won every max() and the whole table
                # printed validation accuracy as if it were test.
                cand = [(n, gm(ds, T, n, "A_avg"), gm(ds, T, n, "A_last"),
                         gm(ds, T, f"{n}|_val", "A_avg"))
                        for n in cells[(ds, T, sds(ds, T)[0])]
                        if n.startswith("sub:") and not n.endswith("|_val")
                        and len(n[4:].split("+")) == k]
                if k == len(MEMBERS):
                    cand = [("uniform", u_a, u_l, uv)]
                if not cand:
                    continue
                b = max(cand, key=lambda r: r[3])          # chosen on VAL
                orc = max(r[1] for r in cand)              # best on TEST = oracle
                print(f"    {k:<3}{b[0].replace('sub:',''):<30}{b[1]:>8.2f}{b[2]:>8.2f}"
                      f"{b[1]-u_a:>+7.2f}{orc:>8.2f}{orc-b[1]:>7.2f}")

    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  1. `uniform` MUST equal exp56's FE|f{R} when SYNTH=0 -- it is asserted per stage. If that
     control is absent (_meta.verified.uniform_vs_exp56 false) treat every delta as unanchored.
  2. LEVER 2 BAR IS +0.30 A-Avg, pre-registered, because exp49's noise floor is 0.27. The
     tau grid contains uniform at tau=inf, so this family cannot lose on VAL -- read the
     val-test column, not the max. EF/JT/FEW all looked good on val and lost on test.
     tau=0 is hard per-row selection; if it is the worst row, that reproduces the routing
     result from a different direction and is worth recording as such.
  3. LEVER 1 needs two runs: R=64 SYNTH=0 (the exp56-anchored baseline) and R=128 SYNTH=3.
     PRE-REGISTERED: under +0.10 A-Avg for the second means the ray ceiling is real and the
     monotone-in-R curve was really "more rays ~ more training points".
  4. LEVER 3 is a cost argument, not an accuracy one. If K=3 holds ~all of K=5's A-Avg, the
     inference claim against GR-LoRA improves from 5 passes to 3 at no accuracy cost.""")
    print("=" * W)
    log(f"wrote {OUT}")
