#!/usr/bin/env python3
"""exp61_cone_only.py -- how good is the method with RanPAC DELETED?

THE QUESTION AND WHY IT IS WORTH ONE RUN
    The 10000x10000 RanPAC Gram is the single worst number in this project's cost analysis:
    4.3 GB of persistent state across five members, 6.5x GR-LoRA's 0.66 GB. The cone needs
    197 MB. exp54 measured the two read-outs as near-equals standing alone on IMAGENETR --
    cone 79.92/84.87 against RanPAC 80.41/85.31 -- and ~95% additive when fused. So the
    trade is priced at roughly -1.6 A-Last for -4 GB on a SINGLE member.

    Nobody has measured it on the ENSEMBLE, which is what the method actually is. If the
    +1.44 ensemble gain transfers to cone-only, it lands near 81.4 and loses to GR-LoRA's
    82.09; if the cone ensembles better than RanPAC does, it could land higher. Extrapolating
    is not good enough to decide whether to drop a component, so this measures it.

FIVE ARMS ON IDENTICAL CONES AND IDENTICAL FEATURES, SO THE DELTAS ARE CLEAN
    rp_q32       single member, RanPAC only                  -- exp54's `ranpac` analogue
    cone_q32     single member, cone only                     -- exp54's `cone` analogue
    ens_ranpac   5 members, mean of z-scored RanPAC logits    -- ANCHOR: must equal exp55
    cone_ens     5 members, mean of z-scored cone scores      -- THE ARM THIS FILE EXISTS FOR
    FE           5 members, per-member fusion then average    -- ANCHOR: must equal exp56

    Two of the five are exact reproduction anchors. If either fails, the cones or the features
    are not the ones that produced the published numbers and no delta here is interpretable.

WHAT WOULD MAKE CONE-ONLY WORTH SHIPPING
    cone_ens within ~0.3 of FE would make the memory win (4.3 GB -> 0.3 GB, from 6.5x WORSE
    than GR-LoRA to 2x BETTER) close to free. cone_ens near the extrapolated 81.4 means the
    cone is a co-equal partner to RanPAC rather than a replacement, and the method keeps both.
    Either way this is a decision, not a fishing trip: no beta, no tau, no config search, and
    nothing here is selected on test.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot DS=IMAGENETR T=10 SEED=0,1,2 R=64 \
      MEMBERS=q32,m32,a16,q32b70,q64 VERIFY=1 python -u exp61_cone_only.py
"""
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
ARM = f"f{R}"
MEMBERS = os.environ.get("MEMBERS", "q32,m32,a16,q32b70,q64").split(",")
EPOCHS, LR = int(os.environ.get("EPOCHS", 40)), float(os.environ.get("LR", 3e-4))
METHOD, GAMMA, F_MAX = S54.METHOD, S54.GAMMA, S54.F_MAX
SHRINK, M_RP = S54.SHRINK, S54.M_RP
LAMBDAS, BETAS_V2 = S54.LAMBDAS, S54.BETAS_V2
VERIFY = int(os.environ.get("VERIFY", 0))
OUT = os.path.join(REPO, f"exp61_cone_only{os.environ.get('SUFFIX','')}_{TAG}.json")
EXP55 = os.path.join(REPO, f"exp55_lora_diversity_{TAG}.json")
EXP56 = os.path.join(REPO, f"exp56_ray_ensemble_table_{TAG}.json")

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
acc_margin, pick_beta_v2, score, rays_for = (S54.acc_margin, S54.pick_beta_v2,
                                             S54.score, S54.rays_for)
assert MEMBERS[0] == "q32", f"member 0 must be q32; got {MEMBERS[0]!r}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], f"threads not pinned (OMP={_th[0]} MKL={_th[1]})"


def member_features(ds, T, seed, spec):
    ot = CO.order_tag()
    if spec == "q32":
        f = os.path.join(REPO,
                         f"exp16_feats_{ds}_T{T}_s{seed}_ep40_lr0.0003_aug1{ot}_{TAG}.npz")
    else:
        f = os.path.join(
            REPO, f"exp55_feats_{ds}_T{T}_s{seed}_{spec}_ep{EPOCHS}_lr{LR:g}{ot}_{TAG}.npz")
    assert os.path.exists(f), f"missing feature cache {f}"
    z = np.load(f)
    return un(z["Ftr"]), un(z["Fte"])


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
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    G = {m: torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64) for m in MEMBERS}
    C = {m: torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64) for m in MEMBERS}
    scat = {m: (np.zeros((d, d), np.float64), 0) for m in MEMBERS}
    A = {m: {} for m in MEMBERS}

    def _H(Zm, bs=4096):
        for i in range(0, len(Zm), bs):
            yield i, torch.relu(torch.as_tensor(Zm[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Zm, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Zm)]).cpu().numpy()

    ref55 = ref56 = None
    if verify:
        k55 = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}"
               f"|ep{EPOCHS}_lr{LR:g}_a4{CO.order_tag()}|m{M_RP}|v1")
        if os.path.exists(EXP55):
            d55 = json.load(open(EXP55))
            ref55 = d55[k55]["ensemble"]["accs"] if k55 in d55 else None
        if os.path.exists(EXP56):
            for k, v in json.load(open(EXP56)).items():
                p = k.split("|")
                if (p[0] == ds and int(p[1]) == T and int(p[2]) == seed
                        and p[3] == "+".join(MEMBERS) and f"FE|{ARM}|cone" in v):
                    ref56 = v[f"FE|{ARM}|cone"]["accs"]; break
        log(f"    VERIFY exp55 {'ok' if ref55 else 'N/A'}   exp56 {'ok' if ref56 else 'N/A'}")

    res = {k: [] for k in ("rp_q32", "cone_q32", "ens_ranpac", "cone_ens", "FE")}
    for t in range(T):
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vix = VAL_ALL[:nval]
        yv = ytr[vix]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        col = {int(c): j for j, c in enumerate(seen)}
        tcv = np.array([col[int(v)] for v in yv])

        zLt, zCt, FUt = {}, {}, {}
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
                A[m][int(c)] = X.BUILD[METHOD](Xw, Fw, rays_for(ARM, len(r)),
                                               int(c), GAMMA) @ Wh_inv
            for i, h in _H(un(Ztr_m[FIT[t]])):
                h = h.double()
                Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
                Y[torch.arange(h.shape[0]),
                  torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
                G[m] += h.T @ h; C[m] += h.T @ Y
            best, bw = -1.0, None
            for lam in LAMBDAS:
                Wm = torch.linalg.solve(G[m] + lam * eye, C[m])
                a_ = acc_v1(logits(un(Ztr_m[vix]), Wm), seen, yv)
                if a_ > best:
                    best, bw = a_, Wm
            zLv = zs(logits(un(Ztr_m[vix]), bw), seen)
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
            zCt[m] = ct_
            b, _ = pick_beta_v2(zLv, cv_, seen, tcv, BETAS_V2)
            FUt[m] = zs(zLt[m] + b * ct_, seen)

        M = len(MEMBERS)
        res["rp_q32"].append(acc_v1(zLt["q32"], seen, yt))
        res["cone_q32"].append(acc_v1(zCt["q32"], seen, yt))
        res["ens_ranpac"].append(acc_v1(sum(zLt.values()) / M, seen, yt))
        res["cone_ens"].append(acc_v1(sum(zCt.values()) / M, seen, yt))
        res["FE"].append(acc_v1(sum(FUt.values()) / M, seen, yt))

        if ref55 is not None:
            assert abs(res["ens_ranpac"][-1] - ref55[t]) < 1e-9, (
                f"s{t}: ens_ranpac {res['ens_ranpac'][-1]:.10f} != exp55 {ref55[t]:.10f}")
        if ref56 is not None:
            assert abs(res["FE"][-1] - ref56[t]) < 1e-9, (
                f"s{t}: FE {res['FE'][-1]:.10f} != exp56 {ref56[t]:.10f}")
        log(f"    s{t}: rp1 {res['rp_q32'][-1]*100:.2f}  cone1 {res['cone_q32'][-1]*100:.2f}"
            f"  ensRP {res['ens_ranpac'][-1]*100:.2f}  coneENS {res['cone_ens'][-1]*100:.2f}"
            f"  FE {res['FE'][-1]*100:.2f}")

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()
    out = {k: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for k, v in res.items()}
    out["_meta"] = {"members": MEMBERS, "R": R, "order": CO.mode(),
                    "verified": {"exp55": ref55 is not None, "exp56": ref56 is not None}}
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|R{R}"
                       f"|{METHOD}g{GAMMA:g}|m{M_RP}{CO.order_tag()}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                t_ = time.time()
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                log(f"    cell {time.time()-t_:.0f}s")
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 92
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if p[3] == "+".join(MEMBERS) and p[4] == f"R{R}":
            cells[(p[0], int(p[1]), int(p[2]))] = v
    REF = {("IMAGENETR", 10): (82.09, 86.20), ("IMAGENETR", 20): (80.23, 85.05),
           ("CUB200P", 10): (89.91, 93.85), ("IMAGENETAP", 10): (63.60, 70.24),
           ("CIFAR100", 10): (91.97, 94.65)}
    print("\n" + "=" * W)
    print(f"EXP61 -- what does deleting RanPAC cost?   R={R}  ORDER={CO.mode()}")
    print("=" * W)
    for ds, T in sorted({(a, b) for a, b, _ in cells}):
        ss = sorted(s for a, b, s in cells if a == ds and b == T)
        def g(n, f):
            return np.array([cells[(ds, T, s)][n][f] * 100 for s in ss])
        ref = REF.get((ds, T), (float("nan"),) * 2)
        print(f"\n  {ds} T={T}  seeds {ss}   "
              f"anchors {cells[(ds,T,ss[0])]['_meta']['verified']}")
        print(f"    {'arm':<14}{'A-Last':>16}{'A-Avg':>16}{'vs GR-L':>9}{'vs GR-A':>9}"
              f"{'memory':>10}")
        mem = {"rp_q32": "0.9 GB", "cone_q32": "40 MB", "ens_ranpac": "4.3 GB",
               "cone_ens": "0.3 GB", "FE": "4.3 GB"}
        for n in ("rp_q32", "cone_q32", "ens_ranpac", "cone_ens", "FE"):
            l, a = g(n, "A_last"), g(n, "A_avg")
            sl = l.std(ddof=1) if len(l) > 1 else float("nan")
            sa = a.std(ddof=1) if len(a) > 1 else float("nan")
            print(f"    {n:<14}{l.mean():>9.2f}±{sl:<5.2f}{a.mean():>9.2f}±{sa:<5.2f}"
                  f"{l.mean()-ref[0]:>+9.2f}{a.mean()-ref[1]:>+9.2f}{mem[n]:>10}")
        ce, fe = g("cone_ens", "A_last").mean(), g("FE", "A_last").mean()
        cea, fea = g("cone_ens", "A_avg").mean(), g("FE", "A_avg").mean()
        print(f"    cone_ens - FE = {ce-fe:+.2f} A-Last / {cea-fea:+.2f} A-Avg"
              f"   for 4.3 GB -> 0.3 GB")
        print(f"    ensemble gain: RanPAC {g('ens_ranpac','A_last').mean()-g('rp_q32','A_last').mean():+.2f}"
              f"   cone {ce-g('cone_q32','A_last').mean():+.2f}   (A-Last)")
    print("\n" + "-" * W)
    print("""HOW TO READ THIS
  1. The two anchors must both be `True`. ens_ranpac reproduces exp55 and FE reproduces
     exp56, both to 1e-9, or the cones/features are not the published ones.
  2. cone_ens - FE is the price of deleting RanPAC. Within ~0.3 makes the memory win
     (6.5x WORSE than GR-LoRA -> 2x BETTER) close to free.
  3. Compare the two ensemble gains. If the cone ensembles BETTER than RanPAC does, the
     cone-only ensemble closes more of the single-member gap than exp54's -0.49 suggests,
     and cone-native becomes viable on accuracy rather than only on memory.""")
    print("=" * W)
