#!/usr/bin/env python3
"""exp62_cone_diversity.py -- the cone ensembles worse than RanPAC. Why, and can it be fixed?

THE DEFICIT, LOCATED
    exp61 (IMAGENETR T=10, 3 seeds, pilot order, both anchors green):
        cone_q32   80.80 +-0.40      rp_q32      80.76 +-0.50     <- SINGLE MEMBER IS A TIE
        cone_ens   81.56 +-0.18      ens_ranpac  81.78 +-0.18
        ensemble gain:  cone +0.76   RanPAC +1.02
    The classifier is not the problem -- one cone equals one RanPAC at 1/22 the memory. The
    whole cone-only deficit is in ENSEMBLING, and it is 0.26. Against GR-LoRA's 82.09/86.20,
    cone_ens needs +0.53 A-Last and is already at parity on A-Avg (-0.04).

THE HYPOTHESIS
    All five cones come from the SAME deterministic construction -- opca, gamma=0.5, top-k
    eigenvectors, k-means at R=64, on the same class rows. Only the feature space differs.
    RanPAC's five ridge solutions live in a 10000-d random ReLU space and can differ far more
    freely. So the cone members are probably MORE CORRELATED than the RanPAC members, whose
    errcorr exp55 measured at 0.75-0.82. If so the fix is diversity in the CONSTRUCTION, not
    a better combiner -- and every attempt at a better combiner in this project has lost.

WHAT THIS FILE MEASURES

  [A] DIAGNOSTIC -- exp55's precondition block, computed on CONE scores and on RANPAC scores
      side by side at the final stage: pairwise disagree, errcorr, both_wrong, plus ORACLE and
      the headroom each ensemble is drawing from. This decides whether levers B and C are even
      aimed at the right thing:
        cone headroom << RanPAC headroom  -> members are redundant, construction diversity is
                                             the fix
        headroom similar, capture worse   -> a combiner problem, and combiners keep losing

  [B] CONSTRUCTION DIVERSITY -- FREE, because each member still builds exactly ONE cone set:
        base   every member  (opca, gamma=0.5)          reproduces exp61's cone_ens (ANCHOR)
        gam    member i gets gamma from GAMMAS          same rays algorithm, different
                                                        discriminative pressure per member
        meth   member i gets a different BUILD          opca / kmeans / dkm / spa mixed
      A worse-but-decorrelated member is useful -- that was the whole premise of m32, which is
      2.5 points below q32 alone and still earns its place.

  [C] RULE DIVERSITY -- NEARLY FREE, the rays already exist. cone_ens uses only the `cone`
      rule. exp54 measured `sub` 80.04 and `pm` 78.56 on the SAME rays, and found stacking
      them adds nothing GIVEN RANPAC. That condition has now been removed, so it is untested
      in the regime that matters.

    Every combination is a UNIFORM average of row-z-scored scores. No betas, no taus, no
    weights, nothing selected on test -- four separate attempts at fitted combination weights
    have lost in this project and none is repeated here.

ANCHORS
    ens_ranpac must reproduce exp55 and cone_ens_base must reproduce exp61's cone_ens, both
    per stage to 1e-9. If either trips, the cones are not the ones behind those numbers.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot DS=IMAGENETR T=10 SEED=0 R=64 \
      MEMBERS=q32,m32,a16,q32b70,q64 VERIFY=1 python -u exp62_cone_diversity.py
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
ARM = f"f{R}"
MEMBERS = os.environ.get("MEMBERS", "q32,m32,a16,q32b70,q64").split(",")
GAMMAS = [float(x) for x in os.environ.get("GAMMAS", "0,0.25,0.5,1.0,2.0").split(",")]
METHODS = os.environ.get("METHODS", "opca,dkm,kmeans,opca,dkm").split(",")
RULES = os.environ.get("RULES_LIST", "cone,sub,pm").split(",")
BANKS = os.environ.get("BANKS", "base,gam,meth").split(",")
EPOCHS, LR = int(os.environ.get("EPOCHS", 40)), float(os.environ.get("LR", 3e-4))
G0, F_MAX, SHRINK, M_RP = S54.GAMMA, S54.F_MAX, S54.SHRINK, S54.M_RP
LAMBDAS = S54.LAMBDAS
VERIFY = int(os.environ.get("VERIFY", 0))
OUT = os.path.join(REPO, f"exp62_cone_div{os.environ.get('SUFFIX','')}_{TAG}.json")
EXP55 = os.path.join(REPO, f"exp55_lora_diversity_{TAG}.json")
EXP61 = os.path.join(REPO, f"exp61_cone_only_{TAG}.json")

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
score, rays_for = S54.score, S54.rays_for
assert MEMBERS[0] == "q32", f"member 0 must be q32; got {MEMBERS[0]!r}"
assert len(GAMMAS) >= len(MEMBERS) and len(METHODS) >= len(MEMBERS), \
    "GAMMAS and METHODS must supply one entry per member"
assert set(METHODS[:len(MEMBERS)]) <= set(X.BUILD), f"unknown BUILD in {METHODS}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], f"threads not pinned (OMP={_th[0]} MKL={_th[1]})"


def bank_spec(bank, mi):
    """(build method, gamma) for member index mi under a given bank. Each member always
    builds exactly ONE cone set, so a bank costs the same as the baseline."""
    if bank == "base":
        return "opca", G0
    if bank == "gam":
        return "opca", GAMMAS[mi]
    if bank == "meth":
        return METHODS[mi], G0
    raise AssertionError(f"unknown bank {bank}")


def member_features(ds, T, seed, spec):
    ot = CO.order_tag()
    f = (os.path.join(REPO, f"exp16_feats_{ds}_T{T}_s{seed}_ep40_lr0.0003_aug1{ot}_{TAG}.npz")
         if spec == "q32" else
         os.path.join(REPO,
                      f"exp55_feats_{ds}_T{T}_s{seed}_{spec}_ep{EPOCHS}_lr{LR:g}{ot}_{TAG}.npz"))
    assert os.path.exists(f), f"missing feature cache {f}"
    z = np.load(f)
    return un(z["Ftr"]), un(z["Fte"])


def diversity(preds, corr, names):
    d = {}
    for i, j in itertools.combinations(names, 2):
        ci, cj = corr[i].astype(float), corr[j].astype(float)
        sd = ci.std() * cj.std()
        d[f"{i}|{j}"] = {
            "disagree": float((preds[i] != preds[j]).mean()),
            "errcorr": float(((ci - ci.mean()) * (cj - cj.mean())).mean() / sd)
            if sd > 1e-12 else float("nan")}
    anyc = np.zeros(len(corr[names[0]]), bool)
    for n in names:
        anyc |= corr[n]
    d["ORACLE"] = float(anyc.mean())
    d["best_single"] = max(float(corr[n].mean()) for n in names)
    return d


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
    A = {(b, m): {} for b in BANKS for m in MEMBERS}

    def _H(Zm, bs=4096):
        for i in range(0, len(Zm), bs):
            yield i, torch.relu(torch.as_tensor(Zm[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Zm, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Zm)]).cpu().numpy()

    ref55 = ref61 = None
    if verify:
        k55 = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}"
               f"|ep{EPOCHS}_lr{LR:g}_a4{CO.order_tag()}|m{M_RP}|v1")
        if os.path.exists(EXP55):
            j = json.load(open(EXP55)); ref55 = j[k55]["ensemble"]["accs"] if k55 in j else None
        if os.path.exists(EXP61):
            for k, v in json.load(open(EXP61)).items():
                p = k.split("|")
                if p[0] == ds and int(p[1]) == T and int(p[2]) == seed and "cone_ens" in v:
                    ref61 = v["cone_ens"]["accs"]; break
        log(f"    VERIFY exp55 {'ok' if ref55 else 'N/A'}   exp61 {'ok' if ref61 else 'N/A'}")

    names = ["ens_ranpac"] + [f"cone_ens_{b}" for b in BANKS]
    for b in BANKS:
        for k in range(2, len(RULES) + 1):
            names.append(f"rules_{b}_" + "+".join(RULES[:k]))
    res = {n: [] for n in names}
    diag = {}

    for t in range(T):
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vix = VAL_ALL[:nval]
        yv = ytr[vix]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        zLt, Sc = {}, {}                      # Sc[(bank, member, rule)] -> z-scored test score
        for mi, m in enumerate(MEMBERS):
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

            for b in BANKS:
                meth, gam = bank_spec(b, mi)
                for c in tasks[t]:
                    r = FIT[t][ytr[FIT[t]] == c]
                    if len(r) < 2:
                        continue
                    Xw = un(Ztr_m[r] @ Wh)
                    rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(ARM.encode()) % 1000)
                    Fw = np.zeros((0, d), np.float32)
                    if meth in X.DISCRIM and gam > 0:
                        oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                        past = [A[(b, m)][o] for o in A[(b, m)] if o not in tasks[t]]
                        Fr = np.concatenate([Ztr_m[oth]] + past, 0)
                        if len(Fr) > F_MAX:
                            Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                        Fw = un(Fr @ Wh)
                    A[(b, m)][int(c)] = X.BUILD[meth](Xw, Fw, rays_for(ARM, len(r)),
                                                      int(c), gam) @ Wh_inv

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
            zLt[m] = zs(logits(un(Zte_m), bw)[tei], seen)

            Qtw = un(Zte_m[tei] @ Wh)
            for b in BANKS:
                miss = [c for c in seen if c not in A[(b, m)]]
                for rule in RULES:
                    St = np.full((len(tei), n_cls), -np.inf, np.float32)
                    for c in seen:
                        if c not in A[(b, m)]:
                            continue
                        St[:, c] = score(rule, un(A[(b, m)][c] @ Wh), Qtw)
                    z = zs(St, seen)
                    if miss:
                        z[:, miss] = 0.0
                    Sc[(b, m, rule)] = z

        M = len(MEMBERS)
        res["ens_ranpac"].append(acc_v1(sum(zLt.values()) / M, seen, yt))
        for b in BANKS:
            res[f"cone_ens_{b}"].append(
                acc_v1(sum(Sc[(b, m, "cone")] for m in MEMBERS) / M, seen, yt))
            for k in range(2, len(RULES) + 1):
                rs = RULES[:k]
                tot = sum(Sc[(b, m, r)] for m in MEMBERS for r in rs) / (M * k)
                res[f"rules_{b}_" + "+".join(rs)].append(acc_v1(tot, seen, yt))

        if ref55 is not None:
            assert abs(res["ens_ranpac"][-1] - ref55[t]) < 1e-9, \
                f"s{t}: ens_ranpac != exp55"
        if ref61 is not None and "base" in BANKS:
            assert abs(res["cone_ens_base"][-1] - ref61[t]) < 1e-9, \
                f"s{t}: cone_ens_base {res['cone_ens_base'][-1]:.10f} != exp61 {ref61[t]:.10f}"

        if t == T - 1:                     # [A] diagnostic at the final stage
            for lbl, get in (("cone", lambda m: Sc[("base", m, "cone")]),
                             ("ranpac", lambda m: zLt[m])):
                pr = {m: np.asarray(seen)[get(m)[:, seen].argmax(1)] for m in MEMBERS}
                co = {m: (pr[m] == yt) for m in MEMBERS}
                diag[lbl] = diversity(pr, co, MEMBERS)

        log(f"    s{t}: ensRP {res['ens_ranpac'][-1]*100:.2f}  " +
            "  ".join(f"{b} {res[f'cone_ens_{b}'][-1]*100:.2f}" for b in BANKS))

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()
    out = {k: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for k, v in res.items()}
    out["_diag"] = diag
    out["_meta"] = {"members": MEMBERS, "R": R, "banks": BANKS, "rules": RULES,
                    "gammas": GAMMAS[:len(MEMBERS)], "methods": METHODS[:len(MEMBERS)],
                    "order": CO.mode(),
                    "verified": {"exp55": ref55 is not None, "exp61": ref61 is not None}}
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|R{R}|{'+'.join(BANKS)}"
                       f"|{'+'.join(RULES)}{CO.order_tag()}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                t_ = time.time()
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                log(f"    cell {time.time()-t_:.0f}s")
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 96
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if p[3] == "+".join(MEMBERS) and p[4] == f"R{R}":
            cells[(p[0], int(p[1]), int(p[2]))] = v
    REF = {("IMAGENETR", 10): (82.09, 86.20), ("IMAGENETR", 20): (80.23, 85.05),
           ("CUB200P", 10): (89.91, 93.85), ("IMAGENETAP", 10): (63.60, 70.24)}
    print("\n" + "=" * W)
    print(f"EXP62 -- cone ensemble diversity   R={R}  ORDER={CO.mode()}")
    print("=" * W)
    for ds, T in sorted({(a, b) for a, b, _ in cells}):
        ss = sorted(s for a, b, s in cells if a == ds and b == T)
        c0 = cells[(ds, T, ss[0])]
        mt = c0["_meta"]
        def g(n, f):
            return np.array([cells[(ds, T, s)][n][f] * 100 for s in ss if n in cells[(ds, T, s)]])
        ref = REF.get((ds, T), (float("nan"),) * 2)
        print(f"\n  {ds} T={T} seeds {ss}  anchors {mt['verified']}")
        print(f"  gammas {mt['gammas']}   methods {mt['methods']}")

        print(f"\n  [A] DIVERSITY AT THE FINAL STAGE")
        print(f"      {'source':<9}{'mean disagree':>15}{'mean errcorr':>14}"
              f"{'best single':>13}{'ORACLE':>9}{'headroom':>10}")
        for lbl in ("cone", "ranpac"):
            D = [cells[(ds, T, s)]["_diag"][lbl] for s in ss]
            prs = [k for k in D[0] if "|" in k]
            dis = np.mean([[x[p]["disagree"] for p in prs] for x in D]) * 100
            ec = np.mean([[x[p]["errcorr"] for p in prs] for x in D])
            bs = np.mean([x["best_single"] for x in D]) * 100
            orc = np.mean([x["ORACLE"] for x in D]) * 100
            print(f"      {lbl:<9}{dis:>14.1f}%{ec:>14.2f}{bs:>13.2f}{orc:>9.2f}"
                  f"{orc-bs:>10.2f}")
        print("      -> cone headroom much smaller than RanPAC's means the cone members are\n"
              "         REDUNDANT and construction diversity is the right lever.")

        print(f"\n  [B]/[C] ENSEMBLE ARMS")
        print(f"      {'arm':<26}{'A-Last':>16}{'A-Avg':>16}{'vs base':>9}{'vs GR-L':>9}")
        base_l = g("cone_ens_base", "A_last").mean()
        allarms = [n for n in c0 if not n.startswith("_")]
        for n in sorted(allarms, key=lambda x: -g(x, "A_last").mean()):
            l, a = g(n, "A_last"), g(n, "A_avg")
            sl = l.std(ddof=1) if len(l) > 1 else float("nan")
            sa = a.std(ddof=1) if len(a) > 1 else float("nan")
            print(f"      {n:<26}{l.mean():>9.2f}±{sl:<5.2f}{a.mean():>9.2f}±{sa:<5.2f}"
                  f"{l.mean()-base_l:>+9.2f}{l.mean()-ref[0]:>+9.2f}")
    print("\n" + "-" * W)
    print("""HOW TO READ THIS
  1. Anchors first. cone_ens_base must equal exp61's cone_ens and ens_ranpac must equal
     exp55's ensemble, both to 1e-9 per stage.
  2. [A] is the diagnosis. exp55 measured the RanPAC members at errcorr 0.75-0.82 with 5.87
     points of oracle headroom. If the cone members sit much higher on errcorr and much lower
     on headroom, they are redundant and [B] is aimed correctly.
  3. Every arm is a UNIFORM average. No weights are fitted anywhere -- EF, JT, FEW and per-row
     tau weighting have all lost in this project, and the target here is +0.53 A-Last, which
     is smaller than the overfit those combiners produced.
  4. cone_ens needs +0.53 A-Last to reach GR-LoRA and is already at parity on A-Avg (-0.04).
     Judge the arms against that, not against FE.""")
    print("=" * W)
