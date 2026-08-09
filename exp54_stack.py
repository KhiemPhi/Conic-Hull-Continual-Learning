#!/usr/bin/env python3
"""exp54_stack.py -- joint multi-rule fusion and a repaired beta search.

WHY THIS FILE EXISTS
    exp52 measured where our remaining deficit to the published PTM-CIL numbers lives, and
    it is NOT the conic rule. Per dataset, A-Avg:

        ds           our RanPAC   GR-LoRA   base deficit   what the fusion adds
        CIFAR100        95.13      94.65        +0.48            +0.05
        IMAGENETR       85.31      86.20        -0.89            +0.63
        IMAGENETAP      68.66      70.24        -1.58            +0.63
        CUB200P         93.11      93.85        -0.74            +0.03

    On every losing dataset the BASE classifier is behind by more than the read-out can
    recover. Read-out work therefore cannot win this outright -- and this file does not
    claim it will. What it can do is bounded and was measured: per-stage fuse_cone - ranpac
    is +0.36 early against +0.90 late on IMAGENETR, so making the fusion work as well early
    as it already does late is worth +0.27 A-Avg there, against a -0.26 gap. IMAGENETR is
    the ONE cell a better read-out can flip. CUB200P abstains on 63% of stages and gains
    +0.03 total; no fusion fix touches it. Expectations are set accordingly.

THE TWO DEFECTS THIS REPAIRS, both visible in exp52's own logs

  1. TIES RESOLVE TO "DON'T USE THE CONE".
        beta = max(BETAS, key=lambda b: acc(zL + b*zS))
     Python's max returns the FIRST maximal element and BETAS[0] == 0.0. Val accuracy is a
     0/1 statistic on a small split -- at stage 0 the val set is 10% of ONE task -- so many
     betas tie, and every tie silently becomes "fusion off". The signature is unmissable:
     per-stage fuse_cone - ranpac is EXACTLY +0.00 at s0 on all four datasets, and CUB200P
     seed 0 chose beta=0 at all ten stages, i.e. its "fused" arm was literally RanPAC.
     FIX: select on (accuracy, mean margin) lexicographically. Margin is continuous in beta,
     so it breaks ties on evidence instead of on list order, and it never prefers beta=0
     merely because 0 comes first.

  2. THE GRID CANNOT EXPRESS THE ANSWER IT KEEPS REACHING FOR.
     BETAS ends [..., 3, 5, 10, 100]. IMAGENETA hit 10.0, and the only larger option is 100.
     FIX: fill the 5-100 decade.

THE STACK, which is the actual proposal
    exp52's decomposition on IMAGENETR (A-Last, vs RanPAC): rays alone via `pm` +0.30, plus
    the linear combination via `sub` +1.04, plus non-negativity via `cone` +1.12. Those three
    rules read the SAME rays through different lenses and are only partly redundant -- and
    the whole finding of exp52 was that the gain is ensemble decorrelation, not conic
    structure. So stop picking one lens and fuse them jointly:

        S = z(ranpac) + sum_k beta_k * z(S_k)      k over (arm, rule)

    with beta_k by coordinate ascent on val. `pm` is the interesting component precisely
    because it is the WORST rule standalone (76-86) and therefore the most decorrelated
    from the other two.

    ATTRIBUTION LADDER -- every step isolates one change, and the first must reproduce exp52:
        fuse_{rule}      exp52 protocol, verbatim, original grid, first-max tie-break
        fuse2_{rule}     same single rule, repaired beta search          <- isolates defect 1+2
        stack_{arm}      rules jointly within one ray budget             <- isolates multi-rule
        stack_all        all rules across all ray budgets                <- adds multi-scale

THE RISK, stated before the result: THIS CAN OVERFIT VAL AND IT IS DESIGNED TO SHOW THAT
    `stack_all` fits 6 weights on a val split that is 10% of the classes seen so far. At
    stage 0 on IMAGENETAP that is ~40 rows. Six parameters on forty rows will overfit, the
    val number will look excellent and the test number will not follow. So every stack arm
    reports its VAL accuracy alongside its TEST accuracy and the summary prints the
    val-minus-test gap per arm. If `stack_all` beats `stack_f32` on val and loses on test,
    that is the diagnosis, the component count is the cause, and the answer is fewer
    components -- not a better search. Read the gap column BEFORE the accuracy column.

    Coordinate ascent is deliberately weak (N_PASS=2, a fixed grid, no interactions). A
    stronger optimiser here would fit the val split harder, which is the opposite of what
    this needs.

WHAT WOULD FALSIFY THE WHOLE IDEA
    If `stack_f32 - fuse_cone` is <= 0 on IMAGENETR A-Avg, the three rules are redundant
    given RanPAC, the "it's ensembling" reading of exp52 does not extend to stacking, and
    Tier 1 is done -- the remaining gap is features and nothing in the read-out will move it.

REPRODUCTION IS ASSERTED, NOT ARGUED
    VERIFY=1 checks the in-cell RanPAC against the exp16 bar AND `f32|fuse_cone` against
    exp52's cached cell to 1e-9. The foreign-negative RNG is keyed on the arm name via
    zlib.crc32, so the arm names here are f32/f64 exactly as in exp52 -- renaming them would
    silently draw a different foreign subsample and nothing would match.

PIN YOUR THREADS. exp49 measured the unpinned noise floor at 0.27, larger than anything
    here, and it would also break the exp52 repro assert.

USAGE
    source ~/venvs/ml_env/bin/activate

    # smoke -- NOTE T must stay 10: exp16 feature caches are keyed by T (task 0's class set
    # depends on it), so T=2 would need its own exp16 run before anything here can load.
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETAP T=10 SEED=0 ARMS=f32 RULES=cone,sub SUFFIX=_smoke \
      python -u exp54_stack.py

    # CONTROL FIRST -- asserts repro against exp52 and the exp16 bar
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0 VERIFY=1 python -u exp54_stack.py

    # the run that decides Tier 1
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CIFAR100,IMAGENETR,IMAGENETAP,CUB200P T=10 SEED=0,1,2 python -u exp54_stack.py

    Resumable: cells are written as they finish and existing keys are skipped.
    Run SEQUENTIALLY; concurrency on this box is measured strictly worse.
"""
import json
import os
import time
import zlib

import numpy as np
import torch

_DS = os.environ.get("DS", "CIFAR100,IMAGENETR,IMAGENETAP,CUB200P").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0,1,2").split(",")]
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DSETS, TS, SEEDS = _DS, _TS, _SEEDS
ARMS = os.environ.get("ARMS", "f32,f64").split(",")
RULES = os.environ.get("RULES", "cone,sub,pm").split(",")
METHOD = os.environ.get("METHOD", "opca")
GAMMA = float(os.environ.get("GAMMA", 0.5))
RMAX = int(os.environ.get("RMAX", 128))
RMIN_DEF = int(os.environ.get("RMIN", 8))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
M_RP = int(os.environ.get("MRP", 10000))
ITERS = int(os.environ.get("ITERS", 500))
N_PASS = int(os.environ.get("N_PASS", 2))
VERIFY = int(os.environ.get("VERIFY", 0))
LAMBDAS = [1e2, 1e3, 1e4]

# exp52's grid, VERBATIM. The v1 control arms must use this and nothing else.
BETAS_V1 = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
# Repaired grid: the 5-100 decade filled in. IMAGENETA reached 10.0 under V1 with nothing
# between it and 100, so the search was choosing the edge of the grid, not an optimum.
BETAS_V2 = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
            7.5, 10.0, 15.0, 25.0, 40.0, 65.0, 100.0]

OUT = os.path.join(REPO, f"exp54_stack{os.environ.get('SUFFIX', '')}_{TAG}.json")
EXP52 = os.path.join(REPO, f"exp52_fusion_rule_control_{TAG}.json")
BAR = json.load(open(os.path.join(REPO, f"exp16_full_table_{TAG}.json")))

assert set(RULES) <= {"cone", "sub", "pm"}, f"unknown rule in {RULES}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}). exp49 measured the unpinned noise "
        f"floor at 0.27, larger than every effect here, and it breaks the exp52 repro "
        f"assert. Prefix with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 or set ALLOW_UNPINNED=1.")

un = X.un


def rays_for(arm, n):
    if arm.startswith("f"):
        return int(arm[1:])
    k, _, m = arm[1:].partition("m")
    return int(np.clip(n / float(k), int(m) if m else RMIN_DEF, RMAX))


def bar_for(ds, T, seed):
    v = BAR.get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
    assert v is not None, f"no exp16 bar for {ds} T={T} s={seed}"
    return v


def zs(A, seen):
    """Row-wise z-score over SEEN columns, -inf -> row min. Verbatim from exp35/52/53."""
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def score(rule, Ac, Q):
    if rule == "cone":
        return X.cone_score(Ac, Q, iters=ITERS)
    if rule == "pm":
        return (Q @ Ac.T).max(1)
    U, s, _ = np.linalg.svd(Ac.T, full_matrices=False)
    B = U[:, s > max(s[0], 1e-12) * 1e-6]
    return np.linalg.norm(Q @ B, axis=1)


def acc_v1(S, seen, y):
    """exp52's accuracy, character for character. Used ONLY by the v1 control arms, because
    argmax tie-handling is part of what has to reproduce."""
    return float((np.asarray(seen)[S[:, seen].argmax(1)] == y).mean())


def acc_margin(S, seen, tcol):
    """(accuracy, mean SCALE-FREE margin) over the seen columns. `tcol` is the true class's
    POSITION in `seen`, precomputed once per stage -- it does not depend on beta and
    rebuilding it inside the search was measurably the slowest part of the coordinate ascent.

    Margin = true score minus best rival score. It is continuous in beta, which is the whole
    point: accuracy alone ties constantly on a small val split and every tie in exp52 fell to
    beta=0 by list order.

    THE ROW RE-Z-SCORE IS NOT COSMETIC -- WITHOUT IT THIS CRITERION IS WORSE THAN THE BUG.
    The combined score zL + b*zS grows in MAGNITUDE with b, so a raw margin is monotonically
    increasing in b whenever the component agrees with the label on most rows. The tie-break
    then always runs to the top of the grid. Measured on the IMAGENETAP smoke: the stack
    chose b=100 (the grid maximum) for BOTH components at stage 0 and lost 0.96 on test,
    75.00 -> 74.04, and A-Avg came in 0.17 BELOW exp52's fuse_cone. exp52's tie-to-zero was
    accidentally acting as a regulariser and a naive margin is strictly worse than it.
    Re-z-scoring each row makes the margin scale-free: accuracy is untouched (argmax is
    invariant to a per-row affine map) so the primary criterion is unchanged, and the
    tie-break now measures separation rather than amplitude."""
    Ss = np.asarray(S[:, seen], np.float64)
    Ss = (Ss - Ss.mean(1, keepdims=True)) / (Ss.std(1, keepdims=True) + 1e-8)
    r = np.arange(len(tcol))
    true = Ss[r, tcol]
    tmp = Ss.copy()
    tmp[r, tcol] = -np.inf
    other = tmp.max(1)
    return float((true > other).mean()), float((true - other).mean())


def pick_beta_v2(base_v, zS_v, seen, tcol, betas):
    """argmax over betas of (val accuracy, val margin), lexicographic. Returns the beta and
    the resulting val accuracy."""
    best, bb = (-1.0, -np.inf), betas[0]
    for b in betas:
        key = acc_margin(base_v + b * zS_v, seen, tcol)
        if key > best:
            best, bb = key, b
    return bb, best[0]


def run_cell(ds, T, seed, verify):
    E.T, E.SEED = T, seed
    F_ = E.adapted_features(ds)
    assert F_ is not None, f"no exp16 feature cache for {ds} T={T} s={seed}"
    Ztr, Zte = F_
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm_ = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm_[:nv]]); FIT.append(ix[pm_[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A = {a: {} for a in ARMS}
    COMPS = [(a, r) for a in ARMS for r in RULES]
    STACKS = [f"stack_{a}" for a in ARMS] + (["stack_all"] if len(ARMS) > 1 else [])

    res = {"ranpac": []}
    for a, r in COMPS:
        res[f"{a}|{r}"] = []
        res[f"{a}|fuse_{r}"] = []
        res[f"{a}|fuse2_{r}"] = []
    for s_ in STACKS:
        res[s_] = []
    valacc = {s_: [] for s_ in STACKS}
    wlog = {s_: [] for s_ in STACKS}

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

        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = un(Ztr[r] @ Wh)
            for a in ARMS:
                # Arm names are f32/f64 exactly as in exp52: this crc32 keys the foreign
                # subsample, so a renamed arm draws different negatives and the repro assert
                # below could never pass.
                rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(a.encode()) % 1000)
                Fw = np.zeros((0, d), np.float32)
                if METHOD in X.DISCRIM and GAMMA > 0:
                    oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                    past = [A[a][o] for o in A[a] if o not in tasks[t]]
                    Fr = np.concatenate([Ztr[oth]] + past, 0)
                    if len(Fr) > F_MAX:
                        Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                    Fw = un(Fr @ Wh)
                A[a][int(c)] = X.BUILD[METHOD](Xw, Fw, rays_for(a, len(r)),
                                               int(c), GAMMA) @ Wh_inv

        for i, h in _H(un(Ztr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y

        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vix = VAL_ALL[:nval]
        yv, tei = ytr[vix], np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        col = {int(c): j for j, c in enumerate(seen)}
        tcv = np.array([col[int(v)] for v in yv])
        # Test rows whose class has no cone keep -inf and are still scored (they are simply
        # wrong); but the margin helper must not index a missing column, so the TEST margin
        # is only ever used for logging, never for selection. Selection is val-only.
        tct = np.array([col[int(v)] for v in yt])

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            if acc_v1(logits(un(Ztr[vix]), Wm), seen, yv) > best:
                best = acc_v1(logits(un(Ztr[vix]), Wm), seen, yv); bw = Wm
        Lv, Lt = logits(un(Ztr[vix]), bw), logits(un(Zte), bw)[tei]
        res["ranpac"].append(acc_v1(zs(Lt, seen), seen, yt))
        zLv, zLt = zs(Lv, seen), zs(Lt, seen)

        Qvw, Qtw = un(Ztr[vix] @ Wh), un(Zte[tei] @ Wh)
        zSv, zSt = {}, {}
        for a in ARMS:
            miss = [c for c in seen if c not in A[a]]
            for r_ in RULES:
                Sv = np.full((nval, n_cls), -np.inf, np.float32)
                St = np.full((len(tei), n_cls), -np.inf, np.float32)
                for c in seen:
                    if c not in A[a]:
                        continue
                    Ac = un(A[a][c] @ Wh)
                    Sv[:, c] = score(r_, Ac, Qvw)
                    St[:, c] = score(r_, Ac, Qtw)
                res[f"{a}|{r_}"].append(acc_v1(zs(St, seen), seen, yt))
                zv, zt = zs(Sv, seen), zs(St, seen)
                if miss:
                    # Classes with <2 fit rows have no cone. zs maps their -inf to the row
                    # MINIMUM, which in the fused score actively suppresses classes RanPAC
                    # would have called right. Neutralise to 0 (the z-mean) so they fall back
                    # to RanPAC alone. Raw arms above keep -inf, which is honest.
                    zv[:, miss] = 0.0
                    zt[:, miss] = 0.0
                zSv[(a, r_)], zSt[(a, r_)] = zv, zt

                # v1: exp52 verbatim -- original grid, plain max, first-max tie-break.
                b1 = max(BETAS_V1, key=lambda bb: acc_v1(zLv + bb * zv, seen, yv))
                res[f"{a}|fuse_{r_}"].append(acc_v1(zLt + b1 * zt, seen, yt))
                # v2: repaired grid + (accuracy, margin) tie-break.
                b2, _ = pick_beta_v2(zLv, zv, seen, tcv, BETAS_V2)
                res[f"{a}|fuse2_{r_}"].append(acc_v1(zLt + b2 * zt, seen, yt))

        for s_ in STACKS:
            comps = ([(a, r) for a in ARMS for r in RULES] if s_ == "stack_all"
                     else [(s_[6:], r) for r in RULES])
            wts = {k: 0.0 for k in comps}
            Sv, St = zLv.copy(), zLt.copy()
            for _ in range(N_PASS):
                for k in comps:
                    bv = Sv - wts[k] * zSv[k]
                    b, _va = pick_beta_v2(bv, zSv[k], seen, tcv, BETAS_V2)
                    St = St - wts[k] * zSt[k] + b * zSt[k]
                    Sv = bv + b * zSv[k]
                    wts[k] = b
            res[s_].append(acc_v1(St, seen, yt))
            valacc[s_].append(acc_margin(Sv, seen, tcv)[0])
            wlog[s_].append({f"{a}:{r}": wts[(a, r)] for a, r in comps})

        log(f"    s{t}: rp {res['ranpac'][-1]*100:.2f}  " + "  ".join(
            f"{a}[" + " ".join(f"{r_} {res[f'{a}|fuse_{r_}'][-1]*100:.2f}"
                               f"/{res[f'{a}|fuse2_{r_}'][-1]*100:.2f}" for r_ in RULES)
            + "]" for a in ARMS)
            + "  " + " ".join(f"{s_} {res[s_][-1]*100:.2f}(v{valacc[s_][-1]*100:.1f})"
                              for s_ in STACKS))

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for k, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{k} out of range"
        out[k] = {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
    for s_ in STACKS:
        out[f"{s_}|_val"] = {"A_last": valacc[s_][-1],
                             "A_avg": float(np.mean(valacc[s_])),
                             "accs": valacc[s_]}
        out[f"{s_}|_w"] = {"weights": wlog[s_]}

    if verify:
        b = bar_for(ds, T, seed)
        assert abs(res["ranpac"][-1] - b["A_last"]) < 1e-6, (
            f"recomputed RanPAC {res['ranpac'][-1]:.6f} != exp16 bar {b['A_last']:.6f}")
        log("    VERIFY ok: RanPAC matches the exp16 bar")
        if os.path.exists(EXP52):
            d52 = json.load(open(EXP52))
            hit = [v for k, v in d52.items()
                   if k.startswith(f"{ds}|{T}|{seed}|") and "f32|fuse_cone" in v]
            if hit and "f32" in ARMS and "cone" in RULES:
                got, want = out["f32|fuse_cone"]["A_last"], hit[0]["f32|fuse_cone"]["A_last"]
                assert abs(got - want) < 1e-9, (
                    f"f32|fuse_cone {got:.10f} != exp52 {want:.10f}. The v1 control does not "
                    f"reproduce exp52, so every fuse2/stack delta is measured against a "
                    f"moving baseline. Check thread pinning and the arm names (the foreign "
                    f"subsample is keyed on crc32 of the arm name).")
                log(f"    VERIFY ok: f32|fuse_cone reproduces exp52 exactly ({got:.6f})")
            else:
                log("    VERIFY skipped exp52 repro: no matching cell cached")
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(ARMS)}|{'+'.join(RULES)}"
                       f"|{METHOD}g{GAMMA:g}|R{RMAX}_f{F_MAX}_s{SHRINK:g}_i{ITERS}"
                       f"|np{N_PASS}|m{M_RP}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 104
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if len(p) < 5 or p[3] != "+".join(ARMS) or p[4] != "+".join(RULES):
            continue
        cells[(p[0], int(p[2]))] = v

    def g(v, n, f):
        return v[n][f] * 100 if n in v else float("nan")

    STACKS = [f"stack_{a}" for a in ARMS] + (["stack_all"] if len(ARMS) > 1 else [])
    SOTA = {"CIFAR100": (91.97, 94.65), "IMAGENETR": (82.09, 86.20),
            "IMAGENETAP": (63.60, 70.24), "CUB200P": (89.91, 93.85)}

    print("\n" + "=" * W)
    print("EXP54 — joint multi-rule stack and a repaired beta search")
    print("=" * W)
    print(f"\narms {ARMS}  rules {RULES}  N_PASS {N_PASS}  cells {len(cells)}")

    print(f"\n{'-'*W}\nOVERFITTING CHECK — read this before the accuracy tables\n{'-'*W}")
    print(f"  {'ds':<12}{'arm':<12}{'val':>9}{'test':>9}{'val-test':>11}   "
          f"(a large positive gap means the weights fit the val split, not the problem)")
    for (ds, seed), v in sorted(cells.items()):
        for s_ in STACKS:
            if s_ in v and f"{s_}|_val" in v:
                va, te = g(v, f"{s_}|_val", "A_avg"), g(v, s_, "A_avg")
                print(f"  {ds:<12}{s_+' s'+str(seed):<12}{va:>9.2f}{te:>9.2f}{va-te:>11.2f}")

    for fld, lbl in (("A_last", "A-Last"), ("A_avg", "A-Avg")):
        print(f"\n{'-'*W}\n{lbl} — mean over seeds\n{'-'*W}")
        names = ["ranpac"] + [f"f32|fuse_{r}" for r in RULES] \
            + [f"f32|fuse2_{r}" for r in RULES] + STACKS
        print(f"  {'ds':<12}" + "".join(f"{n.replace('f32|',''):>14}" for n in names)
              + f"{'SOTA':>10}{'best-SOTA':>11}")
        for ds in sorted({d0 for d0, _ in cells}):
            sd_ = sorted(s for (d0, s) in cells if d0 == ds)
            row = f"  {ds:<12}"
            vals = {}
            for n in names:
                m = float(np.mean([g(cells[(ds, s)], n, fld) for s in sd_]))
                vals[n] = m
                row += f"{m:>14.2f}"
            so = SOTA.get(ds, (float("nan"),) * 2)[0 if fld == "A_last" else 1]
            row += f"{so:>10.2f}{max(vals.values())-so:>+11.2f}"
            print(row)

        print(f"\n  PAIRED vs f32|fuse_cone (exp52's best single arm)")
        print(f"  {'contrast':<26}{'mean':>9}{'sd':>9}{'wins':>8}   per-dataset")
        for n in [f"f32|fuse2_{r}" for r in RULES] + STACKS:
            dl = {}
            for (ds, seed), v in cells.items():
                if n in v and "f32|fuse_cone" in v:
                    dl.setdefault(ds, []).append(g(v, n, fld) - g(v, "f32|fuse_cone", fld))
            flat = [x for xs in dl.values() for x in xs]
            if not flat:
                continue
            sdv = float(np.std(flat, ddof=1)) if len(flat) > 1 else float("nan")
            per = "  ".join(f"{k}{np.mean(x):+.2f}" for k, x in sorted(dl.items()))
            print(f"  {n.replace('f32|','')+' - fuse_cone':<26}{np.mean(flat):>+9.2f}"
                  f"{sdv:>9.2f}{sum(x>0 for x in flat):>5}/{len(flat):<3}   {per}")

    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  0. THE OVERFITTING TABLE FIRST. stack_all fits 6 weights on a val split that is 10% of the
     classes seen so far -- ~40 rows at stage 0 on IMAGENETAP. If val-test is large and
     stack_all < stack_f32 on test, the component count is the cause and the fix is fewer
     components, not a better optimiser.
  1. `fuse2_cone - fuse_cone` isolates the beta repair ALONE. Expected small (+0.05-ish):
     s0 was pinned at exactly +0.00 across all four datasets by the tie-to-zero bug, and
     recovering one stage of ten is worth about that much on A-Avg.
  2. `stack_f32 - fuse_cone` is the actual proposal. IF IT IS <= 0 ON IMAGENETR A-AVG, the
     three rules are redundant given RanPAC, exp52's ensembling reading does not extend to
     stacking, and Tier 1 is finished -- the rest of the deficit is features.
  3. `best - SOTA` is context, not a result. Our RanPAC base is already ahead on CIFAR100 and
     behind by 0.74-1.58 A-Avg elsewhere; the read-out was measured to have ~+0.27 of
     headroom on IMAGENETR and ~0 on CUB200P, so only IMAGENETR can flip here.
  4. `_w` records the chosen weights per stage. A component whose weight is 0 at every stage
     contributed nothing and should be dropped from the stack rather than reported as part
     of it.""")
    print("=" * W)
    log(f"wrote {OUT}")
