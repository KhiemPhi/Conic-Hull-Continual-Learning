#!/usr/bin/env python3
"""exp56_ray_ensemble.py -- does the cone read-out COMPOSE with the LoRA-member ensemble,
and how does that composition scale with the ray budget?

WHY THIS FILE EXISTS
    Two mechanisms, measured independently on the SAME base (q32 RanPAC = 80.41 / 85.31 on
    IMAGENETR, identical in exp54 and exp55, so the protocols already agree):

        exp54  cone read-out   f32|fuse_cone   81.52 / 85.94   (+1.12 / +0.63)
        exp55  member ensemble ensemble        81.62 / 86.14   (+1.21 / +0.83)
        GR-LoRA                                82.09 / 86.20

    They are almost exactly the same size, which is the signature of a shared ceiling rather
    than two independent levers. If they were additive we would land at 82.74 / 86.77 and beat
    GR-LoRA outright. This file measures which it is.

THE PRE-REGISTERED PREDICTION -- WRITTEN DOWN BEFORE THE RUN
    exp52 decomposed fuse_cone's +1.12 A-Last on IMAGENETR into three nested pieces:

        rays alone (pm)        +0.30      mechanistically distinct from feature averaging
        + linear combination   +0.74      <- THIS is ensemble decorrelation, which the member
                                             ensemble already captures at the feature level
        + non-negativity       +0.08      distinct

    So the part of the cone gain that should SURVIVE composition is +0.30 + 0.08 ~ +0.38
    A-Last on top of the ensemble's 81.62, landing near 82.0 -- just under GR-LoRA.

        FALSIFIED IF the best honest arm beats ensemble by more than +0.7 A-Last. Then the
        substitution reading of exp52 is wrong, the two levers are independent, and this line
        of work is alive. Under +0.7 and they are substitutes; a read-out on top of a member
        ensemble is re-buying decorrelation that is already paid for.

ORDERING IS THE ACTUAL VARIABLE -- exp55 ALREADY KILLED ONE OF THEM
    exp55 measured `oracle_class_cv` = 80.12 against `best_single` = 80.41 on IMAGENETR: a
    per-class rule that picks among members captures a NEGATIVE share of the 6.22pt oracle
    headroom. Cone rays laid over concatenated member blocks are exactly such a rule, so:

      EF  ensemble-then-fuse   average member RanPAC scores, then fuse cone components that
                               were themselves averaged across members. NOTE: an earlier draft
                               of this file claimed the cv result predicted EF ~ 0. That was
                               wrong and the run disproved it (EF = +0.42 A-Last). EF AVERAGES
                               member cone scores; it never selects among members per class, so
                               oracle_class_cv -- a statement about hard per-class member
                               SELECTION -- does not bear on it. Nothing in this file tests the
                               cv statistic; a per-class member-WEIGHTING rule would, and none
                               of FE/EF/JT is one. EF's role is to isolate WHERE the averaging
                               happens, nothing more.
      FE  fuse-then-ensemble   fuse each member INDEPENDENTLY (its own betas, its own cones in
                               its own whitened space), then average the fused scores. Never
                               weights members per class, so the cv result does not apply.
                               This is the arm the prediction above is about.
      FEW fuse-then-ensemble,  FE with one scalar per member fitted on val ON TOP of the
          weighted             uniform average, so effective weight = 1/M + w_m and w=0 is
                               EXACTLY FE. It therefore cannot lose to FE on val, and any
                               test-side loss is pure val overfit. Motivated by the members
                               being visibly unequal -- q32 80.41 vs m32 78.29 solo, and
                               exp55's per-class winner_counts ran 88 (q32) vs 19 (q32b70) --
                               which makes uniform averaging a real constraint rather than a
                               neutral choice. Still a GLOBAL weight per member, not per
                               class, so exp55's oracle_class_cv still does not apply.
      JT  joint                every (member, arm, rule) triple as its own component in ONE
                               coordinate ascent over the averaged RanPAC base. Strictly the
                               most expressive and therefore the most likely to overfit val;
                               it exists to bound the others from above, not to be believed.

RAY BUDGETS FROM MIN TO MAX, AND EVERY SUBSET OF THEM
    exp53 found the conic rule's fused gain is monotone in the ray count R (+0.08 / +0.45 /
    +0.75 at R = 4 / 16 / 64), i.e. the cone behaves as an AGGREGATOR of noisy atom sets, not
    as a better descriptor. If that is what it is, then a member ensemble -- which is also an
    aggregator of noisy estimates -- should substitute for ray count, and the R-sweep should
    FLATTEN once members are averaged. That is a sharper test of the substitution reading than
    the headline number, and it is why every ray budget is swept rather than just f32.

    ARMS default f4,f8,f16,f32,f64,f128. RAYSETS=all enumerates all 2^6-1 = 63 non-empty
    subsets, ordered by (size, smallest R) -- min to max. Building the cones is the only
    expensive part and it is done ONCE per (member, arm); the 63 subsets x 3 orderings are
    then enumerated over cached z-scores, which is why an exhaustive sweep is affordable.

THE HAZARD, STATED BEFORE THE RESULT: THIS IS A MULTIPLE-COMPARISONS MACHINE
    63 raysets x |rule subsets| x 3 orderings is up to ~1300 configurations per cell. Reporting
    the max over them is not a result, it is shopping. Three guards, and the summary prints
    them before any accuracy:
      1. Every config reports VAL and TEST. The val-minus-test gap is printed first, exactly
         as in exp54.
      2. The HEADLINE number is `sel_{order}` -- the config chosen by VAL accuracy alone, then
         scored on test. That is the only number a deployable system could actually obtain.
         The max-over-configs test number is printed too, labelled as an oracle, because the
         gap between them IS the shopping penalty.
      3. The pre-registered prediction above is about `sel_FE`, fixed before the run.

TWO EXACT REPRODUCTION CONTROLS -- both asserted under VERIFY=1
    a) member q32, arm f32, rule cone, single-member fusion  ==  exp54's `f32|fuse_cone`
       to 1e-9. Helpers (zs / acc_v1 / acc_margin / pick_beta_v2 / score / rays_for) are
       IMPORTED from exp54_stack rather than copied so they cannot drift, and the foreign-
       negative RNG is keyed on crc32 of the arm name, so arms must stay named f{R}.
    b) the no-cone member average  ==  exp55's `ensemble` to 1e-9.
    If either fails, the composition is being measured against a moving baseline and no delta
    in this file is interpretable.

FEATURES ARE ALL CACHED -- THIS FILE NEVER TRAINS
    q32 is exp16's cache; the other members are exp55's. A missing cache is an error with the
    exp55 command to produce it, not a silent retrain.

PIN YOUR THREADS -- exp49 measured the unpinned noise floor at 0.27, larger than the entire
    effect under test, and it breaks both repro asserts.

USAGE
    source ~/venvs/ml_env/bin/activate

    # smoke: measures per-unit cost before committing to the full sweep
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0 MEMBERS=q32,m32 ARMS=f4,f32 RULES=cone \
      RAYSETS=all SUFFIX=_smoke VERIFY=1 python -u exp56_ray_ensemble.py

    # the sweep
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0,1,2 MEMBERS=q32,m32,a16,q32b70,q64 \
      ARMS=f4,f8,f16,f32,f64,f128 RULES=cone RAYSETS=all VERIFY=1 \
      python -u exp56_ray_ensemble.py

    Resumable: cells are written as they finish and existing keys are skipped.
    Run SEQUENTIALLY; concurrency on this box is measured strictly worse.
"""
import itertools
import json
import os
import time
import warnings
import zlib

import numpy as np
import torch

# b_kmeans clips to k = min(R_, len(Xw)), so any arm whose ray budget exceeds a class's fit-row
# count (~108-113 on IMAGENETR, i.e. f128) asks for one cluster per point and sklearn warns that
# duplicates collapsed it. That is the intended degenerate "all points" limit of the cone, not a
# fault -- but it fires once per class per stage and buries the log. Filtered NARROWLY on that
# message so real convergence failures elsewhere still surface. See the f128 note in _meta.
warnings.filterwarnings("ignore", message=r"Number of distinct clusters.*",
                        category=UserWarning, module=r"sklearn\..*")

_DS = os.environ.get("DS", "IMAGENETR").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0,1,2").split(",")]
_ARMS = os.environ.get("ARMS", "f4,f8,f16,f32,f64,f128").split(",")
_RULES = os.environ.get("RULES", "cone").split(",")
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])
# exp54_stack validates ARMS/RULES from the environment at import time and its helpers read
# module-level ITERS/SHRINK/METHOD/GAMMA/F_MAX. Handing it the SAME values this file uses is
# what makes the imported helpers bit-identical to the ones that produced exp54's numbers.
os.environ["ARMS"], os.environ["RULES"] = ",".join(_ARMS), ",".join(_RULES)

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
DSETS, TS, SEEDS = _DS, _TS, _SEEDS
ARMS, RULES = _ARMS, _RULES
MEMBERS = os.environ.get("MEMBERS", "q32,m32,a16,q32b70,q64").split(",")
ORDERS = os.environ.get("ORDERS", "FE,EF,JT").split(",")
RAYSETS = os.environ.get("RAYSETS", "all")
# JT enumerates |members| x |rayset| x |rules| components in a single ascent. Past this many
# it is fitting more weights than the stage-0 val split has rows, and exp54 already showed
# where that ends. Configs over the cap are skipped and COUNTED, never silently dropped.
JT_MAX = int(os.environ.get("JT_MAX", 12))
# exp55's member-feature cache parameters. These name the files; they do not retrain anything.
EPOCHS = int(os.environ.get("EPOCHS", 40))
LR = float(os.environ.get("LR", 3e-4))

METHOD, GAMMA, F_MAX = S54.METHOD, S54.GAMMA, S54.F_MAX
SHRINK, M_RP, N_PASS = S54.SHRINK, S54.M_RP, S54.N_PASS
LAMBDAS, BETAS_V1, BETAS_V2 = S54.LAMBDAS, S54.BETAS_V1, S54.BETAS_V2
VERIFY = int(os.environ.get("VERIFY", 0))

OUT = os.path.join(REPO, f"exp56_ray_ensemble{os.environ.get('SUFFIX', '')}_{TAG}.json")
EXP54 = os.path.join(REPO, f"exp54_stack_{TAG}.json")
EXP55 = os.path.join(REPO, f"exp55_lora_diversity_{TAG}.json")

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
acc_margin, pick_beta_v2, score, rays_for = (S54.acc_margin, S54.pick_beta_v2,
                                             S54.score, S54.rays_for)

assert MEMBERS[0] == "q32", (
    f"member 0 must be q32 -- it is exp16's cache and the control both repro asserts and "
    f"every delta in this file are measured against. Got {MEMBERS[0]!r}.")
assert all(a.startswith("f") and a[1:].isdigit() for a in ARMS), (
    f"arms must be fixed ray budgets named f{{R}} ({ARMS}). The foreign-negative subsample is "
    f"keyed on crc32 of the arm name, so renaming f32 draws different negatives and the exp54 "
    f"repro assert could never pass.")
assert set(ORDERS) <= {"FE", "FEW", "EF", "JT"}, f"unknown ordering in {ORDERS}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}); exp49 measured the unpinned noise "
        f"floor at 0.27, larger than the whole effect under test, and it breaks both repro "
        f"asserts. Prefix with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 or set ALLOW_UNPINNED=1.")


# ------------------------------------------------------------------ configurations
def raysets():
    """Ray-budget subsets, ordered MIN TO MAX: by size, then by smallest budget in the set.

    `all` is exhaustive (2^|ARMS| - 1). `up`/`down` are the nested prefixes from the smallest
    budget upward / the largest downward, which is the cheap way to read whether the sweep is
    monotone. Enumeration runs over cached z-scores, so `all` costs enumeration time only --
    the cones are built once per (member, arm) regardless of how many subsets reference them."""
    idx = list(range(len(ARMS)))
    if RAYSETS == "single":
        sets = [(i,) for i in idx]
    elif RAYSETS == "up":
        sets = [tuple(idx[:k + 1]) for k in idx]
    elif RAYSETS == "down":
        sets = [tuple(idx[len(idx) - k - 1:]) for k in idx]
    elif RAYSETS == "all":
        sets = [c for k in range(1, len(idx) + 1) for c in itertools.combinations(idx, k)]
    else:
        raise AssertionError(f"RAYSETS must be all|up|down|single, got {RAYSETS!r}")
    sets = sorted(set(sets), key=lambda s: (len(s), s))
    return [tuple(ARMS[i] for i in s) for s in sets]


def rulesets():
    return [tuple(c) for k in range(1, len(RULES) + 1)
            for c in itertools.combinations(RULES, k)]


def cfg_name(order, rs, ru):
    return f"{order}|{'+'.join(rs)}|{'+'.join(ru)}"


# ------------------------------------------------------------------ member features
def member_features(ds, T, seed, spec):
    """Cached features only. q32 is exp16's; the rest are exp55's. Never trains."""
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
        f"member {spec} feature cache missing: {f}\nThis file never trains. Produce it with:\n"
        f"  {hint}")
    z = np.load(f)
    return un(z["Ftr"]), un(z["Fte"])


# ------------------------------------------------------------------ one cell
def run_cell(ds, T, seed, verify):
    E.T, E.SEED = T, seed
    ytr, yte, n_cls = E.get_labels(ds)
    Z = {m: member_features(ds, T, seed, m) for m in MEMBERS}
    d = Z[MEMBERS[0]][0].shape[1]
    cpt = n_cls // T

    # Task split is keyed on `seed` alone and is IDENTICAL across members -- otherwise the
    # members are not solving the same problem and nothing here is comparable. Same carve as
    # exp54/exp55, which is what both repro asserts depend on.
    order = CO.class_order(n_cls, seed)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm_ = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm_[:nv]]); FIT.append(ix[pm_[nv:]])
    VAL_ALL = np.concatenate(VAL)

    # P is seeded 0 in exp54 and exp55 alike and d is the same for every member, so all
    # members share one projection. Changing this breaks both repro asserts.
    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    G = {m: torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64) for m in MEMBERS}
    C = {m: torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64) for m in MEMBERS}
    scat = {m: (np.zeros((d, d), np.float64), 0) for m in MEMBERS}
    A = {m: {a: {} for a in ARMS} for m in MEMBERS}

    def _H(Zm, bs=4096):
        for i in range(0, len(Zm), bs):
            yield i, torch.relu(torch.as_tensor(Zm[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Zm, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Zm)]).cpu().numpy()

    # ---- reference trajectories for the two repro controls, loaded BEFORE the stage loop.
    # Both exp54 and exp55 store per-stage `accs`, so the controls can be asserted at STAGE 0
    # rather than at the end of the cell. Checking only A_last would mean paying the whole cell
    # (~20 min on the 2x2 smoke, hours on the full sweep) before discovering the read-out here
    # is not the one that produced their numbers -- at which point nothing in the run is
    # interpretable anyway. Fail in 40 seconds instead.
    ref54 = ref55 = None
    if verify:
        # exp54 predates class_order.py, so its results key carries NO order tag and a
        # `{ds}|{T}|{seed}|` prefix match would happily return a LEGACY-order trajectory.
        # Asserting PILOT-order results against it would fail for a reason that has nothing to
        # do with the read-out. Gate it on the mode it was actually produced under.
        if CO.mode() != "legacy":
            log(f"    VERIFY: exp54 anchor N/A under ORDER={CO.mode()} (exp54 is legacy-order "
                f"only); relying on the exp55 anchor, which is regenerated per order")
        elif "f32" in ARMS and "cone" in RULES and os.path.exists(EXP54):
            d54 = json.load(open(EXP54))
            hit = [v for k, v in d54.items()
                   if k.startswith(f"{ds}|{T}|{seed}|") and "f32|fuse_cone" in v]
            ref54 = hit[0]["f32|fuse_cone"]["accs"] if hit else None
            log(f"    VERIFY: exp54 f32|fuse_cone reference "
                f"{'loaded' if ref54 else 'NOT FOUND (skipping)'}")
        k55 = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}"
               f"|ep{EPOCHS}_lr{LR:g}_a4{CO.order_tag()}|m{M_RP}|v1")
        if os.path.exists(EXP55):
            d55 = json.load(open(EXP55))
            ref55 = d55[k55]["ensemble"]["accs"] if k55 in d55 else None
            log(f"    VERIFY: exp55 ensemble reference "
                f"{'loaded' if ref55 else f'NOT FOUND for {k55} (skipping)'}")
        assert ref54 is not None or ref55 is not None, (
            f"VERIFY=1 but NEITHER anchor is available for {ds} T={T} s={seed} under "
            f"ORDER={CO.mode()}. A run with zero controls is not a verified run -- it would "
            f"report `verified: {{false, false}}` in _meta and look identical to one that "
            f"passed. Produce the exp55 cell first:\n  ORDER={CO.mode()} DS={ds} T={T} "
            f"SEED={seed} MEMBERS={','.join(MEMBERS)} python -u exp55_lora_diversity_pilot.py"
            f"\nor rerun with VERIFY=0 to proceed deliberately unverified.")

    RS, RU = raysets(), rulesets()
    CFGS = [(o, rs, ru) for o in ORDERS for rs in RS for ru in RU]
    log(f"    {len(MEMBERS)} members x {len(ARMS)} arms x {len(RULES)} rules; "
        f"{len(RS)} raysets x {len(RU)} rulesets x {len(ORDERS)} orders = {len(CFGS)} configs")

    res, val = {}, {}
    skipped_jt = 0
    wlog_few = {}   # final-stage member weights per FEW config; a member pinned at ~0 is
                    # one the ensemble does not want, which is a result in itself

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

        zLv, zLt, zSv, zSt = {}, {}, {}, {}
        for m in MEMBERS:
            Ztr_m, Zte_m = Z[m]
            # ---- whitening from this member's own accumulated within-class scatter
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

            # ---- cones, one set per (member, arm). Loop order over classes then arms is
            # exp54's, and the crc32-on-arm-name RNG makes the f32 draw identical to exp54's.
            for c in tasks[t]:
                r = FIT[t][ytr[FIT[t]] == c]
                if len(r) < 2:
                    continue
                Xw = un(Ztr_m[r] @ Wh)
                for a in ARMS:
                    rng = np.random.default_rng(
                        1234 + 97 * t + zlib.crc32(a.encode()) % 1000)
                    Fw = np.zeros((0, d), np.float32)
                    if METHOD in X.DISCRIM and GAMMA > 0:
                        oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                        past = [A[m][a][o] for o in A[m][a] if o not in tasks[t]]
                        Fr = np.concatenate([Ztr_m[oth]] + past, 0)
                        if len(Fr) > F_MAX:
                            Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                        Fw = un(Fr @ Wh)
                    A[m][a][int(c)] = X.BUILD[METHOD](
                        Xw, Fw, rays_for(a, len(r)), int(c), GAMMA) @ Wh_inv

            # ---- RanPAC for this member
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
            Lv, Lt = logits(un(Ztr_m[vix]), bw), logits(un(Zte_m), bw)[tei]
            zLv[m], zLt[m] = zs(Lv, seen), zs(Lt, seen)
            put(f"{m}|ranpac", acc_v1(zLt[m], seen, yt))

            # ---- cone/sub/pm scores for every (arm, rule)
            Qvw, Qtw = un(Ztr_m[vix] @ Wh), un(Zte_m[tei] @ Wh)
            miss = [c for c in seen if c not in A[m][ARMS[0]]]
            for a in ARMS:
                for r_ in RULES:
                    Sv = np.full((nval, n_cls), -np.inf, np.float32)
                    St = np.full((len(tei), n_cls), -np.inf, np.float32)
                    for c in seen:
                        if c not in A[m][a]:
                            continue
                        Ac = un(A[m][a][c] @ Wh)
                        Sv[:, c] = score(r_, Ac, Qvw)
                        St[:, c] = score(r_, Ac, Qtw)
                    zv, zt = zs(Sv, seen), zs(St, seen)
                    if miss:
                        # Classes with <2 fit rows have no cone; zs sends their -inf to the row
                        # MINIMUM, which would actively suppress classes RanPAC calls right.
                        # Neutralise to the z-mean so they fall back to RanPAC. exp54 verbatim.
                        zv[:, miss] = 0.0
                        zt[:, miss] = 0.0
                    zSv[(m, a, r_)] = zv.astype(np.float32)
                    zSt[(m, a, r_)] = zt.astype(np.float32)
                    # exp54's single-member v1 control arm, for member q32 the repro anchor.
                    b1 = max(BETAS_V1, key=lambda bb: acc_v1(zLv[m] + bb * zv, seen, yv))
                    put(f"{m}|{a}|fuse_{r_}", acc_v1(zLt[m] + b1 * zt, seen, yt))

        # ---- no-cone baselines: best single member, and exp55's plain member ensemble
        eLv = sum(zLv[m] for m in MEMBERS) / len(MEMBERS)
        eLt = sum(zLt[m] for m in MEMBERS) / len(MEMBERS)
        put("ens_ranpac", acc_v1(eLt, seen, yt))

        # ---- both controls, asserted per stage so a drifted read-out fails at s0
        if ref54 is not None:
            got, want = res["q32|f32|fuse_cone"][-1], ref54[t]
            assert abs(got - want) < 1e-9, (
                f"s{t}: q32|f32|fuse_cone {got:.10f} != exp54 {want:.10f}. The cone read-out "
                f"here is not the one that produced exp54's numbers, so every composition "
                f"delta is measured against a moving baseline. Check thread pinning and arm "
                f"names (the foreign subsample is keyed on crc32 of the arm name).")
        if ref55 is not None:
            got, want = res["ens_ranpac"][-1], ref55[t]
            assert abs(got - want) < 1e-9, (
                f"s{t}: ens_ranpac {got:.10f} != exp55 ensemble {want:.10f}. The member "
                f"ensemble here is not exp55's, so 'does the cone ADD to the ensemble' is "
                f"being asked of a different ensemble.")
        if t == 0 and (ref54 is not None or ref55 is not None):
            log(f"    VERIFY ok at s0: "
                + "  ".join(filter(None, [
                    f"exp54 f32|fuse_cone {res['q32|f32|fuse_cone'][0]:.6f}"
                    if ref54 is not None else "",
                    f"exp55 ensemble {res['ens_ranpac'][0]:.6f}" if ref55 is not None else ""])))

        def ascend(base_v, base_t, comps, getv, gett):
            """Coordinate ascent on val over `comps`; returns (test, val, val acc, weights).
            Identical in structure to exp54's stack loop -- same N_PASS, same grid, same
            (accuracy, margin) selection. The VAL score matrix is returned as well because
            FEW needs each member's fused val score to fit the member weights on."""
            w = {k: 0.0 for k in comps}
            Sv, St = base_v.copy(), base_t.copy()
            for _ in range(N_PASS):
                for k in comps:
                    bv = Sv - w[k] * getv(k)
                    b, _ = pick_beta_v2(bv, getv(k), seen, tcv, BETAS_V2)
                    St = St - w[k] * gett(k) + b * gett(k)
                    Sv = bv + b * getv(k)
                    w[k] = b
            return St, Sv, acc_margin(Sv, seen, tcv)[0], w

        # ---- the sweep
        for o, rs, ru in CFGS:
            name = cfg_name(o, rs, ru)
            comps = [(a, r_) for a in rs for r_ in ru]
            if o in ("FE", "FEW"):
                # fuse each member on ITS OWN betas and ITS OWN cones, then combine the
                # re-z-scored fused scores. Never weights members per class.
                Svs, Sts, vas = {}, {}, []
                for m in MEMBERS:
                    St, Sv, va, _ = ascend(zLv[m], zLt[m], comps,
                                           lambda k, m=m: zSv[(m,) + k],
                                           lambda k, m=m: zSt[(m,) + k])
                    Svs[m], Sts[m] = zs(Sv, seen), zs(St, seen)
                    vas.append(va)
                bt = sum(Sts.values()) / len(MEMBERS)
                if o == "FE":
                    put(name, acc_v1(bt, seen, yt), float(np.mean(vas)))
                else:
                    # FEW: one scalar per member ON TOP of the uniform average, so the
                    # effective weight is 1/M + w_m and w=0 reproduces FE EXACTLY. FEW
                    # therefore cannot lose to FE on val by construction, and any test-side
                    # loss is pure val overfit -- which is the whole point of measuring it.
                    # Members are not equal (q32 80.41 vs m32 78.29 solo, and exp55's
                    # winner_counts ran 88 vs 19 classes), so uniform is a real constraint.
                    bv = sum(Svs.values()) / len(MEMBERS)
                    St2, Sv2, va2, wm = ascend(bv, bt, list(MEMBERS),
                                               Svs.__getitem__, Sts.__getitem__)
                    put(name, acc_v1(St2, seen, yt), va2)
                    if t == T - 1:
                        wlog_few[name] = {m: round(1.0 / len(MEMBERS) + wm[m], 4)
                                          for m in MEMBERS}
            elif o == "EF":
                # average first, then fuse member-averaged components. This is the arm exp55's
                # oracle_class_cv predicts to be worthless; it is the control, not a hope.
                gv = {k: sum(zSv[(m,) + k] for m in MEMBERS) / len(MEMBERS) for k in comps}
                gt = {k: sum(zSt[(m,) + k] for m in MEMBERS) / len(MEMBERS) for k in comps}
                St, _, va, _ = ascend(eLv, eLt, comps, gv.__getitem__, gt.__getitem__)
                put(name, acc_v1(St, seen, yt), va)
            else:
                jc = [(m, a, r_) for m in MEMBERS for a in rs for r_ in ru]
                if len(jc) > JT_MAX:
                    skipped_jt += 1
                    continue
                St, _, va, _ = ascend(eLv, eLt, jc, zSv.__getitem__, zSt.__getitem__)
                put(name, acc_v1(St, seen, yt), va)

        line = "  ".join(f"{m} {res[f'{m}|ranpac'][-1]*100:.2f}" for m in MEMBERS)
        line += f"  | ens {res['ens_ranpac'][-1]*100:.2f}"
        for o in ORDERS:
            got = [res[cfg_name(o, rs, ru)][-1] for rs in RS for ru in RU
                   if cfg_name(o, rs, ru) in res]
            if got:
                line += f"  best{o} {max(got)*100:.2f}"
        log(f"    s{t}: {line}")

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for k, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{k} out of range"
        out[k] = {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
        if k in val:
            out[f"{k}|_val"] = {"A_last": val[k][-1], "A_avg": float(np.mean(val[k])),
                                "accs": val[k]}
    # Per-class fit-row counts bound the REALISED ray budget: b_kmeans clips to
    # k = min(R_, n_rows). An arm at or above the median row count is not the fixed budget its
    # name claims -- it is "one ray per point", clipped to a DIFFERENT R for every class. On
    # IMAGENETR that is f128 (~108-113 rows/class). Recorded so the R-sweep is not read as
    # flattening on its own merits when the top end is simply saturated.
    nrows = [int((ytr[FIT[t]] == c).sum()) for t in range(T) for c in tasks[t]]
    med = float(np.median(nrows))
    out["_meta"] = {"members": MEMBERS, "arms": ARMS, "rules": RULES, "orders": ORDERS,
                    "n_cfg": len(CFGS), "jt_skipped": skipped_jt, "jt_max": JT_MAX,
                    "cpt": cpt, "order": CO.mode(), "few_member_weights": wlog_few,
                    "fit_rows_per_class": {"min": int(np.min(nrows)), "median": med,
                                           "max": int(np.max(nrows))},
                    "clipped_arms": [a for a in ARMS if rays_for(a, med) >= med]}
    if skipped_jt:
        log(f"    NOTE {skipped_jt} JT configs skipped: more than JT_MAX={JT_MAX} components")

    # Both controls were asserted per stage in the loop above (which covers A_last at t=T-1),
    # so there is nothing left to check here -- only to record what was actually verified, so a
    # cell whose controls were SKIPPED is never mistaken for one that passed them.
    out["_meta"]["verified"] = {"exp54_fuse_cone": ref54 is not None,
                                "exp55_ensemble": ref55 is not None}
    if verify:
        log(f"    VERIFY ok over all {T} stages: "
            f"exp54={'yes' if ref54 is not None else 'SKIPPED'}  "
            f"exp55={'yes' if ref55 is not None else 'SKIPPED'}")
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|{'+'.join(ARMS)}"
                       f"|{'+'.join(RULES)}|{'+'.join(ORDERS)}|rs{RAYSETS}"
                       f"|{METHOD}g{GAMMA:g}|s{SHRINK:g}_i{S54.ITERS}|np{N_PASS}|m{M_RP}"
                       f"{CO.order_tag()}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                t_ = time.time()
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                log(f"    cell took {time.time()-t_:.0f}s")
                json.dump(allres, open(OUT, "w"), indent=2)

    # ------------------------------------------------------------------ summary
    W = 104
    # Cells are keyed by (ds, T, seed). Keying on (ds, seed) alone -- as this did -- makes
    # T=10/20/50 collide on the same key and the LAST one silently wins the whole table.
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if len(p) < 7 or p[3] != "+".join(MEMBERS) or p[4] != "+".join(ARMS):
            continue
        cells[(p[0], int(p[1]), int(p[2]))] = v
    RS, RU = raysets(), rulesets()
    NAMES = [cfg_name(o, rs, ru) for o in ORDERS for rs in RS for ru in RU]

    def seeds_of(ds, T):
        return sorted(s for (d0, t0, s) in cells if d0 == ds and t0 == T)

    def gm(ds, T, n, f):
        xs = [cells[(ds, T, s)][n][f] * 100 for s in seeds_of(ds, T)
              if n in cells[(ds, T, s)]]
        return float(np.mean(xs)) if xs else float("nan")

    # Published GR-LoRA numbers keyed by (dataset, T) -- SOURCE OF TRUTH IS exp16_full_table.REF
    # (GR-LoRA ICML'26 Tables 1,2,6, ViT-B/16-IN21k, mean of 3 seeds). Duplicated rather than
    # imported because importing exp16 builds a timm model at module scope. The previous dict
    # here was keyed by dataset ALONE and therefore printed the T=10 baseline against every T.
    REF = {("CIFAR100", 10): (91.97, 94.65), ("CIFAR100", 20): (91.46, 94.41),
           ("CIFAR100", 50): (90.03, 93.38),
           ("IMAGENETR", 10): (82.09, 86.20), ("IMAGENETR", 20): (80.23, 85.05),
           ("IMAGENETR", 50): (76.74, 82.64),
           ("IMAGENETAP", 10): (63.60, 70.24), ("IMAGENETAP", 20): (62.37, 69.30),
           ("IMAGENETAP", 50): (59.71, 67.23),
           ("CUB200P", 10): (89.91, 93.85), ("CUB200P", 20): (89.76, 94.08),
           ("CUB200P", 50): (89.68, 93.94)}
    dts = sorted({(d0, t0) for d0, t0, _ in cells})

    print("\n" + "=" * W)
    print("EXP56 — does the cone read-out compose with the LoRA-member ensemble?")
    print("=" * W)
    print(f"\nmembers {MEMBERS}\narms {ARMS}  rules {RULES}  orders {ORDERS}  "
          f"raysets {RAYSETS} ({len(RS)})  cells {len(cells)}")

    print(f"\n{'-'*W}\nSHOPPING PENALTY — read before any accuracy. `sel` picks the config on "
          f"VAL;\n`oracle` is the max over configs on TEST and is NOT obtainable.\n{'-'*W}")
    print(f"  {'ds':<12}{'T':>4} {'order':<6}{'sel(test)':>11}{'oracle':>9}{'penalty':>9}"
          f"{'sel val-test':>14}{'chosen config':>30}")
    SEL = {}
    for ds, T in dts:
        for o in ORDERS:
            probe = cells[(ds, T, seeds_of(ds, T)[0])]
            cand = [n for n in NAMES if n.startswith(f"{o}|") and f"{n}|_val" in probe]
            if not cand:
                continue
            for fld in ("A_last", "A_avg"):
                bestn = max(cand, key=lambda n: gm(ds, T, f"{n}|_val", fld))
                SEL[(ds, T, o, fld)] = (bestn, gm(ds, T, bestn, fld))
            bn, bt = SEL[(ds, T, o, "A_avg")]
            orc = max(gm(ds, T, n, "A_avg") for n in cand)
            vt = gm(ds, T, f"{bn}|_val", "A_avg") - bt
            print(f"  {ds:<12}{T:>4} {o:<6}{bt:>11.2f}{orc:>9.2f}{orc-bt:>9.2f}{vt:>14.2f}"
                  f"{bn.split('|', 1)[1]:>30}")

    for fld, lbl in (("A_last", "A-Last"), ("A_avg", "A-Avg")):
        print(f"\n{'-'*W}\n{lbl} — baselines and the honest (val-selected) composition\n{'-'*W}")
        hdr = ["q32|ranpac", "ens_ranpac", "q32|f32|fuse_cone"]
        print(f"  {'ds':<12}{'T':>4}{'seeds':>6}{'q32':>9}{'ens':>9}{'q32+cone':>10}"
              + "".join(f"{'sel_'+o:>9}" for o in ORDERS)
              + f"{'best-ens':>10}{'GR-LoRA':>9}{'best-GR':>9}")
        for ds, T in dts:
            b = [gm(ds, T, n, fld) for n in hdr]
            sels = [SEL.get((ds, T, o, fld), (None, float('nan')))[1] for o in ORDERS]
            so = REF.get((ds, T), (float("nan"),) * 2)[0 if fld == "A_last" else 1]
            bo = max([x for x in sels if x == x] or [float("nan")])
            print(f"  {ds:<12}{T:>4}{len(seeds_of(ds, T)):>6}"
                  + "".join(f"{x:>9.2f}" for x in b[:2]) + f"{b[2]:>10.2f}"
                  + "".join(f"{x:>9.2f}" for x in sels)
                  + f"{bo-b[1]:>+10.2f}{so:>9.2f}{bo-so:>+9.2f}")

    print(f"\n{'-'*W}\nRAY-BUDGET SWEEP — single budgets, min to max. Does the cone's known\n"
          f"monotonicity in R SURVIVE member averaging, or does the ensemble substitute for "
          f"rays?\n{'-'*W}")
    for ds, T in dts:
        probe = cells[(ds, T, seeds_of(ds, T)[0])]
        mt = probe.get("_meta", {})
        clip = set(mt.get("clipped_arms", []))
        fr = mt.get("fit_rows_per_class", {})
        print(f"  {ds} T={T} (cpt={mt.get('cpt', '?')})   fit rows/class "
              f"{fr.get('min','?')}-{fr.get('max','?')} (median {fr.get('median','?')})"
              + (f"   CLIPPED ARMS {sorted(clip)}: k=min(R, n_rows), so these are "
                 f"'one ray per point' at a DIFFERENT R per class, not a fixed budget"
                 if clip else ""))
        print(f"    {'arm':<8}{'q32 alone':>11}" + "".join(f"{o:>9}" for o in ORDERS)
              + "     (A-Last, vs ens_ranpac)   * = clipped")
        e = gm(ds, T, "ens_ranpac", "A_last")
        for a in ARMS:
            solo = (gm(ds, T, f"q32|{a}|fuse_{RULES[0]}", "A_last")
                    - gm(ds, T, "q32|ranpac", "A_last"))
            row = f"    {a + ('*' if a in clip else ''):<8}{solo:>+11.2f}"
            for o in ORDERS:
                n = cfg_name(o, (a,), (RULES[0],))
                row += f"{gm(ds, T, n, 'A_last')-e:>+9.2f}" if n in probe else f"{'--':>9}"
            print(row)

    print(f"\n{'-'*W}\nPRE-REGISTERED TEST — sel_FE vs ens_ranpac, IMAGENETR A-Last, per T\n"
          f"{'-'*W}")
    for ds, T in dts:
        if ds != "IMAGENETR" or (ds, T, "FE", "A_last") not in SEL:
            continue
        e = gm(ds, T, "ens_ranpac", "A_last")
        v = SEL[(ds, T, "FE", "A_last")][1]
        verdict = ("FALSIFIED: levers INDEPENDENT" if v - e > 0.70 else
                   "consistent with SUBSTITUTES")
        print(f"  T={T:<3} ens_ranpac {e:.2f}   sel_FE {v:.2f}   delta {v-e:+.2f}   "
              f"(predicted ~+0.38, FALSIFIED IF > +0.70)  -> {verdict}")

    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  0. SHOPPING PENALTY FIRST. This file evaluates up to ~1300 configs per cell. `oracle` is the
     max over them on TEST and is not obtainable by any deployable system; `sel` picks on VAL
     and is the only honest number. A large penalty means the sweep found noise, and a large
     `sel val-test` means even the val-selected config overfits the split.
  1. THE PRE-REGISTERED TEST is sel_FE - ens_ranpac on IMAGENETR A-Last. exp52's decomposition
     says +0.30 (rays) + 0.08 (non-negativity) ~ +0.38 should survive and the +0.74 linear-
     combination piece should NOT, because the member ensemble already bought it. Above +0.70
     that reading is dead and the levers are independent.
  2. EF/JT ISOLATE WHERE THE AVERAGING HAPPENS -- they do NOT test exp55's oracle_class_cv
     (see the EF note in the header; none of these orderings weights members per class). The
     informative column for them is `sel val-test`, NOT accuracy. Measured on the f32 control:
     FE 0.86, EF 1.98, JT 2.37 -- the overfitting order is the exact inverse of the accuracy
     order. FE fuses each member's cone against THAT member's own RanPAC in matched geometry
     and renormalises before averaging; EF and JT fuse against a POOLED base and pay 2-3x the
     val overfit for it. Treat a future EF/JT win as suspect until its gap matches FE's.
  3. THE RAY SWEEP IS THE MECHANISM READ. exp53 found the fused gain monotone in R because the
     cone is an AGGREGATOR of noisy atom sets. A member ensemble is also an aggregator. If the
     substitution reading is right, the `q32 alone` column stays monotone in R while the FE/EF
     columns FLATTEN -- the ensemble has already done the aggregating. A still-monotone FE
     column would mean rays contribute something averaging cannot.
  4. JT bounds FE and EF from above and is the most likely to overfit; believe it only if its
     val-test gap is comparable to FE's. Configs above JT_MAX components are skipped and the
     count is recorded in _meta, never silently dropped.""")
    print("=" * W)
    log(f"wrote {OUT}")
