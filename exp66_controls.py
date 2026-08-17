#!/usr/bin/env python3
"""exp66_controls.py -- the control battery for the ensemble-LoRA method paper (C2/C3/C5).

WHAT THIS ANSWERS, AND WHY EACH ONE IS A SUBMISSION BLOCKER
    C2 RULE CONTROL -- does "conic" mean anything?
        The read-out scores a query against R rays per class. `cone` solves a non-negative
        least squares against them; `sub` projects onto their SVD basis (same rays, same
        storage, NON-NEGATIVITY DROPPED); `pm` takes the max cosine to any single ray (same
        rays, no combination at all). All three are already implemented in exp54_stack.score
        and cost identical storage, so the comparison is exactly matched on capacity.
        If cone ~= sub, the result is about the SPAN of the rays and the word "conic" should
        come out of the title. If cone ~= pm, it is about nearest-atom retrieval. This is the
        control that closed every previous conic claim in this repo
        ([[cone-is-dead-weight-final]], [[nonneg-space-hypothesis-falsified]]), and
        [[oriented-pca-cone-ties-ranpac]] explicitly lists "is sub ~= cone on those rays?" as
        UNRESOLVED. It is one environment variable and it has never been run in the ensemble.

    C3 ONE FIXED CONFIGURATION -- no per-cell shopping.
        exp56 evaluates up to ~1300 configs per cell and results.txt currently reports f64 on
        two datasets and a val-selected multi-budget arm on the other two. That is post-hoc
        per-cell selection and a referee will read it as shopping. This file evaluates ONE
        arm (default f64) on every cell, chosen before the run and never re-selected. The
        2026-08-13 CUB200P T=20 sweep makes this free: sweeping every unclipped budget bought
        +0.08 A-Last / +0.03 A-Avg over clipped f64, i.e. nothing.

    C5 MEMBER ABLATION -- which members carry the gain, and does it saturate?
        Singletons, leave-one-out, and the nested M-sweep. If the ensemble saturates at M=3
        the method gets 40% cheaper and the paper gets stronger, so this is not a defensive
        control -- it is a result. Nested prefixes are used rather than best-M-on-val because
        a val-selected subset would reintroduce exactly the shopping C3 removes.

WHY THIS FILE HAS ITS OWN LOOP, AND THE ASSERT THAT MAKES THAT SAFE
    Leave-one-out over 5 members through exp56 would rebuild cones 4x per subset -- 20 member
    builds where 5 suffice, and cone construction is the only expensive part of the read-out.
    So the outer loop is re-implemented here. Duplicated loops are how this project has
    previously lost weeks (see splits.py), so two things protect it:
      1. EVERY helper is imported, never copied: zs / acc_v1 / acc_margin / pick_beta_v2 /
         score / rays_for from exp54_stack, BUILD/un from exp39, member caches from exp56.
      2. Under VERIFY=1 the full-member FE|{ARM}|cone A_last is asserted against the value
         exp56 already stored for the same cell, to 1e-9. If this loop has drifted from the
         one that produced the paper's headline table, the run dies before reporting anything.

THE FACTORISATION THAT MAKES THE WHOLE BATTERY CHEAP
    FE fuses each member against ITS OWN RanPAC in its own whitened geometry, re-z-scores,
    and only THEN averages. The per-member fused score therefore does not depend on which
    subset it will be averaged into. So per stage we do |members| x |rules| coordinate
    ascents ONCE and every subset is a mean over cached vectors. 5 members x 3 rules = 15
    ascents serves all 14 subsets x 3 rules + 14 no-cone baselines.

FEATURES ARE CACHED -- THIS FILE NEVER TRAINS. A missing member cache is an error carrying
    the exp55 command that produces it.

PIN YOUR THREADS -- exp49 measured the unpinned noise floor at 0.27, larger than several of
    the effects under test, and it breaks the exp56 assert.

USAGE
    source ~/venvs/ml_env/bin/activate

    # smoke -- one cell, checks the exp56 assert fires and the factorisation is right
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot \
      DS=IMAGENETR T=10 SEED=0 VERIFY=1 SUFFIX=_smoke python -u exp66_controls.py

    # the battery, all four datasets x three task counts
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot \
      DS=CIFAR100,IMAGENETR,CUB200P,IMAGENETAP T=10,20,50 SEED=0,1,2 VERIFY=1 \
      python -u exp66_controls.py

    Resumable: cells are written as they finish and existing keys are skipped.
"""
import itertools
import json
import os
import time
import warnings
import zlib

import numpy as np
import torch

warnings.filterwarnings("ignore", message=r"Number of distinct clusters.*",
                        category=UserWarning, module=r"sklearn\..*")

_DS = os.environ.get("DS", "IMAGENETR").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0,1,2").split(",")]
# ONE arm, fixed before the run. This is C3; do not turn it into a list.
ARM = os.environ.get("ARM", "f64")
_RULES = os.environ.get("RULES", "cone,sub,pm").split(",")
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])
# exp54_stack validates ARMS/RULES from the environment at import time and its helpers read
# module-level ITERS/SHRINK/METHOD/GAMMA/F_MAX. Handing it the SAME values used here is what
# makes the imported helpers bit-identical to the ones that produced exp56's numbers.
os.environ["ARMS"], os.environ["RULES"] = ARM, ",".join(_RULES)

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402
import exp54_stack as S54                   # noqa: E402
import exp56_ray_ensemble as S56            # noqa: E402
import class_order as CO                    # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = S54.TAG
DSETS, TS, SEEDS = _DS, _TS, _SEEDS
RULES = _RULES
MEMBERS = os.environ.get("MEMBERS", "q32,m32,a16,q32b70,q64").split(",")
VERIFY = int(os.environ.get("VERIFY", 0))

METHOD, GAMMA, F_MAX = S54.METHOD, S54.GAMMA, S54.F_MAX
SHRINK, M_RP, N_PASS = S54.SHRINK, S54.M_RP, S54.N_PASS
LAMBDAS, BETAS_V1, BETAS_V2 = S54.LAMBDAS, S54.BETAS_V1, S54.BETAS_V2

OUT = os.path.join(REPO, f"exp66_controls{os.environ.get('SUFFIX', '')}_{TAG}.json")
EXP56 = os.path.join(REPO, f"exp56_ray_ensemble_table_{TAG}.json")

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
acc_margin, pick_beta_v2, score, rays_for = (S54.acc_margin, S54.pick_beta_v2,
                                             S54.score, S54.rays_for)
member_features = S56.member_features          # imported: cache naming cannot drift

assert MEMBERS[0] == "q32", f"member 0 must be q32 (exp16's cache); got {MEMBERS[0]!r}"
assert ARM.startswith("f") and ARM[1:].isdigit(), (
    f"ARM must be a fixed ray budget named f{{R}}, got {ARM!r}. The foreign-negative "
    f"subsample is keyed on crc32 of the arm name, so renaming it draws different negatives.")
assert set(RULES) <= {"cone", "sub", "pm"}, f"unknown rule in {RULES}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}); exp49 measured the unpinned noise "
        f"floor at 0.27 and it breaks the exp56 assert. Prefix with OMP_NUM_THREADS=1 "
        f"MKL_NUM_THREADS=1 or set ALLOW_UNPINNED=1.")


# ------------------------------------------------------------------ member subsets (C5)
def subsets():
    """(label, tuple_of_members), deduplicated, in reporting order.

    singles   each member alone -- the per-member baseline the ensemble must beat
    nested    q32, q32+m32, ... -- the M-sweep. NESTED PREFIXES, NOT best-M-on-val: a
              val-selected subset would reintroduce the per-cell shopping C3 exists to remove.
    loo       full set minus one -- attributes the gain to individual members
    full      the method as published
    """
    # `full` is pre-seeded so the nested prefix of length |MEMBERS| -- which is the SAME tuple
    # -- deduplicates away and the canonical label survives. Registering it last instead would
    # silently leave the full-set arm named `nested:M5`, and the exp56 assert (which looks up
    # `full|FE|cone`) would die with a KeyError.
    full = tuple(MEMBERS)
    out, seen = [], {full}

    def add(lbl, ms):
        k = tuple(ms)
        if k and k not in seen:
            seen.add(k); out.append((lbl, k))

    for m in MEMBERS:
        add(f"single:{m}", [m])
    for k in range(2, len(MEMBERS) + 1):
        add(f"nested:M{k}", MEMBERS[:k])
    for m in MEMBERS:
        add(f"loo:-{m}", [x for x in MEMBERS if x != m])
    out.append(("full", full))
    return out


SUBSETS = subsets()


# ------------------------------------------------------------------ one cell
def run_cell(ds, T, seed, verify):
    E.T, E.SEED = T, seed
    ytr, yte, n_cls = E.get_labels(ds)
    Z = {m: member_features(ds, T, seed, m) for m in MEMBERS}
    d = Z[MEMBERS[0]][0].shape[1]
    cpt = n_cls // T

    # Task split keyed on `seed` alone and IDENTICAL across members -- same carve as
    # exp54/55/56, which is what the exp56 assert depends on.
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
    C = {m: torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64) for m in MEMBERS}
    scat = {m: (np.zeros((d, d), np.float64), 0) for m in MEMBERS}
    A = {m: {} for m in MEMBERS}                      # one arm, so no per-arm nesting

    def _H(Zm, bs=4096):
        for i in range(0, len(Zm), bs):
            yield i, torch.relu(torch.as_tensor(Zm[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Zm, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Zm)]).cpu().numpy()

    ref56 = None
    if verify:
        if os.path.exists(EXP56):
            d56 = json.load(open(EXP56))
            want = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|{ARM}")
            hit = [v for k, v in d56.items() if k.startswith(want) and CO.order_tag() in k]
            if hit and f"FE|{ARM}|cone" in hit[0]:
                ref56 = hit[0][f"FE|{ARM}|cone"]["accs"]
        log(f"    VERIFY: exp56 FE|{ARM}|cone reference "
            f"{'loaded' if ref56 else 'NOT FOUND (skipping)'}")

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

        zLv, zLt, zSv, zSt = {}, {}, {}, {}
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

            # ---- cones, ONE set per member. Loop order and the crc32-on-arm-name RNG are
            # exp56's verbatim; changing either breaks the assert.
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
                A[m][int(c)] = X.BUILD[METHOD](
                    Xw, Fw, rays_for(ARM, len(r)), int(c), GAMMA) @ Wh_inv

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
            put(f"member:{m}|ranpac", acc_v1(zLt[m], seen, yt))

            # ---- C2: the three rules over the SAME rays, identical storage
            Qvw, Qtw = un(Ztr_m[vix] @ Wh), un(Zte_m[tei] @ Wh)
            miss = [c for c in seen if c not in A[m]]
            for r_ in RULES:
                Sv = np.full((nval, n_cls), -np.inf, np.float32)
                St = np.full((len(tei), n_cls), -np.inf, np.float32)
                for c in seen:
                    if c not in A[m]:
                        continue
                    Ac = un(A[m][c] @ Wh)
                    Sv[:, c] = score(r_, Ac, Qvw)
                    St[:, c] = score(r_, Ac, Qtw)
                zv, zt = zs(Sv, seen), zs(St, seen)
                if miss:
                    # Classes with <2 fit rows have no rays; zs sends -inf to the row MINIMUM,
                    # actively suppressing classes RanPAC calls right. Neutralise to the
                    # z-mean so they fall back to RanPAC. exp54/exp56 verbatim.
                    zv[:, miss] = 0.0
                    zt[:, miss] = 0.0
                zSv[(m, r_)], zSt[(m, r_)] = zv.astype(np.float32), zt.astype(np.float32)

        def ascend(base_v, base_t, comps, getv, gett):
            """Coordinate ascent on val -- exp56's, same N_PASS, grid and criterion."""
            w = {k: 0.0 for k in comps}
            Sv, St = base_v.copy(), base_t.copy()
            for _ in range(N_PASS):
                for k in comps:
                    bv = Sv - w[k] * getv(k)
                    b, _ = pick_beta_v2(bv, getv(k), seen, tcv, BETAS_V2)
                    St = St - w[k] * gett(k) + b * gett(k)
                    Sv = bv + b * getv(k)
                    w[k] = b
            return St, Sv, acc_margin(Sv, seen, tcv)[0]

        # ---- THE FACTORISATION: fuse each member ONCE per rule; subsets are then means.
        FT, FV, FA = {}, {}, {}
        for m in MEMBERS:
            for r_ in RULES:
                St, Sv, va = ascend(zLv[m], zLt[m], [(m, r_)],
                                    zSv.__getitem__, zSt.__getitem__)
                FT[(m, r_)], FV[(m, r_)], FA[(m, r_)] = zs(St, seen), zs(Sv, seen), va

        for lbl, ms in SUBSETS:
            put(f"{lbl}|ens_ranpac", acc_v1(sum(zLt[m] for m in ms) / len(ms), seen, yt))
            for r_ in RULES:
                bt = sum(FT[(m, r_)] for m in ms) / len(ms)
                put(f"{lbl}|FE|{r_}", acc_v1(bt, seen, yt),
                    float(np.mean([FA[(m, r_)] for m in ms])))

        if ref56 is not None:
            got, want_ = res[f"full|FE|cone"][-1], ref56[t]
            assert abs(got - want_) < 1e-9, (
                f"s{t}: full|FE|cone {got:.10f} != exp56 {want_:.10f}. This file's loop has "
                f"drifted from the one that produced the paper's headline table, so no "
                f"control below is measured against the published method. Check thread "
                f"pinning, ARM name and CO.mode().")
        line = f"    s{t}: ens {res['full|ens_ranpac'][-1]*100:.2f}"
        for r_ in RULES:
            line += f"  FE-{r_} {res[f'full|FE|{r_}'][-1]*100:.2f}"
        log(line)

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for k, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{k} out of range"
        out[k] = {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
        if k in val:
            out[f"{k}|_val"] = {"A_last": val[k][-1], "A_avg": float(np.mean(val[k]))}
    nrows = [int((ytr[FIT[t]] == c).sum()) for t in range(T) for c in tasks[t]]
    med = float(np.median(nrows))
    out["_meta"] = {"members": MEMBERS, "arm": ARM, "rules": RULES, "cpt": cpt,
                    "order": CO.mode(), "n_subsets": len(SUBSETS),
                    "fit_rows_per_class": {"min": int(np.min(nrows)), "median": med,
                                           "max": int(np.max(nrows))},
                    "arm_clipped": rays_for(ARM, med) >= med,
                    "verified_vs_exp56": ref56 is not None}
    return out


# ------------------------------------------------------------------ summary
def summarise(allres):
    W = 108
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if len(p) < 5 or p[3] != "+".join(MEMBERS) or p[4] != ARM:
            continue
        cells[(p[0], int(p[1]), int(p[2]))] = v
    dts = sorted({(a, b) for a, b, _ in cells})

    def seeds_of(ds, T):
        return sorted(s for (a, b, s) in cells if a == ds and b == T)

    def gm(ds, T, n, f):
        xs = [cells[(ds, T, s)][n][f] * 100 for s in seeds_of(ds, T) if n in cells[(ds, T, s)]]
        return (float(np.mean(xs)),
                float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")) if xs else \
               (float("nan"), float("nan"))

    print("\n" + "=" * W)
    print(f"EXP66 CONTROL BATTERY -- arm {ARM} FIXED on every cell (C3), rules {RULES} (C2), "
          f"{len(SUBSETS)} member subsets (C5)")
    print("=" * W)
    ver = [cells[c]["_meta"].get("verified_vs_exp56") for c in cells]
    print(f"\ncells {len(cells)}   verified against exp56: {sum(bool(v) for v in ver)}/{len(ver)}"
          f"   members {MEMBERS}")

    # ---------------- C2
    print(f"\n{'-'*W}\nC2 RULE CONTROL -- same rays, same storage, full member set.\n"
          f"   cone = non-negative combination | sub = SVD span (non-negativity DROPPED) | "
          f"pm = max single ray\n{'-'*W}")
    print(f"  {'ds':<11}{'T':>4}{'n':>3}{'ens_ranpac':>13}"
          + "".join(f"{'FE-'+r:>13}" for r in RULES)
          + "".join(f"{'d('+r+'-ens)':>13}" for r in RULES))
    for ds, T in dts:
        e, _ = gm(ds, T, "full|ens_ranpac", "A_last")
        vals = [gm(ds, T, f"full|FE|{r}", "A_last")[0] for r in RULES]
        print(f"  {ds:<11}{T:>4}{len(seeds_of(ds, T)):>3}{e:>13.2f}"
              + "".join(f"{v:>13.2f}" for v in vals)
              + "".join(f"{v-e:>+13.2f}" for v in vals))
    if "cone" in RULES and "sub" in RULES:
        ds_ = [gm(ds, T, "full|FE|cone", "A_last")[0] - gm(ds, T, "full|FE|sub", "A_last")[0]
               for ds, T in dts]
        ds_ = [x for x in ds_ if x == x]
        if ds_:
            print(f"\n  cone - sub, mean over {len(ds_)} cells: {np.mean(ds_):+.3f} "
                  f"(min {np.min(ds_):+.3f}, max {np.max(ds_):+.3f})")
            print("  READ: if this is inside seed noise, the gain is the SPAN of the rays, "
                  "not the conic\n        combination, and the paper should not be about cones.")

    # ---------------- C5
    print(f"\n{'-'*W}\nC5 MEMBER ABLATION -- A-Last, rule = {RULES[0]}\n{'-'*W}")
    for ds, T in dts:
        print(f"\n  {ds} T={T}   (seeds {seeds_of(ds, T)})")
        print(f"    {'subset':<16}{'ens_ranpac':>12}{'FE-'+RULES[0]:>12}{'vs full':>10}")
        fullv, _ = gm(ds, T, f"full|FE|{RULES[0]}", "A_last")
        for lbl, ms in SUBSETS:
            a, _ = gm(ds, T, f"{lbl}|FE|{RULES[0]}", "A_last")
            e, _ = gm(ds, T, f"{lbl}|ens_ranpac", "A_last")
            if a != a:
                continue
            print(f"    {lbl:<16}{e:>12.2f}{a:>12.2f}{a-fullv:>+10.2f}")

    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  C2 is the decisive one. cone/sub/pm store the SAME rays at the SAME cost; only the scoring
     rule differs. A cone-sub gap inside seed noise means the read-out is a low-rank region
     score and "conic" is decoration. That outcome is the PRIOR here, not a surprise -- see
     cone-is-dead-weight-final and nonneg-space-hypothesis-falsified.
  C3 there is exactly one arm in this file and it was fixed before the run. Any table built
     from it is free of the per-cell selection in results.txt. Cells where the arm CLIPS
     (rays_for(ARM, median) >= median) are flagged in _meta and must be reported as such.
  C5 read `nested:Mk` for saturation and `loo:-m` for attribution. If nested:M3 ~= full, the
     method is 40% cheaper than published and that is a result, not a concession. If every
     loo row is ~= full, no single member matters and the diversity story is about the
     ENSEMBLE, not the members -- which C1 (seed-only ensemble) then has to rule out.
  NONE of this addresses "is it just ensembling?" -- that is C1 and it needs new training.""")
    print("=" * W)


# ------------------------------------------------------------------ driver
if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|{ARM}|{'+'.join(RULES)}"
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
    summarise(allres)
    log(f"wrote {OUT}")
