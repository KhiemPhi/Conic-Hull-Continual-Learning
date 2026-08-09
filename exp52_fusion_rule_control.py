#!/usr/bin/env python3
"""exp52_fusion_rule_control.py — is the fused win conic, or is it just ray ensembling?

WHY THIS FILE EXISTS
    The best number this project has is a FUSED one: exp35 `fuse_km` at R=64 on
    IMAGENETR s0, 81.03 / 85.99 against RanPAC 80.28 / 85.13 (+0.75 / +0.86). Every
    other conic asset is closed. So the only live claim is "the cone adds something to
    RanPAC", and this file tests whether the word `cone` is load-bearing in that claim.

    Two facts say it may not be.

    1. THE FUSION GAIN IS ANTI-CORRELATED WITH CONE QUALITY.  From exp35, v2 cells,
       IMAGENETR s0, standalone cone vs its own fused score:

            R=4    cone 79.80   fuse 80.80   gain +0.52
            R=16   cone 78.77   fuse 80.88   gain +0.60
            R=32   cone 77.92   fuse 81.02   gain +0.73
            R=64   cone 78.02   fuse 81.03   gain +0.75

       The standalone cone DEGRADES monotonically in R while the fusion gain GROWS.
       And exp41's oPCA cone, which is 1.9pt better standalone (79.92), fuses to 81.02
       — the same place. The fused ceiling is invariant to cone quality. That is the
       signature of ensemble decorrelation, not of a better class descriptor.

    2. NON-NEGATIVITY IS WORTH ~0.14 STANDALONE.  cone vs sub on identical rays:
       89.04 / 88.90 (exp50 base, CUB200 s0), 89.23 / 89.04 (exp48 ce e40-0). The rays
       do the work. What has never been measured is whether that 0.14 survives, grows,
       or vanishes AT THE POINT THE METHOD ACTUALLY WINS — inside the fusion.

    So: three read-out RULES over the SAME oPCA rays, each fused with RanPAC under an
    identical protocol, on the full 4-dataset x 3-seed grid.

THE THREE RULES (identical atoms, identical whitener, identical negatives)
    cone  NNLS conic score.  min_{w>=0} ||q - A^T w||, reported as the projection norm.
          The method. Non-negativity ON, combination ON.
    sub   ||q @ B||, B an orthonormal basis of span(A).  The SAME rays, the SAME linear
          combination, but the coefficients are FREE IN SIGN. This is the exact control
          for non-negativity and nothing else. It is the read-out-side twin of exp48's
          `sub` training arm.
    pm    max_j cos(q, a_j).  Naive multi-prototype: drops non-negativity AND the
          combination, so a cone is worth nothing over a plain nearest-atom rule if
          cone == pm. exp35 called this arm `pm_km`.

    cone - sub  isolates the sign constraint at fixed atoms and fixed span.
    sub  - pm   isolates the combination at fixed atoms.
    cone - pm   is the whole conic apparatus vs the cheapest thing you could do.

WHAT DECIDES IT
    ONE number: `fuse_cone - fuse_sub`, paired per (dataset, seed), reported with an sd.
        |delta| < seed sd   -> the conic constraint contributes nothing where the method
                               wins. The honest finding is "ray-set ensembling beats
                               RanPAC by +0.75" and it is not a conic result.
        delta > seed sd     -> first evidence non-negativity matters at the win, and the
                               fused arm becomes the method.
    Paired is the claim, not the raw cells. Both arms see identical features, identical
    splits, identical rays and an identical beta search, so seed noise is common to the
    pair and cancels; exp49 measured the UNPAIRED seed spread at 0.27-1.38, larger than
    every effect here.

    Issue 3 from the standing review comes free: the fusion has only ever been run at
    IMAGENETR seed 0, and this grid seeds it.

RAY BUDGETS AND THE DEGENERACY YOU MUST NOT IGNORE
    ARMS uses exp49's grammar: `f32` = fixed 32, `k5m8` = clip(n/5, 8, RMAX).
    R=32 and R=64 are the exp35 cells that produced the headline, but they were run on
    IMAGENETR, where classes have ~108 fit rows. CUB200 and IMAGENETA have ~27. Asking
    for 64 rays from 27 points is one ray per point — SPA-like degeneracy, the one
    construction this project knows is catastrophic. `clamp_frac` is reported per arm so
    that shows up in the table instead of being silently averaged into a mean.

    A PREDICTION WORTH WRITING DOWN BEFORE THE RUN: `sub` should collapse as R grows,
    because once span(A) has rank comparable to the ambient dimension ||q @ B|| -> 1 for
    every class and the rule stops discriminating. `cone` has no such failure mode — the
    non-negative orthant does not fill up. If cone - sub is ~0 at R=8 and large at R=64,
    non-negativity is real but only as a REGULARISER against over-large ray sets, which
    is a much weaker claim than "cones model classes better" and should be reported as
    such.

PIN YOUR THREADS
    eigh and KMeans inside b_opca reduce in a thread-count-dependent order. exp49 measured
    three pinned runs bit-identical and two unpinned runs of the same cell at 79.78 and
    80.05. Unpinned, the noise floor is 0.27 and swamps everything here. The usage lines
    below pin, and the file refuses to start otherwise unless you set ALLOW_UNPINNED=1.

READ IN THIS ORDER
    1. `fuse_cone - fuse_sub`, paired mean +/- sd, in the CONTRAST block. That is the file.
    2. `wins` for that contrast: 12 cells, how many are positive. 6/12 is a coin.
    3. fuse_cone - ranpac. Confirms the +0.75 replicates off IMAGENETR s0 at all.
    4. cone - sub raw. If this is +0.14 and fuse_cone - fuse_sub is 0.00, the constraint
       is real but redundant with RanPAC, which is still a negative result for the method.
    5. clamp_frac per arm before believing any CUB200 / IMAGENETA cell at f64.

COST
    The NNLS scoring dominates and it runs once per (rule, arm, stage, class) over val and
    test. Roughly exp35's cell cost x len(ARMS); `sub` and `pm` are an SVD and a matmul and
    are free next to it. Cells are written to JSON as they finish and existing keys are
    skipped, so this is interruptible — kill it and rerun the same command to resume.
    RULES=cone,sub drops `pm` and about a third of the time if you are short on box.

USAGE
    source ~/venvs/ml_env/bin/activate

    # smoke: one dataset, one seed, T=2, tiny ray budget. ~5 min. Own JSON.
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CUB200 T=2 SEED=0 ARMS=f8 SUFFIX=_smoke python -u exp52_fusion_rule_control.py

    # bar check FIRST — asserts the in-cell RanPAC reproduces exp16 exactly
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0 ARMS=f64 VERIFY=1 python -u exp52_fusion_rule_control.py

    # the headline cell: reproduce exp35's 81.03 and split it three ways
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0 ARMS=f32,f64 python -u exp52_fusion_rule_control.py

    # the grid that answers the question (12 cells; run it overnight, it resumes)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CIFAR100,IMAGENETR,IMAGENETA,CUB200 T=10 SEED=0,1,2 ARMS=f32,f64 \
      python -u exp52_fusion_rule_control.py

    # the R-sweep that tests the regulariser reading in the prediction above
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0,1,2 ARMS=f8,f16,f32,f64 python -u exp52_fusion_rule_control.py

    Run cells SEQUENTIALLY. Concurrency on this box is measured strictly worse: four
    streams doubled IMAGENETA's per-task time from 206s to 414s.
"""
import json
import os
import time
import zlib

import numpy as np
import torch

# exp19_dataset_hull parses T and SEED as SCALARS at import time, so importing it with
# T=10,20 or SEED=0,1,2 still in the environment raises ValueError before a line of this
# file runs. Capture the grids here and narrow the environment before the import; E.T and
# E.SEED are reassigned per cell in run_cell, so the scalars left behind are never read.
_DS = os.environ.get("DS", "CIFAR100,IMAGENETR,IMAGENETA,CUB200").split(",")
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
VERIFY = int(os.environ.get("VERIFY", 0))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
OUT = os.path.join(REPO, f"exp52_fusion_rule_control{os.environ.get('SUFFIX', '')}_{TAG}.json")
BAR = json.load(open(os.path.join(REPO, f"exp16_full_table_{TAG}.json")))
EPS = 1e-12

assert set(RULES) <= {"cone", "sub", "pm"}, f"unknown rule in {RULES}"
assert METHOD in X.BUILD, f"unknown method {METHOD}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}). eigh/KMeans reduce in a "
        f"thread-count-dependent order and exp49 measured the unpinned noise floor at "
        f"0.27 -- larger than every effect this file is trying to resolve. Prefix the "
        f"command with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, or set ALLOW_UNPINNED=1 if "
        f"you are only smoke-testing and do not intend to compare the numbers.")

un = X.un


def rays_for(arm, n):
    """f32 -> fixed 32.  k5m24 -> clip(n/5, 24, RMAX).  Same grammar as exp49."""
    if arm.startswith("f"):
        return int(arm[1:])
    k, _, m = arm[1:].partition("m")
    return int(np.clip(n / float(k), int(m) if m else RMIN_DEF, RMAX))


def bar_for(ds, T, seed):
    v = BAR.get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
    assert v is not None, f"no exp16 bar for {ds} T={T} s={seed}"
    return v


def zs(A, seen):
    """Row-wise z-score over the SEEN columns only, -inf mapped to the row min. Verbatim
    from exp35 so a fused number here is comparable to a fused number there."""
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def score(rule, Ac, Q):
    """All three rules take the SAME unit rays Ac (R, d) in the whitened space and the
    same unit queries Q (n, d). Only the rule differs."""
    if rule == "cone":
        return X.cone_score(Ac, Q, iters=ITERS)
    if rule == "pm":
        return (Q @ Ac.T).max(1)
    # sub: orthonormal basis of span(Ac), then the norm of the projection. Rank is taken
    # from the singular values rather than assumed to be R, because oPCA can and does
    # return near-duplicate rays once R exceeds the class's intrinsic dimension -- which
    # is exactly the regime this file is probing.
    U, s, _ = np.linalg.svd(Ac.T, full_matrices=False)
    B = U[:, s > max(s[0], 1e-12) * 1e-6]
    return np.linalg.norm(Q @ B, axis=1)


def run_cell(ds, T, seed, verify):
    E.T, E.SEED = T, seed
    assert (E.T, E.SEED) == (T, seed)
    F_ = E.adapted_features(ds)
    assert F_ is not None, f"no exp16 feature cache for {ds} T={T} s={seed}"
    Ztr, Zte = F_
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    # 10% of each task's train rows are held out as VAL. VAL picks the RanPAC lambda AND
    # the fusion beta; TEST is never used to select anything.
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
    nray = {a: [] for a in ARMS}
    clamp = {a: [0, 0] for a in ARMS}
    res = {"ranpac": []}
    for a in ARMS:
        for r_ in RULES:
            res[f"{a}|{r_}"] = []
            res[f"{a}|fuse_{r_}"] = []
    beta_log = {f"{a}|{r_}": [] for a in ARMS for r_ in RULES}

    for t in range(T):
        # ---- whitener: pooled within-class scatter over everything seen so far
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

        # ---- rays: built ONCE per arm and shared by all three rules. This is the whole
        # design: the rules cannot differ by atoms, only by how the atoms are read.
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = un(Ztr[r] @ Wh)
            for a in ARMS:
                # Self-consistent negatives, per exp49: this arm's own stored rays, never
                # another arm's. crc32 rather than hash() so the foreign subsample is
                # stable across processes and the cell is bit-reproducible.
                rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(a.encode()) % 1000)
                Fw = np.zeros((0, d), np.float32)
                if METHOD in X.DISCRIM and GAMMA > 0:
                    oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                    past = [A[a][o] for o in A[a] if o not in tasks[t]]
                    Fr = np.concatenate([Ztr[oth]] + past, 0)
                    if len(Fr) > F_MAX:
                        Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                    Fw = un(Fr @ Wh)
                Rc = rays_for(a, len(r))
                clamp[a][1] += 1
                clamp[a][0] += int(Rc > len(r))
                A[a][c] = X.BUILD[METHOD](Xw, Fw, Rc, int(c), GAMMA) @ Wh_inv

        # ---- RanPAC accumulation on this task's FIT rows
        for i, h in _H(un(Ztr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y

        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vix = VAL_ALL[:nval]
        yv = ytr[vix]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(S, y):
            return float((np.asarray(seen)[S[:, seen].argmax(1)] == y).mean())

        # ---- RanPAC head, lambda picked on VAL
        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            av = acc(logits(un(Ztr[vix]), Wm), yv)
            if av > best:
                best, bw = av, Wm
        Lv = logits(un(Ztr[vix]), bw)
        Lt = logits(un(Zte), bw)[tei]
        res["ranpac"].append(acc(zs(Lt, seen), yt))
        zLv, zLt = zs(Lv, seen), zs(Lt, seen)

        # ---- the three rules, per arm, on identical rays
        Qvw_raw, Qtw_raw = un(Ztr[vix] @ Wh), un(Zte[tei] @ Wh)
        for a in ARMS:
            miss = [c for c in seen if c not in A[a]]
            if miss:
                # A class with <2 fit rows never gets rays. zs() maps its -inf column to
                # the row MINIMUM, which in zL + b*zS actively SUPPRESSES classes RanPAC
                # would have called correctly -- a protocol artifact, not a property of
                # the rule. Neutralise to 0 (the mean of the z-distribution) in the FUSED
                # score only; the raw arms keep -inf, which is honest, because the rule
                # genuinely cannot model a class it has no atoms for.
                log(f"      s{t} {a}: {len(miss)} seen classes have NO rays -> "
                    f"neutralised in fusion, -inf raw")
            tot = 0
            for r_ in RULES:
                Sv = np.full((nval, n_cls), -np.inf, np.float32)
                St = np.full((len(tei), n_cls), -np.inf, np.float32)
                for c in seen:
                    if c not in A[a]:
                        continue
                    Ac = un(A[a][c] @ Wh)
                    Sv[:, c] = score(r_, Ac, Qvw_raw)
                    St[:, c] = score(r_, Ac, Qtw_raw)
                    if r_ == RULES[0]:
                        tot += len(Ac)
                res[f"{a}|{r_}"].append(acc(zs(St, seen), yt))
                zSv, zSt = zs(Sv, seen), zs(St, seen)
                if miss:
                    zSv[:, miss] = 0.0
                    zSt[:, miss] = 0.0
                b = max(BETAS, key=lambda bb: acc(zLv + bb * zSv, yv))
                beta_log[f"{a}|{r_}"].append(b)
                res[f"{a}|fuse_{r_}"].append(acc(zLt + b * zSt, yt))
            nray[a].append(tot / max(len(seen), 1))

        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}   " + "   ".join(
            f"{a}[" + " ".join(f"{r_} {res[f'{a}|{r_}'][-1]*100:.2f}"
                               f"/{res[f'{a}|fuse_{r_}'][-1]*100:.2f}"
                               for r_ in RULES) + "]" for a in ARMS))

    if verify:
        b = bar_for(ds, T, seed)
        assert abs(res["ranpac"][-1] - b["A_last"]) < 1e-6, (
            f"recomputed RanPAC {res['ranpac'][-1]:.6f} != exp16 bar {b['A_last']:.6f}; "
            f"the replay protocol does not match exp16 and nothing in this grid is "
            f"comparable to anything else in the project")
        log("    VERIFY ok: recomputed RanPAC matches the exp16 bar exactly")

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for k, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{k} out of range"
        out[k] = {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
    for a in ARMS:
        out[f"{a}|_rays"] = {"mean_rays": float(np.mean(nray[a])),
                             "clamp_frac": clamp[a][0] / max(clamp[a][1], 1)}
    for k, v in beta_log.items():
        out[f"{k}|_beta"] = {"betas": v, "beta_last": v[-1]}
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(ARMS)}|{'+'.join(RULES)}"
                       f"|{METHOD}g{GAMMA:g}|R{RMAX}_f{F_MAX}_s{SHRINK:g}_i{ITERS}"
                       f"|m{M_RP}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                json.dump(allres, open(OUT, "w"), indent=2)

    # ------------------------------------------------------------------ summary
    W = 100
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if len(p) < 5 or p[3] != "+".join(ARMS) or p[4] != "+".join(RULES):
            continue
        cells[(p[0], int(p[1]), int(p[2]))] = v

    def g(v, name, fld):
        return v[name][fld] * 100 if name in v else float("nan")

    print("\n" + "=" * W)
    print("EXP52 — the fusion rule control: is the fused win conic, or is it ensembling?")
    print("=" * W)
    print(f"\narms {ARMS}   rules {RULES}   method {METHOD} g={GAMMA}   "
          f"cells {len(cells)}")

    for fld, lbl in (("A_last", "A-Last"), ("A_avg", "A-Avg")):
        print(f"\n{'-'*W}\n{lbl}\n{'-'*W}")
        hdr = f"  {'ds':<10}{'seed':<5}{'ranpac':>8}"
        for a in ARMS:
            for r_ in RULES:
                hdr += f"{a+':'+r_:>12}{a+':f_'+r_:>12}"
        print(hdr)
        for (ds, T, seed), v in sorted(cells.items()):
            row = f"  {ds:<10}{seed:<5}{g(v,'ranpac',fld):>8.2f}"
            for a in ARMS:
                for r_ in RULES:
                    row += (f"{g(v,f'{a}|{r_}',fld):>12.2f}"
                            f"{g(v,f'{a}|fuse_{r_}',fld):>12.2f}")
            print(row)

        # ---- THE CONTRAST. Paired per cell, so seed noise cancels.
        print(f"\n  PAIRED DELTAS ({lbl}), mean +/- sd over {len(cells)} cells, "
              f"wins = cells > 0")
        print(f"  {'contrast':<28}{'mean':>9}{'sd':>9}{'wins':>8}   per-dataset means")
        contrasts = []
        for a in ARMS:
            for r_ in RULES:
                contrasts.append((f"{a}: fuse_{r_} - ranpac",
                                  f"{a}|fuse_{r_}", "ranpac"))
            if "cone" in RULES and "sub" in RULES:
                contrasts.append((f"{a}: fuse_cone - fuse_sub  <<<",
                                  f"{a}|fuse_cone", f"{a}|fuse_sub"))
                contrasts.append((f"{a}: cone - sub  (raw)",
                                  f"{a}|cone", f"{a}|sub"))
            if "cone" in RULES and "pm" in RULES:
                contrasts.append((f"{a}: fuse_cone - fuse_pm",
                                  f"{a}|fuse_cone", f"{a}|fuse_pm"))
                contrasts.append((f"{a}: cone - pm   (raw)",
                                  f"{a}|cone", f"{a}|pm"))
        for lbl2, hi, lo in contrasts:
            dl = {}
            for (ds, T, seed), v in cells.items():
                if hi in v and lo in v:
                    dl.setdefault(ds, []).append(g(v, hi, fld) - g(v, lo, fld))
            flat = [x for xs in dl.values() for x in xs]
            if not flat:
                continue
            sd = float(np.std(flat, ddof=1)) if len(flat) > 1 else float("nan")
            per = "  ".join(f"{d}{np.mean(x):+.2f}" for d, x in sorted(dl.items()))
            print(f"  {lbl2:<28}{np.mean(flat):>+9.2f}{sd:>9.2f}"
                  f"{sum(x>0 for x in flat):>5}/{len(flat):<3}   {per}")

    print(f"\n{'-'*W}")
    print("  rays actually built (clamp_frac = fraction of classes asked for R > n_rows;")
    print("  anything above ~0 means SPA-like degeneracy and that cell is not a ray-budget")
    print("  result, it is a one-ray-per-point result)")
    for (ds, T, seed), v in sorted(cells.items()):
        for a in ARMS:
            r = v.get(f"{a}|_rays")
            if r:
                print(f"    {ds:<10}s{seed} {a:<6} mean_rays {r['mean_rays']:>7.1f}"
                      f"   clamp_frac {r['clamp_frac']:.2f}")

    print("\n" + "-" * W)
    print("""HOW TO READ THIS
  `fuse_cone - fuse_sub` IS THE FILE. Everything else is context for it.
    |mean| < sd, or wins near half the cells  -> non-negativity contributes nothing where
      the method wins. The result is "ray-set ensembling beats RanPAC", the cone is a
      relabelling of it, and the conic framing should be dropped rather than defended.
    mean > sd with wins clustered            -> the constraint is load-bearing at the win.
      Then check whether the effect is concentrated at LARGE R: if cone-sub is ~0 at f8
      and large at f64, non-negativity is acting as a regulariser against over-large ray
      sets, not as a better model of a class, and must be reported that way.
  `fuse_cone - ranpac` replicating off IMAGENETR s0 is a PRECONDITION, not a result. If it
    does not survive the other 11 cells then exp35's +0.75 was a seed and there is nothing
    here to attribute in the first place.
  `cone - pm` is the cheap sanity floor. A conic hull that cannot beat max-cosine over its
    own rays is not doing anything a nearest-prototype rule does not already do.
  BETA: `_beta` records the fused weight chosen on VAL per stage. beta -> 0 means the
    search itself decided the cone was worthless and the fused arm silently collapsed to
    RanPAC; a fused delta of +0.00 with beta 0 is not a tie, it is an abstention.""")
    print("=" * W)
    log(f"wrote {OUT}")
