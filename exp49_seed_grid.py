#!/usr/bin/env python3
"""exp49_seed_grid.py — the best raw cone across every dataset and seed we can run.

WHY THIS FILE EXISTS
    Every read-out number in this project is ImageNet-R T=10 SEED 0. The RanPAC bar alone
    spans 80.28 / 80.55 / 80.38 across three seeds -- a range of 0.27, LARGER than the
    0.21 deficit the method was reported to have -- and on A-Avg it spans 85.13 / 86.09 /
    84.71, a range of 1.38. At one seed neither the deficit nor most of the component
    effects are distinguishable from noise. This runs the grid.

THE 80.07 NUMBER IS AN ARTIFACT AND THIS FILE DOES NOT REPRODUCE IT
    exp41 builds every ray-mode arm against a SHARED foreign set:
        past = [A[RMODES[0]][o] for o in A[RMODES[0]] if o not in tasks[t]]
    i.e. the past-class negatives always come from the FIRST arm's cones. As a control for
    comparing budgets that is exactly right -- only the ray count varies. But k5m24 was the
    third arm in its run, so its discriminative construction consumed k5m32's stored rays
    as negatives, and a deployed k5m24 system has only k5m24's rays. exp45, which builds
    k5m24 alone against its own rays, reads 79.77.
        80.07 = k5m24 with another arm's negatives   (not deployable)
        79.77 = k5m24 with its own negatives         (deployable)
    THIS FILE IS SELF-CONSISTENT: every arm uses its own stored rays.

    THE ABOVE IS WRONG AND THE 0.3 IS THREAD NONDETERMINISM, MEASURED 2026-08-07.
    This cell was run three times pinned to one BLAS thread -- twice with VERIFY=0, once
    with VERIFY=1 -- and all three are bit-identical at every stage, A-Last 0.8005. Two
    UNPINNED runs of the same cell gave 79.78 and 80.05. Since ARMS=k5m24 alone makes
    "own" and "shared" negatives the same set by construction, the pinned 80.05 IS the
    own-negatives number, and exp45's 79.77 was an unpinned draw of it. So:
        - the deployable k5m24 value is ~80.05, not 79.77, and the IMAGENETR s0 paired
          delta is -0.23, not -0.50;
        - own-vs-shared negatives is worth ~0.02, not ~0.3. exp41's 80.07 was fine.
        - eigh/KMeans inside b_opca reduce in a thread-count-dependent order, which flips
          k-means basins. Divergence starts at s1 and compounds to 0.27 by s9.
    PIN THREADS (OMP_NUM_THREADS=1 MKL_NUM_THREADS=1) FOR ANY CELL YOU INTEND TO COMPARE.
    Unpinned, the noise floor is ~0.27 -- larger than every effect in the table but CUB's.

ON PICKING THE BEST SEED
    The request was to find a seed that beats RanPAC. That number is reported --
    `best cell` in the summary -- but it is not a result. With 4 datasets x 3 seeds the
    maximum of 12 draws is biased upward by roughly the seed spread, so a cherry-picked
    cell will not replicate on a fourth seed. The defensible statistic is the PAIRED
    per-seed delta cone - ranpac, which costs nothing extra because both arms see identical
    features and identical splits: seed noise is common to both and cancels in the pair.
    Both are printed. `wins` counts cells, `paired mean +/- sd` is the claim.

SEEDS ARE CAPPED AT THREE and that is a hard limit, not a choice. The features are LoRA-
    adapted per (dataset, T, seed) and exp16 cached seeds 0,1,2 only. A fourth seed means
    retraining the backbone 4 times (one per dataset), which is a different experiment.

ARMS
    k5m24   R_c = clip(n_c/5, 24, 128).  exp41's best ON IMAGENET-R.
    k5m8    R_c = clip(n_c/5,  8, 128).  RMIN=24 was tuned where classes have ~108 fit
            rows. CUB200 has ~27 and ImageNet-A ~27, so RMIN=24 forces k-means to ask for
            24 centroids from ~27 points -- one centroid per point, which is SPA-like
            degeneracy and the one construction this project knows is catastrophic. On
            those datasets k5m8 is the honest setting and k5m24 is expected to lose.
            `mean_rays` and `clamp_frac` are reported so the degeneracy is visible rather
            than inferred.
    f32     fixed R=32. The pre-allocation reference.

THE BAR IS READ FROM exp16, NOT RECOMPUTED
    exp39's internal RanPAC reproduces exp16's ImageNet-R T=10 s0 cell exactly (80.28), so
    the protocols match and the 10000x10000 Gram -- by far the most expensive part of a
    replay -- is pure duplication. VERIFY=1 recomputes it for the FIRST cell only and
    asserts agreement to 1e-6; if that assert fires, the bar is not comparable and nothing
    in the table means anything.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=CIFAR100,IMAGENETR,IMAGENETA,CUB200 T=10 SEED=0,1,2 python -u exp49_seed_grid.py
    DS=IMAGENETR T=10 SEED=0 VERIFY=1 python -u exp49_seed_grid.py      # bar check first
    DS=IMAGENETR T=10,20,50 SEED=0,1,2 ARMS=k5m24 python -u exp49_seed_grid.py
"""
import json
import os
import time
import zlib

import numpy as np
import torch

# exp19_dataset_hull parses T and SEED as SCALARS at import time:
#     T = int(os.environ.get("T", 10))
# so importing it with T=10,20,50 or SEED=0,1,2 in the environment raises ValueError before
# a single line of this file runs. That is what killed the four final_*.log runs. Grid
# values are captured here and the environment is narrowed to the first element before the
# import; E.T / E.SEED are reassigned per cell in run_cell anyway, so the scalar left in
# os.environ is never read again.
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
ARMS = os.environ.get("ARMS", "k5m24,k5m8").split(",")
METHOD = os.environ.get("METHOD", "opca")
GAMMA = float(os.environ.get("GAMMA", 0.5))
RMAX = int(os.environ.get("RMAX", 128))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
RMIN_DEF = int(os.environ.get("RMIN", 8))          # floor when an arm omits its mNN suffix
VERIFY = int(os.environ.get("VERIFY", 0))
# SUFFIX exists so a side run does not share an output file with a grid already in flight.
# Both processes hold the whole dict in memory and rewrite the file wholesale, so concurrent
# writers silently drop each other's cells even when their keys do not collide.
OUT = os.path.join(REPO, f"exp49_seed_grid{os.environ.get('SUFFIX', '')}_{TAG}.json")
BAR = json.load(open(os.path.join(REPO, f"exp16_full_table_{TAG}.json")))

un = X.un


def rays_for(arm, n):
    """f32 -> fixed 32.  k5m24 -> clip(n/5, 24, RMAX).  k20 -> clip(n/20, RMIN_DEF, RMAX).
    The mNN suffix is optional; without it the floor is RMIN_DEF rather than an error."""
    if arm.startswith("f"):
        return int(arm[1:])
    k, _, m = arm[1:].partition("m")
    return int(np.clip(n / float(k), int(m) if m else RMIN_DEF, RMAX))


def bar_for(ds, T, seed):
    v = BAR.get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
    assert v is not None, f"no exp16 bar for {ds} T={T} s={seed}"
    return v


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
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)

    G = C = eye = P = None
    if verify:
        P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
        G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
        C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
        eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A = {a: {} for a in ARMS}
    res = {a: [] for a in ARMS}
    nray = {a: [] for a in ARMS}
    clamp = {a: [0, 0] for a in ARMS}       # [n classes where R_c > n_rows, n classes]
    rp = []

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
                # SELF-CONSISTENT NEGATIVES: this arm's own stored rays, never another
                # arm's. See the module docstring -- this is the whole difference from
                # exp41 and it is worth ~0.3 on the headline.
                # zlib.crc32, NOT hash(): Python salts str hashes per process, so hash(a)
                # would give a different foreign subsample on every run and no cell would
                # be bit-reproducible. Measured drift from this was ~0.03 -- small, but it
                # makes "rerun and confirm" impossible, which is worse than the number.
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

        seen = np.concatenate(tasks[:t + 1])
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        Qw = un(Zte[tei] @ Wh)

        def acc(S):
            return float((np.asarray(seen)[S[:, seen].argmax(1)] == yt).mean())

        for a in ARMS:
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            tot = 0
            for c in seen:
                if c not in A[a]:
                    continue
                Ac = un(A[a][c] @ Wh)
                St[:, c] = X.cone_score(Ac, Qw)
                tot += len(Ac)
            res[a].append(acc(St))
            nray[a].append(tot / max(len(seen), 1))

        if verify:
            for i, h in _H(un(Ztr[FIT[t]])):
                h = h.double()
                Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
                Y[torch.arange(h.shape[0]),
                  torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
                G += h.T @ h; C += h.T @ Y
            nval = sum(len(v) for v in VAL[:t + 1])

            def prj(Z, Wm):
                return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()
            best, ba = -1.0, -1.0
            for lam in LAMBDAS:
                Wm = torch.linalg.solve(G + lam * eye, C)
                L = prj(un(Ztr[VAL_ALL[:nval]]), Wm)
                av = float((np.asarray(seen)[L[:, seen].argmax(1)]
                            == ytr[VAL_ALL[:nval]]).mean())
                if av > best:
                    best, ba = av, acc(prj(un(Zte[tei]), Wm))
            rp.append(ba)

        log(f"    s{t}: " + "  ".join(f"{a} {res[a][-1]*100:.2f}" for a in ARMS)
            + (f"   [ranpac {rp[-1]*100:.2f}]" if verify else ""))

    if verify:
        del G, C, P, eye
        torch.cuda.empty_cache()
        b = bar_for(ds, T, seed)
        assert abs(rp[-1] - b["A_last"]) < 1e-6, (
            f"recomputed RanPAC {rp[-1]:.6f} != exp16 bar {b['A_last']:.6f}; the replay "
            f"protocol does not match exp16 and the whole grid is incomparable")
        log("    VERIFY ok: recomputed RanPAC matches the exp16 bar exactly")

    out = {}
    for a in ARMS:
        out[a] = {"A_last": res[a][-1], "A_avg": float(np.mean(res[a])), "accs": res[a],
                  "mean_rays": float(np.mean(nray[a])),
                  "clamp_frac": clamp[a][0] / max(clamp[a][1], 1)}
    b = bar_for(ds, T, seed)
    out["ranpac"] = {"A_last": b["A_last"], "A_avg": b["A_avg"], "source": "exp16"}
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(ARMS)}|{METHOD}g{GAMMA:g}"
                       f"|R{RMAX}_f{F_MAX}_s{SHRINK:g}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 96
    print("\n" + "=" * W)
    print("EXP49 — best raw cone across datasets and seeds (self-consistent negatives)")
    print("=" * W)
    rows = []
    for key, r in allres.items():
        ds, T, seed = key.split("|")[0], int(key.split("|")[1]), int(key.split("|")[2])
        for a in [x for x in r if x != "ranpac"]:
            rows.append({"ds": ds, "T": T, "seed": seed, "arm": a,
                         "cone": r[a]["A_last"] * 100, "avg": r[a]["A_avg"] * 100,
                         "bar": r["ranpac"]["A_last"] * 100,
                         "baravg": r["ranpac"]["A_avg"] * 100,
                         "rays": r[a]["mean_rays"], "clamp": r[a]["clamp_frac"]})
    if not rows:
        print("no cells"); raise SystemExit

    print(f"\n  {'dataset':<11}{'T':>3}{'seed':>5}{'arm':>8}{'rays':>7}{'clamp':>7}"
          f"{'cone':>8}{'ranpac':>8}{'delta':>8}{'A-Avg d':>9}")
    for x in sorted(rows, key=lambda v: (v["ds"], v["T"], v["arm"], v["seed"])):
        print(f"  {x['ds']:<11}{x['T']:>3}{x['seed']:>5}{x['arm']:>8}{x['rays']:>7.1f}"
              f"{x['clamp']*100:>6.0f}%{x['cone']:>8.2f}{x['bar']:>8.2f}"
              f"{x['cone']-x['bar']:>+8.2f}{x['avg']-x['baravg']:>+9.2f}")

    print("\n  PAIRED delta (cone - ranpac), same features & splits -- seed noise cancels")
    print(f"  {'dataset':<11}{'arm':>8}{'n':>4}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}"
          f"{'wins':>7}")
    for ds in sorted({x["ds"] for x in rows}):
        for a in sorted({x["arm"] for x in rows}):
            g = [x["cone"] - x["bar"] for x in rows if x["ds"] == ds and x["arm"] == a]
            if not g:
                continue
            g = np.array(g)
            sd = f"{g.std(ddof=1):>8.2f}" if len(g) > 1 else f"{'--':>8}"
            print(f"  {ds:<11}{a:>8}{len(g):>4}{g.mean():>+9.2f}{sd}"
                  f"{g.min():>+8.2f}{g.max():>+8.2f}{int((g > 0).sum()):>4}/{len(g)}")
    allg = np.array([x["cone"] - x["bar"] for x in rows])
    print(f"  {'ALL':<11}{'':>8}{len(allg):>4}{allg.mean():>+9.2f}"
          f"{allg.std(ddof=1):>8.2f}{allg.min():>+8.2f}{allg.max():>+8.2f}"
          f"{int((allg > 0).sum()):>4}/{len(allg)}")

    best = max(rows, key=lambda v: v["cone"] - v["bar"])
    print(f"\n  best cell: {best['ds']} T={best['T']} seed {best['seed']} {best['arm']}"
          f"  cone {best['cone']:.2f} vs ranpac {best['bar']:.2f}"
          f"  ({best['cone']-best['bar']:+.2f})")
    print("  ^ this is the MAX OF %d DRAWS and is biased upward by roughly the seed"
          % len(rows))
    print("    spread. Quote the paired mean, not this cell.")
    print("\n" + "-" * W)
    print("SANITY: IMAGENETR T=10 s0 k5m24 should read ~79.77, NOT 80.07. exp41's 80.07")
    print("   used another arm's stored rays as negatives; see the module docstring.")
    print("`clamp` is the fraction of classes where R_c exceeded the class's fit rows, so")
    print("   k-means was asked for more centroids than it had points. Above ~50% the arm")
    print("   is degenerate and its number describes the clamp, not the allocation rule.")
    print("=" * W)
    print(f"wrote {OUT}")
