#!/usr/bin/env python3
"""exp50_crowding.py — three fixes for class crowding, and the control that says whether
any of them is about CONES.

THE DIAGNOSIS THIS IMPLEMENTS
    s_c(q) = ||Pi_C_c q||^2 is a COVERAGE score: it asks how much of q class c's rays
    explain, never how much better they explain it than the neighbours. A ridge is a joint
    least-squares over all seen classes, so it automatically ignores directions with no
    between-class variance. A per-class cone cannot -- a direction every class shares still
    contributes to every class's score.

    Measured, on the A_plus features this file reads (exp16 cache, seed 0):
        dataset     nearest-class cos   mean pairwise cos   delta slope per doubling of C
        IMAGENETA         0.791               0.438              +1.63   (r=+0.76)
        CIFAR100          0.810               0.465              +0.06   (r=+0.33)
        IMAGENETR         0.819               0.509              +0.39   (r=+0.56)
        CUB200            0.856               0.533              -0.24   (r=-0.56)
    The slope is cone - ranpac per stage regressed on log2(classes seen). Three datasets
    have the cone CATCHING UP as classes accumulate; CUB, the crowded one, falls further
    behind, monotonically, -0.12 at 20 classes to -1.05 at 200. IMAGENETA and CUB200 both
    hold ~30 rows/class and slope OPPOSITE ways, which rules out data poverty as the cause.
    n=4 datasets, so the cross-dataset correlation (-0.82) is an ordering, not a p-value;
    the arms below are tested where the effect is monotone and large, on CUB.

THE THREE ARMS
    dK  DEFLATE. Project out the top-K directions of the SEEN class-prototype matrix,
        uncentered, before building cones and before scoring. Uncentered on purpose: the
        leading uncentered direction is the global mean ("bird-ness"), which is shared and
        carries no label information, while the leading CENTERED directions are the
        between-class scatter and are the discriminative signal -- deflating those would be
        self-defeating. K therefore trades one against the other and is swept, not assumed.
        Cost: one d x K basis, shared across classes.
    lK  LOCAL NEGATIVES. Crowding is local: class c is confused with a handful of
        neighbours, not with all 199. The current construction assembles the foreign set
        and then subsamples it UNIFORMLY to F_MAX, so on CUB at stage 9 the single nearest
        confuser contributes 8 of ~1950 rows -- 0.4% of S_F. gamma*S_F is being averaged
        over 199 classes when the damage comes from 5. This spends the same budget on the
        K nearest classes by prototype cosine.
    z   CALIBRATE. Store mu_c, sd_c of cone c's score on the foreign material already
        assembled at build time, and score (s_c - mu_c)/sd_c. A cone sitting in a crowded
        region scores high on everything and is penalised for it. Two floats per class,
        zero images, computed at birth from the negatives available then.
    Arms compose: `d2+l10+z`. `base` is none of them and must reproduce exp49's k5m8 cell.

THE CONTROL, RUN IN THE SAME PASS AND NOT AFTER
    Every one of these is a DISCRIMINATIVE correction to a generative score, so each pushes
    the cone toward being a linear discriminant -- which is where this project's cone
    applications have died before. `sub` is the same rays with non-negativity dropped
    (score ||B_c^T q||, B_c an orthonormal basis of the rays). If an arm lifts `sub` by as
    much as it lifts `cone`, it is better preprocessing, not a better cone. cone - sub is
    reported per arm and it is the column that decides whether any of this is about cones.

WHAT "DISCRIMINATORY POWER" IS MEASURED BY, beyond accuracy
    margin      mean (s_true - max_{c != true} s_c), per-class z-scored so classes are
                commensurable. Accuracy is the sign of this; the magnitude says how much
                room is left before a perturbation flips the decision.
    err_to_nn   share of errors landing on the query's NEAREST-PROTOTYPE class. This is the
                crowding signature: if an arm works by decrowding, this falls. If accuracy
                moves and this does not, the arm helped for some other reason.
    nn_cos      mean nearest-class prototype cosine in the arm's own space. Direct readout
                of whether the intervention actually decrowded anything.
    shared_frac ||D^T q||^2 / ||q||^2 at K=8, and the share of the BASE cone score that
                survives projecting the query onto that shared subspace. This is the
                quantity fix 1 claims to be removing; if it is small on CUB the diagnosis
                is wrong regardless of what accuracy does.

PIN THREADS. exp49 measured a 0.27 A-Last swing on a nominally deterministic cell from
    BLAS thread count alone; pinned runs are bit-identical. Every arm here is compared
    against every other, so an unpinned run makes the whole table unreadable.
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DS=CUB200 SEED=0 \
        ARMS=base,d1,d2,d4,d8,l5,l10,l25,z python -u exp50_crowding.py      # sweep
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DS=CUB200,IMAGENETA SEED=0,1,2 \
        ARMS=base,d2,l10,z,d2+l10+z python -u exp50_crowding.py             # confirm
"""
import json
import os
import time
import zlib

import numpy as np

_DS = os.environ.get("DS", "CUB200").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
# exp19_dataset_hull parses T and SEED as scalars at import time; narrow before importing.
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
TAG = "augreg_in21k"
DSETS, TS, SEEDS = _DS, _TS, _SEEDS
ARMS = os.environ.get("ARMS", "base,d2,l10,z,d2+l10+z").split(",")
GAMMA = float(os.environ.get("GAMMA", 0.5))
KVAL = float(os.environ.get("KVAL", 5))
RMIN = int(os.environ.get("RMIN", 8))
RMAX = int(os.environ.get("RMAX", 128))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
SHARED_K = int(os.environ.get("SHARED_K", 8))      # K used for the shared_frac diagnostic
OUT = os.path.join(REPO, f"exp50_crowding{os.environ.get('SUFFIX', '')}_{TAG}.json")
BAR = json.load(open(os.path.join(REPO, f"exp16_full_table_{TAG}.json")))
un = X.un


def parse_arm(a):
    """'d2+l10+z' -> (deflate_rank, local_K, calibrate). 'base' -> (0, 0, False)."""
    defl, loc, cal = 0, 0, False
    for tok in a.split("+"):
        if tok in ("base", ""):
            continue
        if tok.startswith("d") and tok[1:].isdigit():
            defl = int(tok[1:])
        elif tok.startswith("l") and tok[1:].isdigit():
            loc = int(tok[1:])
        elif tok == "z":
            cal = True
        else:
            raise SystemExit(f"bad arm token {tok!r} in {a!r}")
    return defl, loc, cal


def rays_for(n):
    return int(np.clip(n / KVAL, RMIN, RMAX))


def deflate(Zw, D):
    """Project out the shared subspace, then renormalise. D is (d, K) orthonormal."""
    if D is None or D.shape[1] == 0:
        return un(Zw)
    return un(Zw - (Zw @ D) @ D.T)


def sub_basis(Ac):
    U, s, _ = np.linalg.svd(Ac.T, full_matrices=False)
    return U[:, s > max(s[0], 1e-12) * 1e-6]


def run_cell(ds, T, seed, arm):
    defl, loc, cal = parse_arm(arm)
    E.T, E.SEED = T, seed
    F_ = E.adapted_features(ds)
    assert F_ is not None, f"no exp16 feature cache for {ds} T={T} s={seed}"
    Ztr, Zte = F_
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT = []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        FIT.append(ix[pm[max(int(0.1 * len(ix)), 1):]])

    scatter = np.zeros((d, d), np.float64)
    n_scat = 0
    A, PROTO, CAL = {}, {}, {}          # rays / raw prototype / (mu,sd) per class per head
    acc_cone, acc_sub = [], []
    diag = {}

    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc
            n_scat += len(Xc)
        S_ = scatter / max(n_scat, 1)
        S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)

        # raw prototypes are born with their class; one vector each, same storage class as
        # the rays the method already keeps.
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) >= 2:
                PROTO[int(c)] = un(Ztr[r]).mean(0)

        # shared subspace from the prototypes SEEN SO FAR, in the CURRENT whitened space.
        # Uncentered SVD: V[:, :K] are the directions the seen classes have in common.
        pk = sorted(PROTO)
        Pw = un(np.stack([PROTO[c] for c in pk]) @ Wh)
        D = None
        if defl:
            D = np.linalg.svd(Pw, full_matrices=False)[2][:defl].T.astype(np.float32)
        Dsh = np.linalg.svd(Pw, full_matrices=False)[2][:SHARED_K].T.astype(np.float32)
        Pd = deflate(Pw, D)                        # prototypes in the arm's own space
        pidx = {c: i for i, c in enumerate(pk)}

        # ONE foreign draw shared by every arm: arms must differ by the intervention, not
        # by which negatives they happened to sample. crc32("k5m8") reproduces exp49's
        # base-arm stream so `base` is comparable to that table.
        rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(b"k5m8") % 1000)

        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = deflate(un(Ztr[r] @ Wh), D)
            cur = [o for o in tasks[t] if int(o) != int(c) and int(o) in PROTO]
            past = [o for o in pk if o not in [int(v) for v in tasks[t]]]
            if loc:
                # nearest classes by prototype cosine, in the arm's own space
                cand = [int(o) for o in cur] + list(past)
                cos = Pd[[pidx[o] for o in cand]] @ Pd[pidx[int(c)]]
                cand = [cand[i] for i in np.argsort(-cos)[:loc]]
                cs, ps = set(cand) & {int(o) for o in cur}, [o for o in cand if o in past]
            else:
                cs, ps = {int(o) for o in cur}, past
            blocks = []
            if cs:
                oth = FIT[t][np.isin(ytr[FIT[t]], list(cs))]
                if len(oth):
                    blocks.append(Ztr[oth])
            blocks += [A[o] for o in ps]
            Fr = np.concatenate(blocks, 0) if blocks else np.zeros((0, d), np.float32)
            if len(Fr) > F_MAX:
                Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
            Fw = deflate(un(Fr @ Wh), D) if len(Fr) else np.zeros((0, d), np.float32)

            rays = X.BUILD["opca"](Xw, Fw, rays_for(len(r)), int(c), GAMMA)
            A[int(c)] = rays @ Wh_inv
            if cal and len(Fw):
                sc = X.cone_score(rays, Fw)
                B = sub_basis(un(rays))
                ss = np.linalg.norm(Fw @ B, axis=1)
                CAL[int(c)] = (float(sc.mean()), float(sc.std() + 1e-9),
                               float(ss.mean()), float(ss.std() + 1e-9))

        seen = np.asarray([c for c in np.concatenate(tasks[:t + 1]) if int(c) in A])
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        Qw = deflate(un(Zte[tei] @ Wh), D)
        Sc = np.full((len(tei), n_cls), -np.inf, np.float32)
        Ss = np.full((len(tei), n_cls), -np.inf, np.float32)
        for c in seen:
            Ac = deflate(un(A[int(c)] @ Wh), D)
            sc = X.cone_score(Ac, Qw)
            ss = np.linalg.norm(Qw @ sub_basis(Ac), axis=1)
            if cal and int(c) in CAL:
                m1, s1, m2, s2 = CAL[int(c)]
                sc, ss = (sc - m1) / s1, (ss - m2) / s2
            Sc[:, int(c)], Ss[:, int(c)] = sc, ss
        acc_cone.append(float((seen[Sc[:, seen].argmax(1)] == yt).mean()))
        acc_sub.append(float((seen[Ss[:, seen].argmax(1)] == yt).mean()))

        if t == T - 1:
            Sv = Sc[:, seen]
            zs = (Sv - Sv.mean(1, keepdims=True)) / (Sv.std(1, keepdims=True) + 1e-12)
            col = {int(c): i for i, c in enumerate(seen)}
            tc = np.array([col[int(y)] for y in yt])
            rows = np.arange(len(yt))
            own = zs[rows, tc].copy()
            zo = zs.copy()
            zo[rows, tc] = -np.inf
            best = zo.max(1)
            pred = seen[Sv.argmax(1)]
            # nearest OTHER class to each query, by prototype cosine in the arm's space
            nn = seen[np.argsort(-(Qw @ Pd[[pidx[int(c)] for c in seen]].T), 1)[:, :2]]
            nn_other = np.where(nn[:, 0] == yt, nn[:, 1], nn[:, 0])
            err = pred != yt
            Pn = Pd[[pidx[int(c)] for c in seen]]
            G = Pn @ Pn.T
            np.fill_diagonal(G, -1.0)
            qs = un(Zte[tei] @ Wh)                 # undeflated, for the shared diagnostic
            diag = {
                "margin": float((own - best).mean()),
                "err_to_nn": float((pred[err] == nn_other[err]).mean()) if err.any() else 0.0,
                "nn_cos": float(G.max(1).mean()),
                "shared_frac": float((np.linalg.norm(qs @ Dsh, axis=1) ** 2).mean()),
                "n_err": int(err.sum()),
            }

    b = BAR.get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
    return {"cone_A_last": acc_cone[-1], "cone_A_avg": float(np.mean(acc_cone)),
            "cone_accs": acc_cone,
            "sub_A_last": acc_sub[-1], "sub_A_avg": float(np.mean(acc_sub)),
            "ranpac_A_last": b["A_last"] if b else None,
            "ranpac_A_avg": b["A_avg"] if b else None,
            "n_classes_with_cone": len(A), **diag}


if __name__ == "__main__":
    if os.environ.get("OMP_NUM_THREADS") != "1":
        log("WARNING: threads not pinned -- exp49 measured 0.27 A-Last of drift from this")
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                for arm in ARMS:
                    key = (f"{ds}|{T}|{seed}|{arm}|opca_g{GAMMA:g}_k{KVAL:g}m{RMIN}"
                           f"|f{F_MAX}_s{SHRINK:g}|v1")
                    if key in allres:
                        log(f"skip {key}")
                        continue
                    log(f"=== {key}")
                    allres[key] = run_cell(ds, T, seed, arm)
                    r = allres[key]
                    log(f"    cone {r['cone_A_last']*100:.2f}  sub {r['sub_A_last']*100:.2f}"
                        f"  ranpac {(r['ranpac_A_last'] or 0)*100:.2f}"
                        f"  margin {r['margin']:+.3f}  err_to_nn {r['err_to_nn']*100:.1f}%"
                        f"  nn_cos {r['nn_cos']:.3f}")
                    json.dump(allres, open(OUT, "w"), indent=2)

    W = 104
    print("\n" + "=" * W)
    print("EXP50 — crowding fixes: deflate / local negatives / calibrate")
    print("=" * W)
    print(f"\n  {'dataset':<11}{'seed':>5}{'arm':>12}{'cone':>8}{'sub':>8}{'ranpac':>8}"
          f"{'c-rp':>8}{'c-sub':>8}{'vs base':>9}{'margin':>8}{'err_nn':>8}{'nn_cos':>8}")
    base = {}
    for k, r in allres.items():
        p = k.split("|")
        if p[3] == "base":
            base[(p[0], p[2])] = r["cone_A_last"]
    for k, r in allres.items():
        p = k.split("|")
        ds, seed, arm = p[0], p[2], p[3]
        rp = r["ranpac_A_last"] or 0
        vb = r["cone_A_last"] - base.get((ds, seed), float("nan"))
        print(f"  {ds:<11}{seed:>5}{arm:>12}{r['cone_A_last']*100:>8.2f}"
              f"{r['sub_A_last']*100:>8.2f}{rp*100:>8.2f}"
              f"{(r['cone_A_last']-rp)*100:>+8.2f}"
              f"{(r['cone_A_last']-r['sub_A_last'])*100:>+8.2f}{vb*100:>+9.2f}"
              f"{r['margin']:>+8.3f}{r['err_to_nn']*100:>7.1f}%{r['nn_cos']:>8.3f}")
    print("\n  `vs base` is the fix's effect. `c-sub` is whether it is about CONES: an arm")
    print("     that lifts cone and sub equally is better preprocessing, not a better cone.")
    print("  `err_nn` is the share of errors landing on the nearest class -- the crowding")
    print("     signature. An arm that raises accuracy without lowering it did not decrowd.")
    print("=" * W)
    log(f"wrote {OUT}")
