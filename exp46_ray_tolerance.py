#!/usr/bin/env python3
"""exp46_ray_tolerance.py — GATE 0. How much ray error can a cone absorb, vs a prototype?

WHY THIS IS THE FIRST GATE
    The whole "train features for the full 10 tasks" plan rests on one unproven claim:
    that a cone tolerates more transport error than a prototype does. The repo already
    knows the prototype side of this. [forward-prototype-transport-fidelity-wall] measured
    that a phi_0 -> phi_t linear map cuts feature error 4x but still lands at 0.15 when the
    prototype needs <= 0.01 (gamma/2) to be safe, and [transport-to-phi0-is-dominated]
    closed the other direction. Transport lost for prototypes by 15x.

    If a cone also needs 0.01 the plan is dead on arrival and no amount of feature training
    will save it, because every stored cone goes stale the instant the backbone moves.
    If a cone tolerates 0.05-0.15 the transport path is open. This file measures the number.

WHY A CONE MIGHT TOLERATE MORE, stated so it can be falsified
    A prototype's score is cos(q, mu_c): the score is a coordinate of the stored object, so
    ANY positional error passes straight into the score at first order.
    A cone's score is ||Pi_C q||, a function of the cone as a SET. Two invariances follow:
        - positive rescaling of any generator changes nothing
        - moving a generator anywhere INSIDE the existing cone changes nothing, because the
          conic hull is unchanged
    So a cone should be first-order insensitive to the component of the perturbation that
    lies inside its own span, and only pay for the component that leaves it.
    THAT IS THE MECHANISM, and `inspan` vs `outspan` is the arm that tests it directly. If
    inspan and outspan cost the same, the invariance argument is wrong and the tolerance
    (whatever its level) is not coming from conic geometry.

THE FOUR PERTURBATIONS
    iso      V <- un(V + eps * g),  g ~ N(0, I/d).  The honest default: unstructured error.
    inspan   the same noise PROJECTED ONTO span(V_c). Predicted nearly free for the cone.
    outspan  the same noise projected OFF span(V_c). Predicted to carry the whole cost.
    rot      ONE random rotation applied coherently to every class's stored rays, queries
             left alone. This is what drift actually looks like --
             [drift-is-benign-gauge-rotation] found PEFT drift is a coherent rotation --
             and it is the only arm that resembles a real staleness event rather than iid
             noise. A coherent rotation is the WORST case for an angular reader (it moves
             every class the same way, so no error cancels in the argmax) and the cheapest
             to correct (one Procrustes), which is exactly why it is separated out.

MATCHED PERTURBATION IS THE WHOLE EXPERIMENT
    The prototype comparators are perturbed by the SAME relative magnitude and the measured
    mean cos(original, perturbed) is reported per arm, so the x-axis is a physical feature
    error and not a hyperparameter. Comparing a cone at eps=0.3 against a prototype at
    eps=0.1 would manufacture any conclusion you like.

    Three readers, identical perturbation, identical atoms:
        cone    ||Pi_C q||           the object under test
        maxcos  max_r <q, v_r>       the k=1 member of the cone's own nesting
        ncm     <q, mu_c>            one vector per class, perturbed the same way

    Note ncm has ONE vector and the others have R_c, so its noise does not average down the
    same way. That is a property of the primitive, not a confound: it is precisely why a
    prototype is fragile. It is flagged rather than corrected.

OUTPUT THAT DECIDES THE GATE
    eps_1pt: the perturbation at which A-Last falls 1.00 point, by linear interpolation on
    the measured curve, expressed in mean-cosine-displacement units so it can be compared
    against the 0.15 the repo already measured for a fitted linear transport map.
        cone eps_1pt / ncm eps_1pt  >= 3   -> transport path is open, go to exp48
        ratio ~ 1                          -> cones are no better, drop the transport arm

NO TRAINING, NO FUSION. Final-stage scoring only (exp40-style): the whitener and the
    generators still accumulate over all T stages, nothing is scored until the last.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp46_ray_tolerance.py
    DS=IMAGENETR T=10 SEED=0 KINDS=iso,rot EPS=0,0.05,0.1,0.2 python -u exp46_ray_tolerance.py
"""
import json
import os
import time

import numpy as np
import torch

import exp19_dataset_hull as E
import exp39_cone_construction as X

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
TAG = "augreg_in21k"
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
METHOD = os.environ.get("METHOD", "opca")
GAMMA = float(os.environ.get("GAMMA", 0.5))
KVAL = float(os.environ.get("KVAL", 5))        # R_c = clip(n_c/KVAL, RMIN, RMAX)
RMIN = int(os.environ.get("RMIN", 24))
RMAX = int(os.environ.get("RMAX", 128))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
KINDS = os.environ.get("KINDS", "iso,inspan,outspan,rot").split(",")
EPS = [float(x) for x in os.environ.get("EPS", "0,0.05,0.1,0.2,0.4,0.8").split(",")]
NREP = int(os.environ.get("NREP", 3))          # perturbation draws averaged per (kind, eps)
OUT = os.path.join(REPO, f"exp46_ray_tolerance_{TAG}.json")

un = X.un
E.T, E.SEED = T, SEED
assert (E.T, E.SEED) == (T, SEED)
Ztr, Zte = E.adapted_features(DS)
ytr, yte, n_cls = E.get_labels(DS)
d = Ztr.shape[1]
cpt = n_cls // T
order = np.random.default_rng(SEED).permutation(n_cls)
tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

FIT = []
for t in range(T):
    ix = np.where(np.isin(ytr, tasks[t]))[0]
    pm = np.random.default_rng(t).permutation(len(ix))
    FIT.append(ix[pm[max(int(0.1 * len(ix)), 1):]])


# ---------------------------------------------------------------- state
def build():
    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A, MU = {}, {}
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
            Xw = un(Ztr[r] @ Wh)
            Fw = np.zeros((0, d), np.float32)
            if METHOD in X.DISCRIM and GAMMA > 0:
                oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                past = [A[o] for o in A if o not in tasks[t]]
                Fr = np.concatenate([Ztr[oth]] + past, 0)
                if len(Fr) > F_MAX:
                    Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                Fw = un(Fr @ Wh)
            Rc = int(np.clip(len(r) / KVAL, RMIN, RMAX))
            A[c] = X.BUILD[METHOD](Xw, Fw, Rc, int(c), GAMMA) @ Wh_inv
            MU[c] = un(Xw.mean(0, keepdims=True)) @ Wh_inv
        log(f"  stage {t} accumulated")
    return Wh, A, MU


# ---------------------------------------------------------------- perturbations
def perturb(V, kind, eps, rng, Rglob):
    """V: (R, d) rays in the ORIGINAL space. Returns perturbed UNIT rows.

    THE ROWS ARRIVE NON-UNIT. Generators are built unit in the whitened space and stored as
    `un(...) @ Wh_inv`, so in the original space their norms are ~0.03. Perturbing those
    with a unit-norm g makes eps=0.05 a ~2x-the-signal kick rather than a 5% one, and makes
    the reported displacement meaningless (it read 0.999 at eps=0, which is just 1-||v||^2).
    The score only ever sees `un(V @ Wh)`, so rays are scale-free: normalise first and the
    eps axis becomes an honest relative perturbation.

    `rot` uses the SAME rotation for every class (passed in as Rglob) -- that coherence is
    the point of the arm, so it must not be redrawn per class."""
    V = un(V)
    if eps == 0:
        return V
    if kind == "rot":
        Vr = V @ Rglob
        return un(V + eps * (Vr - V) / (np.linalg.norm(Vr - V, axis=1, keepdims=True) + 1e-12))
    g = rng.standard_normal(V.shape).astype(np.float32) / np.sqrt(V.shape[1])
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    if kind in ("inspan", "outspan"):
        # orthonormal basis of span(V); SVD not QR -- numpy's reduced QR pads rank-deficient
        # inputs with directions OUTSIDE the span, which would put `inspan` noise partly
        # outside the span and destroy the very contrast this arm exists to make.
        U, s, _ = np.linalg.svd(V.T, full_matrices=False)
        B = U[:, s > max(s[0], 1e-12) * 1e-6]
        gp = (g @ B) @ B.T
        g = gp if kind == "inspan" else g - gp
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    elif kind != "iso":
        raise ValueError(kind)
    return un(V + eps * g)


def run():
    Wh, A, MU = build()
    seen = np.asarray(sorted(A))
    Qw = un(Zte @ Wh)
    n = len(yte)
    rngR = np.random.default_rng(7)
    Q_, _ = np.linalg.qr(rngR.standard_normal((d, d)))
    Rglob = Q_.astype(np.float32)                       # one shared rotation for `rot`

    def score_all(Ad, MUd):
        S_c = np.full((n, n_cls), -np.inf, np.float32)
        S_m = np.full((n, n_cls), -np.inf, np.float32)
        S_n = np.full((n, n_cls), -np.inf, np.float32)
        for c in seen:
            Ac = un(Ad[c] @ Wh)
            S_c[:, c] = X.cone_score(Ac, Qw)
            S_m[:, c] = (Qw @ Ac.T).max(1)
            S_n[:, c] = Qw @ un(MUd[c] @ Wh)[0]
        f = lambda S: float((seen[S[:, seen].argmax(1)] == yte).mean())  # noqa: E731
        return f(S_c), f(S_m), f(S_n)

    res = {}
    for kind in KINDS:
        for eps in EPS:
            accs, cosr, cosm = [], [], []
            for rep in range(1 if eps == 0 else NREP):
                rng = np.random.default_rng(1000 * rep + 17)
                Ad, MUd = {}, {}
                for c in seen:
                    Ad[c] = perturb(A[c], kind, eps, rng, Rglob)
                    MUd[c] = perturb(MU[c], kind, eps, rng, Rglob)
                    cosr.append(float((un(A[c]) * Ad[c]).sum(1).mean()))
                    cosm.append(float((un(MU[c]) * MUd[c]).sum(1).mean()))
                accs.append(score_all(Ad, MUd))
            a = np.array(accs)
            res[f"{kind}|{eps:g}"] = {
                "eps": eps, "kind": kind,
                "cone": float(a[:, 0].mean()), "maxcos": float(a[:, 1].mean()),
                "ncm": float(a[:, 2].mean()),
                "cone_sd": float(a[:, 0].std()),
                "disp_ray": 1.0 - float(np.mean(cosr)),      # mean cosine displacement
                "disp_mu": 1.0 - float(np.mean(cosm)),
            }
            r = res[f"{kind}|{eps:g}"]
            log(f"  {kind:8s} eps {eps:<5g} disp {r['disp_ray']:.4f} | "
                f"cone {r['cone']*100:.2f}  maxcos {r['maxcos']*100:.2f}  ncm {r['ncm']*100:.2f}")
    res["_rays"] = float(np.mean([len(A[c]) for c in seen]))
    return res


def eps_1pt(rows, key):
    """Displacement at which `key` drops 1.00 point from its eps=0 value, by linear
    interpolation between the two bracketing measurements. None if never reached."""
    rows = sorted(rows, key=lambda r: r["disp_ray"])
    base = rows[0][key] * 100
    for a, b in zip(rows, rows[1:]):
        ya, yb = a[key] * 100, b[key] * 100
        if yb <= base - 1.0 <= ya:
            f = (ya - (base - 1.0)) / max(ya - yb, 1e-9)
            return a["disp_ray"] + f * (b["disp_ray"] - a["disp_ray"])
    return None


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    key = (f"{DS}|{T}|{SEED}|{METHOD}g{GAMMA:g}_k{KVAL:g}m{RMIN}"
           f"|{'+'.join(KINDS)}|n{NREP}|v1")
    if key not in allres:
        log(f"=== {key}")
        allres[key] = run()
        json.dump(allres, open(OUT, "w"), indent=2)
    else:
        log(f"skip {key}")

    W = 84
    print("\n" + "=" * W)
    print("EXP46 — ray-perturbation tolerance: cone vs prototype at MATCHED feature error")
    print("=" * W)
    for k, r in sorted(allres.items()):
        print(f"\n--- {k}   ({r.get('_rays', 0):.1f} rays/class)")
        for kind in KINDS:
            rows = [v for kk, v in r.items() if kk.startswith(f"{kind}|")]
            if not rows:
                continue
            rows.sort(key=lambda v: v["eps"])
            print(f"\n  {kind}")
            print(f"    {'eps':>6}{'disp':>8}{'cone':>9}{'maxcos':>9}{'ncm':>9}")
            for v in rows:
                print(f"    {v['eps']:>6g}{v['disp_ray']:>8.4f}{v['cone']*100:>9.2f}"
                      f"{v['maxcos']*100:>9.2f}{v['ncm']*100:>9.2f}")
            e = {n: eps_1pt(rows, n) for n in ("cone", "maxcos", "ncm")}
            s = "  ".join(f"{n} {('%.4f' % e[n]) if e[n] else '>max'}" for n in e)
            rat = (e["cone"] / e["ncm"]) if (e["cone"] and e["ncm"]) else None
            print(f"    eps_1pt   {s}" + (f"   |  cone/ncm {rat:.2f}x" if rat else ""))
    print("\n" + "-" * W)
    print("GATE: cone eps_1pt / ncm eps_1pt >= 3 on `iso` and `rot` -> the transport arm of")
    print("   exp48 is worth building. Compare eps_1pt against 0.15, the forward-transport")
    print("   error the repo already measured; if cone eps_1pt < 0.15 transport alone cannot")
    print("   carry a cone either and exp48 must rely on the loss, not the map.")
    print("MECHANISM: inspan vs outspan. The invariance argument predicts inspan is nearly")
    print("   free for `cone` and NOT for `ncm` (a single vector has no span to hide in).")
    print("   If inspan == outspan for the cone, the tolerance is not conic in origin and")
    print("   the eps_1pt ratio is measuring ray averaging, which multiproto gets too --")
    print("   read `maxcos` to separate those: it has the same R vectors and no conic rule.")
    print("=" * W)
    print(f"wrote {OUT}")
