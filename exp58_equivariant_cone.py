#!/usr/bin/env python3
"""exp58_equivariant_cone.py -- is the whitened conic read-out INVARIANT to linear feature drift?

THE CLAIM, DERIVED NOT ASSUMED
    Features are row vectors and the whitener is applied on the right: x -> x Wh, with
    Wh = chol(Sigma^-1) and Sigma the (shrunk) within-class scatter. Let a task update
    transform features by any invertible linear map, phi -> phi Mt. Then

        Sigma_B = Mt^T Sigma_A Mt
        Wh_B    = chol(Sigma_B^-1)
        (Mt Wh_B)(Mt Wh_B)^T = Mt Sigma_B^-1 Mt^T = Sigma_A^-1 = Wh_A Wh_A^T
        =>  Mt Wh_B = Wh_A Q        for some ORTHOGONAL Q.

    So a query transported and re-whitened is phi_A Mt Wh_B = phi_A Wh_A Q, and a stored ray
    transported and re-whitened is A_A Mt Wh_B = A_A Wh_A Q -- THE SAME Q. The cone score is
    cos(q_hat, w* A_c) and the NNLS constraint is w >= 0, both invariant under a common
    rotation. The whitened conic read-out is therefore EXACTLY invariant to invertible linear
    drift. Nothing needs to be re-fit; the coordinate frame moves with the cones.

WHY THIS IS NOT THE FOUR TRANSPORT RESULTS THAT ARE ALREADY CLOSED
    Forward prototype transport, transport-to-phi0, and per-stage frames all APPLIED an
    estimated map to the prototypes and therefore needed the estimate to be accurate --
    forward transport needed feature error <= 0.01 and measured 0.15, and died there.
    Here the estimate enters BOTH the rays and the whitener, and the algebra above shows the
    two errors cancel to whatever extent the residual is orthogonal. Writing M_hat = M E, the
    surviving distortion is E, and

        IF E IS ORTHOGONAL THE SCORE IS UNCHANGED NO MATTER HOW LARGE E IS.

    So the quantity that matters is not ||M_hat - M|| but the ORTHOGONALITY DEFECT of
    M_hat^-1 M. This file measures that alongside accuracy, because it is the whole reason
    the fidelity wall might not apply.

ARMS
    native       cones built in phi_B, whitened by Sigma_B.   Upper bound; needs old data.
    invar_true   cones from phi_A, transported by the TRUE Mt, whitener from Mt^T Sigma_A Mt.
                 Tests the ALGEBRA. Must equal `native` to float precision when drift is
                 exactly linear; if it does not, the derivation or the code is wrong.
    invar_est    same but with M_hat fit by least squares on TASK-0 ROWS ONLY -- replay-free,
                 the deployable version. This is the arm the design lives or dies on.
    stale        cones and whitener both left in phi_A, scored on phi_B. The do-nothing
                 control: what forgetting costs if drift is ignored.

TWO DRIFT SOURCES
    synthetic    phi_B = phi_A Mt + eps * (random nonlinear residual, relative norm eps).
                 eps=0 validates the algebra exactly; sweeping eps gives the TOLERANCE CURVE
                 -- how much non-linearity the invariance survives.
    member       phi_A = q32, phi_B = another cached member. A real change of feature space,
                 though a PESSIMISTIC proxy: two independently trained networks are further
                 from linearly related than one network before/after a task update. Prior work
                 here measured PEFT drift as approximately affine, so real drift should sit
                 easier than this arm. Passing here is sufficient, failing is inconclusive.

SCOPE
    Task-0 classes only, one seed, cached features, no training. This is a pre-flight for a
    per-task-LoRA architecture, not the architecture. It answers exactly one question: does
    the whitened cone survive a change of feature space without being rebuilt?

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot DS=IMAGENETR T=10 SEED=0 \
      python -u exp58_equivariant_cone.py
"""
import json
import os
import warnings

import numpy as np

os.environ.setdefault("T", "10")
os.environ.setdefault("SEED", "0")
os.environ.setdefault("ARMS", "f64")
os.environ.setdefault("RULES", "cone")
warnings.filterwarnings("ignore", message=r"Number of distinct clusters.*",
                        category=UserWarning, module=r"sklearn\..*")

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402
import exp54_stack as S54                   # noqa: E402
import class_order as CO                    # noqa: E402

un = X.un
REPO = os.path.dirname(os.path.abspath(__file__))
TAG = S54.TAG
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
R = int(os.environ.get("R", 64))
SHRINK = S54.SHRINK
GAMMA = S54.GAMMA
A_MEM = os.environ.get("A_MEM", "q32")
B_MEMS = os.environ.get("B_MEMS", "q64,a16,m32").split(",")
EPS = [float(x) for x in os.environ.get("EPS", "0,0.05,0.1,0.2,0.4,0.8").split(",")]
RIDGE = float(os.environ.get("RIDGE", 1e-3))


def feats(spec):
    ot = CO.order_tag()
    if spec == "q32":
        f = os.path.join(REPO,
                         f"exp16_feats_{DS}_T{T}_s{SEED}_ep40_lr0.0003_aug1{ot}_{TAG}.npz")
    else:
        f = os.path.join(REPO,
                         f"exp55_feats_{DS}_T{T}_s{SEED}_{spec}_ep40_lr0.0003{ot}_{TAG}.npz")
    assert os.path.exists(f), f"missing cache {f}"
    z = np.load(f)
    return un(z["Ftr"]).astype(np.float64), un(z["Fte"]).astype(np.float64)


def scatter(Ztr, ytr, rows, classes):
    """Shrunk within-class scatter, exactly as the method builds it."""
    d = Ztr.shape[1]
    sc, ns = np.zeros((d, d)), 0
    for c in classes:
        r = rows[ytr[rows] == c]
        if len(r) < 2:
            continue
        Xc = Ztr[r] - Ztr[r].mean(0)
        sc += Xc.T @ Xc; ns += len(Xc)
    S = sc / max(ns, 1)
    return S + SHRINK * np.trace(S) / d * np.eye(d)


def whiten(S):
    return np.linalg.cholesky(np.linalg.inv(S))


def build_cones(Z, ytr, rows, classes, Wh):
    """Cones in FEATURE space (the method stores A @ Wh_inv). Rotation-equivariant by
    construction: oPCA eigenvectors and k-means centroids both rotate with the frame."""
    Wh32 = Wh.astype(np.float32)
    Wi = np.linalg.inv(Wh32)
    out = {}
    for c in classes:
        r = rows[ytr[rows] == c]
        if len(r) < 2:
            continue
        Xw = un(Z[r].astype(np.float32) @ Wh32)
        oth = rows[~np.isin(ytr[rows], [c])]
        Fr = Z[oth].astype(np.float32)
        if len(Fr) > S54.F_MAX:
            Fr = Fr[np.random.default_rng(int(c)).choice(len(Fr), S54.F_MAX, replace=False)]
        out[int(c)] = X.BUILD[S54.METHOD](Xw, un(Fr), R, int(c), GAMMA) @ Wi
    return out


def score_acc(cones, Wh, Zte, yte_rows, yte, classes):
    """Cone-only classification accuracy over `classes` on the given test rows."""
    Wh32 = Wh.astype(np.float32)
    Q = un(Zte[yte_rows].astype(np.float32) @ Wh32)
    cls = sorted(cones)
    Smat = np.full((len(yte_rows), len(cls)), -np.inf, np.float32)
    for j, c in enumerate(cls):
        Smat[:, j] = S54.score("cone", un(cones[c].astype(np.float32) @ Wh32), Q)
    pred = np.asarray(cls)[Smat.argmax(1)]
    return float((pred == yte[yte_rows]).mean())


def orth_defect(Mhat, Mt):
    """||E^T E - I||_F / sqrt(d) for E = Mhat^-1 Mt, normalised so 0 = perfectly orthogonal.
    THIS is the quantity the invariance depends on, not ||Mhat - Mt||."""
    Emat = np.linalg.solve(Mhat, Mt)
    Emat = Emat / (np.linalg.norm(Emat, 2) + 1e-12)
    d = Emat.shape[0]
    return float(np.linalg.norm(Emat.T @ Emat - np.eye(d) / 1.0 * (np.trace(Emat.T @ Emat) / d),
                                "fro") / np.sqrt(d))


def fit_map(Za, Zb, rows):
    """Least-squares Mt with phi_B ~= phi_A @ Mt, fit on `rows` ONLY (replay-free)."""
    A, B = Za[rows], Zb[rows]
    G = A.T @ A + RIDGE * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ B)


def run(Za_tr, Za_te, Zb_tr, Zb_te, ytr, yte, fit_rows, te_rows, classes, Mt_true=None):
    Sa = scatter(Za_tr, ytr, fit_rows, classes)
    Wa = whiten(Sa)
    cones_A = build_cones(Za_tr, ytr, fit_rows, classes, Wa)

    res = {}
    # upper bound: rebuild everything in B (needs old data -- not deployable)
    Sb = scatter(Zb_tr, ytr, fit_rows, classes)
    Wb = whiten(Sb)
    res["native"] = score_acc(build_cones(Zb_tr, ytr, fit_rows, classes, Wb),
                              Wb, Zb_te, te_rows, yte, classes)
    # do-nothing control
    res["stale"] = score_acc(cones_A, Wa, Zb_te, te_rows, yte, classes)
    # invariance with the TRUE map (algebra check)
    if Mt_true is not None:
        Wb_t = whiten(Mt_true.T @ Sa @ Mt_true)
        res["invar_true"] = score_acc({c: v @ Mt_true for c, v in cones_A.items()},
                                      Wb_t, Zb_te, te_rows, yte, classes)
    # invariance with the estimated map (deployable)
    Mh = fit_map(Za_tr, Zb_tr, fit_rows)
    Wb_h = whiten(Mh.T @ Sa @ Mh)
    res["invar_est"] = score_acc({c: v @ Mh for c, v in cones_A.items()},
                                 Wb_h, Zb_te, te_rows, yte, classes)
    res["_source_A"] = score_acc(cones_A, Wa, Za_te, te_rows, yte, classes)
    # how well does the linear map explain the drift, and how orthogonal is the residual?
    pred = Za_tr[fit_rows] @ Mh
    res["_fit_relerr"] = float(np.linalg.norm(pred - Zb_tr[fit_rows]) /
                               (np.linalg.norm(Zb_tr[fit_rows]) + 1e-12))
    if Mt_true is not None:
        res["_orth_defect"] = orth_defect(Mh, Mt_true)
    return res


if __name__ == "__main__":
    E.T, E.SEED = T, SEED
    ytr, yte, n_cls = E.get_labels(DS)
    cpt = n_cls // T
    order = CO.class_order(n_cls, SEED)
    # 20 task-0 classes put every arm at 94-95%, leaving no dynamic range to separate them.
    # NCLS controls how many classes the cones cover; default is ALL of them, which is not a
    # CIL stage but is a far more discriminative test of the invariance property.
    ncls_use = int(os.environ.get("NCLS", n_cls))
    task0 = order[:ncls_use]
    ix = np.where(np.isin(ytr, task0))[0]
    pm = np.random.default_rng(0).permutation(len(ix))
    fit_rows = ix[pm[max(int(0.1 * len(ix)), 1):]]
    te_rows = np.where(np.isin(yte, task0))[0]
    _cap = int(os.environ.get("MAX_TEST", 1500))
    if len(te_rows) > _cap:
        te_rows = te_rows[np.random.default_rng(7).permutation(len(te_rows))[:_cap]]

    Za_tr, Za_te = feats(A_MEM)
    d = Za_tr.shape[1]
    W = 92
    print("=" * W)
    print(f"EXP58 -- is the whitened cone invariant to linear feature drift?")
    print(f"  {DS} T={T} seed={SEED} ORDER={CO.mode()}  classes={len(task0)}  "
          f"R={R}  d={d}")
    print(f"  fit rows {len(fit_rows)}   test rows {len(te_rows)}   A={A_MEM}")
    print("=" * W)

    print("\n[1] SYNTHETIC DRIFT  phi_B = phi_A @ Mt + eps * nonlinear residual")
    print("    eps=0 MUST reproduce `native` exactly -- that is the algebra check.\n")
    rng = np.random.default_rng(0)
    Mt = rng.normal(size=(d, d)) / np.sqrt(d) + 0.5 * np.eye(d)     # well-conditioned
    print(f"    cond(Mt) = {np.linalg.cond(Mt):.1f}")
    # A random ReLU layer. The previous version used tanh on a pre-activation with |z|~0.03,
    # where tanh is linear to 0.2% -- so the "nonlinear" sweep was adding a LINEAR component,
    # which invar_est absorbed (flat accuracy, flat fit error) and invar_true could not (it
    # was handed only Mt). Both behaviours were artefacts. ReLU on a scaled pre-activation is
    # genuinely nonlinear and cannot be absorbed by a linear fit.
    V1 = rng.normal(size=(d, d)) / np.sqrt(d)
    V2 = rng.normal(size=(d, d)) / np.sqrt(d)
    def nlmap(Z):
        return np.maximum(Z @ V1, 0.0) @ V2
    print(f"    {'eps':>6}{'native':>9}{'invar_true':>12}{'invar_est':>11}{'stale':>9}"
          f"{'fit relerr':>12}{'orth defect':>13}")
    rows_out = []
    for eps in EPS:
        base_tr, base_te = Za_tr @ Mt, Za_te @ Mt
        if eps > 0:
            nl_tr, nl_te = nlmap(Za_tr), nlmap(Za_te)
            s = np.linalg.norm(base_tr) / (np.linalg.norm(nl_tr) + 1e-12)
            Zb_tr, Zb_te = base_tr + eps * s * nl_tr, base_te + eps * s * nl_te
        else:
            Zb_tr, Zb_te = base_tr, base_te
        Zb_tr, Zb_te = un(Zb_tr), un(Zb_te)
        r = run(Za_tr, Za_te, Zb_tr, Zb_te, ytr, yte, fit_rows, te_rows, task0, Mt)
        rows_out.append((eps, r))
        print(f"    {eps:>6.2f}{r['native']*100:>9.2f}{r['invar_true']*100:>12.2f}"
              f"{r['invar_est']*100:>11.2f}{r['stale']*100:>9.2f}"
              f"{r['_fit_relerr']:>12.3f}{r['_orth_defect']:>13.3f}")
    a0 = rows_out[0][1]
    # The invariant is invar_true == the SOURCE-space score, not == native. `native` rebuilds
    # the cones from Zb, and Zb is row-renormalised by un(), which is a projective rather than
    # linear map -- so native legitimately differs. Comparing against it was the wrong check.
    ok = abs(a0["invar_true"] - a0["_source_A"]) < 1e-6
    print(f"\n    ALGEBRA CHECK at eps=0: invar_true {a0['invar_true']*100:.4f} vs source-A "
          f"{a0['_source_A']*100:.4f}  -> "
          f"{'EXACT (invariance confirmed)' if ok else 'MISMATCH -- derivation or code is wrong'}")

    print("\n[2] REAL MEMBER PAIRS  (pessimistic proxy: independently trained networks)\n")
    print(f"    {'A -> B':>14}{'source A':>10}{'native':>9}{'invar_est':>11}{'stale':>9}"
          f"{'fit relerr':>12}{'invar-stale':>13}")
    out = {}
    for b in B_MEMS:
        try:
            Zb_tr, Zb_te = feats(b)
        except AssertionError as e:
            print(f"    {A_MEM+' -> '+b:>14}   skipped ({e})")
            continue
        r = run(Za_tr, Za_te, Zb_tr, Zb_te, ytr, yte, fit_rows, te_rows, task0)
        out[f"{A_MEM}->{b}"] = r
        print(f"    {A_MEM+' -> '+b:>14}{r['_source_A']*100:>10.2f}{r['native']*100:>9.2f}"
              f"{r['invar_est']*100:>11.2f}{r['stale']*100:>9.2f}{r['_fit_relerr']:>12.3f}"
              f"{(r['invar_est']-r['stale'])*100:>+13.2f}")

    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  1. eps=0 must give invar_true == native EXACTLY. If not, stop -- the derivation or the
     implementation is wrong and nothing below means anything.
  2. `invar_est` vs `native` is the real question: how much does using a REPLAY-FREE estimated
     map cost against rebuilding the cones with old data you are not allowed to have?
  3. `invar_est` vs `stale` is the value of the mechanism. If it is ~0, the whitened cone was
     already robust to this drift and no transport is needed at all -- which would be an even
     better result for the architecture, and a cheaper one.
  4. `orth defect` is the quantity the theory says matters. Watch it against `fit relerr`:
     prior work here died at fit error 0.15 against a 0.01 requirement, so if accuracy holds
     while fit relerr is large and orth defect is small, THAT is the finding -- the fidelity
     wall does not bind on a whitened conic read-out.
  5. The member-pair arm is pessimistic. Two separately trained LoRAs are further apart than
     one network before/after a task update, and prior work measured PEFT drift as roughly
     affine. Passing here is sufficient; failing here is not decisive.""")
    print("=" * W)
    json.dump({"synthetic": [{"eps": e, **{k: v for k, v in r.items()}} for e, r in rows_out],
               "members": out},
              open(os.path.join(REPO, f"exp58_equivariant_cone_{DS}_{TAG}.json"), "w"),
              indent=2)
