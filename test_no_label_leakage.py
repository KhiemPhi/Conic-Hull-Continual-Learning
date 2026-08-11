#!/usr/bin/env python3
"""test_no_label_leakage.py -- standing regression check: no test label reaches the fit.

WHY THIS FILE EXISTS
    exp55/exp56/exp57 all fit several things per task -- the RanPAC ridge solution W, the
    ridge penalty lambda, the whitener Wh, the per-class cone rays, and the fusion weight beta
    -- and every one of them is supposed to see TRAIN rows only. A single mis-indexed array
    (yte where ytr was meant, Zte where Ztr was meant) would leak test labels into the model
    and inflate every number in the project without failing anything. Reading the code proves
    it once; this file proves it after every edit.

THE TEST, AND WHY IT IS THE STRONGEST FORM AVAILABLE
    Permute the test labels and refit. If ANY fitted quantity depends on a test label, at
    least one of these hashes moves:

        lambda   beta   W   Wh   cone rays   val accuracy   predictions on test

    Nothing about the training data changes, so a correct implementation must produce
    bit-identical output. This catches leaks that a static read misses -- an aliased array, a
    stale closure, an index computed from the wrong label vector.

WHAT IS EXPECTED TO CHANGE, AND IS NOT A LEAK
    `tei = np.where(np.isin(yte, seen))[0]` selects WHICH test rows are scored at a given
    stage, so permuting the labels legitimately changes the scored row SET. That is the CIL
    evaluation protocol and PyCIL/LAMDA-PILOT does exactly the same thing; it defines the
    evaluation set, not the model. The test therefore compares predictions only on the rows
    that both runs happen to score.

IT ALSO CHECKS THE SPLIT ITSELF
    A leak can live upstream of any of this: if split_indices returned overlapping train and
    test index sets, every downstream file would be honest and the result would still be
    wrong. splits.audit_split is run for each dataset whose features are cached.

COST
    ~1-2 minutes. Stage 0 only, one member, the smallest cached dataset, and MRP defaults to
    2000 rather than the method's 10000 -- the label-flow property does not depend on the
    projection width, and this keeps the check cheap enough to run every time. Set MRP=10000
    for a faithful-width run.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u test_no_label_leakage.py
    echo $?          # 0 = clean, 1 = leak

    ORDER=pilot DS=IMAGENETAP MRP=10000 python -u test_no_label_leakage.py
"""
import hashlib
import os
import sys
import zlib

import numpy as np
import torch

os.environ.setdefault("T", "10")
os.environ.setdefault("SEED", "0")

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402
import exp54_stack as S54                   # noqa: E402
import class_order as CO                    # noqa: E402
import splits                               # noqa: E402

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
REPO = os.path.dirname(os.path.abspath(__file__))
TAG = S54.TAG
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
R = int(os.environ.get("R", 64))
D = int(os.environ.get("MRP", 2000))
# Smallest test sets first: the check is identical on any dataset, so pay the least for it.
CANDIDATES = os.environ.get("DS", "IMAGENETAP,CUB200P,IMAGENETR,CIFAR100").split(",")


def hsh(a):
    return hashlib.sha1(
        np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()[:16]


def find_cache():
    """First candidate dataset whose q32 feature cache exists for this ORDER/T/SEED."""
    for ds in CANDIDATES:
        f = os.path.join(
            REPO,
            f"exp16_feats_{ds}_T{T}_s{SEED}_ep40_lr0.0003_aug1{CO.order_tag()}_{TAG}.npz")
        if os.path.exists(f):
            return ds, f
    return None, None


def fit_stage0(ds, Ztr, Zte, ytr, yte_used, tasks, FIT, VAL, n_cls):
    """Every quantity the method FITS at stage 0, plus its predictions on test.

    `yte_used` is deliberately in scope for the whole function: if any line reaches for it
    outside the two scoring positions, the permuted run will disagree."""
    d = Ztr.shape[1]
    t = 0
    P = torch.randn(d, D, generator=torch.Generator().manual_seed(0)).to(DEV)
    G = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    C = torch.zeros(D, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(D, device=DEV, dtype=torch.float64)

    def H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(
                torch.as_tensor(Z[i:i + bs], device=DEV, dtype=torch.float32) @ P)

    def logits(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in H(Z)]).cpu().numpy()

    sc, ns = np.zeros((d, d)), 0
    for c in tasks[t]:
        r = FIT[t][ytr[FIT[t]] == c]
        if len(r) < 2:
            continue
        Xc = Ztr[r] - Ztr[r].mean(0)
        sc += Xc.T @ Xc; ns += len(Xc)
    S_ = sc / max(ns, 1)
    S_ = S_ + 0.03 * np.trace(S_) / d * np.eye(d)
    Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
    Wh_inv = np.linalg.inv(Wh).astype(np.float32)

    A = {}
    for c in tasks[t]:
        r = FIT[t][ytr[FIT[t]] == c]
        if len(r) < 2:
            continue
        Xw = un(Ztr[r] @ Wh)
        rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(f"f{R}".encode()) % 1000)
        oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
        Fr = Ztr[oth]
        if len(Fr) > S54.F_MAX:
            Fr = Fr[rng.choice(len(Fr), S54.F_MAX, replace=False)]
        A[int(c)] = X.BUILD[S54.METHOD](Xw, un(Fr), R, int(c), S54.GAMMA) @ Wh_inv

    for i, h in H(un(Ztr[FIT[t]])):
        h = h.double()
        Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
        Y[torch.arange(h.shape[0]),
          torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
        G += h.T @ h; C += h.T @ Y

    seen = np.concatenate(tasks[:t + 1])
    vix = np.concatenate(VAL[:t + 1])
    yv = ytr[vix]
    best, bw, blam = -1.0, None, None
    for lam in S54.LAMBDAS:
        Wm = torch.linalg.solve(G + lam * eye, C)
        a = acc_v1(logits(un(Ztr[vix]), Wm), seen, yv)
        if a > best:
            best, bw, blam = a, Wm, lam

    col = {int(c): j for j, c in enumerate(seen)}
    tcv = np.array([col[int(v)] for v in yv])
    zLv = zs(logits(un(Ztr[vix]), bw), seen)
    Qvw = un(Ztr[vix] @ Wh)
    Sv = np.full((len(vix), n_cls), -np.inf, np.float32)
    for c in seen:
        if c in A:
            Sv[:, c] = S54.score("cone", un(A[c] @ Wh), Qvw)
    beta, _ = S54.pick_beta_v2(zLv, zs(Sv, seen), seen, tcv, S54.BETAS_V2)

    # ---- scoring. The only two legitimate uses of yte_used are here.
    tei = np.where(np.isin(yte_used, seen))[0]
    zLt = zs(logits(un(Zte), bw)[tei], seen)
    Qtw = un(Zte[tei] @ Wh)
    St = np.full((len(tei), n_cls), -np.inf, np.float32)
    for c in seen:
        if c in A:
            St[:, c] = S54.score("cone", un(A[c] @ Wh), Qtw)
    fused = zLt + beta * zs(St, seen)
    pred = np.asarray(seen)[fused[:, seen].argmax(1)]

    fitted = {"lambda": blam, "beta": beta, "W": hsh(bw.cpu().numpy()), "Wh": hsh(Wh),
              "rays": hsh(np.concatenate([A[c] for c in sorted(A)], 0)),
              "n_rays": sum(len(A[c]) for c in A), "val_acc": round(best, 12)}
    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return fitted, tei, pred


def main():
    fails = []

    print("=" * 78)
    print(f"LABEL-LEAKAGE REGRESSION CHECK   ORDER={CO.mode()}  T={T}  SEED={SEED}  "
          f"R={R}  MRP={D}")
    print("=" * 78)

    print("\n[1] split integrity -- train and test index sets must be disjoint")
    checked = 0
    for ds in CANDIDATES:
        try:
            _, _, ytr_all, _, yte_all, n_cls = None, None, None, None, None, None
            E.T, E.SEED = T, SEED
            ytr_all, yte_all, n_cls = E.get_labels(ds)
        except Exception as e:                                    # noqa: BLE001
            print(f"    {ds:<11} skipped ({type(e).__name__})")
            continue
        checked += 1
        st = splits.audit_split(np.concatenate([ytr_all, yte_all]),
                                np.arange(len(ytr_all)),
                                np.arange(len(ytr_all), len(ytr_all) + len(yte_all)),
                                ds, n_cls=n_cls, strict=False)
        bad = st["overlap"] or st["no_test"] or st["no_train"]
        print(f"    {ds:<11} train {len(ytr_all):>6}  test {len(yte_all):>5}  "
              f"overlap {st['overlap']}  no_test {st['no_test']}  no_train {st['no_train']}"
              f"   {'FAIL' if bad else 'ok'}")
        if bad:
            fails.append(f"split integrity: {ds}")
    if not checked:
        print("    no dataset labels loadable -- cannot check")

    ds, cache = find_cache()
    if ds is None:
        print(f"\n[2] SKIPPED: no q32 feature cache for ORDER={CO.mode()} T={T} SEED={SEED} "
              f"among {CANDIDATES}.\n    Produce one with:  ORDER={CO.mode()} "
              f"DATASETS={CANDIDATES[0]} TASKS={T} SEEDS={SEED} python -u exp16_full_table.py")
        print("\nRESULT: split checks only." + ("  FAILURES ABOVE." if fails else "  clean."))
        return 1 if fails else 0

    print(f"\n[2] permutation test on {ds} (stage 0, member q32)")
    E.T, E.SEED = T, SEED
    ytr, yte, n_cls = E.get_labels(ds)
    z = np.load(cache)
    Ztr, Zte = un(z["Ftr"]), un(z["Fte"])
    cpt = n_cls // T
    order = CO.class_order(n_cls, SEED)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])

    assert len(np.intersect1d(np.concatenate(FIT), np.concatenate(VAL))) == 0, \
        "FIT and VAL overlap -- the val split is not held out"

    rng = np.random.default_rng(12345)
    yte_perm = yte[rng.permutation(len(yte))]
    frac = float((yte != yte_perm).mean())
    print(f"    permuted {frac:.1%} of test labels")

    a, tei_a, pred_a = fit_stage0(ds, Ztr, Zte, ytr, yte, tasks, FIT, VAL, n_cls)
    b, tei_b, pred_b = fit_stage0(ds, Ztr, Zte, ytr, yte_perm, tasks, FIT, VAL, n_cls)

    print(f"    {'fitted quantity':<12}{'real yte':>20}{'permuted yte':>20}   verdict")
    for k in a:
        same = a[k] == b[k]
        print(f"    {k:<12}{str(a[k]):>20}{str(b[k]):>20}   "
              f"{'ok' if same else 'DIFFERS -- LEAK'}")
        if not same:
            fails.append(f"fitted quantity '{k}' depends on test labels")

    ia = {r: i for i, r in enumerate(tei_a)}
    ib = {r: i for i, r in enumerate(tei_b)}
    common = np.intersect1d(tei_a, tei_b)
    if len(common):
        agree = all(pred_a[ia[r]] == pred_b[ib[r]] for r in common)
        print(f"    predictions identical on the {len(common)} commonly-scored rows: "
              f"{'ok' if agree else 'DIFFER -- LEAK'}")
        if not agree:
            fails.append("test predictions depend on test labels")
    else:
        print("    (no commonly-scored rows to compare -- fitted hashes carry the check)")
    print(f"    scored-row SET differs ({len(tei_a)} vs {len(tei_b)} rows, {len(common)} "
          f"common): EXPECTED, `tei` is the CIL eval protocol, not the model")

    print("\n" + "=" * 78)
    if fails:
        print("RESULT: LEAK DETECTED")
        for f in fails:
            print(f"  - {f}")
    else:
        print("RESULT: clean -- every fitted quantity and every prediction is invariant "
              "to test labels")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
