#!/usr/bin/env python3
"""exp29_subspace_overlap.py — is class-subspace OVERLAP the binding constraint?

THE CHAIN THIS TESTS
    exp23 measured that the discriminative signal is radial: the out-of-span residual has
    AUROC 0.964 while the orthant coordinate adds zero incremental value. If every descriptor
    is really reading "how much energy does q have in class c's subspace", then:

        classification  ==  which class subspace is q nearest to
        and that is fixed by HOW SEPARATED THE CLASS SUBSPACES ARE IN phi.

    No descriptor can separate subspaces that overlap. That would explain why eleven
    geometric objects, five dictionaries and three combiners all plateau: they are
    re-parameterisations of one quantity that phi has already fixed.

MEASUREMENT
    Per class c, take the top-k right singular vectors B_c of its (unit, uncentred) rows.
    For a pair (c, c') the squared cosines of the principal angles are the squared singular
    values of B_c B_c'^T, so

        affinity(c, c') = ||B_c B_c'^T||_F^2 / k   in [0, 1]

    1 = identical subspaces, 0 = orthogonal. Per class we report the mean and the max over
    all other classes (the worst confuser).

WHAT WOULD CONFIRM THE CHAIN
    H1  overlap predicts error        spearman(overlap_c, err_c) strongly positive
    H2  adaptation works by decorrelating   A_plus has lower overlap than frozen, AND
                                            delta-overlap predicts delta-error per class
    Either failing breaks the chain and the "read-out is saturated because phi fixed it"
    story does not hold.

CONTROLS (without these H1 is unattributable)
    n_c          per-class sample count -- small classes are badly estimated, not overlapping
    spread_c     within-class dispersion (1 - mean cosine to the class mean)
    energy_c     mean ||P_c q|| for the class's own held-out rows -- the radial quantity
                 itself. If energy alone predicts error as well as overlap does, overlap is
                 just restating it.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR python -u exp29_subspace_overlap.py
    DS=IMAGENETR,CUB200,CIFAR100,IMAGENETA KS=4,16,32 python -u exp29_subspace_overlap.py
"""
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

import exp19_dataset_hull as E

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
KS = [int(x) for x in os.environ.get("KS", "4,16,32").split(",")]
T_CIL = int(os.environ.get("T", "10"))
SEED = int(os.environ.get("SEED", "0"))
M_RP = int(os.environ.get("MRP", 10000))
LAM = float(os.environ.get("LAM", 1e3))
OUT = os.path.join(REPO, "exp29_subspace_overlap.json")


def basis(X, k):
    """Top-k right singular vectors of the unit, UNCENTRED rows -- uncentred because the
    signal is directional and a cone/subspace here is anchored at the origin."""
    k = int(min(k, min(X.shape)))
    return np.linalg.svd(np.asarray(X, np.float64), full_matrices=False)[2][:k]


def affinity_matrix(Bs, k):
    """A[c,c'] = ||B_c B_c'^T||_F^2 / k, the mean squared cosine of the principal angles."""
    n = len(Bs)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = float(np.sum((Bs[i] @ Bs[j].T) ** 2) / k)
            A[i, j] = A[j, i] = v
    return A


def ranpac_err(Ztr, ytr, Zte, yte, n_cls):
    """Per-class error under the real head, fitted on everything (the A-Last situation)."""
    P = torch.randn(Ztr.shape[1], M_RP,
                    generator=torch.Generator().manual_seed(0)).to(DEV)

    def H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.tensor(X[i:i + bs], device=DEV,
                                             dtype=torch.float32) @ P).double()
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    for i, h in H(Ztr):
        Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
        Y[torch.arange(h.shape[0]), torch.tensor(ytr[i:i + h.shape[0]], device=DEV)] = 1.0
        G += h.T @ h
        C += h.T @ Y
    W = torch.linalg.solve(G + LAM * torch.eye(M_RP, device=DEV, dtype=torch.float64), C)
    pred = torch.cat([h @ W for _, h in H(Zte)]).argmax(1).cpu().numpy()
    del G, C, P
    torch.cuda.empty_cache()
    return np.array([1.0 - (pred[yte == c] == c).mean() if (yte == c).any() else np.nan
                     for c in range(n_cls)]), float((pred == yte).mean())


def run(ds):
    ytr, yte, n_cls = E.get_labels(ds)
    E.T, E.SEED = T_CIL, SEED
    spaces = {}
    fa = E.adapted_features(ds)
    assert fa is not None, f"missing exp16 cache for {ds} T={T_CIL} s={SEED}"
    spaces["aplus"] = fa
    fp = os.path.join(REPO, f"exp19_frozen_{ds}_{TAG}.npz")
    if os.path.exists(fp):
        z = np.load(fp)
        spaces["frozen"] = (E.un(z["Ftr"]), E.un(z["Fte"]))
    log(f"{ds}: {n_cls} classes, spaces {list(spaces)}")

    out = {}
    per = {}
    for name, (Ztr, Zte) in spaces.items():
        err, acc = ranpac_err(Ztr, ytr, Zte, yte, n_cls)
        mu = E.un(np.stack([Ztr[ytr == c].mean(0) for c in range(n_cls)]))
        spread = np.array([1.0 - float((E.un(Ztr[ytr == c]) @ mu[c]).mean())
                           for c in range(n_cls)])
        cnt = np.array([int((ytr == c).sum()) for c in range(n_cls)])
        log(f"  {name}: RanPAC acc {acc*100:.2f}")
        for k in KS:
            Bs = [basis(Ztr[ytr == c], k) for c in range(n_cls)]
            A = affinity_matrix(Bs, k)
            np.fill_diagonal(A, np.nan)
            ov_mean = np.nanmean(A, 1)
            ov_max = np.nanmax(A, 1)
            # the radial quantity itself: own-subspace energy of the class's TEST rows
            energy = np.array([float(np.linalg.norm(E.un(Zte[yte == c]) @ Bs[c].T,
                                                    axis=1).mean())
                               if (yte == c).any() else np.nan for c in range(n_cls)])
            ok = np.isfinite(err) & np.isfinite(ov_mean)
            out[f"{name}|k{k}"] = dict(
                acc=acc,
                affinity_mean=float(np.nanmean(A)), affinity_p95=float(np.nanpercentile(A, 95)),
                rho_ovmean_err=float(spearmanr(ov_mean[ok], err[ok]).statistic),
                rho_ovmax_err=float(spearmanr(ov_max[ok], err[ok]).statistic),
                rho_energy_err=float(spearmanr(energy[ok], err[ok]).statistic),
                rho_spread_err=float(spearmanr(spread[ok], err[ok]).statistic),
                rho_count_err=float(spearmanr(cnt[ok], err[ok]).statistic))
            per[f"{name}|k{k}"] = dict(ov_mean=ov_mean.tolist(), ov_max=ov_max.tolist(),
                                       err=err.tolist(), energy=energy.tolist())
            o = out[f"{name}|k{k}"]
            log(f"    k={k:<3d} affinity {o['affinity_mean']:.4f} (p95 {o['affinity_p95']:.4f})"
                f"  rho(ov,err) {o['rho_ovmean_err']:+.3f}  rho(max,err) {o['rho_ovmax_err']:+.3f}"
                f"  | controls energy {o['rho_energy_err']:+.3f} spread"
                f" {o['rho_spread_err']:+.3f} n {o['rho_count_err']:+.3f}")

    # ---- H2: does adaptation decorrelate, and does that explain the per-class gain? ----
    if "frozen" in spaces:
        for k in KS:
            f, a = per[f"frozen|k{k}"], per[f"aplus|k{k}"]
            d_ov = np.array(a["ov_mean"]) - np.array(f["ov_mean"])
            d_err = np.array(a["err"]) - np.array(f["err"])
            ok = np.isfinite(d_ov) & np.isfinite(d_err)
            out[f"delta|k{k}"] = dict(
                d_affinity=float(np.nanmean(d_ov)),
                d_err=float(np.nanmean(d_err)),
                rho_dov_derr=float(spearmanr(d_ov[ok], d_err[ok]).statistic))
            o = out[f"delta|k{k}"]
            log(f"  A_plus - frozen  k={k:<3d} d_affinity {o['d_affinity']:+.4f}  "
                f"d_err {o['d_err']*100:+.2f}pt  rho(d_ov, d_err) {o['rho_dov_derr']:+.3f}")
    return out


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    allres.setdefault(ds, {}).update(run(ds))
    json.dump(allres, open(OUT, "w"), indent=2)

W = 104
print("\n" + "=" * W)
print("EXP29 — class-subspace overlap as the binding constraint")
print("=" * W)
for ds, res in allres.items():
    print(f"\n--- {ds}")
    print(f"{'space':>8}{'k':>4}{'RanPAC':>9}{'affinity':>10}{'p95':>8}"
          f"{'rho(ov,err)':>13}{'rho(max,err)':>14} | {'energy':>8}{'spread':>8}{'n':>7}")
    for key in [k for k in res if not k.startswith("delta")]:
        sp, kk = key.split("|")
        o = res[key]
        print(f"{sp:>8}{kk[1:]:>4}{o['acc']*100:>9.2f}{o['affinity_mean']:>10.4f}"
              f"{o['affinity_p95']:>8.4f}{o['rho_ovmean_err']:>13.3f}"
              f"{o['rho_ovmax_err']:>14.3f} | {o['rho_energy_err']:>8.3f}"
              f"{o['rho_spread_err']:>8.3f}{o['rho_count_err']:>7.3f}")
    for key in [k for k in res if k.startswith("delta")]:
        o = res[key]
        print(f"  A_plus-frozen k={key[6:]:>3}: d_affinity {o['d_affinity']:+.4f}  "
              f"d_err {o['d_err']*100:+.2f}pt  rho(d_ov,d_err) {o['rho_dov_derr']:+.3f}")
print("\n" + "-" * W)
print("H1  rho(ov,err) strongly positive  =>  overlap predicts which classes fail.")
print("H2  d_affinity < 0 AND rho(d_ov,d_err) > 0  =>  adaptation works by decorrelating,")
print("    and the per-class gain tracks the per-class decorrelation.")
print("CONTROLS: if `energy`, `spread` or `n` predict error as well as overlap does, then")
print("    overlap is restating sample size or dispersion and H1 is unattributable.")
print("=" * W)
print(f"wrote {OUT}")
