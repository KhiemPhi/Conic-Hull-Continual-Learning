#!/usr/bin/env python3
"""exp19_dataset_hull.py — ONE big conic hull per DATASET: can it tell datasets apart?

THE QUESTION
    Every prior cone test in this repo fitted a hull per CLASS (24 rays, ~24-450 rows).
    This one goes the other way: fit ONE hull with LOTS of rays over an ENTIRE dataset's
    exp16 feature cache, and ask whether the hull describes "the set of things this dataset
    contains" better than the obvious set-description baselines -- multi-prototype, kNN,
    k-means -- at a MATCHED storage budget.

WHY BUDGET-MATCHING IS THE WHOLE EXPERIMENT
    [cone-diagnostic-instrument-fails] and [cone-is-dead-weight-final] both died the same
    way: the cone's apparent gain was capacity, not conic structure. A 256-ray hull stores
    256 vectors; so must k-means (256 centroids), kNN (256 exemplars) and multi-proto (256
    prototypes). Anything that beats the baselines only at unmatched R is a capacity result
    and is reported as such.

    Two further controls decide whether "conic" means anything here:
      subspace  R-dim SVD basis of the same rows, score = cos(q, P_R q). Identical
                reconstruction-fidelity score with the non-negativity DROPPED. If cone
                ~= subspace, the answer is "reconstruction", not "cones".
      mean      1 vector. The floor. If everything ties this, the task is saturated.
      maha      OFF-BUDGET reference (768x768 shrunk covariance, ~2300x the budget).
                Included because Mahalanobis is the standard strong OOD baseline and
                [zero-image-losses-work-but-converge-below-bar] found it strong here.

THE BACKBONE CONFOUND (and the control for it)
    exp16 features are A_plus: the backbone was LoRA-adapted on task 0 of ITS OWN dataset,
    so CIFAR features and CUB features come from DIFFERENT networks. Cross-dataset
    separation measured on them is partly "which LoRA produced this vector", which is not
    the question. SPACE=frozen re-extracts all four datasets through the SHARED pretrained
    ViT-B/16 (no LoRA, cached), which removes the confound entirely. Both spaces are run;
    if they disagree, only the frozen numbers are about dataset geometry.

THREE PROTOCOLS (the third is the one that can actually discriminate)
    P1 dataset-ID   fit on each dataset's train, argmax over the 4 descriptor scores on a
                    class-balanced pooled test set. Scores are z-calibrated on a held-out
                    slice of each dataset's OWN train, because a tighter hull on a bigger
                    dataset would otherwise win the argmax for free. Chance = 25%.
    P2 far-OOD      per ordered pair (A ID vs B OOD), AUROC of A's descriptor. Expected to
                    saturate near 1.0 for everything -- reported so that saturation is
                    visible rather than assumed.
    P3 near-OOD     THE INFORMATIVE CELL. Within one dataset, fit on the first half of the
                    classes only; ID = test rows of seen classes, OOD = test rows of the
                    UNSEEN half. Same dataset, same backbone, same statistics -- the only
                    difference is semantic coverage, which is exactly what a hull claims to
                    model. P2 saturating and P3 not is the expected shape.

READING THE OUTPUT
    A win only counts if it holds (a) at matched R, (b) over `subspace`, and (c) in the
    frozen space. Anything else is capacity, reconstruction, or the LoRA fingerprint.

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp19_dataset_hull.py                            # both spaces, full grid
    SPACE=adapted python -u exp19_dataset_hull.py              # no GPU featurisation
    SPACE=frozen BUDGETS=8,64,256 python -u exp19_dataset_hull.py
    DS=CIFAR100,IMAGENETR T=10 SEED=0 N_FIT=8000 python -u exp19_dataset_hull.py
"""
import json
import os
import time

import numpy as np
import torch

from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
TAG = MODEL.split(".")[-1]
DSETS = os.environ.get("DS", "CIFAR100,IMAGENETR,IMAGENETA,CUB200").split(",")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
SPACES = os.environ.get("SPACE", "adapted,frozen").split(",")
BUDGETS = [int(b) for b in os.environ.get("BUDGETS", "8,24,64,128,256").split(",")]
N_FIT = int(os.environ.get("N_FIT", 8000))       # rows subsampled for fitting (SPA is O(m*N*D))
CAL_FRAC = float(os.environ.get("CAL_FRAC", 0.1))
SPLIT_SEED = 1993
RNG_SEED = 0
REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, f"exp19_dataset_hull_{TAG}.json")

METHODS = ["cone_spa", "cone_km", "cone_km_cov", "subspace", "facet", "facet_soft",
           "facet_sum", "kmeans", "multiproto", "knn", "mean", "maha"]
Q_EXT = float(os.environ.get("Q_EXT", 0.01))     # robust extent quantile for the facet box
TAU = float(os.environ.get("TAU", 8.0))          # softmin temperature for facet_soft
WANT = [m for m in os.environ.get("METHODS", ",".join(METHODS)).split(",") if m in METHODS]
# `mean` is R=1 and `maha` is off-budget: both are constant across the R sweep, so they are
# fitted once at the smallest budget and their number is repeated (marked * in the table).
BUDGET_FREE = {"mean", "maha"}


def un(A):
    return np.asarray(A, np.float32) / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ labels / features
def get_labels(ds):
    """Same split convention as exp16/exp17 (80/20 at SPLIT_SEED for single-split sets)."""
    from datasets import load_dataset
    if ds == "CIFAR100":
        from torchvision import datasets as tvd
        tr = tvd.CIFAR100(os.path.join(REPO, "data"), train=True, download=False)
        te = tvd.CIFAR100(os.path.join(REPO, "data"), train=False, download=False)
        return np.array(tr.targets), np.array(te.targets), 100
    if ds == "CUB200":
        dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
        ytr, yte = np.array(dd["train"]["label"]), np.array(dd["test"]["label"])
        return ytr, yte, int(max(ytr.max(), yte.max())) + 1
    if ds == "IMAGENETR":
        d = load_dataset("axiong/imagenet-r", cache_dir=os.path.join(REPO, "data/hf"))["test"]
        w = np.array(d["wnid"])
        lab = np.searchsorted(np.array(sorted(set(w.tolist()))), w)
    elif ds == "IMAGENETA":
        d = load_dataset("barkermrl/imagenet-a",
                         cache_dir=os.path.join(REPO, "data/hf"))["train"]
        lab = np.array(d["label"])
    else:
        raise ValueError(ds)
    p = np.random.default_rng(SPLIT_SEED).permutation(len(lab))
    n = int(0.8 * len(lab))
    return lab[p[:n]], lab[p[n:]], int(lab.max()) + 1


def adapted_features(ds):
    f = os.path.join(REPO, f"exp16_feats_{ds}_T{T}_s{SEED}_ep40_lr0.0003_aug1_{TAG}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return un(z["Ftr"]), un(z["Fte"])


def frozen_features(ds):
    """Shared pretrained backbone, no LoRA -- the control for the per-dataset-LoRA confound."""
    cache = os.path.join(REPO, f"exp19_frozen_{ds}_{TAG}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return un(z["Ftr"]), un(z["Fte"])
    log(f"    extracting frozen features for {ds} (one-off, cached)")
    import timm
    from timm.data import create_transform, resolve_model_data_config
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset
    from backbone import load_backbone

    cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
    tf = create_transform(**cfg, is_training=False)

    class HFWrap(Dataset):
        def __init__(self, d, idx):
            self.d, self.idx = d, np.asarray(idx)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            img = self.d[int(self.idx[i])]["image"]
            return tf(img.convert("RGB") if img.mode != "RGB" else img), 0

    class TVWrap(Dataset):
        def __init__(self, d):
            self.d = d

        def __len__(self):
            return len(self.d)

        def __getitem__(self, i):
            img = self.d[i][0]
            return tf(img.convert("RGB") if img.mode != "RGB" else img), 0

    if ds == "CIFAR100":
        from torchvision import datasets as tvd
        sets = [TVWrap(tvd.CIFAR100(os.path.join(REPO, "data"), train=True, download=False)),
                TVWrap(tvd.CIFAR100(os.path.join(REPO, "data"), train=False, download=False))]
    elif ds == "CUB200":
        dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
        sets = [HFWrap(dd["train"], np.arange(len(dd["train"]))),
                HFWrap(dd["test"], np.arange(len(dd["test"])))]
    else:
        if ds == "IMAGENETR":
            d = load_dataset("axiong/imagenet-r",
                             cache_dir=os.path.join(REPO, "data/hf"))["test"]
            n_all = len(d["wnid"])
        else:
            d = load_dataset("barkermrl/imagenet-a",
                             cache_dir=os.path.join(REPO, "data/hf"))["train"]
            n_all = len(d["label"])
        p = np.random.default_rng(SPLIT_SEED).permutation(n_all)
        ntr = int(0.8 * n_all)
        sets = [HFWrap(d, p[:ntr]), HFWrap(d, p[ntr:])]

    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV, lora_rank=0)
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        for s in sets:
            acc = []
            for x, _ in DataLoader(s, batch_size=256, num_workers=8, pin_memory=True):
                acc.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
            out.append(np.concatenate(acc, 0))
    del model
    torch.cuda.empty_cache()
    np.savez(cache, Ftr=out[0], Fte=out[1])
    return un(out[0]), un(out[1])


# ------------------------------------------------------------------ descriptors
# Every fit_* returns score(Q) -> (n,) with HIGHER = more in-distribution, and stores
# exactly R vectors of dimension D (except `mean`, R=1, and `maha`, flagged off-budget).
def _hull(X, R, init="kmeans"):
    """PCA dim must exceed the SPA candidate pool (3R) or the residual hits zero after
    pca_dim picks and the remaining rays are arbitrary -- hence max(64, 3R+16)."""
    n, d = X.shape
    pdim = int(min(d, n, max(64, 3 * R + 16)))
    # ray_init MUST be explicit and MUST appear in the arm name: the library default changed
    # to "kmeans" in 2026-08, so passing only ray_diversity would silently build a different
    # object under the same label -- and the results key would not record which one.
    return ConicHull(n_rays=int(min(R, n - 2)), use_pca=pdim < d, pca_dim=pdim,
                     ray_init=init, ray_diversity="hybrid").fit(X)


def fit_cone(X, y, R, rng, coverage=False, init="kmeans"):
    h = _hull(X, R, init)

    def score(Q):
        Qn = un(Q)
        W = h._reconstruct_norm(Qn)                 # one NNLS solve, two scores off it
        rec = W @ h.extreme_rays_
        rn = np.linalg.norm(rec, axis=1)
        cos = np.sum(Qn * (rec / (rn[:, None] + 1e-12)), axis=1)
        return np.clip(cos, -1, 1) * np.clip(rn, 0, 1) if coverage else cos
    return score


_SVD = {}


def _basis(X, R):
    """Top-R right singular vectors, memoised so subspace/facet/facet_sum/facet_soft all
    share ONE basis per (matrix, protocol). Uncentred, like the hull -- keeps the facet
    rows directly comparable to `subspace`, which is the whole point of the control."""
    k = (X.shape, float(X[0, 0]), float(X[-1, -1]), float(X.sum()))
    if k not in _SVD:
        _SVD[k] = np.linalg.svd(X, full_matrices=False)[2]
    return _SVD[k][:int(min(R, min(X.shape) - 1))]


def fit_subspace(X, y, R, rng):
    """Same reconstruction-fidelity score, non-negativity dropped: the `is it conic?` control."""
    B = _basis(X, R)                                 # (R, D) orthonormal

    def score(Q):
        Qn = un(Q)
        return np.linalg.norm(Qn @ B.T, axis=1)      # = cos(q, projection) for unit q
    return score


def fit_facet(X, y, R, rng, mode="min"):
    """H-REPRESENTATION cone: an axis-aligned orthotope in the shared SVD frame.

    Stores R directions (shared) + 2R extent scalars per set, and accepts CONJUNCTIVELY:
    l <= Bq <= u on every coordinate. Unlike the V-representation hull, adding a facet can
    only SHRINK the accept region -- the fix for the monotone-cover failure the cone rows show.

    The three modes are the same box read three ways, and the comparison between them IS the
    pre-registered gate: t is the per-direction deviation in half-width units (0 at centre,
    +-1 on the box face).
        min   -max|t|      min-margin over the 2R facets  -> the CONIC reading
        sum   -sum t^2     diagonal Mahalanobis in the frame -> the NON-conic reading
        soft  -logsumexp   the interpolation between them
    If min ~= sum the conjunction is carrying nothing and the honest result is
    'diagonal Mahalanobis in a learned subspace', which is not a conic result.

    v1 FAILED and the failure is kept here because it is the informative part: a two-sided
    box on the RAW projections Bq scores 0.16-0.34 AUROC -- systematically INVERTED. Reason:
    off-distribution points project near ZERO on directions that are specific to the fitted
    set, and an interval straddling zero counts near-zero as maximally interior. A box around
    the data is an affine polytope, not a cone; it treats the origin as the safest point.

    v2 factors the cone properly into radial x angular, which is what makes it scale-free:
        energy  e = ||Bq||          one-sided facet, only LOW energy is a violation
                                    (for unit q this is exactly the `subspace` score)
        shape   u = Bq/||Bq||       two-sided facets on the unit in-subspace direction
    A point near the origin of the subspace now fails the energy facet instead of passing
    every shape facet, and u is unit by construction so the degeneracy cannot recur.
    """
    B = _basis(X, R)
    Z = X @ B.T
    e = np.linalg.norm(Z, axis=1) + 1e-12
    U = Z / e[:, None]
    lo = np.quantile(U, Q_EXT, axis=0)
    hi = np.quantile(U, 1.0 - Q_EXT, axis=0)
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 + 1e-8                    # robust angular half-width per direction
    e_lo = float(np.quantile(e, Q_EXT))
    e_half = float(np.quantile(e, 0.5) - e_lo) + 1e-8

    def score(Q):
        Zq = un(Q) @ B.T
        eq = np.linalg.norm(Zq, axis=1) + 1e-12
        t = np.abs((Zq / eq[:, None] - mid) / half)                  # (n, R) shape facets
        te = np.clip((e_lo - eq) / e_half, 0.0, None)[:, None]       # (n, 1) energy facet
        t = np.concatenate([t, te], 1)
        if mode == "min":
            return -t.max(1)
        if mode == "sum":
            return -(t ** 2).sum(1)
        m = t.max(1, keepdims=True)                  # stabilised softmin over facets
        return -(m[:, 0] + np.log(np.exp(TAU * (t - m)).sum(1)) / TAU)
    return score


def _max_cos(P):
    Pn = un(P)

    def score(Q):
        return (un(Q) @ Pn.T).max(1)
    return score


def fit_kmeans(X, y, R, rng):
    from sklearn.cluster import KMeans
    R = int(min(R, len(X)))
    C = KMeans(n_clusters=R, n_init=3, random_state=RNG_SEED).fit(X).cluster_centers_
    return _max_cos(C)


def fit_multiproto(X, y, R, rng):
    """The only descriptor here that gets to see LABELS -- deliberately advantaged.
    R >= n_cls: split each class into R//n_cls k-means sub-prototypes.
    R <  n_cls: merge class means into R groups so the budget is still exactly R."""
    from sklearn.cluster import KMeans
    cls = np.unique(y)
    mus = np.stack([X[y == c].mean(0) for c in cls])
    if R <= len(cls):
        if R == len(cls):
            return _max_cos(mus)
        km = KMeans(n_clusters=int(R), n_init=3, random_state=RNG_SEED).fit(mus)
        return _max_cos(km.cluster_centers_)
    per = int(R // len(cls))
    P = []
    for c in cls:
        Xc = X[y == c]
        k = int(min(per, len(Xc)))
        P.append(KMeans(n_clusters=k, n_init=3, random_state=RNG_SEED).fit(Xc).cluster_centers_
                 if k > 1 else Xc.mean(0, keepdims=True))
    return _max_cos(np.concatenate(P, 0)[:R])


def fit_knn(X, y, R, rng):
    idx = rng.choice(len(X), size=int(min(R, len(X))), replace=False)
    return _max_cos(X[idx])                          # 1-NN cosine over R stored exemplars


def fit_mean(X, y, R, rng):
    return _max_cos(X.mean(0, keepdims=True))


def fit_maha(X, y, R, rng):
    """OFF-BUDGET: a full DxD covariance is ~2300x the R=256 vector budget."""
    mu = X.mean(0)
    Xc = X - mu
    S = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    S += 1e-3 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])   # Ledoit-style ridge
    Pm = np.linalg.inv(S).astype(np.float32)

    def score(Q):
        D = un(Q) - mu
        return -np.einsum("ij,jk,ik->i", D, Pm, D)   # negated: higher = more ID
    return score


FIT = {"cone_spa": lambda X, y, R, g: fit_cone(X, y, R, g, init="spa"),
       "cone_km": lambda X, y, R, g: fit_cone(X, y, R, g, init="kmeans"),
       "cone_km_cov": lambda X, y, R, g: fit_cone(X, y, R, g, coverage=True,
                                                  init="kmeans"),
       "subspace": fit_subspace,
       "facet": lambda X, y, R, g: fit_facet(X, y, R, g, mode="min"),
       "facet_soft": lambda X, y, R, g: fit_facet(X, y, R, g, mode="soft"),
       "facet_sum": lambda X, y, R, g: fit_facet(X, y, R, g, mode="sum"),
       "kmeans": fit_kmeans, "multiproto": fit_multiproto,
       "knn": fit_knn, "mean": fit_mean, "maha": fit_maha}


# ------------------------------------------------------------------ protocols
def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def run_space(space, prior_cells):
    rng = np.random.default_rng(RNG_SEED)
    data = {}
    for ds in DSETS:
        F = adapted_features(ds) if space == "adapted" else frozen_features(ds)
        if F is None:
            log(f"  MISSING adapted cache for {ds} -- skipped")
            continue
        Ztr, Zte = F
        ytr, yte, n_cls = get_labels(ds)
        assert len(Ztr) == len(ytr) and len(Zte) == len(yte), \
            f"feature/label mismatch {ds}: {Ztr.shape} vs {ytr.shape}"
        p = rng.permutation(len(Ztr))
        ncal = max(int(CAL_FRAC * len(p)), 50)
        cal, fit = p[:ncal], p[ncal:]
        sub = fit if len(fit) <= N_FIT else fit[:N_FIT]
        data[ds] = dict(Xfit=Ztr[sub], yfit=ytr[sub], Xcal=Ztr[cal],
                        Xte=Zte, yte=yte, n_cls=n_cls)
        log(f"  {ds}: fit {len(sub)}/{len(fit)} rows, cal {ncal}, test {len(Zte)}, "
            f"{n_cls} classes")
    names = list(data)
    if len(names) < 2:
        return None

    # class-balanced pooled test set for P1 (else argmax accuracy tracks test-set sizes)
    per = min(len(data[d]["Xte"]) for d in names)
    pool = {d: data[d]["Xte"][rng.permutation(len(data[d]["Xte"]))[:per]] for d in names}
    Qpool = np.concatenate([pool[d] for d in names], 0)
    ypool = np.concatenate([np.full(per, i) for i in range(len(names))])
    log(f"  P1 pool: {len(names)} x {per} = {len(Qpool)} queries")

    # P3 near-OOD split: first half of each dataset's classes is "seen"
    half = {}
    for d in names:
        D = data[d]
        seen = np.arange(D["n_cls"])[: D["n_cls"] // 2]
        m = np.isin(D["yfit"], seen)
        half[d] = dict(X=D["Xfit"][m], y=D["yfit"][m],
                       id=D["Xte"][np.isin(D["yte"], seen)],
                       ood=D["Xte"][~np.isin(D["yte"], seen)])

    res = dict(prior_cells)          # keep cells computed by earlier partial-METHODS runs
    for meth in WANT:
        for R in BUDGETS:
            if f"{meth}|{R}" in res:
                continue
            if meth in BUDGET_FREE and R != BUDGETS[0]:
                res[f"{meth}|{R}"] = res[f"{meth}|{BUDGETS[0]}"]
                continue
            t = time.time()
            cell = {}
            # ---- P1 + P2 (whole-dataset descriptors) ----
            sc, cal_stats = {}, {}
            for d in names:
                g = np.random.default_rng(RNG_SEED)
                f = FIT[meth](data[d]["Xfit"], data[d]["yfit"], R, g)
                s = f(data[d]["Xcal"])
                cal_stats[d] = (float(s.mean()), float(s.std() + 1e-12))
                sc[d] = {"pool": f(Qpool)}
                for e in names:
                    sc[d][e] = f(data[e]["Xte"])
            Sp = np.stack([sc[d]["pool"] for d in names], 1)
            Sz = np.stack([(sc[d]["pool"] - cal_stats[d][0]) / cal_stats[d][1]
                           for d in names], 1)
            cell["p1_acc_raw"] = float((Sp.argmax(1) == ypool).mean())
            cell["p1_acc_cal"] = float((Sz.argmax(1) == ypool).mean())
            pairs = [auroc(sc[d][d], sc[d][e]) for d in names for e in names if e != d]
            cell["p2_auroc_mean"] = float(np.mean(pairs))
            cell["p2_auroc_min"] = float(np.min(pairs))
            cell["p2_pairs"] = {f"{d}->{e}": auroc(sc[d][d], sc[d][e])
                                for d in names for e in names if e != d}
            # ---- P3 near-OOD (unseen classes of the SAME dataset) ----
            near = {}
            for d in names:
                h = half[d]
                if len(h["X"]) < 16 or len(h["ood"]) < 16:
                    continue
                g = np.random.default_rng(RNG_SEED)
                f = FIT[meth](h["X"], h["y"], R, g)
                near[d] = auroc(f(h["id"]), f(h["ood"]))
            cell["p3_near_auroc"] = near
            cell["p3_mean"] = float(np.mean(list(near.values()))) if near else float("nan")
            cell["secs"] = round(time.time() - t, 1)
            res[f"{meth}|{R}"] = cell
            log(f"    {space:8s} {meth:11s} R={R:<4d} P1cal {cell['p1_acc_cal']*100:5.1f}  "
                f"P2 {cell['p2_auroc_mean']:.4f}  P3 {cell['p3_mean']:.4f}  "
                f"({cell['secs']}s)")
    return {"datasets": names, "cells": res}


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for space in SPACES:
        key = f"{space}|T{T}|s{SEED}"
        log(f"=== {key}")
        r = run_space(space, allres.get(key, {}).get("cells", {}))
        if r:
            allres[key] = r
            json.dump(allres, open(OUT, "w"), indent=2)

    W = 104
    print("\n" + "=" * W)
    print(f"EXP19 — one conic hull per DATASET vs matched-budget set descriptors ({MODEL})")
    print("=" * W)
    for key, r in allres.items():
        names = r["datasets"]
        print(f"\n--- {key}   datasets: {', '.join(names)}   (P1 chance = "
              f"{100/len(names):.1f}%)")
        print(f"{'method':<12}{'R':>5}{'P1 raw':>9}{'P1 cal':>9}{'P2 mean':>10}{'P2 min':>9}"
              f"{'P3 near':>10}   P3 per-dataset")
        for meth in METHODS:
            for R in BUDGETS:
                c = r["cells"].get(f"{meth}|{R}")
                if c is None:
                    continue
                star = "*" if meth in BUDGET_FREE else " "
                pd = "  ".join(f"{k[:6]} {v:.3f}" for k, v in c["p3_near_auroc"].items())
                print(f"{meth:<12}{R:>4}{star}{c['p1_acc_raw']*100:>9.1f}"
                      f"{c['p1_acc_cal']*100:>9.1f}{c['p2_auroc_mean']:>10.4f}"
                      f"{c['p2_auroc_min']:>9.4f}{c['p3_mean']:>10.4f}   {pd}")
                if meth in BUDGET_FREE:
                    break
    print("\n" + "-" * W)
    print("* = budget-free row (mean stores 1 vector; maha stores a full DxD covariance,")
    print("    ~2300x the R=256 budget) -- printed once, not swept.")
    print("P1 cal is the honest dataset-ID number: raw argmax rewards whichever dataset happens")
    print("    to have the tightest descriptor, so scores are z-normalised on held-out own-train.")
    print("A cone win counts ONLY if it holds at matched R, over `subspace` (= same score without")
    print("    non-negativity), and in the frozen space (= no per-dataset LoRA fingerprint).")
    print("=" * W)
    print(f"wrote {OUT}")
