#!/usr/bin/env python3
"""exp10_lorentz_phase0.py — is an anisotropic second-order (Lorentz) cone a valid
                             CROWDING INSTRUMENT?  Kill-gate before any CIL training.

WHY THIS EXISTS
    The polyhedral ConicHull cannot say WHERE a class is crowded: overlap between two
    hulls needs an LP per pair (200x200 = 20k LPs). A second-order cone can, in closed
    form -- which is the only reason to generalise.

    But `second_order.py:generate_soft_socs` builds the ISOTROPIC cone: axis = mean of
    hull.extreme_rays_, aperture = MAX angle to any ray. That is a strict LOOSENING of
    the hull, so on d=768 (eff_rank ~36) it crowds MORE, not less. The generalisation
    that can buy something is the ANISOTROPIC SOC  ||S^-1/2 P^T (x/(u^T x) - u)|| <= kappa,
    i.e. the real ||Ax+b|| <= c^T x + d form with a MATRIX shape.

    Family, from loosest to tightest:  NCM point  <  isotropic SOC  <  anisotropic SOC
                                                                    <  polyhedral hull
    The endpoints are already implemented (ConicHull / generate_soft_socs). The middle
    is the new object, and this script asks only whether it MEASURES anything.

WHY A GATE AND NOT JUST A RUN
    logs record that a cone diagnostic already failed once by reading +0.12 conic gain on
    a UNIMODAL class -- it was measuring capacity, not structure. So the cone here is not
    allowed to be judged on its own: it must (a) correlate with where classification
    actually fails and (b) beat a no-cone control that only knows prototype distances.

THE FOUR PRE-REGISTERED GATES              (thresholds fixed HERE, before seeing numbers)
    A  MEASURES FAILURE   spearman(crowding_aniso, per-class frozen-NCM error) >= 0.40
                          fails -> the geometry does not track real confusion; it is a
                          capacity read again, exactly the prior failure mode.
    B  BEATS NO-CONE      rho(aniso) - max(rho(dist_only), rho(nn_angle)) >= 0.10
                          fails -> prototype distances already contain it; the per-class
                          extent (the whole point of a cone) adds nothing.
    C  EXTENT HAS STRUCTURE  CV over classes of the equivalent isotropic radius >= 0.25
                          fails -> one global aperture is as good as per-class cones and
                          the method collapses to NCM + a threshold. NOTE this is a
                          DIFFERENT quantity from spread_check.py's gate B, which measured
                          prototype ESTIMATION uncertainty s_c = sqrt(tr Sigma_c / N_c)
                          (ImageNet-R: CV 0.216, FAILED). Class EXTENT may still have
                          structure where estimation error did not.
    D  FREE SPACE EXISTS  fraction of on-manifold probe directions inside ZERO cones >= 0.05
                          fails -> "restrict new-task updates to the other regions" has
                          nowhere to go and Phase 2 is dead before it is written.

    ALL FOUR must pass for Phase 1/2 to be worth the 3.5 GPU-hours.

WHAT IS DELIBERATELY NOT HERE
    * No training. Frozen phi_0 only.
    * No polyhedral-LP comparison -- 20k LPs is the one expensive piece and gate B's
      cheap controls answer the same question (does the extent add anything?).
    * No classification with cones -- that is Phase 1, and crux_stagecone already showed
      cone scoring TIES NCM (0.8398 vs 0.8413 oracle). Phase 0 is about measurement only.

PROTOCOL
    Identical to exp8_combined.py: same backbone, same 80/20 split at seed 1993, same
    deterministic transform, same L2 normalisation. Features cached to .npz -- the first
    run pays ~90 s of extraction, every re-run is ~30 s.

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp10_lorentz_phase0.py                    # ImageNet-R, N in {1,2,4}
    DATASET=CIFAR100 python -u exp10_lorentz_phase0.py   # negative control (no headroom)
    N_CONES=1,2,4 M_PER=20,0 python -u exp10_lorentz_phase0.py
"""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import timm
from timm.data import create_transform, resolve_model_data_config

from backbone import load_backbone
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ config
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
DATASET = os.environ.get("DATASET", "IMAGENETR")
SPLIT_SEED = 1993                 # exp8_combined.py:_p / spread_check.py:SPLIT_SEED

K_RAYS = int(os.environ.get("K_RAYS", 16))       # extreme rays per cone (ConicHull.n_rays)
RANK = int(os.environ.get("RANK", 16))           # tangent-shape rank (eff_rank ~36)
SHRINK = float(os.environ.get("SHRINK", 0.1))    # same convention as exp8 MAHA_SHRINK
COVER = float(os.environ.get("COVER", 0.95))     # kappa calibrated to cover this fraction
N_PROBE = int(os.environ.get("N_PROBE", 20_000))   # se on a fraction ~0.003; plenty for gate D

# N cones per class (k-means sub-cones); M examples used to build them (0 = all)
N_CONES = [int(v) for v in os.environ.get("N_CONES", "1,2,4").split(",")]
M_PER = [int(v) for v in os.environ.get("M_PER", "0,20").split(",")]

# pre-registered gates -- DO NOT EDIT AFTER SEEING RESULTS
GATE_A_RHO = 0.40
GATE_B_LIFT = 0.10
GATE_C_CV = 0.25
GATE_D_FREE = 0.05


# ------------------------------------------------------------------ data
_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
TF = create_transform(**_cfg, is_training=False)   # deterministic: statistics, never crops


class HFWrap(Dataset):
    def __init__(self, ds, idx, labels):
        self.ds, self.idx, self.labels = ds, np.asarray(idx), np.asarray(labels)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return TF(img), int(self.labels[i])


def load_data():
    """Return (train_ds, y_train, test_ds, y_test) under the exp8 80/20 convention."""
    from datasets import load_dataset

    if DATASET == "CIFAR100":
        from torchvision import datasets as tvd
        tr = tvd.CIFAR100("./data", train=True, download=False, transform=TF)
        te = tvd.CIFAR100("./data", train=False, download=False, transform=TF)
        return tr, np.array(tr.targets), te, np.array(te.targets)

    if DATASET == "IMAGENETR":
        d = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
        w = np.array(d["wnid"])
        lab = np.searchsorted(np.array(sorted(set(w.tolist()))), w)
    else:                                     # IMAGENETA / CUB200 / CARS
        repo = {"IMAGENETA": "barkermrl/imagenet-a",
                "CUB200": "Donghyun99/cub-200-2011",
                "CARS": "Donghyun99/stanford-cars"}[DATASET]
        d = load_dataset(repo, cache_dir="./data/hf")
        d = d["train"] if "train" in d else d[list(d.keys())[0]]
        lab = np.array(d["label"])

    p = np.random.default_rng(SPLIT_SEED).permutation(len(lab))
    n = int(0.8 * len(lab))
    return (HFWrap(d, p[:n], lab[p[:n]]), lab[p[:n]],
            HFWrap(d, p[n:], lab[p[n:]]), lab[p[n:]])


@torch.no_grad()
def extract(model, ds, bs=256):
    out = []
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def get_features():
    cache = f"exp10_feats_{DATASET}_{MODEL.split('.')[-1]}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"features from cache {cache}  train {z['Ftr'].shape} test {z['Fte'].shape}")
        return z["Ftr"], z["ytr"], z["Fte"], z["yte"]
    tr, ytr, te, yte = load_data()
    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV).eval()
    log(f"extracting {len(ytr)} train + {len(yte)} test through FROZEN {MODEL}")
    Ftr, Fte = extract(model, tr), extract(model, te)
    del model
    torch.cuda.empty_cache()
    np.savez_compressed(cache, Ftr=Ftr, ytr=ytr, Fte=Fte, yte=yte)
    log(f"cached -> {cache}")
    return Ftr, ytr, Fte, yte


# ------------------------------------------------------------------ the cone
class SOCone:
    """One anisotropic second-order cone.

        C = { x : u^T x > 0,  q(x) <= kappa },
        q(x)^2 = z^T S^-1 z + ||t - P z||^2 / sigma_r^2,   t = x/(u^T x) - u,  z = P^T t

    `t` is the gnomonic projection of x onto the tangent hyperplane at u (u^T t = 0 by
    construction), which is exactly what makes membership a SECOND-ORDER cone constraint
    in x rather than a general quadratic.

    The isotropic residual sigma_r^2 on the complement of span(P) is not cosmetic: without
    it the cone is a measure-zero slab in R^768 and every off-subspace point is instantly
    outside. It is the probabilistic-PCA completion of the truncated shape.

    isotropic=True collapses S to a scalar -- this reproduces `generate_soft_socs`'s
    ice-cream cone and is the control for "is the anisotropy load-bearing?".
    """

    def __init__(self, u, P, Sinv, sig2, kappa, iso_radius):
        self.u, self.P, self.Sinv, self.sig2 = u, P, Sinv, sig2
        self.Pu = P.T @ u                  # tangent-coord offset used by q()
        self.kappa = kappa
        self.iso_radius = iso_radius       # equivalent isotropic angular radius (gate C)

    # -- membership -------------------------------------------------------
    def q(self, X):
        """Mahalanobis radius of each row of X (UNIT NORM) in tangent coords.

        Never materialises the (n,d) tangent matrix. For unit x and unit u, with P having
        orthonormal columns:
            t = x/(u.x) - u,   u.t = 0
            ||t||^2 = 1/(u.x)^2 - 1
            z = P^T t = (P^T x)/(u.x) - P^T u
        so only P^T x (n,r) and u.x (n,) are needed -- a 48x smaller footprint at r=16,
        which is what makes the 20k-probe free-space sweep cheap.
        """
        proj = X @ self.u
        safe = np.maximum(proj, 1e-6)
        Z = (X @ self.P) / safe[:, None] - self.Pu
        t2 = 1.0 / (safe * safe) - 1.0
        resid = np.maximum(t2 - (Z * Z).sum(1), 0.0)
        q2 = np.einsum("ij,jk,ik->i", Z, self.Sinv, Z) + resid / self.sig2
        out = np.sqrt(np.maximum(q2, 0))
        out[proj <= 0] = np.inf            # behind the apex is outside the cone
        return out

    def inside(self, X):
        return self.q(X) <= self.kappa

    # -- directional extent -----------------------------------------------
    def radius_toward(self, W):
        """Angular radius (rad) of the cone along each unit tangent direction in W (n,d).

        Along direction w the boundary sits at tangent distance s = kappa / sqrt(w^T Sigma^-1 w);
        the angular radius is arctan(s). This closed form is the whole reason to leave the
        polyhedral hull: for two hulls the same query is an LP.
        """
        A = W @ self.P
        quad = np.einsum("ij,jk,ik->i", A, self.Sinv, A)
        quad = quad + np.maximum(1.0 - (A * A).sum(1), 0) / self.sig2
        return np.arctan(self.kappa / np.sqrt(np.maximum(quad, 1e-12)))


def hull_axis(F):
    """Cone axis exactly as second_order.py:generate_soft_socs -- mean of the SPA extreme
    rays, via the repo's own ConicHull. Computed ONCE per cone and shared by the
    anisotropic and isotropic fits, which differ only in the shape matrix."""
    n, d = F.shape
    if n < 8:
        # ConicHull's hybrid path runs NearestNeighbors(n_neighbors=6) internally, so it
        # cannot be fitted below 8 points. The extreme-ray mean degenerates to the sample
        # mean at that size anyway, so fall back rather than fail.
        u = F.mean(0)
        return u / (np.linalg.norm(u) + 1e-12)
    k = int(np.clip(K_RAYS, 2, max(n - 2, 2)))
    # ConicHull's PCA runs with svd_solver='full', so pca_dim must not exceed n_samples
    pdim = int(min(64, n, d))
    hull = ConicHull(n_rays=k, use_pca=pdim < d, pca_dim=pdim,
                     ray_diversity="hybrid").fit(F)
    u = hull.extreme_rays_.mean(0)
    return u / (np.linalg.norm(u) + 1e-12)


def fit_cone(F, isotropic=False, u=None):
    """Fit one SOCone to L2-normalised rows F (n,d)."""
    n, d = F.shape
    if u is None:
        u = hull_axis(F)

    proj = F @ u
    keep = proj > 1e-3
    if keep.sum() < 3:                     # degenerate class: fall back to the mean direction
        u = F.mean(0)
        u /= np.linalg.norm(u) + 1e-12
        proj = F @ u
        keep = proj > 1e-3
    T = F[keep] / proj[keep, None] - u     # (n', d), u^T T = 0 exactly
    nk = len(T)

    r = int(np.clip(min(RANK, nk - 1, d - 1), 1, d - 1))
    _, sv, Vt = np.linalg.svd(T, full_matrices=False)
    P = Vt[:r].T                                           # (d, r)
    Z = T @ P

    # residual variance per dimension on the complement of span(P) -- see SOCone docstring
    d_eff = max(min(nk - 1, d - 1), r + 1)
    resid_energy = max(float((T * T).sum() - (Z * Z).sum()), 0.0)
    sig2 = resid_energy / (nk * max(d_eff - r, 1))
    tot_var = float((T * T).sum()) / (nk * d_eff)
    sig2 = max(sig2, 1e-8 * max(tot_var, 1e-12), 1e-12)

    S = (Z.T @ Z) / max(nk - 1, 1)
    if isotropic:
        s_iso = max(float(np.trace(S)) / r, 1e-12)
        S = np.eye(r) * s_iso
        sig2 = max(s_iso, 1e-12)           # a genuine ice-cream cone: one scalar everywhere
    else:
        S = (1 - SHRINK) * S + SHRINK * (np.trace(S) / r) * np.eye(r)
    Sinv = np.linalg.inv(S + 1e-10 * np.eye(r))

    cone = SOCone(u, P, Sinv, sig2, kappa=1.0, iso_radius=0.0)
    q = cone.q(F[keep])
    cone.kappa = float(np.quantile(q[np.isfinite(q)], COVER)) if np.isfinite(q).any() else 1.0
    cone.kappa = max(cone.kappa, 1e-6)
    # Equivalent isotropic radius, used ONLY for gate C (does extent vary across classes?).
    # It must not depend on n_c: an earlier version divided the tangent energy by
    # d_eff = min(n_c-1, d-1), which made the radius shrink with class size and manufactured
    # (or, at fixed M, suppressed) the very CV the gate measures. The COVER-quantile of the
    # angle to the axis is n-independent and is generate_soft_socs' own notion of aperture,
    # with the max replaced by a quantile so one outlier cannot set it.
    cone.iso_radius = float(np.quantile(np.arccos(np.clip(F @ u, -1.0, 1.0)), COVER))
    return cone


def fit_class(F, n_cones, seed=0):
    """N sub-cones per class via k-means (N=1 -> a single cone). Returns list[SOCone] x2."""
    if n_cones <= 1 or len(F) < 8 * n_cones:
        u = hull_axis(F)
        return [fit_cone(F, False, u)], [fit_cone(F, True, u)]
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=n_cones, n_init=4, random_state=seed).fit_predict(F)
    ani, iso = [], []
    for c in range(n_cones):
        Fc = F[lab == c]
        # ConicHull's hybrid path runs NearestNeighbors(n_neighbors=6), so a sub-cone needs
        # at least 8 points; smaller k-means clusters are dropped, not fitted.
        if len(Fc) < 8:
            continue
        u = hull_axis(Fc)
        ani.append(fit_cone(Fc, False, u))
        iso.append(fit_cone(Fc, True, u))
    if not ani:
        u = hull_axis(F)
        return [fit_cone(F, False, u)], [fit_cone(F, True, u)]
    return ani, iso


# ------------------------------------------------------------------ crowding
def crowding_matrix(cones):
    """m[i,j] = angle(u_i,u_j) - r_i(toward j) - r_j(toward i);  m < 0 means the cones OVERLAP.

    cones: list (per class) of list (per sub-cone) of SOCone.
    For multi-cone classes the pair margin is the MIN over sub-cone pairs -- one overlapping
    lobe is an overlap.
    """
    flat, owner = [], []
    for ci, group in enumerate(cones):
        for c in group:
            flat.append(c)
            owner.append(ci)
    owner = np.asarray(owner)
    U = np.stack([c.u for c in flat])                       # (F, d)
    F_ = len(flat)
    ang = np.arccos(np.clip(U @ U.T, -1.0, 1.0))

    R = np.zeros((F_, F_))                                  # R[i,j] = r_i(toward j)
    for i, c in enumerate(flat):
        W = U - np.outer(U @ c.u, c.u)                      # tangent component at u_i
        nrm = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.maximum(nrm, 1e-12)
        R[i] = c.radius_toward(W)
    Mf = ang - R - R.T

    C = owner.max() + 1
    M = np.full((C, C), np.inf)
    for a in range(F_):                                     # min over sub-cone pairs
        np.minimum.at(M[owner[a]], owner, Mf[a])
    np.fill_diagonal(M, np.inf)
    return M, ang, owner


def crowd_score(M):
    """Total signed overlap each class suffers: sum_j max(0, -m_ij)."""
    ov = np.maximum(-M, 0.0)
    ov[~np.isfinite(M)] = 0.0
    return ov.sum(1), (M < 0).sum(1)


# ------------------------------------------------------------------ main
def main():
    Ftr, ytr, Fte, yte = get_features()
    Ftr, Fte = un(Ftr), un(Fte)
    classes = np.unique(ytr)
    C, d = len(classes), Ftr.shape[1]
    log(f"{DATASET}: {C} classes, d={d}, train {len(ytr)}, test {len(yte)}")

    # ---- reference signal: per-class frozen-NCM error on the TEST split -------------
    mu = un(np.stack([Ftr[ytr == c].mean(0) for c in classes]))
    pred = classes[(Fte @ mu.T).argmax(1)]
    err = np.array([1.0 - (pred[yte == c] == c).mean() for c in classes])
    log(f"frozen NCM: overall acc {(pred == yte).mean():.4f} | "
        f"per-class err mean {err.mean():.3f} sd {err.std():.3f}")

    # ---- no-cone controls (gate B) ---------------------------------------------------
    ang_mu = np.arccos(np.clip(mu @ mu.T, -1.0, 1.0))
    np.fill_diagonal(ang_mu, np.inf)
    nn_angle = -ang_mu.min(1)                       # closest other prototype: the crudest read
    tau = float(np.median(ang_mu.min(1)))
    dist_only = np.maximum(tau - ang_mu, 0.0)
    dist_only[~np.isfinite(ang_mu)] = 0.0
    dist_only = dist_only.sum(1)                    # knows distances, knows NO extent

    from scipy.stats import spearmanr
    rho_nn = float(spearmanr(nn_angle, err).correlation)
    rho_dist = float(spearmanr(dist_only, err).correlation)
    ctrl_best = max(rho_nn, rho_dist)
    log(f"controls: rho(nn_angle) {rho_nn:+.3f}  rho(dist_only) {rho_dist:+.3f} "
        f"-> best control {ctrl_best:+.3f}")

    rows = []
    for m_per in M_PER:
        # per-class subsample: M examples used to BUILD the cones (0 = all available)
        rng = np.random.default_rng(0)
        sub = {}
        for c in classes:
            ix = np.where(ytr == c)[0]
            sub[c] = ix if m_per <= 0 or len(ix) <= m_per else rng.choice(ix, m_per, False)
        m_lab = "all" if m_per <= 0 else str(m_per)
        n_used = int(np.median([len(v) for v in sub.values()]))

        for n_cones in N_CONES:
            t_fit = time.time()
            ani, iso = [], []
            for c in classes:
                a, i_ = fit_class(Ftr[sub[c]], n_cones, seed=int(c))
                ani.append(a)
                iso.append(i_)

            res = {}
            for name, cones in (("aniso", ani), ("iso", iso)):
                M, _, _ = crowding_matrix(cones)
                score, n_ov = crowd_score(M)
                rho = float(spearmanr(score, err).correlation)
                frac_ov = float((M < 0).sum() / (len(classes) * (len(classes) - 1)))
                radii = np.array([np.mean([c.iso_radius for c in g]) for g in cones])
                res[name] = dict(rho=rho, frac_overlap=frac_ov,
                                 n_ov_med=float(np.median(n_ov)),
                                 cv_extent=float(radii.std() / (radii.mean() + 1e-12)),
                                 radius_med=float(np.median(radii)))

            # ---- gate D: free space, measured ON THE DATA MANIFOLD -------------------
            # Uniform directions on S^767 are vacuously "free" (curse of dimensionality),
            # so the honest probe is directions the data can actually reach: normalised
            # convex blends of two real features from DIFFERENT classes.
            g = np.random.default_rng(7)
            ia, ib = g.integers(0, len(Ftr), N_PROBE), g.integers(0, len(Ftr), N_PROBE)
            ok = ytr[ia] != ytr[ib]
            ia, ib = ia[ok], ib[ok]
            lam = g.uniform(0.2, 0.8, len(ia))[:, None]
            probe = un(lam * Ftr[ia] + (1 - lam) * Ftr[ib])
            hits = np.zeros(len(probe), dtype=np.int32)
            for grp in ani:
                inside = np.zeros(len(probe), dtype=bool)
                for c in grp:
                    inside |= c.inside(probe)
                hits += inside
            free = float((hits == 0).mean())
            # reference only -- expected to be ~1.0 and therefore meaningless
            Q = np.linalg.qr(g.normal(size=(d, 64)))[0]
            unif = un(g.normal(size=(20_000, 64)) @ Q.T)
            uh = np.zeros(len(unif), dtype=bool)
            for grp in ani:
                for c in grp:
                    uh |= c.inside(unif)
            free_unif = float((~uh).mean())

            row = dict(dataset=DATASET, M=m_lab, M_used=n_used, N=n_cones,
                       rho_aniso=res["aniso"]["rho"], rho_iso=res["iso"]["rho"],
                       rho_ctrl=ctrl_best, rho_nn=rho_nn, rho_dist=rho_dist,
                       lift=res["aniso"]["rho"] - ctrl_best,
                       cv_extent=res["aniso"]["cv_extent"],
                       frac_overlap=res["aniso"]["frac_overlap"],
                       frac_overlap_iso=res["iso"]["frac_overlap"],
                       radius_med=res["aniso"]["radius_med"],
                       radius_med_iso=res["iso"]["radius_med"],
                       free=free, free_unif=free_unif,
                       storage_floats_per_class=int(n_cones * (d + d * RANK)))
            row["pass_A"] = bool(row["rho_aniso"] >= GATE_A_RHO)
            row["pass_B"] = bool(row["lift"] >= GATE_B_LIFT)
            row["pass_C"] = bool(row["cv_extent"] >= GATE_C_CV)
            row["pass_D"] = bool(row["free"] >= GATE_D_FREE)
            rows.append(row)
            log(f"  M={m_lab:>3} (med {n_used:3d})  N={n_cones}  "
                f"rho_aniso {row['rho_aniso']:+.3f} rho_iso {row['rho_iso']:+.3f} "
                f"lift {row['lift']:+.3f}  CV {row['cv_extent']:.3f}  "
                f"ovl {row['frac_overlap']:.3f}  free {free:.3f}  "
                f"[{time.time()-t_fit:.0f}s]")

    # ---------------------------------------------------------------- report
    W = 118
    print("\n" + "=" * W)
    print(f"EXP10 PHASE 0 — Lorentz/SOC cone as a CROWDING INSTRUMENT   "
          f"[{DATASET}, {MODEL}, frozen]")
    print(f"reference signal: per-class frozen-NCM test error  "
          f"(overall acc {(pred == yte).mean():.4f})")
    print("=" * W)
    print(f"{'M':>4}{'N':>3}{'rho_ani':>9}{'rho_iso':>9}{'rho_ctl':>9}{'lift':>8}"
          f"{'CV_ext':>8}{'overlap':>9}{'free':>8}{'fl/class':>10}  gates")
    for r in rows:
        g = ("A" if r["pass_A"] else "-") + ("B" if r["pass_B"] else "-") + \
            ("C" if r["pass_C"] else "-") + ("D" if r["pass_D"] else "-")
        star = "  <== ALL PASS" if g == "ABCD" else ""
        print(f"{r['M']:>4}{r['N']:>3}{r['rho_aniso']:>+9.3f}{r['rho_iso']:>+9.3f}"
              f"{r['rho_ctrl']:>+9.3f}{r['lift']:>+8.3f}{r['cv_extent']:>8.3f}"
              f"{r['frac_overlap']:>9.3f}{r['free']:>8.3f}"
              f"{r['storage_floats_per_class']:>10d}   {g}{star}")
    print("-" * W)
    print(f"A rho(crowding_aniso, NCM err) >= {GATE_A_RHO}   does the cone track real confusion?")
    print(f"B lift over best no-cone control >= {GATE_B_LIFT}   "
          f"(controls: nn_angle {rho_nn:+.3f}, dist_only {rho_dist:+.3f})")
    print(f"C CV of per-class extent >= {GATE_C_CV}   else one global aperture suffices")
    print(f"D on-manifold free fraction >= {GATE_D_FREE}   else Phase 2 has nowhere to move")
    print(f"  (uniform-in-64d-subspace free fraction {rows[0]['free_unif']:.3f} — reported "
          f"only to show it is vacuous)")
    best = max(rows, key=lambda r: sum(r[k] for k in ("pass_A", "pass_B", "pass_C", "pass_D")))
    n_pass = sum(best[k] for k in ("pass_A", "pass_B", "pass_C", "pass_D"))
    print("-" * W)
    if n_pass == 4:
        print(f"VERDICT: PROCEED — best config M={best['M']} N={best['N']} passes all four.")
    else:
        failed = [k[-1] for k in ("pass_A", "pass_B", "pass_C", "pass_D") if not best[k]]
        print(f"VERDICT: STOP — best config M={best['M']} N={best['N']} passes {n_pass}/4; "
              f"gate(s) {','.join(failed)} failed.")
        print("         The SOC does not measure crowding here. Phase 1/2 would be "
              "unfalsifiable; close the axis instead of spending 3.5 GPU-hours.")
    print("=" * W)

    # merge on (M, N) so a partial re-run (e.g. one extra N) does not clobber earlier configs
    out = f"exp10_phase0_{DATASET}_{MODEL.split('.')[-1]}.json"
    merged = {}
    if os.path.exists(out):
        try:
            merged = {(r["M"], r["N"]): r for r in json.load(open(out))}
        except Exception as e:
            print(f"[warn] could not merge {out}: {e}")
    merged.update({(r["M"], r["N"]): r for r in rows})
    with open(out, "w") as f:
        json.dump(list(merged.values()), f, indent=2)
    print(f"wrote {out}  ({len(merged)} configs)")


if __name__ == "__main__":
    main()
