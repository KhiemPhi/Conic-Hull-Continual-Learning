"""
ood_hybrid.py
-------------
Density-gated conic OOD: model each class as (density INTERSECT support).
  Mahalanobis (shared-cov Gaussian)  -> DENSITY (center+spread); misses off-support tails
  cone NNLS residual (ConicHull)     -> SUPPORT boundary; misses inter-mode fill
Hypothesis: they make COMPLEMENTARY errors, so fusing them beats either alone -- and the
gain concentrates on NON-GAUSSIAN (fine-grained) class supports, where the Gaussian's
tails over-extend and the cone corrects them.

Setup: cached CLIP features. Each dataset = ID (per-class 50/50 support/query), OOD =
every other dataset's test (capped). Report OOD-detection AUROC for:
  maha    : -min_c Mahalanobis distance          (density baseline)
  residual: -||x - proj_k(x)||  (NuSA/ViM-style) (subspace baseline)
  cone    : max_c score_nnls_residual            (support)
  hybrid  : z(maha) + z(cone)                     (density-gated cone, OURS)
Plus complementarity diagnostics: corr(maha,cone) and hybrid vs max(maha,cone).

    HF_HUB_OFFLINE=1 python -u ood_hybrid.py
"""
import os, json
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import cone_ood as CO

DEVICE = CO.DEVICE
OUT = "./ood_hybrid_out"
GAUSSIANISH = {"CIFAR10", "CIFAR100", "STL10"}          # coarse ~Gaussian
# everything else cached is fine-grained / non-Gaussian


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def fit_gauss(Fn, y, classes):
    mus = np.stack([Fn[y == c].mean(0) for c in classes]).astype(np.float32)
    centered = np.concatenate([Fn[y == c] - mus[i] for i, c in enumerate(classes)])
    prec = LedoitWolf().fit(centered).precision_.astype(np.float32)
    return mus, prec


def idness_maha(mus, prec, X, chunk=8192):
    # d_c(x) = xPx - 2 xP mu_c + mu_c P mu_c  (no (b,C,D) tensor -> no OOM)
    P = torch.tensor(prec, device=DEVICE); M = torch.tensor(mus, device=DEVICE)
    MP = M @ P                                        # (C,D)
    term3 = (MP * M).sum(1)                            # (C,)
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        Xc = torch.tensor(X[i:i+chunk], device=DEVICE)
        XP = Xc @ P                                    # (b,D)
        term1 = (XP * Xc).sum(1, keepdim=True)         # (b,1)
        d = term1 - 2.0 * (Xc @ MP.T) + term3[None]    # (b,C)
        out[i:i+chunk] = (-d.min(1).values).cpu().numpy()  # idness = -min maha
    return out


def idness_residual(pca, X):
    Xhat = pca.inverse_transform(pca.transform(X))
    return -np.linalg.norm(X - Xhat, axis=1)                    # -complement norm


def z(v):
    v = np.asarray(v, np.float64); return (v - v.mean()) / (v.std() + 1e-8)


def auroc(idn_id, idn_ood):
    y = np.r_[np.zeros(len(idn_id)), np.ones(len(idn_ood))]     # OOD positive
    return float(roc_auc_score(y, -np.r_[idn_id, idn_ood]))


def main():
    os.makedirs(OUT, exist_ok=True)
    avail = CO.available_datasets()
    feats = {d: CO.load_clip_feats(d) for d in avail}
    feats = {k: v for k, v in feats.items() if v is not None}
    print(f"[data] {list(feats)}", flush=True)
    methods = ["maha", "residual", "cone", "hybrid"]
    rows = {}
    for idd in feats:
        f_id, y_id = feats[idd]
        si, qi = CO.split_support_query(f_id, y_id, 150, seed=0)
        Fn_sup, ysup = unit(f_id[si]), y_id[si]
        Fq = unit(CO.cap(f_id[qi], 2000))
        classes = np.unique(ysup)
        cones, _, _ = CO.fit_id_models(f_id[si], ysup, 24)
        mus, prec = fit_gauss(Fn_sup, ysup, classes)
        pca = PCA(n_components=64).fit(Fn_sup)
        # OOD pool
        ood = np.concatenate([unit(CO.cap(feats[o][0], 1500, seed=1))
                              for o in feats if o != idd])
        def scores(X):
            return dict(maha=idness_maha(mus, prec, X),
                        residual=idness_residual(pca, X),
                        cone=CO.idness_cone(cones, X))
        s_id, s_ood = scores(Fq), scores(ood)
        # hybrid = z(maha) + z(cone), z-scored over the full eval pool
        allmaha = np.r_[s_id["maha"], s_ood["maha"]]; allcone = np.r_[s_id["cone"], s_ood["cone"]]
        zc = z(allcone); zm = z(allmaha); hyb = zm + zc
        n = len(s_id["maha"]); s_id["hybrid"], s_ood["hybrid"] = hyb[:n], hyb[n:]
        au = {m: auroc(s_id[m], s_ood[m]) for m in methods}
        corr = float(np.corrcoef(zm, zc)[0, 1])
        rows[idd] = {**au, "corr_maha_cone": corr,
                     "hyb_gain_over_maha": au["hybrid"] - au["maha"]}
        print(f"[{idd:14s}] maha {au['maha']:.3f} resid {au['residual']:.3f} "
              f"cone {au['cone']:.3f} | HYBRID {au['hybrid']:.3f} "
              f"(Δmaha {au['hybrid']-au['maha']:+.3f}) corr {corr:+.2f}", flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    gg = [d for d in rows if d in GAUSSIANISH]; fg = [d for d in rows if d not in GAUSSIANISH]
    print("\n=== mean AUROC ===")
    for grp, name in ((gg, "Gaussian-ish"), (fg, "fine-grained"), (list(rows), "all")):
        if not grp: continue
        mm = {m: np.mean([rows[d][m] for d in grp]) for m in methods}
        print(f"  [{name:12s}] maha {mm['maha']:.3f}  cone {mm['cone']:.3f}  "
              f"hybrid {mm['hybrid']:.3f}  (Δ hybrid-maha {mm['hybrid']-mm['maha']:+.3f})")


if __name__ == "__main__":
    main()
