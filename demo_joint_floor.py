"""
demo_joint_floor.py
-------------------
Joint (offline, non-incremental) conic-hull FLOOR on a FROZEN backbone.

    1. Load all 100 CIFAR-100 classes.
    2. Frozen pretrained ViT-B/16 — NO tuning, NO LoRA (pure features).
    3. Build one conic hull per class on the frozen train features (all 100 at once).
    4. Classify the test set by argmax cone score.

This is the reference point: the best the cone classifier does on these frozen
features with every class available together (no drift, no incremental crowding,
no feature tuning).  Continual / tuned variants are measured against it, and it is
the clean substrate for plugging in cone-separation methods (feature tuning or
optimization) between BUILD and CLASSIFY — see the marked hook in main().

Reuses: cone_boundary.load_cifar100 / make_vit_backbone_factory ;
        conic_hull.build_class_conic_hulls (the same builder demo_incremental uses).

    python -u demo_joint_floor.py
"""

import os

import numpy as np
import torch
from cone_boundary import make_vit_backbone_factory
from conic_hull import build_class_conic_hulls, ConicHull
from sklearn.preprocessing import normalize as _l2n
from tqdm import tqdm

# ── knobs ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "vit_base_patch16_224.orig_in21k"
# torchvision datasets that download cleanly (one feature cache each):
#   CIFAR100(100) | FGVCAircraft(100, fine-grained) | Flowers102(102) |
#   OxfordIIITPet(37) | Food101(101).  FGVCAircraft is the multimodal test.
DATASET = "FGVCAircraft"
N_CLASSES = 100  # auto-overwritten from the data in main()
N_RAYS = 25  # extreme rays per class (match cone_boundary; raise for a higher ceiling)
USE_PCA = False  # PCA before SPA (speed); rays stored in original space
PCA_DIM = 128  # PCA dim (match cone_boundary; raise for a higher ceiling)
# ray construction: "spa"|"fps"|"hybrid" (reconstruction, via build_class_conic_hulls)
#   | "kmeans" (multi-prototype centroids) | "disc" (DISCRIMINATIVE — keep each
#   class's most distinctive samples, then place rays there; uses inter-class info)
RAY_METHOD = "disc"
DISC_KEEP_FRAC = 0.6  # disc: fraction of most-distinctive samples to keep per class
BATCH_SIZE = 64
DATA_DIR = "./data"
# Optional frozen Random-Projection + ReLU on the features (post-hoc; no re-forward)
USE_RP = False
N_PROJ = 10000
RP_SEED = 0
# Open-set OOD: build cones/means on the first OOD_ID_FRAC of classes; rest = unseen
EVAL_OOD = True
OOD_ID_FRAC = 0.8
# Separability transform applied to features before building/scoring cones.
# Fit on TRAIN (ID-only for OOD). A global linear map rearranges all cones — you
# cannot separate a cone in isolation (its class data is fixed).
# modes: "none" | "whiten" (full ZCA) | "partial_whiten" (Σ^-α) | "pca" (reduce, no
# whiten) | "lda" (discriminant).  Denoise↔decorrelate axis: pca/lda denoise (cone-
# friendly); whiten decorrelates+amplifies noise (prototype-friendly, cone-hostile).
TRANSFORM = "lda"
WHITEN_RIDGE = 1.0
WHITEN_ALPHA = 0.25  # partial_whiten strength: 0 = identity, 0.5 = full ZCA
PCA_KEEP = 512  # pca: number of top-variance components to keep
LDA_SHRINK = "auto"

SCORE_NAMES = ["cosine", "angular_margin", "blended", "max_ray_sim"]


def cache_path():
    tag = MODEL_NAME.replace("/", "_").replace(".", "_")
    return f"feats_{DATASET}_{tag}_modelnorm.npz"


def build_torchvision_datasets(name, transform, data_dir):
    """Return (train_ds, test_ds) for a clean-download torchvision dataset.  Labels
    come from the DataLoader (0..C−1), so no per-dataset label attribute needed."""
    from torchvision import datasets as D

    if name == "CIFAR100":
        return (D.CIFAR100(data_dir, train=True, download=True, transform=transform),
                D.CIFAR100(data_dir, train=False, download=True, transform=transform))
    if name == "FGVCAircraft":
        return (D.FGVCAircraft(data_dir, split="trainval", download=True, transform=transform),
                D.FGVCAircraft(data_dir, split="test", download=True, transform=transform))
    if name == "Flowers102":
        return (D.Flowers102(data_dir, split="train", download=True, transform=transform),
                D.Flowers102(data_dir, split="test", download=True, transform=transform))
    if name == "OxfordIIITPet":
        return (D.OxfordIIITPet(data_dir, split="trainval", download=True, transform=transform),
                D.OxfordIIITPet(data_dir, split="test", download=True, transform=transform))
    if name == "Food101":
        return (D.Food101(data_dir, split="train", download=True, transform=transform),
                D.Food101(data_dir, split="test", download=True, transform=transform))
    raise ValueError(f"unknown DATASET '{name}'")


@torch.no_grad()
def _extract_loader(backbone, loader, device, desc):
    F, Y = [], []
    for x, y in tqdm(loader, desc=desc, unit="batch"):
        F.append(backbone(x.to(device)).float().cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(F).astype(np.float32), np.concatenate(Y)


def get_features(device):
    """Return (Ftr, ytr, Fte, yte) from the .npz cache if present, else extract
    with the MODEL'S OWN normalization (timm data config — a frozen backbone is
    very sensitive to this) and save.  No backbone is built on a cache hit."""
    path = cache_path()
    if os.path.exists(path):
        d = np.load(path)
        print(f"[cache] loaded {path}  train{d['Ftr'].shape}  test{d['Fte'].shape}")
        return d["Ftr"], d["ytr"], d["Fte"], d["yte"]

    import timm
    from torch.utils.data import DataLoader
    from torchvision import datasets

    backbone, _ = make_vit_backbone_factory(MODEL_NAME, lora_blocks=0)(0, device)
    backbone.to(device).eval()

    cfg = timm.data.resolve_model_data_config(backbone)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print(
        f"[norm] model transform: mean={tuple(round(m, 3) for m in cfg['mean'])} "
        f"std={tuple(round(s, 3) for s in cfg['std'])} input={cfg['input_size']}"
    )

    tr, te = build_torchvision_datasets(DATASET, tf, DATA_DIR)
    nw = min(8, os.cpu_count() or 2)
    trL = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    teL = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)

    Ftr, ytr = _extract_loader(backbone, trL, device, "train feats")
    Fte, yte = _extract_loader(backbone, teL, device, "test feats")
    np.savez(path, Ftr=Ftr, ytr=ytr, Fte=Fte, yte=yte)
    print(f"[cache] saved {path}")
    return Ftr, ytr, Fte, yte


def ncm_accuracy(Ftr, ytr, Fte, yte):
    """Nearest-class-mean (cosine) — the trivial reference on the same features."""
    means = np.stack([Ftr[ytr == c].mean(axis=0) for c in range(N_CLASSES)])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
    Q = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    pred = (Q @ means.T).argmax(axis=1)
    return float((pred == yte).mean())


def maybe_rp(F, W):
    """Frozen RP + ReLU applied to precomputed features (cheap, no re-forward)."""
    return np.maximum(F @ W.T, 0.0).astype(np.float32) if W is not None else F


def fit_feature_transform(
    Ftr, ytr, mode, ridge=1e-2, lda_shrink="auto", alpha=0.5, k=200
):
    """Fit a global linear separability transform on training features and return a
    callable T(F).  All cones move together under T → the space is rearranged to be
    more class-separable, on FROZEN features (training-free, closed-form).

    Denoise ↔ decorrelate axis (cones prefer the denoise end, prototypes the other):
    'none'           : identity.
    'pca'            : project onto top-k variance dirs, NO whitening — denoise
                       without amplifying low-variance noise (cone-friendly).
    'partial_whiten' : Σ^(−α).  α=0 → identity, α=0.5 → full ZCA.  Gentle
                       decorrelation that doesn't blow up tiny eigenvalues.
    'whiten'         : ZCA Σ^(−1/2).  Decorrelates but AMPLIFIES noise dirs —
                       helps NCM/prototypes, hurts cones.
    'lda'            : discriminant projection (≤ n_classes−1 dims) — supervised
                       denoise + between-class emphasis (best closed-set for cones).
    """
    if mode == "none":
        return lambda F: F

    if mode in ("whiten", "partial_whiten", "pca"):
        X = np.asarray(Ftr, np.float64)
        mu = X.mean(axis=0)
        Z = X - mu
        cov = (Z.T @ Z) / max(len(Z) - 1, 1)
        mu32 = mu.astype(np.float32)

        if mode == "pca":  # reduce only, no scaling
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1][: min(k, cov.shape[0])]
            V = vecs[:, order].astype(np.float32)
            return lambda F: ((np.asarray(F, np.float32) - mu32) @ V).astype(np.float32)

        a = 0.5 if mode == "whiten" else float(alpha)  # Σ^(−a)
        vals, vecs = np.linalg.eigh(cov + ridge * np.eye(cov.shape[0]))
        vals = np.clip(vals, 1e-12, None)
        Wt = ((vecs * (vals ** (-a))) @ vecs.T).astype(np.float32)
        return lambda F: ((np.asarray(F, np.float32) - mu32) @ Wt).astype(np.float32)

    if mode == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=lda_shrink)
        lda.fit(np.asarray(Ftr, np.float64), ytr)
        return lambda F: lda.transform(np.asarray(F, np.float64)).astype(np.float32)

    raise ValueError(f"unknown TRANSFORM '{mode}'")


def _rays_to_hull(rays, k_local=None):
    """Wrap a (K, D) ray matrix in a ConicHull (no fit; rays set directly)."""
    rays = _l2n(np.asarray(rays, np.float64), axis=1).astype(np.float32)
    h = ConicHull(n_rays=len(rays), use_pca=False, k_local=k_local)
    h.extreme_rays_ = rays  # setter clears the GPU cache
    h.extreme_rays_index = None
    return h


def build_custom_hulls(feature_dict, n_rays, method, disc_keep_frac=0.6):
    """Build one ConicHull per class with DISCRIMINATIVE or multi-prototype rays.

    'kmeans' : spherical k-means centroids of the class features (tight, mode-
               capturing multi-prototype — uses only within-class info).
    'disc'   : keep each class's most DISTINCTIVE samples — high cosine to own
               class mean, low to the nearest OTHER class mean — then k-means among
               them.  Builds the cone from the discriminative core (drops confusable
               boundary points and outliers), using inter-class info SPA ignores.
    """
    from sklearn.cluster import KMeans

    classes = list(feature_dict)
    means = np.stack(
        [_l2n(feature_dict[c].mean(axis=0)[None], axis=1)[0] for c in classes]
    )  # (C, D) unit means
    cidx = {c: i for i, c in enumerate(classes)}

    def centroids(X, k):
        Xn = _l2n(np.asarray(X, np.float64), axis=1)
        if k >= len(Xn):
            return Xn
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=0).fit(Xn)
        return km.cluster_centers_

    hulls = {}
    for c in tqdm(classes, desc=f"build {method}", unit="cls"):
        Xc = np.asarray(feature_dict[c], np.float32)
        if method == "disc":
            sims = _l2n(Xc, axis=1) @ means.T  # (n, C) cos to all means
            own = sims[:, cidx[c]].copy()
            sims[:, cidx[c]] = -np.inf
            margin = own - sims.max(axis=1)  # distinctive − confusable
            n_keep = max(n_rays, int(disc_keep_frac * len(Xc)))
            keep = np.argsort(margin)[::-1][:n_keep]  # most-distinctive samples
            Xc = Xc[keep]
        hulls[c] = _rays_to_hull(centroids(Xc, min(n_rays, len(Xc))))
    return hulls


def build_hulls(feature_dict):
    """Dispatch cone construction on RAY_METHOD."""
    if RAY_METHOD in ("spa", "fps", "hybrid"):
        return build_class_conic_hulls(
            feature_dict,
            n_rays=N_RAYS,
            use_pca=USE_PCA,
            pca_dim=PCA_DIM,
            ray_diversity=RAY_METHOD,
            spa_oversample=3,
        )
    return build_custom_hulls(
        feature_dict, n_rays=N_RAYS, method=RAY_METHOD, disc_keep_frac=DISC_KEEP_FRAC
    )


def classify_all_scores(test_feats, test_labels, hulls):
    """argmax over per-class cone scores, for every scoring scheme, on precomputed
    test features (one NNLS reconstruction per class, reused across schemes)."""
    classes = list(hulls.keys())
    per_class = [
        hulls[c].score_all(test_feats)
        for c in tqdm(classes, desc="score test", unit="cls")
    ]
    out = {}
    for s in SCORE_NAMES:
        mat = np.stack([pc[s] for pc in per_class], axis=1)  # (N, C)
        pred = np.array([int(classes[i]) for i in mat.argmax(axis=1)])
        out[s] = float((pred == test_labels).mean())
    return out


def auroc_fpr(id_scores, ood_scores):
    """AUROC and FPR@95%TPR for a detector where HIGHER score = in-distribution.
    AUROC 1.0 = perfect ID/OOD separation; FPR@95 = OOD wrongly kept while keeping
    95% of ID (lower is better)."""
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(y, s))
    thr = np.percentile(id_scores, 5)  # keep 95% of ID
    fpr95 = float((ood_scores >= thr).mean())
    return auroc, fpr95


def ood_eval(
    Ftr,
    ytr,
    Fte,
    yte,
    id_classes,
    ood_classes,
    transform="none",
    ridge=1e-2,
    lda_shrink="auto",
    alpha=0.5,
    k=200,
):
    """Open-set OOD: build cones + class means on ID classes only, then score
    ID-test vs OOD-test (unseen classes) by max membership.  HIGHER = ID.
    Compares the cone's calibrated boundary against NCM max-cosine.

    The separability transform is fit on ID TRAIN ONLY (OOD is unseen at fit time)
    and applied to all features — keeping the open-set protocol honest."""
    id_tr = np.isin(ytr, id_classes)
    T = fit_feature_transform(
        Ftr[id_tr], ytr[id_tr], transform, ridge, lda_shrink, alpha=alpha, k=k
    )
    Ftr = T(Ftr)

    fd = {str(c): Ftr[ytr == c] for c in id_classes}
    hulls = build_hulls(fd)
    means = np.stack([Ftr[ytr == c].mean(axis=0) for c in id_classes])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-8

    id_mask, ood_mask = np.isin(yte, id_classes), np.isin(yte, ood_classes)
    F_all = T(np.concatenate([Fte[id_mask], Fte[ood_mask]]))
    n_id = int(id_mask.sum())

    per = [
        hulls[c].score_all(F_all)
        for c in tqdm(list(hulls), desc="ood score", unit="cls")
    ]
    out = {}
    # membership schemes + geo_residual = raw NNLS distance-to-cone (monotone in the
    # reconstruction residual; higher = better reconstructed = more in-distribution)
    ood_keys = list(SCORE_NAMES)
    if per and "geo_residual" in per[0]:
        ood_keys = ood_keys + ["geo_residual"]
    for s in ood_keys:  # cone: max in-cone membership / min residual
        msc = np.stack([pc[s] for pc in per], axis=1).max(axis=1)
        out["cone:" + s] = auroc_fpr(msc[:n_id], msc[n_id:])
    Q = F_all / (np.linalg.norm(F_all, axis=1, keepdims=True) + 1e-8)
    ncm_sc = (Q @ means.T).max(axis=1)  # NCM: max cosine to means
    out["ncm"] = auroc_fpr(ncm_sc[:n_id], ncm_sc[n_id:])
    return out, n_id, len(F_all) - n_id


def main():
    global N_CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] no CUDA — ViT-B on CPU will be very slow.")
    print(
        f"[setup] {DATASET} + frozen {MODEL_NAME} on {device}  "
        f"(rays={RAY_METHOD}×{N_RAYS}, transform={TRANSFORM}, rp={USE_RP})"
    )

    # ── 1-3. features (model's own normalization), cached to disk ───────────────
    Ftr, ytr_np, Fte, yte_np = get_features(device)
    d_feat = Ftr.shape[1]
    N_CLASSES = int(max(int(ytr_np.max()), int(yte_np.max()))) + 1  # from the data
    print(f"[data] {DATASET}: {N_CLASSES} classes, "
          f"train {Ftr.shape[0]}, test {Fte.shape[0]}, dim {d_feat}")

    W = None
    if USE_RP:
        g = torch.Generator().manual_seed(RP_SEED)
        W = (
            (torch.randn(N_PROJ, d_feat, generator=g) / (d_feat**0.5))
            .numpy()
            .astype(np.float32)
        )
        Ftr, Fte = maybe_rp(Ftr, W), maybe_rp(Fte, W)
        print(f"[rp] features expanded {d_feat} → {N_PROJ} (ReLU)")

    # ── separability transform for closed-set (fit on all-100 train).  Keep raw
    #    Ftr/Fte intact: OOD below refits the transform on ID classes only. ───────
    T = fit_feature_transform(
        Ftr,
        ytr_np,
        TRANSFORM,
        ridge=WHITEN_RIDGE,
        lda_shrink=LDA_SHRINK,
        alpha=WHITEN_ALPHA,
        k=PCA_KEEP,
    )
    Ftr_t, Fte_t = T(Ftr), T(Fte)
    if TRANSFORM != "none":
        print(f"[transform] {TRANSFORM} applied → feature dim {Ftr_t.shape[1]}")

    # ── NCM baseline (reference for what the frozen features support) ────────────
    ncm = ncm_accuracy(Ftr_t, ytr_np, Fte_t, yte_np)

    # ── 4. build one cone per class (all 100 at once) ───────────────────────────
    feature_dict = {str(c): Ftr_t[ytr_np == c] for c in range(N_CLASSES)}
    hulls = build_hulls(feature_dict)

    # ── 5. classify the test set ────────────────────────────────────────────────
    accs = classify_all_scores(Fte_t, yte_np, hulls)

    print("\n" + "=" * 56)
    print(
        f"Joint FLOOR — {DATASET} / frozen {MODEL_NAME}  "
        f"(transform={TRANSFORM}, rays={RAY_METHOD})"
    )
    print(f"  {N_CLASSES} classes, {N_RAYS} rays/class, rp={USE_RP}")
    print("=" * 56)
    print(f"  {'NCM (cosine-to-mean)':<22} {ncm:.4f}   <- reference")
    print("  " + "-" * 40)
    for s in SCORE_NAMES:
        star = " *" if s == "cosine" else ""
        print(f"  {'cone:' + s:<22} {accs[s]:.4f}{star}")
    best = max(accs, key=accs.get)
    gap = ncm - accs[best]
    print(f"\n  best cone scheme: {best} = {accs[best]:.4f}   |   NCM = {ncm:.4f}")
    print(
        f"  cone vs NCM gap: {gap:+.4f}  "
        f"({'cone scoring is the bottleneck' if gap > 0.03 else 'cones competitive'})"
    )

    # ── open-set OOD: cones vs NCM at telling seen from unseen classes ───────────
    if EVAL_OOD:
        n_id_classes = max(1, int(round(OOD_ID_FRAC * N_CLASSES)))
        id_cls, ood_cls = list(range(n_id_classes)), list(range(n_id_classes, N_CLASSES))
        res, n_id, n_ood = ood_eval(
            Ftr,
            ytr_np,
            Fte,
            yte_np,
            id_cls,
            ood_cls,
            transform=TRANSFORM,
            ridge=WHITEN_RIDGE,
            lda_shrink=LDA_SHRINK,
            alpha=WHITEN_ALPHA,
            k=PCA_KEEP,
        )
        print("\n" + "=" * 56)
        print(
            f"OPEN-SET OOD (transform={TRANSFORM}) — {n_id_classes} ID vs "
            f"{N_CLASSES - n_id_classes} unseen (ID test {n_id}, OOD test {n_ood})"
        )
        print("=" * 56)
        print(f"  {'detector':<22} {'AUROC':>7} {'FPR@95':>8}")
        print("  " + "-" * 40)
        ordered = ["ncm"] + [k for k in res if k.startswith("cone:")]
        for k in ordered:
            au, fpr = res[k]
            print(f"  {k:<22} {au:>7.4f} {fpr:>8.4f}")
        best_c = max((k for k in res if k.startswith("cone:")), key=lambda k: res[k][0])
        d = res[best_c][0] - res["ncm"][0]
        print(
            f"\n  best cone AUROC: {res[best_c][0]:.4f} ({best_c})  |  "
            f"NCM AUROC: {res['ncm'][0]:.4f}  (Δ={d:+.4f})"
        )
        print(
            f"  {'cone beats NCM on OOD' if d > 0 else 'NCM ahead on OOD'} "
            "— this is the cone primitive's structural edge."
        )


if __name__ == "__main__":
    main()
