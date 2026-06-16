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
# Datasets (one feature cache each).  torchvision (clean download):
#   CIFAR100(100) | FGVCAircraft(100) | Flowers102(102) | OxfordIIITPet(37) |
#   Food101(101).  HuggingFace (CUB/Cars aren't cleanly in torchvision):
#   CUB200(200, fine-grained) | StanfordCars(196, fine-grained).
DATASET = "FGVCAircraft"
# HF repos for the non-torchvision sets (repo names drift on the Hub — override if
# load_dataset fails). Format: name -> (repo, train_split, test_split).
HF_REPOS = {
    "CUB200": ("Donghyun99/CUB-200-2011", "train", "test"),
    "StanfordCars": ("Donghyun99/Stanford-Cars", "train", "test"),
}
N_CLASSES = 100  # auto-overwritten from the data in main()
# Multimodal test: randomly merge MERGE_K fine classes into one label (each merged
# class then has K dissimilar sub-modes — a single mean can't cover it, a cone can).
# MERGE_K=1 → no merge.  Reuses the feature cache (labels remapped only).
MERGE_K = 5
MERGE_SEED = 0
# Use CIFAR-100's REAL semantic superclasses (100 fine → 20 coarse, related sub-
# modes) instead of random merge.  CIFAR-100 only; overrides MERGE_K when True.
SEMANTIC_COARSE = False
# Keep only the first CLASS_LIMIT classes (after any merge). 0 = all.  Use this to
# DISENTANGLE "few classes" from "multimodal": e.g. CLASS_LIMIT=20 + MERGE_K=1
# (20 unimodal) vs MERGE_K=5 (20 multimodal) — same class count, different modality.
CLASS_LIMIT = 0
# Cap train samples PER CLASS (0 = all).  Matches data volume across arms so the
# multimodal win can't be attributed to more data (merge-5 has 5× samples/class).
SAMPLES_PER_CLASS = 0
SUBSET_SEED = 0
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


class _HFImageDataset(torch.utils.data.Dataset):
    """Wrap a HuggingFace image split as (transform(image), label) for a DataLoader."""

    def __init__(self, hf_ds, transform, img_key, lbl_key):
        self.ds, self.tf, self.ik, self.lk = hf_ds, transform, img_key, lbl_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r[self.ik]
        if not hasattr(img, "mode"):                  # bytes dict → PIL
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(r[self.lk])


def build_hf_datasets(name, transform):
    """Load a HuggingFace image dataset (CUB/Cars), auto-detecting the image/label
    columns and falling back validation→test for the eval split."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` needed for CUB/Cars: "
                          "pip install datasets") from e
    repo, tr_split, te_split = HF_REPOS[name]
    dd = load_dataset(repo)
    if te_split not in dd:                             # some repos use 'validation'
        te_split = "validation" if "validation" in dd else tr_split
    tr, te = dd[tr_split], dd[te_split]
    cols = tr.column_names
    ik = next((c for c in ("image", "img", "Image", "picture") if c in cols), None)
    lk = next((c for c in ("label", "labels", "fine_label", "class", "target")
               if c in cols), None)
    if ik is None or lk is None:
        raise ValueError(f"{repo}: can't find image/label cols in {cols}")
    print(f"[hf] {repo}: splits {tr_split}/{te_split}, cols image='{ik}' label='{lk}'")
    return (_HFImageDataset(tr, transform, ik, lk),
            _HFImageDataset(te, transform, ik, lk))


def build_torchvision_datasets(name, transform, data_dir):
    """Return (train_ds, test_ds).  HF for CUB/Cars; else clean-download torchvision.
    Labels come from the DataLoader (0..C−1), so no per-dataset label attr needed."""
    if name in HF_REPOS:
        return build_hf_datasets(name, transform)

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


def check_feature_health(Ftr, Fte, ytr, yte, name):
    """Catch normalization / extraction errors on a new dataset.  Healthy frozen
    ViT features have O(10) L2 norms, no NaN/Inf, no near-zero rows, and a NCM
    train accuracy well above chance.  A mis-normalized transform (the CIFAR bug)
    shows up as collapsed/erratic norms and near-chance train NCM."""
    norms = np.linalg.norm(Ftr, axis=1)
    n_cls = int(max(int(ytr.max()), int(yte.max()))) + 1
    flags = []
    if not np.isfinite(Ftr).all() or not np.isfinite(Fte).all():
        flags.append("NaN/Inf!")
    if (norms < 1e-6).mean() > 0.01:
        flags.append(f"{(norms < 1e-6).mean():.1%} zero rows")
    # train-on-train NCM: bad features ⇒ even this is near chance (1/n_cls)
    tr_acc = ncm_accuracy_generic(Ftr, ytr, Ftr, ytr, n_cls)
    if tr_acc < 3.0 / n_cls:
        flags.append(f"train-NCM {tr_acc:.2f} ≈ chance ({1/n_cls:.3f}) — suspect norm")
    status = "  ⚠ " + "; ".join(flags) if flags else "  ✓"
    print(f"[health:{name}] norm mean={norms.mean():.1f} "
          f"min={norms.min():.1f} max={norms.max():.1f} dim={Ftr.shape[1]}  "
          f"train-NCM={tr_acc:.3f} (chance {1/n_cls:.3f}){status}")


def merge_labels(ytr, yte, k, seed=0):
    """Randomly partition fine classes into groups of k → multimodal coarse labels.
    Reuses cached features; only labels are remapped.  Returns (ytr', yte')."""
    n_fine = int(max(int(ytr.max()), int(yte.max()))) + 1
    perm = np.random.default_rng(seed).permutation(n_fine)
    fine2coarse = np.empty(n_fine, dtype=np.int64)
    for gi, start in enumerate(range(0, n_fine, k)):
        fine2coarse[perm[start:start + k]] = gi
    return fine2coarse[ytr], fine2coarse[yte], int(np.ceil(n_fine / k))


# CIFAR-100 fine→coarse map (fine label 0..99 in torchvision's alphabetical order →
# one of the 20 official superclasses).  Each superclass groups 5 RELATED fine
# classes (e.g. trees = maple/oak/palm/pine/willow) — real, milder multimodality
# than random merge.
_CIFAR100_COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13], dtype=np.int64)


def cifar100_coarse_labels(ytr, yte):
    """Map CIFAR-100 fine labels (0..99) to the 20 official semantic superclasses.
    Only valid for CIFAR-100 features extracted in torchvision's default order."""
    return _CIFAR100_COARSE[ytr], _CIFAR100_COARSE[yte], 20


def ncm_accuracy_generic(Ftr, ytr, Fte, yte, n_classes):
    """NCM (cosine-to-mean) for an explicit class count (used by the health check)."""
    means = np.stack([Ftr[ytr == c].mean(axis=0) for c in range(n_classes)])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
    Q = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    return float(((Q @ means.T).argmax(axis=1) == yte).mean())


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

    # normalization / extraction sanity check for the (possibly new) dataset
    check_feature_health(Ftr, Fte, ytr_np, yte_np, DATASET)

    # multimodal labels: CIFAR-100 semantic superclasses, else random merge-K
    if SEMANTIC_COARSE and DATASET == "CIFAR100":
        ytr_np, yte_np, n_coarse = cifar100_coarse_labels(ytr_np, yte_np)
        print(f"[coarse] CIFAR-100 semantic superclasses: 100 → {n_coarse}")
    elif MERGE_K > 1:
        n_fine = int(max(int(ytr_np.max()), int(yte_np.max()))) + 1
        ytr_np, yte_np, n_coarse = merge_labels(ytr_np, yte_np, MERGE_K, MERGE_SEED)
        print(f"[merge] random merge-{MERGE_K}: {n_fine} → {n_coarse} multimodal classes")

    if CLASS_LIMIT and CLASS_LIMIT > 0:           # few-classes vs multimodal control
        trm, tem = ytr_np < CLASS_LIMIT, yte_np < CLASS_LIMIT
        Ftr, ytr_np, Fte, yte_np = Ftr[trm], ytr_np[trm], Fte[tem], yte_np[tem]
        print(f"[limit] kept first {CLASS_LIMIT} classes "
              f"(train {Ftr.shape[0]}, test {Fte.shape[0]})")

    if SAMPLES_PER_CLASS and SAMPLES_PER_CLASS > 0:   # match data volume across arms
        rng = np.random.default_rng(SUBSET_SEED)
        keep = []
        for c in np.unique(ytr_np):
            idx = np.where(ytr_np == c)[0]
            keep.append(rng.choice(idx, SAMPLES_PER_CLASS, replace=False)
                        if len(idx) > SAMPLES_PER_CLASS else idx)
        keep = np.concatenate(keep)
        Ftr, ytr_np = Ftr[keep], ytr_np[keep]
        print(f"[subsample] ≤{SAMPLES_PER_CLASS} train/class → {len(ytr_np)} total")

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
    d = accs[best] - ncm  # cone − NCM; positive = cone ahead
    verdict = "cone ahead" if d > 0.003 else "NCM ahead" if d < -0.003 else "tied"
    print(f"\n  best cone scheme: {best} = {accs[best]:.4f}   |   NCM = {ncm:.4f}")
    print(f"  cone − NCM: {d:+.4f}  ({verdict})")

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
