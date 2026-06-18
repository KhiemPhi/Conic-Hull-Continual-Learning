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
# Hierarchical coarse→fine eval: coarse route with a CONE per superclass (rays =
# its fine-class centroids), then fine = nearest centroid in the routed group.
# Compared vs flat 100-way prototype, NCM-coarse router, and oracle.  CIFAR-100.
RUN_HIERARCHICAL = False
# Soft routing to avoid hard-route error propagation:
#   top-k : keep top-k coarse cones, fine-rank over their union.
#   fusion: score(f) = α·cone(g(f)) + (1−α)·cos(q,cent_f)   (α=0 ⇒ flat NCM).
HIER_TOPK = [1, 2, 3, 5]
HIER_ALPHA = [0.0, 0.2, 0.4, 0.6, 0.8]
# Keep only the first CLASS_LIMIT classes (after any merge). 0 = all.  Use this to
# DISENTANGLE "few classes" from "multimodal": e.g. CLASS_LIMIT=20 + MERGE_K=1
# (20 unimodal) vs MERGE_K=5 (20 multimodal) — same class count, different modality.
CLASS_LIMIT = 0
# Cap train samples PER CLASS (0 = all).  Matches data volume across arms so the
# multimodal win can't be attributed to more data (merge-5 has 5× samples/class).
SAMPLES_PER_CLASS = 0
SUBSET_SEED = 0
N_RAYS = 50  # extreme rays per class (match cone_boundary; raise for a higher ceiling)
USE_PCA = False  # PCA before SPA (speed); rays stored in original space
PCA_DIM = 128  # PCA dim (match cone_boundary; raise for a higher ceiling)
# ray construction: "spa"|"fps"|"hybrid" (reconstruction, via build_class_conic_hulls)
#   | "kmeans" (multi-prototype centroids) | "disc" (DISCRIMINATIVE — keep each
#   class's most distinctive samples, then place rays there; uses inter-class info)
#   | "mixture" (UNION of N_CONES small cones per class — the structural fix for
#   cone(k) << multiproto-NCM(k): tile the class into local pieces, conic shape per
#   piece, score = max over pieces; capacity-matched to a single cone with N_RAYS)
RAY_METHOD = "mixture"
N_CONES = 2
DISC_KEEP_FRAC = 0.9  # disc: fraction of most-distinctive samples to keep per class
# mixture-of-cones (RAY_METHOD="mixture"): split each class into N_CONES sub-clusters
# (spherical k-means) and fit a small cone (~N_RAYS//N_CONES rays) on each.  Total
# generators stay ≈ N_RAYS, so it's capacity-matched to the single fat cone AND to
# multiproto-NCM(k=N_RAYS) — the only change is piecewise-local tiling.  1 = single cone.
N_CONES = 5
MIXTURE_SUB_METHOD = "spa"  # how each sub-cone's rays are built: "spa" | "kmeans"
MIXTURE_SEED = 0
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
# modes: "none" | "whiten" (full ZCA) | "partial_whiten" (Σ^-α) | "none" (reduce, no
# whiten) | "lda" (discriminant).  Denoise↔decorrelate axis: pca/lda denoise (cone-
# friendly); whiten decorrelates+amplifies noise (prototype-friendly, cone-hostile).
TRANSFORM = "none"
WHITEN_RIDGE = 1.0
WHITEN_ALPHA = 0.25  # partial_whiten strength: 0 = identity, 0.5 = full ZCA
PCA_KEEP = 512  # pca: number of top-variance components to keep
LDA_SHRINK = "auto"

# ── dataset-identification test ─────────────────────────────────────────────────
# Treat each ENTIRE dataset as ONE class — the ultimate multimodal class: it spans
# all its fine classes / many modes.  Build 1 cone vs 1 mean per dataset, then ask
# "which dataset is this query from?"  If the thesis holds (a class is a region, not
# a point), the cone should identify the source dataset far better than a single
# mean, because one mean cannot cover a whole dataset's spread while the cone's rays
# can sit on each mode.  Overrides the normal joint-floor run when DATASET_ID=True.
DATASET_ID = True
ID_DATASETS = ["Food101"]
ID_TRAIN_PER = 0  # cap train samples per dataset (balance + speed; 0 = all)
ID_TEST_PER = 0  # cap test samples per dataset → balanced test, chance = 1/D
ID_TRANSFORM = "none"  # pca/none preserve modes (cone-friendly); lda collapses them
# Split EACH dataset's fine classes into K random groups → D×K multimodal classes
# (each group = a random subset of one dataset's fine classes, so it carries many
# dissimilar sub-modes).  Probes cone vs mean at a finer granularity than whole-
# dataset-id, while still rolling group predictions up to a "which dataset?" answer.
# 1 = no split (whole dataset = one class, the original behavior).
ID_SPLIT_K = 10
ID_SPLIT_SEED = 0
# Cross-dataset grouping: MERGE group i across ALL datasets into one class → K total
# classes, each spanning every dataset (its sub-modes are fine classes drawn from
# different datasets — maximally dissimilar multimodality).  False keeps groups
# inside their dataset (D×K classes, with a which-dataset rollup).  Needs K>1.
ID_CROSS_DATASET = False

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
        if not hasattr(img, "mode"):  # bytes dict → PIL
            import io

            from PIL import Image

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
        raise ImportError(
            "HuggingFace `datasets` needed for CUB/Cars: " "pip install datasets"
        ) from e
    repo, tr_split, te_split = HF_REPOS[name]
    dd = load_dataset(repo)
    if te_split not in dd:  # some repos use 'validation'
        te_split = "validation" if "validation" in dd else tr_split
    tr, te = dd[tr_split], dd[te_split]
    cols = tr.column_names
    ik = next((c for c in ("image", "img", "Image", "picture") if c in cols), None)
    lk = next(
        (c for c in ("label", "labels", "fine_label", "class", "target") if c in cols),
        None,
    )
    if ik is None or lk is None:
        raise ValueError(f"{repo}: can't find image/label cols in {cols}")
    print(f"[hf] {repo}: splits {tr_split}/{te_split}, cols image='{ik}' label='{lk}'")
    return (
        _HFImageDataset(tr, transform, ik, lk),
        _HFImageDataset(te, transform, ik, lk),
    )


def build_torchvision_datasets(name, transform, data_dir):
    """Return (train_ds, test_ds).  HF for CUB/Cars; else clean-download torchvision.
    Labels come from the DataLoader (0..C−1), so no per-dataset label attr needed."""
    if name in HF_REPOS:
        return build_hf_datasets(name, transform)

    from torchvision import datasets as D

    if name == "CIFAR100":
        return (
            D.CIFAR100(data_dir, train=True, download=True, transform=transform),
            D.CIFAR100(data_dir, train=False, download=True, transform=transform),
        )
    if name == "FGVCAircraft":
        return (
            D.FGVCAircraft(
                data_dir, split="trainval", download=True, transform=transform
            ),
            D.FGVCAircraft(data_dir, split="test", download=True, transform=transform),
        )
    if name == "Flowers102":
        return (
            D.Flowers102(data_dir, split="train", download=True, transform=transform),
            D.Flowers102(data_dir, split="test", download=True, transform=transform),
        )
    if name == "OxfordIIITPet":
        return (
            D.OxfordIIITPet(
                data_dir, split="trainval", download=True, transform=transform
            ),
            D.OxfordIIITPet(data_dir, split="test", download=True, transform=transform),
        )
    if name == "Food101":
        return (
            D.Food101(data_dir, split="train", download=True, transform=transform),
            D.Food101(data_dir, split="test", download=True, transform=transform),
        )
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
    print(
        f"[health:{name}] norm mean={norms.mean():.1f} "
        f"min={norms.min():.1f} max={norms.max():.1f} dim={Ftr.shape[1]}  "
        f"train-NCM={tr_acc:.3f} (chance {1/n_cls:.3f}){status}"
    )


def merge_labels(ytr, yte, k, seed=0):
    """Randomly partition fine classes into groups of k → multimodal coarse labels.
    Reuses cached features; only labels are remapped.  Returns (ytr', yte')."""
    n_fine = int(max(int(ytr.max()), int(yte.max()))) + 1
    perm = np.random.default_rng(seed).permutation(n_fine)
    fine2coarse = np.empty(n_fine, dtype=np.int64)
    for gi, start in enumerate(range(0, n_fine, k)):
        fine2coarse[perm[start : start + k]] = gi
    return fine2coarse[ytr], fine2coarse[yte], int(np.ceil(n_fine / k))


# CIFAR-100 fine→coarse map (fine label 0..99 in torchvision's alphabetical order →
# one of the 20 official superclasses).  Each superclass groups 5 RELATED fine
# classes (e.g. trees = maple/oak/palm/pine/willow) — real, milder multimodality
# than random merge.
_CIFAR100_COARSE = np.array(
    [
        4,
        1,
        14,
        8,
        0,
        6,
        7,
        7,
        18,
        3,
        3,
        14,
        9,
        18,
        7,
        11,
        3,
        9,
        7,
        11,
        6,
        11,
        5,
        10,
        7,
        6,
        13,
        15,
        3,
        15,
        0,
        11,
        1,
        10,
        12,
        14,
        16,
        9,
        11,
        5,
        5,
        19,
        8,
        8,
        15,
        13,
        14,
        17,
        18,
        10,
        16,
        4,
        17,
        4,
        2,
        0,
        17,
        4,
        18,
        17,
        10,
        3,
        2,
        12,
        12,
        16,
        12,
        1,
        9,
        19,
        2,
        10,
        0,
        1,
        16,
        12,
        9,
        13,
        15,
        13,
        16,
        19,
        2,
        4,
        6,
        19,
        5,
        5,
        8,
        19,
        18,
        1,
        2,
        15,
        6,
        0,
        17,
        8,
        14,
        13,
    ],
    dtype=np.int64,
)


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

    if mode in ("whiten", "partial_whiten", "none"):
        X = np.asarray(Ftr, np.float64)
        mu = X.mean(axis=0)
        Z = X - mu
        cov = (Z.T @ Z) / max(len(Z) - 1, 1)
        mu32 = mu.astype(np.float32)

        if mode == "none":  # reduce only, no scaling
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


class MixtureConicHull:
    """A class modeled as a UNION of m small cones instead of one fat cone.

    The single-cone-with-k-rays representation is ONE unbounded conic region: with k
    generators it describes the *shape* of a single piece but spends nothing on
    locality.  Multipoint-NCM with k prototypes instead spends its budget on k *local*
    cells, which is exactly what the stage-oracle / class-membership job rewards — and
    that's why cone(k) << multiproto-NCM(k).

    This wrapper closes the structural gap: split a class into m sub-clusters and put a
    small cone (~k/m rays) on each, so the class becomes a piecewise-local tiling
    (like k-NCM) while keeping the conic shape model INSIDE each piece.  A query's class
    score is the MAX over the sub-cones — membership in ANY piece = membership in the
    class.  All score_all() keys are higher-is-better, so the elementwise max picks the
    best-explaining sub-cone per query.

    Conforms to the hull interface used by the benchmark (score / score_all /
    extreme_rays_), so it drops into classify_all_scores, run_dataset_id and ood_eval
    unchanged."""

    def __init__(self, sub_hulls):
        self.sub_hulls = list(sub_hulls)

    @property
    def extreme_rays_(self):
        """Concatenated rays of all sub-cones (for external code that stacks rays)."""
        return np.concatenate([h.extreme_rays_ for h in self.sub_hulls])

    def score(self, queries):
        return np.max(
            np.stack([h.score(queries) for h in self.sub_hulls], axis=0), axis=0
        )

    def score_all(self, queries):
        per = [h.score_all(queries) for h in self.sub_hulls]
        return {
            k: np.max(np.stack([p[k] for p in per], axis=0), axis=0) for k in per[0]
        }


def build_mixture_hulls(feature_dict, n_rays, n_cones, sub_method="spa", seed=0):
    """Build one MixtureConicHull per class: n_cones sub-cones, ~n_rays//n_cones rays
    each, so the per-class generator budget stays ≈ n_rays (capacity-matched to the
    single cone and to multiproto-NCM(k=n_rays)).  Sub-clusters come from spherical
    k-means; each sub-cone's rays come from SPA ('spa') or k-means centroids ('kmeans')
    on its sub-cluster."""
    from sklearn.cluster import KMeans

    classes = list(feature_dict)
    rays_per = max(2, n_rays // max(1, n_cones))

    def centroids(X, k):
        Xn = _l2n(np.asarray(X, np.float64), axis=1)
        if k >= len(Xn):
            return Xn
        return KMeans(
            n_clusters=k, n_init=1, max_iter=50, random_state=seed
        ).fit(Xn).cluster_centers_

    hulls = {}
    for c in tqdm(classes, desc=f"build mixture×{n_cones}", unit="cls"):
        Xc = np.asarray(feature_dict[c], np.float32)
        m = min(n_cones, len(Xc))
        if m <= 1:  # too few samples → single piece
            sub_labels = np.zeros(len(Xc), dtype=int)
        else:
            Xn = _l2n(np.asarray(Xc, np.float64), axis=1)
            sub_labels = KMeans(
                n_clusters=m, n_init=1, max_iter=50, random_state=seed
            ).fit(Xn).labels_
        subs = []
        for g in range(max(1, m)):
            Xg = Xc[sub_labels == g]
            if len(Xg) == 0:
                continue
            if sub_method == "kmeans":
                subs.append(_rays_to_hull(centroids(Xg, min(rays_per, len(Xg)))))
            else:  # "spa": reconstruction rays on the sub-cluster
                h = ConicHull(
                    n_rays=min(rays_per, len(Xg)),
                    use_pca=USE_PCA,
                    pca_dim=PCA_DIM,
                    ray_diversity="spa",
                    spa_oversample=3,
                )
                h.fit(Xg)
                subs.append(h)
        hulls[c] = MixtureConicHull(subs)
    return hulls


def build_hulls(feature_dict):
    """Dispatch cone construction on RAY_METHOD."""
    if RAY_METHOD == "mixture":
        return build_mixture_hulls(
            feature_dict,
            n_rays=N_RAYS,
            n_cones=N_CONES,
            sub_method=MIXTURE_SUB_METHOD,
            seed=MIXTURE_SEED,
        )
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
    Compares the cone's calibrated boundary against BOTH the single-mean NCM
    max-cosine and the capacity-matched multi-prototype (k-NCM, k=N_RAYS) max-cosine —
    so a cone OOD win is attributable to the conic region boundary, not just to having
    k prototypes (the same control as the closed-set multiproto baseline).

    The separability transform is fit on ID TRAIN ONLY (OOD is unseen at fit time)
    and applied to all features — keeping the open-set protocol honest."""
    from sklearn.cluster import KMeans

    id_tr = np.isin(ytr, id_classes)
    T = fit_feature_transform(
        Ftr[id_tr], ytr[id_tr], transform, ridge, lda_shrink, alpha=alpha, k=k
    )
    Ftr = T(Ftr)

    fd = {str(c): Ftr[ytr == c] for c in id_classes}
    hulls = build_hulls(fd)
    means = np.stack([Ftr[ytr == c].mean(axis=0) for c in id_classes])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-8

    # capacity-matched point detector: k=N_RAYS k-means centroids per ID class
    mp_cents = []
    for c in id_classes:
        Xn = _l2n(np.asarray(Ftr[ytr == c], np.float64), axis=1)
        kk = min(N_RAYS, len(Xn))
        C = (
            _l2n(
                KMeans(n_clusters=kk, n_init=1, max_iter=50, random_state=0)
                .fit(Xn)
                .cluster_centers_,
                axis=1,
            )
            if kk < len(Xn)
            else Xn
        )
        mp_cents.append(C.astype(np.float32))
    mp_cents = np.concatenate(mp_cents)

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
    ncm_sc = (Q @ means.T).max(axis=1)  # NCM: max cosine to single mean
    out["ncm"] = auroc_fpr(ncm_sc[:n_id], ncm_sc[n_id:])
    mp_sc = (Q @ mp_cents.T).max(axis=1)  # multiproto: max cosine to ANY centroid
    out["multiproto"] = auroc_fpr(mp_sc[:n_id], mp_sc[n_id:])
    return out, n_id, len(F_all) - n_id


def hierarchical_eval(Ftr, ytr, Fte, yte, coarse_of, transform="none"):
    """Coarse(cone)→fine(centroid) hierarchy vs flat prototype, on fine labels.

    Structure is SHARED: per fine class f a unit centroid (fine prototype); a coarse
    class g's cone has rays = the centroids of its member fine classes.
      coarse: argmax_g cone_g membership(q)         (cone wins — coarse is multimodal)
      fine  : argmax_{f∈g*} cos(q, centroid_f)      (prototype wins — fine is unimodal)
    Reports flat 100-way NCM, hier with CONE vs NCM coarse routers, and oracle-coarse
    (ceiling), plus coarse routing accuracy for cone vs NCM."""
    T = fit_feature_transform(
        Ftr,
        ytr,
        transform,
        ridge=WHITEN_RIDGE,
        lda_shrink=LDA_SHRINK,
        alpha=WHITEN_ALPHA,
        k=PCA_KEEP,
    )
    Ftr, Fte = T(Ftr), T(Fte)
    C = int(ytr.max()) + 1
    G = int(coarse_of[:C].max()) + 1
    groups = {
        g: np.where(coarse_of[:C] == g)[0] for g in range(G)
    }  # fine ids per coarse

    fcent = np.stack(
        [
            _l2n(Ftr[ytr == f].mean(0)[None], axis=1)[0]  # (C,D) fine prototypes
            for f in range(C)
        ]
    )
    Fn = _l2n(Fte, axis=1)
    yc = coarse_of[yte]  # true coarse of test

    fine_cos = Fn @ fcent.T  # (N,C) cos to fine centroids
    flat_acc = float((fine_cos.argmax(1) == yte).mean())  # flat 100-way prototype

    # coarse routers
    cones = {
        g: _rays_to_hull(fcent[groups[g]]) for g in range(G)
    }  # cone rays = fine centroids
    cone_sc = np.stack(
        [cones[g].score(Fte) for g in range(G)], 1
    )  # (N,G) coarse cone membership
    g_cone = cone_sc.argmax(1)
    ccent = np.stack(
        [_l2n(Ftr[coarse_of[ytr] == g].mean(0)[None], axis=1)[0] for g in range(G)]
    )  # coarse class means
    g_ncm = (Fn @ ccent.T).argmax(1)

    def fine_acc(groute):  # hard route → nearest fine centroid in group
        pred = np.empty(len(yte), dtype=int)
        for g in range(G):
            m = groute == g
            if m.any():
                fl = groups[g]
                pred[m] = fl[(Fn[m] @ fcent[fl].T).argmax(1)]
        return float((pred == yte).mean())

    def topk_acc(k):  # soft: fine-rank over top-k coarse cones
        k = min(k, G)
        topg = np.argsort(-cone_sc, axis=1)[:, :k]
        allowed = np.zeros((len(yte), C), dtype=bool)
        cf = coarse_of[:C][None, :]
        for j in range(k):
            allowed |= cf == topg[:, j][:, None]
        return float((np.where(allowed, fine_cos, -np.inf).argmax(1) == yte).mean())

    cone_per_fine = cone_sc[:, coarse_of[:C]]  # (N,C) each fine ← its coarse score

    def fusion_acc(a):  # score(f)=α·cone(g(f))+(1−α)·cos(q,cent_f)
        return float(((a * cone_per_fine + (1 - a) * fine_cos).argmax(1) == yte).mean())

    return dict(
        flat_ncm=flat_acc,
        hier_ncm=fine_acc(g_ncm),
        coarse_ncm=float((g_ncm == yc).mean()),
        hier_cone=fine_acc(g_cone),
        coarse_cone=float((g_cone == yc).mean()),
        oracle=fine_acc(yc),
        n_fine=C,
        n_coarse=G,
        topk={k: topk_acc(k) for k in HIER_TOPK},
        fusion={a: fusion_acc(a) for a in HIER_ALPHA},
    )


def _per_class_recall(pred, true, n):
    """Per-class recall (diagonal of the row-normalized confusion matrix)."""
    return np.array(
        [
            float((pred[true == c] == c).mean()) if (true == c).any() else float("nan")
            for c in range(n)
        ]
    )


def _partition_groups(n, k, seed):
    """Randomly partition the n items 0..n-1 into min(k, n) near-equal groups.
    Returns group_of: array indexed by item → group id in [0, min(k, n)).  Used to
    bundle fine classes into multimodal groups (each group = several fine classes; at
    k=n every group is a single fine class)."""
    k = min(k, n)
    group_of = np.empty(n, dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(n)
    for gi, chunk in enumerate(np.array_split(perm, k)):  # k near-equal item sets
        group_of[chunk] = gi
    return group_of


def multiproto_accuracy(Ftr, ytr, Fte, yte, labels, k):
    """Multi-prototype NCM (k-NCM): the CAPACITY-MATCHED point baseline.  Per class,
    k spherical k-means centroids; classify a query by its single nearest centroid
    (max cosine over every class's centroids).  Sits between single-mean NCM (k=1)
    and the cone: if the cone barely beats this, the cone's win is just multi-prototype
    capacity, not the conic (NNLS) geometry.  Returns (accuracy, predictions)."""
    from sklearn.cluster import KMeans

    cents, owner = [], []
    for c in labels:
        Xn = _l2n(np.asarray(Ftr[ytr == c], np.float64), axis=1)
        kk = min(k, len(Xn))
        C = (
            _l2n(
                KMeans(n_clusters=kk, n_init=1, max_iter=50, random_state=0)
                .fit(Xn)
                .cluster_centers_,
                axis=1,
            )
            if kk < len(Xn)
            else Xn
        )
        cents.append(C.astype(np.float32))
        owner.append(np.full(len(C), c, dtype=np.int64))
    cents = np.concatenate(cents)
    owner = np.concatenate(owner)
    Q = _l2n(np.asarray(Fte, np.float64), axis=1).astype(np.float32)
    pred = owner[(Q @ cents.T).argmax(axis=1)]
    return float((pred == yte).mean()), pred


def run_dataset_id(device):
    """Dataset-identification: one cone vs one mean per class, classify each query.

    Three regimes (ID_SPLIT_K=K, ID_CROSS_DATASET):
      K=1                : a class is a WHOLE dataset (maximally multimodal — one mean
                           summarizes an entire dataset; the cone gets a ray per mode).
      K>1, cross=False   : each dataset's fine classes split into K groups → D×K
                           classes, each a random subset of ONE dataset's fine classes;
                           predictions roll up to a "which dataset?" answer.
      K>1, cross=True    : ALL fine classes across every dataset are pooled and
                           randomly partitioned into K groups → K total classes, each
                           a random mix of fine classes drawn from different datasets
                           (maximally dissimilar multimodality, the hardest case for
                           one mean).  Max K = total fine classes across all datasets
                           (then every group is a single unique fine class).

    Features for every dataset come from the SAME frozen backbone (same dim), so the
    only thing being compared is the class primitive (mean vs cone)."""
    global DATASET, N_CLASSES
    names = list(ID_DATASETS)
    n_ds = len(names)
    K = max(1, ID_SPLIT_K)
    cross = ID_CROSS_DATASET and K > 1  # pool ALL fine classes, partition into K groups
    print(
        f"[dataset-id] {n_ds} datasets: {', '.join(names)}  "
        f"(transform={ID_TRANSFORM}, rays={RAY_METHOD}×{N_RAYS}, split K={K}, cross={cross})"
    )

    saved = DATASET
    rng = np.random.default_rng(SUBSET_SEED)

    def _cap(n, cap):  # balanced subsample indices (or all)
        idx = np.arange(n)
        return rng.choice(idx, cap, replace=False) if cap and cap < n else idx

    # ── load every dataset; collect features + per-sample dataset idx + GLOBAL fine
    #    id (fine classes re-indexed to be unique ACROSS datasets) ─────────────────
    Ftr_l, Fte_l, tr_ds_l, te_ds_l, tr_gf_l, te_gf_l = [], [], [], [], [], []
    fine_counts, offsets, offset = [], [], 0
    for di, name in enumerate(names):
        DATASET = name  # get_features keys its cache off DATASET
        Ftr, ytr_fine, Fte, yte_fine = get_features(device)
        itr, ite = _cap(len(Ftr), ID_TRAIN_PER), _cap(len(Fte), ID_TEST_PER)
        c_d = (
            int(max(int(ytr_fine.max()), int(yte_fine.max()))) + 1
        )  # fine classes here
        Ftr_l.append(Ftr[itr])
        Fte_l.append(Fte[ite])
        tr_ds_l.append(np.full(len(itr), di, dtype=np.int64))
        te_ds_l.append(np.full(len(ite), di, dtype=np.int64))
        tr_gf_l.append(offset + ytr_fine[itr])  # global (cross-dataset unique) fine id
        te_gf_l.append(offset + yte_fine[ite])
        fine_counts.append(c_d)
        offsets.append(offset)
        offset += c_d
        print(
            f"  [{di}] {name:<14} {c_d:>3} fine  train {len(itr):>5}  test {len(ite):>5}"
        )
    DATASET = saved

    Ftr = np.concatenate(Ftr_l)
    Fte = np.concatenate(Fte_l)
    tr_ds = np.concatenate(tr_ds_l)
    te_ds = np.concatenate(te_ds_l)
    tr_gf = np.concatenate(tr_gf_l)
    te_gf = np.concatenate(te_gf_l)
    n_fine_total = offset  # total fine classes across ALL datasets = the cross max K

    # ── build group labels per mode ──────────────────────────────────────────────
    ds_of_label = None  # per-dataset mode only: global class label → source dataset
    if K == 1:  # whole dataset = one class
        ytr, yte, n_cls = tr_ds.copy(), te_ds.copy(), n_ds
    elif cross:
        # pool ALL fine classes across datasets, partition into K groups.  Max useful
        # K = n_fine_total → every group is ONE unique fine class.  Each group mixes
        # fine classes from different datasets (cross-dataset multimodality).
        if K > n_fine_total:
            print(f"  [cross] K={K} capped to {n_fine_total} = total fine classes")
        group_of = _partition_groups(n_fine_total, K, ID_SPLIT_SEED)
        ytr, yte = group_of[tr_gf], group_of[te_gf]
        n_cls = int(group_of.max()) + 1
    else:
        # per-dataset: split each dataset's fine classes into K groups; label = di*K+g
        ytr = np.empty(len(tr_gf), dtype=np.int64)
        yte = np.empty(len(te_gf), dtype=np.int64)
        ds_lab = []
        for di in range(n_ds):
            c_d, off = fine_counts[di], offsets[di]
            local = _partition_groups(
                c_d, K, ID_SPLIT_SEED + di
            )  # fine→group in this ds
            m_tr, m_te = tr_ds == di, te_ds == di
            ytr[m_tr] = di * K + local[tr_gf[m_tr] - off]
            yte[m_te] = di * K + local[te_gf[m_te] - off]
            ds_lab.extend([di] * K)  # labels [di*K, di*K+K) belong to dataset di
        ds_of_label = np.array(ds_lab)
        n_cls = n_ds * K

    labels = np.unique(ytr)  # classes actually present (robust to empty groups)
    N_CLASSES = len(labels)

    if cross:
        shape = f"{n_cls} cross-dataset groups from {n_fine_total} pooled fine classes"
    elif K == 1:
        shape = f"{n_ds} datasets as classes"
    else:
        shape = f"{n_ds} datasets × {K} groups = {len(labels)} non-empty classes"
    print(f"[dataset-id] {shape}  chance={1/n_cls:.4f}")

    check_feature_health(Ftr, Fte, ytr, yte, "dataset-id")

    # separability transform fit on the group labels (pca/none keep modes intact)
    T = fit_feature_transform(
        Ftr,
        ytr,
        ID_TRANSFORM,
        ridge=WHITEN_RIDGE,
        lda_shrink=LDA_SHRINK,
        alpha=WHITEN_ALPHA,
        k=PCA_KEEP,
    )
    Ftr_t, Fte_t = T(Ftr), T(Fte)

    # ── 1 mean per group-class (NCM) ────────────────────────────────────────────
    means = np.stack([Ftr_t[ytr == c].mean(axis=0) for c in labels])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
    Q = Fte_t / (np.linalg.norm(Fte_t, axis=1, keepdims=True) + 1e-8)
    ncm_pred = labels[(Q @ means.T).argmax(axis=1)]
    ncm_acc = float((ncm_pred == yte).mean())

    # ── k prototypes per group-class (multi-prototype NCM) — capacity-matched point
    #    baseline; isolates the cone's conic geometry from mere multi-prototype ────
    mp_acc, mp_pred = multiproto_accuracy(Ftr_t, ytr, Fte_t, yte, labels, N_RAYS)

    # ── 1 cone per group-class ──────────────────────────────────────────────────
    feature_dict = {str(c): Ftr_t[ytr == c] for c in labels}
    hulls = build_hulls(feature_dict)
    per = [
        hulls[str(c)].score_all(Fte_t)
        for c in tqdm(labels, desc="score test", unit="cls")
    ]
    cone_pred, cone_acc = {}, {}
    for s in SCORE_NAMES:
        pred = labels[np.stack([p[s] for p in per], axis=1).argmax(axis=1)]
        cone_pred[s], cone_acc[s] = pred, float((pred == yte).mean())

    best = max(cone_acc, key=cone_acc.get)
    d_acc = cone_acc[best] - ncm_acc

    # ── report ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if K == 1:
        title = "DATASET IDENTIFICATION — which dataset is this query from?"
    elif cross:
        title = f"CROSS-DATASET GROUPS — {K} groups, each pooled across {n_ds} datasets"
    else:
        title = f"GROUPED DATASET-ID — {n_ds} datasets × {K} random groups"
    print(title)
    print(f"  ({n_cls} classes, transform={ID_TRANSFORM}, chance={1/n_cls:.4f})")
    print("=" * 60)
    lab = "1 mean / dataset" if K == 1 else "1 mean / group"
    print(f"  {'NCM (' + lab + ', k=1)':<28} {ncm_acc:.4f}   <- single-prototype point")
    print(
        f"  {'multiproto NCM (k=' + str(N_RAYS) + ')':<28} {mp_acc:.4f}"
        "   <- capacity-matched point"
    )
    print("  " + "-" * 44)
    for s in SCORE_NAMES:
        star = " *" if s == best else ""
        print(f"  {'cone:' + s:<28} {cone_acc[s]:.4f}{star}")
    verdict = (
        "cone classifies better"
        if d_acc > 0.003
        else "NCM classifies better" if d_acc < -0.003 else "tied"
    )
    d_mp = cone_acc[best] - mp_acc
    geom = (
        "conic geometry adds value beyond multi-prototype"
        if d_mp > 0.003
        else (
            "multi-prototype point ≥ cone — conic geometry not the driver"
            if d_mp < -0.003
            else "tied — the gain is multi-prototype capacity, NOT conic geometry"
        )
    )
    level = "group-level: " if K > 1 else ""
    print(f"\n  {level}best cone = {best} {cone_acc[best]:.4f}")
    print(f"    cone − NCM (k=1)         {d_acc:+.4f}  ({verdict})")
    print(f"    cone − multiproto (k={N_RAYS})  {d_mp:+.4f}  ({geom})")

    if cross:
        # each group spans datasets → no which-dataset rollup; show per-group recall
        # plus how many test samples each dataset contributes to every group.
        ncm_rec = _per_class_recall(ncm_pred, yte, len(labels))
        cone_rec = _per_class_recall(cone_pred[best], yte, len(labels))
        hdr = "".join(f"{nm[:8]:>9}" for nm in names)
        print("\n  per-group recall (NCM vs best cone) + test composition per dataset:")
        print(f"    {'group':<6} {'NCM':>7} {'cone':>7} {'Δ':>7}   {hdr}")
        for gi, g in enumerate(labels):
            comp = "".join(
                f"{int(((yte == g) & (te_ds == di)).sum()):>9}" for di in range(n_ds)
            )
            print(
                f"    {'g' + str(int(g)):<6} {ncm_rec[gi]:>7.3f} {cone_rec[gi]:>7.3f} "
                f"{cone_rec[gi] - ncm_rec[gi]:>+7.3f}   {comp}"
            )
        return

    # ── per-dataset mode: roll group predictions up to a "which dataset?" answer ──
    yte_ds = ds_of_label[yte]
    if K > 1:
        ncm_ds_acc = float((ds_of_label[ncm_pred] == yte_ds).mean())
        mp_ds_acc = float((ds_of_label[mp_pred] == yte_ds).mean())
        cone_ds_acc = float((ds_of_label[cone_pred[best]] == yte_ds).mean())
        dd = cone_ds_acc - ncm_ds_acc
        dverdict = (
            "cone IDs dataset better"
            if dd > 0.003
            else "NCM IDs dataset better" if dd < -0.003 else "tied"
        )
        print(
            f"  dataset-level (group→dataset rollup, chance {1/n_ds:.3f}): "
            f"cone {cone_ds_acc:.4f}  |  multiproto {mp_ds_acc:.4f}  |  "
            f"NCM {ncm_ds_acc:.4f}  Δ(cone−NCM) {dd:+.4f}  ({dverdict})"
        )

    # per-dataset recall (aggregated to dataset level): where does each get confused?
    ncm_rec = _per_class_recall(ds_of_label[ncm_pred], yte_ds, n_ds)
    cone_rec = _per_class_recall(ds_of_label[cone_pred[best]], yte_ds, n_ds)
    print("\n  per-dataset recall (NCM vs best cone):")
    print(f"    {'dataset':<14} {'NCM':>7} {'cone':>7} {'Δ':>7}")
    for di, name in enumerate(names):
        print(
            f"    {name:<14} {ncm_rec[di]:>7.3f} {cone_rec[di]:>7.3f} "
            f"{cone_rec[di] - ncm_rec[di]:>+7.3f}"
        )


def main():
    global N_CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] no CUDA — ViT-B on CPU will be very slow.")

    # ── dataset-identification test (each dataset = one class) ──────────────────
    if DATASET_ID:
        run_dataset_id(device)
        return

    print(
        f"[setup] {DATASET} + frozen {MODEL_NAME} on {device}  "
        f"(rays={RAY_METHOD}×{N_RAYS}, transform={TRANSFORM}, rp={USE_RP})"
    )

    # ── 1-3. features (model's own normalization), cached to disk ───────────────
    Ftr, ytr_np, Fte, yte_np = get_features(device)
    d_feat = Ftr.shape[1]

    # normalization / extraction sanity check for the (possibly new) dataset
    check_feature_health(Ftr, Fte, ytr_np, yte_np, DATASET)

    # ── hierarchical coarse(cone)→fine(centroid) eval (CIFAR-100; uses fine labels) ─
    if RUN_HIERARCHICAL:
        assert DATASET == "CIFAR100", "hierarchical eval uses CIFAR-100 superclasses"
        r = hierarchical_eval(Ftr, ytr_np, Fte, yte_np, _CIFAR100_COARSE, TRANSFORM)
        print("\n" + "=" * 60)
        print(
            f"HIERARCHICAL coarse(cone)→fine(centroid) — CIFAR-100 "
            f"({r['n_fine']} fine / {r['n_coarse']} coarse, transform={TRANSFORM})"
        )
        print("=" * 60)
        print(f"  {'flat 100-way NCM (prototype)':<34} {r['flat_ncm']:.4f}")
        print(
            f"  {'hier: NCM-coarse  → centroid':<34} {r['hier_ncm']:.4f}"
            f"   (coarse route {r['coarse_ncm']:.4f})"
        )
        print(
            f"  {'hier: CONE-coarse → centroid':<34} {r['hier_cone']:.4f}"
            f"   (coarse route {r['coarse_cone']:.4f})"
        )
        print(f"  {'hier: ORACLE-coarse → centroid':<34} {r['oracle']:.4f}   (ceiling)")
        print("  -- soft top-k routing (fine-rank over union of top-k coarse) --")
        for k in sorted(r["topk"]):
            print(f"    top-{k:<2} {r['topk'][k]:.4f}")
        print("  -- score fusion  α·cone(g(f)) + (1−α)·cos(q,cent_f)  (α=0 ⇒ flat) --")
        for a in sorted(r["fusion"]):
            print(f"    α={a:<4} {r['fusion'][a]:.4f}")
        best_soft = max([r["hier_cone"], *r["topk"].values(), *r["fusion"].values()])
        print(
            f"\n  cone vs NCM coarse routing: {r['coarse_cone'] - r['coarse_ncm']:+.4f}"
            f"  ({'cone routes better' if r['coarse_cone'] > r['coarse_ncm'] else 'NCM routes better'})"
        )
        print(
            f"  best soft method vs flat NCM: {best_soft - r['flat_ncm']:+.4f}"
            f"  ({'hierarchy helps' if best_soft > r['flat_ncm'] else 'flat still better'})"
            f"   [oracle ceiling {r['oracle'] - r['flat_ncm']:+.4f}]"
        )
        return

    # multimodal labels: CIFAR-100 semantic superclasses, else random merge-K
    if SEMANTIC_COARSE and DATASET == "CIFAR100":
        ytr_np, yte_np, n_coarse = cifar100_coarse_labels(ytr_np, yte_np)
        print(f"[coarse] CIFAR-100 semantic superclasses: 100 → {n_coarse}")
    elif MERGE_K > 1:
        n_fine = int(max(int(ytr_np.max()), int(yte_np.max()))) + 1
        ytr_np, yte_np, n_coarse = merge_labels(ytr_np, yte_np, MERGE_K, MERGE_SEED)
        print(
            f"[merge] random merge-{MERGE_K}: {n_fine} → {n_coarse} multimodal classes"
        )

    if CLASS_LIMIT and CLASS_LIMIT > 0:  # few-classes vs multimodal control
        trm, tem = ytr_np < CLASS_LIMIT, yte_np < CLASS_LIMIT
        Ftr, ytr_np, Fte, yte_np = Ftr[trm], ytr_np[trm], Fte[tem], yte_np[tem]
        print(
            f"[limit] kept first {CLASS_LIMIT} classes "
            f"(train {Ftr.shape[0]}, test {Fte.shape[0]})"
        )

    if SAMPLES_PER_CLASS and SAMPLES_PER_CLASS > 0:  # match data volume across arms
        rng = np.random.default_rng(SUBSET_SEED)
        keep = []
        for c in np.unique(ytr_np):
            idx = np.where(ytr_np == c)[0]
            keep.append(
                rng.choice(idx, SAMPLES_PER_CLASS, replace=False)
                if len(idx) > SAMPLES_PER_CLASS
                else idx
            )
        keep = np.concatenate(keep)
        Ftr, ytr_np = Ftr[keep], ytr_np[keep]
        print(f"[subsample] ≤{SAMPLES_PER_CLASS} train/class → {len(ytr_np)} total")

    N_CLASSES = int(max(int(ytr_np.max()), int(yte_np.max()))) + 1  # from the data
    print(
        f"[data] {DATASET}: {N_CLASSES} classes, "
        f"train {Ftr.shape[0]}, test {Fte.shape[0]}, dim {d_feat}"
    )

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
        id_cls, ood_cls = list(range(n_id_classes)), list(
            range(n_id_classes, N_CLASSES)
        )
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
        points = [p for p in ("ncm", "multiproto") if p in res]
        ordered = points + [k for k in res if k.startswith("cone:")]
        for k in ordered:
            au, fpr = res[k]
            tag = (
                "  <- single-prototype point"
                if k == "ncm"
                else ("  <- capacity-matched point" if k == "multiproto" else "")
            )
            print(f"  {k:<22} {au:>7.4f} {fpr:>8.4f}{tag}")
        best_c = max((k for k in res if k.startswith("cone:")), key=lambda k: res[k][0])
        d_ncm = res[best_c][0] - res["ncm"][0]
        d_mp = res[best_c][0] - res["multiproto"][0]
        geom = (
            "conic boundary adds value beyond multi-prototype"
            if d_mp > 0.003
            else (
                "multi-prototype point ≥ cone on OOD — boundary not the driver"
                if d_mp < -0.003
                else "tied — OOD edge is multi-prototype capacity, NOT the conic boundary"
            )
        )
        print(f"\n  best cone AUROC: {res[best_c][0]:.4f} ({best_c})")
        print(f"    cone − NCM (k=1)         {d_ncm:+.4f}")
        print(f"    cone − multiproto (k={N_RAYS})  {d_mp:+.4f}  ({geom})")
        # geo_residual is the cone's DISTINCTIVE score (distance-to-region boundary,
        # not nearest-ray); its gap over multiproto isolates the conic boundary itself
        if "cone:geo_residual" in res:
            d_geo = res["cone:geo_residual"][0] - res["multiproto"][0]
            print(
                f"    geo_residual − multiproto {d_geo:+.4f}  "
                f"(the conic region boundary vs k prototypes)"
            )


if __name__ == "__main__":
    main()
