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
from conic_hull import build_class_conic_hulls
from tqdm import tqdm

# ── knobs ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "vit_base_patch16_224.orig_in21k"
N_CLASSES = 100
N_RAYS = 200  # extreme rays per class (match cone_boundary; raise for a higher ceiling)
USE_PCA = True  # PCA before SPA (speed); rays stored in original space
PCA_DIM = 64
RAY_DIV = "spa"  # "spa" | "fps" | "hybrid"
BATCH_SIZE = 64
DATA_DIR = "./data"
# Optional frozen Random-Projection + ReLU on the features (post-hoc; no re-forward)
USE_RP = False
N_PROJ = 10000
RP_SEED = 0

SCORE_NAMES = ["cosine", "angular_margin", "blended", "max_ray_sim"]


def cache_path():
    tag = MODEL_NAME.replace("/", "_").replace(".", "_")
    return f"feats_{tag}_modelnorm.npz"


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
    print(f"[norm] model transform: mean={tuple(round(m, 3) for m in cfg['mean'])} "
          f"std={tuple(round(s, 3) for s in cfg['std'])} input={cfg['input_size']}")

    tr = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=tf)
    te = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tf)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] no CUDA — ViT-B on CPU will be very slow.")
    print(
        f"[setup] CIFAR-100 + frozen {MODEL_NAME} on {device}  "
        f"(n_rays={N_RAYS}, rp={USE_RP})"
    )

    # ── 1-3. features (model's own normalization), cached to disk ───────────────
    Ftr, ytr_np, Fte, yte_np = get_features(device)
    d_feat = Ftr.shape[1]

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

    # ── NCM baseline (reference for what the frozen features support) ────────────
    ncm = ncm_accuracy(Ftr, ytr_np, Fte, yte_np)

    # ── 4. build one cone per class (all 100 at once) ───────────────────────────
    feature_dict = {str(c): Ftr[ytr_np == c] for c in range(N_CLASSES)}
    hulls = build_class_conic_hulls(
        feature_dict,
        n_rays=N_RAYS,
        use_pca=USE_PCA,
        pca_dim=PCA_DIM,
        ray_diversity=RAY_DIV,
        spa_oversample=3,
    )

    # ── HOOK: insert cone separation here (feature tuning / optimization) ────────
    #   e.g. whiten Ftr/Fte with a global Gram inverse, or rebuild hulls with a
    #   discriminative/margin objective, then re-run classify_all_scores.
    # ────────────────────────────────────────────────────────────────────────────

    # ── 5. classify the test set ────────────────────────────────────────────────
    accs = classify_all_scores(Fte, yte_np, hulls)

    print("\n" + "=" * 56)
    print(f"Joint FLOOR — frozen {MODEL_NAME}")
    print(f"  100 classes, {N_RAYS} rays/class, rp={USE_RP}")
    print("=" * 56)
    print(f"  {'NCM (cosine-to-mean)':<22} {ncm:.4f}   <- reference")
    print("  " + "-" * 40)
    for s in SCORE_NAMES:
        star = " *" if s == "cosine" else ""
        print(f"  {'cone:' + s:<22} {accs[s]:.4f}{star}")
    best = max(accs, key=accs.get)
    gap = ncm - accs[best]
    print(f"\n  best cone scheme: {best} = {accs[best]:.4f}   |   NCM = {ncm:.4f}")
    print(f"  cone vs NCM gap: {gap:+.4f}  "
          f"({'cone scoring is the bottleneck' if gap > 0.03 else 'cones competitive'})")


if __name__ == "__main__":
    main()
