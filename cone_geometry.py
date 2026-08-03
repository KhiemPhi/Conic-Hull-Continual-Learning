"""
cone_geometry.py
----------------
Cone-separability diagnostic across frozen ViT backbones and datasets.

For every (backbone, dataset) cell we:
  1. Extract the [CLS] / pooled embedding for the test/val split (cached to .npz).
  2. Project to the unit sphere (angles only).
  3. Per class: angular diameter (max pairwise angle, 512-subsample) and the mean
     angle to the L2-normalised centroid ("cone half-angle").
  4. Inter-class centroid angle matrix (median, min).
  5. Apply the decision rule (green / overlap-risk / red).

Three backbones with SHARPLY different embedding geometry — the comparison is
itself the finding:
  * timm   vit_base_patch16_224  (supervised ImageNet-1k, D=768)
  * dinov2 dinov2_vitb14         (self-supervised, D=768, torch.hub)
  * clip   ViT-B-16 laion2b      (contrastive image-text, D=512, open_clip)

Each backbone uses ITS OWN preprocessing — a frozen backbone is very sensitive to
normalization (see demo_joint_floor.get_features). Features are cached so the
geometry can be recomputed for free.

Usage
-----
    python -u cone_geometry.py                       # all backbones × all datasets
    python -u cone_geometry.py --backbones clip      # one backbone
    python -u cone_geometry.py --datasets CIFAR100 CUB200
    python -u cone_geometry.py --max-per-class 300   # cap extraction (speed)
"""
import argparse
import io
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Devserver external downloads (torch.hub / open_clip / HF) route through fwdproxy.
for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
# The HF Xet CAS backend fails through fwdproxy; force the classic HTTP path.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DATA_DIR = "./data"
OUT_DIR = "./cone_geom_out"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> (hf_repo, train_split, test_split); test falls back validation→train.
HF_REPOS = {
    "CUB200": ("Donghyun99/CUB-200-2011", "train", "test"),
    "StanfordCars": ("Donghyun99/Stanford-Cars", "train", "test"),
    "ImageNet-A": ("barkermrl/imagenet-a", "train", "test"),
    "ImageNet-R": ("axiong/imagenet-r", "train", "test"),
}
TORCHVISION = ["CIFAR10", "CIFAR100", "STL10", "FGVCAircraft", "Flowers102",
               "OxfordIIITPet", "Food101"]
ALL_DATASETS = TORCHVISION + list(HF_REPOS)
ALL_BACKBONES = ["timm", "dinov2", "clip"]


# ─────────────────────────────────────────────────────────────────────────────
# Backbones — each returns (forward_fn, transform, dim, label)
# ─────────────────────────────────────────────────────────────────────────────
def load_backbone(name):
    from torchvision import transforms as T
    try:
        bicubic = T.InterpolationMode.BICUBIC
    except AttributeError:
        bicubic = 3

    if name == "timm":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True,
                                  num_classes=0).to(DEVICE).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        tf = timm.data.create_transform(**cfg, is_training=False)
        fwd = lambda x: model(x)
        return fwd, tf, 768, "vit_base_patch16_224 (timm/IN1k)"

    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2",
                               "dinov2_vitb14").to(DEVICE).eval()
        tf = T.Compose([
            T.Resize(256, interpolation=bicubic),
            T.CenterCrop(224),                       # 224 = 16 patches of 14px
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        fwd = lambda x: model(x)                      # returns CLS token (D=768)
        return fwd, tf, 768, "dinov2_vitb14 (SSL)"

    if name == "clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k")
        model = model.to(DEVICE).eval()
        fwd = lambda x: model.encode_image(x)        # image tower (D=512)
        return fwd, preprocess, 512, "CLIP ViT-B-16 (laion2b)"

    raise ValueError(f"unknown backbone '{name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Datasets  (reuses demo_joint_floor conventions)
# ─────────────────────────────────────────────────────────────────────────────
class _HFImageDataset(Dataset):
    """Wrap a HuggingFace image split as (transform(image), label)."""

    def __init__(self, hf_ds, transform, img_key, lbl_key):
        self.ds, self.tf, self.ik, self.lk = hf_ds, transform, img_key, lbl_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r[self.ik]
        if not hasattr(img, "mode"):                 # bytes dict → PIL
            from PIL import Image
            img = Image.open(io.BytesIO(img["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(r[self.lk])


def build_test_dataset(name, transform):
    """Return the test/val split as a torch Dataset yielding (img, label)."""
    if name in HF_REPOS:
        from datasets import load_dataset
        repo, tr_split, te_split = HF_REPOS[name]
        dd = load_dataset(repo, cache_dir=os.path.join(DATA_DIR, "hf"))
        if te_split not in dd:
            te_split = "validation" if "validation" in dd else tr_split
        te = dd[te_split]
        cols = te.column_names
        ik = next((c for c in ("image", "img", "Image", "picture") if c in cols), None)
        lk = next((c for c in ("label", "labels", "fine_label", "class", "target")
                   if c in cols), None)
        if ik is None or lk is None:
            raise ValueError(f"{repo}: can't find image/label cols in {cols}")
        print(f"[hf] {repo}: split '{te_split}', cols image='{ik}' label='{lk}'")
        return _HFImageDataset(te, transform, ik, lk)

    from torchvision import datasets as D
    if name == "CIFAR10":
        return D.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    if name == "CIFAR100":
        return D.CIFAR100(DATA_DIR, train=False, download=True, transform=transform)
    if name == "STL10":
        return D.STL10(DATA_DIR, split="test", download=True, transform=transform)
    if name == "FGVCAircraft":
        return D.FGVCAircraft(DATA_DIR, split="test", download=True, transform=transform)
    if name == "Flowers102":
        return D.Flowers102(DATA_DIR, split="test", download=True, transform=transform)
    if name == "OxfordIIITPet":
        return D.OxfordIIITPet(DATA_DIR, split="test", download=True, transform=transform)
    if name == "Food101":
        return D.Food101(DATA_DIR, split="test", download=True, transform=transform)
    raise ValueError(f"unknown dataset '{name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction (cached)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(bkb_name, ds_name, fwd, transform, batch_size=128,
                     num_workers=8):
    """Extract (feats [N,D] float16, labels [N]) for a cell, caching to .npz."""
    path = os.path.join(CACHE_DIR, f"feats_{ds_name}_{bkb_name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        print(f"[cache] {path}  feats{d['feats'].shape}")
        return d["feats"].astype(np.float32), d["labels"]

    ds = build_test_dataset(ds_name, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    use_amp = DEVICE.startswith("cuda")
    for x, y in tqdm(loader, desc=f"{bkb_name}/{ds_name}", unit="batch"):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            f = fwd(x)
        feats.append(f.float().cpu().numpy().astype(np.float16))
        labels.append(np.asarray(y))
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    np.savez_compressed(path, feats=feats, labels=labels)
    print(f"[save] {path}  feats{feats.shape}  ({len(np.unique(labels))} classes)")
    return feats.astype(np.float32), labels


# ─────────────────────────────────────────────────────────────────────────────
# Geometry  (protocol steps 3-4)
# ─────────────────────────────────────────────────────────────────────────────
def angle_deg(a, b):
    cos = (a @ b.T).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos))


def compute_geometry(feats, labels, max_per_class=None, diam_subsample=512,
                     seed=0):
    """Return per-class diameters/half-angles + inter-centroid stats."""
    g = torch.Generator().manual_seed(seed)
    F = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
    F = torch.nn.functional.normalize(F, dim=1)
    y = torch.tensor(labels)

    diam, half, n_per = [], [], []
    centroids = []
    classes = torch.unique(y).tolist()
    for c in classes:
        X = F[y == c]
        n_per.append(int(len(X)))
        if max_per_class and len(X) > max_per_class:
            idx = torch.randperm(len(X), generator=g)[:max_per_class]
            X = X[idx]
        # intra diameter = max pairwise angle (subsample huge classes)
        if len(X) > diam_subsample:
            idx = torch.randperm(len(X), generator=g)[:diam_subsample]
            Xs = X[idx]
        else:
            Xs = X
        D = angle_deg(Xs, Xs)
        diam.append(D.max().item())
        # spread around the normalised centroid = cone half-angle
        mu = torch.nn.functional.normalize(X.mean(0, keepdim=True), dim=1)
        half.append(angle_deg(X, mu).mean().item())
        centroids.append(mu)

    centroids = torch.cat(centroids)
    inter = angle_deg(centroids, centroids)
    inter.fill_diagonal_(float("nan"))

    diam = np.array(diam)
    half = np.array(half)
    inter_np = inter.cpu().numpy()
    # nearest-neighbour centroid angle per class (min over row)
    nn_inter = np.nanmin(inter_np, axis=1)

    return {
        "n_classes": len(classes),
        "n_samples": int(len(labels)),
        "min_n_per_class": int(min(n_per)),
        "median_n_per_class": int(np.median(n_per)),
        "diam_median": float(np.median(diam)),
        "diam_p95": float(np.percentile(diam, 95)),
        "diam_max": float(diam.max()),
        "half_median": float(np.median(half)),
        "half_p95": float(np.percentile(half, 95)),
        "inter_median": float(np.nanmedian(inter_np)),
        "inter_min": float(np.nanmin(inter_np)),
        "inter_nn_median": float(np.median(nn_inter)),  # median nearest-centroid gap
        "acute_frac": float((diam < 90).mean()),        # diameter < 90°
        "tight_frac": float((diam < 75).mean()),        # diameter < 75° (green thresh)
    }


def verdict(r):
    """Decision rule from the protocol."""
    tight = r["tight_frac"] >= 0.90
    inter_gt_half = r["inter_median"] > r["half_median"]
    nn_gt_half = r["inter_nn_median"] > r["half_median"]
    if tight and inter_gt_half and nn_gt_half:
        return "GREEN", "cones separable — build the conic head"
    if r["acute_frac"] >= 0.50 and r["inter_median"] <= r["half_median"] * 1.5:
        return "OVERLAP", "cones acute but touch — try whitening/γ-shift, re-measure"
    if (1.0 - r["acute_frac"]) >= 0.10:  # ≥10% classes with diameter ≥ 90°
        return "RED", "conic hull degenerates — γ-shift mandatory, clean OOD weakens"
    return "OVERLAP", "acute but inter-class margin thin — anisotropy risk"


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=ALL_BACKBONES,
                    choices=ALL_BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    choices=ALL_DATASETS)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap samples/class in geometry (extraction stays full)")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[device] {DEVICE}  backbones={args.backbones}  "
          f"datasets={args.datasets}")

    results = {}
    labels_map = {}
    for bkb in args.backbones:
        print(f"\n{'='*70}\n=== backbone: {bkb} ===\n{'='*70}")
        fwd, transform, dim, blabel = None, None, None, None
        for ds in args.datasets:
            key = f"{bkb}/{ds}"
            try:
                # Lazy-load the backbone only when a cell needs extraction.
                cache = os.path.join(CACHE_DIR, f"feats_{ds}_{bkb}.npz")
                if fwd is None and not os.path.exists(cache):
                    fwd, transform, dim, blabel = load_backbone(bkb)
                elif blabel is None:
                    # Cache hit but we still want the human label for the report.
                    _labels = {"timm": "vit_base_patch16_224 (timm/IN1k)",
                               "dinov2": "dinov2_vitb14 (SSL)",
                               "clip": "CLIP ViT-B-16 (laion2b)"}
                    blabel = _labels[bkb]
                if os.path.exists(cache):
                    feats, labels = extract_features(bkb, ds, None, None,
                                                     batch_size=args.batch_size)
                else:
                    feats, labels = extract_features(bkb, ds, fwd, transform,
                                                     batch_size=args.batch_size)
                t0 = time.time()
                geo = compute_geometry(feats, labels,
                                       max_per_class=args.max_per_class)
                geo["backbone_dim"] = feats.shape[1]
                v, why = verdict(geo)
                geo["verdict"], geo["verdict_why"] = v, why
                results[key] = geo
                labels_map[bkb] = blabel
                print(f"[{key}]  C={geo['n_classes']:>3} N={geo['n_samples']:>6} | "
                      f"diam med {geo['diam_median']:.1f}° p95 {geo['diam_p95']:.1f}° | "
                      f"half med {geo['half_median']:.1f}° | "
                      f"inter med {geo['inter_median']:.1f}° nn {geo['inter_nn_median']:.1f}° "
                      f"min {geo['inter_min']:.1f}° | "
                      f"tight<75 {geo['tight_frac']*100:.0f}% acute<90 {geo['acute_frac']*100:.0f}% "
                      f"→ {v}  ({time.time()-t0:.1f}s geom)")
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"[FAIL] {key}: {e}")
                traceback.print_exc()
                results[key] = {"error": str(e)}

    _write_report(results, labels_map, args)


def _write_report(results, labels_map, args):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# Cone-separability diagnostic\n",
             "Intra **diameter** = max pairwise angle (512-subsample). "
             "**half** = mean angle to normalised centroid (cone half-angle). "
             "**inter** = centroid-centroid angle; **nn** = nearest-centroid "
             "angle (median). Verdict per protocol decision rule.\n"]
    lines.append("| backbone | dataset | C | N | diam med / p95 | half med | "
                 "inter med / nn / min | tight<75° | acute<90° | verdict |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|---|")
    for bkb in args.backbones:
        for ds in args.datasets:
            r = results.get(f"{bkb}/{ds}")
            if not r:
                continue
            if "error" in r:
                lines.append(f"| {labels_map.get(bkb, bkb)} | {ds} | | | | | | | | "
                             f"ERROR: {r['error'][:40]} |")
                continue
            lines.append(
                f"| {labels_map.get(bkb, bkb)} | {ds} | {r['n_classes']} | "
                f"{r['n_samples']} | {r['diam_median']:.1f}/{r['diam_p95']:.1f} | "
                f"{r['half_median']:.1f} | {r['inter_median']:.1f}/"
                f"{r['inter_nn_median']:.1f}/{r['inter_min']:.1f} | "
                f"{r['tight_frac']*100:.0f}% | {r['acute_frac']*100:.0f}% | "
                f"**{r['verdict']}** |")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print(f"\n{report}")
    print(f"[done] wrote {OUT_DIR}/report.md and results.json")


if __name__ == "__main__":
    main()
