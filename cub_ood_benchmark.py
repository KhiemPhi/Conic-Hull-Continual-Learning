"""
cub_ood_benchmark.py
--------------------
Fine-tune a ViT on CUB-200, then benchmark cone OOD detection against the
standard post-hoc OOD suite on the *fine-tuned* embeddings.

This is the regime the earlier tests couldn't probe: fine-tuning on CUB shapes
the feature space around the ID classes, giving the cone its best remaining shot
(frozen CLIP already tied NCM, and the synthetic control tied even with oracle
generators).

Pipeline
--------
1. timm vit_base_patch16_224 (IN-1k) + LoRA(rank 16, attn.qkv+proj) + 200-way head,
   fine-tuned on CUB train.  A softmax head is required for MSP/Energy/MaxLogit.
2. Extract penultimate features (768-d) + logits for CUB train (fit ID scorers),
   CUB test (ID eval), and each OOD dataset's test split.  Cached to .npz.
3. Score ID vs each OOD with:
     MSP, Energy, MaxLogit          (logit-based benchmarks)
     NCM (cosine), Mahalanobis, KNN (feature-based benchmarks)
     Cone  min_c r_c(x)             (method under test)
   Report AUROC + FPR@95 per OOD set and mean.

Usage
-----
    python -u cub_ood_benchmark.py                 # full pipeline
    python -u cub_ood_benchmark.py --epochs 25
    python -u cub_ood_benchmark.py --skip-train    # reuse cached features
"""
import argparse
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import timm
from backbone import inject_lora, get_lora_params, freeze_non_lora
from conic_hull import ConicHull

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DATA_DIR = "./data"
OUT_DIR = "./cub_ood_out"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set at runtime from --model / --ft-mode (see main()).
MODEL_NAME = "vit_base_patch16_224"
FT_MODE = "lora"          # "lora" (ViT attn) | "full" (e.g. resnet18)
TAG = "vitb16"            # cache/report suffix, derived from model
CKPT = os.path.join(OUT_DIR, "cub_vitb16.pt")

ID_DATASET = "CUB200"
OOD_DATASETS = ["StanfordCars", "FGVCAircraft", "Flowers102", "OxfordIIITPet",
                "Food101", "CIFAR100", "ImageNet-A"]
HF_REPOS = {
    "CUB200": ("Donghyun99/CUB-200-2011", "train", "test"),
    "StanfordCars": ("Donghyun99/Stanford-Cars", "train", "test"),
    "ImageNet-A": ("barkermrl/imagenet-a", "train", "test"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
class _HFImageDataset(Dataset):
    def __init__(self, hf_ds, transform, ik, lk):
        self.ds, self.tf, self.ik, self.lk = hf_ds, transform, ik, lk

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        img = r[self.ik]
        if not hasattr(img, "mode"):
            from PIL import Image
            img = Image.open(io.BytesIO(img["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(r[self.lk])


def build_dataset(name, split, transform):
    """split ∈ {'train','test'}."""
    if name in HF_REPOS:
        from datasets import load_dataset
        repo, tr, te = HF_REPOS[name]
        dd = load_dataset(repo, cache_dir=os.path.join(DATA_DIR, "hf"))
        want = tr if split == "train" else te
        if want not in dd:
            want = "validation" if "validation" in dd else tr
        ds = dd[want]
        cols = ds.column_names
        ik = next((c for c in ("image", "img", "Image", "picture") if c in cols), None)
        lk = next((c for c in ("label", "labels", "fine_label", "class", "target")
                   if c in cols), None)
        return _HFImageDataset(ds, transform, ik, lk)

    from torchvision import datasets as D
    if name == "CIFAR100":
        return D.CIFAR100(DATA_DIR, train=(split == "train"), download=True, transform=transform)
    tv_split = {"train": {"FGVCAircraft": "trainval", "Flowers102": "train",
                          "OxfordIIITPet": "trainval", "Food101": "train"},
                "test": {"FGVCAircraft": "test", "Flowers102": "test",
                         "OxfordIIITPet": "test", "Food101": "test"}}[split][name]
    return getattr(D, name)(DATA_DIR, split=tv_split, download=True, transform=transform)


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune (LoRA + head)
# ─────────────────────────────────────────────────────────────────────────────
class ClassifierModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, return_feat=False):
        f = self.backbone(x)
        z = self.head(f)
        return (z, f) if return_feat else z


def build_model(num_classes, lora_rank=16, lora_alpha=16, ft_mode="lora"):
    bb = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    feat_dim = bb(torch.zeros(1, 3, 224, 224)).shape[-1]
    if ft_mode == "lora":
        inject_lora(bb, rank=lora_rank, alpha=lora_alpha,
                    target_modules=["attn.qkv", "attn.proj"])
        freeze_non_lora(bb)
    # "full": all backbone params stay trainable (e.g. resnet18 has no attn layers)
    head = nn.Linear(feat_dim, num_classes)
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)
    return ClassifierModel(bb, head).to(DEVICE), feat_dim


def finetune(args):
    train_tf = timm.data.create_transform(
        **timm.data.resolve_data_config({}, model=timm.create_model(MODEL_NAME)),
        is_training=True)
    eval_tf = timm.data.create_transform(
        **timm.data.resolve_data_config({}, model=timm.create_model(MODEL_NAME)),
        is_training=False)

    tr_ds = build_dataset(ID_DATASET, "train", train_tf)
    te_ds = build_dataset(ID_DATASET, "test", eval_tf)
    num_classes = int(max(max(l for _, l in _labels(tr_ds)),
                          max(l for _, l in _labels(te_ds))) + 1)
    print(f"[cub] train={len(tr_ds)} test={len(te_ds)} classes={num_classes}")

    model, _ = build_model(num_classes, args.lora_rank, args.lora_alpha, ft_mode=FT_MODE)
    if FT_MODE == "lora":
        params = get_lora_params(model.backbone) + list(model.head.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params if p.requires_grad)
    print(f"[ft] trainable params: {n_train:,} (mode={FT_MODE})")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=8, pin_memory=True, drop_last=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=8, pin_memory=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for imgs, y in tqdm(tr_loader, desc=f"ep{ep:02d}", leave=False):
            imgs, y = imgs.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(imgs), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, y in te_loader:
                imgs = imgs.to(DEVICE)
                with torch.cuda.amp.autocast():
                    pred = model(imgs).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"[ft] epoch {ep:02d}  test_acc {acc:.4f}  (best {best_acc:.4f})")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": num_classes,
                "acc": best_acc, "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha, "ft_mode": FT_MODE}, CKPT)
    print(f"[ft] saved {CKPT}  best_acc {best_acc:.4f}")
    return model, num_classes


def _labels(ds):
    """Yield (idx, label) cheaply for counting classes (avoids decoding images)."""
    if isinstance(ds, _HFImageDataset):
        for i, l in enumerate(ds.ds[ds.lk]):
            yield i, int(l)
    else:
        targets = getattr(ds, "_labels", None)
        if targets is None:
            targets = getattr(ds, "targets", None)
        if targets is None:  # last resort
            targets = [ds[i][1] for i in range(len(ds))]
        for i, l in enumerate(targets):
            yield i, int(l)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (penultimate + logits), cached
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract(model, name, split, tag):
    path = os.path.join(CACHE_DIR, f"{name}_{split}_{tag}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["feats"], d["logits"], d["labels"]
    eval_tf = timm.data.create_transform(
        **timm.data.resolve_data_config({}, model=timm.create_model(MODEL_NAME)),
        is_training=False)
    ds = build_dataset(name, split, eval_tf)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    F, Z, Y = [], [], []
    model.eval()
    for imgs, y in tqdm(loader, desc=f"extract {name}/{split}", leave=False):
        imgs = imgs.to(DEVICE)
        with torch.cuda.amp.autocast():
            z, f = model(imgs, return_feat=True)
        F.append(f.float().cpu().numpy())
        Z.append(z.float().cpu().numpy())
        Y.append(np.asarray(y))
    F, Z, Y = np.concatenate(F), np.concatenate(Z), np.concatenate(Y)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, feats=F.astype(np.float16),
                        logits=Z.astype(np.float16), labels=Y)
    return F.astype(np.float32), Z.astype(np.float32), Y


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# OOD scorers — each returns ID-ness (higher = more in-distribution)
# ─────────────────────────────────────────────────────────────────────────────
def fit_scorers(Ftr, ytr, n_rays=24):
    classes = np.unique(ytr)
    Ftr_n = _unit(Ftr)
    # NCM centroids (normalized)
    cents = np.stack([_unit(Ftr_n[ytr == c].mean(0, keepdims=True))[0] for c in classes])
    # Mahalanobis: class means (raw) + tied LedoitWolf precision
    mus = np.stack([Ftr[ytr == c].mean(0) for c in classes])
    centered = np.concatenate([Ftr[ytr == c] - mus[i] for i, c in enumerate(classes)])
    cov = LedoitWolf().fit(centered)
    prec = cov.precision_.astype(np.float32)
    # Cones (per class, normalized feats), reuse ConicHull with small-class fallback
    cones = []
    for c in classes:
        Xc = Ftr_n[ytr == c]
        k = int(min(n_rays, len(Xc)))
        ch = ConicHull(n_rays=k, use_pca=False, ray_diversity="hybrid")
        if len(Xc) < 12:
            ch.extreme_rays_ = Xc
            ch.extreme_rays_index = np.arange(len(Xc))
        else:
            ch.fit(Xc)
        cones.append(ch)
    return dict(cents=cents, mus=mus, prec=prec, cones=cones, bank=Ftr_n)


def score_all(scorers, F, Z):
    """Return dict method -> ID-ness array (higher = ID)."""
    Fn = _unit(F)
    out = {}
    # ---- logit-based ----
    out["MSP"] = np.max(_softmax(Z), axis=1)
    out["Energy"] = logsumexp(Z, axis=1)                     # -energy
    out["MaxLogit"] = np.max(Z, axis=1)
    # ---- feature-based ----
    C = torch.tensor(scorers["cents"], device=DEVICE)
    Q = torch.tensor(Fn, device=DEVICE)
    out["NCM"] = (Q @ C.T).max(1).values.cpu().numpy()
    out["Mahalanobis"] = -_maha_min(F, scorers["mus"], scorers["prec"])
    out["KNN"] = -_knn_dist(Fn, scorers["bank"], k=50)
    # ---- cone ----
    best = np.full(len(F), -np.inf, np.float32)
    for ch in scorers["cones"]:
        best = np.maximum(best, ch.score_nnls_residual(Fn))  # 1 - min_c r_c/2
    out["Cone(min_c r_c)"] = best
    return out


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


def _maha_min(F, mus, prec, chunk=8192):
    """min_c (x-mu_c)^T prec (x-mu_c), over classes."""
    P = torch.tensor(prec, device=DEVICE)
    M = torch.tensor(mus, device=DEVICE)                     # (C,D)
    out = np.empty(len(F), np.float32)
    for i in range(0, len(F), chunk):
        X = torch.tensor(F[i:i + chunk], device=DEVICE)      # (B,D)
        diff = X[:, None, :] - M[None, :, :]                 # (B,C,D)
        d = torch.einsum("bcd,de,bce->bc", diff, P, diff)    # (B,C)
        out[i:i + chunk] = d.min(1).values.cpu().numpy()
    return out


def _knn_dist(Qn, bankn, k=50, chunk=4096):
    """Distance (1 - cos) to the k-th nearest training feature."""
    B = torch.tensor(bankn, device=DEVICE)
    out = np.empty(len(Qn), np.float32)
    for i in range(0, len(Qn), chunk):
        Q = torch.tensor(Qn[i:i + chunk], device=DEVICE)
        sims = Q @ B.T                                        # (b, N)
        kth = sims.topk(k, dim=1).values[:, -1]              # k-th largest cos
        out[i:i + chunk] = (1.0 - kth).cpu().numpy()
    return out


def auroc_fpr(idness_id, idness_ood):
    """AUROC (ID positive) and FPR@95TPR."""
    y = np.r_[np.ones(len(idness_id)), np.zeros(len(idness_ood))]
    s = np.r_[idness_id, idness_ood]
    auc = roc_auc_score(y, s)
    thr = np.quantile(idness_id, 0.05)                       # keep 95% of ID
    fpr = float(np.mean(idness_ood >= thr))
    return float(auc), fpr


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224")
    ap.add_argument("--ft-mode", choices=["lora", "full", "auto"], default="auto")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--n-rays", type=int, default=24)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    # Resolve model-dependent globals. LoRA needs ViT attn layers; anything else
    # (resnet18, ...) is full fine-tuned.
    global MODEL_NAME, FT_MODE, TAG, CKPT
    MODEL_NAME = args.model
    TAG = MODEL_NAME.replace("_", "").replace("patch", "p")[:14]
    if args.ft_mode == "auto":
        FT_MODE = "lora" if "vit" in MODEL_NAME.lower() else "full"
    else:
        FT_MODE = args.ft_mode
    CKPT = os.path.join(OUT_DIR, f"cub_{TAG}.pt")
    print(f"[cfg] model={MODEL_NAME} ft_mode={FT_MODE} tag={TAG}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── model ──
    if args.skip_train and os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEVICE)
        model, _ = build_model(ck["num_classes"], ck["lora_rank"], ck["lora_alpha"],
                               ft_mode=ck.get("ft_mode", "lora"))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        print(f"[load] {CKPT}  cub_acc {ck['acc']:.4f}")
    else:
        model, _ = finetune(args)

    # ── features ──
    Ftr, _, ytr = extract(model, ID_DATASET, "train", TAG)
    Fid, Zid, _ = extract(model, ID_DATASET, "test", TAG)
    print(f"[feat] CUB train {Ftr.shape}  test {Fid.shape}")

    scorers = fit_scorers(Ftr, ytr, n_rays=args.n_rays)
    idn_id = score_all(scorers, Fid, Zid)

    methods = list(idn_id.keys())
    results = {m: {} for m in methods}
    for ood in OOD_DATASETS:
        Fo, Zo, _ = extract(model, ood, "test", TAG)
        idn_ood = score_all(scorers, Fo, Zo)
        for m in methods:
            auc, fpr = auroc_fpr(idn_id[m], idn_ood[m])
            results[m][ood] = {"auroc": auc, "fpr95": fpr}
        print(f"[ood] {ood:14s} "
              + "  ".join(f"{m.split('(')[0]}:{results[m][ood]['auroc']:.3f}"
                          for m in methods))

    _report(results, methods)


def _report(results, methods):
    for m in methods:
        aucs = [results[m][o]["auroc"] for o in OOD_DATASETS]
        fprs = [results[m][o]["fpr95"] for o in OOD_DATASETS]
        results[m]["MEAN"] = {"auroc": float(np.mean(aucs)), "fpr95": float(np.mean(fprs))}
    with open(os.path.join(OUT_DIR, f"results_{TAG}.json"), "w") as f:
        json.dump(results, f, indent=2)

    cols = OOD_DATASETS + ["MEAN"]
    lines = [f"# CUB fine-tuned OOD ({MODEL_NAME}, {FT_MODE}): cone vs benchmark detectors\n",
             "AUROC (ID = CUB test, OOD = other datasets' test). Higher better. "
             "FPR@95 in the second table.\n",
             "## AUROC\n",
             "| method | " + " | ".join(cols) + " |",
             "|---|" + "--:|" * len(cols)]
    # order: benchmarks then cone, cone highlighted
    order = ["MSP", "Energy", "MaxLogit", "NCM", "Mahalanobis", "KNN", "Cone(min_c r_c)"]
    order = [m for m in order if m in methods]
    for m in order:
        row = [f"{results[m][c]['auroc']:.3f}" for c in cols]
        name = f"**{m}**" if m.startswith("Cone") else m
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    lines.append("\n## FPR@95 (lower better)\n")
    lines.append("| method | " + " | ".join(cols) + " |")
    lines.append("|---|" + "--:|" * len(cols))
    for m in order:
        row = [f"{results[m][c]['fpr95']:.3f}" for c in cols]
        name = f"**{m}**" if m.startswith("Cone") else m
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, f"report_{TAG}.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"[done] wrote {OUT_DIR}/report_{TAG}.md")


if __name__ == "__main__":
    main()
