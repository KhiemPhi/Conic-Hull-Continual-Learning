"""
cil_conic.py
------------
Class-incremental learning on CIFAR-100 (frozen ViT-B/16, PTM-CL setting).
Tests the paradigm: use the conic hull NOT as a classifier (that loses) but as a
FORGETTING-FREE ADAPTATION mechanism — project LoRA gradients onto the null space of
old classes' extreme rays (your GradientProjector) so adapting for new classes keeps
old-class features intact. A plain NCM head classifies.

Conditions (avg-incremental accuracy, higher better):
  frozen+NCM   : no backbone adaptation, NCM prototypes (ADAM/APER floor)
  RanPAC       : frozen feats -> random ReLU projection -> Gram-decorrelated prototypes (SOTA bar)
  lora-naive   : adapt LoRA per task (CE), NCM prototypes -> backbone drifts -> forgets
  lora-conicGP : same, but GradientProjector(old extreme rays) preserves old features

    HF_HUB_OFFLINE=1 python -u cil_conic.py --tasks 10 --epochs 5
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import timm

from backbone import inject_lora, freeze_non_lora, get_lora_params, GradientProjector
from conic_hull import ConicHull

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./cil_out"
MODEL = "vit_base_patch16_224"


def transforms_():
    cfg = timm.data.resolve_data_config({}, model=timm.create_model(MODEL))
    return timm.data.create_transform(**cfg, is_training=False)


def load_cifar(tf):
    tr = CIFAR100("./data", train=True, download=False, transform=tf)
    te = CIFAR100("./data", train=False, download=False, transform=tf)
    return tr, te


@torch.no_grad()
def feats_of(model, ds, idx, bs=256):
    """Parallel (num_workers) feature extraction over a subset, order preserved."""
    model.eval()
    loader = DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
                        num_workers=8, pin_memory=True)
    F, Y = [], []
    for xb, yb in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F.append(model(xb.to(DEVICE)).float().cpu().numpy())
        Y.append(np.asarray(yb))
    return np.concatenate(F), np.concatenate(Y)


def idx_for(targets, classes):
    cset = set(classes)
    return [i for i, y in enumerate(targets) if y in cset]


def ncm_eval(protos, proto_lbl, Fte, yte):
    P = torch.tensor(protos / (np.linalg.norm(protos, 1, keepdims=True) + 1e-8), device=DEVICE)
    Fn = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    pred = np.array(proto_lbl)[(torch.tensor(Fn, device=DEVICE) @ P.T).argmax(1).cpu().numpy()]
    return float((pred == yte).mean())


# ── frozen feature-based conditions (extract once) ───────────────────────────
def run_frozen_ncm(model, tr, te, order, tasks):
    tasksz = len(order) // tasks
    protos, plbl, accs = [], [], []
    seen = []
    for t in range(tasks):
        cls = order[t*tasksz:(t+1)*tasksz]; seen += list(cls)
        Ftr, ytr = feats_of(model, tr, idx_for(tr.targets, cls))
        for c in cls:
            protos.append(Ftr[ytr == c].mean(0)); plbl.append(c)
        Fte, yte = feats_of(model, te, idx_for(te.targets, seen))
        accs.append(ncm_eval(np.stack(protos), plbl, Fte, yte))
        print(f"  [frozen+NCM] task {t} acc {accs[-1]*100:.1f}", flush=True)
    return accs


def run_ranpac(model, tr, te, order, tasks, M=4096, lam=1e3):
    tasksz = len(order) // tasks
    rng = np.random.default_rng(0)
    D = model(torch.zeros(1, 3, 224, 224).to(DEVICE)).shape[-1] if False else 768
    W = rng.standard_normal((D, M)).astype(np.float32) / np.sqrt(D)
    G = np.zeros((M, M), np.float32); csum = {}; ccount = {}
    seen, accs = [], []
    for t in range(tasks):
        cls = order[t*tasksz:(t+1)*tasksz]; seen += list(cls)
        Ftr, ytr = feats_of(model, tr, idx_for(tr.targets, cls))
        phi = np.maximum(Ftr @ W, 0)                       # ReLU random projection
        G += phi.T @ phi
        for c in cls:
            csum[c] = phi[ytr == c].sum(0); ccount[c] = (ytr == c).sum()
        Ginv = np.linalg.inv(G + lam * np.eye(M, dtype=np.float32))
        labels = sorted(seen)
        Wc = np.stack([Ginv @ (csum[c] / ccount[c]) for c in labels])   # (C,M)
        Fte, yte = feats_of(model, te, idx_for(te.targets, seen))
        phite = np.maximum(Fte @ W, 0)
        pred = np.array(labels)[(phite @ Wc.T).argmax(1)]
        accs.append(float((pred == yte).mean()))
        print(f"  [RanPAC] task {t} acc {accs[-1]*100:.1f}", flush=True)
    return accs


# ── LoRA-adapting conditions ─────────────────────────────────────────────────
def run_lora(model0, tr, te, order, tasks, epochs, use_gp, lr=1e-3, rank=16):
    tasksz = len(order) // tasks
    model = model0
    inject_lora(model, rank=rank, alpha=rank, target_modules=["attn.qkv", "attn.proj"])
    freeze_non_lora(model); model.to(DEVICE)
    protos, plbl, accs = [], [], []
    all_rays = None; seen = []
    for t in range(tasks):
        cls = list(order[t*tasksz:(t+1)*tasksz]); seen += cls
        cls_to_local = {c: i for i, c in enumerate(cls)}
        head = nn.Linear(768, len(cls)).to(DEVICE)
        params = get_lora_params(model) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        gp = GradientProjector(all_rays, torch.device(DEVICE)) if (use_gp and all_rays is not None) else None
        idx = idx_for(tr.targets, cls)
        sub = Subset(tr, idx)
        loader = DataLoader(sub, batch_size=128, shuffle=True, num_workers=8, drop_last=True)
        model.train()
        for ep in range(epochs):
            for x, y in loader:
                x = x.to(DEVICE); y = torch.tensor([cls_to_local[int(v)] for v in y], device=DEVICE)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = crit(head(model(x)), y)
                scaler.scale(loss).backward()
                if gp is not None:
                    scaler.unscale_(opt); gp.project_grads(model)   # null-space of old rays
                scaler.step(opt); scaler.update()
        # prototypes for new classes (current backbone)
        Ftr, ytr = feats_of(model, tr, idx)
        for c in cls:
            protos.append(Ftr[ytr == c].mean(0)); plbl.append(c)
        # extreme rays of new classes -> accumulate protected subspace
        rays = []
        for c in cls:
            Xc = Ftr[ytr == c]; k = min(10, len(Xc))
            ch = ConicHull(n_rays=k, use_pca=False, ray_diversity="hybrid")
            (ch.fit(Xc) if len(Xc) >= 12 else setattr(ch, "extreme_rays_",
             Xc / (np.linalg.norm(Xc, 1, keepdims=True) + 1e-8)))
            rays.append(ch.extreme_rays_)
        rays = np.concatenate(rays)
        all_rays = rays if all_rays is None else np.concatenate([all_rays, rays])
        Fte, yte = feats_of(model, te, idx_for(te.targets, seen))
        accs.append(ncm_eval(np.stack(protos), plbl, Fte, yte))
        tag = "conicGP" if use_gp else "naive"
        print(f"  [lora-{tag}] task {t} acc {accs[-1]*100:.1f} (rays={all_rays.shape[0]})", flush=True)
    return accs


def summarize(accs):
    return dict(avg_inc=float(np.mean(accs)*100), last=float(accs[-1]*100), per_task=[round(a*100,1) for a in accs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--conditions", nargs="+",
                    default=["frozen", "ranpac", "lora-naive", "lora-conicGP"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tf = transforms_()
    tr, te = load_cifar(tf)
    order = np.random.default_rng(1993).permutation(100)
    print(f"[cil] CIFAR-100, {args.tasks} tasks x {100//args.tasks} classes", flush=True)

    res = {}
    if "frozen" in args.conditions or "ranpac" in args.conditions:
        m = timm.create_model(MODEL, pretrained=True, num_classes=0).to(DEVICE).eval()
        if "frozen" in args.conditions:
            res["frozen+NCM"] = summarize(run_frozen_ncm(m, tr, te, order, args.tasks))
        if "ranpac" in args.conditions:
            res["RanPAC"] = summarize(run_ranpac(m, tr, te, order, args.tasks))
    if "lora-naive" in args.conditions:
        m = timm.create_model(MODEL, pretrained=True, num_classes=0)
        res["lora-naive"] = summarize(run_lora(m, tr, te, order, args.tasks, args.epochs, use_gp=False))
    if "lora-conicGP" in args.conditions:
        m = timm.create_model(MODEL, pretrained=True, num_classes=0)
        res["lora-conicGP"] = summarize(run_lora(m, tr, te, order, args.tasks, args.epochs, use_gp=True))

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n| method | avg-inc acc | last acc |\n|---|--:|--:|")
    for k, v in res.items():
        print(f"| {k} | {v['avg_inc']:.1f} | {v['last']:.1f} |")


if __name__ == "__main__":
    main()
