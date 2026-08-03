"""
crux_ranpac.py — can transported-adapted features beat frozen features under RanPAC's head?

The design (user's "transform back to the old frame" idea, done with ONE fixed reference frame):
  * reference frame = phi_0 (frozen base model: always reproducible, never drifts)
  * task t: adapt -> phi_t, fit Psi_t: phi_t -> phi_0 on TASK-t DATA ITSELF (in-region, no replay)
  * accumulate RanPAC statistics in phi_0 coordinates:  G += H^T H,  C += H^T Y
    -> statistics never go stale (frame is fixed) => EXACT additivity => zero forgetting,
       the same guarantee RanPAC has, but the features carry the adapted backbone's quality.

Variants (identical RanPAC head + identical lambda-selection protocol; ONLY the features differ):
  A  frozen      : train phi_0(x),        test phi_0(x)          <- this IS RanPAC
  B  transported : train Psi_t(phi_t(x)), test Psi_t(phi_t(x))   <- the proposal
  B2 hybrid      : train Psi_t(phi_t(x)), test phi_0(x)          <- exact test features
  C  denoised    : train/test rank-r projected phi_0             <- denoising control
                   (does any gain in B just come from ridge shrinkage, not adaptation?)
Reference: NCM on frozen phi_0.

Decisive question:  is Psi_t(phi_t(x)) a BETTER representation than native phi_0(x)?
  B > A  => beat RanPAC (same head, better features).   B <= A => the mechanism adds nothing.
Prediction: on CIFAR-100 there is no adaptation headroom, so B <= A (negative control).
            ImageNet-R (renditions) has a real domain gap -> the place a win could appear.

Run:  DATASET=CIFAR100 python -u crux_ranpac.py
      DATASET=IMAGENETR python -u crux_ranpac.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"   # IN21k-only ckpt is CDN-blocked here
DATASET = os.environ.get("DATASET", "CIFAR100").upper()
N_TASKS = 10
EPOCHS, LR, BS = 4, 1e-4, 128
M_RP = 10000              # RanPAC random-projection width
LAMBDAS = [1e-1, 1.0, 1e1, 1e2, 1e3]
VAL_FRAC = 0.10           # per-task held-out split for lambda selection (same for all variants)
DENOISE_RANK = 256        # rank for the denoising control

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)


# ------------------------------- data -------------------------------
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


def build_data():
    """Return (train_ds, y_train, test_ds, y_test, n_classes, cpt)."""
    if DATASET == "CIFAR100":
        tr = datasets.CIFAR100("./data", train=True,  download=False, transform=TF)
        te = datasets.CIFAR100("./data", train=False, download=False, transform=TF)
        return tr, np.array(tr.targets), te, np.array(te.targets), 100, 10
    if DATASET == "IMAGENETR":
        from datasets import load_dataset
        ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
        wnid = np.array(ds["wnid"])
        classes = np.array(sorted(set(wnid.tolist())))
        lab = np.searchsorted(classes, wnid)
        # standard Split-ImageNet-R convention: random 80/20 train/test
        rs = np.random.default_rng(1993).permutation(len(lab))
        ntr = int(0.8 * len(lab))
        tr_i, te_i = rs[:ntr], rs[ntr:]
        return (HFWrap(ds, tr_i, lab[tr_i]), lab[tr_i],
                HFWrap(ds, te_i, lab[te_i]), lab[te_i], len(classes), 20)
    raise ValueError(DATASET)


TRAIN, TR_Y, TEST, TE_Y, N_CLS, CPT = build_data()
print(f"[{DATASET}] train {len(TR_Y)}  test {len(TE_Y)}  classes {N_CLS}  "
      f"{N_TASKS} tasks x {CPT}")
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]


@torch.no_grad()
def extract(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ------------------------------- RanPAC head -------------------------------
class RanPAC:
    """RP -> ReLU -> decorrelated ridge, with exactly-additive (G, C) accumulation."""

    def __init__(self, d, M, n_cls, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.P = torch.randn(d, M, generator=g).to(DEV)
        self.G = torch.zeros(M, M, device=DEV, dtype=torch.float64)
        self.C = torch.zeros(M, n_cls, device=DEV, dtype=torch.float64)
        self.n_cls, self.M = n_cls, M

    def _h(self, Z):
        return torch.relu(torch.tensor(un(Z), device=DEV, dtype=torch.float32) @ self.P)

    def accumulate(self, Z, y, bs=4096):
        for i in range(0, len(Z), bs):
            H = self._h(Z[i:i + bs]).double()
            Y = torch.zeros(H.shape[0], self.n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(H.shape[0]), torch.tensor(y[i:i + bs], device=DEV)] = 1.0
            self.G += H.T @ H
            self.C += H.T @ Y

    def solve(self, lam):
        A = self.G + lam * torch.eye(self.M, device=DEV, dtype=torch.float64)
        return torch.linalg.solve(A, self.C)

    def predict(self, Z, W, seen, bs=4096):
        seen_t = torch.tensor(np.asarray(seen), device=DEV)
        out = []
        for i in range(0, len(Z), bs):
            logits = (self._h(Z[i:i + bs]).double() @ W)[:, seen_t]
            out.append(seen_t[logits.argmax(1)].cpu().numpy())
        return np.concatenate(out)

    def fit_predict(self, Zval, yval, Zte, seen):
        """Pick lambda on the accumulated validation split, then predict test."""
        best, bestW = -1.0, None
        for lam in LAMBDAS:
            W = self.solve(lam)
            a = float((self.predict(Zval, W, seen) == yval).mean())
            if a > best:
                best, bestW, bestlam = a, W, lam
        return self.predict(Zte, bestW, seen), bestlam, best


def fit_linear(X, Y, ridge_grid=(1e-3, 1e-2, 1e-1, 1.0)):
    """Ridge map X->Y with lambda chosen on a held-out 20% of the fit data."""
    X, Y = un(X), un(Y)
    n, d = X.shape
    perm = np.random.default_rng(0).permutation(n)
    ntr = max(int(0.8 * n), 1)
    tr, va = perm[:ntr], perm[ntr:]
    s = np.trace(X[tr].T @ X[tr]) / d
    best, bestlam = -2.0, 1e-2
    for lam in ridge_grid:
        W = np.linalg.solve(X[tr].T @ X[tr] + lam * s * np.eye(d), X[tr].T @ Y[tr])
        v = float((un(X[va] @ W) * Y[va]).sum(1).mean()) if len(va) else 0.0
        if v > best:
            best, bestlam = v, lam
    return np.linalg.solve(X.T @ X + bestlam * s * np.eye(d), X.T @ Y)


# ------------------------------- phi_0 pass -------------------------------
print("=== phi_0 (frozen reference frame) ===")
phi0 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
F0_tr = extract(phi0, TRAIN, np.arange(len(TR_Y)))
F0_te = extract(phi0, TEST,  np.arange(len(TE_Y)))
del phi0; torch.cuda.empty_cache()

# denoising control: rank-r PCA basis fit on frozen train features
Xc = un(F0_tr) - un(F0_tr).mean(0, keepdims=True)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
Vr = Vt[:DENOISE_RANK].T
den = lambda Z: un(un(Z) @ Vr @ Vr.T)

VARIANTS = ["A_frozen", "B_transported", "B2_hybrid", "C_denoised"]
heads = {v: RanPAC(768, M_RP, N_CLS, seed=0) for v in VARIANTS}
val_store = {v: {"Z": [], "y": []} for v in VARIANTS}

model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
lora_params = list(get_lora_params(model))
rows = []

for t in range(N_TASKS):
    cls = np.asarray(TASKS[t])
    tr_idx = np.where(np.isin(TR_Y, cls))[0]
    # ---- adapt on task t ----
    remap = {int(c): i for i, c in enumerate(cls)}
    loader = DataLoader(Subset(TRAIN, tr_idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
    head = nn.Linear(768, CPT).to(DEV)
    opt = torch.optim.AdamW(lora_params + list(head.parameters()), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        for x, lab in loader:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ce(head(model(x).float()), y)
            opt.zero_grad(); loss.backward(); opt.step()

    seen = np.concatenate(TASKS[:t + 1])
    te_idx = np.where(np.isin(TE_Y, seen))[0]

    # ---- Psi_t : phi_t -> phi_0, fit on THIS task's data (in-region, no replay) ----
    Ft_tr = extract(model, TRAIN, tr_idx)
    Psi_W = fit_linear(Ft_tr, F0_tr[tr_idx])
    Zt_tr = un(un(Ft_tr) @ Psi_W)                    # transported task-t train
    Ft_te = extract(model, TEST, te_idx)
    Zt_te = un(un(Ft_te) @ Psi_W)                    # transported seen test (current map)

    y_task = TR_Y[tr_idx]
    nval = max(int(VAL_FRAC * len(tr_idx)), 1)
    vperm = np.random.default_rng(t).permutation(len(tr_idx))
    vi, ti = vperm[:nval], vperm[nval:]

    feats_tr = {"A_frozen": un(F0_tr[tr_idx]), "B_transported": Zt_tr,
                "B2_hybrid": Zt_tr, "C_denoised": den(F0_tr[tr_idx])}
    feats_te = {"A_frozen": un(F0_te[te_idx]), "B_transported": Zt_te,
                "B2_hybrid": un(F0_te[te_idx]), "C_denoised": den(F0_te[te_idx])}

    line = {}
    for v in VARIANTS:
        heads[v].accumulate(feats_tr[v][ti], y_task[ti])
        val_store[v]["Z"].append(feats_tr[v][vi]); val_store[v]["y"].append(y_task[vi])
        Zval = np.concatenate(val_store[v]["Z"]); yval = np.concatenate(val_store[v]["y"])
        pred, lam, vacc = heads[v].fit_predict(Zval, yval, feats_te[v], seen)
        line[v] = float((pred == TE_Y[te_idx]).mean())

    # NCM on frozen features, for reference (frozen => birth prototype == current prototype)
    mu = un(np.stack([un(F0_tr[np.where(TR_Y == c)[0]]).mean(0) for c in seen]))
    ncm = float((seen[np.argmax(un(F0_te[te_idx]) @ mu.T, axis=1)] == TE_Y[te_idx]).mean())

    # transport fidelity, for context
    fid = float((Zt_te * un(F0_te[te_idx])).sum(1).mean())
    rows.append(dict(t=t, seen=len(seen), ncm=ncm, fid=fid, **line))
    print(f"[t={t}] seen={len(seen):3d} | NCM {ncm:.4f} | "
          + " | ".join(f"{v.split('_')[0]} {line[v]:.4f}" for v in VARIANTS)
          + f" | transp-fid {fid:.3f}")

np.save(f"crux_ranpac_{DATASET}.npy", np.array(rows, dtype=object), allow_pickle=True)
print("\n" + "=" * 88)
print(f"{DATASET}  —  RanPAC head (M={M_RP}), identical for all variants; only FEATURES differ")
print("=" * 88)
print(f"{'t':>2} {'seen':>5} {'NCM-froz':>9} {'A frozen':>9} {'B transp':>9} "
      f"{'B2 hybrid':>10} {'C denois':>9} {'B-A':>7}")
for r in rows:
    print(f"{r['t']:>2} {r['seen']:>5} {r['ncm']:>9.4f} {r['A_frozen']:>9.4f} "
          f"{r['B_transported']:>9.4f} {r['B2_hybrid']:>10.4f} {r['C_denoised']:>9.4f} "
          f"{r['B_transported']-r['A_frozen']:>+7.4f}")
avg = {v: np.mean([r[v] for r in rows]) for v in VARIANTS}
print("-" * 88)
print(f"{'avg':>2} {'':>5} {np.mean([r['ncm'] for r in rows]):>9.4f} "
      + " ".join(f"{avg[v]:>9.4f}" for v in VARIANTS))
print(f"\nFINAL  A(RanPAC) {rows[-1]['A_frozen']:.4f} | B(transported) "
      f"{rows[-1]['B_transported']:.4f} | delta {rows[-1]['B_transported']-rows[-1]['A_frozen']:+.4f}")
print("B > A  => transported-adapted features beat frozen under the same head.")
print("=" * 88)
