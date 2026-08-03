"""
crux_drift.py  — the "which invariant survives adaptation?" experiment.

Question: when the backbone adapts on NEW classes, which stored geometric
structure of the OLD classes stays valid (forgetting-free) AND discriminative?

Protocol
--------
1. phi0 = frozen pretrained ViT-B/16 (IN21k, model's own normalization).
   Extract OLD-class train/test features (birth frame).
2. Adapt: inject repo LoRA (rank 32, task_specific), freeze base, train a
   50-way linear head + LoRA on NEW classes only (CE). This is the standard
   "learn task 2" step that drifts task-1 features.
3. phi1 = adapted backbone. Re-extract the SAME OLD-class samples (current frame).
4. Compare OLD-class geometry phi0 vs phi1:
     (a) centroid angular drift  eps      (the thing that forgets)
     (b) Procrustes rigidity     resid    (is drift ~ a global rotation? -> idea 1)
     (c) Gram / relative-geom    stability(is the class arrangement preserved? -> idea 3)
     (d) separability retained under each correction (stale / oracle / rotation).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
# orig_in21k weights are behind a blocked CDN on this box; use the locally-cached
# in21k->in1k ViT-B/16 (same arch/feature_dim/normalization). Drift dynamics are
# backbone-agnostic, so this is a faithful stand-in for the crux measurement.
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_OLD = 50            # classes 0..49 are "old"; 50..99 are "new"
EPOCHS = 6
LR = 1e-4
BS = 128
ROOT = "./data"


# ---- transform: MODEL'S OWN normalization (the findings' critical bug fix) ----
_probe = timm.create_model(MODEL, pretrained=False, num_classes=0)
TF = create_transform(**resolve_model_data_config(_probe), is_training=False)
print("[transform]", TF)


def _loader(train, class_lo, class_hi, shuffle=False):
    ds = datasets.CIFAR100(ROOT, train=train, download=False, transform=TF)
    y = np.array(ds.targets)
    idx = np.where((y >= class_lo) & (y < class_hi))[0]
    sub = Subset(ds, idx.tolist())
    return DataLoader(sub, batch_size=256, shuffle=shuffle, num_workers=8,
                      pin_memory=True), y[idx]


@torch.no_grad()
def extract(model, train, lo, hi):
    """Return (F [N,D] fp32, y [N]) in the given class range, deterministic order."""
    model.eval()
    loader, y = _loader(train, lo, hi, shuffle=False)
    feats = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for imgs, _ in loader:
            feats.append(model(imgs.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(feats, 0), y


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def class_means(F, y, classes):
    """L2-normalized mean direction per class, row-aligned to `classes`."""
    M = np.stack([unit(F[y == c]).mean(0) for c in classes])
    return unit(M)


def ncm_acc(F_query, y_query, means, classes):
    q = unit(F_query)
    sims = q @ means.T                      # (N, C)
    pred = classes[np.argmax(sims, axis=1)]
    return float((pred == y_query).mean())


def deg(cos):
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# ============================ 1. BIRTH FRAME phi0 =============================
print("\n=== phi0: frozen pretrained backbone (birth frame) ===")
phi0 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
F0_old_tr, y0_old_tr = extract(phi0, True,  0, N_OLD)
F0_old_te, y0_old_te = extract(phi0, False, 0, N_OLD)
F0_new_te, y0_new_te = extract(phi0, False, N_OLD, 100)
del phi0; torch.cuda.empty_cache()

# ============================ 2. ADAPT ON NEW ================================
print("\n=== adapting LoRA on NEW classes (50..99) only ===")
phi1 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                     lora_rank=32, lora_alpha=4.0, lora_config="task_specific")
freeze_non_lora(phi1)
head = nn.Linear(768, 100 - N_OLD).to(DEV)      # 50-way head over NEW classes
params = list(get_lora_params(phi1)) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()

train_ds = datasets.CIFAR100(ROOT, train=True, download=False, transform=TF)
ty = np.array(train_ds.targets)
new_idx = np.where(ty >= N_OLD)[0]
train_loader = DataLoader(Subset(train_ds, new_idx.tolist()), batch_size=BS,
                          shuffle=True, num_workers=8, pin_memory=True)

phi1.train()
for ep in range(EPOCHS):
    tot, correct, run = 0, 0, 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(DEV, non_blocking=True)
        labels = (labels - N_OLD).to(DEV)       # remap 50..99 -> 0..49
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(phi1(imgs).float())
            loss = lossf(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        run += loss.item() * imgs.size(0); tot += imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    print(f"  epoch {ep+1}/{EPOCHS}  loss {run/tot:.4f}  new-train-acc {correct/tot:.4f}")

# ============================ 3. CURRENT FRAME phi1 ==========================
print("\n=== phi1: re-extract OLD classes in adapted frame ===")
F1_old_tr, y1_old_tr = extract(phi1, True,  0, N_OLD)
F1_old_te, y1_old_te = extract(phi1, False, 0, N_OLD)
F1_new_te, y1_new_te = extract(phi1, False, N_OLD, 100)
assert np.array_equal(y0_old_te, y1_old_te)     # same samples, same order

np.savez("crux_drift_feats.npz",
         F0_old_tr=F0_old_tr, F0_old_te=F0_old_te, F0_new_te=F0_new_te,
         F1_old_tr=F1_old_tr, F1_old_te=F1_old_te, F1_new_te=F1_new_te,
         y_old_tr=y0_old_tr, y_old_te=y0_old_te, y_new_te=y0_new_te)

# ============================ 4. METRICS ====================================
OLD = np.arange(0, N_OLD)
NEW = np.arange(N_OLD, 100)

M0 = class_means(F0_old_tr, y0_old_tr, OLD)      # birth means (50, D)
M1 = class_means(F1_old_tr, y1_old_tr, OLD)      # current means (oracle refit)

# (a) centroid angular drift eps (cross-frame, per class) + crowding gamma
eps_raw = deg((M0 * M1).sum(1))                  # angle between birth & current mean
G0 = M0 @ M0.T
gamma = deg(G0[~np.eye(N_OLD, dtype=bool)].reshape(N_OLD, N_OLD - 1).max(1))  # nearest neighbor angle
gamma_min = float(gamma.min())

# (b) Procrustes rigidity on row-aligned OLD-test features (phi1 -> phi0)
A = unit(F1_old_te); B = unit(F0_old_te)
U, _, Vt = np.linalg.svd(A.T @ B)
R = U @ Vt                                       # orthogonal, maps phi1 -> phi0
resid = np.linalg.norm(A @ R - B) / np.linalg.norm(B)
# residual (non-rigid) drift of class means after removing the global rotation
M1_rot = unit(M1 @ R)
eps_resid = deg((M0 * M1_rot).sum(1))

# (c) Gram / relative-geometry stability
G1 = M1 @ M1.T
off = ~np.eye(N_OLD, dtype=bool)
gram_rel = np.linalg.norm(G1[off] - G0[off]) / np.linalg.norm(G0[off])
gram_corr = float(np.corrcoef(G0[off], G1[off])[0, 1])

# (d) separability of OLD test in phi1 under each stored quantity
acc_stale   = ncm_acc(F1_old_te, y1_old_te, M0, OLD)              # birth means, no correction (FORGETTING)
acc_oracle  = ncm_acc(F1_old_te, y1_old_te, M1, OLD)             # refit means (needs old data; upper bound)
acc_rot     = ncm_acc(unit(F1_old_te) @ R, y1_old_te, M0, OLD)  # rotate query into birth frame, birth means (idea 1)
acc_birth   = ncm_acc(F0_old_te, y0_old_te, M0, OLD)             # no adaptation at all (frozen SOTA regime)

# sanity: did adaptation actually improve NEW-class features?
Mn0 = class_means(F0_new_te, y0_new_te, NEW)   # birth new means (test-as-proto, quick proxy)
Mn1 = class_means(F1_new_te, y1_new_te, NEW)
acc_new0 = ncm_acc(F0_new_te, y0_new_te, Mn0, NEW)
acc_new1 = ncm_acc(F1_new_te, y1_new_te, Mn1, NEW)

# ============================ REPORT ========================================
print("\n" + "=" * 68)
print("CRUX RESULTS  (CIFAR-100, 50 old / 50 new, LoRA r32, %d epochs)" % EPOCHS)
print("=" * 68)
print("\n-- adaptation sanity (was drift 'paid for'?) --")
print(f"  NEW-class NCM   phi0 {acc_new0:.4f}  ->  phi1 {acc_new1:.4f}   (Δ {acc_new1-acc_new0:+.4f})")

print("\n-- (a) centroid drift eps  vs  crowding gamma --")
print(f"  eps (birth->current mean angle):  mean {eps_raw.mean():5.2f}°  median {np.median(eps_raw):5.2f}°  max {eps_raw.max():5.2f}°")
print(f"  gamma (nearest inter-class angle): min {gamma_min:5.2f}°   -> forgetting-free needs eps < gamma/2 = {gamma_min/2:5.2f}°")
print(f"  eps < gamma/2 ?  {'YES (safe)' if eps_raw.mean() < gamma_min/2 else 'NO (drift exceeds crowding budget)'}")

print("\n-- (b) Procrustes rigidity (is drift ~ a single global rotation?) --")
print(f"  relative residual ||phi1·R - phi0|| / ||phi0|| : {resid:.4f}   (0=perfectly rigid)")
print(f"  residual mean class drift after removing R      : {eps_resid.mean():5.2f}°  (vs raw {eps_raw.mean():5.2f}°)")
print(f"  -> global rotation explains {100*(1-eps_resid.mean()/eps_raw.mean()):.1f}% of the mean drift")

print("\n-- (c) Gram / relative-geometry stability --")
print(f"  relative Frobenius change of off-diag Gram : {gram_rel:.4f}  (0=arrangement frozen)")
print(f"  off-diagonal Pearson corr (G0 vs G1)       : {gram_corr:.4f}  (1=arrangement frozen)")

print("\n-- (d) OLD-class separability in the adapted frame phi1 --")
print(f"  frozen (no adapt, SOTA regime)      : {acc_birth:.4f}")
print(f"  STALE birth means (forgetting)      : {acc_stale:.4f}   <- the disease")
print(f"  ORACLE refit means (needs old data) : {acc_oracle:.4f}   <- upper bound")
print(f"  GLOBAL-ROTATION transport (idea 1)  : {acc_rot:.4f}   <- forgetting-free candidate")
print(f"  recovered by rotation: {100*(acc_rot-acc_stale)/(acc_oracle-acc_stale+1e-9):.1f}% of the stale->oracle gap")
print("=" * 68)
