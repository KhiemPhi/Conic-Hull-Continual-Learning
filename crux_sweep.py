"""
crux_sweep.py — does "drift is a benign coherent rotation" survive DEPTH?

10-task sequential CIL on CIFAR-100 (10 classes/task, shared LoRA adapter that is
continually reshaped each task = the drift generator). At every step we measure:

  FIXED COHORT (task 0, tracked for all 9 later adaptations, birth frame = phi_1):
    - eps        : birth->current class-mean angle (cumulative drift)
    - rigidity   : Procrustes residual + residual class-drift after removing one R
    - Gram corr  : is the class arrangement preserved?
    - accuracy   : stale birth means / global-rotation transport / oracle refit

  RUNNING seen-way CIL accuracy (all classes seen so far):
    - frozen     : never adapt (SOTA-ish regime), means & queries in phi_0
    - stale      : adapt, store means at birth, classify current queries (no fix)
    - oracle     : adapt, refit all means in the current frame (needs old data)

If cohort stale ~ oracle and rigidity stays high across 10 steps -> coherent-rotation
thesis holds, drift correction is cheap/unnecessary. If they diverge with depth ->
idea 1 (rigid-constrained transport) is a real, needed method.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import timm
from timm.data import resolve_model_data_config, create_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = int(os.environ.get("SEED", 0))   # was hardcoded 0; seeds now env-driven
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # orig_in21k CDN blocked; same arch/norm
N_TASKS = 10
CPT = 10                 # classes per task
EPOCHS = 4
LR = 1e-4
BS = 128
N_TR = 128               # train samples/class used for means + Procrustes (fixed indices)

rng = np.random.default_rng(SEED)
ORDER = rng.permutation(100)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)

TRAIN = datasets.CIFAR100("./data", train=True,  download=False, transform=TF)
TEST  = datasets.CIFAR100("./data", train=False, download=False, transform=TF)
TR_Y = np.array(TRAIN.targets); TE_Y = np.array(TEST.targets)
# fixed per-class index lists (deterministic row-alignment across frames)
TR_IDX = {c: np.where(TR_Y == c)[0][:N_TR] for c in range(100)}
TE_IDX = {c: np.where(TE_Y == c)[0]        for c in range(100)}


@torch.no_grad()
def feats(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for imgs, _ in loader:
            out.append(model(imgs.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def mean_of(F):
    return unit(unit(F).mean(0, keepdims=True))[0]


def acc(query_by_c, means_by_c, classes):
    """NCM over `classes`: query_by_c[c] (N,D), means_by_c[c] (D,)."""
    classes = np.asarray(classes)
    M = np.stack([means_by_c[c] for c in classes])
    tot = corr = 0
    for c in classes:
        q = unit(query_by_c[c])
        pred = classes[np.argmax(q @ M.T, axis=1)]
        corr += int((pred == c).sum()); tot += len(q)
    return corr / tot


def deg(cos):
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# ---- frozen phi_0 reference (base weights == adaptive model before training) ----
print("=== extracting frozen phi_0 reference (all 100 classes) ===")
frozen = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
F0_te = {c: feats(frozen, TEST,  TE_IDX[c]) for c in range(100)}
F0_tr = {c: feats(frozen, TRAIN, TR_IDX[c]) for c in range(100)}
frozen_mean = {c: mean_of(F0_tr[c]) for c in range(100)}
del frozen; torch.cuda.empty_cache()

# ---- adaptive model: one shared LoRA adapter, reshaped every task ----
model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
lora_params = list(get_lora_params(model))

birth_mean = {}                          # class -> mean at its birth frame
COHORT = list(TASKS[0])                  # tracked forever; birth frame = phi_1
cohort_birth = {}                        # phi_1 test feats + train feats + means
hist = {k: [] for k in ("eps", "rigid_resid", "rigid_deg", "gram_corr",
                        "coh_stale", "coh_transport", "coh_oracle",
                        "seen_frozen", "seen_stale", "seen_oracle", "new_acc")}

for t in range(N_TASKS):
    cls = list(TASKS[t])
    # ---- train shared LoRA + fresh head on THIS task's 10 classes ----
    remap = {c: i for i, c in enumerate(cls)}
    tr_idx = np.concatenate([np.where(TR_Y == c)[0] for c in cls])
    loader = DataLoader(Subset(TRAIN, tr_idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
    head = nn.Linear(768, CPT).to(DEV)
    opt = torch.optim.AdamW(lora_params + list(head.parameters()), lr=LR, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for ep in range(EPOCHS):
        c_ok = n = 0
        for imgs, labels in loader:
            imgs = imgs.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in labels], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(model(imgs).float()); loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            c_ok += int((logits.argmax(1) == y).sum()); n += len(y)
    new_acc = c_ok / n                          # this task's train acc (adaptation sanity)

    # ---- current frame phi_{t+1}: extract all seen ----
    seen = [c for tk in TASKS[:t + 1] for c in tk]
    te_cur = {c: feats(model, TEST,  TE_IDX[c]) for c in seen}
    tr_cur = {c: feats(model, TRAIN, TR_IDX[c]) for c in seen}
    oracle_mean = {c: mean_of(tr_cur[c]) for c in seen}
    for c in cls:                                # newly-born classes: store birth means
        birth_mean[c] = oracle_mean[c]
    if t == 0:                                   # freeze the cohort birth frame
        cohort_birth = {"te": {c: te_cur[c].copy() for c in COHORT},
                        "tr": {c: tr_cur[c].copy() for c in COHORT},
                        "mean": {c: oracle_mean[c] for c in COHORT}}

    # ---- seen-way CIL accuracy ----
    seen_frozen = acc(F0_te,  frozen_mean, seen)      # never adapted
    seen_stale  = acc(te_cur, birth_mean,  seen)      # adapt, birth means, no fix
    seen_oracle = acc(te_cur, oracle_mean, seen)      # adapt, refit means

    # ---- fixed-cohort diagnostics (drift phi_1 -> phi_{t+1}) ----
    A = unit(np.concatenate([tr_cur[c]           for c in COHORT]))  # current frame
    B = unit(np.concatenate([cohort_birth["tr"][c] for c in COHORT]))  # birth frame (aligned rows)
    U, _, Vt = np.linalg.svd(A.T @ B); R = U @ Vt                     # phi_{t+1} -> phi_1
    rigid_resid = float(np.linalg.norm(A @ R - B) / np.linalg.norm(B))
    eps = np.mean([deg(cohort_birth["mean"][c] @ oracle_mean[c]) for c in COHORT])
    eps_resid = np.mean([deg(cohort_birth["mean"][c] @ unit((oracle_mean[c] @ R)[None])[0])
                         for c in COHORT])
    M0 = np.stack([cohort_birth["mean"][c] for c in COHORT])
    M1 = np.stack([oracle_mean[c]          for c in COHORT])
    G0, G1 = M0 @ M0.T, M1 @ M1.T
    off = ~np.eye(len(COHORT), dtype=bool)
    gram_corr = float(np.corrcoef(G0[off], G1[off])[0, 1])

    coh_stale     = acc(te_cur, cohort_birth["mean"], COHORT)
    coh_oracle    = acc(te_cur, oracle_mean,          COHORT)
    te_rot = {c: te_cur[c] @ R for c in COHORT}                        # query -> birth frame
    coh_transport = acc(te_rot, cohort_birth["mean"], COHORT)

    for k, v in dict(eps=eps, rigid_resid=rigid_resid, rigid_deg=eps_resid,
                     gram_corr=gram_corr, coh_stale=coh_stale,
                     coh_transport=coh_transport, coh_oracle=coh_oracle,
                     seen_frozen=seen_frozen, seen_stale=seen_stale,
                     seen_oracle=seen_oracle, new_acc=new_acc).items():
        hist[k].append(v)

    print(f"[t={t}] seen={len(seen):3d}  newAcc {new_acc:.3f} | "
          f"COHORT eps {eps:5.1f}° rigid {100*(1-eps_resid/max(eps,1e-6)):4.0f}% "
          f"gram {gram_corr:.2f} | stale {coh_stale:.3f} transp {coh_transport:.3f} "
          f"orac {coh_oracle:.3f} | SEEN froz {seen_frozen:.3f} stale {seen_stale:.3f} "
          f"orac {seen_oracle:.3f}")

# ============================ REPORT + PLOT ================================
np.savez(f"crux_sweep_hist_s{SEED}.npz", **{k: np.array(v) for k, v in hist.items()},
         order=ORDER)
steps = np.arange(1, N_TASKS + 1)
print("\n" + "=" * 72)
print("FINAL (after 10 tasks, 100-way):")
print(f"  seen-way  frozen {hist['seen_frozen'][-1]:.4f} | "
      f"stale {hist['seen_stale'][-1]:.4f} | oracle {hist['seen_oracle'][-1]:.4f}")
print(f"  cohort    stale  {hist['coh_stale'][-1]:.4f} | "
      f"transport {hist['coh_transport'][-1]:.4f} | oracle {hist['coh_oracle'][-1]:.4f}")
print(f"  cohort drift eps {hist['eps'][-1]:.1f}°  rigidity "
      f"{100*(1-hist['rigid_deg'][-1]/max(hist['eps'][-1],1e-6)):.0f}%  "
      f"gram-corr {hist['gram_corr'][-1]:.2f}")
print("=" * 72)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot(steps, hist["coh_stale"], "o-", label="stale (no fix)")
ax[0, 0].plot(steps, hist["coh_transport"], "s-", label="transport (1 rotation)")
ax[0, 0].plot(steps, hist["coh_oracle"], "^-", label="oracle (refit)")
ax[0, 0].set_title("Fixed cohort (task 0) 10-way accuracy vs depth")
ax[0, 0].set_xlabel("tasks trained"); ax[0, 0].set_ylabel("accuracy"); ax[0, 0].legend()

ax[0, 1].plot(steps, hist["eps"], "o-", label="eps (birth→now mean angle)")
ax[0, 1].plot(steps, hist["rigid_deg"], "s-", label="residual after 1 rotation")
ax[0, 1].axhline(0, color="k", lw=.5)
ax[0, 1].set_title("Cohort drift: total vs non-rigid residual")
ax[0, 1].set_xlabel("tasks trained"); ax[0, 1].set_ylabel("degrees"); ax[0, 1].legend()

ax[1, 0].plot(steps, hist["gram_corr"], "o-")
ax[1, 0].set_title("Cohort Gram off-diagonal correlation (arrangement preserved)")
ax[1, 0].set_xlabel("tasks trained"); ax[1, 0].set_ylabel("corr"); ax[1, 0].set_ylim(0, 1)

ax[1, 1].plot(steps, hist["seen_frozen"], "o-", label="frozen (no adapt)")
ax[1, 1].plot(steps, hist["seen_stale"], "s-", label="adapt + stale means")
ax[1, 1].plot(steps, hist["seen_oracle"], "^-", label="adapt + oracle refit")
ax[1, 1].set_title("Running seen-way CIL accuracy")
ax[1, 1].set_xlabel("tasks trained"); ax[1, 1].set_ylabel("accuracy"); ax[1, 1].legend()
plt.tight_layout(); plt.savefig("crux_sweep.png", dpi=110)
print(f"saved crux_sweep.png and crux_sweep_hist_s{SEED}.npz")
