"""
crux_relational.py — attack representational forgetting with Gram-preserving adaptation.

Identical protocol to crux_sweep.py (same seed/order/epochs) PLUS a relational
distillation loss on a replay buffer:

    L = CE(new task) + lambda * || Ghat(phi_now(replay)) - Ghat(phi_ref(replay)) ||^2

where Ghat = normalized-feature Gram (cosine matrix). This PERMITS the free global
rotation (rotation preserves Gram; the net wants it anyway) but FORBIDS the non-rigid
distortion that destroys old-class separability.

Hypothesis: with lambda>0, seen-way ORACLE stays >= frozen (representational forgetting
killed) AND rigidity -> ~100% (so stale+transport = oracle). lambda=0 reproduces the
crux_sweep baseline (frozen 0.815 | oracle 0.725 | stale 0.300 at 100 classes).

Run:  LAMBDA_GRAM=50 python crux_relational.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT, EPOCHS, LR, BS, N_TR = 10, 10, 4, 1e-4, 128, 128
M_REPLAY = 20                                  # exemplars/class in replay buffer
GB = 128                                        # Gram-loss batch size
LAM = float(os.environ.get("LAMBDA_GRAM", "50"))
print(f">>> LAMBDA_GRAM = {LAM}")

rng = np.random.default_rng(SEED)
ORDER = rng.permutation(100)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)
TRAIN = datasets.CIFAR100("./data", train=True,  download=False, transform=TF)
TEST  = datasets.CIFAR100("./data", train=False, download=False, transform=TF)
TR_Y = np.array(TRAIN.targets); TE_Y = np.array(TEST.targets)
TR_IDX = {c: np.where(TR_Y == c)[0][:N_TR] for c in range(100)}
TE_IDX = {c: np.where(TE_Y == c)[0]        for c in range(100)}
REP_IDX = {c: np.where(TR_Y == c)[0][:M_REPLAY] for c in range(100)}


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


def load_images(ds, idx):
    """Preload transformed images into one CPU tensor (for replay re-forwarding)."""
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    return torch.cat([x for x, _ in loader], 0)


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
def mean_of(F_): return unit(unit(F_).mean(0, keepdims=True))[0]
def deg(cos): return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def acc(query_by_c, means_by_c, classes):
    classes = np.asarray(classes)
    M = np.stack([means_by_c[c] for c in classes])
    tot = corr = 0
    for c in classes:
        q = unit(query_by_c[c])
        pred = classes[np.argmax(q @ M.T, axis=1)]
        corr += int((pred == c).sum()); tot += len(q)
    return corr / tot


print("=== frozen phi_0 reference ===")
frozen = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
F0_te = {c: feats(frozen, TEST,  TE_IDX[c]) for c in range(100)}
F0_tr = {c: feats(frozen, TRAIN, TR_IDX[c]) for c in range(100)}
frozen_mean = {c: mean_of(F0_tr[c]) for c in range(100)}
del frozen; torch.cuda.empty_cache()

model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
lora_params = list(get_lora_params(model))

birth_mean, birth_mean_phi1 = {}, {}           # birth frame / transported-to-phi_1
COHORT = list(TASKS[0]); cohort_birth = {}
X_rep = None                                     # (n_old*M, C,H,W) cpu tensor of old replay
hist = {k: [] for k in ("eps", "rigid_deg", "gram_corr", "seen_frozen", "seen_stale",
                        "seen_transport", "seen_oracle", "coh_stale", "coh_transport",
                        "coh_oracle", "new_acc", "gram_loss")}

for t in range(N_TASKS):
    cls = list(TASKS[t])
    remap = {c: i for i, c in enumerate(cls)}
    tr_idx = np.concatenate([np.where(TR_Y == c)[0] for c in cls])
    loader = DataLoader(Subset(TRAIN, tr_idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
    head = nn.Linear(768, CPT).to(DEV)
    opt = torch.optim.AdamW(lora_params + list(head.parameters()), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    # snapshot reference geometry of OLD replay (current frame, detached)
    B_ref = None
    if X_rep is not None and LAM > 0:
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            B_ref = torch.cat([model(X_rep[i:i+256].to(DEV)).float()
                               for i in range(0, len(X_rep), 256)], 0)
        B_ref = F.normalize(B_ref, dim=1)        # (n_old*M, 768)

    model.train()
    last_gram = 0.0
    for ep in range(EPOCHS):
        c_ok = n = 0
        for imgs, labels in loader:
            imgs = imgs.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in labels], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(model(imgs).float())
                loss = ce(logits, y)
            if B_ref is not None:
                ridx = torch.randint(0, B_ref.shape[0], (min(GB, B_ref.shape[0]),))
                rimgs = X_rep[ridx].to(DEV, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    A = F.normalize(model(rimgs).float(), dim=1)
                Gr = A @ A.T
                with torch.no_grad():
                    Gref = B_ref[ridx] @ B_ref[ridx].T
                l_gram = F.mse_loss(Gr, Gref)
                loss = loss + LAM * l_gram
                last_gram = float(l_gram)
            opt.zero_grad(); loss.backward(); opt.step()
            c_ok += int((logits.argmax(1) == y).sum()); n += len(y)
    new_acc = c_ok / n

    # grow replay buffer with THIS task's exemplars (they become "old" next task)
    new_rep = load_images(TRAIN, np.concatenate([REP_IDX[c] for c in cls]))
    X_rep = new_rep if X_rep is None else torch.cat([X_rep, new_rep], 0)

    # ---- current frame phi_{t+1} ----
    seen = [c for tk in TASKS[:t + 1] for c in tk]
    te_cur = {c: feats(model, TEST,  TE_IDX[c]) for c in seen}
    tr_cur = {c: feats(model, TRAIN, TR_IDX[c]) for c in seen}
    oracle_mean = {c: mean_of(tr_cur[c]) for c in seen}
    if t == 0:
        cohort_birth = {"tr": {c: tr_cur[c].copy() for c in COHORT},
                        "mean": {c: oracle_mean[c] for c in COHORT}}

    # global rotation R: phi_{t+1} -> phi_1, fit on cohort replay geometry
    A_ = unit(np.concatenate([tr_cur[c]              for c in COHORT]))
    B_ = unit(np.concatenate([cohort_birth["tr"][c]  for c in COHORT]))
    U, _, Vt = np.linalg.svd(A_.T @ B_); R = U @ Vt
    for c in cls:                                # store birth means (both frames)
        birth_mean[c] = oracle_mean[c]
        birth_mean_phi1[c] = unit((oracle_mean[c] @ R)[None])[0]

    # ---- metrics ----
    seen_frozen = acc(F0_te,  frozen_mean, seen)
    seen_stale  = acc(te_cur, birth_mean,  seen)
    seen_oracle = acc(te_cur, oracle_mean, seen)
    te_rot = {c: te_cur[c] @ R for c in seen}   # map queries -> phi_1
    seen_transport = acc(te_rot, birth_mean_phi1, seen)

    eps = np.mean([deg(cohort_birth["mean"][c] @ oracle_mean[c]) for c in COHORT])
    eps_resid = np.mean([deg(cohort_birth["mean"][c] @ unit((oracle_mean[c] @ R)[None])[0])
                         for c in COHORT])
    M0 = np.stack([cohort_birth["mean"][c] for c in COHORT])
    M1 = np.stack([oracle_mean[c]          for c in COHORT])
    off = ~np.eye(len(COHORT), dtype=bool)
    gram_corr = float(np.corrcoef((M0 @ M0.T)[off], (M1 @ M1.T)[off])[0, 1])
    coh_stale = acc(te_cur, cohort_birth["mean"], COHORT)
    coh_oracle = acc(te_cur, oracle_mean, COHORT)
    coh_transport = acc({c: te_cur[c] @ R for c in COHORT}, cohort_birth["mean"], COHORT)

    for k, v in dict(eps=eps, rigid_deg=eps_resid, gram_corr=gram_corr,
                     seen_frozen=seen_frozen, seen_stale=seen_stale,
                     seen_transport=seen_transport, seen_oracle=seen_oracle,
                     coh_stale=coh_stale, coh_transport=coh_transport,
                     coh_oracle=coh_oracle, new_acc=new_acc, gram_loss=last_gram).items():
        hist[k].append(v)

    print(f"[t={t}] seen={len(seen):3d} newAcc {new_acc:.3f} gramL {last_gram:.4f} | "
          f"COH eps {eps:4.0f}° rigid {100*(1-eps_resid/max(eps,1e-6)):3.0f}% "
          f"gram {gram_corr:.2f} | SEEN froz {seen_frozen:.3f} stale {seen_stale:.3f} "
          f"transp {seen_transport:.3f} orac {seen_oracle:.3f}")

np.savez(f"crux_relational_lam{LAM:g}.npz", **{k: np.array(v) for k, v in hist.items()})
print("\n" + "=" * 72)
print(f"FINAL (lambda={LAM:g}, 100-way):  frozen {hist['seen_frozen'][-1]:.4f} | "
      f"stale {hist['seen_stale'][-1]:.4f} | transport {hist['seen_transport'][-1]:.4f} | "
      f"oracle {hist['seen_oracle'][-1]:.4f}")
print(f"  vs baseline lambda=0:            frozen 0.8145 | stale 0.3000 | "
      f"transport   ??? | oracle 0.7247")
print(f"  cohort rigidity {100*(1-hist['rigid_deg'][-1]/max(hist['eps'][-1],1e-6)):.0f}%  "
      f"gram-corr {hist['gram_corr'][-1]:.2f}")
print("  WIN if oracle >= frozen (repr. forgetting killed) AND transport ~ oracle")
print("=" * 72)
