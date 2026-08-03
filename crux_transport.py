"""
crux_transport.py — can we map NEW features back into the OLD backbone's space?

Fit transport maps Psi: phi_adapted(x) -> phi_0(x) on OLD-class replay pairs, across a
hierarchy of map classes (identity -> orthogonal -> linear -> nonlinear MLP). Then, on
HELD-OUT old test, measure:
  (a) reconstruction: cosine(Psi(phi_adapted(x)), phi_0(x))   -- how well old features rebuild
  (b) decision recovery: NCM accuracy with phi_0-frame prototypes on Psi(phi_adapted(x_test))

Theory predictions:
  * transport accuracy <= oracle (Bayes info in adapted features; data-processing bound)
  * NAIVE (lam=0): oracle < frozen (collapse) => even nonlinear Psi plateaus BELOW frozen
    => the gap is IRREDUCIBLE (non-invertible drift, real information loss)
  * GRAM (lam=50): oracle ~ frozen (invertible) => Psi reconstructs phi_0 well, even
    orthogonal/linear recovers ~frozen => drift is (near-)invertible / gauge
  * if MLP-transport >> linear-transport: retained info was there but NCM read it linearly
    => "representational forgetting" partly a readout limit, not true collapse
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
N_OLD = 50
EPOCHS, LR, BS = 4, 1e-4, 128
N_FIT = 200        # old train samples/class used to FIT transports (the "replay")
M_GRAM = 20        # old exemplars/class for the Gram loss during adaptation
GB = 128

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)
TRAIN = datasets.CIFAR100("./data", train=True,  download=False, transform=TF)
TEST  = datasets.CIFAR100("./data", train=False, download=False, transform=TF)
TR_Y, TE_Y = np.array(TRAIN.targets), np.array(TEST.targets)
FIT_IDX = np.concatenate([np.where(TR_Y == c)[0][:N_FIT] for c in range(N_OLD)])   # old fit set
TE_IDX  = np.concatenate([np.where(TE_Y == c)[0] for c in range(N_OLD)])           # old test
GRAM_IDX = np.concatenate([np.where(TR_Y == c)[0][:M_GRAM] for c in range(N_OLD)])


@torch.no_grad()
def extract(model, idx, ds):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV)).float().cpu().numpy())
    return np.concatenate(out, 0)


def load_imgs(idx, ds):
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    return torch.cat([x for x, _ in loader], 0)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def protos(Fn, y):
    return un(np.stack([un(Fn[y == c]).mean(0) for c in range(N_OLD)]))


def ncm(Fn, y, P):
    return float((np.argmax(un(Fn) @ P.T, axis=1) == y).mean())


y_fit = TR_Y[FIT_IDX]; y_te = TE_Y[TE_IDX]

# ---------------- frozen phi_0 (the OLD backbone) ----------------
print("=== frozen phi_0 (old backbone) ===")
frozen = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
F0_fit = extract(frozen, FIT_IDX, TRAIN)      # old train in phi_0
F0_te  = extract(frozen, TE_IDX,  TEST)       # old test in phi_0
P0 = protos(F0_fit, y_fit)                     # phi_0-frame prototypes
frozen_acc = ncm(F0_te, y_te, P0)
# Gram reference = phi_0 features of the gram exemplars.
# NOTE: FIT_IDX is class-blocked, NOT globally sorted -> np.searchsorted is invalid here
# (it mis-mapped 983/1000 rows). Use a boolean mask: GRAM_IDX subset of FIT_IDX and both
# are in (class, ascending) order, so isin preserves the pairing.
gram_pos = np.where(np.isin(FIT_IDX, GRAM_IDX))[0]
assert np.array_equal(FIT_IDX[gram_pos], GRAM_IDX), "gram reference misaligned"
B_ref = F.normalize(torch.tensor(F0_fit[gram_pos]).to(DEV), dim=1)
X_gram = load_imgs(GRAM_IDX, TRAIN)
del frozen; torch.cuda.empty_cache()
print(f"frozen old-{N_OLD} NCM = {frozen_acc:.4f}")


def adapt(lam):
    """Adapt shared LoRA on NEW classes (50..99); optional Gram loss preserving old geometry."""
    m = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(m)
    lp = list(get_lora_params(m))
    new_idx = np.where(TR_Y >= N_OLD)[0]
    loader = DataLoader(Subset(TRAIN, new_idx.tolist()), batch_size=BS, shuffle=True, num_workers=8)
    head = nn.Linear(768, 100 - N_OLD).to(DEV)
    opt = torch.optim.AdamW(lp + list(head.parameters()), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    m.train()
    for ep in range(EPOCHS):
        for x, lab in loader:
            x = x.to(DEV); y = (lab - N_OLD).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ce(head(m(x).float()), y)
            if lam > 0:
                ridx = torch.randint(0, X_gram.shape[0], (GB,))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    A = F.normalize(m(X_gram[ridx].to(DEV)).float(), dim=1)
                loss = loss + lam * F.mse_loss(A @ A.T, (B_ref[ridx] @ B_ref[ridx].T))
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def fit_transports(Xtr, Ytr):
    """Return dict name -> callable Psi on normalized features (predict phi_0-normalized)."""
    Xn = un(Xtr); Yn = un(Ytr)
    out = {"identity": lambda Z: un(Z)}
    # orthogonal Procrustes  (rotation)
    U, _, Vt = np.linalg.svd(Xn.T @ Yn); R = U @ Vt
    out["orthogonal"] = lambda Z, R=R: un(un(Z) @ R)
    # linear ridge  (GL / affine)
    d = Xn.shape[1]; A = Xn.T @ Xn + 1e-1 * np.eye(d); W = np.linalg.solve(A, Xn.T @ Yn)
    out["linear"] = lambda Z, W=W: un(un(Z) @ W)
    # nonlinear MLP
    net = nn.Sequential(nn.Linear(d, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(),
                        nn.Linear(2048, d)).to(DEV)
    Xt = torch.tensor(Xn, device=DEV); Yt = torch.tensor(Yn, device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    net.train()
    for ep in range(300):
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            pred = F.normalize(net(Xt[b]), dim=1)
            loss = (1 - (pred * F.normalize(Yt[b], dim=1)).sum(1)).mean()  # 1 - cosine
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def mlp(Z, net=net):
        with torch.no_grad():
            P = F.normalize(net(torch.tensor(un(Z), device=DEV, dtype=torch.float32)), dim=1)
        return P.cpu().numpy()
    out["mlp"] = mlp
    return out


print(f"\n{'='*78}\n{'variant':>10} {'transport':>11} {'recon_cos':>10} {'NCM(phi0 proto)':>16}")
print(f"{'':>10} {'frozen*':>11} {'1.000':>10} {frozen_acc:>16.4f}   <- target (old backbone)")
print("=" * 78)

results = {}
for lam in [0.0, 50.0]:
    tag = f"lam={lam:g}"
    m = adapt(lam)
    Fa_fit = extract(m, FIT_IDX, TRAIN)
    Fa_te  = extract(m, TE_IDX,  TEST)
    del m; torch.cuda.empty_cache()
    oracle = ncm(Fa_te, y_te, protos(Fa_fit, y_fit))   # info ceiling (linear readout on adapted)
    stale  = ncm(Fa_te, y_te, P0)                      # identity transport, phi_0 protos
    Ts = fit_transports(Fa_fit, F0_fit)
    print(f"{tag:>10} {'oracle':>11} {'-':>10} {oracle:>16.4f}   <- best on adapted feats")
    for name, Psi in Ts.items():
        Zte = Psi(Fa_te)                               # transported test feats (phi_0-normalized)
        recon = float((Zte * un(F0_te)).sum(1).mean()) # cosine to TRUE old features (held out)
        acc = ncm(Zte, y_te, P0)
        print(f"{tag:>10} {name:>11} {recon:>10.4f} {acc:>16.4f}")
        results[(tag, name)] = (recon, acc)
    print(f"{tag:>10} {'stale=id':>11} {'-':>10} {stale:>16.4f}   (sanity == identity above)")
    print("-" * 78)

np.savez("crux_transport.npz", frozen=frozen_acc,
         **{f"{t}|{n}": np.array(v) for (t, n), v in results.items()})
print("\nRead: does any Psi's NCM reach frozen? if best-Psi << frozen => irreducible collapse;")
print("if orthogonal already ~frozen => pure gauge; if mlp>>linear => info there but nonlinear.")
