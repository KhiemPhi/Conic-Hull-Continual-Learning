"""
crux_recovered.py — how different are ORACLE features (image -> new net) from
RECOVERED features (old feature -> transport map -> new space)?

For each task t, fit the best map from crux_forward (replay-fit LINEAR) and compare, on
HELD-OUT test images of all seen classes:

    B = phi_t(x)          "oracle"     : image -> NEW network -> feature
    A = V_t(phi_0(x))     "recovered"  : OLD feature -> transport map -> new space

Diagnostics
-----------
1. per-sample agreement      cos(A_i, B_i)                     (what fwd_err measured)
2. PROTOTYPE agreement       cos(mean_c A, mean_c B) in DEGREES, vs the crowding budget
                             gamma/2 (min inter-class angle of oracle prototypes).
                             For linear V, mean_c A = V(mu0_c) exactly -> this IS the
                             prototype placement error that decides accuracy.
3. bias/variance split       e_i = A_i - B_i decomposed into a per-class systematic part
                             (mean_c e) and a random part. Random error averages out of a
                             prototype; systematic error does NOT.
4. structural similarity     linear CKA(A, B) and Gram correlation of prototypes
                             -> is the recovered space the SAME shape, just misplaced?
5. usability (4-way)         NCM accuracy for {oracle,recovered} features
                             x {oracle,recovered} prototypes.
                             recovered/recovered high but oracle/recovered low
                             => coherent-but-misaligned space (a frame error, not noise).
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = int(os.environ.get("SEED", 0))   # was hardcoded 0; seeds now env-driven
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT, EPOCHS, LR, BS = 10, 10, 4, 1e-4, 128
N_POOL, M_REPLAY = 128, 20

rng = np.random.default_rng(SEED)
ORDER = rng.permutation(100)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)
TRAIN = datasets.CIFAR100("./data", train=True,  download=False, transform=TF)
TEST  = datasets.CIFAR100("./data", train=False, download=False, transform=TF)
TR_Y, TE_Y = np.array(TRAIN.targets), np.array(TEST.targets)
POOL_IDX = np.concatenate([np.where(TR_Y == c)[0][:N_POOL] for c in range(100)])
POOL_Y = TR_Y[POOL_IDX]
REPLAY_MASK = np.concatenate([np.arange(N_POOL) < M_REPLAY for _ in range(100)])


@torch.no_grad()
def extract(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
def protos(F_, y, cs): return un(np.stack([un(F_[y == c]).mean(0) for c in cs]))
def deg(c): return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def acc(Fq, yq, P, cs):
    cs = np.asarray(cs)
    return float((cs[np.argmax(un(Fq) @ P.T, axis=1)] == yq).mean())


def fit_linear(X, Y):
    X, Y = un(X), un(Y)
    n, d = X.shape
    perm = np.random.default_rng(0).permutation(n)
    ntr = max(int(0.8 * n), 1)
    tr, va = perm[:ntr], perm[ntr:]
    s = np.trace(X[tr].T @ X[tr]) / d
    best, bestlam = -2.0, 1e-2
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        W = np.linalg.solve(X[tr].T @ X[tr] + lam * s * np.eye(d), X[tr].T @ Y[tr])
        v = float((un(X[va] @ W) * Y[va]).sum(1).mean()) if len(va) else 0.0
        if v > best:
            best, bestlam = v, lam
    return np.linalg.solve(X.T @ X + bestlam * s * np.eye(d), X.T @ Y)


def cka(X, Y):
    """Linear CKA between two feature sets on the SAME rows (centered)."""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12))


print("=== phi_0 anchor frame ===")
phi0 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
P0_pool = extract(phi0, TRAIN, POOL_IDX)
P0_test = extract(phi0, TEST,  np.arange(len(TE_Y)))
MU0 = protos(P0_pool, POOL_Y, np.arange(100))
del phi0; torch.cuda.empty_cache()

model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
lora_params = list(get_lora_params(model))
hist = []

for t in range(N_TASKS):
    cls = np.asarray(TASKS[t])
    remap = {c: i for i, c in enumerate(cls)}
    tr_idx = np.concatenate([np.where(TR_Y == c)[0] for c in cls])
    loader = DataLoader(Subset(TRAIN, tr_idx.tolist()), batch_size=BS, shuffle=True, num_workers=8)
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
    old = np.setdiff1d(seen, cls)
    if len(old) == 0:
        continue
    seen_pool = np.where(np.isin(POOL_Y, seen))[0]
    seen_te = np.where(np.isin(TE_Y, seen))[0]
    Pt_pool = extract(model, TRAIN, POOL_IDX[seen_pool])
    Pt_test = extract(model, TEST, seen_te)
    py, ty = POOL_Y[seen_pool], TE_Y[seen_te]

    # map fit on OLD-class replay exemplars (best source from crux_forward)
    m = np.where(np.isin(py, old) & REPLAY_MASK[seen_pool])[0]
    W = fit_linear(P0_pool[seen_pool][m], Pt_pool[m])

    B = un(Pt_test)                       # ORACLE  : image -> new net
    A = un(un(P0_test[seen_te]) @ W)      # RECOVERED: old feat -> map -> new space

    # 1. per-sample
    per_sample = float((A * B).sum(1).mean())
    # 2. prototype-level (linear V => mean_c A == V(mu0_c))
    PA = protos(A, ty, seen); PB = protos(B, ty, seen)
    proto_cos = float((PA * PB).sum(1).mean())
    G = PB @ PB.T
    gamma = deg(np.max(G[~np.eye(len(seen), dtype=bool)]))     # min inter-class angle
    # 3. bias / variance of the error
    E = A - B
    Ebar = np.stack([E[ty == c].mean(0) for c in seen])
    sysE = float(np.mean([np.sum(Ebar[i] ** 2) for i in range(len(seen))]))
    rndE = float(np.mean([np.mean(np.sum((E[ty == c] - Ebar[i]) ** 2, 1))
                          for i, c in enumerate(seen)]))
    frac_sys = sysE / (sysE + rndE + 1e-12)
    # 4. structure
    ck = cka(A, B)
    off = ~np.eye(len(seen), dtype=bool)
    gram_corr = float(np.corrcoef((PA @ PA.T)[off], G[off])[0, 1])
    # 5. usability 4-way
    a_oo = acc(B, ty, PB, seen); a_ro = acc(A, ty, PB, seen)
    a_or = acc(B, ty, PA, seen); a_rr = acc(A, ty, PA, seen)

    hist.append(dict(t=t, per_sample=per_sample, proto_cos=proto_cos,
                     proto_deg=deg(proto_cos), gamma=gamma, frac_sys=frac_sys,
                     cka=ck, gram_corr=gram_corr,
                     a_oo=a_oo, a_ro=a_ro, a_or=a_or, a_rr=a_rr))
    h = hist[-1]
    print(f"\n[t={t}] seen={len(seen)}  gamma={gamma:.1f}deg  (budget gamma/2={gamma/2:.1f}deg)")
    print(f"   per-sample cos {per_sample:.4f} ({deg(per_sample):5.1f}deg) | "
          f"PROTOTYPE cos {proto_cos:.4f} ({h['proto_deg']:5.1f}deg)  "
          f"{'OK' if h['proto_deg'] < gamma/2 else 'EXCEEDS BUDGET'}")
    print(f"   error is {100*frac_sys:.1f}% systematic / {100*(1-frac_sys):.1f}% random | "
          f"CKA {ck:.4f} | proto-Gram corr {gram_corr:.4f}")
    print(f"   acc  oracleF/oracleP {a_oo:.4f} | recovF/oracleP {a_ro:.4f} | "
          f"oracleF/recovP {a_or:.4f} | recovF/recovP {a_rr:.4f}")

np.save("crux_recovered_hist.npy", np.array(hist, dtype=object), allow_pickle=True)
print("\n" + "=" * 92)
print(f"{'t':>2} {'persampl':>9} {'protoDeg':>9} {'gam/2':>7} {'%sys':>6} {'CKA':>7} "
      f"{'gramCor':>8} {'or/or':>7} {'rec/or':>7} {'or/rec':>7} {'rec/rec':>7}")
for h in hist:
    print(f"{h['t']:>2} {deg(h['per_sample']):>8.1f}d {h['proto_deg']:>8.1f}d "
          f"{h['gamma']/2:>6.1f}d {100*h['frac_sys']:>5.1f}% {h['cka']:>7.4f} "
          f"{h['gram_corr']:>8.4f} {h['a_oo']:>7.4f} {h['a_ro']:>7.4f} "
          f"{h['a_or']:>7.4f} {h['a_rr']:>7.4f}")
print("=" * 92)
print("or/or = oracle feats+protos (ceiling) | rec/rec = recovered world self-consistent")
print("If rec/rec >> or/rec: recovered space is COHERENT but MISALIGNED (frame error).")
print("If proto_deg > gamma/2: prototypes land inside a neighbour's territory.")
