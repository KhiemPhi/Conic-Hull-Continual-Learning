"""
crux_forward.py — FORWARD prototype transport:  phi_0  --V_t-->  phi_t

THE metric:  take an OLD feature phi_0(x), push it through the learned map V_t into the
CURRENT space, and compare against the TRUE current feature phi_t(x) (image -> new
network -> new feature).  We want this error LOW.

    fwd_err = 1 - cos( V_t(phi_0(x)), phi_t(x) )     on HELD-OUT old test images
              (fwd_err = 0  <=>  transported old feature == what the new net would produce)

Method under test — "anchor at phi_0" prototype maintenance:
  * phi_0 (frozen pretrained backbone) is ALWAYS reproducible, so use it as the anchor frame.
  * store each class prototype ONCE in phi_0:            mu0_c
  * at task t fit ONE map V_t: phi_0 -> phi_t
  * transport ALL prototypes in one shot:                 mu_c^(t) = V_t(mu0_c)
  * classify current queries phi_t(x) against them
  => a single d x d matrix updates every old class, including classes with ZERO exemplars,
     and there is no map composition (no error compounding).

Ablation — WHICH images are used to fit V_t (matched budget, same count):
  replay  : 20/class exemplars of OLD classes        (needs a buffer)
  curtask : the CURRENT task's own training images   (REPLAY-FREE)
  generic : CIFAR-10 images, zero CIFAR-100          (REPLAY-FREE + task-agnostic)

Map classes : orthogonal (Procrustes) | linear (ridge, lambda picked on a held-out fit split)
              | mlp (nonlinearity check; note Psi(mean) != mean(Psi) so it is approximate
                     for prototype transport -- linear is exact in that respect)
Baselines   : stale (no transport) | oracle (refit from current features) | frozen (never adapt)
"""
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
N_TASKS, CPT, EPOCHS, LR, BS = 10, 10, 4, 1e-4, 128
N_POOL = 128        # train imgs/class in the feature pool (prototypes + fit sources)
M_REPLAY = 20       # exemplars/class in the replay buffer
N_FIT = 1000        # matched fit budget per source
N_GENERIC = 2000    # CIFAR-10 pool size
USE_MLP = True

rng = np.random.default_rng(SEED)
ORDER = rng.permutation(100)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)
TRAIN = datasets.CIFAR100("./data", train=True,  download=False, transform=TF)
TEST  = datasets.CIFAR100("./data", train=False, download=False, transform=TF)
GEN   = datasets.CIFAR10 ("./data", train=True,  download=False, transform=TF)
TR_Y, TE_Y = np.array(TRAIN.targets), np.array(TEST.targets)

POOL_IDX = np.concatenate([np.where(TR_Y == c)[0][:N_POOL] for c in range(100)])
POOL_Y   = TR_Y[POOL_IDX]
TEST_IDX = np.arange(len(TE_Y))
GEN_IDX  = rng.choice(len(GEN), N_GENERIC, replace=False)
# replay = first M_REPLAY of each class's pool block  (aligned by construction)
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
def protos(Fp, y, classes): return un(np.stack([un(Fp[y == c]).mean(0) for c in classes]))


def acc(Fq, yq, P, classes):
    classes = np.asarray(classes)
    return float((classes[np.argmax(un(Fq) @ P.T, axis=1)] == yq).mean())


# ---------------- transport map classes  (fit X=phi_0 -> Y=phi_t) ----------------
def fit_orthogonal(X, Y):
    U, _, Vt = np.linalg.svd(un(X).T @ un(Y))
    R = U @ Vt
    return lambda Z, R=R: un(un(Z) @ R)


def fit_linear(X, Y):
    """Ridge phi_0 -> phi_t; lambda chosen on a held-out 20% split of the FIT data."""
    X, Y = un(X), un(Y)
    n, d = X.shape
    ntr = max(int(0.8 * n), d // 8 + 1)
    perm = np.random.default_rng(0).permutation(n)
    tr, va = perm[:ntr], perm[ntr:]
    s = np.trace(X[tr].T @ X[tr]) / d
    best, bestW = -2.0, None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        W = np.linalg.solve(X[tr].T @ X[tr] + lam * s * np.eye(d), X[tr].T @ Y[tr])
        v = float((un(X[va] @ W) * Y[va]).sum(1).mean()) if len(va) else 0.0
        if v > best:
            best, bestlam = v, lam
    W = np.linalg.solve(X.T @ X + bestlam * s * np.eye(d), X.T @ Y)
    return lambda Z, W=W: un(un(Z) @ W)


def fit_mlp(X, Y, epochs=200):
    X, Y = un(X), un(Y)
    d = X.shape[1]
    net = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, d)).to(DEV)
    Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
    Yt = torch.tensor(Y, device=DEV, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = (1 - (F.normalize(net(Xt[b]), dim=1) * Yt[b]).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def f(Z, net=net):
        with torch.no_grad():
            P = F.normalize(net(torch.tensor(un(Z), device=DEV, dtype=torch.float32)), dim=1)
        return P.cpu().numpy()
    return f


FITTERS = {"orthogonal": fit_orthogonal, "linear": fit_linear}
if USE_MLP:
    FITTERS["mlp"] = fit_mlp

# ============================ phi_0 (anchor frame) ============================
print("=== phi_0: frozen pretrained backbone (anchor frame, always reproducible) ===")
phi0 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
P0_pool = extract(phi0, TRAIN, POOL_IDX)     # (12800, 768)
P0_test = extract(phi0, TEST,  TEST_IDX)     # (10000, 768)
P0_gen  = extract(phi0, GEN,   GEN_IDX)      # (2000, 768)
MU0 = protos(P0_pool, POOL_Y, np.arange(100))         # anchor prototypes, phi_0 frame
del phi0; torch.cuda.empty_cache()

COHORT = np.asarray(TASKS[0])                          # fixed old cohort (task 0)
coh_te = np.where(np.isin(TE_Y, COHORT))[0]            # held-out old test rows

# ============================ sequential adaptation ============================
model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
lora_params = list(get_lora_params(model))
rows = []

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
    seen_pool = np.where(np.isin(POOL_Y, seen))[0]
    seen_te   = np.where(np.isin(TE_Y,  seen))[0]

    # current-frame features
    Pt_pool = extract(model, TRAIN, POOL_IDX[seen_pool])
    Pt_test = extract(model, TEST,  TEST_IDX[seen_te])
    Pt_gen  = extract(model, GEN,   GEN_IDX)
    pool_y_seen = POOL_Y[seen_pool]
    te_y_seen   = TE_Y[seen_te]

    # ---- fit sources (matched budget), each as (phi_0 rows, phi_t rows) ----
    old = np.setdiff1d(seen, cls)                       # classes seen BEFORE this task
    src = {}
    if len(old):
        m = np.where(np.isin(pool_y_seen, old) & REPLAY_MASK[seen_pool])[0]
        m = m[:N_FIT]
        src["replay"] = (P0_pool[seen_pool][m], Pt_pool[m])
    m = np.where(np.isin(pool_y_seen, cls))[0][:N_FIT]
    src["curtask"] = (P0_pool[seen_pool][m], Pt_pool[m])
    src["generic"] = (P0_gen[:N_FIT], Pt_gen[:N_FIT])

    # ---- baselines ----
    oracle_P = protos(Pt_pool, pool_y_seen, seen)
    a_oracle = acc(Pt_test, te_y_seen, oracle_P, seen)
    a_stale  = acc(Pt_test, te_y_seen, MU0[seen], seen)               # no transport
    a_frozen = acc(P0_test[seen_te], te_y_seen, MU0[seen], seen)      # never adapt
    # held-out old-test fidelity target: what the NEW net actually produces
    Pt_coh = Pt_test[np.isin(te_y_seen, COHORT)]
    P0_coh = P0_test[coh_te]

    for sname, (Xf, Yf) in src.items():
        for mname, fitter in FITTERS.items():
            V = fitter(Xf, Yf)
            fwd_err = 1.0 - float((V(P0_coh) * un(Pt_coh)).sum(1).mean())   # <-- THE metric
            Pm = un(V(MU0[seen]))                                            # transported protos
            a_tr = acc(Pt_test, te_y_seen, Pm, seen)
            rows.append(dict(t=t, seen=len(seen), source=sname, map=mname, n_fit=len(Xf),
                             fwd_err=fwd_err, acc=a_tr, oracle=a_oracle,
                             stale=a_stale, frozen=a_frozen))
    # identity reference (no map): how far apart are phi_0 and phi_t to begin with
    id_err = 1.0 - float((un(P0_coh) * un(Pt_coh)).sum(1).mean())
    rows.append(dict(t=t, seen=len(seen), source="-", map="identity", n_fit=0,
                     fwd_err=id_err, acc=a_stale, oracle=a_oracle,
                     stale=a_stale, frozen=a_frozen))

    cur = [r for r in rows if r["t"] == t]
    print(f"\n[t={t}] seen={len(seen)}  frozen {a_frozen:.3f} | stale {a_stale:.3f} | "
          f"oracle {a_oracle:.3f}")
    print(f"    {'source':>8} {'map':>11} {'n_fit':>6} {'fwd_err':>9} {'acc':>7} {'vs oracle':>10}")
    for r in cur:
        print(f"    {r['source']:>8} {r['map']:>11} {r['n_fit']:>6} {r['fwd_err']:>9.4f} "
              f"{r['acc']:>7.4f} {r['acc']-r['oracle']:>+10.4f}")

np.save("crux_forward_rows.npy", np.array(rows, dtype=object), allow_pickle=True)

# ============================ summary ============================
print("\n" + "=" * 84)
print("FORWARD TRANSPORT FIDELITY  fwd_err = 1 - cos(V(phi_0(x)), phi_t(x))   [LOWER=BETTER]")
print("=" * 84)
print(f"{'source':>8} {'map':>11} | " + " ".join(f"t{t}" .rjust(7) for t in range(N_TASKS)))
for sname in ["-", "replay", "curtask", "generic"]:
    for mname in ["identity"] + list(FITTERS):
        by_t = {r["t"]: r for r in rows if r["source"] == sname and r["map"] == mname}
        if not by_t:
            continue
        # pad missing steps (e.g. 'replay' has no t=0) so columns stay aligned
        line = " ".join(f"{by_t[t]['fwd_err']:7.4f}" if t in by_t else " " * 7
                        for t in range(N_TASKS))
        print(f"{sname:>8} {mname:>11} | {line}")

print("\n" + "=" * 84)
print("SEEN-WAY ACCURACY with transported prototypes   [oracle/frozen for reference]")
print("=" * 84)
print(f"{'source':>8} {'map':>11} | " + " ".join(f"t{t}".rjust(7) for t in range(N_TASKS)))
for nm, key in [("baseline", "frozen"), ("baseline", "oracle"), ("baseline", "stale")]:
    rs = sorted([r for r in rows if r["map"] == "identity"], key=lambda r: r["t"])
    print(f"{nm:>8} {key:>11} | " + " ".join(f"{r[key]:7.4f}" for r in rs))
for sname in ["replay", "curtask", "generic"]:
    for mname in FITTERS:
        by_t = {r["t"]: r for r in rows if r["source"] == sname and r["map"] == mname}
        if by_t:
            print(f"{sname:>8} {mname:>11} | " + " ".join(
                f"{by_t[t]['acc']:7.4f}" if t in by_t else " " * 7 for t in range(N_TASKS)))
print("=" * 84)
