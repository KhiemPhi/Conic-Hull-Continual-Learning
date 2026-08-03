"""
crux_perstage.py — the per-stage-frame design (user's proposal), done properly.

Idea under test
---------------
Do NOT force every class into one frame. Keep each stage's classifier frozen in the frame
it was BORN in, and at test time transport the query INTO each stage's frame, so every
comparison happens within a single consistent frame.

  stage s: after training task s the backbone is phi_{s+1} = the stage's BIRTH FRAME.
           store  (a) prototypes of stage-s classes in phi_{s+1}
                  (b) M replay exemplars/class  +  their phi_{s+1} features   <- gives
                      (phi_now, phi_birth) correspondence pairs later, IN-REGION.

  test at task t (backbone phi_{t+1}):
           for each stage s <= t:
               fit  Psi_{t->s}: phi_{t+1} -> phi_{s+1}   on stage-s replay pairs (in-region)
               z_s = Psi_{t->s}(phi_{t+1}(x));  score z_s against stage-s prototypes
           combine across stages.

This avoids the enrollment/query mismatch that killed the single-phi_0-frame variant
(crux_ranpac B): here a class is ALWAYS queried in the frame it was enrolled in.

Combination rules compared
--------------------------
  raw      : argmax of raw cosine over all (stage, class)          [no calibration]
  cal      : per-stage z-scored cosines, then argmax               [soft, calibrated]
  cascade  : walk stages oldest->newest, take the first whose calibrated max exceeds
             tau (tau = percentile of that stage's own in-stage replay scores) [hard gate]

References
----------
  frozen   : NCM on phi_0, prototypes from phi_0            (no adaptation at all)
  stale    : birth prototypes, query NOT transported        (the naive failure)
  oracle   : all prototypes refit in the CURRENT frame      (single-frame upper bound)
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
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT, EPOCHS, LR, BS = 10, 10, 4, 1e-4, 128
M_REPLAY = 20                 # exemplars/class kept per stage (fits the transport maps)
N_PROTO = 128                 # train imgs/class used to build prototypes
CASCADE_PCT = 25.0            # tau = this percentile of in-stage replay scores

rng = np.random.default_rng(SEED)
ORDER = rng.permutation(100)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)
TRAIN = datasets.CIFAR100("./data", train=True,  download=False, transform=TF)
TEST  = datasets.CIFAR100("./data", train=False, download=False, transform=TF)
TR_Y, TE_Y = np.array(TRAIN.targets), np.array(TEST.targets)


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


def fit_linear(X, Y, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Ridge X->Y, lambda picked on a held-out 20% of the fit pairs."""
    X, Y = un(X), un(Y)
    n, d = X.shape
    perm = np.random.default_rng(0).permutation(n)
    ntr = max(int(0.8 * n), 1)
    tr, va = perm[:ntr], perm[ntr:]
    s = np.trace(X[tr].T @ X[tr]) / d
    best, bestlam = -2.0, 1e-1
    for lam in grid:
        W = np.linalg.solve(X[tr].T @ X[tr] + lam * s * np.eye(d), X[tr].T @ Y[tr])
        v = float((un(X[va] @ W) * Y[va]).sum(1).mean()) if len(va) else 0.0
        if v > best:
            best, bestlam = v, lam
    return np.linalg.solve(X.T @ X + bestlam * s * np.eye(d), X.T @ Y)


# ---------------- phi_0 reference ----------------
print("=== phi_0 (frozen reference) ===")
phi0 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
PROTO_IDX = {c: np.where(TR_Y == c)[0][:N_PROTO] for c in range(100)}
F0_proto = {c: extract(phi0, TRAIN, PROTO_IDX[c]) for c in range(100)}
MU0 = {c: un(un(F0_proto[c]).mean(0, keepdims=True))[0] for c in range(100)}
F0_te = extract(phi0, TEST, np.arange(len(TE_Y)))
del phi0; torch.cuda.empty_cache()

model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
lora_params = list(get_lora_params(model))

stage = []      # per-stage store: classes, prototypes (birth frame), replay idx + birth feats
rows = []

for t in range(N_TASKS):
    cls = np.asarray(TASKS[t])
    remap = {int(c): i for i, c in enumerate(cls)}
    tr_idx = np.concatenate([np.where(TR_Y == c)[0] for c in cls])
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
    # ---- register stage t in its BIRTH frame phi_{t+1} ----
    rep_idx = np.concatenate([np.where(TR_Y == c)[0][:M_REPLAY] for c in cls])
    P = np.stack([un(un(extract(model, TRAIN, PROTO_IDX[c])).mean(0, keepdims=True))[0]
                  for c in cls])
    stage.append(dict(cls=cls, proto=P, rep_idx=rep_idx,
                      rep_birth=extract(model, TRAIN, rep_idx)))

    # ---- evaluate over all seen ----
    seen = np.concatenate(TASKS[:t + 1])
    te_idx = np.where(np.isin(TE_Y, seen))[0]
    y_te = TE_Y[te_idx]
    Fte_now = extract(model, TEST, te_idx)

    raw_blocks, cal_blocks, cls_blocks, gates = [], [], [], []
    fids = []
    for s in range(t + 1):
        st = stage[s]
        if s == t:                                   # native frame, no transport
            Z = un(Fte_now)
            cal_src = un(st["rep_birth"])
        else:
            rep_now = extract(model, TRAIN, st["rep_idx"])          # phi_{t+1}(replay_s)
            W = fit_linear(rep_now, st["rep_birth"])                # -> phi_{s+1}
            Z = un(un(Fte_now) @ W)
            cal_src = un(un(rep_now) @ W)
            fids.append(float((cal_src * un(st["rep_birth"])).sum(1).mean()))
        sc = Z @ st["proto"].T                                       # (n_te, |cls_s|)
        ref = cal_src @ st["proto"].T                                # in-stage score sample
        mu, sd = float(ref.mean()), float(ref.std() + 1e-8)
        raw_blocks.append(sc); cal_blocks.append((sc - mu) / sd)
        cls_blocks.append(st["cls"])
        gates.append(np.percentile(ref.max(1), CASCADE_PCT))

    allcls = np.concatenate(cls_blocks)
    RAW = np.concatenate(raw_blocks, 1); CAL = np.concatenate(cal_blocks, 1)
    a_raw = float((allcls[RAW.argmax(1)] == y_te).mean())
    a_cal = float((allcls[CAL.argmax(1)] == y_te).mean())

    # hard cascade: oldest -> newest, first stage clearing its own gate
    pred = np.full(len(y_te), -1)
    for s in range(t + 1):
        m = raw_blocks[s].max(1) >= gates[s]
        take = m & (pred < 0)
        pred[take] = cls_blocks[s][raw_blocks[s].argmax(1)][take]
    fall = pred < 0
    if fall.any():                                   # nothing fired -> best calibrated
        pred[fall] = allcls[CAL[fall].argmax(1)]
    a_cas = float((pred == y_te).mean())

    # ---- references ----
    mu_frozen = np.stack([MU0[int(c)] for c in seen])
    a_frozen = float((seen[np.argmax(un(F0_te[te_idx]) @ mu_frozen.T, 1)] == y_te).mean())
    mu_stale = np.concatenate([st["proto"] for st in stage[:t + 1]])
    a_stale = float((allcls[np.argmax(un(Fte_now) @ mu_stale.T, 1)] == y_te).mean())
    mu_orac = np.stack([un(un(extract(model, TRAIN, PROTO_IDX[int(c)])).mean(0, keepdims=True))[0]
                        for c in seen])
    a_orac = float((seen[np.argmax(un(Fte_now) @ mu_orac.T, 1)] == y_te).mean())

    rows.append(dict(t=t, seen=len(seen), raw=a_raw, cal=a_cal, cascade=a_cas,
                     frozen=a_frozen, stale=a_stale, oracle=a_orac,
                     fid=float(np.mean(fids)) if fids else 1.0))
    r = rows[-1]
    print(f"[t={t}] seen={len(seen):3d} | PER-STAGE raw {a_raw:.4f} cal {a_cal:.4f} "
          f"cascade {a_cas:.4f} | frozen {a_frozen:.4f} stale {a_stale:.4f} "
          f"oracle {a_orac:.4f} | map-fid {r['fid']:.3f}")

np.save("crux_perstage.npy", np.array(rows, dtype=object), allow_pickle=True)
print("\n" + "=" * 96)
print("PER-STAGE FRAMES (query transported into each stage's birth frame, replay-fit maps)")
print("=" * 96)
print(f"{'t':>2} {'seen':>5} {'raw':>8} {'cal':>8} {'cascade':>8} | {'frozen':>8} "
      f"{'stale':>8} {'oracle':>8} | {'cal-orac':>9} {'mapfid':>7}")
for r in rows:
    print(f"{r['t']:>2} {r['seen']:>5} {r['raw']:>8.4f} {r['cal']:>8.4f} {r['cascade']:>8.4f} | "
          f"{r['frozen']:>8.4f} {r['stale']:>8.4f} {r['oracle']:>8.4f} | "
          f"{r['cal']-r['oracle']:>+9.4f} {r['fid']:>7.3f}")
print("-" * 96)
f = rows[-1]
print(f"FINAL: per-stage(cal) {f['cal']:.4f} vs single-frame oracle {f['oracle']:.4f} "
      f"({f['cal']-f['oracle']:+.4f}) vs frozen {f['frozen']:.4f}")
print("WIN for the per-stage idea if cal > oracle (beats the best single-frame option).")
print("Also check: cal > raw (calibration needed?), cal > cascade (soft > hard gate?).")
print("=" * 96)
