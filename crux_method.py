"""
crux_method.py — the proposed CIL method vs the REAL RanPAC bar, on Split-ImageNet-R.

Proposed algorithm (per task t):
  1. snapshot  G_ref = cosine-Gram of the replay buffer under the current backbone
  2. adapt LoRA with   CE(task t) + lam * ||Gram(phi(B)) - G_ref||^2
     (relational constraint: PERMITS the harmless global rotation, FORBIDS the
      non-rigid distortion that destroys old-class features)
  3. grow replay buffer with M exemplars/class
  4. RECOMPUTE RanPAC statistics in the new frame from (task-t data + buffer),
     optionally inflated with synthetic features ~ N(mu_c, Sigma_shared) to restore
     rank and class balance   [do NOT accumulate: the frame moved, old stats are stale]
  5. solve W = (G + ridge)^-1 C ; predict argmax over seen classes

Variants
  A        frozen phi_0, incremental RanPAC                       -> the floor
  Aplus    adapt on task 0 only then FREEZE, incremental RanPAC   -> the REAL RanPAC bar
                                                                     (RanPAC's first-session PETL)
  B1       adapt every task (CE), accumulate stats in the then-current frame (stale)
  B2       adapt every task (CE), recompute from replay          [+/- synthetic]
  B3       adapt every task (Gram loss), recompute from replay   [+/- synthetic]
  C        joint upper bound = 0.8355 (already measured by crux_headroom.py)

B1 vs B2 isolates whether FRAME STALENESS or REPLAY-LIMITED STATISTICS costs more.
B3 vs B2 isolates the value of the relational loss.

Run:  python -u crux_method.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT = 10, 20            # Split-ImageNet-R: 200 classes / 10 tasks
EPOCHS, LR, BS = 10, 1e-4, 128   # matches crux_headroom total gradient steps
M_REPLAY = 20                    # exemplars/class
LAM_GRAM = float(os.environ.get("LAM_GRAM", 50))
M_RP = 10000                     # same as the headroom run, for comparability
LAMBDAS = [1e2, 1e3, 1e4]
SYNTH_PER_CLASS = 120            # synthetic feats/class to restore rank + balance
GB = 128

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)


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


from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"]); _cl = np.array(sorted(set(_w.tolist())))
_lab = np.searchsorted(_cl, _w)
_perm = np.random.default_rng(1993).permutation(len(_lab))
_ntr = int(0.8 * len(_lab))
TRAIN = HFWrap(_ds, _perm[:_ntr], _lab[_perm[:_ntr]]); TR_Y = _lab[_perm[:_ntr]]
TEST  = HFWrap(_ds, _perm[_ntr:], _lab[_perm[_ntr:]]); TE_Y = _lab[_perm[_ntr:]]
N_CLS = len(_cl)
print(f"[ImageNet-R] train {len(TR_Y)} test {len(TE_Y)} classes {N_CLS}")
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


P_RP = torch.randn(768, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)


def _H(Z, bs=4096):
    for i in range(0, len(Z), bs):
        yield i, torch.relu(torch.tensor(un(Z[i:i + bs]), device=DEV,
                                         dtype=torch.float32) @ P_RP)


def build_GC(Z, y):
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    for i, h in _H(Z):
        h = h.double()
        Y = torch.zeros(h.shape[0], N_CLS, device=DEV, dtype=torch.float64)
        Y[torch.arange(h.shape[0]), torch.tensor(y[i:i + h.shape[0]], device=DEV)] = 1.0
        G += h.T @ h; C += h.T @ Y
    return G, C


def solve_eval(G, C, Zval, yval, Zte, yte, seen):
    seen_t = torch.tensor(np.asarray(seen), device=DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def acc(W, Z, y):
        pr = []
        for _, h in _H(Z):
            pr.append(seen_t[(h.double() @ W)[:, seen_t].argmax(1)].cpu().numpy())
        return float((np.concatenate(pr) == y).mean())

    best, bestW = -1.0, None
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, C)
        a = acc(W, Zval, yval)
        if a > best:
            best, bestW = a, W
    return acc(bestW, Zte, yte)


def synth(Z, y, n_per):
    """Class-balanced synthetic features: per-class mean + SHARED covariance
    (a full per-class Sigma is unestimable from M_REPLAY samples in 768-d)."""
    Zn = un(Z)
    cls = np.unique(y)
    R = np.concatenate([Zn[y == c] - Zn[y == c].mean(0, keepdims=True) for c in cls])
    Sig = (R.T @ R) / max(len(R) - len(cls), 1) + 1e-4 * np.eye(Zn.shape[1])
    L = np.linalg.cholesky(Sig)
    rng = np.random.default_rng(0)
    out, lab = [], []
    for c in cls:
        mu = Zn[y == c].mean(0)
        out.append(mu + rng.standard_normal((n_per, Zn.shape[1])) @ L.T)
        lab.append(np.full(n_per, c))
    return np.concatenate(out), np.concatenate(lab)


def adapt_task(model, lora_params, cls, X_ref_idx, G_ref, lam):
    """One task of LoRA training, with optional Gram-relational loss on the buffer."""
    remap = {int(c): i for i, c in enumerate(cls)}
    idx = np.where(np.isin(TR_Y, cls))[0]
    loader = DataLoader(Subset(TRAIN, idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
    head = nn.Linear(768, len(cls)).to(DEV)
    opt = torch.optim.AdamW(lora_params + list(head.parameters()), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    Xb = None
    if lam > 0 and X_ref_idx is not None and len(X_ref_idx) > 0 and G_ref is not None:
        Xb = torch.cat([x for x, _ in DataLoader(Subset(TRAIN, X_ref_idx.tolist()),
                                                 batch_size=256, num_workers=8)], 0)
    for ep in range(EPOCHS):
        model.train(); ok = tot = 0
        for x, lab in loader:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(model(x).float()); loss = ce(lg, y)
            if Xb is not None:
                r = torch.randint(0, Xb.shape[0], (min(GB, Xb.shape[0]),))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    A = Fn.normalize(model(Xb[r].to(DEV)).float(), dim=1)
                loss = loss + lam * Fn.mse_loss(A @ A.T, G_ref[r][:, r])
            opt.zero_grad(); loss.backward(); opt.step()
            ok += int((lg.argmax(1) == y).sum()); tot += len(y)
        sch.step()
    return ok / tot


@torch.no_grad()
def gram_of(model, idx):
    Z = torch.tensor(un(extract(model, TRAIN, idx)), device=DEV, dtype=torch.float32)
    return Z @ Z.T


results = {}
# ONLY=B3 skips the already-measured A / A+ / B2 (saves ~40 min).
ONLY = set(x for x in os.environ.get("ONLY", "").split(",") if x)
def want(k): return (not ONLY) or (k in ONLY)

# ---------------- A: frozen phi_0, incremental RanPAC ----------------
if want("A") or want("Aplus"):
    print("\n=== A: frozen phi_0 (incremental RanPAC) ===")
    phi0 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
    F0_tr = extract(phi0, TRAIN, np.arange(len(TR_Y)))
    F0_te = extract(phi0, TEST, np.arange(len(TE_Y)))
    del phi0; torch.cuda.empty_cache()


def run_incremental(Ftr, Fte, tag):
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    vZ, vy, accs = [], [], []
    for t in range(N_TASKS):
        cls = TASKS[t]
        idx = np.where(np.isin(TR_Y, cls))[0]
        pm = np.random.default_rng(t).permutation(len(idx))
        nv = max(int(0.1 * len(idx)), 1)
        vi, ti = idx[pm[:nv]], idx[pm[nv:]]
        g, c = build_GC(Ftr[ti], TR_Y[ti]); G += g; C += c
        vZ.append(Ftr[vi]); vy.append(TR_Y[vi])
        seen = np.concatenate(TASKS[:t + 1])
        te = np.where(np.isin(TE_Y, seen))[0]
        a = solve_eval(G, C, np.concatenate(vZ), np.concatenate(vy),
                       Fte[te], TE_Y[te], seen)
        accs.append(a); print(f"  [{tag} t={t}] seen={len(seen):3d} acc {a:.4f}")
    return accs


if want("A"):
    results["A_frozen"] = run_incremental(F0_tr, F0_te, "A")

# ---------------- A+: first-session adaptation then freeze ----------------
if want("Aplus"):
    print("\n=== A+: adapt task 0 only, then FREEZE (the real RanPAC bar) ===")
    m = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(m)
    adapt_task(m, list(get_lora_params(m)), TASKS[0], None, None, 0.0)
    F1_tr = extract(m, TRAIN, np.arange(len(TR_Y)))
    F1_te = extract(m, TEST, np.arange(len(TE_Y)))
    del m; torch.cuda.empty_cache()
    results["Aplus_firstsession"] = run_incremental(F1_tr, F1_te, "A+")

# ---------------- B: sequential adaptation ----------------
for tag, name, lam in [("B2", "seqCE", 0.0), ("B3", "seqGram", LAM_GRAM)]:
    if not want(tag):
        continue
    print(f"\n=== {tag} (lam_gram={lam:g}) ===")
    m = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(m)
    lp = list(get_lora_params(m))
    buf = np.array([], dtype=int)
    Gs = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)   # B1 stale accumulate
    Cs = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    a_stale, a_rec, a_syn = [], [], []
    for t in range(N_TASKS):
        cls = TASKS[t]
        G_ref = gram_of(m, buf) if (lam > 0 and len(buf)) else None
        tr_acc = adapt_task(m, lp, cls, buf if lam > 0 else None, G_ref, lam)
        buf = np.concatenate([buf, np.concatenate(
            [np.where(TR_Y == c)[0][:M_REPLAY] for c in cls])]).astype(int)

        seen = np.concatenate(TASKS[:t + 1])
        idx_task = np.where(np.isin(TR_Y, cls))[0]
        te = np.where(np.isin(TE_Y, seen))[0]
        Z_task = extract(m, TRAIN, idx_task)
        Z_buf = extract(m, TRAIN, buf)
        Z_te = extract(m, TEST, te)

        # --- B1: stale accumulation (full task data, then-current frame) ---
        g, c = build_GC(Z_task, TR_Y[idx_task]); Gs += g; Cs += c
        pm = np.random.default_rng(t).permutation(len(idx_task))
        nv = max(int(0.1 * len(idx_task)), 1)
        a1 = solve_eval(Gs, Cs, Z_task[pm[:nv]], TR_Y[idx_task][pm[:nv]],
                        Z_te, TE_Y[te], seen)

        # --- B2/B3: recompute from (task data + buffer), +/- synthetic ---
        Zr = np.concatenate([Z_task, Z_buf]); yr = np.concatenate([TR_Y[idx_task], TR_Y[buf]])
        pm = np.random.default_rng(t + 99).permutation(len(Zr))
        nv = max(int(0.1 * len(Zr)), 1)
        vi, ti = pm[:nv], pm[nv:]
        G2, C2 = build_GC(Zr[ti], yr[ti])
        a2 = solve_eval(G2, C2, Zr[vi], yr[vi], Z_te, TE_Y[te], seen)
        Zs, ys = synth(Zr[ti], yr[ti], SYNTH_PER_CLASS)
        G3, C3 = build_GC(np.concatenate([Zr[ti], Zs]), np.concatenate([yr[ti], ys]))
        a3 = solve_eval(G3, C3, Zr[vi], yr[vi], Z_te, TE_Y[te], seen)

        a_stale.append(a1); a_rec.append(a2); a_syn.append(a3)
        print(f"  [{tag} t={t}] seen={len(seen):3d} trainacc {tr_acc:.3f} | "
              f"stale {a1:.4f} | recompute {a2:.4f} | +synth {a3:.4f}")
    results[f"{name}_stale"] = a_stale
    results[f"{name}_recompute"] = a_rec
    results[f"{name}_recompute+synth"] = a_syn
    del m; torch.cuda.empty_cache()

# ---------------- merge with previously-measured runs ----------------
# Results measured in the first (crashed-at-B3) run, so ONLY= re-runs still show a full table.
KNOWN = {
    "A_frozen": [0.8653, 0.8423, 0.8116, 0.7961, 0.7744, 0.7475, 0.7620, 0.7461,
                 0.7349, 0.7272],
    "Aplus_firstsession": [0.9050, 0.8784, 0.8638, 0.8477, 0.8267, 0.8134, 0.8076,
                           0.7955, 0.7889, 0.7858],
    "seqCE_stale": [0.8752, 0.7937, 0.6362, 0.5637, 0.5561, 0.4786, 0.4263, 0.3296,
                    0.3096, 0.2617],
    "seqCE_recompute": [0.8891, 0.8261, 0.7687, 0.7391, 0.7343, 0.6901, 0.6997,
                        0.6593, 0.6544, 0.6237],
    "seqCE_recompute+synth": [0.8772, 0.8477, 0.8010, 0.7662, 0.7671, 0.7274, 0.7244,
                              0.6921, 0.6800, 0.6557],
}
OUT = "crux_method_imagenetr.npy"
merged = dict(KNOWN)
if os.path.exists(OUT):                       # anything from earlier invocations
    try:
        merged.update(np.load(OUT, allow_pickle=True).item())
    except Exception as e:
        print(f"[warn] could not read {OUT}: {e}")
merged.update(results)                        # this run wins
np.save(OUT, merged, allow_pickle=True)

JOINT_CEILING = 0.8355                        # crux_headroom.py, LoRA
BAR = "Aplus_firstsession"
ORDER = ["A_frozen", "Aplus_firstsession",
         "seqCE_stale", "seqCE_recompute", "seqCE_recompute+synth",
         "seqGram_stale", "seqGram_recompute", "seqGram_recompute+synth"]
keys = [k for k in ORDER if k in merged] + [k for k in merged if k not in ORDER]

bl = merged[BAR][-1] if BAR in merged else None
ba = float(np.mean(merged[BAR])) if BAR in merged else None

print("\n" + "=" * 96)
print("SPLIT-IMAGENET-R (200 cls / 10 tasks)   A-last = accuracy after final task,"
      "   A-avg = mean over tasks")
print("=" * 96)
print(f"{'method':>26} {'A-last':>9} {'A-avg':>9} {'dA-last':>10} {'dA-avg':>9}   note")
for k in keys:
    v = merged[k]
    last, avg = v[-1], float(np.mean(v))
    d1 = f"{last-bl:>+10.4f}" if bl is not None else f"{'-':>10}"
    d2 = f"{avg-ba:>+9.4f}" if ba is not None else f"{'-':>9}"
    note = "<- BAR (real RanPAC)" if k == BAR else ("floor" if k == "A_frozen" else "")
    print(f"{k:>26} {last:>9.4f} {avg:>9.4f} {d1} {d2}   {note}")
print("-" * 96)
print(f"{'C_joint (upper bound)':>26} {JOINT_CEILING:>9.4f} {'-':>9} "
      f"{JOINT_CEILING-bl:>+10.4f} {'-':>9}   ceiling, not a CIL method"
      if bl is not None else "")
print(f"\nWIN CONDITION: seqGram_recompute+synth  A-last > {bl if bl else 'BAR'}"
      f"  (and A-avg > {ba:.4f})" if bl else "")
print("deltas are vs the BAR (RanPAC with first-session PETL), not vs the frozen floor.")
print("=" * 96)
