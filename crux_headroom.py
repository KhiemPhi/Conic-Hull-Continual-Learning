"""
crux_headroom.py — is there ANY headroom for adaptation over frozen features?

The only live question left. Strip away CIL entirely and ask the joint/offline version:

    does adapting the backbone produce BETTER features than the frozen PTM,
    as measured by the heads we actually care about (NCM and RanPAC's decorrelated ridge)?

    frozen phi_0        --->  NCM / RanPAC        (what RanPAC gets, zero forgetting, free)
    jointly adapted phi --->  NCM / RanPAC        (the ceiling adaptation could ever reach)

If adapted <= frozen there is NO headroom: no CIL adaptation scheme can beat RanPAC on this
benchmark, because even the offline upper bound doesn't. Stop.
If adapted >> frozen the headroom is real and it is worth engineering a CIL method that
captures it (simplest: small replay + recompute statistics in the current frame -- no transport).

Three feature sets, identical heads/protocol:
    frozen   : phi_0
    lora     : LoRA r32 fine-tuned jointly on ALL train classes
    fullft   : all parameters fine-tuned jointly (higher-capacity upper bound)

Run:  DATASET=IMAGENETR python -u crux_headroom.py
      DATASET=CIFAR100  python -u crux_headroom.py     # negative control: expect ~0 headroom
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
# Which backbone's CEILING are we measuring? This is the decisive number: no CIL method can
# exceed the joint/offline result, so it bounds every target.
#   augreg2_in21k_ft_in1k -> 0.8355 (measured; IN1k-finetuned, inflates the frozen baseline)
#   augreg_in21k          -> standard PTM-CIL backbone, NOT yet measured
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
DATASET = os.environ.get("DATASET", "IMAGENETR").upper()
BS = 128
M_RP = 10000
LAMBDAS = [1e-1, 1.0, 1e1, 1e2, 1e3]
VAL_FRAC = 0.10

# ----------------------------- RECIPE KNOBS -----------------------------------------
# The original measurement (ceiling 0.8332) trained with the EVAL transform: no crop, no flip,
# no augmentation of any kind, 10 epochs, on 24k images. That is almost certainly underfit, and
# the ceiling bounds every target, so it is worth probing before chasing a better CIL method.
#   AUG=0 EPOCHS=10                -> reproduces the original number
#   AUG=1 EPOCHS=40                -> the candidate recipe
#   SWEEP=1                        -> runs a grid of recipes (see RECIPES below)
EPOCHS = int(os.environ.get("EPOCHS", 10))
AUG = int(os.environ.get("AUG", 0))              # RandAugment + RRC + flip + random-erasing
LR = float(os.environ.get("LR", 1e-4))           # LoRA lr (fullft uses LR/10)
WARMUP = int(os.environ.get("WARMUP", 0))        # linear warmup epochs
LORA_R = int(os.environ.get("LORA_R", 32))
RRC_MIN = float(os.environ.get("RRC_MIN", 0.7))  # RandomResizedCrop lower scale
SWEEP = int(os.environ.get("SWEEP", 0))
MODES = [m for m in os.environ.get("MODES", "lora,fullft").split(",") if m]

_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
# EVAL transform: ALWAYS used for feature extraction. Features must be deterministic or the
# prototypes/Gram statistics are computed on random crops and the whole measurement is noise.
TF_EVAL = create_transform(**_cfg, is_training=False)


def make_train_tf(aug, rrc_min=RRC_MIN):
    if not aug:
        return TF_EVAL
    return create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                            re_prob=0.25, scale=(rrc_min, 1.0), hflip=0.5)


TF = TF_EVAL          # kept for any legacy reference; datasets below take an explicit tf


class HFWrap(Dataset):
    def __init__(self, ds, idx, labels, tf=None):
        self.ds, self.idx, self.labels = ds, np.asarray(idx), np.asarray(labels)
        self.tf = tf or TF_EVAL

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(self.labels[i])


def build_data(train_tf):
    """Returns TRAIN_AUG (for the training loop), TRAIN_EVAL (for feature extraction —
    ALWAYS deterministic), TEST, and labels."""
    if DATASET == "CIFAR100":
        tra = datasets.CIFAR100("./data", train=True, download=False, transform=train_tf)
        tre = datasets.CIFAR100("./data", train=True, download=False, transform=TF_EVAL)
        te = datasets.CIFAR100("./data", train=False, download=False, transform=TF_EVAL)
        return tra, tre, np.array(tre.targets), te, np.array(te.targets), 100
    if DATASET == "IMAGENETR":
        from datasets import load_dataset
        ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
        wnid = np.array(ds["wnid"])
        classes = np.array(sorted(set(wnid.tolist())))
        lab = np.searchsorted(classes, wnid)
        perm = np.random.default_rng(1993).permutation(len(lab))   # standard 80/20
        ntr = int(0.8 * len(lab))
        tr_i, te_i = perm[:ntr], perm[ntr:]
        return (HFWrap(ds, tr_i, lab[tr_i], train_tf),      # augmented, training only
                HFWrap(ds, tr_i, lab[tr_i], TF_EVAL),       # deterministic, extraction
                lab[tr_i],
                HFWrap(ds, te_i, lab[te_i], TF_EVAL), lab[te_i], len(classes))
    raise ValueError(DATASET)


TRAIN_AUG, TRAIN, TR_Y, TEST, TE_Y, N_CLS = build_data(make_train_tf(AUG))
print(f"[{DATASET}] train {len(TR_Y)}  test {len(TE_Y)}  classes {N_CLS}")


@torch.no_grad()
def extract(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def ncm(Ftr, ytr, Fte, yte):
    P = un(np.stack([un(Ftr[ytr == c]).mean(0) for c in range(N_CLS)]))
    return float((np.argmax(un(Fte) @ P.T, axis=1) == yte).mean())


def ranpac(Ftr, ytr, Fte, yte, seed=0):
    """RP -> ReLU -> decorrelated ridge; lambda picked on a held-out train split."""
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(Ftr.shape[1], M_RP, generator=g).to(DEV)

    def H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(
                torch.tensor(un(Z[i:i + bs]), device=DEV, dtype=torch.float32) @ P)

    n = len(Ftr)
    perm = np.random.default_rng(0).permutation(n)
    nval = int(VAL_FRAC * n)
    vi, ti = perm[:nval], perm[nval:]
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    Xt, yt = Ftr[ti], ytr[ti]
    for i, h in H(Xt):
        h = h.double()
        Y = torch.zeros(h.shape[0], N_CLS, device=DEV, dtype=torch.float64)
        Y[torch.arange(h.shape[0]), torch.tensor(yt[i:i + h.shape[0]], device=DEV)] = 1.0
        G += h.T @ h
        C += h.T @ Y

    def acc(W, Z, y):
        pred = []
        for _, h in H(Z):
            pred.append((h.double() @ W).argmax(1).cpu().numpy())
        return float((np.concatenate(pred) == y).mean())

    best, bestW, bestlam = -1.0, None, None
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * torch.eye(M_RP, device=DEV, dtype=torch.float64), C)
        a = acc(W, Ftr[vi], ytr[vi])
        if a > best:
            best, bestW, bestlam = a, W, lam
    return acc(bestW, Fte, yte), bestlam


def adapt(mode, train_ds, epochs=None, lr=None, warmup=None, lora_r=None):
    """Joint fine-tune on ALL classes. mode in {'lora','fullft'}."""
    epochs = EPOCHS if epochs is None else epochs
    lr = LR if lr is None else lr
    warmup = WARMUP if warmup is None else warmup
    lora_r = LORA_R if lora_r is None else lora_r
    if mode == "lora":
        m = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=lora_r, lora_alpha=4.0, lora_config="task_shared")
        freeze_non_lora(m)
        params = list(get_lora_params(m))
    else:
        m = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
        for p in m.parameters():
            p.requires_grad_(True)
        params, lr = list(m.parameters()), lr / 10.0     # full FT needs a much smaller lr
    head = nn.Linear(768, N_CLS).to(DEV)
    opt = torch.optim.AdamW(params + list(head.parameters()), lr=lr, weight_decay=1e-4)
    if warmup > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=warmup),
                  torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1))],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True)
    for ep in range(epochs):
        m.train(); ok = tot = 0
        for x, y in loader:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(m(x).float())
                loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ok += int((logits.argmax(1) == y).sum()); tot += len(y)
        sched.step()
        if ep % max(epochs // 10, 1) == 0 or ep == epochs - 1:
            print(f"    [{mode}] epoch {ep+1}/{epochs} train-acc {ok/tot:.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    return m


results = {}
print("\n=== frozen phi_0 ===")
phi0 = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
Ftr = extract(phi0, TRAIN); Fte = extract(phi0, TEST)
del phi0; torch.cuda.empty_cache()
a_ncm = ncm(Ftr, TR_Y, Fte, TE_Y)
a_rp, lam = ranpac(Ftr, TR_Y, Fte, TE_Y)
results["frozen"] = (a_ncm, a_rp)
print(f"  frozen : NCM {a_ncm:.4f} | RanPAC {a_rp:.4f} (lam={lam:g})")


def subsample(ds_aug, per_class=None, n_classes=None):
    """Restrict the TRAINING set (extraction sets are untouched) to isolate the two
    confounded factors in `joint > A_plus`: class DIVERSITY vs data VOLUME."""
    keep = np.arange(len(TR_Y))
    if n_classes:
        keep = keep[TR_Y[keep] < n_classes]
    if per_class:
        sel = []
        for c in np.unique(TR_Y[keep]):
            sel.append(keep[TR_Y[keep] == c][:per_class])
        keep = np.concatenate(sel)
    return Subset(ds_aug, keep.tolist()), len(keep)


def measure(tag, mode, aug, epochs, lr, warmup, lora_r, per_class=None, n_classes=None):
    """One recipe -> (NCM, RanPAC) of the jointly-adapted features."""
    print(f"\n=== {tag}  [{mode} aug={aug} ep={epochs} lr={lr:g} warm={warmup} r={lora_r}] ===",
          flush=True)
    ds = TRAIN_AUG if aug == AUG else build_data(make_train_tf(aug))[0]
    if per_class or n_classes:
        ds, n_used = subsample(ds, per_class, n_classes)
        print(f"    subsampled training set -> {n_used} images "
              f"(per_class={per_class}, n_classes={n_classes})", flush=True)
    t0 = time.time()
    m = adapt(mode, ds, epochs=epochs, lr=lr, warmup=warmup, lora_r=lora_r)
    F1 = extract(m, TRAIN); F2 = extract(m, TEST)      # extraction is ALWAYS deterministic
    del m; torch.cuda.empty_cache()
    n_, (r_, l_) = ncm(F1, TR_Y, F2, TE_Y), ranpac(F1, TR_Y, F2, TE_Y)
    results[tag] = (n_, r_)
    print(f"  -> {tag}: NCM {n_:.4f} | RanPAC {r_:.4f} (lam={l_:g})  "
          f"[{(time.time()-t0)/60:.0f} min]", flush=True)
    return n_, r_


# RECIPE SWEEP. The baseline row reproduces the original 0.8332 measurement exactly; the rest
# probe whether the ceiling is underfit. Ordered cheapest-first so an early stop is informative.
#            tag                mode     aug  ep   lr    warm  r
RECIPES = [
    ("lora_base_noaug_10ep",   "lora",    0,  10, 1e-4,   0,  32),   # == the original number
    ("lora_aug_10ep",          "lora",    1,  10, 1e-4,   0,  32),   # augmentation alone
    ("lora_aug_40ep",          "lora",    1,  40, 1e-4,   3,  32),   # + longer schedule
    ("lora_aug_40ep_lr3e-4",   "lora",    1,  40, 3e-4,   3,  32),   # + higher lr
    ("lora_aug_40ep_r64",      "lora",    1,  40, 1e-4,   3,  64),   # + more adapter capacity
]

# DIVERSITY vs VOLUME control. `joint (0.8498) > A_plus (0.7897)` confounds two things:
# joint sees 200 CLASSES *and* 24000 IMAGES; A_plus sees 20 classes and 2400 images.
#   div_only : 200 classes, 2400 images (12/class)  -- full diversity, one task's volume
#   vol_only :  20 classes, 2400 images             -- one task's diversity and volume (~A_plus)
# div_only ~ joint  => DIVERSITY is what matters -> a cross-task loss can close the gap
# div_only ~ vol_only => VOLUME is what matters -> no prototype-free loss will; only replay
CONTROLS = [
    ("div_only_200cls_12each", "lora", 1, 40, 1e-4, 3, 32, 12,  None),
    ("vol_only_20cls_120each", "lora", 1, 40, 1e-4, 3, 32, 120, 20),
]

# ---------------------------------------------------------------------------------------
# STAGE 0 -- IS 0.8498 EVEN THE RIGHT CEILING?
#
# Our bound was measured with a SINGLE rank-32 LoRA. MACIL (SOTA, 0.8188) aggregates ten
# task-specific adapters:  W = W0 + sum_{i=1..T} B_i A_i.  If that architecture has a higher
# achievable bound, every target recomputes and "+2 over SOTA" changes character.
#
# KEY EQUIVALENCE: sum_i B_i A_i is a sum of T rank-r terms, hence a single matrix of rank
# <= T*r. Trained JOINTLY (all adapters simultaneously, all data), the aggregated
# task-specific architecture is therefore EXACTLY as expressive as one LoRA of rank T*r.
# So the ceiling question reduces to a RANK SWEEP -- no need to simulate their construction,
# and this is the BEST case for that parameter budget (their adapters each see only one
# task's 20 classes, so their sequential build can only be <= this).
#
#   T=10, r=32  ->  rank 320 is the aggregated-architecture ceiling
#   r64 was already measured: +0.0012 over r32, i.e. rank is NOT a lever at small steps.
#
# READ: if r320 ~ r32, capacity is not the constraint and 0.8498 stands regardless of how
# many adapters SOTA stacks -- their advantage is then the sequential SPECIALISATION, not
# the parameter count. If r320 >> r32, we have been measuring against the wrong bound.
CEILING_RANKS = [
    ("lora_aug_40ep_r128", "lora", 1, 40, 1e-4, 3, 128),
    ("lora_aug_40ep_r320", "lora", 1, 40, 1e-4, 3, 320),   # == 10 aggregated r32 adapters
]

if int(os.environ.get("RANKS", 0)):
    for tag, mode, aug, ep, lr, warm, r in CEILING_RANKS:
        measure(tag, mode, aug, ep, lr, warm, r)
elif int(os.environ.get("CONTROL", 0)):
    for tag, mode, aug, ep, lr, warm, r, pc, nc in CONTROLS:
        measure(tag, mode, aug, ep, lr, warm, r, per_class=pc, n_classes=nc)
elif SWEEP:
    for tag, mode, aug, ep, lr, warm, r in RECIPES:
        measure(tag, mode, aug, ep, lr, warm, r)
else:
    for mode in MODES:
        measure(mode, mode, AUG, EPOCHS, LR, WARMUP, LORA_R)

OUT = f"crux_headroom_{DATASET}_{MODEL.split('.')[-1]}.npy"
merged = {}
if os.path.exists(OUT):
    try:
        merged = np.load(OUT, allow_pickle=True).item()
    except Exception as e:
        print(f"[warn] {e}")
merged.update(results)
np.save(OUT, merged, allow_pickle=True)

print("\n" + "=" * 84)
print(f"HEADROOM — {DATASET} / {MODEL}  (joint/offline; adaptation's best case)")
print("=" * 84)
print(f"{'recipe':>24} {'NCM':>9} {'RanPAC':>9} | {'dNCM':>8} {'dRanPAC':>9}")
fn, fr = merged["frozen"]
for k, (n_, r_) in sorted(merged.items(), key=lambda kv: kv[1][1]):
    star = "  <- frozen floor" if k == "frozen" else ""
    print(f"{k:>24} {n_:>9.4f} {r_:>9.4f} | {n_-fn:>+8.4f} {r_-fr:>+9.4f}{star}")
best_k, (best_n, best_r) = max(merged.items(), key=lambda kv: kv[1][1])
print("-" * 84)
print(f"CEILING = {best_r:.4f}  ({best_k})")
base = merged.get("lora_base_noaug_10ep", merged.get("lora"))
if base and best_k != "frozen":
    print(f"recipe gain over the original 10ep/no-aug recipe: {best_r-base[1]:+.4f}")
print(f"\nNo CIL method can exceed {best_r:.4f}. Targets:")
for tgt in (0.80, 0.82, 0.84):
    frac = (tgt - fr) / max(best_r - fr, 1e-9)
    ok = "reachable" if tgt < best_r else "ABOVE CEILING — impossible"
    print(f"  A-last {tgt:.2f}: needs {100*frac:5.1f}% of headroom captured   ({ok})")
print(f"RanPAC headroom = {best_r - fr:+.4f}")
print("  <= 0  -> NO headroom: no CIL adaptation scheme can beat RanPAC here. Stop.")
print("  >> 0  -> headroom is real; build the CIL method that captures it.")

# ---- STAGE 0 verdict: is the single-r32 bound the real ceiling? ----
r32 = merged.get("lora_aug_40ep", merged.get("lora"))
ranks = [(int(k.split("_r")[-1]), v[1]) for k, v in merged.items()
         if "lora_aug_40ep_r" in k and k.split("_r")[-1].isdigit()]
if r32 and ranks:
    print("\n" + "-" * 84)
    print("STAGE 0 — CEILING vs ADAPTER RANK  (rank T*r == T aggregated task-specific adapters)")
    print(f"{'rank':>6} {'RanPAC':>9} {'vs r32':>9}")
    print(f"{32:>6} {r32[1]:>9.4f} {0.0:>+9.4f}")
    for rk, acc in sorted(ranks):
        print(f"{rk:>6} {acc:>9.4f} {acc-r32[1]:>+9.4f}")
    top_rank, top_acc = max(ranks, key=lambda kv: kv[1])
    d = top_acc - r32[1]
    print(f"\nbest rank {top_rank}: {top_acc:.4f}  ({d:+.4f} over r32)")
    if d < 0.005:
        print("=> CAPACITY IS NOT THE CONSTRAINT. 0.8498 stands as the ceiling however many")
        print("   adapters SOTA stacks; their edge is sequential SPECIALISATION, not parameters.")
        print("   Targets are unchanged: A-last 0.84 remains ~98.7% of ceiling.")
    else:
        print("=> WE WERE MEASURING AGAINST THE WRONG BOUND. Recompute every target against")
        print(f"   the new ceiling {top_acc:.4f}; '+2 over SOTA' just got easier.")
print("=" * 84)
