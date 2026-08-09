#!/usr/bin/env python3
"""exp16_full_table.py — the publication table: 3 seeds x 4 datasets x {10,20,50} tasks.

WHAT THIS IS FOR
    A_plus (adapt task 0 with the tuned recipe, then FREEZE) currently sits 3rd on A-Last and
    2nd on A-Avg on ImageNet-R 10-task -- but that is ONE dataset, ONE task count, ONE seed,
    and the margins involved (-0.0011 to GR-LoRA, +0.0033 over MACIL on A-Avg) are far inside
    cross-seed variance. No claim is supportable until this table exists.

    It is cheap for one reason: A_plus trains ONCE, on task 0 only. Everything after that is
    feature extraction plus closed-form head arithmetic. 36 configs is hours, not days.

WHAT IS BEING CLAIMED (and therefore what has to be measured)
    Not "we beat SOTA" -- on A-Last we do not. The claim is:
        a first-session baseline with zero stored per-class state ranks 2nd-3rd against
        eleven published PTM-CIL methods that each store 473 MB of per-class covariance.
    That claim needs every cell of the table published methods report, not the one cell where
    the number happens to look best.

THE SECOND RESULT THIS SWEEP TESTS FOR FREE
    exp12 found A_plus accuracy is single-peaked in TASK-0 TRAIN ACCURACY, at ~0.98:
        seed1 ep40:  0.968 -> 0.8027 | 0.983 -> 0.8090 | 0.986 -> 0.8055 | 0.989 -> 0.7985
    If that holds, "early-stop the first session at ~0.98" replaces tuning lr x epochs x rank,
    and it transfers across datasets without re-tuning -- which is the distinguishing result
    against ADAM/APER, whose "first-session adaptation is strong" observation we would
    otherwise merely be re-confirming. So task-0 train accuracy is RECORDED FOR EVERY CONFIG.
    lr3e-4 was tuned on ImageNet-R 10-task; where it lands far from 0.98 (CIFAR has 500
    img/class, ImageNet-A ~30, and T=50 gives task 0 only 2-4 classes) the recipe has not
    transferred and the cell needs a targeted fix rather than a footnote.
    TARGET_ACC=0.98 switches from a fixed schedule to early-stopping on that criterion --
    that is the METHOD version; leave it at 0 for the fixed-recipe table.

PROTOCOL
    Identical to exp12/exp8 in every respect that matters: ViT-B/16 augreg_in21k, LoRA r32
    alpha4 on attn.qkv+attn.proj, AdamW lr3e-4 wd1e-4 cosine, 40 epochs, RandAug, batch 128;
    RanPAC head M_RP=10000, lambda from {1e2,1e3,1e4} on a 10% per-task val carve-out;
    statistics ACCUMULATED (exactly additive -- verified 2.9e-15 in exp14).
    Class order = rng(SEED).permutation(n_classes), the PyCIL/LAMDA-PILOT convention.

COST  ~5-8 min per config (CIFAR is slowest: 60k images to featurise), 36 configs ~4-6 h.
      Every config is checkpointed to JSON and skipped on resume, so kill it freely.

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp16_full_table.py                                   # everything
    DATASETS=IMAGENETR SEEDS=0,1,2 TASKS=10,20,50 python -u exp16_full_table.py
    DATASETS=CIFAR100,CUB200 python -u exp16_full_table.py
    REPORT_ONLY=1 python -u exp16_full_table.py                     # re-print the table
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import timm
from timm.data import create_transform, resolve_model_data_config

from backbone import load_backbone, freeze_non_lora, get_lora_params
from splits import split_indices

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


DEV = "cuda"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
DATASETS = os.environ.get("DATASETS", "CIFAR100,IMAGENETR,IMAGENETA,CUB200").split(",")
TASKCOUNTS = [int(t) for t in os.environ.get("TASKS", "10,20,50").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", 40))
LR = float(os.environ.get("LR", 3e-4))
TARGET_ACC = float(os.environ.get("TARGET_ACC", 0.0))   # >0 => early-stop task 0 on train acc
# Per-dataset epoch override, e.g. EPOCHS_DS=IMAGENETA:150 . ImageNet-A task 0 is 96-480
# images and never reaches TARGET_ACC in 40 epochs, so its two worst cells were UNDER-fit.
# NOTE this is not just a ceiling: EPOCHS is also the cosine T_max, so a different epoch
# budget is a different LR schedule and therefore a different recipe, not "the same recipe
# for longer". That is deliberate here -- lr3e-4/ep40 was tuned on ImageNet-R and the whole
# point is that it did not transfer -- but it must be reported as its own recipe.
EPOCHS_DS = {k: int(v) for k, v in
             (p.split(":") for p in os.environ.get("EPOCHS_DS", "").split(",") if ":" in p)}
BS, GRAD_CLIP = 128, 1.0
AUG = int(os.environ.get("AUG", 1))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SPLIT_SEED = 1993
REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, f"exp16_full_table_{MODEL.split('.')[-1]}.json")
REPORT_ONLY = int(os.environ.get("REPORT_ONLY", 0))

# Published numbers (GR-LoRA ICML'26 Tables 1,2,6), ViT-B/16-IN21k, mean of 3 seeds.
# (A-Last, A-Avg) as percentages.
REF = {
    ("CIFAR100", 10): {"GR-LoRA": (91.97, 94.65), "MACIL": (91.86, 94.44)},
    ("CIFAR100", 20): {"GR-LoRA": (91.46, 94.41), "MACIL": (90.31, 93.47)},
    ("CIFAR100", 50): {"GR-LoRA": (90.03, 93.38), "MACIL": (85.09, 90.89)},
    ("IMAGENETR", 10): {"GR-LoRA": (82.09, 86.20), "MACIL": (81.82, 85.76)},
    ("IMAGENETR", 20): {"GR-LoRA": (80.23, 85.05), "MACIL": (79.46, 84.25)},
    ("IMAGENETR", 50): {"GR-LoRA": (76.74, 82.64), "MACIL": (70.10, 77.47)},
    ("IMAGENETA", 10): {"GR-LoRA": (63.60, 70.24), "MACIL": (63.15, 70.54)},
    ("IMAGENETA", 20): {"GR-LoRA": (62.37, 69.30), "MACIL": (59.40, 67.79)},
    ("IMAGENETA", 50): {"GR-LoRA": (59.71, 67.23), "MACIL": (47.86, 59.96)},
    ("CUB200", 10): {"GR-LoRA": (89.91, 93.85), "MACIL": (90.23, 93.78)},
    ("CUB200", 20): {"GR-LoRA": (89.76, 94.08), "MACIL": (88.63, 93.52)},
    ("CUB200", 50): {"GR-LoRA": (89.68, 93.94), "MACIL": (82.06, 91.04)},
}

def epochs_for(ds):
    return EPOCHS_DS.get(ds, EPOCHS)


def recipe_tag(ds):
    """The recipe identity. THE RESULTS KEY MUST CARRY THIS.

    Bug this fixes: the results key used to be f"{ds}|{T}|{seed}" while the FEATURE cache
    filename carried ep/lr/aug/ta. So a run with a different recipe found the old cell
    already present, logged 'skip (done)', and re-printed the OLD numbers under the NEW
    recipe's header. logs/exp16_target98.txt is exactly that: all 36 cells skipped, so the
    TARGET_ACC=0.98 result -- the one this file's docstring calls the distinguishing result
    against ADAM/APER -- was never actually measured.
    """
    t = f"ep{epochs_for(ds)}_lr{LR:g}_aug{AUG}"
    if TARGET_ACC > 0:
        t += f"_ta{TARGET_ACC:g}"
    return t


_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
TF_EVAL = create_transform(**_cfg, is_training=False)
TF_TRAIN = (create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                             re_prob=0.25, scale=(0.7, 1.0), hflip=0.5) if AUG else TF_EVAL)


# ------------------------------------------------------------------ data
class HFWrap(Dataset):
    def __init__(self, ds, idx, labels, tf):
        self.ds, self.idx, self.labels, self.tf = ds, np.asarray(idx), np.asarray(labels), tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(self.labels[i])


class TVWrap(Dataset):
    """torchvision dataset + swappable transform (CIFAR ships PIL images)."""
    def __init__(self, ds, labels, tf):
        self.ds, self.labels, self.tf = ds, np.asarray(labels), tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i][0]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(self.labels[i])


def get_data(name):
    """(train_aug, train_eval, ytr, test_eval, yte, n_classes). 80/20 at SPLIT_SEED for
    single-split datasets -- the same convention exp8/exp12/spread_check use."""
    from datasets import load_dataset
    if name == "CIFAR100":
        from torchvision import datasets as tvd
        tr = tvd.CIFAR100(os.path.join(REPO, "data"), train=True, download=False)
        te = tvd.CIFAR100(os.path.join(REPO, "data"), train=False, download=False)
        ytr, yte = np.array(tr.targets), np.array(te.targets)
        return (TVWrap(tr, ytr, TF_TRAIN), TVWrap(tr, ytr, TF_EVAL), ytr,
                TVWrap(te, yte, TF_EVAL), yte, 100)

    if name == "IMAGENETR":
        d = load_dataset("axiong/imagenet-r", cache_dir=os.path.join(REPO, "data/hf"))["test"]
        w = np.array(d["wnid"])
        lab = np.searchsorted(np.array(sorted(set(w.tolist()))), w)
        single = True
    elif name in ("IMAGENETA", "IMAGENETAP"):
        # IMAGENETAP is the SAME images under a STRATIFIED per-class 80/20 rather than a
        # global one. ImageNet-A is 7500 images over 200 classes with per-class totals from
        # 3 to 100, and a global cut leaves 4 CLASSES WITH ZERO TEST IMAGES plus one with a
        # single train row -- columns that can never be scored but can still fire false
        # positives, and the source of the "N seen classes have NO rays" warnings. See
        # splits.py. Separate dataset name, for the same reason as CUB200P.
        d = load_dataset("barkermrl/imagenet-a",
                         cache_dir=os.path.join(REPO, "data/hf"))["train"]
        lab = np.array(d["label"]); single = True
    elif name == "CUB200":
        dd = load_dataset("Donghyun99/cub-200-2011",
                          cache_dir=os.path.join(REPO, "data/hf"))
        tr, te = dd["train"], dd["test"]
        ytr, yte = np.array(tr["label"]), np.array(te["label"])
        n = int(max(ytr.max(), yte.max())) + 1
        return (HFWrap(tr, np.arange(len(ytr)), ytr, TF_TRAIN),
                HFWrap(tr, np.arange(len(ytr)), ytr, TF_EVAL), ytr,
                HFWrap(te, np.arange(len(yte)), yte, TF_EVAL), yte, n)
    elif name == "CUB200P":
        # CUB under the PyCIL / LAMDA-PILOT convention: ALL 11,788 images re-split 80/20 at
        # SPLIT_SEED -> 9430 train / 2358 test. `CUB200` above uses the OFFICIAL CUB split,
        # 5994 / 5794, which is a different benchmark: after the read-out's 10% val carve it
        # fits on 5,395 images (27/class) against the published protocol's 9,430 (47/class),
        # i.e. 57% of the data, while testing on a 2.5x larger test set. Every published
        # PTM-CIL CUB number is on THIS split, so `CUB200` cells are not comparable to them.
        #
        # This is a SEPARATE dataset name on purpose. Feature caches and every results key
        # in the repo are keyed by dataset name; redefining CUB200 in place would silently
        # change the meaning of numbers already sitting in exp48/49/50/52's JSONs.
        from datasets import concatenate_datasets
        dd = load_dataset("Donghyun99/cub-200-2011",
                          cache_dir=os.path.join(REPO, "data/hf"))
        d = concatenate_datasets([dd["train"], dd["test"]])
        lab = np.array(d["label"])
        single = True
    else:
        raise ValueError(name)

    tri, tei = split_indices(name, lab)
    n = int(lab.max()) + 1
    return (HFWrap(d, tri, lab[tri], TF_TRAIN), HFWrap(d, tri, lab[tri], TF_EVAL), lab[tri],
            HFWrap(d, tei, lab[tei], TF_EVAL), lab[tei], n)


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


@torch.no_grad()
def extract(model, ds, rows=None):
    model.eval()
    sub = ds if rows is None else Subset(ds, np.asarray(rows).tolist())
    ld = DataLoader(sub, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in ld:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


# ------------------------------------------------------------------ A_plus
def aplus_features(ds_name, T, seed, tr_aug, tr_ev, ytr, te_ev, n_cls):
    """Train task 0 with the tuned recipe, FREEZE, return (Ftr, Fte, acc0, epochs_used)."""
    cpt = n_cls // T
    nep = epochs_for(ds_name)
    tag = f"{ds_name}_T{T}_s{seed}_{recipe_tag(ds_name)}"
    cache = os.path.join(REPO, f"exp16_feats_{tag}_{MODEL.split('.')[-1]}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Ftr"], z["Fte"], float(z["acc0"]), int(z["ep_used"]) if "ep_used" in z \
            else nep

    torch.manual_seed(seed)
    np.random.seed(seed)
    order = np.random.default_rng(seed).permutation(n_cls)
    task0 = order[:cpt]
    idx = np.where(np.isin(ytr, task0))[0]
    remap = {int(c): i for i, c in enumerate(task0)}

    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))
    head = nn.Linear(model.num_features, cpt).to(DEV)
    opt = torch.optim.AdamW(lp + list(head.parameters()), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nep)
    ce = nn.CrossEntropyLoss()
    ld = DataLoader(Subset(tr_aug, idx.tolist()), batch_size=BS, shuffle=True,
                    num_workers=8, pin_memory=True)
    acc0, ep_used = 0.0, nep
    for ep in range(nep):
        model.train()
        ok = tot = 0
        for x, lab in ld:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x).float()
                loss = ce(head(f), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lp + list(head.parameters()), GRAD_CLIP)
            opt.step()
            ok += int((head(f).argmax(1) == y).sum())
            tot += len(y)
        sch.step()
        acc0 = ok / tot
        # exp12's fit curve peaks at ~0.98 task-0 train accuracy; TARGET_ACC turns that
        # observation into the stopping rule instead of tuning lr/epochs per dataset.
        if TARGET_ACC > 0 and acc0 >= TARGET_ACC:
            ep_used = ep + 1
            log(f"    early-stopped at epoch {ep_used}/{nep} (train acc {acc0:.3f})")
            break
    else:
        if TARGET_ACC > 0:
            log(f"    NEVER REACHED TARGET {TARGET_ACC:g} in {nep} epochs "
                f"(train acc {acc0:.3f}) -- this cell is UNDER-fit, raise EPOCHS_DS")
    Ftr, Fte = extract(model, tr_ev), extract(model, te_ev)
    del model
    torch.cuda.empty_cache()
    np.savez(cache, Ftr=Ftr, Fte=Fte, acc0=np.array(acc0), ep_used=np.array(ep_used))
    return Ftr, Fte, acc0, ep_used


# ------------------------------------------------------------------ RanPAC replay
def replay(Ftr, ytr, Fte, yte, T, seed, n_cls):
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    P = torch.randn(Ftr.shape[1], M_RP,
                    generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.tensor(un(Z[i:i + bs]), device=DEV,
                                             dtype=torch.float32) @ P)

    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    vZ, vy, accs = [], [], []
    for t in range(T):
        idx = np.where(np.isin(ytr, tasks[t]))[0]
        Zr, yt_ = un(Ftr[idx]), ytr[idx]
        pm = np.random.default_rng(t).permutation(len(idx))
        nv = max(int(0.1 * len(idx)), 1)
        vZ.append(Zr[pm[:nv]]); vy.append(yt_[pm[:nv]])
        for i, h in _H(Zr[pm[nv:]]):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(yt_[pm[nv:]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        seen_t = torch.tensor(np.asarray(seen), device=DEV)
        te = np.where(np.isin(yte, seen))[0]
        Zv, yv = np.concatenate(vZ), np.concatenate(vy)

        def acc(W, Z, y):
            pr = []
            for _, h in _H(Z):
                pr.append(seen_t[(h.double() @ W)[:, seen_t].argmax(1)].cpu().numpy())
            return float((np.concatenate(pr) == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            W = torch.linalg.solve(G + lam * eye, C)
            a = acc(W, Zv, yv)
            if a > best:
                best, bw = a, W
        accs.append(acc(bw, Fte[te], yte[te]))
    del G, C, P, eye
    torch.cuda.empty_cache()
    return accs


# ------------------------------------------------------------------ main
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

if not REPORT_ONLY:
    for ds_name in DATASETS:
        tr_aug, tr_ev, ytr, te_ev, yte, n_cls = get_data(ds_name)
        log(f"### {ds_name}: {n_cls} classes, train {len(ytr)}, test {len(yte)}")
        for T in TASKCOUNTS:
            if n_cls % T:
                log(f"  skip T={T}: {n_cls} not divisible")
                continue
            for seed in SEEDS:
                key = f"{ds_name}|{T}|{seed}|{recipe_tag(ds_name)}"
                if key in res:
                    log(f"  skip {key} (done)")
                    continue
                t0 = time.time()
                Ftr, Fte, acc0, ep_used = aplus_features(ds_name, T, seed, tr_aug, tr_ev,
                                                         ytr, te_ev, n_cls)
                accs = replay(Ftr, ytr, Fte, yte, T, seed, n_cls)
                res[key] = {"A_last": accs[-1], "A_avg": float(np.mean(accs)),
                            "acc0": acc0, "ep_used": ep_used, "accs": accs,
                            "n_cls": n_cls, "cpt": n_cls // T}
                json.dump(res, open(OUT, "w"), indent=2)
                log(f"  {key:44s} A-Last {accs[-1]*100:6.2f}  A-Avg "
                    f"{np.mean(accs)*100:6.2f}  task0-acc {acc0:.3f}  ep {ep_used}  "
                    f"[{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ report
# One block per RECIPE present in the JSON, so fixed-schedule and target-0.98 sit side by
# side instead of one silently masquerading as the other.
W = 126
TAGS = sorted({k.split("|")[3] for k in res if len(k.split("|")) == 4})
for tag in TAGS:
    print("\n" + "=" * W)
    print(f"EXP16 — A_plus full table   ({MODEL})   RECIPE: {tag}")
    print("A_plus = adapt task 0 with the tuned recipe, then FREEZE. Zero stored per-class "
          "state; the compared methods store 473 MB.")
    print("=" * W)
    print(f"{'dataset':<11}{'T':>4}{'cpt':>5}{'seeds':>6} | {'A-Last (mean+-sd)':>19}"
          f"{'A-Avg (mean+-sd)':>19} | {'GR-LoRA':>15}{'MACIL':>15} | "
          f"{'dLast':>8}{'task0':>8}{'ep':>5}")
    for ds_name in DATASETS:
        for T in TASKCOUNTS:
            ks = [f"{ds_name}|{T}|{s}|{tag}" for s in SEEDS
                  if f"{ds_name}|{T}|{s}|{tag}" in res]
            if not ks:
                continue
            la = np.array([res[k]["A_last"] for k in ks]) * 100
            av = np.array([res[k]["A_avg"] for k in ks]) * 100
            a0 = np.mean([res[k]["acc0"] for k in ks])
            ep = np.mean([res[k].get("ep_used", float("nan")) for k in ks])
            ref = REF.get((ds_name, T), {})
            g, m = ref.get("GR-LoRA"), ref.get("MACIL")
            dl = la.mean() - g[0] if g else float("nan")
            win = " *" if (g and (la.mean() > g[0] or av.mean() > g[1])) else "  "
            print(f"{ds_name:<11}{T:>4}{res[ks[0]]['cpt']:>5}{len(ks):>6} | "
                  f"{la.mean():>12.2f}+-{la.std():<5.2f}"
                  f"{av.mean():>12.2f}+-{av.std():<5.2f} | "
                  f"{(f'{g[0]:.2f}/{g[1]:.2f}' if g else '-'):>15}"
                  f"{(f'{m[0]:.2f}/{m[1]:.2f}' if m else '-'):>15} | "
                  f"{dl:>+8.2f}{a0:>8.3f}{ep:>5.0f}{win}")
    print("-" * W)
print("dLast = ours - GR-LoRA A-Last.  '*' = beats GR-LoRA on either metric.")
print("task0 = mean task-0 TRAIN accuracy, ep = mean epochs actually used.")
print("Under TARGET_ACC, task0 well BELOW the target means the cell never converged and is")
print("        UNDER-fit -- raise EPOCHS_DS for that dataset; it is not a method result.")
print("=" * W)
print(f"wrote {OUT}")
