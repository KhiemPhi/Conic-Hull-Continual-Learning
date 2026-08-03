#!/usr/bin/env python3
"""exp11_mrp_sweep.py — how much of A_plus_aug40 survives a SMALLER RanPAC projection?

WHY THIS EXISTS
    We are building a storage-matched argument against the "exemplar-free" SOTA line
    (GR-LoRA / MACIL / SSIAT / SLCA), which stores mu + Sigma per class:

        589,824 + 768 floats/class = 2.362 MB/class  ->  473 MB on ImageNet-R (200 cls)

    That argument collapses if OUR baseline is just as expensive. It nearly is. A_plus
    uses RanPAC with M_RP = 10000, and the accumulator G is M_RP x M_RP:

        G (float64) = 10000^2 x 8 B = 800 MB      <- MORE than the covariances we criticise

    G is the state that must PERSIST across tasks (C too, but C is only M_RP x 200).
    So before claiming a storage win we have to know the accuracy/storage curve in M_RP.
    If M_RP=2000 (32 MB fp64) keeps most of 0.8027, the baseline beats ten published
    methods at ~7% of their storage. If it needs all 10000, the framing is dead and we
    should learn that now rather than in review.

WHAT MAKES THIS CHEAP
    A_plus freezes the backbone after task 0 (freeze_after=0), so features are IDENTICAL
    at every stage -- verified: forensics_aplus_s0.npz has F0 == F9 exactly. Therefore the
    whole 10-stage protocol is head arithmetic over one cached feature matrix. Train once,
    extract once, replay per M_RP. Every M_RP sees the SAME features, so the comparison is
    exactly paired: only the projection width changes.

FIDELITY CHECK
    At M_RP=10000 this must reproduce exp8's A_plus_aug40 = 0.8027 / 0.8552. The script
    asserts closeness and says so loudly if not -- a replay that does not reproduce the
    original number is not measuring what we think it is.

STORAGE ACCOUNTING (reported three ways, because they differ by 20x)
    persist_fp64  G + C in float64   -- what exp8 actually holds across tasks
    persist_fp32  G + C in float32   -- the honest minimum for the same algorithm
    inference     P_RP + W           -- deployment only; P_RP is regenerable from a seed,
                                        so the true inference floor is W alone (M x 200)
    The number comparable to GR-LoRA's 473 MB is persist_*, since that is per-task state.

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp11_mrp_sweep.py                     # train/cache if needed, then sweep
    MRP=1000,2000,10000 python -u exp11_mrp_sweep.py
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

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ---- protocol: copied verbatim from exp8_combined.py so the replay is comparable -------
SEED = int(os.environ.get("SEED", 0))
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
N_TASKS, CPT = 10, 20
EPOCHS = int(os.environ.get("EPOCHS", 40))      # aug40 recipe
LR, BS = float(os.environ.get("LR", 1e-4)), 128
GRAD_CLIP = 1.0
LAMBDAS = [1e2, 1e3, 1e4]                       # identical grid, or fidelity is meaningless
AUG = int(os.environ.get("AUG", 1))
MRP_LIST = [int(v) for v in os.environ.get("MRP", "256,512,1000,2000,5000,10000").split(",")]

# reference to reproduce (exp8_results_augreg_in21k.npy: A_plus_aug40)
REF_LAST, REF_AVG = 0.8027, 0.8552

_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
TF_EVAL = create_transform(**_cfg, is_training=False)
TF_TRAIN = (create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                             re_prob=0.25, scale=(0.7, 1.0), hflip=0.5) if AUG else TF_EVAL)


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
        return self.tf(img), int(self.labels[i]), i


from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"]); _cl = np.array(sorted(set(_w.tolist())))
_lab = np.searchsorted(_cl, _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
TRAIN_AUG = HFWrap(_ds, _p[:_n], _lab[_p[:_n]], TF_TRAIN)
TRAIN = HFWrap(_ds, _p[:_n], _lab[_p[:_n]], TF_EVAL);  TR_Y = _lab[_p[:_n]]
TEST = HFWrap(_ds, _p[_n:], _lab[_p[_n:]], TF_EVAL);   TE_Y = _lab[_p[_n:]]
N_CLS = len(_cl)
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- stage 1: features
@torch.no_grad()
def extract(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def get_aplus_features():
    """Train A_plus (task 0 only, then FREEZE) and cache train/test features."""
    cache = f"exp11_aplus_feats_{'aug40' if AUG else 'ep10'}_s{SEED}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"A_plus features from cache {cache}")
        return z["Ftr"], z["Fte"]

    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))

    cls = np.asarray(TASKS[0])
    remap = {int(c): i for i, c in enumerate(cls)}
    idx = np.where(np.isin(TR_Y, cls))[0]
    loader = DataLoader(Subset(TRAIN_AUG, idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
    head = nn.Linear(768, CPT).to(DEV)
    opt = torch.optim.AdamW(lp + list(head.parameters()), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    log(f"training A_plus task 0 ({len(idx)} imgs, {EPOCHS} ep, AUG={AUG})")
    for ep in range(EPOCHS):
        model.train(); ok = tot = 0
        for x, lab, _ in loader:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x).float()
                loss = ce(head(f), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(lp + list(head.parameters()), GRAD_CLIP)
            opt.step()
            ok += int((head(f).argmax(1) == y).sum()); tot += len(y)
        sch.step()
    log(f"  task-0 train acc {ok/tot:.3f}  -> BACKBONE NOW FROZEN (this is A_plus)")

    Ftr = extract(model, TRAIN, np.arange(len(TR_Y)))
    Fte = extract(model, TEST, np.arange(len(TE_Y)))
    del model; torch.cuda.empty_cache()
    np.savez(cache, Ftr=Ftr, Fte=Fte)
    log(f"cached -> {cache}  train {Ftr.shape} test {Fte.shape}")
    return Ftr, Fte


# ---------------------------------------------------------------- stage 2: head replay
def replay(Ftr, Fte, M_RP):
    """Exact exp8 accum protocol, in feature space. Only M_RP changes."""
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

        best, bestW, bestlam = -1.0, None, None
        for lam in LAMBDAS:
            W = torch.linalg.solve(G + lam * eye, C)
            a = acc(W, Zval, yval)
            if a > best:
                best, bestW, bestlam = a, W, lam
        return acc(bestW, Zte, yte), bestlam

    Gacc = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    Cacc = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    vZ, vy, accs, lams = [], [], [], []
    for t in range(N_TASKS):
        cls = np.asarray(TASKS[t])
        idx = np.where(np.isin(TR_Y, cls))[0]
        y_task = TR_Y[idx]
        Zr = un(Ftr[idx])
        pm = np.random.default_rng(t).permutation(len(idx))
        nv = max(int(0.1 * len(idx)), 1)
        vZ.append(Zr[pm[:nv]]); vy.append(y_task[pm[:nv]])
        g, c_ = build_GC(Zr[pm[nv:]], y_task[pm[nv:]])
        Gacc += g; Cacc += c_
        seen = np.concatenate(TASKS[:t + 1])
        te = np.where(np.isin(TE_Y, seen))[0]
        a, lam = solve_eval(Gacc, Cacc, np.concatenate(vZ), np.concatenate(vy),
                            Fte[te], TE_Y[te], seen)
        accs.append(a); lams.append(lam)
    del Gacc, Cacc, P_RP
    torch.cuda.empty_cache()
    return accs, lams


def storage(M_RP):
    persist = M_RP * M_RP + M_RP * N_CLS          # G + C
    infer = 768 * M_RP + M_RP * N_CLS             # P_RP + W (P_RP regenerable from seed)
    return dict(persist_fp64_MB=persist * 8 / 1e6, persist_fp32_MB=persist * 4 / 1e6,
                infer_fp32_MB=infer * 4 / 1e6, W_only_MB=M_RP * N_CLS * 4 / 1e6)


def main():
    Ftr, Fte = get_aplus_features()
    rows = []
    for M in sorted(MRP_LIST):
        t0 = time.time()
        accs, lams = replay(Ftr, Fte, M)
        s = storage(M)
        r = dict(M_RP=M, A_last=accs[-1], A_avg=float(np.mean(accs)),
                 lam_last=lams[-1], lam_all=lams, accs=accs, **s)
        rows.append(r)
        log(f"  M_RP {M:>5}  A-last {r['A_last']:.4f}  A-avg {r['A_avg']:.4f}  "
            f"persist {s['persist_fp64_MB']:>7.1f} MB(f64) / {s['persist_fp32_MB']:>6.1f} "
            f"MB(f32)  lam={lams[-1]:g}  [{time.time()-t0:.0f}s]")

    ref = [r for r in rows if r["M_RP"] == 10000]
    print("\n" + "=" * 104)
    print(f"EXP11 — RanPAC projection width vs storage   [A_plus{'_aug40' if AUG else ''}, "
          f"ImageNet-R, {MODEL}]")
    if ref:
        d_last, d_avg = ref[0]["A_last"] - REF_LAST, ref[0]["A_avg"] - REF_AVG
        ok = abs(d_last) < 0.01 and abs(d_avg) < 0.01
        print(f"FIDELITY @M_RP=10000: {ref[0]['A_last']:.4f}/{ref[0]['A_avg']:.4f} vs exp8 "
              f"{REF_LAST}/{REF_AVG}  (d {d_last:+.4f}/{d_avg:+.4f})  "
              f"{'OK' if ok else '*** MISMATCH — replay is not reproducing exp8 ***'}")
    else:
        print("FIDELITY: M_RP=10000 not in sweep — cannot verify the replay.")
    print("=" * 104)
    print(f"{'M_RP':>7}{'A-last':>9}{'A-avg':>9}{'dA-last':>9}{'persist f64':>13}"
          f"{'persist f32':>13}{'infer':>9}{'vs SOTA 473MB':>15}")
    base = ref[0]["A_last"] if ref else max(r["A_last"] for r in rows)
    for r in rows:
        print(f"{r['M_RP']:>7}{r['A_last']:>9.4f}{r['A_avg']:>9.4f}"
              f"{r['A_last']-base:>+9.4f}{r['persist_fp64_MB']:>12.1f}M"
              f"{r['persist_fp32_MB']:>12.1f}M{r['infer_fp32_MB']:>8.1f}M"
              f"{100*r['persist_fp32_MB']/473:>14.1f}%")
    print("-" * 104)
    print("SOTA reference (ImageNet-R 10-task, ViT-B/16-IN21k):")
    print("  GR-LoRA 82.09/86.20 | MACIL 81.82/85.76 | CL-LoRA 79.78/85.10 | "
          "SSIAT 79.54/83.67 | SLCA 79.35/83.29")
    print("  all store mu+Sigma per class = 473 MB on ImageNet-R")
    print("-" * 104)
    beats = [r for r in rows if r["A_last"] > 0.7978]     # > CL-LoRA
    if beats:
        cheap = min(beats, key=lambda r: r["persist_fp32_MB"])
        print(f"CHEAPEST CONFIG STILL BEATING CL-LoRA (79.78): M_RP={cheap['M_RP']} at "
              f"{cheap['persist_fp32_MB']:.1f} MB fp32 = "
              f"{100*cheap['persist_fp32_MB']/473:.1f}% of the SOTA budget "
              f"(A-last {cheap['A_last']:.4f})")
    else:
        print("No config beats CL-LoRA — the storage-matched framing needs rethinking.")
    print("=" * 104)

    out = f"exp11_mrp_sweep_{MODEL.split('.')[-1]}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
