#!/usr/bin/env python3
"""exp12_free_points.py — ZERO-MEMORY levers on A_plus: backbone checkpoint + first-session recipe.

WHY THIS EXISTS
    A_plus (adapt task 0, then freeze) is the bar, and its ENTIRE accuracy rests on one
    training run over 20 classes. Two knobs cost zero stored bytes and have never been
    tuned as such:

      1. WHICH IN21k CHECKPOINT.  Papers say only "ViT-B/16 IN21k". timm has two
         pretrain-only candidates, both already cached locally:
             vit_base_patch16_224.augreg_in21k   (what every result so far used)
             vit_base_patch16_224.orig_in21k     (original Google ViT)
         This choice already flipped a conclusion once: "continued adaptation is
         net-negative" was an artifact of augreg2_in21k_ft_in1k. augreg2 / *_ft_in1k are
         EXCLUDED here -- they are IN1k-finetuned and ImageNet-R's 200 classes ARE IN1k
         classes, so they inflate the frozen baseline and are not comparable to published
         numbers.
      2. FIRST-SESSION RECIPE.  epochs and lr for that single task-0 run.

    Target: A_plus_aug40 is 80.27/85.52 (exp8) against GR-LoRA 82.09/86.20. The A-Avg gap
    is only -0.68, which is inside what these levers could plausibly close.

FAIR-COMPARISON RULES BAKED IN
    * The transform is resolved PER MODEL. orig_in21k and augreg_in21k do not share
      normalisation; a globally-resolved transform would silently handicap one of them.
    * Every config re-trains task 0 and re-extracts features. The feature cache key
      includes model/epochs/lr/aug/seed -- exp11's cache key did NOT include the model,
      which would have silently served augreg features for an orig_in21k run.
    * Deltas are reported against THIS script's own baseline row (augreg_in21k, 40ep,
      1e-4), never against exp8's recorded 80.27. Different runs of the same config differ;
      mixing them would attribute run-to-run variation to the lever under test.
    * The head replay is byte-identical to exp8/exp11 (same RP seed, same lambda grid,
      same accum protocol), so only the backbone/recipe varies.

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp12_free_points.py ckpt      # stage A: checkpoints @ 40ep/1e-4   (~12 min)
    python -u exp12_free_points.py recipe    # stage B: epochs x lr on best ckpt  (~30 min)
    MRP=10000,3000 python -u exp12_free_points.py ckpt
"""
import json
import os
import sys
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


SEED = int(os.environ.get("SEED", 0))
DEV = "cuda"
N_TASKS, CPT, BS = 10, 20, 128
GRAD_CLIP = 1.0
LAMBDAS = [1e2, 1e3, 1e4]
AUG = int(os.environ.get("AUG", 1))
MRP_LIST = [int(v) for v in os.environ.get("MRP", "10000,3000").split(",")]

def _full(name):
    """Allow short checkpoint names on the command line: 'orig_in21k' -> full timm id."""
    return name if "." in name else f"vit_base_patch16_224.{name}"


# Delta reference. Overridable, because the right baseline depends on what you are testing:
# comparing checkpoints at lr3e-4 needs BASE_LR=3e-4, or every row diffs against a config
# that is not in the run and the deltas are meaningless.
BASE_MODEL = _full(os.environ.get("BASE_MODEL", "vit_base_patch16_224.augreg_in21k"))
BASE_EP = int(os.environ.get("BASE_EP", 40))
BASE_LR = float(os.environ.get("BASE_LR", 1e-4))
EXP8_REF = (0.8027, 0.8552)          # A_plus_aug40 as recorded (SEED=1); context only


# LoRA target-module sets. NOTE timm's ViT uses a FUSED attn.qkv Linear, so GR-LoRA's
# "key and value projections only" is not expressible without slicing that matrix -- these
# are the variations the existing inject_lora supports.
TARGETS = {
    "qkvproj": ["attn.qkv", "attn.proj"],                              # backbone.py default
    "qkv": ["attn.qkv"],
    "qkvprojmlp": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
}
DEF_RANK, DEF_ALPHA, DEF_TGT = 32, 4.0, "qkvproj"


def cfg(model, ep, lr, rank=DEF_RANK, alpha=DEF_ALPHA, tgt=DEF_TGT):
    return (_full(model), int(ep), float(lr), int(rank), float(alpha), tgt)


def parse_configs(spec):
    """CONFIGS='model:ep:lr[:rank:alpha:targets]', comma-separated.

    LoRA scaling is alpha/rank (backbone.py:LoRALinear), so a rank sweep at FIXED alpha is
    also a scale sweep. Pass alpha explicitly to hold alpha/rank constant and isolate rank.
    """
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        p = item.split(":")
        if len(p) == 3:
            out.append(cfg(p[0], p[1], p[2]))
        elif len(p) == 6:
            out.append(cfg(p[0], p[1], p[2], p[3], p[4], p[5]))
        else:
            raise ValueError(f"bad config '{item}': need model:ep:lr or "
                             f"model:ep:lr:rank:alpha:targets")
    return out

# ---- data (images/labels/split are model-independent; the TRANSFORM is not) -------------
from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"]); _cl = np.array(sorted(set(_w.tolist())))
_lab = np.searchsorted(_cl, _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
TR_IDX, TR_Y = _p[:_n], _lab[_p[:_n]]
TE_IDX, TE_Y = _p[_n:], _lab[_p[_n:]]
N_CLS = len(_cl)
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]


class HFWrap(Dataset):
    def __init__(self, idx, labels, tf):
        self.idx, self.labels, self.tf = np.asarray(idx), np.asarray(labels), tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = _ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(self.labels[i])


def transforms_for(model_name):
    cfg = resolve_model_data_config(timm.create_model(model_name, pretrained=False,
                                                      num_classes=0))
    tf_eval = create_transform(**cfg, is_training=False)
    tf_train = (create_transform(**cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                                 re_prob=0.25, scale=(0.7, 1.0), hflip=0.5)
                if AUG else tf_eval)
    return tf_train, tf_eval


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


@torch.no_grad()
def extract(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def aplus_features(model_name, epochs, lr, rank=DEF_RANK, alpha=DEF_ALPHA, tgt=DEF_TGT):
    """Train task 0 then FREEZE (= A_plus); return (train, test) features."""
    stem = f"{model_name.split('.')[-1]}_ep{epochs}_lr{lr:g}"
    # Default LoRA settings keep the ORIGINAL cache key so the rows already computed
    # (exp12 stages ckpt/recipe/lr) are reused byte-for-byte instead of retrained.
    if (rank, alpha, tgt) != (DEF_RANK, DEF_ALPHA, DEF_TGT):
        stem += f"_r{rank}_a{alpha:g}_{tgt}"
    key = f"{stem}_aug{AUG}_s{SEED}"
    cache = f"exp12_feats_{key}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"  [{key}] features from cache")
        return z["Ftr"], z["Fte"]

    torch.manual_seed(SEED); np.random.seed(SEED)     # same init state for every config
    tf_train, tf_eval = transforms_for(model_name)
    tr_aug = HFWrap(TR_IDX, TR_Y, tf_train)
    tr_ev = HFWrap(TR_IDX, TR_Y, tf_eval)
    te_ev = HFWrap(TE_IDX, TE_Y, tf_eval)

    model = load_backbone(model_name, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=rank, lora_alpha=alpha, lora_config="task_shared",
                          lora_target_modules=TARGETS[tgt])
    freeze_non_lora(model)
    lp = list(get_lora_params(model))

    cls = np.asarray(TASKS[0])
    remap = {int(c): i for i, c in enumerate(cls)}
    idx = np.where(np.isin(TR_Y, cls))[0]
    loader = DataLoader(Subset(tr_aug, idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
    head = nn.Linear(model.num_features, CPT).to(DEV)
    opt = torch.optim.AdamW(lp + list(head.parameters()), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    log(f"  [{key}] training task 0 ({len(idx)} imgs)")
    ok = tot = 1
    for _ in range(epochs):
        model.train(); ok = tot = 0
        for x, lab in loader:
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
    log(f"  [{key}] task-0 train acc {ok/tot:.3f} -> frozen")
    Ftr, Fte = extract(model, tr_ev), extract(model, te_ev)
    del model; torch.cuda.empty_cache()
    np.savez(cache, Ftr=Ftr, Fte=Fte)
    return Ftr, Fte


def replay(Ftr, Fte, M_RP):
    """exp8 accum protocol in feature space -- identical to exp11."""
    d = Ftr.shape[1]
    P_RP = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

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

    Gacc = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    Cacc = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    vZ, vy, accs = [], [], []
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
        accs.append(solve_eval(Gacc, Cacc, np.concatenate(vZ), np.concatenate(vy),
                               Fte[te], TE_Y[te], seen))
    del Gacc, Cacc, P_RP
    torch.cuda.empty_cache()
    return accs


def run_configs(configs, tag):
    # SEED must be in the filename: ORDER = rng(SEED).permutation(N_CLS), so different seeds
    # are different CLASS ORDERS -- a different benchmark instance, not a repeat run. Mixing
    # them in one file would silently diff a seed-1 config against a seed-0 baseline.
    out = f"exp12_free_points_{tag}_s{SEED}.json"
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["model"], r["epochs"], r["lr"], r.get("rank", DEF_RANK),
             r.get("alpha", DEF_ALPHA), r.get("targets", DEF_TGT)) for r in rows}
    for model_name, ep, lr, rank, alpha, tgt in configs:
        if (model_name, ep, lr, rank, alpha, tgt) in done:
            log(f"skip (already done) {model_name} ep{ep} lr{lr:g} r{rank} a{alpha:g} {tgt}")
            continue
        t0 = time.time()
        Ftr, Fte = aplus_features(model_name, ep, lr, rank, alpha, tgt)
        r = dict(model=model_name, epochs=ep, lr=lr, rank=rank, alpha=alpha, targets=tgt)
        for M in MRP_LIST:
            a = replay(Ftr, Fte, M)
            r[f"A_last@{M}"] = a[-1]
            r[f"A_avg@{M}"] = float(np.mean(a))
            r[f"accs@{M}"] = a
        rows.append(r)
        json.dump(rows, open(out, "w"), indent=2)
        m = MRP_LIST[0]
        log(f"  -> {model_name.split('.')[-1]:13s} ep{ep:<3d} lr{lr:<7g} r{rank:<3d} "
            f"a{alpha:<5g} {tgt:<11s} A-last {r[f'A_last@{m}']:.4f}  "
            f"A-avg {r[f'A_avg@{m}']:.4f}  [{time.time()-t0:.0f}s]")
    return rows, out


def report(rows, tag):
    m0 = MRP_LIST[0]
    base = next((r for r in rows if r["model"] == BASE_MODEL and r["epochs"] == BASE_EP
                 and abs(r["lr"] - BASE_LR) < 1e-12
                 and r.get("rank", DEF_RANK) == DEF_RANK
                 and r.get("alpha", DEF_ALPHA) == DEF_ALPHA
                 and r.get("targets", DEF_TGT) == DEF_TGT), None)
    W = 112
    print("\n" + "=" * W)
    print(f"EXP12 [{tag}] — zero-memory levers on A_plus   (ImageNet-R, 10 tasks, AUG={AUG})")
    print(f"deltas are vs THIS run's baseline: {BASE_MODEL.split('.')[-1]} ep{BASE_EP} "
          f"lr{BASE_LR:g}" + (f" = {base[f'A_last@{m0}']:.4f}/{base[f'A_avg@{m0}']:.4f}"
                              if base else " (NOT RUN)"))
    print(f"context only, do not diff against these: exp8 A_plus_aug40 "
          f"{EXP8_REF[0]:.4f}/{EXP8_REF[1]:.4f} | GR-LoRA 0.8209/0.8620 | MACIL 0.8182/0.8576")
    print("=" * W)
    cols = "".join(f"{'A-last@'+str(M):>13}{'A-avg@'+str(M):>13}" for M in MRP_LIST)
    print(f"{'checkpoint':<15}{'ep':>4}{'lr':>7}{'rank':>5}{'alpha':>6}{'scale':>7}"
          f"{'targets':>12}{cols}{'dA-last':>10}{'dA-avg':>9}")
    for r in sorted(rows, key=lambda r: -r[f"A_last@{m0}"]):
        vals = "".join(f"{r[f'A_last@{M}']:>13.4f}{r[f'A_avg@{M}']:>13.4f}" for M in MRP_LIST)
        dl = da = float("nan")
        if base:
            dl = r[f"A_last@{m0}"] - base[f"A_last@{m0}"]
            da = r[f"A_avg@{m0}"] - base[f"A_avg@{m0}"]
        rk, al = r.get('rank', DEF_RANK), r.get('alpha', DEF_ALPHA)
        print(f"{r['model'].split('.')[-1]:<15}{r['epochs']:>4}{r['lr']:>7g}{rk:>5}"
              f"{al:>6g}{al/rk:>7.4f}{r.get('targets', DEF_TGT):>12}{vals}"
              f"{dl:>+10.4f}{da:>+9.4f}")
    print("=" * W)


STAGES = {
    "ckpt": [cfg(BASE_MODEL, BASE_EP, BASE_LR),
             cfg("orig_in21k", BASE_EP, BASE_LR)],
    # cross, not full grid: vary one knob at a time off the baseline
    "recipe": [cfg(BASE_MODEL, BASE_EP, BASE_LR), cfg(BASE_MODEL, 20, BASE_LR),
               cfg(BASE_MODEL, 80, BASE_LR), cfg(BASE_MODEL, BASE_EP, 3e-5),
               cfg(BASE_MODEL, BASE_EP, 3e-4)],
    # LR probe. 3e-5 -> 1e-4 -> 3e-4 was monotone increasing at seed 0, so the top of the
    # curve has not been found. lr1e-4 is included as the in-run baseline (and as a check
    # against exp8's recorded seed-1 A_plus_aug40 = 0.8027/0.8552); lr3e-4 because if 1e-3
    # turns over, 3e-4 is the config that would actually be used and it must be measured on
    # the SAME class order to be comparable.
    "lr": [cfg(BASE_MODEL, BASE_EP, BASE_LR), cfg(BASE_MODEL, BASE_EP, 3e-4),
           cfg(BASE_MODEL, BASE_EP, 1e-3)],
    # LoRA screen at the new lr3e-4 optimum. Scale = alpha/rank, so each off-baseline rank
    # appears TWICE: once at fixed alpha (scale moves with rank) and once scale-matched to
    # the baseline's 0.125 -- otherwise "rank helped" is indistinguishable from "scale did".
    "lora": [cfg(BASE_MODEL, 40, 3e-4, 32, 4.0, "qkvproj"),      # baseline (cached)
             cfg(BASE_MODEL, 40, 3e-4, 10, 4.0, "qkvproj"),      # scale 0.400
             cfg(BASE_MODEL, 40, 3e-4, 10, 1.25, "qkvproj"),     # scale 0.125 (rank only)
             cfg(BASE_MODEL, 40, 3e-4, 64, 4.0, "qkvproj"),      # scale 0.0625
             cfg(BASE_MODEL, 40, 3e-4, 64, 8.0, "qkvproj"),      # scale 0.125 (rank only)
             cfg(BASE_MODEL, 40, 3e-4, 32, 4.0, "qkv"),          # drop attn.proj
             cfg(BASE_MODEL, 40, 3e-4, 32, 4.0, "qkvprojmlp")],  # add MLP
}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "ckpt"
    spec = os.environ.get("CONFIGS")
    cfgs = parse_configs(spec) if spec else STAGES[stage]
    log(f"stage '{stage}' seed {SEED}: {len(cfgs)} configs, M_RP={MRP_LIST}, "
        f"delta-ref {BASE_MODEL.split('.')[-1]} ep{BASE_EP} lr{BASE_LR:g}")
    rows, out = run_configs(cfgs, stage)
    report(rows, stage)
    print(f"wrote {out}")
