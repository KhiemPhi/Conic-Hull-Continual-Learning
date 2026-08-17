#!/usr/bin/env python3
"""exp59_seq_lora.py -- train ONE LoRA PER TASK, GR-LoRA's adapter budget, cache every stage.

WHY THIS FILE EXISTS
    Everything in this project so far adapts only in the first session and freezes. That buys
    "no backprop after task 0" but caps the features at whatever task 0 could produce, and
    exp56/exp58 both point at the features as the binding constraint. This file produces the
    other regime -- a NEW LoRA for every task, summed into the backbone exactly as GR-LoRA
    does -- so that the sequential-drift question can be asked with real data instead of a
    proxy.

    It deliberately applies NO forgetting protection: no gradient projection, no orthogonality
    constraint, no distillation, no Gram-preserving term. This is the UNPROTECTED baseline.
    Its whole job is to make drift happen and record it. Protection belongs in the method,
    not in the measurement of how big the problem is.

WHAT "SAME AMOUNT OF LORA AS GR-LORA" MEANS HERE
    GR-LoRA instantiates a new branch per task on k and v, DOUBLED into a shared {A,B} and a
    task-specific {A_fit,B_fit}:
        per task = L * 4 pairs * 2 * d * r = 73728 r      (L=12, d=768)
    We instantiate a new branch per task on attn.qkv + attn.proj:
        per task = L * r * (768 + 2304 + 768 + 768) = 55296 r
    So OUR rank r is equivalent to THEIR rank 0.75 r on parameter count. RANK is printed
    against their budget at import; set RANK=85 to match r=64, or MATCH=64 to have it solved
    for you. Their rank is not verified from any config we hold, so the match is stated
    rather than assumed.

ARCHITECTURE
    lora_config="task_specific" gives W = W0 + sum_j B_j A_j * (alpha/r) with one (A_j,B_j)
    per task; advance_lora_task() freezes the existing pairs and allocates a fresh trainable
    one. Only the newest pair ever receives gradient, so this is per-task LoRA in the same
    sense GR-LoRA's shared branch is -- not a single adapter that keeps training.

WHAT IT WRITES
    exp59_feats_{ds}_T{T}_s{seed}_stage{t}_r{rank}{order}_{TAG}.npz   for every t
    each containing Ftr and Fte over the FULL train and test sets in the feature space that
    exists AFTER task t. exp60 needs phi_{t-1} and phi_t on the same rows to fit the drift
    map, which is why full splits are cached at every stage rather than per-task slices.

COST AND DISK
    T trainings (task 0 at INIT_EPOCHS, the rest at EPOCHS) plus T full extractions.
    ~60 min and ~0.9 GB for IMAGENETR T=10. Stages are cached and skipped on resume.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot DS=IMAGENETR T=10 SEED=0 RANK=32 \
      python -u exp59_seq_lora.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_DS = os.environ.get("DS", "IMAGENETR").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])

import fsa_train as F                                                   # noqa: E402
import class_order as CO                                                # noqa: E402
from backbone import (advance_lora_task, freeze_non_lora,               # noqa: E402
                      get_lora_params, load_backbone)

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = F.TAG
DSETS, TS, SEEDS = _DS, _TS, _SEEDS
TARGETS = os.environ.get("TARGETS", "attn.qkv,attn.proj").split(",")
INIT_EPOCHS = int(os.environ.get("INIT_EPOCHS", 40))
EPOCHS = int(os.environ.get("EPOCHS", 20))
LR = float(os.environ.get("LR", 3e-4))
LR_LATER = float(os.environ.get("LR_LATER", LR))
ALPHA = float(os.environ.get("ALPHA", 4.0))
BS, GRAD_CLIP = 128, 1.0

_DIM = {"attn.qkv": 2304, "attn.proj": 768, "mlp.fc1": 3072, "mlp.fc2": 768}
_IN = {"attn.qkv": 768, "attn.proj": 768, "mlp.fc1": 768, "mlp.fc2": 3072}
_L, _D = 12, 768
PER_TASK_PER_RANK = _L * sum(_IN[t] + _DIM[t] for t in TARGETS)
GR_PER_TASK_PER_RANK = _L * 4 * 2 * _D                     # 73728
MATCH = int(os.environ.get("MATCH", 0))
RANK = (int(round(MATCH * GR_PER_TASK_PER_RANK / PER_TASK_PER_RANK)) if MATCH
        else int(os.environ.get("RANK", 32)))

if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}). Prefix with "
        f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 or set ALLOW_UNPINNED=1.")


def cache_path(ds, T, seed, t):
    return os.path.join(
        REPO, f"exp59_feats_{ds}_T{T}_s{seed}_stage{t}_r{RANK}{CO.order_tag()}_{TAG}.npz")


def run_cell(ds, T, seed):
    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = F.get_data(ds)
    cpt = n_cls // T
    order = CO.class_order(n_cls, seed)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    if all(os.path.exists(cache_path(ds, T, seed, t)) for t in range(T)):
        log(f"    all {T} stages cached")
        return {"skipped": True}

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = load_backbone(F.MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=RANK, lora_alpha=ALPHA,
                          lora_target_modules=TARGETS, lora_config="task_specific")
    freeze_non_lora(model)

    # ---- live CIL diagnostic. The per-task head accuracy printed during training is TRAIN
    # accuracy on the current task's own classes and says nothing about retention. These two
    # nearest-class-mean numbers cost ~nothing and preview exp60's Q2 while the run proceeds:
    #   ncm_stale   prototype frozen at the stage the class was FIRST SEEN -> pays for drift
    #   ncm_oracle  every prototype recomputed in the CURRENT space        -> no drift, needs
    #               old data, so it is an upper bound rather than a method
    # The gap between them IS the drift cost, visible live.
    protos_birth = {}

    def cil_diag(Ftr, Fte, upto):
        seen = np.concatenate(tasks[:upto + 1])
        ftr, fte = F.un(Ftr), F.un(Fte)
        for c in tasks[upto]:                       # freeze this task's prototypes at birth
            r = np.where(ytr == c)[0]
            if len(r):
                protos_birth[int(c)] = F.un(ftr[r].mean(0, keepdims=True))[0]
        tei = np.where(np.isin(yte, seen))[0]
        cls = [c for c in seen if int(c) in protos_birth]
        Pb = np.stack([protos_birth[int(c)] for c in cls])
        Po = np.stack([F.un(ftr[np.where(ytr == c)[0]].mean(0, keepdims=True))[0]
                       for c in cls])
        q = fte[tei]
        acc_s = float((np.asarray(cls)[(q @ Pb.T).argmax(1)] == yte[tei]).mean())
        acc_o = float((np.asarray(cls)[(q @ Po.T).argmax(1)] == yte[tei]).mean())
        return acc_s, acc_o, len(cls)

    stats = []
    for t in range(T):
        cp = cache_path(ds, T, seed, t)
        if t > 0:
            # Freeze every existing pair and allocate a fresh trainable one. From here on
            # only the newest adapter receives gradient; the backbone is W0 + sum_j B_j A_j.
            advance_lora_task(model)
        if os.path.exists(cp):
            # Cached stages still have to be REPLAYED through the model so the adapter stack
            # is in the right state for later stages. Nothing is retrained, but a resumed run
            # cannot skip the advance_lora_task() calls above.
            log(f"    stage {t}: cached, adapter advanced without training")
            stats.append({"task": t, "cached": True})
            continue

        idx = np.where(np.isin(ytr, tasks[t]))[0]
        remap = {int(c): i for i, c in enumerate(tasks[t])}
        nep = INIT_EPOCHS if t == 0 else EPOCHS
        lr = LR if t == 0 else LR_LATER
        trainable = [p for p in get_lora_params(model) if p.requires_grad]
        head = nn.Linear(model.num_features, cpt).to(DEV)
        opt = torch.optim.AdamW(trainable + list(head.parameters()), lr=lr,
                                weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nep)
        ce = nn.CrossEntropyLoss()
        ld = DataLoader(Subset(tr_aug, idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
        acc = 0.0
        for _ in range(nep):
            model.train()
            ok = tot = 0
            for x, lab in ld:
                x = x.to(DEV, non_blocking=True)
                y = torch.tensor([remap[int(v)] for v in lab], device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fx = model(x).float()
                    loss = ce(head(fx), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), GRAD_CLIP)
                opt.step()
                ok += int((head(fx).argmax(1) == y).sum()); tot += len(y)
            sch.step()
            acc = ok / max(tot, 1)
        n_train = sum(p.numel() for p in trainable)
        log(f"    stage {t}: {len(idx)} imgs, {nep} ep, trainable {n_train/1e6:.2f}M, "
            f"task acc {acc:.3f}")
        Ftr, Fte = F.extract(model, tr_ev), F.extract(model, te_ev)
        np.savez(cp, Ftr=Ftr, Fte=Fte, acc=np.array(acc))
        stats.append({"task": t, "n_img": int(len(idx)), "epochs": nep,
                      "train_acc": float(acc), "trainable_M": n_train / 1e6})
        del Ftr, Fte

    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return {"stages": stats}


if __name__ == "__main__":
    W = 92
    print("=" * W)
    print(f"EXP59 -- sequential per-task LoRA (UNPROTECTED), ORDER={CO.mode()}")
    print("=" * W)
    print(f"  targets {TARGETS}   rank {RANK}   alpha {ALPHA}")
    print(f"  params/task  ours {PER_TASK_PER_RANK*RANK/1e6:.2f} M"
          f"   GR-LoRA at r=64 {GR_PER_TASK_PER_RANK*64/1e6:.2f} M"
          f"   at r=16 {GR_PER_TASK_PER_RANK*16/1e6:.2f} M")
    print(f"  rank that MATCHES GR-LoRA r=64 on our targets: "
          f"{round(64*GR_PER_TASK_PER_RANK/PER_TASK_PER_RANK)}   (set MATCH=64)")
    print(f"  NOTE GR-LoRA's actual rank is NOT verified from any config we hold; the model "
          f"signature\n       defaults to 64. Report the budget, not a claimed parity.")
    out = {}
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = f"{ds}|{T}|{seed}|r{RANK}|{'+'.join(TARGETS)}{CO.order_tag()}"
                log(f"=== {key}   total over {T} tasks "
                    f"{PER_TASK_PER_RANK*RANK*T/1e6:.1f} M")
                out[key] = run_cell(ds, T, seed)
    json.dump(out, open(os.path.join(
        REPO, f"exp59_seq_lora{CO.order_tag()}_{TAG}.json"), "w"), indent=2)
    log("done -- run exp60_seq_drift.py for the analysis")
