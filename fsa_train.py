#!/usr/bin/env python3
"""fsa_train.py — shared first-session-adaptation harness for exp30/31/32.

WHY A SHARED MODULE
    exp30/31/32 change ONE thing each: the task-0 training objective. Everything else --
    data, LoRA config, optimiser, feature extraction, the RanPAC replay, the exp16 bar --
    must be bit-identical or the comparison is worthless. Three copies of a training loop
    would drift, and this project has already lost weeks to two copies of a solver
    disagreeing. One loop, one objective switch.

THE OBJECTIVES
    ce        LoRA + throwaway Linear(768, cpt), cross-entropy.  THE exp16 BASELINE.
              Must reproduce exp16's cached features exactly at the same recipe.
    ce_kd     ce + lam * (1 - cos(phi_lora(x), phi_frozen(x)))          [exp30]
              A proximal anchor to the PTM. exp12 found A_plus is single-peaked in task-0
              TRAIN accuracy (over-fitting task 0 costs later tasks); the 0.98 early-stop
              rule was a crude proxy for that and was falsified (10/12 cells worse). This is
              the continuous version of the same intent.
    cosine    LoRA + cosine head, logits = s * cos(phi, w_c)            [exp32]
              A linear softmax head optimises for linear separability with per-class bias
              and scale; the deployed reader (NCM / RanPAC / cone) wants compact equidistant
              clusters. `proto` uses batch class means instead of learnable w_c.
    subspace  DSN-style episodic subspace classifier                    [exp31]
              Split each class's batch into support/query, orthonormalise phi(support) by QR,
              score queries by ||B_c^T phi(q)||. Trains phi so that R samples of a class SPAN
              the class -- which is exactly what exp29 measured as the dominant per-class
              error predictor (own-subspace energy, rho -0.585, strengthening with k) while
              subspace OVERLAP was noise (+0.037). QR not SVD: we need the span, and QR is
              stable where SVD gradients blow up on near-degenerate spectra.

CACHE KEYS CARRY THE FULL RECIPE
    exp16's results key once omitted ep/lr/aug/target and silently reported stale numbers
    under a new recipe. Every cache name here embeds the objective and its hyperparameters.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Dataset, Subset

import timm
from timm.data import create_transform, resolve_model_data_config

from backbone import load_backbone, freeze_non_lora, get_lora_params
from splits import split_indices

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
TAG = MODEL.split(".")[-1]
EPOCHS = int(os.environ.get("EPOCHS", 40))
LR = float(os.environ.get("LR", 3e-4))
AUG = int(os.environ.get("AUG", 1))
BS, GRAD_CLIP = 128, 1.0
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SPLIT_SEED = 1993

_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
TF_EVAL = create_transform(**_cfg, is_training=False)
TF_TRAIN = (create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                             re_prob=0.25, scale=(0.7, 1.0), hflip=0.5) if AUG else TF_EVAL)


# ------------------------------------------------------------------ data (from exp16)
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
    """(train_aug, train_eval, ytr, test_eval, yte, n_classes) -- exp16's splits verbatim."""
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
    elif name in ("IMAGENETA", "IMAGENETAP"):
        # IMAGENETAP: stratified per-class 80/20. See splits.py and exp16_full_table.get_data.
        d = load_dataset("barkermrl/imagenet-a",
                         cache_dir=os.path.join(REPO, "data/hf"))["train"]
        lab = np.array(d["label"])
    elif name == "CUB200":
        dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
        tr, te = dd["train"], dd["test"]
        ytr, yte = np.array(tr["label"]), np.array(te["label"])
        n = int(max(ytr.max(), yte.max())) + 1
        return (HFWrap(tr, np.arange(len(ytr)), ytr, TF_TRAIN),
                HFWrap(tr, np.arange(len(ytr)), ytr, TF_EVAL), ytr,
                HFWrap(te, np.arange(len(yte)), yte, TF_EVAL), yte, n)
    elif name == "CUB200P":
        # PyCIL / LAMDA-PILOT CUB split: all 11,788 images re-split 80/20 -> 9430 / 2358.
        # See the long note in exp16_full_table.get_data; kept as a separate dataset name
        # so no cached CUB200 feature file or results key changes meaning.
        from datasets import concatenate_datasets
        dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
        d = concatenate_datasets([dd["train"], dd["test"]])
        lab = np.array(d["label"])
    else:
        raise ValueError(name)
    tri, tei = split_indices(name, lab)
    return (HFWrap(d, tri, lab[tri], TF_TRAIN), HFWrap(d, tri, lab[tri], TF_EVAL), lab[tri],
            HFWrap(d, tei, lab[tei], TF_EVAL), lab[tei], int(lab.max()) + 1)


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


# ------------------------------------------------------------------ heads / losses
class CosHead(nn.Module):
    """logits = s * cos(phi, w_c). `proto=True` replaces the learnable w_c with the batch
    class means (prototypical networks), so the head carries no parameters at all."""
    def __init__(self, dim, n_out, scale=16.0, proto=False):
        super().__init__()
        self.proto, self.s = proto, scale
        self.w = None if proto else nn.Parameter(torch.randn(n_out, dim) * 0.02)
        self.n_out = n_out

    def forward(self, f, y=None):
        fn = Fn.normalize(f, dim=1)
        if not self.proto:
            return self.s * fn @ Fn.normalize(self.w, dim=1).T
        # class means of THIS batch; classes absent from the batch get -inf and are masked
        P = torch.zeros(self.n_out, f.shape[1], device=f.device, dtype=f.dtype)
        P.index_add_(0, y, fn)
        cnt = torch.zeros(self.n_out, device=f.device, dtype=f.dtype).index_add_(
            0, y, torch.ones_like(y, dtype=f.dtype))
        keep = cnt > 0
        P[keep] = Fn.normalize(P[keep] / cnt[keep, None], dim=1)
        out = self.s * fn @ P.T
        out[:, ~keep] = -1e4
        return out


def subspace_logits(f, y, n_out, r_sup, scale, gen):
    """DSN-style: per class, QR-orthonormalise the support features and score every query by
    ||B_c^T phi(q)||. Returns (logits, query_mask). Classes with < r_sup+1 samples in the
    batch are skipped and masked out -- scoring them would use a rank-deficient basis."""
    fn = Fn.normalize(f, dim=1)
    logits = torch.full((f.shape[0], n_out), -1e4, device=f.device, dtype=f.dtype)
    is_q = torch.zeros(f.shape[0], dtype=torch.bool, device=f.device)
    bases = {}
    for c in y.unique().tolist():
        idx = (y == c).nonzero(as_tuple=True)[0]
        if len(idx) < r_sup + 1:
            continue
        perm = idx[torch.randperm(len(idx), generator=gen, device=f.device)]
        sup, qry = perm[:r_sup], perm[r_sup:]
        # QR gives an orthonormal basis of span(support). We need the SPAN, and QR is stable
        # where SVD gradients blow up on near-degenerate spectra.
        B, _ = torch.linalg.qr(fn[sup].T)              # (d, r_sup)
        bases[c] = B
        is_q[qry] = True
    if not bases:
        return None, None
    for c, B in bases.items():
        logits[:, c] = scale * torch.linalg.norm(fn @ B, dim=1)
    return logits, is_q


# ------------------------------------------------------------------ task-0 training
def train_task0(ds_name, T, seed, objective="ce", tag_extra="", *, lam_kd=0.0,
                head_scale=16.0, proto=False, r_sup=4, epochs=None, lr=None):
    """Train task 0 under `objective`, FREEZE, extract features for everything, cache.

    Returns (Ftr, Fte, acc0). The cache name embeds the objective and every hyperparameter
    that changes the result -- omitting one is how exp16 silently reported stale numbers.
    """
    ep = EPOCHS if epochs is None else epochs
    lrate = LR if lr is None else lr
    tag = f"{ds_name}_T{T}_s{seed}_ep{ep}_lr{lrate:g}_aug{AUG}_{objective}{tag_extra}"
    cache = os.path.join(REPO, f"fsa_feats_{tag}_{TAG}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"  cached {tag}")
        return z["Ftr"], z["Fte"], float(z["acc0"])

    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = get_data(ds_name)
    cpt = n_cls // T
    torch.manual_seed(seed)
    np.random.seed(seed)
    task0 = np.random.default_rng(seed).permutation(n_cls)[:cpt]
    idx = np.where(np.isin(ytr, task0))[0]
    remap = {int(c): i for i, c in enumerate(task0)}

    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))

    teacher = None
    if objective == "ce_kd" and lam_kd > 0:
        teacher = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                                lora_rank=0).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    if objective in ("ce", "ce_kd"):
        head = nn.Linear(model.num_features, cpt).to(DEV)
    elif objective == "cosine":
        head = CosHead(model.num_features, cpt, head_scale, proto).to(DEV)
    elif objective == "subspace":
        head = nn.Identity().to(DEV)
    else:
        raise ValueError(objective)

    params = lp + [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lrate, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    ce = nn.CrossEntropyLoss()
    ld = DataLoader(Subset(tr_aug, idx.tolist()), batch_size=BS, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=(objective == "subspace"))
    gen = torch.Generator(device=DEV).manual_seed(seed)
    acc0 = 0.0
    for e in range(ep):
        model.train()
        ok = tot = 0
        for x, lab in ld:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x).float()
                if objective == "subspace":
                    logits, is_q = subspace_logits(f, y, cpt, r_sup, head_scale, gen)
                    if logits is None or is_q.sum() == 0:
                        continue
                    loss = ce(logits[is_q], y[is_q])
                    pred, ytgt = logits[is_q].argmax(1), y[is_q]
                else:
                    logits = head(f, y) if objective == "cosine" else head(f)
                    loss = ce(logits, y)
                    pred, ytgt = logits.argmax(1), y
                    if teacher is not None:
                        with torch.no_grad():
                            ft = teacher(x).float()
                        loss = loss + lam_kd * (1.0 - Fn.cosine_similarity(f, ft, dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            ok += int((pred == ytgt).sum())
            tot += len(ytgt)
        sch.step()
        acc0 = ok / max(tot, 1)
    Ftr, Fte = extract(model, tr_ev), extract(model, te_ev)
    del model, teacher
    torch.cuda.empty_cache()
    np.savez(cache, Ftr=Ftr, Fte=Fte, acc0=np.array(acc0))
    log(f"  trained {tag}: task0-acc {acc0:.3f}")
    return Ftr, Fte, acc0


# ------------------------------------------------------------------ RanPAC replay (exp16)
def replay(Ftr, ytr, Fte, yte, T, seed, n_cls):
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    Ztr, Zte = un(Ftr), un(Fte)
    P = torch.randn(Ztr.shape[1], M_RP,
                    generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.tensor(X[i:i + bs], device=DEV,
                                             dtype=torch.float32) @ P)

    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    accs, nval = [], 0
    for t in range(T):
        for i, h in _H(Ztr[FIT[t]]):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval += len(VAL[t])
        tei = np.where(np.isin(yte, seen))[0]

        def acc(W, X, y):
            L = torch.cat([(h.double() @ W) for _, h in _H(X)]).cpu().numpy()
            return float((np.asarray(seen)[L[:, seen].argmax(1)] == y).mean())

        best, ba = -1.0, -1.0
        for lam in LAMBDAS:
            W = torch.linalg.solve(G + lam * eye, C)
            a = acc(W, Ztr[VAL_ALL[:nval]], ytr[VAL_ALL[:nval]])
            if a > best:
                best, ba = a, acc(W, Zte[tei], yte[tei])
        accs.append(ba)
    del G, C, P, eye
    torch.cuda.empty_cache()
    return accs


def bar_for(ds, T, seed):
    """exp16's paired A_plus cell, so every run prints what it has to beat."""
    import json
    p = os.path.join(REPO, f"exp16_full_table_{TAG}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
