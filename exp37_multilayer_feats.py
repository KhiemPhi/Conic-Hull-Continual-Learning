#!/usr/bin/env python3
"""exp37_multilayer_feats.py — read MORE of the frozen backbone, at zero training cost.

THE ARGUMENT
    Every number in this project reads ONE vector per image: the final-block CLS token.
    A ViT-B/16 has 12 blocks and 196 patch tokens per image; all of that is computed on
    every forward pass and then thrown away.

    This is the only representation lever that is NOT bounded by the 29% accumulation
    cap, because it is not about accumulation at all. The 71/29 decomposition says
    cross-task losses can reach at most 29% of the remaining 3.3 points of feature
    headroom (~0.96). Reading more of an ALREADY-adapted, ALREADY-frozen backbone sits
    outside that budget entirely -- it is extraction, not learning.

    It is also the cheapest thing available: task 0 is trained ONCE and every variant is
    extracted from the same forward pass.

WHAT IS EXTRACTED
    Forward hooks on model.blocks[i] capture the full token sequence (B, 1+P, C) before
    the final norm. From each captured block we form two pooled vectors:
        cls_i  = tokens[:, 0]                 the class token at depth i
        gap_i  = tokens[:, 1:].mean(1)        mean over patch tokens at depth i
    Hooks rather than timm's get_intermediate_layers: the LoRA wrapper may not forward
    that method, and hooks work on any nn.Module regardless of the timm version.

    EACH COMPONENT IS L2-NORMALISED BEFORE CONCATENATION. Deep and shallow blocks differ
    in scale by a large factor (no final norm is applied to intermediate blocks), so a
    raw concat would let one depth dominate the cosine geometry that every downstream
    arm -- whitener, k-means, cone, RanPAC -- is built on. Normalising per component
    makes the concatenation a genuine direct sum of directions.

VARIANTS (v = the concatenation, d = 768 * n_components)
    cls          cls_12                            768   BASELINE, must reproduce exp16
    cls_gap      cls_12 + gap_12                  1536   patch tokens, same depth
    multi_cls    cls_12 + cls_9 + cls_6           2304   depth, class tokens only
    multi_gap    gap_12 + gap_9 + gap_6           2304   depth, patch tokens only
    full         cls_12 + gap_12 + cls_9 + gap_9  3072   both axes

REGRESSION CHECK -- the whole file is worthless without it
    The `cls` variant re-derives exp16's features from a freshly trained task 0 and must
    land on the known bar (ImageNet-R T=10 s0: ranpac 80.28). Task-0 training is seeded,
    so this should match closely; a large drift means the retrain does not reproduce
    exp16 and no variant comparison below it means anything.

WHY THE REPLAY IS DUPLICATED HERE RATHER THAN IMPORTED FROM exp35
    exp35's results key does not encode the feature variant. Importing it and swapping
    E.adapted_features would silently collide five variants onto one cache key -- the
    exact skip-key bug this repo has already been bitten by twice. Separate file,
    separate json, variant in the key.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp37_multilayer_feats.py
    DS=IMAGENETR T=10 SEED=0 VARIANTS=cls,cls_gap python -u exp37_multilayer_feats.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset

import exp19_dataset_hull as E
import fsa_train as F
from backbone import freeze_non_lora, get_lora_params, load_backbone
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]

# blocks to hook (0-indexed); 11 is the last block of a 12-block ViT-B
BLOCKS = [5, 8, 11]
SPECS = {
    "cls":       [("cls", 11)],
    "cls_gap":   [("cls", 11), ("gap", 11)],
    "multi_cls": [("cls", 11), ("cls", 8), ("cls", 5)],
    "multi_gap": [("gap", 11), ("gap", 8), ("gap", 5)],
    "full":      [("cls", 11), ("gap", 11), ("cls", 8), ("gap", 8)],
}
VARIANTS = os.environ.get("VARIANTS", ",".join(SPECS)).split(",")
assert all(v in SPECS for v in VARIANTS), f"unknown variant; pick from {list(SPECS)}"

R = int(os.environ.get("R", 32))                   # exp35's ImageNet-R recipe
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
ITERS = int(os.environ.get("ITERS", 500))
EPOCHS = int(os.environ.get("EPOCHS", 40))
LR = float(os.environ.get("LR", 3e-4))
OUT = os.path.join(REPO, f"exp37_multilayer_feats_{TAG}.json")


def un(A):
    return np.asarray(A, np.float32) / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def zs(A, seen):
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


# ---------------------------------------------------------------- extraction
@torch.no_grad()
def extract_all(model, ds, rows=None):
    """One forward pass -> {(kind, block): (N, C)} for every hooked block.

    Returns raw (unnormalised) pools; normalisation happens per-variant in assemble()
    so the same forward pass serves all five variants.
    """
    model.eval()
    cap = {}
    hs = [model.blocks[b].register_forward_hook(
        lambda m, i, o, b=b: cap.__setitem__(b, o if torch.is_tensor(o) else o[0]))
        for b in BLOCKS]
    sub = ds if rows is None else Subset(ds, np.asarray(rows).tolist())
    ld = DataLoader(sub, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    acc = {(k, b): [] for b in BLOCKS for k in ("cls", "gap")}
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for x, _ in ld:
                cap.clear()
                model(x.to(DEV, non_blocking=True))
                for b in BLOCKS:
                    t = cap[b].float()
                    assert t.ndim == 3, f"block {b} output is {t.shape}, expected (B,N,C)"
                    acc[("cls", b)].append(t[:, 0].cpu().numpy())
                    acc[("gap", b)].append(t[:, 1:].mean(1).cpu().numpy())
    finally:
        for h in hs:
            h.remove()
    return {k: np.concatenate(v, 0) for k, v in acc.items()}


def assemble(pools, spec):
    """L2-normalise each component, then concatenate. See the header on why."""
    return np.concatenate([un(pools[(kind, blk)]) for kind, blk in spec], 1)


def train_task0_pools(ds_name, T, seed):
    """Train task 0 exactly as fsa_train does, then extract every pooled component once.

    Cached, because the training is the expensive part and all five variants share it.
    """
    cache = os.path.join(REPO, f"exp37_pools_{ds_name}_T{T}_s{seed}"
                               f"_ep{EPOCHS}_lr{LR:g}_{TAG}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"  cached pools {ds_name} T{T} s{seed}  (acc0 {float(z['acc0']):.4f})")
        return ({tuple(k.split("|")[:1] + [int(k.split("|")[1])]): z[k]
                 for k in z.files if "|" in k}, float(z["acc0"]))

    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = F.get_data(ds_name)
    cpt = n_cls // T
    torch.manual_seed(seed)
    np.random.seed(seed)
    task0 = np.random.default_rng(seed).permutation(n_cls)[:cpt]
    idx = np.where(np.isin(ytr, task0))[0]
    remap = {int(c): i for i, c in enumerate(task0)}

    model = load_backbone(F.MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    head = nn.Linear(model.num_features, cpt).to(DEV)
    params = list(get_lora_params(model)) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    ld = DataLoader(Subset(tr_aug, idx.tolist()), batch_size=F.BS, shuffle=True,
                    num_workers=8, pin_memory=True)
    acc0 = 0.0
    for e in range(EPOCHS):
        model.train()
        ok = tot = 0
        for x, lab in ld:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = head(model(x).float())
                loss = ce(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, F.GRAD_CLIP)
            opt.step()
            ok += int((logit.argmax(1) == y).sum())
            tot += len(y)
        sch.step()
        acc0 = ok / max(tot, 1)
        if e % 10 == 0 or e == EPOCHS - 1:
            log(f"    ep{e:3d}  task0 train acc {acc0:.4f}")

    ptr = extract_all(model, tr_ev)
    pte = extract_all(model, te_ev)
    pools = {("tr",) + k: v for k, v in ptr.items()}
    pools.update({("te",) + k: v for k, v in pte.items()})
    np.savez_compressed(cache, acc0=acc0,
                        **{f"{s}|{k}|{b}": v for (s, k, b), v in pools.items()})
    del model, head
    torch.cuda.empty_cache()
    return pools, acc0


# ---------------------------------------------------------------- replay
def km(X, k, seed):
    k = int(min(k, len(X)))
    return un(X.mean(0, keepdims=True) if k <= 1 else
              KMeans(k, n_init=4, random_state=seed).fit(X).cluster_centers_)


def cone_score(A, Q):
    h = ConicHull(n_rays=len(A), nnls_iters=ITERS)
    h.extreme_rays_ = un(A)
    return h.score(Q)


def replay(Ztr, Zte, ytr, yte, n_cls, T, seed):
    """exp35's staged replay, reduced to the four arms this question needs."""
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    Qv, Qt = Ztr[VAL_ALL], Zte

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.as_tensor(X[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def logits(X, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(X)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    Aorig = {}
    res = {a: [] for a in ("ranpac", "ncm", "cone_km", "fuse_km")}

    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        S = scatter / max(n_scat, 1)
        S = S + SHRINK * np.trace(S) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Aorig[c] = km(un(Ztr[r] @ Wh), R, c) @ Wh_inv

        for i, h in _H(un(Ztr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        yv = ytr[VAL_ALL[:nval]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(Z, y):
            return float((np.asarray(seen)[Z[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(logits(un(Qv[:nval]), Wm), yv)
            if a > best:
                best, bw = a, Wm
        zLv, zLt = zs(logits(un(Qv[:nval]), bw), seen), zs(logits(un(Qt), bw)[tei], seen)

        Qvw, Qtw = un(Qv[:nval] @ Wh), un(Qt[tei] @ Wh)
        Cv = np.full((nval, n_cls), -np.inf, np.float32)
        Ct = np.full((len(tei), n_cls), -np.inf, np.float32)
        NCt = np.full((len(tei), n_cls), -np.inf, np.float32)
        for c in seen:
            if c not in Aorig:
                continue
            Ac = un(Aorig[c] @ Wh)
            Cv[:, c] = cone_score(Ac, Qvw)
            Ct[:, c] = cone_score(Ac, Qtw)
            NCt[:, c] = Qtw @ un(Ac.mean(0, keepdims=True))[0]

        miss = [c for c in seen if c not in Aorig]
        zSv, zSt = zs(Cv, seen), zs(Ct, seen)
        if miss:
            zSv[:, miss] = 0.0
            zSt[:, miss] = 0.0
        b = max(BETAS, key=lambda bb: acc(zLv + bb * zSv, yv))

        res["ranpac"].append(acc(zLt, yt))
        res["ncm"].append(acc(zs(NCt, seen), yt))
        res["cone_km"].append(acc(zs(Ct, seen), yt))
        res["fuse_km"].append(acc(zLt + b * zSt, yt))
        log(f"      s{t}: " + "  ".join(f"{a} {res[a][-1]*100:.2f}" for a in res))

    del G, C, P, eye
    torch.cuda.empty_cache()
    return {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
            for a, v in res.items()}


# ---------------------------------------------------------------- driver
allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            todo = [v for v in VARIANTS
                    if f"{ds}|{T}|{seed}|{v}|R{R}_s{SHRINK:g}_m{M_RP}_ep{EPOCHS}|v1"
                    not in allres]
            if not todo:
                log(f"skip {ds} T{T} s{seed} (all variants cached)"); continue
            log(f"=== {ds} T{T} s{seed}  training task 0 once for {len(todo)} variant(s)")
            pools, acc0 = train_task0_pools(ds, T, seed)
            E.T, E.SEED = T, seed
            ytr, yte, n_cls = E.get_labels(ds)
            for v in todo:
                spec = SPECS[v]
                Ztr = assemble({(k, b): pools[("tr", k, b)] for k, b in spec}, spec)
                Zte = assemble({(k, b): pools[("te", k, b)] for k, b in spec}, spec)
                key = f"{ds}|{T}|{seed}|{v}|R{R}_s{SHRINK:g}_m{M_RP}_ep{EPOCHS}|v1"
                log(f"  --- {v}  d={Ztr.shape[1]}  ({len(spec)} components)")
                blob = replay(un(Ztr), un(Zte), ytr, yte, n_cls, T, seed)
                blob["d"], blob["acc0"], blob["spec"] = Ztr.shape[1], acc0, \
                    [f"{k}{b}" for k, b in spec]
                allres[key] = blob
                json.dump(allres, open(OUT, "w"), indent=2)

W = 96
print("\n" + "=" * W)
print("EXP37 — how much does reading more of the frozen backbone buy?")
print("=" * W)
for key, r in sorted(allres.items()):
    ds, T, seed, v = key.split("|")[:4]
    base = allres.get(f"{ds}|{T}|{seed}|cls|" + key.split("|", 4)[4])
    dl = (f"{(r['fuse_km']['A_last']-base['fuse_km']['A_last'])*100:>+8.2f}"
          if base else f"{'--':>8}")
    print(f"{ds:<10}T{T:<3}s{seed}  {v:<10} d{r['d']:<5}"
          f" ranpac {r['ranpac']['A_last']*100:>6.2f}"
          f"  cone {r['cone_km']['A_last']*100:>6.2f}"
          f"  fuse {r['fuse_km']['A_last']*100:>6.2f}/{r['fuse_km']['A_avg']*100:<6.2f}"
          f" | vs cls {dl}")
print("-" * W)
print("REGRESSION: the `cls` variant retrains task 0 from scratch and must land on the")
print("   known bar (ImageNet-R T=10 s0: ranpac 80.28). Task-0 training is seeded, so a")
print("   drift beyond ~0.3 means the retrain does not reproduce exp16 and every variant")
print("   comparison below it is meaningless. Check this column FIRST.")
print("Read `ranpac` and `cone` separately: a variant can help the ridge head (more")
print("   dimensions for the random projection to exploit) while doing nothing for the")
print("   cone geometry, or the reverse. Only `fuse` decides the headline.")
print("Components are L2-normalised BEFORE concatenation -- intermediate blocks get no")
print("   final norm and would otherwise dominate or vanish by scale alone.")
print("This lever is NOT bounded by the 29% accumulation cap: it is extraction from an")
print("   already-adapted frozen backbone, not cross-task learning.")
print("=" * W)
print(f"wrote {OUT}")
