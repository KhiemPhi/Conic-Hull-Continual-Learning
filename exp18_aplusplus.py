#!/usr/bin/env python3
"""exp18_aplusplus.py — A++: keep adapting after task 0, but on a DECAYING budget.

THE GAP THIS FILLS
    Only the two endpoints of the adaptation axis have ever been measured:
        freeze_after=0    (A_plus)          0.8027 / 0.8552   [ImageNet-R aug40 lr1e-4]
        freeze_after=None (full adapt)      0.7903 / 0.8416
    1.2 points apart, and nothing in between has been sampled. exp12 separately showed the
    backbone has an INTERIOR optimum (accuracy is single-peaked in task-0 train accuracy at
    ~0.98, falling on both sides), so there is no reason to believe the optimum for LATER
    stages is exactly zero -- that is simply the only value tested.

WHY A DECAYING SCHEDULE SPECIFICALLY
    At task t, adapting buys exposure to cpt new classes on top of the t*cpt already seen,
    so the marginal DIVERSITY gain falls ~1/t. The cost is that drift invalidates the
    accumulated statistics of t*cpt old classes, which grows ~t. Benefit/cost ~ 1/t^2, so the
    budget should decay steeply. Hence e_t = round(E0 * gamma^t), swept in gamma:
        gamma=0    -> A_plus exactly (0 epochs for every t>=1)
        gamma=0.5  -> 20,10,5,2,1,1,1,0,0   (~39 extra epochs total, ~= one more task-0)
        gamma=1    -> full adaptation, the other endpoint
    A one-dimensional interpolation between two measured endpoints, with a shape derived
    from the cost model rather than guessed.

WHAT MUST BE MEASURED, OR THE RESULT IS UNINTERPRETABLE
    Adapting breaks the property that makes A_plus special: exp14 verified
    ||G_accum - G_joint||/||G|| = 2.9e-15, i.e. the head has ZERO incremental penalty only
    while the backbone is frozen. So every arm reports three things:
      acc     accum head, drifting statistics       -- what you would actually ship
      oracle  head rebuilt from ALL real seen data in the CURRENT frame -- isolates whether
              the BACKBONE improved, independent of what drift did to the head
      drift   mean cosine( phi_t(x), phi_0(x) ) on a fixed probe set
    The decisive pattern: if oracle RISES while acc FALLS, tiny adaptation is improving the
    representation and the head just needs refreshing -- which is exp15's exemplars, not a
    reason to abandon adaptation. Without the oracle column a null is unreadable.

HONEST PRIOR
    Probably monotone decreasing in gamma -- this project's whole history says continued
    adaptation is net-negative, and protection losses converge DOWN to freezing. Two things
    make it worth the compute anyway: the endpoints are only 1.2 apart so the downside is
    bounded, and the fit-curve result says the backbone's optimal adaptation is an interior
    quantity. Also note later tasks train with TASK-WISE CE (cpt-way head, no replay), which
    is precisely the objective that never asks class 5 vs class 150 -- so A++ adds
    representation diversity WITHOUT adding cross-task discrimination. That asymmetry is the
    most likely reason for it to fail, and it is worth knowing.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR GAMMAS=0,0.25,0.5,0.75,1.0 python -u exp18_aplusplus.py     # ~3 h
    DS=IMAGENETR GAMMAS=0.5 ORACLE=0 python -u exp18_aplusplus.py            # ~25 min
    DS=IMAGENETA GAMMAS=0,0.5,1.0 python -u exp18_aplusplus.py
"""
import json
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

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


DEV = "cuda"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 1))
E0 = int(os.environ.get("E0", 40))
LR = float(os.environ.get("LR", 3e-4))
LR_LATER = float(os.environ.get("LR_LATER", LR))   # separate axis: gentler lr, same epochs
GAMMAS = [float(g) for g in os.environ.get("GAMMAS", "0,0.25,0.5,0.75,1.0").split(",")]
BS, GRAD_CLIP, AUG = 128, 1.0, int(os.environ.get("AUG", 1))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
ORACLE = int(os.environ.get("ORACLE", 1))
# XCE: what loss later tasks use.
#   none   cpt-way CE over the CURRENT task only. This is the default and the problem --
#          it never asks the backbone to separate class 5 from class 150, which is where
#          88.7% of the remaining error lives (exp14).
#   proto  cosine CE over ALL SEEN classes, with old-class rows FROZEN at the unit-norm
#          class mean recorded in that class's birth frame. Current-task images then get a
#          real gradient pushing them AWAY from old-class directions: cross-task
#          discrimination with zero stored images, at 768 floats = 3 KB per class
#          (0.6 MB for 200 classes, vs the 473 MB the compared methods spend).
#          Caveat worth measuring rather than assuming: mu_c is frozen in the birth frame,
#          so it goes stale exactly as fast as the backbone drifts -- which is why this
#          belongs in the SAME sweep as gamma rather than a separate one.
XCE = os.environ.get("XCE", "none")             # none | proto
XCE_SCALE = float(os.environ.get("XCE_SCALE", 16.0))
N_PROBE = 512
SPLIT_SEED = 1993
REPO = os.path.dirname(os.path.abspath(__file__))
# Bump when the METHOD changes. v1 = learnable per-task head (degenerate: random-head noise
# at small ep, and a learnable current row that shunted the cross-task gradient away from
# phi). v2 = fixed cosine head for t>0, seeded-before-backbone. Without this the resume logic
# happily reports v1 rows next to v2 rows in the same table.
METHOD_VERSION = "v2"
OUT = os.path.join(REPO,
                   f"exp18_aplusplus_{METHOD_VERSION}_{DS}_T{T}_s{SEED}"
                   + (f"_{XCE}" if XCE != "none" else "") + ".json")

_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
TF_EVAL = create_transform(**_cfg, is_training=False)
TF_TRAIN = (create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                             re_prob=0.25, scale=(0.7, 1.0), hflip=0.5) if AUG else TF_EVAL)


def epochs_at(t, gamma):
    """e_0 = E0 always; e_t = round(E0 * gamma^t) for t>=1. gamma=0 reproduces A_plus."""
    if t == 0:
        return E0
    return int(max(0, round(E0 * (gamma ** t))))


# ------------------------------------------------------------------ data (mirrors exp16)
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


def get_data():
    from datasets import load_dataset
    if DS == "CIFAR100":
        from torchvision import datasets as tvd
        tr = tvd.CIFAR100(os.path.join(REPO, "data"), train=True, download=False)
        te = tvd.CIFAR100(os.path.join(REPO, "data"), train=False, download=False)
        ytr, yte = np.array(tr.targets), np.array(te.targets)
        return (TVWrap(tr, ytr, TF_TRAIN), TVWrap(tr, ytr, TF_EVAL), ytr,
                TVWrap(te, yte, TF_EVAL), yte, 100)
    if DS == "CUB200":
        dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
        tr, te = dd["train"], dd["test"]
        ytr, yte = np.array(tr["label"]), np.array(te["label"])
        n = int(max(ytr.max(), yte.max())) + 1
        return (HFWrap(tr, np.arange(len(ytr)), ytr, TF_TRAIN),
                HFWrap(tr, np.arange(len(ytr)), ytr, TF_EVAL), ytr,
                HFWrap(te, np.arange(len(yte)), yte, TF_EVAL), yte, n)
    if DS == "IMAGENETR":
        d = load_dataset("axiong/imagenet-r", cache_dir=os.path.join(REPO, "data/hf"))["test"]
        w = np.array(d["wnid"])
        lab = np.searchsorted(np.array(sorted(set(w.tolist()))), w)
    elif DS == "IMAGENETA":
        d = load_dataset("barkermrl/imagenet-a",
                         cache_dir=os.path.join(REPO, "data/hf"))["train"]
        lab = np.array(d["label"])
    else:
        raise ValueError(DS)
    p = np.random.default_rng(SPLIT_SEED).permutation(len(lab))
    n_tr = int(0.8 * len(lab))
    tri, tei = p[:n_tr], p[n_tr:]
    return (HFWrap(d, tri, lab[tri], TF_TRAIN), HFWrap(d, tri, lab[tri], TF_EVAL), lab[tri],
            HFWrap(d, tei, lab[tei], TF_EVAL), lab[tei], int(lab.max()) + 1)


TR_AUG, TR_EV, YTR, TE_EV, YTE, N_CLS = get_data()
CPT = N_CLS // T
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(T)]
FIT, VAL = [], []
for t in range(T):
    idx = np.where(np.isin(YTR, TASKS[t]))[0]
    pm = np.random.default_rng(t).permutation(len(idx))
    nv = max(int(0.1 * len(idx)), 1)
    VAL.append(idx[pm[:nv]]); FIT.append(idx[pm[nv:]])
PROBE = np.random.default_rng(0).permutation(len(YTE))[:N_PROBE]
log(f"{DS} T={T} cpt={CPT} seed={SEED} | train {len(YTR)} test {len(YTE)} | E0={E0} "
    f"lr={LR:g} lr_later={LR_LATER:g}")


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


@torch.no_grad()
def extract(model, ds, rows):
    model.eval()
    ld = DataLoader(Subset(ds, np.asarray(rows).tolist()), batch_size=256, shuffle=False,
                    num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in ld:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


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


def solve_eval(G, C, Zv, yv, Zt, yt, seen):
    seen_t = torch.tensor(np.asarray(seen), device=DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

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
    return acc(bw, Zt, yt)


def run(gamma):
    sched = [epochs_at(t, gamma) for t in range(T)]
    log(f"=== gamma={gamma:g}  epoch schedule {sched}  (total extra {sum(sched[1:])})")
    # Seed BEFORE load_backbone. inject_lora draws lora_A from the global torch RNG, so
    # seeding afterwards leaves the initialisation at the mercy of whatever consumed the
    # stream earlier -- which differs between the first and second gamma in a single process,
    # and differs again between exp12/exp16/exp18. That is worth ~0.5 A-Last and it destroys
    # the pairing this sweep depends on: every arm must start from the SAME backbone.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV, lora_rank=32,
                          lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))

    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    vZ, vy, accs, orcs, drifts = [], [], [], [], []
    PROTO = {}          # class -> unit-norm birth-frame mean (768 floats = 3 KB/class)
    F0 = None

    for t in range(T):
        ep = sched[t]
        cls = TASKS[t]
        use_xce = (XCE == "proto" and t > 0 and len(PROTO) > 0 and ep > 0)
        if ep > 0:
            lr_t = LR if t == 0 else LR_LATER
            if t == 0:
                # UNCHANGED from A_plus: learnable linear head. Task 0 must stay bit-identical
                # or gamma=0 stops reproducing the exp16 reference and there is no baseline.
                head = nn.Linear(768, CPT).to(DEV)
                remap = {int(c): i for i, c in enumerate(cls)}
                head_params = list(head.parameters())

                def logit_fn(f):
                    return head(f)
            else:
                # LATER TASKS: FIXED cosine head, NOTHING learnable but the LoRA params.
                # Two degeneracies this removes, both of which made the gamma sweep measure
                # the wrong thing:
                #   (a) a freshly RANDOM linear head + fresh Adam state means that at ep=1-2
                #       (~17 steps) the backbone is chasing a classifier that has not learned
                #       anything yet -- "train a little" became "inject a little noise",
                #       which corrupts exactly the small-gamma regime this sweep exists for.
                #   (b) with XCE=proto, a LEARNABLE current-class row is a shunt: CE over all
                #       seen classes is minimised more cheaply by moving W_cur away from the
                #       frozen W_old than by moving phi. The cross-task pressure never
                #       reached the representation.
                # Fixing every row makes the loss meaningful from step 1 and sends every
                # gradient into phi, which is what "adapt the backbone a bit" was meant to be.
                Zi = un(extract(model, TR_EV, FIT[t]))
                yi = YTR[FIT[t]]
                W_cur = torch.tensor(un(np.stack([Zi[yi == c].mean(0) for c in cls])),
                                     device=DEV, dtype=torch.float32)
                if use_xce:
                    old_c = sorted(PROTO.keys())
                    W_old = torch.tensor(np.stack([PROTO[c] for c in old_c]),
                                         device=DEV, dtype=torch.float32)
                    Wfix = torch.cat([W_old, W_cur], 0)
                    remap = {c: i for i, c in enumerate(old_c + [int(c) for c in cls])}
                else:
                    Wfix = W_cur
                    remap = {int(c): i for i, c in enumerate(cls)}
                Wfix = Fn.normalize(Wfix, dim=1).detach()
                head_params = []

                def logit_fn(f, _W=Wfix):
                    return XCE_SCALE * (Fn.normalize(f, dim=1) @ _W.T)
            opt = torch.optim.AdamW(lp + head_params, lr=lr_t, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(ep, 1))
            ce = nn.CrossEntropyLoss()
            if t > 0:
                log(f"    [t={t}] FIXED cosine head over {len(remap)} classes "
                    + (f"({len(PROTO)} frozen old prototypes + {len(cls)} current means, "
                       f"cross-task)" if use_xce else f"({len(cls)} current means, "
                       f"current-task only)") + f"  scale={XCE_SCALE:g}, 0 learnable rows")
            ld = DataLoader(Subset(TR_AUG, FIT[t].tolist()), batch_size=BS, shuffle=True,
                            num_workers=8, pin_memory=True)
            ok = tot = 1
            for _ in range(ep):
                model.train(); ok = tot = 0
                for x, lab in ld:
                    x = x.to(DEV, non_blocking=True)
                    y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        f = model(x).float()
                        lg = logit_fn(f)
                        loss = ce(lg, y)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(lp + head_params, GRAD_CLIP)
                    opt.step()
                    ok += int((lg.argmax(1) == y).sum()); tot += len(y)
                sch.step()
            tr_acc = ok / tot
        else:
            tr_acc = float("nan")

        Fp = un(extract(model, TE_EV, PROBE))
        if F0 is None:
            F0 = Fp
        drift = float((Fp * F0).sum(1).mean())
        drifts.append(drift)

        # accum head -- exact only while the backbone is frozen; that is the thing at risk
        Zr = un(extract(model, TR_EV, FIT[t]))
        pm = np.random.default_rng(t).permutation(len(FIT[t]))
        nv = max(int(0.1 * len(FIT[t])), 1)
        vZ.append(Zr[pm[:nv]]); vy.append(YTR[FIT[t]][pm[:nv]])
        for c_ in cls:                      # birth-frame prototypes, recorded POST-training
            m_ = YTR[FIT[t]] == c_
            if m_.any():
                PROTO[int(c_)] = un(Zr[m_].mean(0)[None])[0].astype(np.float32)
        g, c = build_GC(Zr[pm[nv:]], YTR[FIT[t]][pm[nv:]])
        G += g; C += c
        seen = np.concatenate(TASKS[:t + 1])
        te = np.where(np.isin(YTE, seen))[0]
        Zte = un(extract(model, TE_EV, te))
        a = solve_eval(G, C, np.concatenate(vZ), np.concatenate(vy), Zte, YTE[te], seen)
        accs.append(a)

        if ORACLE:
            allr = np.concatenate([FIT[s] for s in range(t + 1)])
            Zo = un(extract(model, TR_EV, allr))
            Go, Co = build_GC(Zo, YTR[allr])
            ao = solve_eval(Go, Co, np.concatenate(vZ), np.concatenate(vy),
                            Zte, YTE[te], seen)
            orcs.append(ao)
            ts = f"{tr_acc:.3f}" if ep > 0 else " -- "
            log(f"  [g={gamma:g} t={t}] ep={ep:<3d} tr={ts} drift={drift:.4f} "
                f"seen={len(seen):3d} acc {a:.4f} | oracle {ao:.4f} ({a-ao:+.4f})")
        else:
            ts = f"{tr_acc:.3f}" if ep > 0 else " -- "
            log(f"  [g={gamma:g} t={t}] ep={ep:<3d} tr={ts} drift={drift:.4f} "
                f"seen={len(seen):3d} acc {a:.4f}")
    del model, G, C
    torch.cuda.empty_cache()
    return dict(gamma=gamma, sched=sched, xce=XCE, accs=accs, oracle=orcs, drift=drifts,
                proto_MB=len(PROTO) * 768 * 4 / 1e6,
                A_last=accs[-1], A_avg=float(np.mean(accs)),
                O_last=(orcs[-1] if orcs else None))


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for g in GAMMAS:
    k = f"{g:g}"
    if k in res:
        log(f"skip gamma={k} (done)")
        continue
    res[k] = run(g)
    json.dump(res, open(OUT, "w"), indent=2)

W = 100
print("\n" + "=" * W)
print(f"EXP18 — A++ : decaying adaptation budget   ({DS} T={T} seed={SEED}, "
      f"e_t = {E0}*gamma^t, XCE={XCE}"
      + (f", scale={XCE_SCALE:g}" if XCE != "none" else "") + ")")
if XCE == "proto":
    print(f"cross-task CE on: later tasks use cosine CE over ALL seen classes with frozen "
          f"birth-frame prototypes ({N_CLS*768*4/1e6:.2f} MB total)")
print("=" * W)
print(f"{'gamma':>7}{'extra ep':>10}{'A-Last':>9}{'A-Avg':>9}{'oracle':>9}"
      f"{'head cost':>11}{'drift':>9}{'vs A_plus':>11}")
base = res.get("0", {}).get("A_last")
for k in sorted(res, key=float):
    r = res[k]
    o = r["O_last"]
    print(f"{float(k):>7g}{sum(r['sched'][1:]):>10d}{r['A_last']*100:>9.2f}"
          f"{r['A_avg']*100:>9.2f}{(o*100 if o else float('nan')):>9.2f}"
          f"{((r['A_last']-o)*100 if o else float('nan')):>+11.2f}"
          f"{r['drift'][-1]:>9.4f}"
          f"{((r['A_last']-base)*100 if base else float('nan')):>+11.2f}")
print("-" * W)
print("gamma=0 IS A_plus (0 epochs for every t>=1) and is the in-run reference.")
print("drift = mean cos(phi_t, phi_0) on a fixed 512-image probe; 1.0 = frozen.")
print("READ 'oracle' FIRST: for gamma>0 the accum head sums Gram matrices computed under")
print("      DIFFERENT backbones, so 'A-Last' conflates backbone change with head incoherence.")
print("      'oracle' rebuilds the head in the current frame and is the clean backbone read.")
print("READ: if 'oracle' RISES with gamma while 'A-Last' falls, tiny adaptation improved the")
print("      BACKBONE and only the head needs refreshing -> that is exp15's exemplars, not a")
print("      reason to stop adapting. If oracle falls too, the adaptation itself is harmful.")
print("=" * W)
print(f"wrote {OUT}")
