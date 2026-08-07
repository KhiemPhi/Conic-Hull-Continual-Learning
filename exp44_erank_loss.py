#!/usr/bin/env python3
"""exp44_erank_loss.py — shape task-0 features FOR a subspace read-out, then run the head.

THE MISMATCH
    Task 0 is trained with a 20-way linear softmax. The head is a 200-way per-class
    SUBSPACE classifier. The feature objective has no knowledge of the read-out: it
    optimises linear separability of 20 classes and says nothing about subspace structure.
    Every gain so far (oPCA +1.58, allocation +0.57) has been won on the head side of that
    mismatch, with the features held fixed.

WHAT THE HEAD ACTUALLY WANTS
    exp29 measured that WITHIN-CLASS COVERAGE is the binding constraint (rho -0.585) while
    between-class overlap is not (+0.037). So the head wants each class to occupy FEW
    directions -- then R_c rays cover it. But the space must stay globally rich or 200
    classes stop being separable.

THE GLOBAL TERM HAS TO BE THE POOLED WITHIN-CLASS SCATTER, NOT THE BETWEEN-CLASS ONE
    This is the correction that matters. The head whitens by S_w = mean_c S_c, so the
    pooled within-class scatter becomes the identity BY CONSTRUCTION. If the loss makes
    every class uniformly low-rank, S_w is low-rank too, whitening amplifies its null
    directions, and the structure the loss built is destroyed. What survives whitening is
    only RELATIVE rank: classes lower-rank than the pool.
    So the objective must be "each class low-rank, but their subspaces pointing in
    DIFFERENT directions so the POOL stays high-rank":

        L = CE  +  lam_w * mean_c erank(S_c)/(K-1)  -  lam_p * erank(S_pool)/(n-P)

    with erank(S) = (tr S)^2 / tr(S^2), the participation ratio -- smooth, differentiable,
    and computable from Gram matrices rather than d x d scatters. Per-class eranks come
    from the (K,K) diagonal blocks of one (n,n) Gram of the within-class-centred batch;
    erank of the pool comes from the whole Gram. One matmul serves both.

    Batches are CLASS-BALANCED (P classes x K rows) because a per-class scatter estimated
    from whatever a random batch happens to contain is useless.

THE FALSIFIABLE PREDICTION, which is sharper than accuracy
    If this works, mean within-class effective rank should DROP -- so fewer rays should
    achieve the same coverage, and the R_c optimum should move DOWN. Better accuracy AND
    less storage. `erank` columns are reported for exactly this reason.

THE TRANSFER QUESTION, which decides everything
    The loss sees 20 classes. The head serves 200. If erank drops for task-0 classes but
    not for the 180 never seen during training, the loss has overfitted the first session
    and cannot help. `erank_t0` vs `erank_rest` is that measurement, and it should be read
    before the accuracy column.

WHY exp30's FAILURE DOES NOT PREDICT THIS
    exp30 (KD anchor to the frozen backbone) was monotonically harmful: 80.28 / 78.87 /
    76.03 at lam 0 / 0.1 / 0.5. But it CONSTRAINED features toward a fixed target, forfeiting
    adaptation. This SHAPES them with no target. Different failure mode.

HEAD: exp41's best -- oPCA gamma=0.5, R_c = clip(n_c/5, 24, 128) (k5m24, 80.07 A-Last).

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp44_erank_loss.py
    DS=IMAGENETR T=10 SEED=0 ARMS=0:0,0.1:0,0.1:0.1 python -u exp44_erank_loss.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import exp19_dataset_hull as E
import exp39_cone_construction as X
import fsa_train as F
from backbone import freeze_non_lora, get_lora_params, load_backbone

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
# arms are "lam_w:lam_p"; 0:0 is the CE baseline and must reproduce A+
ARMS = os.environ.get("ARMS", "0:0,0.1:0,0.1:0.1,0.3:0.1").split(",")
PCLS = int(os.environ.get("PCLS", 16))         # classes per batch
KPER = int(os.environ.get("KPER", 8))          # rows per class per batch
EPOCHS = int(os.environ.get("EPOCHS", 40))
LR = float(os.environ.get("LR", 3e-4))
KDIV = float(os.environ.get("KDIV", 5))
RMIN = int(os.environ.get("RMIN", 24))
RMAX = int(os.environ.get("RMAX", 128))
GAMMA = float(os.environ.get("GAMMA", 0.5))
F_MAX = int(os.environ.get("F_MAX", 2000))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
OUT = os.path.join(REPO, f"exp44_erank_loss_{TAG}.json")

un = X.un


def erank_from_gram(Gm):
    """(tr G)^2 / tr(G^2). Same nonzero spectrum as the d x d scatter, so this is the
    participation ratio of the covariance computed in O(n^2) instead of O(d^2)."""
    tr = torch.diagonal(Gm).sum()
    return tr * tr / (Gm.pow(2).sum() + 1e-8)


def erank_terms(f, P, K):
    """f is (P*K, d) ordered class-major. Returns (mean per-class erank, pooled erank),
    both normalised to [0,1] by their maxima so lam_w and lam_p are on one scale."""
    fb = f.view(P, K, -1)
    fb = fb - fb.mean(1, keepdim=True)          # centre WITHIN each class
    Z = fb.reshape(P * K, -1)
    Gm = Z @ Z.T
    per = torch.stack([erank_from_gram(Gm[i * K:(i + 1) * K, i * K:(i + 1) * K])
                       for i in range(P)])
    return per.mean() / max(K - 1, 1), erank_from_gram(Gm) / max(P * K - P, 1)


class Balanced(torch.utils.data.Sampler):
    """P classes x K rows per batch. A per-class scatter from an unbalanced batch is noise."""

    def __init__(self, labels, P, K, nb, seed):
        self.by = {int(c): np.where(labels == c)[0] for c in np.unique(labels)}
        self.P, self.K, self.nb = min(P, len(self.by)), K, nb
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.nb

    def __iter__(self):
        for _ in range(self.nb):
            cs = self.rng.choice(list(self.by), self.P, replace=False)
            yield np.concatenate([self.rng.choice(self.by[int(c)], self.K,
                                                  replace=len(self.by[int(c)]) < self.K)
                                  for c in cs]).tolist()


def train_feats(lam_w, lam_p):
    tag = f"{DS}_T{T}_s{SEED}_ep{EPOCHS}_lr{LR:g}_P{PCLS}K{KPER}_w{lam_w:g}_p{lam_p:g}"
    cache = os.path.join(REPO, f"exp44_feats_{tag}_{TAG}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"  cached {tag}  (acc0 {float(z['acc0']):.4f})")
        return un(z["Ftr"]), un(z["Fte"]), float(z["acc0"])

    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = F.get_data(DS)
    cpt = n_cls // T
    torch.manual_seed(SEED); np.random.seed(SEED)
    task0 = np.random.default_rng(SEED).permutation(n_cls)[:cpt]
    idx = np.where(np.isin(ytr, task0))[0]
    remap = {int(c): i for i, c in enumerate(task0)}
    sub_lab = np.array([remap[int(c)] for c in ytr[idx]])

    model = load_backbone(F.MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    head = nn.Linear(model.num_features, cpt).to(DEV)
    params = list(get_lora_params(model)) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    nb = max(1, len(idx) // (PCLS * KPER))
    ld = DataLoader(Subset(tr_aug, idx.tolist()), num_workers=8, pin_memory=True,
                    batch_sampler=Balanced(sub_lab, PCLS, KPER, nb, SEED))
    acc0 = 0.0
    for e in range(EPOCHS):
        model.train()
        ok = tot = 0; ew = ep_ = 0.0
        for x, lab in ld:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x).float()
            logit = head(f)
            loss = ce(logit, y)
            if lam_w > 0 or lam_p > 0:
                P_ = len(y) // KPER
                rw, rp = erank_terms(f[:P_ * KPER], P_, KPER)
                loss = loss + lam_w * rw - lam_p * rp
                ew += float(rw); ep_ += float(rp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, F.GRAD_CLIP)
            opt.step()
            ok += int((logit.argmax(1) == y).sum()); tot += len(y)
        sch.step()
        acc0 = ok / max(tot, 1)
        if e % 10 == 0 or e == EPOCHS - 1:
            log(f"    ep{e:3d} acc0 {acc0:.4f}  erank_w {ew/nb:.3f}  erank_pool {ep_/nb:.3f}")

    Ftr = F.extract(model, tr_ev)
    Fte = F.extract(model, te_ev)
    np.savez_compressed(cache, Ftr=Ftr, Fte=Fte, acc0=acc0)
    del model, head
    torch.cuda.empty_cache()
    return un(Ftr), un(Fte), acc0


def erank_diag(Ztr, ytr, task0):
    """Per-class effective rank IN THE WHITENED SPACE -- what the head actually sees --
    split by whether the class was visible to the loss. If t0 drops and rest does not,
    the loss overfitted the first session and cannot help the other 180 classes."""
    d = Ztr.shape[1]
    sc = np.zeros((d, d)); n = 0
    for c in np.unique(ytr):
        r = np.where(ytr == c)[0]
        Xc = Ztr[r] - Ztr[r].mean(0)
        sc += Xc.T @ Xc; n += len(Xc)
    S = sc / max(n, 1); S = S + SHRINK * np.trace(S) / d * np.eye(d)
    Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)
    out = {True: [], False: []}
    for c in np.unique(ytr):
        r = np.where(ytr == c)[0]
        Xw = un(Ztr[r] @ Wh)
        Xw = Xw - Xw.mean(0)
        Gm = Xw @ Xw.T
        out[int(c) in set(int(x) for x in task0)].append(
            float(np.trace(Gm) ** 2 / ((Gm ** 2).sum() + 1e-9)))
    return float(np.mean(out[True])), float(np.mean(out[False]))


def replay(Ztr, Zte, ytr, yte, n_cls):
    """exp41's k5m24 head: oPCA gamma=0.5, R_c = clip(n_c/5, 24, 128). Plus ranpac."""
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(SEED).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def proj(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A = {}
    res = {"ranpac": [], "cone": []}
    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        S_ = scatter / max(n_scat, 1)
        S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)
        rng = np.random.default_rng(1234 + t)
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw = un(Ztr[r] @ Wh)
            oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
            past = [A[o] for o in A if o not in tasks[t]]
            Fr = np.concatenate([Ztr[oth]] + past, 0)
            if len(Fr) > F_MAX:
                Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
            Rc = int(np.clip(len(r) / KDIV, RMIN, RMAX))
            A[c] = X.BUILD["opca"](Xw, un(Fr @ Wh), Rc, int(c), GAMMA) @ Wh_inv
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
        sa = np.asarray(seen)

        def acc(Z, y):
            return float((sa[Z[:, seen].argmax(1)] == y).mean())
        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(proj(un(Ztr[VAL_ALL[:nval]]), Wm), yv)
            if a > best:
                best, bw = a, Wm
        res["ranpac"].append(acc(proj(un(Zte[tei]), bw), yt))
        Qw = un(Zte[tei] @ Wh)
        St = np.full((len(tei), n_cls), -np.inf, np.float32)
        for c in seen:
            if c in A:
                St[:, c] = X.cone_score(un(A[c] @ Wh), Qw)
        res["cone"].append(acc(St, yt))
        log(f"      s{t}: ranpac {res['ranpac'][-1]*100:.2f}  cone {res['cone'][-1]*100:.2f}")
    del G, C, P, eye
    torch.cuda.empty_cache()
    return {k: {"A_last": v[-1], "A_avg": float(np.mean(v))} for k, v in res.items()}


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
E.T, E.SEED = T, SEED
ytr_, yte_, n_cls_ = E.get_labels(DS)
task0_ = np.random.default_rng(SEED).permutation(n_cls_)[:n_cls_ // T]
for arm in ARMS:
    lw, lp = (float(x) for x in arm.split(":"))
    key = (f"{DS}|{T}|{SEED}|w{lw:g}p{lp:g}|P{PCLS}K{KPER}ep{EPOCHS}"
           f"|k{KDIV:g}m{RMIN}g{GAMMA:g}|m{M_RP}_s{SHRINK:g}|v1")
    if key in allres:
        log(f"skip {key}"); continue
    log(f"=== {key}")
    Ztr, Zte, acc0 = train_feats(lw, lp)
    e0, e1 = erank_diag(Ztr, ytr_, task0_)
    log(f"  erank(whitened): task0 {e0:.2f}   unseen {e1:.2f}   acc0 {acc0:.4f}")
    blob = replay(Ztr, Zte, ytr_, yte_, n_cls_)
    blob.update({"acc0": acc0, "erank_t0": e0, "erank_rest": e1})
    allres[key] = blob
    json.dump(allres, open(OUT, "w"), indent=2)

W = 96
print("\n" + "=" * W)
print("EXP44 — erank-shaped task-0 features + the k5m24 conic head")
print("=" * W)
print(f"  {'arm':<12}{'acc0':>7}{'erank_t0':>10}{'erank_rest':>12}"
      f"{'cone':>9}{'ranpac':>9}{'cone A-Avg':>12}")
base = None
for key, r in sorted(allres.items()):
    arm = key.split("|")[3]
    if arm == "w0p0":
        base = r
    print(f"  {arm:<12}{r['acc0']:>7.4f}{r['erank_t0']:>10.2f}{r['erank_rest']:>12.2f}"
          f"{r['cone']['A_last']*100:>9.2f}{r['ranpac']['A_last']*100:>9.2f}"
          f"{r['cone']['A_avg']*100:>12.2f}")
if base:
    print(f"\n  baseline w0p0: cone {base['cone']['A_last']*100:.2f} "
          f"erank_t0 {base['erank_t0']:.2f} erank_rest {base['erank_rest']:.2f}")
    for key, r in sorted(allres.items()):
        arm = key.split("|")[3]
        if arm != "w0p0":
            print(f"  {arm:<12} d_cone {(r['cone']['A_last']-base['cone']['A_last'])*100:+6.2f}"
                  f"   d_erank_t0 {r['erank_t0']-base['erank_t0']:+6.2f}"
                  f"   d_erank_rest {r['erank_rest']-base['erank_rest']:+6.2f}")
print("\n" + "-" * W)
print("READ erank_rest FIRST. The loss sees 20 classes; the head serves 200. If erank_t0")
print("   drops but erank_rest does not, the loss overfitted the first session and the")
print("   accuracy column is beside the point.")
print("w0p0 is the CE baseline: cone must be ~80.07 and ranpac ~80.28. It retrains task 0")
print("   from scratch with a BALANCED sampler, so small drift is expected -- beyond ~0.3")
print("   means the sampler changed the recipe, not that the loss did anything.")
print("If erank_rest drops AND cone improves, re-run the R_c sweep: the prediction is that")
print("   the optimum moves DOWN, i.e. fewer rays for the same coverage. That would be a")
print("   storage win on top of the accuracy one, and it is the sharper confirmation.")
print("=" * W)
print(f"wrote {OUT}")
