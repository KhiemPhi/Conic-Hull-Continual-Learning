#!/usr/bin/env python3
"""exp14_ranpac_headroom.py — how much room is there ABOVE RanPAC, and of what kind?

WHY
    Every conic variant in exp13 is head-side: it either replaces RanPAC's readout, reranks
    its logits, or fuses a score into them. So RanPAC's own headroom is a hard bound on the
    entire conic program. exp13's d_add (the variant with bounded downside -- beta=0 is in
    the grid) still lost -0.0012, which says the 768-d cone score is uninformative. Before
    running more variants, measure WHERE the remaining error actually lives.

    Bar: A_plus lr3e-4 + RanPAC M_RP=10000, accum protocol -> A-Last 0.8090 / A-Avg 0.8609.
    Gate already known: best OFFLINE head on the raw 768-d features is ~0.8055, i.e. BELOW
    the bar. The RP expansion is doing real work.

THE SIX READS, and what each one bounds
    1. ACCUM vs JOINT
       RanPAC's G and C are exactly additive, so accumulating per task should equal fitting
       on all data at once. If the delta is ~0, the incremental protocol costs the head
       NOTHING and there is no "CIL penalty" for a better head to recover. Anything a cone
       wins has to come from the classifier, not from the task structure.
    2. TOP-K ACCURACY               -> ceiling for ANY reranker (cone, OOD score, anything)
       If top-5 is 0.95 and top-1 is 0.809, a perfect top-5 reranker is worth +14 points and
       reranking is worth pursuing. If top-5 is 0.83, the ceiling is +2 and it is not.
    3. CONE RERANK vs ORACLE RERANK -> how much of that ceiling the cone actually captures
    4. TASK-ORACLE                  -> ceiling for anything task-structural (routing, per-task
       cones, stage confinement). Restricts argmax to the true task's 20 classes.
    5. WITHIN- vs CROSS-TASK ERRORS -> WHERE the error lives. Cross-task errors are what
       task-level geometry could address; within-task errors are fine-grained confusions no
       cone between tasks can touch.
    6. RIDGE vs LOGISTIC on h       -> is the squared loss itself the limitation? Ridge is
       the only reason RanPAC is closed-form; if CE on the same features is much better, the
       headroom is in the LOSS, not the geometry.
    Plus: per-class recall spread, to see whether error is concentrated (a few bad classes a
    specialised head could fix) or diffuse (nothing local to exploit).

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp14_ranpac_headroom.py                 # ~10-15 min
    SKIP_LOGREG=1 python -u exp14_ranpac_headroom.py   # ~5 min
"""
import json
import os
import time

import numpy as np
import torch

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


DEV = "cuda"
SEED = int(os.environ.get("SEED", 1))
N_TASKS, CPT, N_CLS = 10, 20, 200
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
WIDE_LAMBDAS = [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
TOPKS = [1, 2, 3, 5, 10, 20]
N_RAYS = int(os.environ.get("N_RAYS", 12))
FEATS = os.environ.get("FEATS", "exp12_feats_augreg_in21k_ep40_lr0.0003_aug1_s1.npz")
SKIP_LOGREG = int(os.environ.get("SKIP_LOGREG", 0))
SKIP_CONE = int(os.environ.get("SKIP_CONE", 0))
BAR_LAST = 0.8090


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


z = np.load(FEATS)
from datasets import load_dataset
_d = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_d["wnid"])
_lab = np.searchsorted(np.array(sorted(set(_w.tolist()))), _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
Ztr, ytr = un(z["Ftr"]).astype(np.float32), _lab[_p[:_n]]
Zte, yte = un(z["Fte"]).astype(np.float32), _lab[_p[_n:]]
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]
STAGE_OF = np.empty(N_CLS, int)
for s, cs in enumerate(TASKS):
    STAGE_OF[cs] = s

FIT_IDX, VAL_IDX = [], []
for t in range(N_TASKS):
    idx = np.where(np.isin(ytr, TASKS[t]))[0]
    pm = np.random.default_rng(t).permutation(len(idx))
    nv = max(int(0.1 * len(idx)), 1)
    VAL_IDX.append(idx[pm[:nv]])
    FIT_IDX.append(idx[pm[nv:]])
VAL_ALL, FIT_ALL = np.concatenate(VAL_IDX), np.concatenate(FIT_IDX)
log(f"train {Ztr.shape} test {Zte.shape} | fit {len(FIT_ALL)} val {len(VAL_ALL)}")

P_RP = torch.randn(768, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)


def H(Z, bs=4096):
    return torch.cat([torch.relu(torch.tensor(Z[i:i + bs], device=DEV,
                                              dtype=torch.float32) @ P_RP)
                      for i in range(0, len(Z), bs)])


Htr, Hte = H(Ztr), H(Zte)
Hfit, Hval = Htr[torch.as_tensor(FIT_ALL, device=DEV)], Htr[torch.as_tensor(VAL_ALL, device=DEV)]
yfit, yval = ytr[FIT_ALL], ytr[VAL_ALL]
log(f"h-space {tuple(Htr.shape)}")


def build_GC(Hs, ys):
    Hd = Hs.double()
    Y = torch.zeros(len(ys), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(ys)), torch.as_tensor(ys, device=DEV)] = 1.0
    return Hd.T @ Hd, Hd.T @ Y


def solve(G, C, lam):
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV,
                                                  dtype=torch.float64), C)


def pick_lam(G, C, grid):
    best, bw, bl = -1, None, None
    for lam in grid:
        W = solve(G, C, lam)
        a = float((( Hval.double() @ W).argmax(1).cpu().numpy() == yval).mean())
        if a > best:
            best, bw, bl = a, W, lam
    return bw, bl, best


# ---------------------------------------------------------------- 1. accum vs joint
log("=== 1. accum vs joint (RanPAC's G,C are exactly additive -> expect ~0 delta)")
Ga = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
Ca = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
for t in range(N_TASKS):
    g, c = build_GC(Htr[torch.as_tensor(FIT_IDX[t], device=DEV)], ytr[FIT_IDX[t]])
    Ga += g; Ca += c
Gj, Cj = build_GC(Hfit, yfit)
log(f"  ||G_accum - G_joint||/||G|| = "
    f"{float((Ga-Gj).norm()/Gj.norm()):.2e}   (exact additivity check)")

Wa, lam_a, _ = pick_lam(Ga, Ca, LAMBDAS)
L = (Hte.double() @ Wa)
pred = L.argmax(1).cpu().numpy()
acc1 = float((pred == yte).mean())
log(f"  RanPAC accum A-Last {acc1:.4f} (lambda={lam_a:g})   [bar {BAR_LAST:.4f}]")

Ww, lam_w, _ = pick_lam(Ga, Ca, WIDE_LAMBDAS)
accw = float(((Hte.double() @ Ww).argmax(1).cpu().numpy() == yte).mean())
log(f"  wider lambda grid   {accw:.4f} (lambda={lam_w:g})   delta {accw-acc1:+.4f}")

# ---------------------------------------------------------------- 2. top-k
log("=== 2. top-k accuracy -> the ceiling for ANY reranker")
Lnp = L.cpu().numpy()
order = np.argsort(-Lnp, 1)
topk = {}
for k in TOPKS:
    topk[k] = float((order[:, :k] == yte[:, None]).any(1).mean())
    log(f"  top-{k:<2d} {topk[k]:.4f}   (perfect reranker over top-{k} would gain "
        f"{topk[k]-acc1:+.4f})")

# ---------------------------------------------------------------- 3. cone rerank vs oracle
if not SKIP_CONE:
    log("=== 3. what a cone actually recovers of that reranking ceiling")
    from conic_hull import ConicHull

    def fit_one(X, n_rays=N_RAYS):
        n, d = X.shape
        pdim = int(min(64, n, d))
        return ConicHull(n_rays=int(np.clip(n_rays, 2, max(n - 2, 2))),
                         use_pca=pdim < d, pca_dim=pdim, ray_diversity="hybrid").fit(X)

    S = np.full((len(Zte), N_CLS), -np.inf, np.float32)
    for c in range(N_CLS):
        rows = FIT_ALL[ytr[FIT_ALL] == c]
        if len(rows) >= 8:
            S[:, c] = fit_one(Ztr[rows]).score(Zte)
    for k in [2, 3, 5, 10]:
        top = order[:, :k]
        pick = np.take_along_axis(S, top, 1).argmax(1)
        a = float((np.take_along_axis(top, pick[:, None], 1)[:, 0] == yte).mean())
        log(f"  top-{k:<2d} cone rerank {a:.4f}  vs oracle {topk[k]:.4f}  "
            f"-> captures {(a-acc1)/max(topk[k]-acc1,1e-9)*100:5.1f}% of the room")

# ---------------------------------------------------------------- 4/5. task structure
log("=== 4/5. task structure: is the error cross-task or within-task?")
true_stage = STAGE_OF[yte]
Lmask = Lnp.copy()
to = np.full(len(yte), -1)
for s in range(N_TASKS):
    m = true_stage == s
    sub = Lmask[m][:, TASKS[s]]
    to[m] = np.asarray(TASKS[s])[sub.argmax(1)]
task_oracle = float((to == yte).mean())
log(f"  task-oracle accuracy {task_oracle:.4f}   "
    f"(ceiling for ANY task-routing / per-task-geometry method: {task_oracle-acc1:+.4f})")

err = pred != yte
cross = (STAGE_OF[pred] != true_stage) & err
log(f"  errors {err.sum()} ({err.mean():.4f})  |  cross-task {cross.sum()} "
    f"({cross.sum()/max(err.sum(),1):.3f} of errors)  within-task "
    f"{(err.sum()-cross.sum())/max(err.sum(),1):.3f}")
log(f"  predicted-stage accuracy {float((STAGE_OF[pred] == true_stage).mean()):.4f}")

# ---------------------------------------------------------------- 6. loss family
if not SKIP_LOGREG:
    log("=== 6. ridge vs logistic on the SAME h -> is the squared loss the limitation?")
    lin = torch.nn.Linear(M_RP, N_CLS, bias=False).to(DEV)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    Yf = torch.as_tensor(yfit, device=DEV)
    Hf = Hfit.float()
    best_val, best_te = -1, 0
    for i in range(3000):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(Hf), Yf).backward()
        opt.step()
        if (i + 1) % 250 == 0:
            with torch.no_grad():
                av = float((lin(Hval.float()).argmax(1).cpu().numpy() == yval).mean())
                at = float((lin(Hte.float()).argmax(1).cpu().numpy() == yte).mean())
            if av > best_val:
                best_val, best_te = av, at
    log(f"  logistic on h (early-stopped on val) {best_te:.4f}   "
        f"delta vs ridge {best_te-acc1:+.4f}")

# ---------------------------------------------------------------- per-class structure
rec = np.array([(pred[yte == c] == c).mean() if (yte == c).sum() else np.nan
                for c in range(N_CLS)])
rec = rec[~np.isnan(rec)]
nfit = np.array([len(FIT_ALL[ytr[FIT_ALL] == c]) for c in range(N_CLS)])
from scipy.stats import spearmanr
log(f"=== per-class recall: mean {rec.mean():.3f} sd {rec.std():.3f} "
    f"min {rec.min():.3f} p10 {np.percentile(rec,10):.3f} max {rec.max():.3f}")
log(f"  spearman(recall, n_train) {float(spearmanr(rec, nfit).correlation):+.3f}  "
    f"| classes below 0.5 recall: {(rec<0.5).sum()}/{len(rec)}")

# ---------------------------------------------------------------- report
W_ = 96
print("\n" + "=" * W_)
print(f"EXP14 — RanPAC headroom on A_plus lr3e-4 features (ImageNet-R, seed {SEED})")
print("=" * W_)
print(f"  RanPAC A-Last (bar)          {acc1:.4f}")
print(f"  wider lambda grid            {accw:.4f}   {accw-acc1:+.4f}")
print(f"  task-oracle                  {task_oracle:.4f}   {task_oracle-acc1:+.4f}"
      f"   <- ceiling for task-structural methods")
for k in TOPKS[1:]:
    print(f"  top-{k:<2d} (perfect rerank)      {topk[k]:.4f}   {topk[k]-acc1:+.4f}")
print("-" * W_)
print("READ: a cone can only help where the true class is already in the candidate set and")
print("      the geometry separates it. Compare 'cone rerank' above to the top-k ceilings;")
print("      and if errors are mostly WITHIN-task, no between-task geometry can reach them.")
print("=" * W_)
out = dict(acc=acc1, acc_wide=accw, lam=lam_a, lam_wide=lam_w, task_oracle=task_oracle,
           topk={str(k): v for k, v in topk.items()},
           cross_task_frac=float(cross.sum() / max(err.sum(), 1)),
           recall_mean=float(rec.mean()), recall_sd=float(rec.std()))
json.dump(out, open(f"exp14_ranpac_headroom_s{SEED}.json", "w"), indent=2)
print(f"wrote exp14_ranpac_headroom_s{SEED}.json")
