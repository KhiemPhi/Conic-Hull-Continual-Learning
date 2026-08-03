#!/usr/bin/env python3
"""exp15_exemplar_replay.py — exemplar replay into the BACKBONE, with selection ablated.

WHY THIS AND NOT ANOTHER HEAD EXPERIMENT
    exp14 closed the head. On A_plus the RanPAC head is EXACTLY optimal-additive
    (||G_accum - G_joint||/||G|| = 2.9e-15), a wider lambda grid gives +0.0000, and the only
    head-side lift left anywhere is ridge->CE at +0.0062. Meanwhile 88.7% of remaining errors
    are CROSS-TASK and the joint ceiling is +0.0420 away. That pool is the backbone, and the
    only lever that reaches it is real old-class images with a real gradient path.

    Bar (seed 1, augreg_in21k, ep40 lr3e-4 AUG=1):  A_plus  A-Last 0.8090 / A-Avg 0.8609
    Targets: MACIL 0.8182/0.8576   GR-LoRA 0.8209/0.8620   joint ceiling ~0.8510 (stale,
    measured at lr1e-4 -- probably higher at lr3e-4).

WHAT IS ACTUALLY BEING TESTED
    PRIMARY   does replaying B images/class beat freezing after task 0 at all?
              -> compare any replay_* arm against A_plus (0.8090) and against `none`.
    SECONDARY does WHICH images you keep matter, and is the conic hull good at choosing?
              -> the selection axis. This is a second-order knob on top of the primary
              effect, worth maybe 1-2 points of it. A null here is NOT a failure of the
              experiment; read the two questions separately.

WHY THE HULL MIGHT WIN THE SELECTION AXIS (and why every prior conic attempt lost)
    Every conic failure in this repo asked the hull to DISCRIMINATE -- scoring, routing,
    penalties, reranking. exp14 put a number on how bad that is: reranking a two-element
    candidate set cost -7.7 points. Selection is the opposite task, and it is what SPA is
    actually built for: directionally diverse extremal points with outlier filtering.
    _robust_density_spa's own docstring says "m: number of images to select for the replay
    buffer (e.g., 20)". The code was written for this.

    Open question, genuinely: herding picks points CLOSEST to the class mean (representative);
    SPA picks the BOUNDARY (atypical). These are opposite strategies. Boundary points are the
    informative ones for discriminative training, but they are also where mislabelled and
    freak images live. Hence the controls below -- without `random` and `herding` in the same
    run a hull win is unattributable, and this project has enough unattributable results.

SELECTION ARMS (identical buffer size, identical everything else)
    none      no buffer -- isolates "does replay help" from "does selection help"
    random    uniform. The baseline coreset papers routinely fail to beat.
    herding   iCaRL greedy mean-matching (representative)
    kcenter   greedy farthest-point (pure diversity, no extremity)
    spa       YOUR ConicHull, ray_diversity="spa"   -> hull.extreme_rays_index
    hybrid    YOUR ConicHull, ray_diversity="hybrid" (SPA oversample then FPS)

HEAD HANDLING, and why the oracle diagnostic is not optional
    Once the backbone adapts, old features drift, so the head CANNOT accumulate. It is
    rebuilt each task from re-extracted features -- which is the whole argument for storing
    IMAGES rather than (mu, Sigma): pixels can be re-featurised in the current frame, stored
    statistics cannot.
    But that leaves the head seeing only B rows per old class versus ~108 for the current
    task, while A_plus's head sees ALL 21605 rows. So a replay arm can have a BETTER backbone
    and still lose on total accuracy. ORACLE_STATS (on by default) rebuilds the head from all
    real seen data in the current frame -- cheating, and the only way to separate
    "the backbone improved" from "the head lost data". Without it a null is uninterpretable.

COST
    ~75 min per arm (training grows with the buffer; 40 epochs x 10 tasks) plus ~8 min for
    the oracle diagnostic. Budget ~1.4 h/arm. The default 4 arms is ~5.5 h.
    Cheap screen:  ARMS=none,random,spa BUFFER=20 ORACLE=0   (~3.5 h)

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp15_exemplar_replay.py
    ARMS=random,herding,spa,hybrid BUFFER=20 python -u exp15_exemplar_replay.py
    ARMS=spa BUFFER=5 ORACLE=0 python -u exp15_exemplar_replay.py
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
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


DEV = "cuda"
SEED = int(os.environ.get("SEED", 1))
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
N_TASKS, CPT, N_CLS = 10, 20, 200
EPOCHS = int(os.environ.get("EPOCHS", 40))
LR = float(os.environ.get("LR", 3e-4))          # the exp12 optimum
BS = 128
GRAD_CLIP = 1.0
AUG = int(os.environ.get("AUG", 1))
BUFFER = int(os.environ.get("BUFFER", 20))      # images stored per class
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
ORACLE = int(os.environ.get("ORACLE", 1))
ARMS = [a for a in os.environ.get("ARMS", "none,random,herding,spa").split(",") if a]
BAR_LAST, BAR_AVG = 0.8090, 0.8609              # A_plus lr3e-4, seed 1 (exp12)

torch.manual_seed(SEED)
np.random.seed(SEED)

_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
TF_EVAL = create_transform(**_cfg, is_training=False)
TF_TRAIN = (create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                             re_prob=0.25, scale=(0.7, 1.0), hflip=0.5) if AUG else TF_EVAL)

from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"])
_lab = np.searchsorted(np.array(sorted(set(_w.tolist()))), _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
TR_IDX, TR_Y = _p[:_n], _lab[_p[:_n]]
TE_IDX, TE_Y = _p[_n:], _lab[_p[_n:]]
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

FIT_IDX, VAL_IDX = [], []
for t in range(N_TASKS):
    idx = np.where(np.isin(TR_Y, TASKS[t]))[0]
    pm = np.random.default_rng(t).permutation(len(idx))
    nv = max(int(0.1 * len(idx)), 1)
    VAL_IDX.append(idx[pm[:nv]])
    FIT_IDX.append(idx[pm[nv:]])
log(f"ImageNet-R seed {SEED} | fit {sum(len(f) for f in FIT_IDX)} | buffer {BUFFER}/class "
    f"({BUFFER*N_CLS} imgs ~ {BUFFER*N_CLS*15/1024:.0f} MB as 224px JPEG) | arms {ARMS}")


class Wrap(Dataset):
    """Row index is into TR_IDX / TE_IDX (the local train/test arrays)."""
    def __init__(self, base_idx, labels, tf):
        self.base, self.labels, self.tf = base_idx, labels, tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img = _ds[int(self.base[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(self.labels[i]), i


TRAIN_AUG = Wrap(TR_IDX, TR_Y, TF_TRAIN)
TRAIN_EV = Wrap(TR_IDX, TR_Y, TF_EVAL)
TEST_EV = Wrap(TE_IDX, TE_Y, TF_EVAL)


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


@torch.no_grad()
def extract(model, ds, rows):
    model.eval()
    ld = DataLoader(Subset(ds, np.asarray(rows).tolist()), batch_size=256, shuffle=False,
                    num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _, _ in ld:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


# ====================== selection strategies ======================
# Each returns POSITIONS into `rows` (length B). Features F are current-frame, L2-normalised.
def sel_random(F, rows, B, seed):
    return np.random.default_rng(seed).permutation(len(rows))[:B]


def sel_herding(F, rows, B, seed):
    """iCaRL: greedily pick so the running mean tracks the class mean. Representative."""
    mu = F.mean(0)
    chosen, acc = [], np.zeros_like(mu)
    for k in range(min(B, len(F))):
        scores = (mu * (k + 1) - acc) @ F.T
        scores[chosen] = -np.inf
        chosen.append(int(scores.argmax()))
        acc = acc + F[chosen[-1]]
    return np.array(chosen)


def sel_kcenter(F, rows, B, seed):
    """Greedy farthest-point. Pure diversity, no extremity criterion."""
    start = int(np.argmax(F @ F.mean(0)))
    chosen = [start]
    d = 1.0 - F @ F[start]
    for _ in range(min(B, len(F)) - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, 1.0 - F @ F[nxt])
    return np.array(chosen)


def _hull_select(F, B, diversity):
    """YOUR ConicHull. extreme_rays_index gives positions into F, which is exactly a
    replay-buffer selection -- see _robust_density_spa's docstring.

    pca_dim/n_rays clamped per class: ImageNet-R class sizes run ~28..309 and sklearn PCA
    needs n_components <= n_samples."""
    n, d = F.shape
    pdim = int(min(64, n, d))
    hull = ConicHull(n_rays=int(np.clip(B, 2, max(n - 2, 2))), use_pca=pdim < d,
                     pca_dim=pdim, ray_diversity=diversity, spa_oversample=3).fit(F)
    return np.asarray(hull.extreme_rays_index)[:B]


def sel_spa(F, rows, B, seed):
    return _hull_select(F, B, "spa")


def sel_hybrid(F, rows, B, seed):
    return _hull_select(F, B, "hybrid")


SELECT = {"random": sel_random, "herding": sel_herding, "kcenter": sel_kcenter,
          "spa": sel_spa, "hybrid": sel_hybrid}

# ====================== RanPAC head ======================
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
        G += h.T @ h
        C += h.T @ Y
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


# ====================== one arm ======================
def run(arm):
    log(f"=== ARM '{arm}'  buffer={BUFFER}/class  lr={LR:g} ep={EPOCHS} ===")
    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))
    buf_rows, accs, oracles = [], [], []

    for t in range(N_TASKS):
        cur = FIT_IDX[t]
        # ---- 1. train on current task + everything in the buffer -------------------
        tr_rows = np.concatenate([cur, np.array(buf_rows, int)]) if buf_rows else cur
        seen = np.concatenate(TASKS[:t + 1])
        cls2i = {int(c): i for i, c in enumerate(seen)}       # head over ALL seen classes,
        head = nn.Linear(768, len(seen)).to(DEV)              # so replay supplies real
        opt = torch.optim.AdamW(lp + list(head.parameters()), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        ce = nn.CrossEntropyLoss()
        ld = DataLoader(Subset(TRAIN_AUG, tr_rows.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
        ok = tot = 1
        for _ in range(EPOCHS):
            model.train()
            ok = tot = 0
            for x, lab, _ in ld:
                x = x.to(DEV, non_blocking=True)
                y = torch.tensor([cls2i[int(l)] for l in lab], device=DEV)
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
        log(f"  [{arm} t={t}] trained on {len(tr_rows)} imgs "
            f"({len(cur)} new + {len(buf_rows)} replayed)  train-acc {ok/tot:.3f}")

        # ---- 2. select this task's exemplars IN THE CURRENT FRAME ------------------
        if arm != "none":
            Fcur = un(extract(model, TRAIN_EV, cur))
            for c in TASKS[t]:
                m = np.where(TR_Y[cur] == c)[0]
                if len(m) == 0:
                    continue
                pos = SELECT[arm](Fcur[m], cur[m], min(BUFFER, len(m)), int(c))
                buf_rows.extend(cur[m][np.asarray(pos, int)].tolist())

        # ---- 3. head: rebuild from re-extracted buffer + current (frames changed, so
        #        accumulation is invalid -- this is exactly why images beat stored stats)
        head_rows = np.concatenate([cur, np.array(buf_rows, int)]) if buf_rows else cur
        head_rows = np.unique(head_rows)
        Zh = extract(model, TRAIN_EV, head_rows)
        yh = TR_Y[head_rows]
        # lambda is picked on this task's val rows + the buffer (old val rows were never
        # stored). Identical for every arm, so the comparison stays paired.
        val_rows = np.unique(np.concatenate(
            [VAL_IDX[t], np.array(buf_rows, int)] if buf_rows else [VAL_IDX[t]]))
        Zv, yv = extract(model, TRAIN_EV, val_rows), TR_Y[val_rows]
        te = np.where(np.isin(TE_Y, seen))[0]
        Zte = extract(model, TEST_EV, te)
        G, C = build_GC(un(Zh), yh)
        a = solve_eval(G, C, un(Zv), yv, Zte, TE_Y[te], seen)
        accs.append(a)

        # ---- 4. ORACLE: head from ALL real seen data in the current frame ----------
        if ORACLE:
            allr = np.concatenate([FIT_IDX[s] for s in range(t + 1)])
            Zo = un(extract(model, TRAIN_EV, allr))
            Go, Co = build_GC(Zo, TR_Y[allr])
            ao = solve_eval(Go, Co, un(Zv), yv, Zte, TE_Y[te], seen)
            oracles.append(ao)
            log(f"  [{arm} t={t}] seen={len(seen):3d} acc {a:.4f} | oracle-head {ao:.4f} "
                f"(head-data cost {a-ao:+.4f})")
        else:
            log(f"  [{arm} t={t}] seen={len(seen):3d} acc {a:.4f}")

    del model
    torch.cuda.empty_cache()
    return accs, oracles


# ====================== main ======================
OUT = f"exp15_exemplar_replay_s{SEED}_B{BUFFER}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for arm in ARMS:
    if arm in res:
        log(f"skip '{arm}' (already in {OUT})")
        continue
    a, o = run(arm)
    res[arm] = {"accs": a, "oracle": o, "A_last": a[-1], "A_avg": float(np.mean(a))}
    json.dump(res, open(OUT, "w"), indent=2)

W = 104
print("\n" + "=" * W)
print(f"EXP15 — exemplar replay into the backbone, selection ablated "
      f"(ImageNet-R seed {SEED}, B={BUFFER}/class)")
print(f"storage: {BUFFER*N_CLS} imgs ~ {BUFFER*N_CLS*15/1024:.0f} MB   "
      f"(SOTA stores 473 MB of per-class covariance)")
print("=" * W)
print(f"{'arm':<10}{'A-Last':>9}{'A-Avg':>9}{'dLast':>9}{'dAvg':>9}{'oracle-head':>13}"
      f"{'head cost':>11}")
for arm, r in sorted(res.items(), key=lambda kv: -kv[1]["A_last"]):
    orc = r["oracle"][-1] if r.get("oracle") else float("nan")
    print(f"{arm:<10}{r['A_last']:>9.4f}{r['A_avg']:>9.4f}{r['A_last']-BAR_LAST:>+9.4f}"
          f"{r['A_avg']-BAR_AVG:>+9.4f}{orc:>13.4f}{r['A_last']-orc:>+11.4f}")
print("-" * W)
print(f"{'A_plus':<10}{BAR_LAST:>9.4f}{BAR_AVG:>9.4f}   <- THE BAR (freeze after task 0, "
      f"0 images stored)")
print(f"{'MACIL':<10}{0.8182:>9.4f}{0.8576:>9.4f}\n{'GR-LoRA':<10}{0.8209:>9.4f}"
      f"{0.8620:>9.4f}   <- SOTA")
print(f"{'joint':<10}{0.8510:>9.4f}      (stale: measured at lr1e-4)")
print("-" * W)
print("READ 1 (primary): does any replay arm beat A_plus 0.8090? If none does, replay at")
print("        this buffer size does not pay and the selection axis is moot.")
print("READ 2 (selection): spa/hybrid vs random AND herding. random is the honest baseline.")
print("READ 3: 'head cost' = acc - oracle-head. Large negative means the backbone may have")
print("        improved while the head starved on B rows/class -- opposite fixes.")
print("=" * W)
print(f"wrote {OUT}")
