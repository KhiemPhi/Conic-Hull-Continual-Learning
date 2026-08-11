#!/usr/bin/env python3
"""exp55_lora_diversity_pilot.py -- do K first-session LoRAs make DIFFERENT mistakes?

WHY THIS FILE EXISTS, AND WHY IT DELIBERATELY DOES NOT TOUCH THE CONE
    exp54 closed the read-out. Its pre-registered falsification criterion fired
    (stack_f32 - fuse_cone = -0.00 on IMAGENETR A-Avg) and every stacking variant came in a
    coin across 12 cells. The reason is not subtle -- our deficit is in the BASE:

        A-Avg      our RanPAC   GR-LoRA   base deficit   fusion recovers
        CIFAR100      95.13      94.65       +0.48            +0.05
        IMAGENETR     85.31      86.20       -0.89            +0.63
        IMAGENETAP    68.66      70.24       -1.58            +0.63
        CUB200P       93.11      93.85       -0.74            +0.03

    On every losing dataset the base classifier is behind by more than the read-out can
    recover, and we win exactly where our base already wins. So this file measures ONLY the
    base: RanPAC on features, no cones, no fusion, no beta search. If K adapters cannot move
    RanPAC, nothing built on top of them matters and Tier 2 is over in 30 minutes instead of
    six hours.

THE HYPOTHESIS AND ITS OBVIOUS FAILURE MODE
    exp52 decomposed our fused gain as 66% linear combination / 27% rays / 7% non-negativity
    -- ensemble decorrelation, not conic structure. exp54 then showed that decorrelation is
    exhausted at the RULE level: given RanPAC + cone, adding `sub` and `pm` over THE SAME
    rays buys nothing (the optimiser had `pm` available and gave it weight on 13% of
    IMAGENETR stages at mean 0.11). The open question is whether decorrelation at the
    FEATURE level behaves differently.

    The failure mode is equally obvious: K adapters trained on the SAME task-0 data with the
    SAME objective and recipe will land in nearly the same solution, and an ensemble of
    identical members gains exactly nothing. So members must differ structurally, and the
    diversity has to be MEASURED rather than assumed.

DIVERSITY IS THE PRECONDITION AND IT PRINTS FIRST
    Classic ensemble decomposition: gain ~ average individual error - diversity. This file
    reports, before any headline number:
        disagree_ij   fraction of test rows where members i and j predict differently
        errcorr_ij    correlation of their per-row correctness
        ORACLE        accuracy of an oracle that picks a correct member when one exists
    ORACLE - best_single is the ENTIRE headroom available to any combination rule. If that
    is ~0 the members are redundant, no fusion recovers anything, and the answer is more
    aggressive diversity (or a different mechanism) rather than a better combiner. Read it
    before the accuracy table; an ensemble that gains nothing against a large oracle gap is
    a combiner problem, an ensemble that gains nothing against a small one is a dead idea.

MEMBERS -- spec grammar `{targets}{rank}[b{pct}]`
    q32     attn.qkv + attn.proj, rank 32.  THE exp16 RECIPE.
    m32     mlp.fc1 + mlp.fc2, rank 32.
    a16     all four, rank 16.
    q32b70  q32 trained on a random 70% of task-0's CLASSES (bagging).
    qkv/proj adapt WHAT TOKENS ATTEND TO; mlp.fc1/fc2 adapt the PER-TOKEN transform. Those
    are the two members most likely to make different mistakes, which is why the default
    member set spans them rather than varying seeds -- differing only by init would test the
    weakest diversity axis available and would almost certainly read ~0 disagreement.

MEMBER q32 IS NOT RETRAINED -- IT IS exp16's CACHE
    Retraining it would only introduce GPU nondeterminism between this file and every number
    already recorded (bf16 autocast + cuDNN make bit-exactness across runs unreliable, and
    exp49 measured thread-count nondeterminism flipping k-means basins even on CPU). Loading
    `exp16_feats_{ds}_T{T}_s{seed}_ep40_lr0.0003_aug1_{TAG}.npz` makes member 0 an EXACT
    control by construction, and saves one training per cell. VERIFY=1 additionally asserts
    its RanPAC reproduces the exp16 bar.

WHAT WOULD KILL TIER 2
    `concat - q32` on IMAGENETR A-Avg. Our base deficit there is -0.89, so anything under
    about +0.5 means concatenating adapters does not close it and no read-out on top will.
    Combined with a small ORACLE gap that is a clean kill. This file is scoped so that
    outcome costs ~30 minutes.

COST
    Member q32 is free (cached). Each other member is one task-0 training plus a feature
    extraction, ~2-3 min measured (CUB200P features landed 2-3 min apart, IMAGENETAP 2 min).
    Default pilot = 2 new members x 3 seeds x 1 dataset ~ 20-30 min. Features are cached per
    member, so widening to the full grid later reuses everything.

PIN YOUR THREADS -- the read-out side inherits exp49's 0.27 unpinned noise floor.

USAGE
    source ~/venvs/ml_env/bin/activate

    # the pilot that decides Tier 2
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0,1,2 MEMBERS=q32,m32,a16 VERIFY=1 \
      python -u exp55_lora_diversity_pilot.py

    # if diversity is real but small, push it harder
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=IMAGENETR T=10 SEED=0,1,2 MEMBERS=q32,m32,a16,q32b70,q64 \
      python -u exp55_lora_diversity_pilot.py

    # widen only after the pilot passes
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CIFAR100,IMAGENETR,IMAGENETAP,CUB200P T=10 SEED=0,1,2 \
      python -u exp55_lora_diversity_pilot.py

    Run SEQUENTIALLY; concurrency on this box is measured strictly worse.
"""
import itertools
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# exp19_dataset_hull parses T and SEED as SCALARS at import time:
#     SEED = int(os.environ.get("SEED", 0))
# so importing it with SEED=0,1,2 still in the environment raises ValueError before a line of
# this file runs. exp49/52/53/54 all narrow the environment before the import for exactly
# this reason; this file originally imported it lazily inside run_cell, which only moved the
# crash to AFTER the members had been trained. Grid values are captured here and the
# environment is narrowed to the first element; E.T / E.SEED are reassigned per cell in
# run_cell, so the scalars left in os.environ are never read again.
_DS = os.environ.get("DS", "IMAGENETR").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0,1,2").split(",")]
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])

import exp19_dataset_hull as E                                         # noqa: E402
import fsa_train as F                                                  # noqa: E402
import class_order as CO                                               # noqa: E402
from backbone import freeze_non_lora, get_lora_params, load_backbone   # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = F.TAG
DSETS, TS, SEEDS = _DS, _TS, _SEEDS
MEMBERS = os.environ.get("MEMBERS", "q32,m32,a16").split(",")
EPOCHS = int(os.environ.get("EPOCHS", 40))
LR = float(os.environ.get("LR", 3e-4))
ALPHA = float(os.environ.get("ALPHA", 4.0))
BS, GRAD_CLIP = 128, 1.0
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
VERIFY = int(os.environ.get("VERIFY", 0))
OUT = os.path.join(REPO,
                   f"exp55_lora_diversity{os.environ.get('SUFFIX', '')}_{TAG}.json")
BAR = json.load(open(os.path.join(REPO, f"exp16_full_table_{TAG}.json")))

assert MEMBERS[0] == "q32", (
    f"member 0 must be q32 -- it is the exp16 recipe and the control every delta in this "
    f"file is measured against. Got {MEMBERS[0]!r}.")
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}); exp49 measured the unpinned noise "
        f"floor at 0.27. Prefix with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 or ALLOW_UNPINNED=1.")

un = F.un
_SPEC = re.compile(r"^([qma])(\d+)(?:b(\d+))?$")
_TARGETS = {"q": ["attn.qkv", "attn.proj"],
            "m": ["mlp.fc1", "mlp.fc2"],
            "a": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]}


def parse_member(spec):
    m = _SPEC.match(spec)
    assert m, (f"bad member spec {spec!r}; expected e.g. q32, m32, a16, q32b70 "
               f"({{q|m|a}}{{rank}}[b{{pct}}])")
    kind, rank, bag = m.group(1), int(m.group(2)), m.group(3)
    return _TARGETS[kind], rank, (int(bag) / 100.0 if bag else 1.0)


def bar_for(ds, T, seed):
    # recipe_tag() now appends the order tag, so exp16's key moves with the class order too.
    # Hardcoding the legacy recipe here would silently compare a pilot-order run against a
    # legacy-order bar -- the exact class of mismatch class_order.py exists to prevent.
    k = f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1{CO.order_tag()}"
    v = BAR.get(k)
    assert v is not None, (
        f"no exp16 bar for {k}. Run "
        f"`ORDER={CO.mode()} DATASETS={ds} SEEDS={seed} TASKS={T} python -u "
        f"exp16_full_table.py` first.")
    return v


# ------------------------------------------------------------------ member features
def member_features(ds, T, seed, spec):
    """(Ftr, Fte) for one member. q32 loads exp16's cache; everything else trains once.

    The training path mirrors fsa_train.train_task0's `ce` objective exactly, with the LoRA
    target modules, rank and class-bagging fraction lifted out as parameters. It is a copy
    rather than a call because train_task0 hardcodes lora_rank=32 with the default targets
    and has no bagging hook -- the same reason get_data is duplicated across exp16/fsa_train,
    and the same hazard, so any change here must be mirrored there."""
    if spec == "q32":
        f = os.path.join(
            REPO,
            f"exp16_feats_{ds}_T{T}_s{seed}_ep40_lr0.0003_aug1{CO.order_tag()}_{TAG}.npz")
        assert os.path.exists(f), (
            f"member q32 IS exp16's cache and it is missing: {f}\nRun "
            f"`ORDER={CO.mode()} DATASETS={ds} SEEDS={seed} TASKS={T} python -u "
            f"exp16_full_table.py` first.")
        z = np.load(f)
        return z["Ftr"], z["Fte"]

    cache = os.path.join(
        REPO, f"exp55_feats_{ds}_T{T}_s{seed}_{spec}_ep{EPOCHS}_lr{LR:g}"
              f"{CO.order_tag()}_{TAG}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"    cached member {spec}")
        return z["Ftr"], z["Fte"]

    targets, rank, bagfrac = parse_member(spec)
    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = F.get_data(ds)
    cpt = n_cls // T
    # Seed offset by the member index so members do not share an init, while the class ORDER
    # stays keyed on `seed` alone -- the task split must be identical across members or they
    # are not solving the same problem and nothing is comparable.
    torch.manual_seed(seed * 1000 + MEMBERS.index(spec))
    np.random.seed(seed)
    task0 = CO.class_order(n_cls, seed)[:cpt]
    if bagfrac < 1.0:
        keep = np.random.default_rng(10_000 + seed * 100 + MEMBERS.index(spec)).permutation(
            len(task0))[:max(2, int(round(bagfrac * len(task0))))]
        task0_tr = task0[np.sort(keep)]
    else:
        task0_tr = task0
    idx = np.where(np.isin(ytr, task0_tr))[0]
    remap = {int(c): i for i, c in enumerate(task0_tr)}

    model = load_backbone(F.MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=rank, lora_alpha=ALPHA,
                          lora_target_modules=targets, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))
    head = nn.Linear(model.num_features, len(task0_tr)).to(DEV)
    opt = torch.optim.AdamW(lp + list(head.parameters()), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    ld = DataLoader(Subset(tr_aug, idx.tolist()), batch_size=BS, shuffle=True,
                    num_workers=8, pin_memory=True)
    acc0 = 0.0
    for ep in range(EPOCHS):
        model.train()
        ok = tot = 0
        for x, lab in ld:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fx = model(x).float()
                loss = ce(head(fx), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lp + list(head.parameters()), GRAD_CLIP)
            opt.step()
            ok += int((head(fx).argmax(1) == y).sum())
            tot += len(y)
        sch.step()
        acc0 = ok / max(tot, 1)
    log(f"    trained member {spec} (targets={targets} rank={rank} "
        f"bag={bagfrac:g} cls={len(task0_tr)}) task0 train acc {acc0:.3f}")
    Ftr, Fte = F.extract(model, tr_ev), F.extract(model, te_ev)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    np.savez(cache, Ftr=Ftr, Fte=Fte, acc0=np.array(acc0))
    return Ftr, Fte


# ------------------------------------------------------------------ RanPAC, exp52 protocol
def ranpac_staged(Z_tr, Z_te, ytr, yte, T, seed, n_cls):
    """Staged RanPAC identical in protocol to exp52/54 -- same 10% per-task val carve, same
    lambda grid, same accumulation. Returns (accs, per-stage row-z-scored TEST logits,
    per-stage test row indices). The z-scored logits are what the score ensemble averages;
    z-scoring per row first is required because members have different logit scales."""
    d = Z_tr.shape[1]
    cpt = n_cls // T
    order = CO.class_order(n_cls, seed)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()

    def zs(A, seen):
        B = np.full(A.shape, -1e9, np.float64)
        sub = np.asarray(A[:, seen], np.float64)
        B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
        return B

    accs, zl, tidx = [], [], []
    for t in range(T):
        for i, h in _H(un(Z_tr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vix = VAL_ALL[:nval]
        tei = np.where(np.isin(yte, seen))[0]
        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            L = logits(un(Z_tr[vix]), Wm)
            a = float((np.asarray(seen)[L[:, seen].argmax(1)] == ytr[vix]).mean())
            if a > best:
                best, bw = a, Wm
        Lt = logits(un(Z_te), bw)[tei]
        accs.append(float((np.asarray(seen)[Lt[:, seen].argmax(1)] == yte[tei]).mean()))
        zl.append(zs(Lt, seen))
        tidx.append((tei, np.asarray(seen)))
    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return accs, zl, tidx


def run_cell(ds, T, seed, verify):
    feats = {}
    for spec in MEMBERS:
        Ftr, Fte = member_features(ds, T, seed, spec)
        feats[spec] = (un(Ftr), un(Fte))
    E.T, E.SEED = T, seed
    ytr, yte, n_cls = E.get_labels(ds)

    res, zls = {}, {}
    for spec in MEMBERS:
        a, zl, tidx = ranpac_staged(feats[spec][0], feats[spec][1], ytr, yte, T, seed, n_cls)
        res[spec] = a
        zls[spec] = zl
        log(f"    member {spec:<8} A-Last {a[-1]*100:.2f}  A-Avg {np.mean(a)*100:.2f}")

    Ztr = np.concatenate([feats[s][0] for s in MEMBERS], 1)
    Zte = np.concatenate([feats[s][1] for s in MEMBERS], 1)
    a, _, tidx = ranpac_staged(Ztr, Zte, ytr, yte, T, seed, n_cls)
    res["concat"] = a
    log(f"    concat   ({Ztr.shape[1]}d) A-Last {a[-1]*100:.2f}  A-Avg {np.mean(a)*100:.2f}")

    ens = []
    for t in range(T):
        tei, seen = tidx[t]
        S = sum(zls[s][t] for s in MEMBERS) / len(MEMBERS)
        ens.append(float((seen[S[:, seen].argmax(1)] == yte[tei]).mean()))
    res["ensemble"] = ens
    log(f"    ensemble           A-Last {ens[-1]*100:.2f}  A-Avg {np.mean(ens)*100:.2f}")

    # ---- diversity at the FINAL stage: the precondition for any of this working
    tei, seen = tidx[-1]
    yt = yte[tei]
    pred = {s: seen[zls[s][-1][:, seen].argmax(1)] for s in MEMBERS}
    corr = {s: (pred[s] == yt) for s in MEMBERS}
    div = {}
    for i, j in itertools.combinations(MEMBERS, 2):
        ci, cj = corr[i].astype(float), corr[j].astype(float)
        sd = ci.std() * cj.std()
        div[f"{i}|{j}"] = {
            "disagree": float((pred[i] != pred[j]).mean()),
            "errcorr": float(((ci - ci.mean()) * (cj - cj.mean())).mean() / sd)
            if sd > 1e-12 else float("nan"),
            "both_wrong": float(((~corr[i]) & (~corr[j])).mean())}
    anyc = np.zeros(len(yt), bool)
    for s in MEMBERS:
        anyc |= corr[s]
    div["ORACLE"] = float(anyc.mean())
    div["best_single"] = max(float(corr[s].mean()) for s in MEMBERS)

    # ---- IS THE HEADROOM CLASS-STRUCTURED OR ROW-RANDOM?
    # ORACLE picks a correct member PER ROW, which no deployable rule can do. The question
    # that decides whether a PER-CLASS combiner (e.g. cone/subspace rays over concatenated
    # member blocks, which weight each member differently for every class) can capture any
    # of it is whether the right member is a property of the CLASS or of the row.
    #   best_single      <= oracle_class_cv <= oracle_class <= ORACLE
    # oracle_class is fitted on the same test rows it scores, so it is an upper bound and is
    # badly inflated when classes have few test rows (IMAGENETAP has ~7). oracle_class_cv
    # cross-fits -- choose the member on half a class's rows, score the other half, both ways
    # -- and is the REALIZABLE number. Judge on the cv one.
    cls_list = np.unique(yt)
    winner, pc = {}, np.zeros(len(yt), bool)
    for c in cls_list:
        m = yt == c
        b = max(MEMBERS, key=lambda s: corr[s][m].mean())
        winner[int(c)] = b
        pc[m] = corr[b][m]
    rng = np.random.default_rng(0)
    half = np.zeros(len(yt), bool)
    for c in cls_list:
        ic = np.where(yt == c)[0]
        half[ic[rng.permutation(len(ic))[:len(ic) // 2]]] = True
    cv = np.zeros(len(yt), bool)
    for fit_m, ev_m in ((half, ~half), (~half, half)):
        for c in cls_list:
            mf, me = (yt == c) & fit_m, (yt == c) & ev_m
            if not me.any():
                continue
            b = (max(MEMBERS, key=lambda s: corr[s][mf].mean()) if mf.any() else MEMBERS[0])
            cv[me] = corr[b][me]
    div["oracle_class"] = float(pc.mean())
    div["oracle_class_cv"] = float(cv.mean())
    div["winner_counts"] = {s: int(sum(v == s for v in winner.values())) for s in MEMBERS}

    if verify:
        b = bar_for(ds, T, seed)
        assert abs(res["q32"][-1] - b["A_last"]) < 1e-6, (
            f"member q32 RanPAC {res['q32'][-1]:.6f} != exp16 bar {b['A_last']:.6f}. q32 is "
            f"supposed to BE exp16's cached features under exp52's read-out protocol; if it "
            f"disagrees, the protocol here has drifted and no delta is interpretable.")
        log("    VERIFY ok: member q32 reproduces the exp16 bar")

    out = {k: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
           for k, v in res.items()}
    out["_diversity"] = div
    out["_dim"] = {"per_member": int(feats[MEMBERS[0]][0].shape[1]),
                   "concat": int(Ztr.shape[1])}
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}"
                       f"|ep{EPOCHS}_lr{LR:g}_a{ALPHA:g}{CO.order_tag()}|m{M_RP}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 100
    # THIS FILTER USED TO CHECK ONLY p[3] (the member list) AND KEY ON (ds, seed).
    # The order tag lives in p[4] and T in p[1], so a pilot-order cell silently OVERWROTE the
    # legacy cell with the same (ds, seed) and the table averaged 1 pilot seed with 2 legacy
    # seeds while printing "cells 3". Two different class orders in one mean, invisibly.
    # Match the recipe+order field EXACTLY and key on (ds, T, seed).
    recipe = f"ep{EPOCHS}_lr{LR:g}_a{ALPHA:g}{CO.order_tag()}"
    cells, dropped = {}, 0
    for k, v in allres.items():
        p = k.split("|")
        if len(p) < 5 or p[3] != "+".join(MEMBERS):
            continue
        if p[4] != recipe:
            dropped += 1
            continue
        cells[(p[0], int(p[1]), int(p[2]))] = v
    dts = sorted({(d0, t0) for d0, t0, _ in cells})

    def seeds_of(ds, T):
        return sorted(s for (d0, t0, s) in cells if d0 == ds and t0 == T)

    print("\n" + "=" * W)
    print("EXP55 — LoRA member diversity (RanPAC only; no cones, no fusion, no beta)")
    print("=" * W)
    print(f"\nmembers {MEMBERS}   recipe {recipe}   ORDER={CO.mode()}   cells {len(cells)}"
          + (f"   ({dropped} cells in the JSON belong to a DIFFERENT recipe/order and were "
             f"excluded)" if dropped else ""))
    for ds, T in dts:
        print(f"    {ds} T={T}: seeds {seeds_of(ds, T)}")

    print(f"\n{'-'*W}\nPRECONDITION — is there anything to ensemble? (read before accuracy)"
          f"\n{'-'*W}")
    print(f"  {'ds':<12}{'pair':<18}{'disagree':>10}{'errcorr':>10}{'both_wrong':>12}")
    for ds, T in dts:
        sd_ = seeds_of(ds, T)
        pairs = [k for k in cells[(ds, T, sd_[0])]["_diversity"] if "|" in k]
        for pr in pairs:
            dv = [cells[(ds, T, s)]["_diversity"][pr] for s in sd_]
            print(f"  {ds:<12}{pr:<18}"
                  f"{np.mean([x['disagree'] for x in dv])*100:>9.1f}%"
                  f"{np.mean([x['errcorr'] for x in dv]):>10.2f}"
                  f"{np.mean([x['both_wrong'] for x in dv])*100:>11.1f}%")
    print(f"\n  {'ds':<12}{'best single':>13}{'per-class cv':>14}{'per-class*':>12}"
          f"{'ORACLE':>9}{'cv share':>10}   <- can a PER-CLASS rule reach it?")
    for ds, T in dts:
        sd_ = seeds_of(ds, T)
        D = [cells[(ds, T, s)]["_diversity"] for s in sd_]
        bs = np.mean([x["best_single"] for x in D]) * 100
        pcv = np.mean([x["oracle_class_cv"] for x in D]) * 100
        pc = np.mean([x["oracle_class"] for x in D]) * 100
        orc = np.mean([x["ORACLE"] for x in D]) * 100
        sh = (pcv - bs) / (orc - bs) if orc - bs > 1e-9 else float("nan")
        print(f"  {ds:<12}{bs:>13.2f}{pcv:>14.2f}{pc:>12.2f}{orc:>9.2f}{sh:>9.0%}")
    print("   * per-class fitted on the rows it scores = upper bound; `cv` cross-fits within"
          " each class and is the realizable number. `cv share` = (cv - best) / (ORACLE -"
          " best).")
    print(f"\n  {'ds':<12}per-class winner counts (which member is best on how many classes)")
    for ds, T in dts:
        sd_ = seeds_of(ds, T)
        wc = {s: int(np.mean([cells[(ds, T, sx)]["_diversity"]["winner_counts"][s]
                              for sx in sd_])) for s in MEMBERS}
        print(f"  {ds:<12}" + "  ".join(f"{s}:{n}" for s, n in wc.items()))

    # Published GR-LoRA numbers keyed by (dataset, T). SOURCE OF TRUTH IS exp16_full_table.REF
    # (ICML'26 Tables 1,2,6, ViT-B/16-IN21k, mean of 3 seeds). exp16 keys these as IMAGENETA
    # and CUB200; here they are mapped onto the PILOT-CORRECTED variants IMAGENETAP / CUB200P,
    # which are the splits those published numbers actually correspond to -- see
    # splits.py and the CUB/ImageNet-A note in exp16.get_data. The dict was previously keyed by
    # DATASET ALONE, which printed the T=10 baseline against every task count.
    REF = {("CIFAR100", 10): (91.97, 94.65), ("CIFAR100", 20): (91.46, 94.41),
           ("CIFAR100", 50): (90.03, 93.38),
           ("IMAGENETR", 10): (82.09, 86.20), ("IMAGENETR", 20): (80.23, 85.05),
           ("IMAGENETR", 50): (76.74, 82.64),
           ("IMAGENETAP", 10): (63.60, 70.24), ("IMAGENETAP", 20): (62.37, 69.30),
           ("IMAGENETAP", 50): (59.71, 67.23),
           ("CUB200P", 10): (89.91, 93.85), ("CUB200P", 20): (89.76, 94.08),
           ("CUB200P", 50): (89.68, 93.94)}
    names = MEMBERS + ["concat", "ensemble"]
    for fld, lbl in (("A_last", "A-Last"), ("A_avg", "A-Avg")):
        print(f"\n{'-'*W}\n{lbl} — RanPAC on each feature set, mean over seeds\n{'-'*W}")
        print(f"  {'ds':<12}{'T':>4}{'sd':>4}" + "".join(f"{n:>12}" for n in names)
              + f"{'concat-q32':>13}{'GR-LoRA':>10}{'ens-GR':>9}")
        for ds, T in dts:
            sd_ = seeds_of(ds, T)
            g = {n: np.mean([cells[(ds, T, s)][n][fld] for s in sd_]) * 100 for n in names}
            so = REF.get((ds, T), (float("nan"),) * 2)[0 if fld == "A_last" else 1]
            print(f"  {ds:<12}{T:>4}{len(sd_):>4}"
                  + "".join(f"{g[n]:>12.2f}" for n in names)
                  + f"{g['concat']-g['q32']:>+13.2f}{so:>10.2f}"
                  + f"{g['ensemble']-so:>+9.2f}")

    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  1. PRECONDITION FIRST. `headroom` = ORACLE - best_single is the ceiling for ANY combiner:
     it is the accuracy of an oracle that picks a correct member whenever one exists. A
     headroom near zero means the members are redundant and NO fusion recovers anything --
     the answer is then harsher diversity (bagging, disjoint target modules, different
     ranks) or a different mechanism entirely, NOT a cleverer combination rule. That is
     exactly the trap exp54 fell into at the rule level.
  2. `disagree` near 0 with `errcorr` near 1 means the members are the same network wearing
     different hats. Expected for members differing only by init, which is precisely why the
     default member set varies TARGET MODULES instead.
  3. `concat - q32` is the Tier 2 verdict. Our IMAGENETR base deficit is -0.89 A-Avg, so
     under about +0.5 means concatenation does not close it and nothing built on top will.
  4. `ensemble` vs `concat` separates the two mechanisms: score averaging tests
     decorrelation, concatenation tests representational richness. They can disagree, and
     which one wins says what the members are actually contributing.
  5. Members are NOT expected to be individually equal. `m32` adapting only the MLP may well
     be worse alone than `q32`; a worse-but-decorrelated member is useful and that is the
     whole premise. Judge members by the diversity table, not by their solo accuracy.""")
    print("=" * W)
    log(f"wrote {OUT}")
