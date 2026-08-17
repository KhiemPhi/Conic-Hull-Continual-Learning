#!/usr/bin/env python3
"""exp65_backbone_selection.py -- P1 ONLY: does backbone ranking REORDER across adaptation
budgets? This is the cheap kill-shot for the whole budget-aware-backbone-selection line.

WHAT THIS FILE IS AND IS NOT
    It is NOT about cones. It measures GROUND TRUTH ONLY: accuracy A(b, d, beta) for a grid
    of backbones b, datasets d and adaptation budgets beta. No transferability score of any
    kind is computed here.

    That is deliberate. The proposed research question is "a few-shot conic descriptor picks
    the backbone for your dataset AT YOUR BUDGET". The novel half of that claim -- the budget
    conditioning -- has a PREMISE that can be falsified without building a single predictor:

        if the ranking of backbones is the SAME at every budget, then budget-conditioning
        buys nothing, one ranking serves all budgets, and the question collapses to classical
        transferability estimation (LogME / GBC / LEEP / H-score), where the cone would have
        to win head-on against methods built for exactly that job -- and where three prior
        results in this repo say it will not (see PRIOR below).

    So the ordering is: settle the premise, THEN decide whether to build the predictor. Doing
    it the other way round is how you spend GPU-weeks scoring a question that was already
    answered by its own setup.

THE PRIOR, STATED HONESTLY BEFORE THE RUN
    Three closures bear on the conic half of this line and all three are negative:
      cone-is-dead-weight-final          removing the cone IMPROVED the method (0.7910 vs
                                         0.7903); cone alone 0.6333 < never-adapt 0.6867.
      cone-diagnostic-instrument-fails   as an instrument the cone is not monotone in
                                         multimodality, blind to label noise, chance on
                                         subgroups -- and K-MEANS WAS THE BETTER INSTRUMENT.
      conic-rule-value-scales-with-rays  the conic gain is monotone in ray count R
                                         (+0.08 / +0.45 / +0.75 at R = 4 / 16 / 64).
    The last one is the sharpest warning for a FEW-SHOT selector: at K shots/class the hull
    gets at most K-2 rays, which is the regime where the cone was measured to be worth ~zero
    over multi-prototype. The one honest reason to reopen is that all three asked the cone to
    CLASSIFY, whereas a selector only has to RANK -- a descriptor can be a poor classifier and
    still a good ordinal statistic. That is a real distinction, and it is a narrow one.

    Estimated before the run: P(the conic predictor beats k-means at matched budget) ~ 0.2.
    P(the ranking reorders across budgets, i.e. THIS file returns alive) ~ 0.6-0.7. Note the
    second does not depend on cones at all, which is the likely shape of the surviving result.

THE PRE-REGISTERED TEST
    reorder_rate = fraction of (dataset, budget-pair) cells whose TOP-1 backbone changes.

        reorder_rate >= 0.20   -> ALIVE. Budget conditioning is a real axis; a budget-blind
                                  score is leaving accuracy on the table and the selector is
                                  worth building.
        reorder_rate <  0.10   -> DEAD. One ranking serves every budget. Drop the budget
                                  framing and either compete with LogME head-on or stop.
        0.10 - 0.20            -> INCONCLUSIVE, report as such, do not round in our favour.

    The practically meaningful companion number, reported alongside and arguably the one a
    practitioner cares about:

        regret(frozen -> beta) = A(b*_beta, d, beta) - A(b*_frozen, d, beta)   in acc points

    i.e. what you LOSE by choosing your backbone with a cheap frozen probe and then training
    at budget beta. Top-1 identity can be stable while regret is large (or vice versa), so
    both are pre-registered. A reorder_rate under 0.10 WITH a regret over 1.0 point would mean
    the interesting variation is below the top-1 rank and the metric, not the premise, is wrong.

WHY THESE SIX BACKBONES
    Architecture is HELD FIXED at ViT-B and only the PRETRAINING OBJECTIVE varies. Mixing in a
    ConvNeXt would confound "objective reorders rankings" with "architecture reorders
    rankings", and worse, inject_lora targets attn.qkv/attn.proj which a ConvNeXt does not
    have -- it would silently replace ZERO layers and every LoRA budget would be a duplicate
    of frozen. All six inject exactly 24 LoRA layers; that is asserted at build time.

    in21k   supervised IN-21k            the repo's incumbent, and the in-domain reference
    in1k    supervised IN-21k -> IN-1k   same data, extra supervised fine-tune
    clip    image-text contrastive       different objective, web-scale data
    siglip  sigmoid image-text           contrastive but a different loss/data mix
    dinov2  self-supervised distillation no labels, strong frozen features by reputation
    mae     masked autoencoding          notoriously WEAK frozen, strong fine-tuned --
                                         the single most likely source of a reorder, and the
                                         reason a null result here would be genuinely
                                         surprising rather than merely disappointing

    DINOv2 is patch14 at a native 518px, which is ~7x the tokens of the others; it is built
    with img_size=224 so every backbone sees the same resolution and the same compute. The
    data config is overridden to match, because timm's pretrained_cfg keeps reporting 518.

THE BUDGET LADDER
    frozen   backbone frozen, linear head only          0 backbone params
    lora4    LoRA r=4 on attn.qkv/attn.proj + head      ~0.15M
    lora32   LoRA r=32 on attn.qkv/attn.proj + head     ~1.2M
    full     everything                                 ~86M

LEARNING RATE IS SWEPT, AND THAT IS NOT OPTIONAL
    A single shared lr across six pretraining objectives does not measure "which backbone is
    better", it measures "which backbone likes this lr". MAE in particular needs a different
    operating point from CLIP. Every (backbone, dataset, budget) gets its own lr chosen on a
    VAL split carved from train; test is scored once at the chosen lr. Skipping this is the
    standard way transferability benchmarks manufacture rankings that do not replicate.

SPLITS COME FROM splits.py, AND THE LABELS ARE CROSS-CHECKED
    get_data is already duplicated across exp16/fsa_train and splits.py exists because that
    duplication silently detaches labels from features. This file builds its own loaders (it
    must -- each backbone needs its OWN transform, and fsa_train resolves ONE transform at
    import time from $MODEL), so it imports splits.split_indices for the split itself and
    ASSERTS its label arrays against exp19_dataset_hull.get_labels before training anything.

USAGE
    source ~/venvs/ml_env/bin/activate

    # smoke: 1 epoch, 2 backbones, 1 dataset, 2 budgets, 1 lr -- checks the harness only
    DS=CUB200P BACKBONES=in21k,mae BUDGETS=frozen,lora4 EPOCHS=1 LRS_OVERRIDE=3e-4 \
      SUFFIX=_smoke python -u exp65_backbone_selection.py

    # stage 1: the P1 grid on the two mid-size datasets
    DS=IMAGENETR,CUB200P BACKBONES=in21k,in1k,clip,siglip,dinov2,mae \
      BUDGETS=frozen,lora4,lora32,full EPOCHS=10 python -u exp65_backbone_selection.py

    # stage 2: add the remaining two datasets (CIFAR100 is 50k images and dominates the cost)
    DS=CIFAR100,IMAGENETAP ... python -u exp65_backbone_selection.py

    Resumable: every (backbone, dataset, budget, lr) cell is written as it finishes and
    existing keys are skipped, so the grid can be interrupted and continued.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import timm
from timm.data import create_transform, resolve_model_data_config

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# exp19_dataset_hull reads its OWN grid from the environment at import time and parses two
# of those names as numbers:  T and SEED as scalar ints, BUDGETS as a list of ints. Our
# BUDGETS is a list of NAMES ("frozen,lora4,..."), so importing with it set raises
#     ValueError: invalid literal for int() with base 10: 'frozen'
# before a line of this file runs. The colliding name is captured and popped for the duration
# of the import, then restored -- the alternative, renaming our variable, would leave the
# trap armed for the next file that imports exp19 while using the obvious word for a budget.
_BUDGETS_ENV = os.environ.pop("BUDGETS", None)
os.environ.setdefault("T", "10")
os.environ.setdefault("SEED", "0")

# Six sets of pretrained weights come from the HF hub on first use. Two devserver-specific
# facts, both discovered the hard way and neither of them optional:
#   1. external fetches need fwdproxy (same as download_datasets._set_proxy);
#   2. huggingface_hub's default Xet transfer backend does NOT work through it -- it fails in
#      the CAS client with "error sending request for url https://cas-server.xethub.hf.co/..."
#      AFTER the proxy is set, which reads like a proxy problem and is not one. Forcing the
#      plain-HTTP path fixes it. Weights are cached, so this matters on the first run only.
for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import exp19_dataset_hull as E                                   # noqa: E402
import fsa_train as F                                            # noqa: E402
from backbone import freeze_non_lora, inject_lora                # noqa: E402
from splits import split_indices, stratified_indices             # noqa: E402

if _BUDGETS_ENV is not None:
    os.environ["BUDGETS"] = _BUDGETS_ENV

REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

BACKBONES = {
    "in21k":  ("vit_base_patch16_224.augreg_in21k", {}),
    "in1k":   ("vit_base_patch16_224.augreg2_in21k_ft_in1k", {}),
    "clip":   ("vit_base_patch16_clip_224.openai", {}),
    "siglip": ("vit_base_patch16_siglip_224.webli", {}),
    "dinov2": ("vit_base_patch14_dinov2.lvd142m", {"img_size": 224}),
    "mae":    ("vit_base_patch16_224.mae", {}),
}

# Per-budget lr grids. Ranges differ by budget because they must: a linear head wants ~1e-2,
# a full fine-tune of a pretrained ViT wants ~1e-5, and one shared grid would hand every
# budget a badly-scaled operating point and call the result a backbone ranking.
LR_GRID = {"frozen": [1e-3, 3e-3, 1e-2],
           "lora4":  [1e-4, 3e-4, 1e-3],
           "lora32": [1e-4, 3e-4, 1e-3],
           "full":   [1e-5, 3e-5, 1e-4]}

DSETS = os.environ.get("DS", "IMAGENETR,CUB200P").split(",")
BTAGS = os.environ.get("BACKBONES", ",".join(BACKBONES)).split(",")
BUDGETS = os.environ.get("BUDGETS", "frozen,lora4,lora32,full").split(",")
EPOCHS = int(os.environ.get("EPOCHS", 10))
SEED = int(os.environ.get("SEED_TRAIN", 0))
BS = int(os.environ.get("BS", 128))
WD = float(os.environ.get("WD", 0.05))
GRAD_CLIP = 1.0
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", 4.0))
LORA_TARGETS = ["attn.qkv", "attn.proj"]
VAL_FRAC = 0.9                                   # 90/10 train/val carve for lr selection
_LRO = os.environ.get("LRS_OVERRIDE", "")
LRS_OVERRIDE = [float(x) for x in _LRO.split(",")] if _LRO else None
OUT = os.path.join(REPO, f"exp65_backbone_selection{os.environ.get('SUFFIX', '')}.json")

assert all(b in BACKBONES for b in BTAGS), f"unknown backbone in {BTAGS}; have {list(BACKBONES)}"
assert all(b in LR_GRID for b in BUDGETS), f"unknown budget in {BUDGETS}; have {list(LR_GRID)}"


# ------------------------------------------------------------------ data
def build_datasets(name, tf_train, tf_eval):
    """(train_aug, train_eval, ytr, test_eval, yte, n_cls) under exp16's splits.

    Mirrors exp16_full_table.get_data / fsa_train.get_data in WHICH IMAGES it selects, but
    takes the transforms as arguments -- fsa_train bakes in one transform resolved at import
    from $MODEL, and this file needs a different one per backbone. The split itself is not
    reimplemented: it comes from splits.split_indices, and the labels are asserted against
    exp19_dataset_hull.get_labels by the caller."""
    from datasets import load_dataset
    if name == "CIFAR100":
        from torchvision import datasets as tvd
        tr = tvd.CIFAR100(os.path.join(REPO, "data"), train=True, download=False)
        te = tvd.CIFAR100(os.path.join(REPO, "data"), train=False, download=False)
        ytr, yte = np.array(tr.targets), np.array(te.targets)
        return (F.TVWrap(tr, ytr, tf_train), F.TVWrap(tr, ytr, tf_eval), ytr,
                F.TVWrap(te, yte, tf_eval), yte, 100)

    if name == "IMAGENETR":
        d = load_dataset("axiong/imagenet-r", cache_dir=os.path.join(REPO, "data/hf"))["test"]
        w = np.array(d["wnid"])
        lab = np.searchsorted(np.array(sorted(set(w.tolist()))), w)
    elif name in ("IMAGENETA", "IMAGENETAP"):
        d = load_dataset("barkermrl/imagenet-a",
                         cache_dir=os.path.join(REPO, "data/hf"))["train"]
        lab = np.array(d["label"])
    elif name == "CUB200P":
        from datasets import concatenate_datasets
        dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
        d = concatenate_datasets([dd["train"], dd["test"]])
        lab = np.array(d["label"])
    else:
        raise ValueError(f"unknown dataset {name!r}")

    tri, tei = split_indices(name, lab)
    n = int(lab.max()) + 1
    return (F.HFWrap(d, tri, lab[tri], tf_train), F.HFWrap(d, tri, lab[tri], tf_eval), lab[tri],
            F.HFWrap(d, tei, lab[tei], tf_eval), lab[tei], n)


def transforms_for(btag):
    """Each backbone's OWN preprocessing. Feeding CLIP's normalisation to DINOv2 would be a
    silent accuracy tax on one arm of the comparison, which is the whole measurement."""
    spec, kw = BACKBONES[btag]
    probe = timm.create_model(spec, pretrained=False, num_classes=0, **kw)
    cfg = resolve_model_data_config(probe)
    if "img_size" in kw:
        # timm's pretrained_cfg still reports the native resolution (518 for DINOv2) even
        # after img_size=224 is honoured by patch_embed, so the transform would build 518px
        # crops for a model expecting 224 and every DINOv2 number would be garbage.
        s = kw["img_size"]
        cfg["input_size"] = (cfg["input_size"][0], s, s)
    del probe
    tf_eval = create_transform(**cfg, is_training=False)
    tf_train = create_transform(**cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                                re_prob=0.25, scale=(0.7, 1.0), hflip=0.5)
    return tf_train, tf_eval, cfg


# ------------------------------------------------------------------ model
def make_model(btag, budget, n_cls):
    """Backbone at the given adaptation budget, head always trainable.

    Returns (model, n_trainable). The n_trainable count IS the budget axis and is recorded
    with every cell -- "lora4" is a label, the parameter count is the quantity."""
    spec, kw = BACKBONES[btag]
    model = timm.create_model(spec, pretrained=True, num_classes=n_cls, **kw)

    if budget == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
    elif budget.startswith("lora"):
        rank = int(budget[4:])
        model, n_rep = inject_lora(model, rank=rank, alpha=LORA_ALPHA,
                                   target_modules=LORA_TARGETS)
        assert n_rep > 0, (
            f"{btag}: inject_lora replaced ZERO layers for targets {LORA_TARGETS}. The "
            f"'{budget}' budget would be identical to 'frozen' and the budget ladder would "
            f"silently collapse. This is the ConvNeXt failure the backbone set avoids.")
        freeze_non_lora(model)
    elif budget != "full":
        raise ValueError(budget)

    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model.to(DEV), n_tr


@torch.no_grad()
def evaluate(model, ds, rows=None):
    from torch.utils.data import DataLoader, Subset
    model.eval()
    sub = ds if rows is None else Subset(ds, np.asarray(rows).tolist())
    ld = DataLoader(sub, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    hit = tot = 0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        for x, y in ld:
            p = model(x.to(DEV, non_blocking=True)).float().argmax(1).cpu()
            hit += int((p == y).sum()); tot += len(y)
    return hit / max(tot, 1)


def train_one(btag, ds_name, budget, lr, data, fit_rows):
    """Train at one (budget, lr) and return (val_acc, test_acc, n_trainable, seconds)."""
    from torch.utils.data import DataLoader, Subset
    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = data
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, n_tr = make_model(btag, budget, n_cls)

    ld = DataLoader(Subset(tr_aug, fit_rows.tolist()), batch_size=BS, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS * len(ld), 1))
    lossf = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        # A frozen backbone stays in eval mode: with only the head trainable, train-mode
        # dropout/stochastic-depth would inject noise into features that no gradient can
        # compensate for, penalising exactly the backbones with the most regularisation.
        model.train() if budget != "frozen" else model.eval()
        for x, y in ld:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = lossf(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step(); sched.step()

    val_rows = np.setdiff1d(np.arange(len(ytr)), fit_rows)
    va = evaluate(model, tr_ev, val_rows)
    ta = evaluate(model, te_ev)
    del model, opt
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return va, ta, n_tr, time.time() - t0


# ------------------------------------------------------------------ summary
def kendall(a, b):
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic)


def summarise(res):
    W = 100
    print("\n" + "=" * W)
    print("EXP65 P1 -- does backbone ranking REORDER across adaptation budgets?")
    print("=" * W)
    print(f"\nbackbones {BTAGS}\nbudgets {BUDGETS}  datasets {DSETS}  epochs {EPOCHS}  "
          f"seed {SEED}")

    # sel[(ds, budget, btag)] = (test_acc, chosen_lr, val_acc); lr chosen on VAL, never test.
    sel = {}
    for ds in DSETS:
        for bu in BUDGETS:
            for bt in BTAGS:
                cand = [(v["val"], v["test"], v["lr"]) for k, v in res.items()
                        if v["ds"] == ds and v["budget"] == bu and v["backbone"] == bt]
                if cand:
                    va, ta, lr = max(cand)
                    sel[(ds, bu, bt)] = (ta, lr, va)

    print(f"\n{'-'*W}\nTEST ACCURACY (lr chosen on val)   * = top-1 at that budget\n{'-'*W}")
    for ds in DSETS:
        print(f"\n  {ds}")
        print(f"    {'budget':<9}" + "".join(f"{b:>12}" for b in BTAGS) + f"{'top-1':>9}")
        for bu in BUDGETS:
            row = [sel.get((ds, bu, bt)) for bt in BTAGS]
            if not any(row):
                continue
            accs = [r[0] * 100 if r else float("nan") for r in row]
            best = int(np.nanargmax(accs)) if not all(np.isnan(accs)) else -1
            cells = "".join(f"{a:>11.2f}{'*' if i == best else ' '}"
                            for i, a in enumerate(accs))
            print(f"    {bu:<9}{cells}{BTAGS[best] if best >= 0 else '--':>9}")
        print(f"    {'lr':<9}" + "".join(
            f"{(f'{sel[(ds, BUDGETS[0], bt)][1]:g}' if (ds, BUDGETS[0], bt) in sel else '--'):>12}"
            for bt in BTAGS) + f"   (at {BUDGETS[0]})")

    # ---- the pre-registered statistic
    print(f"\n{'-'*W}\nPRE-REGISTERED: reorder rate over (dataset, budget-pair) cells\n{'-'*W}")
    pairs, flips = [], 0
    for ds in DSETS:
        for i, b1 in enumerate(BUDGETS):
            for b2 in BUDGETS[i + 1:]:
                r1 = [sel.get((ds, b1, bt)) for bt in BTAGS]
                r2 = [sel.get((ds, b2, bt)) for bt in BTAGS]
                if not all(r1) or not all(r2):
                    continue
                a1 = [r[0] for r in r1]; a2 = [r[0] for r in r2]
                t1, t2 = BTAGS[int(np.argmax(a1))], BTAGS[int(np.argmax(a2))]
                tau = kendall(a1, a2)
                flip = t1 != t2
                flips += flip
                pairs.append((ds, b1, b2, t1, t2, tau, flip))
                print(f"  {ds:<11}{b1:>7} -> {b2:<7}  top1 {t1:>7} -> {t2:<7}"
                      f"  tau {tau:>+6.2f}   {'REORDER' if flip else 'stable'}")
    rate = flips / len(pairs) if pairs else float("nan")
    verdict = ("ALIVE -- budget conditioning is real" if rate >= 0.20 else
               "DEAD -- one ranking serves every budget" if rate < 0.10 else
               "INCONCLUSIVE")
    print(f"\n  reorder_rate = {flips}/{len(pairs)} = {rate:.2f}   "
          f"(ALIVE >= 0.20, DEAD < 0.10)  -> {verdict}")

    # ---- regret of choosing with a cheap frozen probe, then training anyway
    if "frozen" in BUDGETS:
        print(f"\n{'-'*W}\nREGRET of picking the backbone by FROZEN accuracy, then training at "
              f"budget beta\n{'-'*W}")
        print(f"  {'ds':<11}{'budget':<9}{'best':>9}{'frozen-pick':>13}{'regret (pts)':>14}")
        regs = []
        for ds in DSETS:
            fr = [sel.get((ds, "frozen", bt)) for bt in BTAGS]
            if not all(fr):
                continue
            pick = BTAGS[int(np.argmax([r[0] for r in fr]))]
            for bu in BUDGETS:
                if bu == "frozen":
                    continue
                row = [sel.get((ds, bu, bt)) for bt in BTAGS]
                if not all(row):
                    continue
                best = max(r[0] for r in row) * 100
                got = sel[(ds, bu, pick)][0] * 100
                regs.append(best - got)
                print(f"  {ds:<11}{bu:<9}{best:>9.2f}{got:>13.2f}{best-got:>14.2f}")
        if regs:
            print(f"\n  mean regret {np.mean(regs):>.2f} pts   max {np.max(regs):>.2f} pts")

    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  1. reorder_rate IS the result. It decides whether a budget-CONDITIONED selector can beat a
     budget-blind one even in principle. Nothing about cones is measured here.
  2. REGRET is the practitioner's version and can disagree with reorder_rate: a stable top-1
     with large regret means the action is below rank 1, and the metric needs rethinking
     before the premise does.
  3. Kendall tau is the full-ranking companion to the top-1 flip. tau near +1 with a flip
     means two near-tied backbones swapped and the reorder is cosmetic; tau near 0 with no
     flip means the tail reordered violently under a stable winner. Read both.
  4. If this returns DEAD, the honest next step is NOT to re-cut the metric until it returns
     ALIVE. It is to drop the budget framing and ask whether a conic score beats LogME at a
     fixed budget -- a question three prior closures say we should expect to lose.""")
    print("=" * W)


# ------------------------------------------------------------------ driver
if __name__ == "__main__":
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    log(f"device {DEV}  out {os.path.basename(OUT)}  resuming from {len(res)} cells")

    for ds_name in DSETS:
        # Labels are dataset-level, not backbone-level: cross-check ONCE per dataset against
        # exp19's loader, which is what every cached feature file in this repo is aligned to.
        e_ytr, e_yte, e_n = E.get_labels(ds_name)
        for bt in BTAGS:
            tf_train, tf_eval, cfg = transforms_for(bt)
            data = build_datasets(ds_name, tf_train, tf_eval)
            tr_aug, tr_ev, ytr, te_ev, yte, n_cls = data
            assert np.array_equal(ytr, e_ytr) and np.array_equal(yte, e_yte), (
                f"{ds_name}: label arrays disagree with exp19_dataset_hull.get_labels. The "
                f"split in this file has drifted from the canonical one and every accuracy "
                f"below would be measured on mismatched rows. See splits.py.")
            assert n_cls == e_n

            # 90/10 stratified carve, keyed on the dataset only -- identical for every
            # backbone and budget, or the lr selection is done on different data per arm.
            fit_rows, _ = stratified_indices(ytr, frac=VAL_FRAC)
            log(f"=== {ds_name} [{bt}] {BACKBONES[bt][0]}  n_cls {n_cls}  "
                f"train {len(ytr)} (fit {len(fit_rows)})  test {len(yte)}  "
                f"input {cfg['input_size']}")

            for bu in BUDGETS:
                for lr in (LRS_OVERRIDE or LR_GRID[bu]):
                    key = f"{bt}|{ds_name}|{bu}|lr{lr:g}|ep{EPOCHS}|bs{BS}|s{SEED}|v1"
                    if key in res:
                        log(f"    skip {key}")
                        continue
                    va, ta, n_tr, secs = train_one(bt, ds_name, bu, lr, data, fit_rows)
                    res[key] = {"backbone": bt, "model": BACKBONES[bt][0], "ds": ds_name,
                                "budget": bu, "lr": lr, "val": va, "test": ta,
                                "n_trainable": n_tr, "epochs": EPOCHS, "seed": SEED,
                                "secs": round(secs, 1)}
                    json.dump(res, open(OUT, "w"), indent=2)
                    log(f"    {bu:<7} lr {lr:<7g} val {va*100:6.2f}  test {ta*100:6.2f}  "
                        f"({n_tr/1e6:.2f}M trainable, {secs:.0f}s)")

    summarise(res)
    log(f"wrote {OUT}")
