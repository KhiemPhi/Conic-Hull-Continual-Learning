#!/usr/bin/env python3
"""exp32_cosine_head.py — stop training with a head we throw away and never deploy.

THE MISMATCH
    A_plus trains LoRA against a throwaway Linear(768, cpt) softmax head, discards it, and
    then deploys NCM / RanPAC / prototypes / cones. A linear softmax optimises for linear
    separability WITH a per-class bias and scale it is free to learn; every reader we
    actually deploy is scale-free and bias-free and wants compact, equidistant clusters.
    The training objective and the deployed objective are not the same objective.

ARMS
    ce       Linear(768, cpt) + CE                        the exp16 baseline
    cosine   logits = s * cos(phi, w_c), w_c learnable     scale-free, bias-free
    proto    logits = s * cos(phi, batch class mean)       prototypical; head has NO parameters
             ^ the closest training-time match to an NCM/RanPAC reader

    `proto` is the sharper test: if a head with zero parameters trains better features than a
    learned linear head, the mismatch is real and not just a normalisation detail.

CHEAPEST OF THE THREE
    No teacher model, no episodes, no extra forward pass -- the same cost as the baseline.
    If it moves A-Last at all it is free accuracy, and it composes with exp30 and exp31.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 SCALE=16 python -u exp32_cosine_head.py
"""
import json, os
import numpy as np
import fsa_train as F

DS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
SCALES = [float(x) for x in os.environ.get("SCALE", "16").split(",")]
ARMS = os.environ.get("ARMS", "ce,cosine,proto").split(",")
OUT = os.path.join(F.REPO, f"exp32_cosine_head_{F.TAG}.json")

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DS:
    for T in TS:
        for seed in SEEDS:
            _, _, ytr, _, yte, n_cls = F.get_data(ds)
            for arm in ARMS:
                for s in (SCALES if arm != "ce" else [0.0]):
                    key = f"{ds}|{T}|{seed}|{arm}" + ("" if arm == "ce" else f"_s{s:g}")
                    if key in res:
                        F.log(f"skip {key}"); continue
                    Ftr, Fte, a0 = F.train_task0(
                        ds, T, seed,
                        "ce" if arm == "ce" else "cosine",
                        tag_extra=("" if arm == "ce" else
                                   f"_{'proto' if arm == 'proto' else 'cos'}_s{s:g}"),
                        head_scale=s, proto=(arm == "proto"))
                    accs = F.replay(Ftr, ytr, Fte, yte, T, seed, n_cls)
                    res[key] = {"A_last": accs[-1], "A_avg": float(np.mean(accs)),
                                "acc0": a0, "accs": accs}
                    json.dump(res, open(OUT, "w"), indent=2)
                    F.log(f"  {key}: A-Last {accs[-1]*100:.2f}  A-Avg "
                          f"{np.mean(accs)*100:.2f}  task0 {a0:.3f}")

print("\n" + "=" * 78)
print(f"EXP32 — training head: linear-CE vs cosine vs prototypical ({F.MODEL})")
print("=" * 78)
for ds in DS:
    for T in TS:
        for seed in SEEDS:
            b = F.bar_for(ds, T, seed)
            print(f"\n--- {ds} T={T} s={seed}" +
                  (f"   [exp16 bar: A-Last {b['A_last']*100:.2f}]" if b else ""))
            print(f"{'arm':>14}{'A-Last':>9}{'A-Avg':>9}{'task0':>8}{'vs ce':>9}")
            z = res.get(f"{ds}|{T}|{seed}|ce")
            for k, r in sorted(res.items()):
                if not k.startswith(f"{ds}|{T}|{seed}|"): continue
                nm = k.split("|")[-1]
                d = f"{(r['A_last']-z['A_last'])*100:>+9.2f}" if z and nm != "ce" else " " * 9
                print(f"{nm:>14}{r['A_last']*100:>9.2f}{r['A_avg']*100:>9.2f}"
                      f"{r['acc0']:>8.3f}{d}")
print("\n`ce` shares the harness with exp30/exp31 and should match the exp16 bar. If it does")
print("not, fsa_train has diverged from exp16 and no cross-arm number here is meaningful.")
print("=" * 78)
