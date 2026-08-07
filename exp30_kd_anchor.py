#!/usr/bin/env python3
"""exp30_kd_anchor.py — a distillation anchor to the PTM, replacing the falsified 0.98 rule.

WHY
    exp12 found A_plus is single-peaked in task-0 TRAIN accuracy: over-fitting the first task
    costs accuracy on classes 20-200. exp16's answer was to early-stop at 0.98, and we
    falsified it hard -- 10/12 cells worse, aggregate -15.94 -> -20.04, CIFAR stopping at
    epoch 3-8 because train accuracy saturates long before the representation is good.
    Task-0 train accuracy is a MARKER, not a lever.

    The intent behind that rule was "do not drift too far from the PTM". This is the
    continuous, principled version of the same intent -- a proximal term rather than a
    stopping heuristic:

        L = CE(head(phi_lora(x)), y) + lam * (1 - cos(phi_lora(x), phi_frozen(x)))

    PLASTIC uses teacher-student KL at TEST time for stability; this is the training-time
    analogue, and it costs nothing at inference.

INVARIANT
    lam=0 must reproduce the plain CE arm bit-for-bit. It is in the sweep for that reason,
    and because it means the sweep cannot lose except by over-regularising.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 LAMS=0,0.1,0.5,1,2 python -u exp30_kd_anchor.py
"""
import json, os
import numpy as np
import fsa_train as F

DS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
LAMS = [float(x) for x in os.environ.get("LAMS", "0,0.1,0.5,1,2").split(",")]
OUT = os.path.join(F.REPO, f"exp30_kd_anchor_{F.TAG}.json")

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DS:
    for T in TS:
        for seed in SEEDS:
            _, _, ytr, _, yte, n_cls = F.get_data(ds)
            for lam in LAMS:
                key = f"{ds}|{T}|{seed}|lam{lam:g}"
                if key in res:
                    F.log(f"skip {key}"); continue
                obj = "ce" if lam == 0 else "ce_kd"
                Ftr, Fte, a0 = F.train_task0(ds, T, seed, obj,
                                             tag_extra=("" if lam == 0 else f"_kd{lam:g}"),
                                             lam_kd=lam)
                accs = F.replay(Ftr, ytr, Fte, yte, T, seed, n_cls)
                res[key] = {"A_last": accs[-1], "A_avg": float(np.mean(accs)),
                            "acc0": a0, "accs": accs}
                json.dump(res, open(OUT, "w"), indent=2)
                F.log(f"  {key}: A-Last {accs[-1]*100:.2f}  A-Avg {np.mean(accs)*100:.2f}"
                      f"  task0 {a0:.3f}")

print("\n" + "=" * 76)
print(f"EXP30 — PTM distillation anchor during first-session adaptation ({F.MODEL})")
print("=" * 76)
for ds in DS:
    for T in TS:
        for seed in SEEDS:
            b = F.bar_for(ds, T, seed)
            print(f"\n--- {ds} T={T} s={seed}" +
                  (f"   [exp16 bar: A-Last {b['A_last']*100:.2f} A-Avg {b['A_avg']*100:.2f}]"
                   if b else ""))
            print(f"{'lambda':>8}{'A-Last':>9}{'A-Avg':>9}{'task0':>8}{'vs lam=0':>10}")
            z = res.get(f"{ds}|{T}|{seed}|lam0")
            for lam in LAMS:
                r = res.get(f"{ds}|{T}|{seed}|lam{lam:g}")
                if not r: continue
                d = f"{(r['A_last']-z['A_last'])*100:>+10.2f}" if z else " " * 10
                print(f"{lam:>8g}{r['A_last']*100:>9.2f}{r['A_avg']*100:>9.2f}"
                      f"{r['acc0']:>8.3f}{d}")
print("\nlam=0 IS the plain CE arm and should match the exp16 bar; if it does not, the")
print("harness diverges from exp16 and nothing else here is comparable.")
print("=" * 76)
