#!/usr/bin/env python3
"""exp31_subspace_train.py — train phi so that R samples of a class SPAN the class.

WHY THIS OBJECTIVE AND NOT ANOTHER
    exp29 measured the dominant per-class error predictor in A_plus's feature space:

        own-subspace energy ||P_c q||   rho = -0.585  (and STRENGTHENING with k)
        class-subspace overlap          rho = +0.037  (noise; every control beat it)

    Classes fail because their own held-out rows fall OUTSIDE their own training subspace --
    a within-class coverage failure, not a between-class collision. Independently, the
    accumulation decomposition says the gap is 71% data VOLUME.

    But A_plus trains LoRA with cross-entropy on a throwaway 20-way linear head and then
    deploys a subspace / prototype / cone reader. Nothing in that objective asks a handful of
    a class's samples to span its test distribution. This closes that mismatch by training
    the deployed criterion directly (Deep Subspace Networks, Simon et al. CVPR'20):

        support S_c, query Q_c  ->  B_c = QR(phi(S_c))  ->  logit(q, c) = s * ||B_c^T phi(q)||

    QR rather than SVD: we need the SPAN, and QR gradients are stable where SVD's blow up on
    near-degenerate spectra.

WHAT WOULD CONFIRM IT
    Not just A-Last. Re-run exp29's energy measurement on the resulting features: if this
    works, mean own-subspace energy should RISE and rho(energy, err) should stay dominant.
    A-Last improving without energy rising would mean it worked for some other reason.

CAVEAT
    r_sup must be < the smallest per-class count in a batch or that class is skipped
    (rank-deficient basis). With batch 128 over cpt classes the per-class count is small, so
    r_sup=2..4 is the usable range; the run logs how many classes were skipped.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 RSUP=2,4 SCALE=16 python -u exp31_subspace_train.py
"""
import json, os
import numpy as np
import fsa_train as F

DS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
RSUP = [int(x) for x in os.environ.get("RSUP", "2,4").split(",")]
SCALES = [float(x) for x in os.environ.get("SCALE", "16").split(",")]
OUT = os.path.join(F.REPO, f"exp31_subspace_train_{F.TAG}.json")

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DS:
    for T in TS:
        for seed in SEEDS:
            _, _, ytr, _, yte, n_cls = F.get_data(ds)
            # the CE baseline, same harness, so the comparison is paired
            key0 = f"{ds}|{T}|{seed}|ce"
            if key0 not in res:
                Ftr, Fte, a0 = F.train_task0(ds, T, seed, "ce")
                accs = F.replay(Ftr, ytr, Fte, yte, T, seed, n_cls)
                res[key0] = {"A_last": accs[-1], "A_avg": float(np.mean(accs)),
                             "acc0": a0, "accs": accs}
                json.dump(res, open(OUT, "w"), indent=2)
                F.log(f"  {key0}: A-Last {accs[-1]*100:.2f}")
            for r in RSUP:
                for s in SCALES:
                    key = f"{ds}|{T}|{seed}|r{r}_s{s:g}"
                    if key in res:
                        F.log(f"skip {key}"); continue
                    Ftr, Fte, a0 = F.train_task0(ds, T, seed, "subspace",
                                                 tag_extra=f"_r{r}_s{s:g}",
                                                 r_sup=r, head_scale=s)
                    accs = F.replay(Ftr, ytr, Fte, yte, T, seed, n_cls)
                    res[key] = {"A_last": accs[-1], "A_avg": float(np.mean(accs)),
                                "acc0": a0, "accs": accs}
                    json.dump(res, open(OUT, "w"), indent=2)
                    F.log(f"  {key}: A-Last {accs[-1]*100:.2f}  A-Avg "
                          f"{np.mean(accs)*100:.2f}  episodic-acc {a0:.3f}")

print("\n" + "=" * 80)
print(f"EXP31 — subspace-coverage training vs cross-entropy ({F.MODEL})")
print("=" * 80)
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
print("\ntask0 for the subspace arms is EPISODIC accuracy, not linear-head accuracy -- the")
print("two are not comparable, only A-Last / A-Avg are.")
print("Follow-up if this wins: re-run exp29 on the new features. Mean own-subspace energy")
print("should RISE; if A-Last improves without it, the mechanism is something else.")
print("=" * 80)
