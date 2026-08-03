#!/usr/bin/env python3
"""explore_headroom.py — locate the remaining +0.0600, which is ~94% FEATURES.

WHAT IS ALREADY SETTLED (do not re-test)
    statistics      +0.0035   split_full_accum.txt, and corroborated twice:
                              - newpen family (kfac/transport/between/within): all +-0.0008
                              - heads_frozen.txt: no head beats plain ridge (best cone -0.0402)
                              - freepoints_lam.txt: lambda has an INTERIOR optimum at 1e3,
                                0/10 stages clipped, free points exactly +0.0000
    capacity        +0.0012   ceiling_ranks.txt: r320 ~ r64 ~ r32
    virtual feats   FATAL     virt_gauss 0.4408 / virt_dirichlet 0.4448
    the cone        HARMFUL   cone_accum_aug40 0.6333 < A_frozen 0.6867

WHAT THESE TESTS ADD
    Two numbers that do not exist yet, and one decomposition that has never been made.

    (1) AN A-AVG CEILING. 0.8510 bounds A-last only. A-avg averages ten stages measured on
        20..200 classes, so the 200-class joint bound says nothing about it. Chasing +2-3
        A-avg without a bound is chasing an unknown target.  -> stage `curve`

    (2) DIVERSITY vs SEQUENTIALITY. The ~0.0565 feature gap confounds two causes needing
        opposite fixes:
          diversity     adaptation only ever sees 20 classes at once
          sequentiality adaptation happens in ten steps rather than one
        curve[t] is joint-trained on tasks 0..t; seq[k] is sequentially trained on the same
        data. Their difference at matched data isolates sequentiality; what remains against
        the full 200-class ceiling is diversity.  -> stages `curve` + `seq`

        seq[k] ~ curve[k]  -> sequential training is fine; the deficit is DIVERSITY, and the
                              lever is adapting over more tasks before freezing (legal, free)
        seq[k] << curve[k] -> drift during adaptation is real after all, which would be a
                              surprise given the penalty family produced nothing

    (3) The statistics/features split RE-MEASURED at the winning recipe on the current best
        method, rather than at 10ep on a superseded one.  -> stage `split`

NOTE ON AN EXISTING CONTROL
    crux_headroom.py:243 subsets classes by raw label (`TR_Y[keep] < n_classes`), NOT by the
    CIL class order. So div_only/vol_only were never class-matched to A_plus's task 0. These
    tests use exp8's ORDER = rng(SEED).permutation(200), so curve[t] and seq[k] are matched.

METHOD
    exp8_combined.py and crux_headroom.py are patched IN MEMORY (read, substituted, exec'd).
    Nothing is written to your source. Results go to headroom_*.npy, never to the real
    exp8_results_*.npy / crux_headroom_*.npy. Every substitution asserts it applied exactly
    once, so source drift fails loudly before any GPU work.

USAGE
    source ~/venvs/ml_env/bin/activate
    python explore_headroom.py curve     # per-stage joint ceiling + A-avg ceiling  (~80 min)
    python explore_headroom.py seq       # freeze_after ladder k=1,2,3,5            (~45 min)
    python explore_headroom.py recipe    # LR x EPOCHS on the bar                   (~35 min)
    python explore_headroom.py split     # stats/features split at winning recipe   (~60 min)
    python explore_headroom.py report    # combine whatever has been run (no GPU)
    python explore_headroom.py all

    T_LIST=0,1,3,5,7,9  K_LIST=1,2,3,5  MODEL=... SEED=... override.

Recommended order: curve (the map) -> seq (the cheapest legal lever) -> recipe -> split.
"""

import os
import sys
import json
import shutil
import contextlib

REPO = os.path.dirname(os.path.abspath(__file__))
EXP8 = os.path.join(REPO, "exp8_combined.py")
CRUX = os.path.join(REPO, "crux_headroom.py")

MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
SEED = int(os.environ.get("SEED", 0))
SLUG = MODEL.split(".")[-1]
LOGDIR = os.path.join(REPO, "logs", SLUG)
STORE = os.path.join(LOGDIR, f"headroom_map_s{SEED}.npy")

T_LIST = [int(x) for x in os.environ.get("T_LIST", "0,1,3,5,7,9").split(",")]
K_LIST = [int(x) for x in os.environ.get("K_LIST", "1,2,3,5").split(",")]

N_TASKS, CPT = 10, 20
CEILING_200 = 0.8510          # ceiling_ranks.txt, lora_aug_40ep_r64
FROZEN_FLOOR = 0.6902


def _sub(src, old, new, what, path):
    n = src.count(old)
    if n != 1:
        raise SystemExit(
            f"[explore_headroom] cannot patch '{what}' in {os.path.basename(path)}: "
            f"expected 1 occurrence, found {n}. Source drifted; update the pattern:\n"
            f"--- expected ---\n{old}\n----------------")
    return src.replace(old, new)


# ----------------------------------------------------------------- crux_headroom as a library
# One substitution neuters the top-level dispatch so the module can be exec'd for its helpers
# (build_data / adapt / extract / ncm / ranpac) without running any of its own experiments.
CRUX_DISPATCH = """if int(os.environ.get("RANKS", 0)):"""
CRUX_DISPATCH_NEW = """if int(os.environ.get("_HM_LIB", 0)):
    pass          # explore_headroom.py: library mode, dispatch + report suppressed
elif int(os.environ.get("RANKS", 0)):"""


def load_crux():
    """Exec crux_headroom.py with its dispatch disabled; return its namespace."""
    with open(CRUX) as f:
        src = f.read()
    src = _sub(src, CRUX_DISPATCH, CRUX_DISPATCH_NEW, "dispatch guard", CRUX)
    # The report block below the dispatch reads merged["frozen"], which library mode never
    # populates. Truncate the module at the results-file write.
    cut = src.index('OUT = f"crux_headroom_')
    src = src[:cut]
    g = {"__name__": "_crux_lib", "__file__": CRUX}
    os.environ["_HM_LIB"] = "1"
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        exec(compile(src, CRUX, "exec"), g)
    finally:
        os.chdir(cwd)
        os.environ.pop("_HM_LIB", None)
    return g


# ----------------------------------------------------------------- exp8 as a runnable
EXP8_VAR_ANCHOR = """}
WANT = [v for v in os.environ.get("VARIANTS", """
EXP8_VAR_NEW = """}
VAR.update(_HM_EXTRA_VAR)   # explore_headroom.py: freeze_after ladder
WANT = [v for v in os.environ.get("VARIANTS", """

EXP8_OUT_OLD = "OUT = f\"exp8_results_{MODEL.split('.')[-1]}.npy\""
EXP8_OUT_NEW = "OUT = _HM_OUT   # redirected: never clobber the real results file"


def patched_exp8(extra_var):
    with open(EXP8) as f:
        src = f.read()
    src = _sub(src, EXP8_VAR_ANCHOR, EXP8_VAR_NEW, "VAR injection", EXP8)
    src = _sub(src, EXP8_OUT_OLD, EXP8_OUT_NEW, "results output path", EXP8)
    return src


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)

    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


def run_exp8(name, variant, env, extra_var=None):
    """Execute the patched exp8 for one variant; return its per-stage accuracy list."""
    os.makedirs(LOGDIR, exist_ok=True)
    logpath = os.path.join(LOGDIR, f"headroom_{name}.txt")
    out = os.path.join(REPO, f"headroom_exp8_results_{SLUG}.npy")
    real = os.path.join(REPO, f"exp8_results_{SLUG}.npy")
    if os.path.exists(real) and not os.path.exists(out):
        shutil.copyfile(real, out)     # so BAR_KEY resolves to a measured bar

    prev = dict(os.environ)
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["VARIANTS"] = variant
    os.environ["MODEL"] = MODEL
    os.environ["SEED"] = str(SEED)
    # Pair against the SAME class order. exp8_combined.py:765 writes the unsuffixed key
    # regardless of SEED, so the plain key can hold another seed's bar.
    os.environ.setdefault("BAR_KEY", f"A_plus_aug40@l1=1,l2=1,l3=0.1,s{SEED}")

    g = {"__name__": "__main__", "__file__": EXP8,
         "_HM_EXTRA_VAR": dict(extra_var or {}), "_HM_OUT": out}
    tee = Tee(logpath)
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        with contextlib.redirect_stdout(tee):
            print(f"### headroom:{name} variant={variant} seed={SEED}")
            print(f"### env: {json.dumps(env, sort_keys=True)}\n")
            exec(compile(patched_exp8(extra_var), EXP8, "exec"), g)
    finally:
        os.chdir(cwd); tee.flush()
        os.environ.clear(); os.environ.update(prev)

    res = g.get("results", {})
    return (list(res.values())[0] if res else None), logpath


# ----------------------------------------------------------------- store
def load_store():
    import numpy as np
    if os.path.exists(STORE):
        try:
            return np.load(STORE, allow_pickle=True).item()
        except Exception as e:
            print(f"[warn] {e}")
    return {}


def save_store(d):
    import numpy as np
    os.makedirs(LOGDIR, exist_ok=True)
    np.save(STORE, d, allow_pickle=True)


# ============================================================ STAGE: curve
def stage_curve():
    """Per-stage joint ceiling: for each t, train jointly on tasks 0..t and evaluate on the
    classes seen by then. Gives the A-avg ceiling and shows WHERE the loss accumulates."""
    import numpy as np
    import torch
    from torch.utils.data import Subset

    C = load_crux()
    TR_Y, TE_Y = C["TR_Y"], C["TE_Y"]
    TRAIN, TEST, TRAIN_AUG = C["TRAIN"], C["TEST"], C["TRAIN_AUG"]

    # EXACTLY exp8's class order (exp8_combined.py:168) so curve[t] and seq[k] are matched.
    ORDER = np.random.default_rng(SEED).permutation(200)
    TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

    store = load_store()
    curve = store.get("curve", {})
    os.makedirs(LOGDIR, exist_ok=True)
    tee = Tee(os.path.join(LOGDIR, f"headroom_curve_s{SEED}.txt"))
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        with contextlib.redirect_stdout(tee):
            print(f"### per-stage joint ceiling  model={MODEL} seed={SEED}")
            print(f"### t in {T_LIST}   (recipe: aug=1 ep=40 lr=1e-4 warm=3 r=32)\n")
            for t in T_LIST:
                if t in curve:
                    print(f"  [skip] t={t} already measured: {curve[t]['ranpac']:.4f}")
                    continue
                seen = np.concatenate(TASKS[:t + 1])
                tr = np.where(np.isin(TR_Y, seen))[0]
                te = np.where(np.isin(TE_Y, seen))[0]
                print(f"\n=== curve t={t}  {len(seen)} classes  "
                      f"{len(tr)} train  {len(te)} test ===")
                m = C["adapt"]("lora", Subset(TRAIN_AUG, tr.tolist()),
                               epochs=40, lr=1e-4, warmup=3, lora_r=32)
                F1, F2 = C["extract"](m, TRAIN), C["extract"](m, TEST)
                del m; torch.cuda.empty_cache()
                # Remap to contiguous 0..len(seen)-1: unseen columns of the ridge solve are
                # then identically zero and cannot win the argmax.
                rmap = {int(c): i for i, c in enumerate(seen)}
                y1 = np.array([rmap[int(v)] for v in TR_Y[tr]])
                y2 = np.array([rmap[int(v)] for v in TE_Y[te]])
                n_ = C["ncm"](F1[tr], y1, F2[te], y2)
                r_, l_ = C["ranpac"](F1[tr], y1, F2[te], y2)
                curve[t] = dict(t=t, n_classes=int(len(seen)), n_train=int(len(tr)),
                                ncm=float(n_), ranpac=float(r_), lam=float(l_))
                print(f"  -> curve t={t}: NCM {n_:.4f} | RanPAC {r_:.4f} (lam={l_:g})")
                store["curve"] = curve; save_store(store)
    finally:
        os.chdir(cwd); tee.flush()

    store["curve"] = curve; save_store(store)
    return curve


# ============================================================ STAGE: seq
def stage_seq():
    """freeze_after ladder. k=0 is A_plus (the bar). k>0 adapts over the first k+1 tasks then
    freezes -- storage-free and CIL-legal, just a different choice of when to stop."""
    store = load_store()
    seq = store.get("seq", {})
    extra = {f"A_plus_k{k}": (0.0, 0.0, 0.0, "accum", k) for k in K_LIST}
    env = dict(AUG=1, EPOCHS=40, TAG="aug40")
    for k in K_LIST:
        if k in seq:
            print(f"  [skip] k={k} already measured: {seq[k]['A_last']:.4f}")
            continue
        print(f"\n>>> seq k={k}: adapt tasks 0..{k} ({(k+1)*CPT} classes) then freeze")
        accs, log = run_exp8(f"seq_k{k}_s{SEED}", f"A_plus_k{k}", env, extra_var=extra)
        if accs is None:
            print(f"  [warn] k={k} produced no result; see {log}")
            continue
        import numpy as np
        seq[k] = dict(k=k, accs=[float(a) for a in accs],
                      A_last=float(accs[-1]), A_avg=float(np.mean(accs)))
        print(f"  -> k={k}: A-last {seq[k]['A_last']:.4f}  A-avg {seq[k]['A_avg']:.4f}")
        store["seq"] = seq; save_store(store)
    return seq


# ============================================================ STAGE: recipe
def stage_recipe():
    """The only lever that has ever paid multi-point (aug40: +0.0433 A-last on the bar).
    The CIL runs inherited the JOINT recipe, tuned on 24000 images; task 0 has 2400."""
    store = load_store()
    rec = store.get("recipe", {})
    grid = [(lr, ep) for lr in (1e-4, 3e-4) for ep in (40, 80)]
    for lr, ep in grid:
        key = f"lr{lr:g}_ep{ep}"
        if key in rec:
            print(f"  [skip] {key} already measured: {rec[key]['A_last']:.4f}")
            continue
        print(f"\n>>> recipe {key} on the BAR")
        env = dict(AUG=1, EPOCHS=ep, LR=lr, TAG=f"r_{key}")
        accs, log = run_exp8(f"recipe_{key}_s{SEED}", "A_plus", env)
        if accs is None:
            print(f"  [warn] {key} produced no result; see {log}")
            continue
        import numpy as np
        rec[key] = dict(lr=lr, epochs=ep, accs=[float(a) for a in accs],
                        A_last=float(accs[-1]), A_avg=float(np.mean(accs)))
        print(f"  -> {key}: A-last {rec[key]['A_last']:.4f}  A-avg {rec[key]['A_avg']:.4f}")
        store["recipe"] = rec; save_store(store)
    return rec


# ============================================================ STAGE: split
def stage_split():
    """Re-measure the statistics/features split at the winning recipe on the CURRENT best
    method. The shipped +0.0035/+0.0963 was measured at 10ep on the superseded full_accum."""
    print("\n>>> split: oracle-stats on maha_distill_accum at aug40")
    env = dict(AUG=1, EPOCHS=40, TAG="aug40_orc", ORACLE_STATS=1, CEILING=CEILING_200)
    accs, log = run_exp8(f"split_s{SEED}", "maha_distill_accum", env)
    store = load_store()
    if accs is not None:
        import numpy as np
        store["split"] = dict(accs=[float(a) for a in accs], A_last=float(accs[-1]),
                              A_avg=float(np.mean(accs)), log=log)
        save_store(store)
    print(f"  -> see the HEADROOM SPLIT line in {log}")
    return store.get("split")


# ============================================================ report
def report():
    import numpy as np
    store = load_store()
    curve, seq, rec = store.get("curve", {}), store.get("seq", {}), store.get("recipe", {})

    print("\n" + "=" * 86)
    print(f"HEADROOM MAP — {MODEL}  seed {SEED}")
    print("=" * 86)

    bar = None
    real = os.path.join(REPO, f"exp8_results_{SLUG}.npy")
    if os.path.exists(real):
        d = np.load(real, allow_pickle=True).item()
        k = f"A_plus_aug40@l1=1,l2=1,l3=0.1,s{SEED}"
        if k in d:
            bar = [float(x) for x in d[k]]

    if curve:
        print("\nPER-STAGE JOINT CEILING  (joint training on tasks 0..t, eval on those classes)")
        print(f"{'t':>3} {'cls':>5} {'ceiling':>9} {'bar':>9} {'gap':>9}")
        gaps = []
        for t in sorted(curve):
            c = curve[t]["ranpac"]
            b = bar[t] if bar and t < len(bar) else float("nan")
            g = c - b
            gaps.append((t, g))
            print(f"{t:>3} {curve[t]['n_classes']:>5} {c:>9.4f} {b:>9.4f} {g:>+9.4f}")
        full = [curve[t]["ranpac"] for t in sorted(curve)]
        print(f"\nA-avg CEILING (mean over measured t): {np.mean(full):.4f}"
              + ("   [SUBSET of stages — interpolate before quoting]"
                 if len(curve) < N_TASKS else ""))
        if bar:
            print(f"A-avg bar                          : {np.mean(bar):.4f}")
            print(f"A-avg headroom                     : "
                  f"{np.mean(full) - np.mean(bar):+.4f}")
        if len(gaps) > 1:
            d0, d9 = gaps[0][1], gaps[-1][1]
            print(f"\ngap at t={gaps[0][0]}: {d0:+.4f}   gap at t={gaps[-1][0]}: {d9:+.4f}")
            print("  gap flat        -> loss is a constant per-stage tax; fix the recipe")
            print("  gap grows with t-> loss COMPOUNDS with tasks; fix adaptation/drift")

    if seq:
        print("\nfreeze_after LADDER  (storage-free; k = last task adapted on before freezing)")
        print(f"{'k':>3} {'cls adapted':>12} {'A-last':>9} {'A-avg':>9} {'vs k=0':>9}")
        base = None
        if bar:
            base = (bar[-1], float(np.mean(bar)))
            print(f"{0:>3} {CPT:>12} {base[0]:>9.4f} {base[1]:>9.4f} {'--':>9}")
        for k in sorted(seq):
            s = seq[k]
            d = f"{s['A_last'] - base[0]:+.4f}" if base else "--"
            print(f"{k:>3} {(k+1)*CPT:>12} {s['A_last']:>9.4f} {s['A_avg']:>9.4f} {d:>9}")
        if curve and seq:
            print("\nSEQUENTIALITY COST  (joint vs sequential on the SAME tasks 0..k)")
            print(f"{'k':>3} {'curve[k]':>10} {'seq[k]@k':>10} {'cost':>9}")
            for k in sorted(seq):
                if k in curve and k < len(seq[k]["accs"]):
                    c, s = curve[k]["ranpac"], seq[k]["accs"][k]
                    print(f"{k:>3} {c:>10.4f} {s:>10.4f} {c - s:>+9.4f}")
            print("  cost ~ 0 -> sequential training is fine; the deficit is DIVERSITY")
            print("  cost >> 0-> drift is real; revisit adaptation, not the head")

    if rec:
        print("\nRECIPE GRID on the bar")
        print(f"{'recipe':>14} {'A-last':>9} {'A-avg':>9}")
        for k in sorted(rec, key=lambda x: -rec[x]["A_last"]):
            print(f"{k:>14} {rec[k]['A_last']:>9.4f} {rec[k]['A_avg']:>9.4f}")

    if not (curve or seq or rec):
        print("\nnothing measured yet — run `curve` first (it is the map).")
    print("=" * 86)


STAGES = {"curve": (stage_curve, "~80 min"), "seq": (stage_seq, "~45 min"),
          "recipe": (stage_recipe, "~35 min"), "split": (stage_split, "~60 min"),
          "report": (report, "no GPU")}


def main():
    want = sys.argv[1:] or ["report"]
    if want == ["all"]:
        want = ["curve", "seq", "recipe", "split"]
    bad = [w for w in want if w not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {list(STAGES)} or 'all'")
    # Fail fast on source drift before any GPU work.
    if any(w in ("seq", "recipe", "split") for w in want):
        patched_exp8({})
    print(f"repo   : {REPO}\nmodel  : {MODEL}\nseed   : {SEED}")
    print(f"stages : {', '.join(f'{w} ({STAGES[w][1]})' for w in want)}\n")
    for w in want:
        print(f"\n{'=' * 30} {w} {'=' * 30}")
        STAGES[w][0]()
    if "report" not in want:
        report()


if __name__ == "__main__":
    main()
