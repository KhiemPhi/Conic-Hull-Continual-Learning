#!/usr/bin/env python3
"""find_free_points.py — measure the accuracy left on the table by head-side hyperparameters.

WHY THIS EXISTS
    The ORACLE_STATS diagnostic (logs/augreg_in21k/split_full_accum.txt) bounds every
    statistics-side improvement at +0.0035:

        method 0.7512 -> oracle-stats 0.7547 (statistics gap +0.0035) -> ceiling 0.8510

    The ridge regulariser lambda is NOT inside that bound. oracle-stats rebuilds G and C from
    real data but re-selects lambda from the SAME grid, so the grid itself was never tested.
    And the grid looks clipped:

        crux_headroom.py:48   LAMBDAS = [1e-1, 1, 1e1, 1e2, 1e3]  -> picks 1e2 every time
        exp8_combined.py:71   LAMBDAS = [1e2, 1e3, 1e4]           -> 1e2 IS the floor

    Every method and the bar select lambda at the lower edge of their grid. solve_eval() never
    logs which lambda won, so this has been invisible.

WHAT IT MEASURES
    Both grids in the SAME run, at zero extra training cost: G and C are already built at every
    stage, so a wider grid is just a few more 10000x10000 solves (~1-2 min per run total).
    Reporting old-grid and wide-grid side by side makes the comparison exactly paired -- same
    backbone, same features, same statistics, same stage. The only thing that differs is lambda.

FIDELITY CHECK
    The old-grid A-last/A-avg must reproduce the number already in exp8_results_*.npy. The
    script asserts this. If it does not match, the patch perturbed something and the delta is
    not trustworthy -- it says so loudly rather than reporting a bogus win.

WHAT IT DOES NOT TOUCH
    exp8_combined.py is patched IN MEMORY (source read, substituted, exec'd). Nothing is written
    to your source. Results go to freepoints_*.npy, never to exp8_results_*.npy.

USAGE
    source ~/venvs/ml_env/bin/activate
    python find_free_points.py lam           # lambda probe on the BAR      (~7 min)  [default]
    python find_free_points.py rank          # r32 vs r64 on the BAR        (~14 min)
    python find_free_points.py lam-method    # lambda probe on best method  (~55 min)
    python find_free_points.py all           # all three                    (~75 min)

    MODEL=... SEED=... override as in run_all.sh.

Each stage writes logs/<model>/freepoints_<stage>.txt and freepoints_<stage>.npy.
"""

import os
import re
import sys
import copy
import json
import shutil
import contextlib

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "exp8_combined.py")

MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
LOGDIR = os.path.join(REPO, "logs", MODEL.split(".")[-1])
REAL_RESULTS = os.path.join(REPO, f"exp8_results_{MODEL.split('.')[-1]}.npy")

# Superset of both existing grids, so the old selection is always reachable and the comparison
# is a strict widening rather than a different experiment.
WIDE = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
OLD = [1e2, 1e3, 1e4]


# ----------------------------------------------------------------- source patching
# Each substitution asserts it applied exactly once. If exp8_combined.py drifts, this fails
# loudly at startup instead of silently probing the wrong thing.

def _sub(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise SystemExit(
            f"[find_free_points] cannot patch '{what}': expected 1 occurrence in "
            f"exp8_combined.py, found {n}.\nThe source has drifted; update the pattern:\n"
            f"--- expected ---\n{old}\n----------------"
        )
    return src.replace(old, new)


PATCH_LAMBDAS_OLD = "LAMBDAS = [1e2, 1e3, 1e4]"
PATCH_LAMBDAS_NEW = (
    "LAMBDAS = _FP_WIDE          # patched by find_free_points.py\n"
    "LAMBDAS_OLD = _FP_OLD_GRID  # the shipped grid, kept for the paired counterfactual"
)

# Original selection block (exp8_combined.py:208-214).
PATCH_SOLVE_OLD = """    best, bestW = -1.0, None
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, C)
        a = acc(W, Zval, yval)
        if a > best:
            best, bestW = a, W
    return acc(bestW, Zte, yte)"""

# Solve once per lambda, score on BOTH val and test, then select twice -- once restricted to the
# shipped grid, once over the wide grid. Selection is always by VAL accuracy; test accuracy is
# only ever read out at the already-chosen lambda, so nothing here tunes on test.
PATCH_SOLVE_NEW = """    val, tst = {}, {}
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, C)
        val[lam] = acc(W, Zval, yval)
        tst[lam] = acc(W, Zte, yte)
        del W
    torch.cuda.empty_cache()

    def _pick(grid):
        g = [l for l in grid if l in val]
        b = max(g, key=lambda l: val[l])
        return b, tst[b]

    lam_w, a_w = _pick(LAMBDAS)
    lam_o, a_o = _pick(LAMBDAS_OLD)
    clipped = lam_o == min(LAMBDAS_OLD) and lam_w < min(LAMBDAS_OLD)
    _FP_RECORDS.append(dict(n_seen=int(len(seen)), lam_old=lam_o, lam_wide=lam_w,
                            acc_old=a_o, acc_wide=a_w, clipped=bool(clipped),
                            val={float(k): v for k, v in val.items()},
                            test={float(k): v for k, v in tst.items()}))
    log(f"  [lam] seen={len(seen):3d} | old-grid lam={lam_o:<7g} acc {a_o:.4f}"
        f" | wide-grid lam={lam_w:<7g} acc {a_w:.4f} | free {a_w - a_o:+.4f}"
        + ("   <-- OLD GRID CLIPPED AT ITS FLOOR" if clipped else ""))
    return a_w"""

PATCH_RANK_OLD = "lora_rank=32, lora_alpha=4.0, lora_config=\"task_shared\")"
PATCH_RANK_NEW = "lora_rank=_FP_RANK, lora_alpha=4.0, lora_config=\"task_shared\")"

PATCH_OUT_OLD = "OUT = f\"exp8_results_{MODEL.split('.')[-1]}.npy\""
PATCH_OUT_NEW = "OUT = _FP_OUT   # redirected: never clobber the real results file"


def build_patched_source():
    with open(SRC) as f:
        src = f.read()
    src = _sub(src, PATCH_LAMBDAS_OLD, PATCH_LAMBDAS_NEW, "LAMBDAS grid")
    src = _sub(src, PATCH_SOLVE_OLD, PATCH_SOLVE_NEW, "solve_eval selection")
    src = _sub(src, PATCH_RANK_OLD, PATCH_RANK_NEW, "lora_rank")
    src = _sub(src, PATCH_OUT_OLD, PATCH_OUT_NEW, "results output path")
    return src


# ----------------------------------------------------------------- runner

class Tee:
    """Mirror stdout to the log file, so the run is inspectable exactly like run_all.sh logs."""

    def __init__(self, path):
        self.f = open(path, "w", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def run_probe(name, variant, env, rank=32):
    """Execute the patched exp8 in a fresh namespace and return the per-stage records."""
    os.makedirs(LOGDIR, exist_ok=True)
    logpath = os.path.join(LOGDIR, f"freepoints_{name}.txt")

    # Seed the probe's results file from the real one so BAR_KEY resolves to a measured bar
    # rather than the stale augreg2 constants. Read-only w.r.t. the real file.
    out = os.path.join(REPO, f"freepoints_exp8_results_{MODEL.split('.')[-1]}.npy")
    if os.path.exists(REAL_RESULTS) and not os.path.exists(out):
        shutil.copyfile(REAL_RESULTS, out)

    prev = dict(os.environ)
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["VARIANTS"] = variant
    os.environ["MODEL"] = MODEL
    os.environ.pop("ORACLE_STATS", None)  # its extra solve_eval call would pollute the records

    records = []
    g = {
        "__name__": "__main__",
        "__file__": SRC,
        "_FP_WIDE": list(WIDE),
        "_FP_OLD_GRID": list(OLD),
        "_FP_RECORDS": records,
        "_FP_RANK": int(rank),
        "_FP_OUT": out,
    }

    tee = Tee(logpath)
    cwd = os.getcwd()
    try:
        os.chdir(REPO)          # exp8 resolves data/ relative to cwd
        with contextlib.redirect_stdout(tee):
            print(f"### freepoints:{name}  variant={variant} rank={rank}")
            print(f"### model={MODEL}")
            print(f"### env: {json.dumps(env, sort_keys=True)}")
            print(f"### grids: old={OLD}  wide={WIDE}\n")
            exec(compile(build_patched_source(), SRC, "exec"), g)
    finally:
        os.chdir(cwd)
        tee.flush()
        os.environ.clear()
        os.environ.update(prev)

    return records, logpath


# ----------------------------------------------------------------- reporting

def summarise(name, variant, tag, records):
    """Paired old-grid vs wide-grid summary, plus the fidelity check against the shipped run."""
    if not records:
        print(f"  [{name}] no records -- the run produced no stages")
        return None

    a_old = [r["acc_old"] for r in records]
    a_new = [r["acc_wide"] for r in records]
    last_o, last_n = a_old[-1], a_new[-1]
    avg_o, avg_n = sum(a_old) / len(a_old), sum(a_new) / len(a_new)
    n_clip = sum(r["clipped"] for r in records)

    print("\n" + "=" * 88)
    print(f"FREE POINTS — {name}  ({variant}{tag})")
    print("=" * 88)
    print(f"{'stage':>6} {'seen':>5} {'lam_old':>9} {'acc_old':>9} "
          f"{'lam_wide':>9} {'acc_wide':>9} {'free':>8}")
    for i, r in enumerate(records):
        print(f"{i:>6} {r['n_seen']:>5} {r['lam_old']:>9g} {r['acc_old']:>9.4f} "
              f"{r['lam_wide']:>9g} {r['acc_wide']:>9.4f} "
              f"{r['acc_wide'] - r['acc_old']:>+8.4f}"
              + ("  clipped" if r["clipped"] else ""))
    print("-" * 88)
    print(f"{'A-last':>14}   old {last_o:.4f}   wide {last_n:.4f}   "
          f"free {last_n - last_o:+.4f}  ({100 * (last_n - last_o):+.2f} points)")
    print(f"{'A-avg':>14}   old {avg_o:.4f}   wide {avg_n:.4f}   "
          f"free {avg_n - avg_o:+.4f}  ({100 * (avg_n - avg_o):+.2f} points)")
    print(f"stages where the shipped grid sat on its floor with a better lambda "
          f"below it: {n_clip}/{len(records)}")

    # ---- fidelity: the old-grid replay must reproduce the shipped number ----
    verdict = "unchecked (no prior result for this key)"
    try:
        import numpy as np
        if os.path.exists(REAL_RESULTS):
            ref = np.load(REAL_RESULTS, allow_pickle=True).item()
            key = variant + tag
            if key in ref:
                r_last = float(ref[key][-1])
                r_avg = float(np.mean(ref[key]))
                d_last, d_avg = last_o - r_last, avg_o - r_avg
                ok = abs(d_last) < 5e-4 and abs(d_avg) < 5e-4
                verdict = ("PASS" if ok else "*** FAIL ***")
                print(f"\nfidelity vs shipped {key}: "
                      f"A-last {r_last:.4f} (d {d_last:+.4f})  "
                      f"A-avg {r_avg:.4f} (d {d_avg:+.4f})  -> {verdict}")
                if not ok:
                    print("  The in-memory patch perturbed the run. The 'free' column above is\n"
                          "  NOT trustworthy -- reconcile before acting on it.")
    except Exception as e:  # reporting must never mask the measurement
        print(f"\n[warn] fidelity check skipped: {e}")

    print("=" * 88)
    return dict(name=name, variant=variant + tag, records=records,
                A_last_old=last_o, A_last_wide=last_n,
                A_avg_old=avg_o, A_avg_wide=avg_n,
                n_clipped=n_clip, fidelity=verdict)


# ----------------------------------------------------------------- stages

SEED = os.environ.get("SEED", "0")
# The winning recipe. Both arms must be measured here -- a tuned method against an untuned bar
# is not a comparison (run_all.sh:158-160).
AUG40 = dict(AUG=1, EPOCHS=40, TAG="aug40", SEED=SEED)

STAGES = {
    # name            variant                  env      rank  est
    "lam": ("A_plus", AUG40, 32, "~7 min"),
    "rank": ("A_plus", AUG40, 64, "~14 min"),
    "lam-method": ("maha_distill_accum", AUG40, 32, "~55 min"),
}


def main():
    want = sys.argv[1:] or ["lam"]
    if want == ["all"]:
        want = ["lam", "rank", "lam-method"]
    bad = [w for w in want if w not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {list(STAGES)} or 'all'")

    import numpy as np
    build_patched_source()  # fail fast on source drift, before any GPU work
    print(f"repo    : {REPO}")
    print(f"model   : {MODEL}")
    print(f"stages  : {', '.join(f'{w} ({STAGES[w][3]})' for w in want)}\n")

    summaries = []
    for w in want:
        variant, env, rank, est = STAGES[w]
        env = dict(env)
        name = w if rank == 32 else f"{w}_r{rank}"
        print(f"\n>>> {name}: variant={variant} rank={rank} env={env}  [{est}]")
        records, logpath = run_probe(name, variant, env, rank=rank)
        tag = "_" + env["TAG"] if env.get("TAG") else ""
        s = summarise(name, variant, tag, records)
        if s:
            s["log"] = logpath
            s["rank"] = rank
            np.save(os.path.join(LOGDIR, f"freepoints_{name}.npy"), s, allow_pickle=True)
            summaries.append(s)
        print(f"  -> {logpath}")

    if len(summaries) > 1:
        print("\n" + "=" * 88)
        print("ALL STAGES")
        print("=" * 88)
        print(f"{'stage':<16} {'r':>3} {'A-last old':>11} {'A-last wide':>12} "
              f"{'A-avg old':>10} {'A-avg wide':>11} {'clip':>6}")
        for s in summaries:
            print(f"{s['name']:<16} {s['rank']:>3} {s['A_last_old']:>11.4f} "
                  f"{s['A_last_wide']:>12.4f} {s['A_avg_old']:>10.4f} "
                  f"{s['A_avg_wide']:>11.4f} {s['n_clipped']:>4}/10")
        print("=" * 88)

    print("\nREAD:")
    print("  free >= +0.005 on either metric -> the grid was clipped; widen LAMBDAS in")
    print("      exp8_combined.py:71 and crux_headroom.py:48 and re-run BOTH arms.")
    print("  free ~ 0 and clip count 0       -> lambda is genuinely settled; the head is done,")
    print("      consistent with the +0.0035 statistics bound. Spend the time on features.")
    print("  fidelity FAIL                   -> ignore the deltas, reconcile the patch first.")


if __name__ == "__main__":
    main()
