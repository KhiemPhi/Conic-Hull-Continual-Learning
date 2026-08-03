#!/usr/bin/env python3
"""feature_forensics.py — HOW do sequential features fail? A longitudinal cohort study.

WHY
    oracle-stats already bounded the statistics side at +0.0035 and left +0.0963 in FEATURES.
    But nothing in this repo says *how* features degrade, so every candidate fix is a guess.
    Four mechanisms are consistent with the same accuracy drop and each needs a DIFFERENT fix:

      (1) dimensional collapse   representation contracts onto ~20 discriminative directions;
                                 later classes have no dimensions left to occupy
      (2) between-class erosion  old class MEANS drift together -- the directions that
                                 separated them stop being represented
      (3) within-class swelling  old class CLUSTERS tear apart; means may be fine
      (4) interference           new classes crowd into old neighbourhoods; global geometry
                                 is intact but specific old/new pairs collide

DESIGN
    Fix a COHORT = task 0's 20 classes. Measure that same cohort in all ten successive feature
    frames phi_0 .. phi_9. Same classes, same test images, only the backbone changes -- so any
    movement is caused by adaptation and nothing else. Prototypes are ALWAYS recomputed in the
    current frame, which removes statistics staleness by construction and isolates pure feature
    geometry (that is the part oracle-stats says is 96% of the problem).

ARMS
    seqce    freeze_after=None, all lambdas 0  -- the pathology, undefended
    aplus    freeze_after=0                    -- control: features frozen, every curve must be
                                                 FLAT. If it is not, the instrument is broken.
    best     maha_distill_accum                -- does the current best actually defend anything?

READ (decision tree)
    eff_rank falls .............. (1) collapse      -> anti-collapse / logdet (pf_anticollapse)
    tr(S_B) falls, S_W flat ..... (2) erosion       -> preserve the between-class subspace;
                                                       this is what LoDA's general branch does
    tr(S_W) rises ............... (3) swelling      -> distillation / stability term
    interference rises .......... (4) interference  -> isolation / routing (GR-LoRA, LoDA U_I)
    all flat but NCM falls ...... instrument or head bug; contradicts oracle-stats

    Fisher = tr(S_B)/tr(S_W) is the summary; the decomposition is what picks the fix.

METHOD
    exp8_combined.py is patched IN MEMORY (one substitution) to dump all-class test features at
    every stage. Nothing is written to your source; results go to forensics_*.npz.

USAGE
    source ~/venvs/ml_env/bin/activate
    python feature_forensics.py run aplus       # control first -- must be flat  (~6 min)
    python feature_forensics.py run seqce       # the pathology                  (~35 min)
    python feature_forensics.py run best        # current best                   (~50 min)
    python feature_forensics.py report          # no GPU
"""

import os
import sys
import json
import contextlib

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
EXP8 = os.path.join(REPO, "exp8_combined.py")
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
SEED = int(os.environ.get("SEED", 0))
SLUG = MODEL.split(".")[-1]
TOPK = 32          # subspace dimension for the rotation measurement

ARMS = {                       # name -> (variant, env)
    "aplus": ("A_plus", dict(AUG=1, EPOCHS=40, TAG="ff")),
    "seqce": ("null_baseline", dict(AUG=1, EPOCHS=40, TAG="ff")),
    "best": ("maha_distill_accum", dict(AUG=1, EPOCHS=40, TAG="ff")),
}

# One substitution: dump all-class test features right after the per-stage extraction.
ANCHOR = "        Zte = extract(model, TEST, te)"
INJECT = ANCHOR + "\n        _FF_HOOK(t, model, extract, TEST, TE_Y)"


def patched_source():
    with open(EXP8) as f:
        src = f.read()
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"[forensics] expected 1 occurrence of the extract anchor, found {n}. "
                         f"exp8_combined.py drifted; update ANCHOR.")
    return src.replace(ANCHOR, INJECT)


# ----------------------------------------------------------------- geometry
def _stats(F, y, classes):
    """Trace of between/within scatter and per-class means, for a fixed class set."""
    mus, nb, sw, n = [], [], 0.0, 0
    for c in classes:
        Z = F[y == c]
        if len(Z) == 0:
            continue
        m = Z.mean(0)
        mus.append(m)
        nb.append(len(Z))
        sw += float(((Z - m) ** 2).sum())
        n += len(Z)
    mus = np.stack(mus)
    nb = np.array(nb, dtype=np.float64)
    gm = (mus * nb[:, None]).sum(0) / nb.sum()
    sb = float((nb[:, None] * (mus - gm) ** 2).sum())
    return mus, sb / n, sw / n


def _eff_rank(F):
    X = F - F.mean(0)
    G = X.T @ X / max(len(X) - 1, 1)
    t1 = float(np.trace(G))
    t2 = float((G * G).sum())
    return t1 * t1 / (t2 + 1e-30)


def _sb_basis(mus, k):
    C = mus - mus.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[:k].T


def _principal_angle(U, V):
    """Mean principal angle (degrees) between two orthonormal subspaces."""
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s, -1, 1))).mean())


def _load(arm):
    """Combined npz if the run completed, else whatever stages survived on disk."""
    full = os.path.join(REPO, f"forensics_{arm}_s{SEED}.npz")
    if os.path.exists(full):
        return np.load(full, allow_pickle=True)
    meta = os.path.join(REPO, f"forensics_{arm}_s{SEED}_meta.npz")
    if not os.path.exists(meta):
        return None
    d = dict(np.load(meta, allow_pickle=True))
    t = 0
    while os.path.exists(os.path.join(REPO, f"forensics_{arm}_s{SEED}_F{t}.npy")):
        d[f"F{t}"] = np.load(os.path.join(REPO, f"forensics_{arm}_s{SEED}_F{t}.npy"))
        t += 1
    if t == 0:
        return None
    d["files"] = list(d.keys())
    print(f"    [{arm}] partial run recovered: {t} stage(s) from disk")
    return d


def analyse(d):
    y = d["y"]
    cohort = d["cohort"]            # task 0's classes -- the longitudinal cohort
    tasks = d["tasks"]
    rows = []
    ref_mu = ref_basis = None
    nstage = len([k for k in (d["files"] if isinstance(d, dict) else d.files)
                  if str(k).startswith("F")])
    for t in range(nstage):
        F = d[f"F{t}"]
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        mus, sb, sw = _stats(F, y, cohort)
        # Cohort-only: a FIXED class set, so this is comparable across stages. Measuring it
        # on the growing seen-class pool would rise trivially as classes are added (verified
        # on the frozen-feature control: 51.2 -> 85.8 with zero actual change).
        er = _eff_rank(F[np.isin(y, cohort)])
        er_pool = _eff_rank(F[np.isin(y, np.concatenate(tasks[: t + 1]))])
        basis = _sb_basis(mus, TOPK)
        if ref_mu is None:
            ref_mu, ref_basis = mus.copy(), basis.copy()
        dnn = np.median(np.sort(
            np.linalg.norm(mus[:, None] - mus[None], axis=2) + np.eye(len(mus)) * 1e9, axis=1)[:, 0])
        # cohort NCM, prototypes recomputed in the CURRENT frame (no staleness)
        M = F[np.isin(y, cohort)]
        yc = y[np.isin(y, cohort)]
        pred = np.array(cohort)[np.argmin(
            ((M[:, None, :] - mus[None]) ** 2).sum(2), axis=1)]
        ncm_cohort = float((pred == yc).mean())
        # interference: cohort prototypes whose nearest neighbour is a class from a LATER task
        seen = np.concatenate(tasks[: t + 1])
        later = np.setdiff1d(seen, cohort)
        interf = margin = np.nan
        if len(later):
            lm, _, _ = _stats(F, y, later)
            d_in = np.linalg.norm(mus[:, None] - mus[None], axis=2) + np.eye(len(mus)) * 1e9
            d_out = np.linalg.norm(mus[:, None] - lm[None], axis=2)
            interf = float((d_out.min(1) < d_in.min(1)).mean())
            # margin < 1 => a later-task class is closer than any cohort peer. Compare to the
            # frozen control at the SAME t; the binary rate alone saturates at 1.0 even when
            # nothing moved.
            margin = float(np.median(d_out.min(1) / (d_in.min(1) + 1e-30)))
        rows.append(dict(
            t=t, eff_rank=er, tr_SB=sb, tr_SW=sw, fisher=sb / (sw + 1e-30),
            ncm_cohort=ncm_cohort, d_nn=float(dnn),
            drift=float(np.linalg.norm(mus - ref_mu, axis=1).mean() / (dnn + 1e-30)),
            rot_deg=_principal_angle(ref_basis, basis), interference=interf,
            margin=margin, eff_rank_pool=er_pool))
    return rows


# ----------------------------------------------------------------- run
class Tee:
    def __init__(self, p):
        self.f = open(p, "w", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)

    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


def run_arm(arm):
    variant, env = ARMS[arm]
    logdir = os.path.join(REPO, "logs", SLUG)
    os.makedirs(logdir, exist_ok=True)
    store, meta = {}, {}

    def hook(t, model, extract, TEST, TE_Y):
        allidx = np.arange(len(TE_Y))
        F = extract(model, TEST, allidx).astype(np.float32)
        store[f"F{t}"] = F
        # Write through to disk at once: the first run of this was orphaned by a session
        # migration at t=7 and lost all eight completed stages from RAM.
        np.save(os.path.join(REPO, f"forensics_{arm}_s{SEED}_F{t}.npy"), F)
        if t == 0:
            ncls = int(np.asarray(TE_Y).max()) + 1
            order = np.random.default_rng(SEED).permutation(ncls)
            np.savez(os.path.join(REPO, f"forensics_{arm}_s{SEED}_meta.npz"),
                     y=np.asarray(TE_Y), tasks=order.reshape(10, ncls // 10),
                     cohort=order[: ncls // 10])

    prev = dict(os.environ)
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["VARIANTS"] = variant
    os.environ["MODEL"] = MODEL
    os.environ["SEED"] = str(SEED)
    os.environ.setdefault("BAR_KEY", f"A_plus_aug40@l1=1,l2=1,l3=0.1,s{SEED}")
    out_npy = os.path.join(REPO, f"forensics_exp8_results_{SLUG}.npy")

    g = {"__name__": "__main__", "__file__": EXP8, "_FF_HOOK": hook}
    tee = Tee(os.path.join(logdir, f"forensics_{arm}.txt"))
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        with contextlib.redirect_stdout(tee):
            print(f"### forensics:{arm}  variant={variant} seed={SEED}\n")
            src = patched_source().replace(
                'OUT = f"exp8_results_{MODEL.split(\'.\')[-1]}.npy"', f'OUT = {out_npy!r}')
            exec(compile(src, EXP8, "exec"), g)
    finally:
        os.chdir(cwd); tee.flush()
        os.environ.clear(); os.environ.update(prev)

    meta["y"] = g["TE_Y"]
    meta["tasks"] = np.array(g["TASKS"], dtype=object)
    meta["cohort"] = np.asarray(g["TASKS"][0])
    path = os.path.join(REPO, f"forensics_{arm}_s{SEED}.npz")
    np.savez_compressed(path, **store, **meta)
    print(f"  -> {path}  ({len(store)} stages)")
    return path


def report():
    print("\n" + "=" * 100)
    print(f"FEATURE FORENSICS — cohort = task 0's classes, tracked across all frames (seed {SEED})")
    print("=" * 100)
    found = {}
    for arm in ARMS:
        d = _load(arm)
        if d is not None:
            found[arm] = analyse(d)
    if not found:
        print("nothing yet — run `aplus` first (it is the instrument check).")
        return
    for arm, rows in found.items():
        print(f"\n--- {arm} ---")
        print(f"{'t':>3}{'effrank':>9}{'tr(S_B)':>10}{'tr(S_W)':>10}{'Fisher':>9}"
              f"{'NCM':>8}{'drift':>8}{'rot°':>7}{'margin':>8}")
        for r in rows:
            it = "  --  " if np.isnan(r["margin"]) else f"{r['margin']:.3f}"
            print(f"{r['t']:>3}{r['eff_rank']:>9.1f}{r['tr_SB']:>10.5f}{r['tr_SW']:>10.5f}"
                  f"{r['fisher']:>9.4f}{r['ncm_cohort']:>8.4f}{r['drift']:>8.3f}"
                  f"{r['rot_deg']:>7.1f}{it:>8}")
        a, b = rows[0], rows[-1]
        print(f"    delta t0->t9:  effrank {b['eff_rank']-a['eff_rank']:+.1f}   "
              f"tr(S_B) {100*(b['tr_SB']/a['tr_SB']-1):+.1f}%   "
              f"tr(S_W) {100*(b['tr_SW']/a['tr_SW']-1):+.1f}%   "
              f"Fisher {100*(b['fisher']/a['fisher']-1):+.1f}%   "
              f"NCM {b['ncm_cohort']-a['ncm_cohort']:+.4f}")
    if "aplus" in found:
        r = found["aplus"]
        moved = abs(r[-1]["fisher"] / r[0]["fisher"] - 1)
        print(f"\nINSTRUMENT CHECK (aplus features are frozen after t=0, so every curve must be "
              f"flat): Fisher moved {100*moved:.2f}%  -> {'OK' if moved < 0.01 else '*** BROKEN ***'}")
    print("\nREAD: effrank falls -> collapse (anti-collapse loss) | tr(S_B) falls -> between-class")
    print("      erosion (preserve the B-subspace; LoDA general branch) | tr(S_W) rises ->")
    print("      swelling (distillation) | interference rises -> isolation/routing")
    print("=" * 100)


if __name__ == "__main__":
    a = sys.argv[1:] or ["report"]
    if a[0] == "run":
        patched_source()
        for arm in (a[1:] or ["aplus"]):
            print(f">>> {arm}")
            run_arm(arm)
        report()
    else:
        report()
