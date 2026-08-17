#!/usr/bin/env python3
"""exp60_seq_drift.py -- CIL accuracy (A-Last / A-Avg) for sequential per-task LoRA read out
   by cones alone, plus the drift diagnostic that explains the result.

WHAT THE FIRST VERSION OF THIS FILE ESTABLISHED, AND WHY THE ARMS CHANGED
    Q1  Drift is real and compounds: cos(phi_0, phi_t) falls 1.00 -> 0.87 -> ... -> ~0.70,
        about 0.90 per step. So the feature space genuinely moves.
    Q2  TRANSPORT IS DEAD. Two separate causes, and only one is fixable:
          - composing ridge-fitted maps multiplies condition numbers (222 -> 3.8e3 -> 1.2e5),
            which is what produced the 28.55 collapse at t=3 and the not-positive-definite
            crash at t=5. Fixable by fitting phi_0 -> phi_t DIRECTLY (cond 659 at t=3).
          - but at t=1 there is NO composition, and transport ALREADY loses to doing nothing
            (93.19 vs 94.06). The map has to be fit on task-t data and applied to task-0 class
            geometry; that extrapolation carries relerr 0.46 even at one step. Direct fitting
            removes the catastrophe, not the deficit.
        The invariance theorem from exp58 is not wrong -- it is exact. You simply cannot
        estimate the map on data that makes it accurate where it is used.
    SURVIVES: frozen cones are robust. `stale` held 95.2 -> 91.2 while the space rotated to
        cos 0.77, and `native` stayed flat, so the FEATURES keep their discriminability and
        only the frozen geometry decays, slowly.

    So the transport arm is demoted to optional (TRANSPORT=1, direct-fit, guarded whitener)
    and this file now answers the question that actually decides the direction: what CIL
    accuracy does sequential per-task LoRA reach with a cone-only read-out, and is it better
    than not adapting at all?

THE FOUR ARMS -- all cone-only, no RanPAC, no Gram anywhere
    seq_stale     sequential LoRA. Each class's rays are built at the stage it ARRIVES and
                  never touched again. One global whitener, frozen at stage 0. Replay-free
                  and deployable. THIS IS THE CANDIDATE METHOD.
    seq_refit     sequential LoRA, every cone and the whitener rebuilt at every stage from
                  all data seen so far. Needs replay -> ORACLE, not a method. Upper bound on
                  what the sequential features can support.
    frozen        NO adaptation after stage 0: phi_0 for everything, cones built as classes
                  arrive. Replay-free. This is the first-session regime the rest of the
                  project uses, stripped to a single member and cone-only.
    frozen_refit  phi_0, everything rebuilt each stage. Oracle for the frozen regime.

    seq_stale  vs frozen        -> does sequential adaptation help a DEPLOYABLE read-out?
    seq_refit  vs frozen_refit  -> does it improve the FEATURES at all, ignoring forgetting?
    The second is the old Q3 and it can close the direction on its own: if the sequential
    features are not better under an oracle, nothing downstream can rescue them.

REFERENCE POINTS (IMAGENETR, cone-only, so directly comparable)
    exp54 measured cone-alone at 79.92 A-Last / 84.87 A-Avg against RanPAC-alone 80.41/85.31
    on the legacy order. Those were first-session single-member numbers, i.e. the same regime
    as `frozen` here modulo the class order.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot DS=IMAGENETR T=10 SEED=0 RANK=32 \
      python -u exp60_seq_drift.py
    # add TRANSPORT=1 to also run the (dead) direct-fit transport arm
"""
import json
import os
import warnings

import numpy as np

os.environ.setdefault("T", "10")
os.environ.setdefault("SEED", "0")
os.environ.setdefault("ARMS", "f64")
os.environ.setdefault("RULES", "cone")
warnings.filterwarnings("ignore", message=r"Number of distinct clusters.*",
                        category=UserWarning, module=r"sklearn\..*")

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402
import exp54_stack as S54                   # noqa: E402
import class_order as CO                    # noqa: E402

un = X.un
REPO = os.path.dirname(os.path.abspath(__file__))
TAG = S54.TAG
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
RANK = int(os.environ.get("RANK", 32))
R = int(os.environ.get("R", 64))
SHRINK, GAMMA = S54.SHRINK, S54.GAMMA
RIDGE = float(os.environ.get("RIDGE", 1e-3))
MAX_TEST = int(os.environ.get("MAX_TEST", 1500))
DO_TRANSPORT = int(os.environ.get("TRANSPORT", 0))


def stage_path(t):
    return os.path.join(
        REPO, f"exp59_feats_{DS}_T{T}_s{SEED}_stage{t}_r{RANK}{CO.order_tag()}_{TAG}.npz")


def n_cached():
    n = 0
    while n < T and os.path.exists(stage_path(n)):
        n += 1
    return n


def stage_feats(t):
    z = np.load(stage_path(t))
    return un(z["Ftr"]).astype(np.float64), un(z["Fte"]).astype(np.float64)


def scatter(Z, ytr, rows, classes):
    d = Z.shape[1]
    sc, ns = np.zeros((d, d)), 0
    for c in classes:
        r = rows[ytr[rows] == c]
        if len(r) < 2:
            continue
        Xc = Z[r] - Z[r].mean(0)
        sc += Xc.T @ Xc; ns += len(Xc)
    S = sc / max(ns, 1)
    return S + SHRINK * np.trace(S) / d * np.eye(d)


def whiten(S):
    """Cholesky of the inverse, made robust. A transported scatter M^T S M loses positive
    definiteness once cond(M) reaches ~1e5 -- that is what crashed the previous version at
    t=5. Symmetrise, then clip eigenvalues to a floor rather than failing."""
    S = 0.5 * (S + S.T)
    try:
        return np.linalg.cholesky(np.linalg.inv(S))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(S)
        w = np.clip(w, max(w.max(), 1e-12) * 1e-8, None)
        return np.linalg.cholesky(V @ np.diag(1.0 / w) @ V.T)


def build_cones(Z, ytr, rows, classes, Wh):
    Wh32 = Wh.astype(np.float32); Wi = np.linalg.inv(Wh32)
    out = {}
    for c in classes:
        r = rows[ytr[rows] == c]
        if len(r) < 2:
            continue
        Xw = un(Z[r].astype(np.float32) @ Wh32)
        oth = rows[~np.isin(ytr[rows], [c])]
        Fr = Z[oth].astype(np.float32)
        if len(Fr) > S54.F_MAX:
            Fr = Fr[np.random.default_rng(int(c)).choice(len(Fr), S54.F_MAX, replace=False)]
        out[int(c)] = X.BUILD[S54.METHOD](Xw, un(Fr), R, int(c), GAMMA) @ Wi
    return out


def score_acc(cones, Wh, Zte, rows, yte):
    """Rays and queries get the SAME whitener, which is what keeps the scores commensurable
    even when the rays were built in an older feature space."""
    Wh32 = Wh.astype(np.float32)
    Q = un(Zte[rows].astype(np.float32) @ Wh32)
    cls = sorted(cones)
    S = np.full((len(rows), len(cls)), -np.inf, np.float32)
    for j, c in enumerate(cls):
        S[:, j] = S54.score("cone", un(cones[c].astype(np.float32) @ Wh32), Q)
    return float((np.asarray(cls)[S.argmax(1)] == yte[rows]).mean())


def fit_map(Za, Zb, rows):
    A, B = Za[rows], Zb[rows]
    G = A.T @ A + RIDGE * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ B)


if __name__ == "__main__":
    E.T, E.SEED = T, SEED
    ytr, yte, n_cls = E.get_labels(DS)
    cpt = n_cls // T
    order = CO.class_order(n_cls, SEED)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT = []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        FIT.append(ix[pm[max(int(0.1 * len(ix)), 1):]])

    NS = n_cached()
    assert NS >= 2, f"need >=2 cached stages, found {NS}"
    Z = [stage_feats(t) for t in range(NS)]
    W = 100
    print("=" * W)
    print(f"EXP60 -- sequential per-task LoRA, cone-only read-out   {DS} T={T} seed={SEED} "
          f"rank={RANK} R={R}")
    print(f"  ORDER={CO.mode()}   stages cached {NS}/{T}"
          + ("   PARTIAL -- A-Last is the last CACHED stage, not task T" if NS < T else ""))
    print("=" * W)

    # ---------------- drift diagnostic
    print("\nDRIFT   per-image cosine on the test set")
    print(f"  {'t':>3}{'cos(phi_0,phi_t)':>19}{'cos(phi_t-1,phi_t)':>21}{'fit relerr':>13}")
    Ms = []
    for t in range(NS):
        c0 = float((Z[0][1] * Z[t][1]).sum(1).mean())
        if t == 0:
            print(f"  {t:>3}{c0:>19.4f}{'--':>21}{'--':>13}"); continue
        cp = float((Z[t - 1][1] * Z[t][1]).sum(1).mean())
        Mt = fit_map(Z[t - 1][0], Z[t][0], FIT[t]); Ms.append(Mt)
        rel = float(np.linalg.norm(Z[t - 1][0][FIT[t]] @ Mt - Z[t][0][FIT[t]]) /
                    np.linalg.norm(Z[t][0][FIT[t]]))
        print(f"  {t:>3}{c0:>19.4f}{cp:>21.4f}{rel:>13.3f}")

    # ---------------- staged CIL, identical test rows across arms at each stage
    rng = np.random.default_rng(7)
    TEI = []
    for t in range(NS):
        seen = np.concatenate(tasks[:t + 1])
        tei = np.where(np.isin(yte, seen))[0]
        if len(tei) > MAX_TEST:
            tei = tei[rng.permutation(len(tei))[:MAX_TEST]]
        TEI.append(tei)

    arms = ["seq_stale", "seq_refit", "frozen", "frozen_refit"] + \
           (["seq_transport"] if DO_TRANSPORT else [])
    accs = {a: [] for a in arms}
    cones_seq, cones_frz = {}, {}          # birth-built rays, never rebuilt
    W0_seq = W0_frz = None

    print(f"\nSTAGED CIL   test rows capped at {MAX_TEST}/stage, identical across arms")
    print(f"  {'t':>3}{'#cls':>6}" + "".join(f"{a:>15}" for a in arms))
    for t in range(NS):
        seen = np.concatenate(tasks[:t + 1])
        allrows = np.concatenate(FIT[:t + 1])
        if t == 0:
            W0_seq = whiten(scatter(Z[0][0], ytr, FIT[0], tasks[0]))
            W0_frz = W0_seq
        # rays for the classes that ARRIVE now, built in the space that exists now
        cones_seq.update(build_cones(Z[t][0], ytr, FIT[t], tasks[t], W0_seq))
        cones_frz.update(build_cones(Z[0][0], ytr, FIT[t], tasks[t], W0_frz))

        row = {}
        row["seq_stale"] = score_acc(cones_seq, W0_seq, Z[t][1], TEI[t], yte)
        row["frozen"] = score_acc(cones_frz, W0_frz, Z[0][1], TEI[t], yte)
        Wr = whiten(scatter(Z[t][0], ytr, allrows, seen))
        row["seq_refit"] = score_acc(build_cones(Z[t][0], ytr, allrows, seen, Wr), Wr,
                                     Z[t][1], TEI[t], yte)
        Wf = whiten(scatter(Z[0][0], ytr, allrows, seen))
        row["frozen_refit"] = score_acc(build_cones(Z[0][0], ytr, allrows, seen, Wf), Wf,
                                        Z[0][1], TEI[t], yte)
        if DO_TRANSPORT:
            # DIRECT phi_0 -> phi_t fit, not composed: composition is what blew up.
            if t == 0:
                row["seq_transport"] = row["seq_stale"]
            else:
                Md = fit_map(Z[0][0], Z[t][0], FIT[t])
                Wt = whiten(Md.T @ scatter(Z[0][0], ytr, FIT[0], tasks[0]) @ Md)
                row["seq_transport"] = score_acc({c: v @ Md for c, v in cones_frz.items()},
                                                 Wt, Z[t][1], TEI[t], yte)
        for a in arms:
            accs[a].append(row[a])
        print(f"  {t:>3}{len(seen):>6}" + "".join(f"{row[a]*100:>15.2f}" for a in arms))

    print(f"\n{'-'*W}")
    print(f"  {'arm':<16}{'A-Last':>9}{'A-Avg':>9}{'vs frozen':>12}{'replay?':>10}")
    fl, fa = accs["frozen"][-1] * 100, np.mean(accs["frozen"]) * 100
    need = {"seq_stale": "no", "frozen": "no", "seq_refit": "YES (oracle)",
            "frozen_refit": "YES (oracle)", "seq_transport": "no"}
    for a in arms:
        l, av = accs[a][-1] * 100, np.mean(accs[a]) * 100
        print(f"  {a:<16}{l:>9.2f}{av:>9.2f}{l-fl:>+7.2f}/{av-fa:>+5.2f}{need[a]:>10}")
    print(f"\n  DEPLOYABLE COMPARISON   seq_stale - frozen = "
          f"{(accs['seq_stale'][-1]-accs['frozen'][-1])*100:+.2f} A-Last / "
          f"{(np.mean(accs['seq_stale'])-np.mean(accs['frozen']))*100:+.2f} A-Avg")
    print(f"  FEATURE-QUALITY ORACLE  seq_refit - frozen_refit = "
          f"{(accs['seq_refit'][-1]-accs['frozen_refit'][-1])*100:+.2f} A-Last / "
          f"{(np.mean(accs['seq_refit'])-np.mean(accs['frozen_refit']))*100:+.2f} A-Avg")

    json.dump({"accs": {a: accs[a] for a in arms}, "n_stages": NS},
              open(os.path.join(REPO,
                   f"exp60_seq_cil_{DS}_T{T}_s{SEED}_r{RANK}{CO.order_tag()}_{TAG}.json"),
                   "w"), indent=2)
    print(f"\n{'-'*W}")
    print("""HOW TO READ THIS
  1. seq_stale - frozen is the only comparison between two DEPLOYABLE systems. If it is <= 0,
     sequential per-task LoRA does not pay for itself under a cone read-out and the
     first-session design wins on accuracy as well as on cost.
  2. seq_refit - frozen_refit removes forgetting from the question entirely: both arms get
     full replay, so this is purely "are the sequential features better". If THIS is <= 0 the
     direction closes -- no read-out, protection scheme or transport can recover features
     that are not there.
  3. The gap seq_refit - seq_stale is what a perfect drift-compensation mechanism could win
     back. exp58/exp60 showed estimated linear transport cannot claim it: the map must be fit
     on the current task and applied to old-class geometry, and that extrapolation loses even
     at one step, before any composition.
  4. Cone-only reference on IMAGENETR from exp54: cone-alone 79.92/84.87, RanPAC-alone
     80.41/85.31, fused 81.52/85.94 -- first-session, single member, legacy class order.""")
    print("=" * W)
