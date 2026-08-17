#!/usr/bin/env python3
"""exp68_vote_structure.py -- WHY does the member ensemble capture only ~14% of its own oracle?

THE FACT THIS FILE EXPLAINS
    exp55's diversity block on IMAGENETR T=10 s0 (structural members):
        ORACLE (some member right)   0.869
        best single member           0.807
        uniform ensemble             0.816
    So 6.2 points sit between the best single member and the oracle, and averaging collects
    0.9 of them. Worse, the structural design demonstrably WORKS at creating diversity --
    pairwise disagreement 0.141 vs 0.111 for a seed-only ensemble, error correlation 0.779 vs
    0.831, oracle ceiling +0.88 -- and yet its ensemble accuracy is IDENTICAL (81.6 vs 81.7,
    exp66/C1). Diversity is created and then thrown away. This file measures where it goes.

THE MECHANISM UNDER TEST
    Uniform averaging of z-scored member scores is, near the decision boundary, a VOTE. A vote
    can only recover a sample that a majority of members already get right. If the headroom
    lives in MINORITY-correct samples (k = 1 or 2 of 5 members right) then averaging is
    structurally incapable of reaching it, no reweighting fixes it, and the only route is
    per-sample routing -- which exp55 already closed at the per-class level
    (oracle_class_cv 0.803 < best_single 0.807, i.e. a NEGATIVE share of the headroom).

    So: bin every test sample by k = #members that classify it correctly, and report
      (a) the mass at each k,
      (b) what the uniform average actually gets at each k,
      (c) how much of the oracle headroom sits at k < majority -- the INACCESSIBLE part.

    PRE-REGISTERED, written before the run: if >=60% of the headroom mass sits at k<=2, the
    combiner is not fixable and the ensemble is a dead end for this member pool. If a majority
    of the headroom sits at k>=3, uniform averaging is leaving ACCESSIBLE points on the table
    and a confidence- or rank-based combiner is worth exactly one experiment.

WHY THE READ-OUT IS REBUILT HERE RATHER THAN IMPORTED
    exp55/exp56 report only aggregate accuracy; nothing on disk carries PER-SAMPLE member
    correctness, which is the whole quantity of interest. The staged RanPAC below is exp56's,
    with its helpers imported (never copied) and its per-member accuracy asserted against
    exp56's stored `{m}|ranpac` and `ens_ranpac` values for the same cell. If those do not
    match, this file is describing a different ensemble than the paper's and the vote
    structure it reports is meaningless.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot \\
      DS=IMAGENETR T=10 SEED=0,1,2 python -u exp68_vote_structure.py

    # the seed-only pool, for the same cell -- does redundancy explain the C1 tie?
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot MEMBERS=q32,q32v1,q32v2,q32v3,q32v4 \\
      DS=IMAGENETR T=10 SEED=0,1,2 SUFFIX=_seedonly python -u exp68_vote_structure.py
"""
import json
import os
import time

import numpy as np
import torch

_DS = os.environ.get("DS", "IMAGENETR").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0,1,2").split(",")]
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])
os.environ.setdefault("ARMS", "f64")
os.environ.setdefault("RULES", "cone")
for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import class_order as CO  # noqa: E402
import exp19_dataset_hull as E  # noqa: E402
import exp39_cone_construction as X  # noqa: E402
import exp54_stack as S54  # noqa: E402
import exp56_ray_ensemble as S56  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = S54.TAG
MEMBERS = os.environ.get("MEMBERS", "q32,m32,a16,q32b70,q64").split(",")
M_RP, SHRINK = S54.M_RP, S54.SHRINK
LAMBDAS = S54.LAMBDAS
OUT = os.path.join(
    REPO, f"exp68_vote_structure{os.environ.get('SUFFIX', '')}_{TAG}.json"
)
EXP56 = os.path.join(REPO, f"exp56_ray_ensemble_table_{TAG}.json")
un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
member_features = S56.member_features
M = len(MEMBERS)
MAJ = M // 2 + 1  # 3 of 5


def run_cell(ds, T, seed):
    E.T, E.SEED = T, seed
    ytr, yte, n_cls = E.get_labels(ds)
    Z = {m: member_features(ds, T, seed, m) for m in MEMBERS}
    d = Z[MEMBERS[0]][0].shape[1]
    cpt = n_cls // T
    order = CO.class_order(n_cls, seed)
    tasks = [order[i * cpt : (i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm_ = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm_[:nv]])
        FIT.append(ix[pm_[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    G = {m: torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64) for m in MEMBERS}
    C = {m: torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64) for m in MEMBERS}

    def _H(Zm, bs=4096):
        for i in range(0, len(Zm), bs):
            yield i, torch.relu(
                torch.as_tensor(Zm[i : i + bs], device=DEV, dtype=torch.float32) @ P
            )

    def logits(Zm, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Zm)]).cpu().numpy()

    # Only the FINAL stage matters for vote structure, but RanPAC is accumulated, so every
    # stage must still be walked to build G/C exactly as exp56 does.
    for t in range(T):
        seen = np.concatenate(tasks[: t + 1])
        nval = sum(len(v) for v in VAL[: t + 1])
        vix = VAL_ALL[:nval]
        yv = ytr[vix]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]
        zLt, per_acc = {}, {}
        for m in MEMBERS:
            Ztr_m, Zte_m = Z[m]
            for i, h in _H(un(Ztr_m[FIT[t]])):
                h = h.double()
                Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
                Y[
                    torch.arange(h.shape[0]),
                    torch.tensor(ytr[FIT[t]][i : i + h.shape[0]], device=DEV),
                ] = 1.0
                G[m] += h.T @ h
                C[m] += h.T @ Y
            best, bw = -1.0, None
            for lam in LAMBDAS:
                Wm = torch.linalg.solve(G[m] + lam * eye, C[m])
                a_ = acc_v1(logits(un(Ztr_m[vix]), Wm), seen, yv)
                if a_ > best:
                    best, bw = a_, Wm
            zLt[m] = zs(logits(un(Zte_m), bw)[tei], seen)
            per_acc[m] = acc_v1(zLt[m], seen, yt)
        ens = sum(zLt.values()) / M
        ens_acc = acc_v1(ens, seen, yt)
        if t < T - 1:
            continue

        # ---- final stage: per-sample correctness per member
        pred = {m: np.asarray(seen)[zLt[m][:, seen].argmax(1)] for m in MEMBERS}
        corr = np.stack([(pred[m] == yt) for m in MEMBERS])  # (M, n)
        k = corr.sum(0)  # members correct
        ens_pred = np.asarray(seen)[ens[:, seen].argmax(1)]
        ens_ok = ens_pred == yt
        best_m = max(MEMBERS, key=lambda m: per_acc[m])
        best_ok = pred[best_m] == yt

        n = len(yt)
        rows = []
        for kk in range(M + 1):
            sel = k == kk
            rows.append(
                {
                    "k": kk,
                    "mass": float(sel.mean()),
                    "n": int(sel.sum()),
                    "ens_acc_here": (
                        float(ens_ok[sel].mean()) if sel.any() else float("nan")
                    ),
                    "best_acc_here": (
                        float(best_ok[sel].mean()) if sel.any() else float("nan")
                    ),
                }
            )
        # Headroom = samples SOME member gets right but the BEST SINGLE member does not.
        head = (k >= 1) & (~best_ok)
        hmass = {
            kk: float(((k == kk) & head).sum()) / max(head.sum(), 1)
            for kk in range(1, M + 1)
        }
        inaccessible = sum(v for kk, v in hmass.items() if kk < MAJ)
        out = {
            "per_member": {m: per_acc[m] for m in MEMBERS},
            "best_single": float(best_ok.mean()),
            "ensemble": ens_acc,
            "oracle": float((k >= 1).mean()),
            "all_wrong": float((k == 0).mean()),
            "k_rows": rows,
            "headroom_frac_by_k": hmass,
            "headroom_below_majority": inaccessible,
            "ens_recovers_of_headroom": float((ens_ok & head).sum())
            / max(head.sum(), 1),
            "ens_breaks_of_unanimous": float((~ens_ok & (k == M)).sum())
            / max((k == M).sum(), 1),
            "n_test": int(n),
            "majority": MAJ,
            "members": MEMBERS,
        }

        # ---- controls against exp56's stored numbers for this exact cell
        if os.path.exists(EXP56):
            d56 = json.load(open(EXP56))
            pre = f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|f64"
            hit = [
                v
                for kk_, v in d56.items()
                if kk_.startswith(pre) and CO.order_tag() in kk_
            ]
            if hit:
                for m in MEMBERS:
                    want = hit[0][f"{m}|ranpac"]["A_last"]
                    assert abs(per_acc[m] - want) < 1e-9, (
                        f"{m}|ranpac {per_acc[m]:.10f} != exp56 {want:.10f}; this file is "
                        f"describing a different ensemble than the paper's."
                    )
                want_e = hit[0]["ens_ranpac"]["A_last"]
                assert (
                    abs(ens_acc - want_e) < 1e-9
                ), f"ens_ranpac {ens_acc:.10f} != exp56 {want_e:.10f}"
                out["verified_vs_exp56"] = True
                log(
                    "    VERIFY ok: all per-member and ensemble A_last match exp56 to 1e-9"
                )
            else:
                out["verified_vs_exp56"] = False
                log(
                    "    VERIFY: no exp56 cell for this member set (expected for seed-only)"
                )

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in _DS:
        for T in _TS:
            for seed in _SEEDS:
                key = f"{ds}|{T}|{seed}|{'+'.join(MEMBERS)}|m{M_RP}{CO.order_tag()}|v1"
                if key in allres:
                    log(f"skip {key}")
                    continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed)
                json.dump(allres, open(OUT, "w"), indent=2)
                r = allres[key]
                log(
                    f"    best {r['best_single']*100:.2f}  ens {r['ensemble']*100:.2f}  "
                    f"oracle {r['oracle']*100:.2f}  headroom<maj "
                    f"{r['headroom_below_majority']*100:.0f}%"
                )

    # ------------------------------------------------------------------ summary
    W = 96
    print("\n" + "=" * W)
    print(
        "EXP68 -- VOTE STRUCTURE: where the ensemble's unclaimed oracle headroom lives"
    )
    print("=" * W)
    cells = {}
    for k_, v in allres.items():
        p = k_.split("|")
        cells.setdefault((p[0], int(p[1])), []).append(v)

    def hk(v, kk):
        """headroom_frac_by_k[kk] regardless of whether this cell was just computed (int keys)
        or reloaded from JSON (str keys) -- json.dump stringifies int keys."""
        h = v["headroom_frac_by_k"]
        return h.get(kk, h.get(str(kk), 0.0))

    for (ds, T), vs in sorted(cells.items()):
        print(
            f"\n{ds} T={T}   {len(vs)} seed(s), members {vs[0]['members']}, "
            f"majority = {vs[0]['majority']}/{len(vs[0]['members'])}"
        )
        bs = np.mean([v["best_single"] for v in vs]) * 100
        en = np.mean([v["ensemble"] for v in vs]) * 100
        orc = np.mean([v["oracle"] for v in vs]) * 100
        aw = np.mean([v["all_wrong"] for v in vs]) * 100
        print(
            f"  best single {bs:.2f}   ensemble {en:.2f}   ORACLE {orc:.2f}   "
            f"no member right {aw:.2f}"
        )
        print(
            f"  headroom (oracle - best single) {orc-bs:+.2f} pts; "
            f"ensemble captures {en-bs:+.2f} = {(en-bs)/max(orc-bs,1e-9)*100:.0f}%"
        )
        print(
            f"\n  {'k correct':>10}{'mass':>9}{'ens acc here':>14}{'best acc here':>15}"
        )
        for kk in range(len(vs[0]["members"]) + 1):
            ms = np.mean([v["k_rows"][kk]["mass"] for v in vs]) * 100
            ea = np.nanmean([v["k_rows"][kk]["ens_acc_here"] for v in vs]) * 100
            ba = np.nanmean([v["k_rows"][kk]["best_acc_here"] for v in vs]) * 100
            print(f"  {kk:>10}{ms:>8.1f}%{ea:>13.1f}%{ba:>14.1f}%")
        print(f"\n  {'headroom mass by k':>22}", end="")
        for kk in range(1, len(vs[0]["members"]) + 1):
            print(f"  k={kk}: {np.mean([hk(v, kk) for v in vs])*100:.0f}%", end="")
        below = np.mean([v["headroom_below_majority"] for v in vs]) * 100
        rec = np.mean([v["ens_recovers_of_headroom"] for v in vs]) * 100
        brk = np.mean([v["ens_breaks_of_unanimous"] for v in vs]) * 100
        print(
            f"\n  headroom BELOW majority (k<{vs[0]['majority']}): {below:.0f}%   "
            f"ensemble recovers {rec:.0f}% of headroom   "
            f"breaks {brk:.2f}% of unanimous-correct"
        )
        verdict = (
            "COMBINER NOT FIXABLE -- headroom is minority-correct"
            if below >= 60
            else (
                "ACCESSIBLE -- a better combiner is worth one experiment"
                if below < 50
                else "AMBIGUOUS"
            )
        )
        print(f"  -> {verdict}")

    print("\n" + "-" * W)
    print(
        """HOW TO READ THIS
  1. `mass` at k=0 is the irreducible floor for THIS member pool: no combination rule, however
     clever, can classify those samples. It bounds every ensemble result in the paper.
  2. Uniform averaging behaves like a vote near the boundary, so it can only recover samples a
     MAJORITY already gets right. `headroom BELOW majority` is therefore the share of the
     unclaimed oracle gap that averaging is STRUCTURALLY unable to reach. Pre-registered: >=60%
     means the combiner is not the problem to fix -- the member pool is.
  3. `breaks % of unanimous-correct` is the cost side: samples every member gets right that the
     average gets wrong. Small here, but it is why the ensemble does not strictly dominate.
  4. Compare the structural pool against the seed-only pool (SUFFIX=_seedonly). C1 found their
     ensembles tie despite the structural pool having a HIGHER oracle; if its extra headroom
     sits below majority, that tie is explained and the diversity design is not at fault --
     the combiner is."""
    )
    print("=" * W)
    log(f"wrote {OUT}")
