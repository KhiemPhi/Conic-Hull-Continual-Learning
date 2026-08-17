#!/usr/bin/env python3
"""exp64_cmv_cone.py -- CMV-Cone: a cone-native ensemble over MULTI-VIEW features.

THE METHOD
    Every number in this project reads ONE vector per image: the final-block CLS token. A
    ViT-B/16 computes 12 blocks and 196 patch tokens on every forward pass and throws almost
    all of it away. exp37 cached six read-outs of the SAME adapted backbone -- {CLS, GAP} x
    blocks {5, 8, 11} -- so an ensemble over views costs ONE forward pass instead of five.

    CMV-Cone: per view, its own whitener and its own oPCA cone bank; predict by the uniform
    average of row-z-scored cone scores. No RanPAC, no random projection, no Gram.

WHY THIS IS NOT exp37 AGAIN
    exp37 CONCATENATED the views and ran one RanPAC, and lost: multi_cls 79.77 against plain
    cls 80.37. It never ENSEMBLED them. exp55 measured ensembling BEATING concatenation by
    +0.63 A-Avg on the LoRA members (81.62 vs 80.99), so the concatenation result does not
    settle the ensembling question. That gap is what this file fills.

ARMS -- every one a UNIFORM average, nothing fitted, nothing selected on test
    cone_1        cls|11 cone only                       1 pass   the single-view baseline
    rp_1          cls|11 RanPAC only                     1 pass   the exp54 `ranpac` analogue
    cone_views    all 6 views, cone                      1 PASS   <- THE METHOD
    cone_cls      cls|5,8,11 only                        1 pass   declared subset
    cone_b11      cls|11 + gap|11 only                   1 pass   declared subset
    rp_views      all 6 views, RanPAC                    1 pass   is any gain view- or
                                                                  read-out-specific?
    fe_views      per-view RanPAC+cone fusion, averaged  1 pass   the full analogue of FE
    cone_obj      5 TRAINING OBJECTIVES, cone            5 passes ce/kd0.1/kd0.5/cos/proto
    rp_obj        same, RanPAC                           5 passes

    The three view subsets are STRUCTURAL choices declared up front (all / CLS-only /
    last-block-only), not a search. Blocks 5 and 8 are individually much weaker -- exp37 had
    multi_gap at 77.20 -- so a uniform average over all six may be dragged down, and the
    subsets show that rather than hiding it.

LEGACY CLASS ORDER, AND WHY THAT IS THE RIGHT COMPARISON
    exp37_pools and fsa_feats predate class_order.py and exist only under ORDER=legacy. The
    baselines are therefore the LEGACY ones: exp54 measured cone-alone 79.92/84.87 and
    RanPAC-alone 80.41/85.31; the 5-member shape ensemble at legacy seed 0 is 81.12 A-Last.
    DO NOT put these numbers in the pilot-order results table.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=legacy DS=IMAGENETR T=10 SEED=0 \
      python -u exp64_cmv_cone.py
    # FAMS=view  to skip the 5-objective family and halve the runtime
"""
import json
import os
import time
import warnings
import zlib

import numpy as np
import torch

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

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = S54.TAG
DS = os.environ.get("DS", "IMAGENETR")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
R = int(os.environ.get("R", 64))
M_RP = int(os.environ.get("MRP", 10000))
FAMS = os.environ.get("FAMS", "view,obj").split(",")
LAMBDAS, SHRINK, GAMMA, F_MAX = S54.LAMBDAS, S54.SHRINK, S54.GAMMA, S54.F_MAX
OUT = os.path.join(REPO, f"exp64_cmv_{DS}_T{T}_s{SEED}_{TAG}.json")

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
score, rays_for = S54.score, S54.rays_for
VIEWS = ["cls|11", "cls|8", "cls|5", "gap|11", "gap|8", "gap|5"]
OBJS = [("ce", "ce"), ("kd0.1", "ce_kd_kd0.1"), ("kd0.5", "ce_kd_kd0.5"),
        ("cos", "cosine_cos_s16"), ("proto", "cosine_proto_s16")]

assert CO.mode() == "legacy", (
    f"ORDER={CO.mode()} but exp37_pools/fsa_feats exist only in LEGACY order. Re-run with "
    f"ORDER=legacy and keep these numbers OUT of the pilot results table.")
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], f"threads not pinned (OMP={_th[0]} MKL={_th[1]})"


def load_views():
    f = os.path.join(REPO, f"exp37_pools_{DS}_T{T}_s{SEED}_ep40_lr0.0003_{TAG}.npz")
    assert os.path.exists(f), f"missing {f}"
    z = np.load(f)
    return {v: (un(z[f"tr|{v}"]), un(z[f"te|{v}"])) for v in VIEWS if f"tr|{v}" in z.files}


def load_objs():
    out = {}
    for lbl, suf in OBJS:
        f = os.path.join(REPO,
                         f"fsa_feats_{DS}_T{T}_s{SEED}_ep40_lr0.0003_aug1_{suf}_{TAG}.npz")
        if os.path.exists(f):
            z = np.load(f)
            out[lbl] = (un(z["Ftr"]), un(z["Fte"]))
    return out


def run_family(mem, ytr, yte, n_cls, tasks, label):
    """Staged protocol. Returns per-stage cone / ranpac / fused z-scores per member."""
    names = list(mem)
    d = mem[names[0]][0].shape[1]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    G = {n: torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64) for n in names}
    C = {n: torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64) for n in names}
    scat = {n: (np.zeros((d, d)), 0) for n in names}
    A = {n: {} for n in names}

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()

    per_stage = []
    for t in range(T):
        seen = np.concatenate(tasks[:t + 1])
        vix = VAL_ALL[:sum(len(v) for v in VAL[:t + 1])]
        yv = ytr[vix]
        tei = np.where(np.isin(yte, seen))[0]
        col = {int(c): j for j, c in enumerate(seen)}
        tcv = np.array([col[int(v)] for v in yv])
        zc, zr, zf = {}, {}, {}
        for n in names:
            Ztr_m, Zte_m = mem[n]
            sc, ns = scat[n]
            for c in tasks[t]:
                r = FIT[t][ytr[FIT[t]] == c]
                if len(r) < 2:
                    continue
                Xc = Ztr_m[r] - Ztr_m[r].mean(0)
                sc += Xc.T @ Xc; ns += len(Xc)
            scat[n] = (sc, ns)
            S_ = sc / max(ns, 1)
            S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
            Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
            Wi = np.linalg.inv(Wh)
            for c in tasks[t]:
                r = FIT[t][ytr[FIT[t]] == c]
                if len(r) < 2:
                    continue
                Xw = un(Ztr_m[r] @ Wh)
                rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(b"f64") % 1000)
                oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                past = [A[n][o] for o in A[n] if o not in tasks[t]]
                Fr = np.concatenate([Ztr_m[oth]] + past, 0)
                if len(Fr) > F_MAX:
                    Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                A[n][int(c)] = X.BUILD[S54.METHOD](Xw, un(Fr), rays_for("f64", len(r)),
                                                   int(c), GAMMA) @ Wi
            for i, h in _H(Ztr_m[FIT[t]]):
                h = h.double()
                Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
                Y[torch.arange(h.shape[0]),
                  torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
                G[n] += h.T @ h; C[n] += h.T @ Y
            best, bw = -1.0, None
            for lam in LAMBDAS:
                Wm = torch.linalg.solve(G[n] + lam * eye, C[n])
                a = acc_v1(logits(Ztr_m[vix], Wm), seen, yv)
                if a > best:
                    best, bw = a, Wm
            zLv = zs(logits(Ztr_m[vix], bw), seen)
            zr[n] = zs(logits(Zte_m, bw)[tei], seen)
            Qvw, Qtw = un(Ztr_m[vix] @ Wh), un(Zte_m[tei] @ Wh)
            miss = [c for c in seen if c not in A[n]]
            Sv = np.full((len(vix), n_cls), -np.inf, np.float32)
            St = np.full((len(tei), n_cls), -np.inf, np.float32)
            for c in seen:
                if c not in A[n]:
                    continue
                Ac = un(A[n][c] @ Wh)
                Sv[:, c] = score("cone", Ac, Qvw)
                St[:, c] = score("cone", Ac, Qtw)
            cv_, ct_ = zs(Sv, seen), zs(St, seen)
            if miss:
                cv_[:, miss] = 0.0
                ct_[:, miss] = 0.0
            zc[n] = ct_
            b, _ = S54.pick_beta_v2(zLv, cv_, seen, tcv, S54.BETAS_V2)
            zf[n] = zs(zr[n] + b * ct_, seen)
        per_stage.append((tei, seen, zc, zr, zf))
        log(f"    {label} s{t}: " +
            "  ".join(f"{n} {acc_v1(zc[n], seen, yte[tei])*100:.1f}" for n in names[:3]))
    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return per_stage


def avg_arm(per_stage, subset, which, yte):
    accs = []
    for tei, seen, zc, zr, zf in per_stage:
        src = {"cone": zc, "rp": zr, "fe": zf}[which]
        accs.append(acc_v1(sum(src[n] for n in subset) / len(subset), seen, yte[tei]))
    return accs


if __name__ == "__main__":
    E.T, E.SEED = T, SEED
    ytr, yte, n_cls = E.get_labels(DS)
    cpt = n_cls // T
    order = CO.class_order(n_cls, SEED)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    res = {}

    if "view" in FAMS:
        mv = load_views()
        log(f"view family: {list(mv)}")
        ps = run_family(mv, ytr, yte, n_cls, tasks, "view")
        vn = list(mv)
        res["cone_1"] = avg_arm(ps, ["cls|11"], "cone", yte)
        res["rp_1"] = avg_arm(ps, ["cls|11"], "rp", yte)
        res["cone_views"] = avg_arm(ps, vn, "cone", yte)
        res["cone_cls"] = avg_arm(ps, [v for v in vn if v.startswith("cls")], "cone", yte)
        res["cone_b11"] = avg_arm(ps, [v for v in vn if v.endswith("11")], "cone", yte)
        res["rp_views"] = avg_arm(ps, vn, "rp", yte)
        res["fe_views"] = avg_arm(ps, vn, "fe", yte)
        for v in vn:
            res[f"solo_cone_{v}"] = avg_arm(ps, [v], "cone", yte)
        json.dump(res, open(OUT, "w"), indent=2)

    if "obj" in FAMS:
        mo = load_objs()
        log(f"objective family: {list(mo)}")
        if len(mo) >= 2:
            ps = run_family(mo, ytr, yte, n_cls, tasks, "obj")
            on = list(mo)
            res["cone_obj"] = avg_arm(ps, on, "cone", yte)
            res["rp_obj"] = avg_arm(ps, on, "rp", yte)
            res["fe_obj"] = avg_arm(ps, on, "fe", yte)
            for n in on:
                res[f"solo_cone_obj_{n}"] = avg_arm(ps, [n], "cone", yte)
            json.dump(res, open(OUT, "w"), indent=2)

    W = 88
    print("\n" + "=" * W)
    print(f"EXP64 -- CMV-Cone   {DS} T={T} seed={SEED} ORDER={CO.mode()} R={R}")
    print("=" * W)
    passes = {"cone_1": 1, "rp_1": 1, "cone_views": 1, "cone_cls": 1, "cone_b11": 1,
              "rp_views": 1, "fe_views": 1, "cone_obj": 5, "rp_obj": 5, "fe_obj": 5}
    print(f"  {'arm':<16}{'A-Last':>9}{'A-Avg':>9}{'passes':>8}{'vs cone_1':>11}")
    base = res["cone_1"][-1] * 100 if "cone_1" in res else float("nan")
    for n in ("cone_1", "rp_1", "cone_b11", "cone_cls", "cone_views", "rp_views",
              "fe_views", "cone_obj", "rp_obj", "fe_obj"):
        if n not in res:
            continue
        a = np.array(res[n]) * 100
        print(f"  {n:<16}{a[-1]:>9.2f}{a.mean():>9.2f}{passes[n]:>8}{a[-1]-base:>+11.2f}")
    print(f"\n  per-view solo cone (A-Last / A-Avg)")
    for k in sorted(res):
        if k.startswith("solo_"):
            a = np.array(res[k]) * 100
            print(f"    {k[5:]:<20}{a[-1]:>8.2f}{a.mean():>9.2f}")
    print(f"""
  LEGACY-ORDER REFERENCES (exp54/exp56, same protocol, seed 0)
    cone alone   79.92 / 84.87       RanPAC alone  80.41 / 85.31
    RanPAC+cone fused  81.52 / 85.94
    5-member SHAPE ensemble (5 passes), seed 0:  81.12 A-Last
  Judge cone_views against cone_1 -- it is the same forward pass. Judge it against the
  5-member shape ensemble on ACCURACY PER PASS.""")
    print("=" * W)
    log(f"wrote {OUT}")
