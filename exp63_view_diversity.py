#!/usr/bin/env python3
"""exp63_view_diversity.py -- the two gates for a cone-native multi-view method.

WHY THESE TWO THINGS, TOGETHER, ON CACHED DATA
    The proposed architecture (CMV-Cone) ensembles over a grid of ADAPTER x VIEW, where views
    are extra read-outs of a SINGLE forward pass, and shortlists classes with the cheap `pm`
    rule before paying for NNLS. Two unknowns decide whether that is an architecture or a
    sketch, and both are answerable from features already on disk.

  GATE 1 -- DO THE VIEWS DECORRELATE?
    exp37 extracted {CLS, GAP} x blocks {5,8,11} from one adapted backbone, CONCATENATED them,
    and lost: multi_cls 79.77 vs plain cls 80.37. But concatenation is not ensembling, and
    exp55 measured ensemble BEATING concat by +0.63 A-Avg on the LoRA members (81.62 vs 80.99).
    So the ensembling question was never asked. If six views of one forward pass decorrelate
    even moderately, the ensemble axis costs ONE backbone pass instead of five.

    Three families are compared on the same protocol, so the numbers mean the same thing:
       shape      q32,m32,a16,q32b70,q64        5 passes   adapter architecture (current)
       objective  ce,ce_kd0.1,ce_kd0.5,          5 passes   TRAINING LOSS -- the axis most
                  cosine_s16,proto_s16                      likely to genuinely decorrelate,
                                                            since different objectives find
                                                            different solutions rather than
                                                            the same one via other parameters
       view       cls/gap x blocks 5,8,11        1 PASS     read-out of one backbone
    Reference: exp55 measured the shape family at errcorr 0.75-0.82 with 6.22 headroom.

  GATE 2 -- HOW DEEP MUST THE `pm` SHORTLIST BE?
    Inference is NNLS-BOUND, not backbone-bound: 37 ms of cone scoring against 15.9 ms of
    backbone. Cutting passes from 5 to 1 therefore saves almost nothing (53 -> 48 ms). What
    changes the picture is scoring only a shortlist: `pm` is a matmul with no NNLS and scored
    78.56 standalone, which is far more than enough to SHORTLIST. If top-20 of 200 recalls the
    true class >=99.5% of the time, the NNLS budget drops 10x and total inference goes to
    ~8 ms -- 4x faster than GR-LoRA and constant in T. If it needs top-100, the argument halves.

EVERYTHING HERE IS LEGACY CLASS ORDER
    exp37_pools and fsa_feats were produced before class_order.py, so they exist only under
    ORDER=legacy. That is fine and it is the RIGHT comparison, because exp55's legacy diversity
    block (errcorr 0.75-0.82, headroom 6.22, seed-0 ensemble A-Last 81.12) is the baseline.
    Do NOT mix these numbers with the pilot-order results table.

RANPAC-ONLY FOR THE FAMILIES
    exp62 measured cone members and RanPAC members at the SAME diversity (errcorr 0.79 vs 0.78,
    headroom 5.85 vs 6.17), so RanPAC answers the diversity question at a fifth of the cost.
    Cones are built only for gate 2, and only for one feature set.

USAGE
    source ~/venvs/ml_env/bin/activate
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=legacy DS=IMAGENETR T=10 SEED=0 \
      python -u exp63_view_diversity.py
"""
import itertools
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
LAMBDAS = S54.LAMBDAS
SHRINK, GAMMA, F_MAX = S54.SHRINK, S54.GAMMA, S54.F_MAX
DO_GATE2 = int(os.environ.get("GATE2", 1))
KS = [1, 5, 10, 20, 40, 100]

un, zs, acc_v1 = X.un, S54.zs, S54.acc_v1
score, rays_for = S54.score, S54.rays_for

assert CO.mode() == "legacy", (
    f"ORDER={CO.mode()} but exp37_pools / fsa_feats exist only in LEGACY order. Re-run with "
    f"ORDER=legacy, and do not mix these numbers with the pilot-order results table.")
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], f"threads not pinned (OMP={_th[0]} MKL={_th[1]})"


# ------------------------------------------------------------------ feature families
def shape_family():
    out = {}
    f = os.path.join(REPO, f"exp16_feats_{DS}_T{T}_s{SEED}_ep40_lr0.0003_aug1_{TAG}.npz")
    if os.path.exists(f):
        z = np.load(f); out["q32"] = (un(z["Ftr"]), un(z["Fte"]))
    for m in ("m32", "a16", "q32b70", "q64"):
        f = os.path.join(REPO, f"exp55_feats_{DS}_T{T}_s{SEED}_{m}_ep40_lr0.0003_{TAG}.npz")
        if os.path.exists(f):
            z = np.load(f); out[m] = (un(z["Ftr"]), un(z["Fte"]))
    return out


def objective_family():
    out = {}
    for lbl, suf in (("ce", "ce"), ("kd0.1", "ce_kd_kd0.1"), ("kd0.5", "ce_kd_kd0.5"),
                     ("cos", "cosine_cos_s16"), ("proto", "cosine_proto_s16")):
        f = os.path.join(REPO,
                         f"fsa_feats_{DS}_T{T}_s{SEED}_ep40_lr0.0003_aug1_{suf}_{TAG}.npz")
        if os.path.exists(f):
            z = np.load(f); out[lbl] = (un(z["Ftr"]), un(z["Fte"]))
    return out


def view_family():
    f = os.path.join(REPO, f"exp37_pools_{DS}_T{T}_s{SEED}_ep40_lr0.0003_{TAG}.npz")
    if not os.path.exists(f):
        return {}
    z = np.load(f)
    out = {}
    for pool in ("cls", "gap"):
        for blk in (5, 8, 11):
            k = f"{pool}|{blk}"
            if f"tr|{k}" in z.files:
                out[k] = (un(z[f"tr|{k}"]), un(z[f"te|{k}"]))
    return out


# ------------------------------------------------------------------ staged RanPAC
def ranpac_stages(Ztr, Zte, ytr, yte, tasks):
    d = Ztr.shape[1]
    n_cls = int(max(ytr.max(), yte.max())) + 1
    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()

    FIT, VAL = [], []
    for t in range(len(tasks)):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)

    accs, zl, tidx = [], [], []
    for t in range(len(tasks)):
        for i, h in _H(Ztr[FIT[t]]):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        vix = VAL_ALL[:sum(len(v) for v in VAL[:t + 1])]
        tei = np.where(np.isin(yte, seen))[0]
        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc_v1(logits(Ztr[vix], Wm), seen, ytr[vix])
            if a > best:
                best, bw = a, Wm
        Lt = zs(logits(Zte, bw)[tei], seen)
        accs.append(acc_v1(Lt, seen, yte[tei]))
        zl.append(Lt); tidx.append((tei, seen))
    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return accs, zl, tidx


def diversity_block(zl_last, tei, seen, yte, names):
    yt = yte[tei]
    pr = {n: np.asarray(seen)[zl_last[n][:, seen].argmax(1)] for n in names}
    co = {n: (pr[n] == yt) for n in names}
    dis, ec = [], []
    for i, j in itertools.combinations(names, 2):
        ci, cj = co[i].astype(float), co[j].astype(float)
        sd = ci.std() * cj.std()
        dis.append(float((pr[i] != pr[j]).mean()))
        ec.append(float(((ci - ci.mean()) * (cj - cj.mean())).mean() / sd)
                  if sd > 1e-12 else np.nan)
    anyc = np.zeros(len(yt), bool)
    for n in names:
        anyc |= co[n]
    bs = max(float(co[n].mean()) for n in names)
    return (float(np.mean(dis)) * 100, float(np.nanmean(ec)), bs * 100,
            float(anyc.mean()) * 100)


if __name__ == "__main__":
    E.T, E.SEED = T, SEED
    ytr, yte, n_cls = E.get_labels(DS)
    cpt = n_cls // T
    order = CO.class_order(n_cls, SEED)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    fams = {"shape": shape_family(), "objective": objective_family(), "view": view_family()}
    W = 96
    print("=" * W)
    print(f"EXP63 -- view / objective / shape diversity   {DS} T={T} seed={SEED} "
          f"ORDER={CO.mode()}")
    print("=" * W)
    for f, mem in fams.items():
        print(f"  {f:<10} {len(mem)} members: {list(mem)}")

    out = {}
    print(f"\n[GATE 1] per-family diversity and uniform ensemble  (RanPAC read-out)")
    print(f"  {'family':<11}{'n':>3}{'passes':>8}{'disagree':>10}{'errcorr':>9}"
          f"{'best1':>8}{'ORACLE':>8}{'headroom':>10}{'ens A-Last':>12}{'gain':>7}")
    for fam, mem in fams.items():
        if len(mem) < 2:
            print(f"  {fam:<11}{len(mem):>3}   -- not enough cached members, skipped")
            continue
        names = list(mem)
        zl, accs, tidx = {}, {}, None
        for n in names:
            a, z, ti = ranpac_stages(mem[n][0], mem[n][1], ytr, yte, tasks)
            accs[n] = a; zl[n] = z; tidx = ti
            log(f"    {fam}/{n}: A-Last {a[-1]*100:.2f}  A-Avg {np.mean(a)*100:.2f}")
        tei, seen = tidx[-1]
        ens = []
        for t in range(T):
            te_t, se_t = tidx[t]
            ens.append(acc_v1(sum(zl[n][t] for n in names) / len(names), se_t, yte[te_t]))
        dis, ec, bs, orc = diversity_block({n: zl[n][-1] for n in names}, tei, seen,
                                           yte, names)
        passes = 1 if fam == "view" else len(names)
        print(f"  {fam:<11}{len(names):>3}{passes:>8}{dis:>9.1f}%{ec:>9.2f}{bs:>8.2f}"
              f"{orc:>8.2f}{orc-bs:>10.2f}{ens[-1]*100:>12.2f}{ens[-1]*100-bs:>+7.2f}")
        out[fam] = {"members": {n: {"A_last": accs[n][-1], "A_avg": float(np.mean(accs[n]))}
                                for n in names},
                    "ens": {"A_last": ens[-1], "A_avg": float(np.mean(ens)), "accs": ens},
                    "disagree": dis, "errcorr": ec, "best_single": bs, "oracle": orc,
                    "passes": passes}
    print("  reference: exp55 measured the SHAPE family at errcorr 0.75-0.82, headroom 6.22.\n"
          "  `view` costs ONE forward pass -- judge its gain per pass, not in absolute terms.")

    if DO_GATE2 and "q32" in fams["shape"]:
        print(f"\n[GATE 2] `pm` shortlist recall -- how deep before the cone NNLS?")
        Ztr, Zte = fams["shape"]["q32"]
        d = Ztr.shape[1]
        FIT = []
        for t in range(T):
            ix = np.where(np.isin(ytr, tasks[t]))[0]
            pm_ = np.random.default_rng(t).permutation(len(ix))
            FIT.append(ix[pm_[max(int(0.1 * len(ix)), 1):]])
        sc, ns, A = np.zeros((d, d)), 0, {}
        for t in range(T):
            for c in tasks[t]:
                r = FIT[t][ytr[FIT[t]] == c]
                if len(r) < 2:
                    continue
                Xc = Ztr[r] - Ztr[r].mean(0)
                sc += Xc.T @ Xc; ns += len(Xc)
            S_ = sc / max(ns, 1)
            S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
            Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
            Wi = np.linalg.inv(Wh)
            for c in tasks[t]:
                r = FIT[t][ytr[FIT[t]] == c]
                if len(r) < 2:
                    continue
                Xw = un(Ztr[r] @ Wh)
                rng = np.random.default_rng(1234 + 97 * t + zlib.crc32(b"f64") % 1000)
                oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                past = [A[o] for o in A if o not in tasks[t]]
                Fr = np.concatenate([Ztr[oth]] + past, 0)
                if len(Fr) > F_MAX:
                    Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                A[int(c)] = X.BUILD[S54.METHOD](Xw, un(Fr), rays_for("f64", len(r)),
                                                int(c), GAMMA) @ Wi
        log(f"    built {len(A)} cones")
        seen = np.concatenate(tasks)
        tei = np.where(np.isin(yte, seen))[0]
        Q = un(Zte[tei] @ Wh)
        cls = sorted(A)
        Spm = np.full((len(tei), len(cls)), -np.inf, np.float32)
        for j, c in enumerate(cls):
            Spm[:, j] = score("pm", un(A[c] @ Wh), Q)
        rank = np.argsort(-Spm, 1)
        truth = np.array([cls.index(int(v)) if int(v) in cls else -1 for v in yte[tei]])
        print(f"  {'k':>5}{'recall of true class':>24}{'NNLS per query':>17}"
              f"{'est. NNLS ms':>14}")
        rec = {}
        for k in KS:
            hit = float(np.mean([(truth[i] in rank[i, :k]) for i in range(len(tei))])) * 100
            rec[k] = hit
            print(f"  {k:>5}{hit:>23.2f}%{k:>17}{k*0.037:>14.2f}")
        print(f"  {len(cls):>5}{'100.00':>23}%{len(cls):>17}{len(cls)*0.037:>14.2f}"
              f"   <- no shortlist (current)")
        out["gate2_pm_recall"] = rec

    json.dump(out, open(os.path.join(REPO, f"exp63_view_div_{DS}_s{SEED}_{TAG}.json"), "w"),
              indent=2)
    print("\n" + "-" * W)
    print("""HOW TO READ THIS
  GATE 1. Compare `view` to `shape` on errcorr and headroom. `view` costs ONE forward pass, so
    even 60% of the shape family's ensemble gain makes it the better deal per unit compute.
    errcorr >= 0.85 with headroom well under 6 means six read-outs of one backbone are simply
    the same classifier six times and the view axis is closed.
  GATE 1b. `objective` is the axis most likely to genuinely decorrelate -- its members were
    trained with DIFFERENT LOSSES, not different parameterisations of the same loss. All of
    them are individually WORSE than ce (exp30: kd0.1 78.87, kd0.5 76.03; exp32: cos 79.45,
    proto 78.77), which is the m32 situation and not an objection.
  GATE 2. Inference is NNLS-bound (37 ms of 53 ms). Read the smallest k whose recall is
    >=99.5%: that sets the NNLS budget. k=20 -> ~4.4 ms -> total ~8 ms, 4x faster than
    GR-LoRA and constant in T. k=100 -> the cost argument halves.""")
    print("=" * W)
