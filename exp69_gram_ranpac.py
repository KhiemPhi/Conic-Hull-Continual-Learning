#!/usr/bin/env python3
"""exp69_gram_ranpac.py -- does Gram-preserving CONTINUAL adaptation + a RanPAC head beat
first-session-only adaptation on a benchmark that HAS adaptation headroom?

THE CHAIN THIS TESTS, AND WHY EACH LINK IS ALREADY MEASURED
    1. We lose to GR-LoRA exactly on the datasets where our BASE is behind, not where our
       read-out is behind (readout-is-exhausted-gap-is-features: per-dataset A-Avg base vs
       GR-LoRA is +0.48 / -0.89 / -1.58 / -0.74, and fusion only ever adds +0.03..+0.63).
    2. Our base is behind because we adapt ONCE; GR-LoRA / SSIAT / MACIL adapt EVERY task.
    3. We adapt once because continual adaptation DEGRADES old features. Measured on
       CIFAR-100: with no penalty the current-frame ORACLE falls to 0.749 against a frozen
       backbone at 0.815, and the gap widens with depth.
    4. A cosine-Gram penalty FIXES that degradation: oracle 0.749 -> 0.816 >= frozen 0.815,
       rigidity 80.6% -> 91.8%, new-task accuracy unchanged (crux_relational, 3 seeds).
    5. So the constraint that FORCED first-session-only is removed, and ImageNet-R has
       +10.83 points of real RanPAC headroom (frozen 0.7272 -> joint LoRA r32 0.8355,
       measured OFFLINE by crux_headroom, so it is a genuine ceiling and not a CIL artifact).

    Step 5 is why this runs on IMAGENETR and not CIFAR-100. Every negative result in the
    Gram line was measured on CIFAR-100, where adaptation HURTS because the backbone is
    already near its ceiling -- that was benchmark-specific, not fundamental.

THE OBJECTION THAT MAKES THIS A REAL TEST AND NOT A FORMALITY
    The Gram penalty is derived from the gauge freedom of a PROTOTYPE read-out: cosine
    similarity is O(d)-invariant, so a simultaneous rotation of features and prototypes is
    unobservable, and Gram is exactly the O(d) quotient.

    RanPAC's head is NOT O(d)-invariant. It scores h = ReLU(z @ P) for a FIXED random P, and
    ReLU(RzP) != ReLU(zP) for a rotation R -- the nonlinearity breaks the symmetry the penalty
    protects. So preserving Gram does NOT preserve RanPAC's accumulated G/C statistics, and
    there is a principled reason the composition might fail even though each half works. That
    is the scientific content here: does gauge preservation help a read-out that is not
    gauge-invariant?

    This also forces a design decision the naive version gets wrong. Accumulating G/C across
    tasks while the backbone moves mixes statistics from different frames; covariance-cone-
    penalty-result measured that collapse directly (drifting accumulation 0.3537 A-last vs
    0.7065 for the same method with the backbone held still). So the `cont*` arms RECOMPUTE
    the RanPAC statistics each stage from the replay buffer plus the current task -- which is
    honest, because the method ALREADY stores that replay for the Gram loss and pays no extra
    storage. The naive accumulating variant is run too, as `cont50_accum`, so the collapse is
    visible rather than assumed.

ARMS (env ARMS, default all)
    frozen        no adaptation at all; RanPAC on phi_0. The floor.
    fs            first-session-only LoRA on task 0 then FREEZE; RanPAC accumulated across
                  tasks. This is OUR CURRENT METHOD'S BASE and the number to beat
                  (IMAGENETR T=10: q32|ranpac = 80.76 A-Last / 85.49 A-Avg, 3 seeds).
    cont0         continual LoRA every task, NO penalty; RanPAC recomputed from replay.
                  Isolates "adapt every task" from "adapt every task SAFELY". Expected to be
                  the worst trained arm if step 3 above generalises to ImageNet-R.
    cont50        continual LoRA + cosine-Gram penalty; RanPAC recomputed from replay.
                  THE PROPOSAL.
    cont50_accum  cont50 but with NAIVELY accumulated RanPAC stats, to expose the
                  frame-mixing collapse.

PRE-REGISTERED, WRITTEN BEFORE THE RUN
    Primary: cont50 beats `fs` by >= +1.0 A-Last on IMAGENETR T=10. Below +0.3 the whole
    "adapt every task safely" path is dead and first-session-only was the right call.
    Secondary and NECESSARY: cont50 - cont0 >= +0.5. If cont50 ~= cont0 then the Gram penalty
    is not the mechanism -- continual adaptation alone did it -- and the paper's derivation
    does not get credit for the gain.
    Diagnostic: rigidity and Gram-corr must MOVE in the cont50 arm relative to cont0, the way
    they did on CIFAR-100. If they do not, the penalty is not biting on this dataset and a
    null result says nothing about the hypothesis.

    Also recorded: NCM alongside RanPAC for every arm. The Gram penalty was validated with an
    NCM read-out, so if it lifts NCM but not RanPAC that localises the failure to the
    non-invariance of the random-ReLU lift rather than to the penalty.

PROTOCOL MATCHES THE PAPER'S TABLE
    PILOT class order (class_order.py, MT19937 1993+s), splits.py, vit_base_patch16_224
    .augreg_in21k, LoRA r32 task_shared alpha 4, M_RP 10000, lambda grid on val. Deviating on
    any of these would make the result incomparable to the 12-cell table -- which is the whole
    point of running it.

USAGE
    source ~/venvs/ml_env/bin/activate
    # pilot: one seed, all arms -- does it improve AT ALL?
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORDER=pilot \\
      DS=IMAGENETR T=10 SEED=0 python -u exp69_gram_ranpac.py

    # then scale
    ... SEED=0,1,2 python -u exp69_gram_ranpac.py
    # lambda sensitivity
    ... SEED=0 ARMS=cont0,cont50 LAM_GRID=10,50,200 python -u exp69_gram_ranpac.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

_DS = os.environ.get("DS", "IMAGENETR").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])
os.environ.setdefault("MODEL", "vit_base_patch16_224.augreg_in21k")
for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import class_order as CO  # noqa: E402
import fsa_train as F  # noqa: E402
from backbone import freeze_non_lora, get_lora_params, load_backbone  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ["MODEL"]
TAG = F.TAG
ARMS = os.environ.get("ARMS", "frozen,fs,cont0,cont50,cont50_accum").split(",")
LAM_GRID = [float(x) for x in os.environ.get("LAM_GRID", "50").split(",")]
EPOCHS_T0 = int(os.environ.get("EPOCHS_T0", 40))  # task 0 matches the table's recipe
EPOCHS_T = int(os.environ.get("EPOCHS_T", 4))  # later tasks: crux_relational's budget
LR = float(os.environ.get("LR", 3e-4))
LR_T = float(os.environ.get("LR_T", 1e-4))
BS = int(os.environ.get("BS", 128))
M_REPLAY = int(os.environ.get("M_REPLAY", 20))
GB = int(os.environ.get("GB", 128))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
GRAD_CLIP = 1.0
OUT = os.path.join(REPO, f"exp69_gram_ranpac{os.environ.get('SUFFIX', '')}_{TAG}.json")

assert set(ARMS) <= {"frozen", "fs", "cont0", "cont50", "cont50_accum"}, ARMS
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}); exp49 measured the unpinned noise "
        f"floor at 0.27, comparable to the effect under test."
    )

un = F.un


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def deg(c):
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ------------------------------------------------------------------ read-outs
class RanPAC:
    """Staged RanPAC. `accum` mixes frames when the backbone moves -- that is the point of the
    cont50_accum arm -- while recompute=True rebuilds G/C from whatever rows it is given.
    """

    def __init__(self, d, n_cls):
        self.P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(
            DEV
        )
        self.eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
        self.n_cls = n_cls
        self.reset()

    def reset(self):
        self.G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
        self.C = torch.zeros(M_RP, self.n_cls, device=DEV, dtype=torch.float64)

    def _h(self, Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(
                torch.as_tensor(Z[i : i + bs], device=DEV, dtype=torch.float32) @ self.P
            )

    def add(self, Z, y):
        for i, h in self._h(un(Z)):
            h = h.double()
            Y = torch.zeros(h.shape[0], self.n_cls, device=DEV, dtype=torch.float64)
            Y[
                torch.arange(h.shape[0]),
                torch.tensor(y[i : i + h.shape[0]], device=DEV),
            ] = 1.0
            self.G += h.T @ h
            self.C += h.T @ Y

    def logits(self, Z, W):
        return torch.cat([(h.double() @ W) for _, h in self._h(un(Z))]).cpu().numpy()

    def fit_predict(self, Zv, yv, Zt, seen):
        """lambda chosen on val, then applied to test -- the table's convention."""
        best, bw = -1.0, None
        for lam in LAMBDAS:
            W = torch.linalg.solve(self.G + lam * self.eye, self.C)
            a = ncm_acc_from_logits(self.logits(Zv, W), seen, yv)
            if a > best:
                best, bw = a, W
        return self.logits(Zt, bw), best


def ncm_acc_from_logits(L, seen, y):
    return float((np.asarray(seen)[L[:, seen].argmax(1)] == y).mean())


def ncm(Ztr, ytr_, Zte, yte_, seen):
    """Nearest-class-mean on unit features -- the read-out the Gram penalty was validated with."""
    M = np.stack([unit(unit(Ztr[ytr_ == c]).mean(0, keepdims=True))[0] for c in seen])
    pred = np.asarray(seen)[np.argmax(unit(Zte) @ M.T, axis=1)]
    return float((pred == yte_).mean())


# ------------------------------------------------------------------ one arm
def run_arm(arm, ds, T, seed):
    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = F.get_data(ds)
    cpt = n_cls // T
    order = CO.class_order(n_cls, seed)
    tasks = [order[i * cpt : (i + 1) * cpt] for i in range(T)]

    # 90/10 fit/val carve per task, keyed on the task index -- exp56's convention.
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]])
        FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    # replay: M_REPLAY rows/class, drawn from FIT only so val stays clean
    rep_idx = {}
    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            rep_idx[int(c)] = r[:M_REPLAY]

    torch.manual_seed(seed * 1000)
    np.random.seed(seed)
    lora = 0 if arm == "frozen" else 32
    model = load_backbone(
        MODEL,
        pretrained=True,
        num_classes=0,
        device=DEV,
        lora_rank=lora,
        lora_alpha=4.0,
        lora_config="task_shared",
    )
    if lora:
        freeze_non_lora(model)
        lp = list(get_lora_params(model))
    d = model.num_features
    rp = RanPAC(d, n_cls)
    recompute = arm in ("cont0", "cont50")
    lam_g = 50.0 if arm.startswith("cont50") else 0.0
    if arm.startswith("cont") and LAM_GRID != [50.0] and lam_g > 0:
        lam_g = LAM_GRID[0]

    def train_task(t, epochs, lr, use_gram):
        """LoRA + throwaway head on task t's FIT rows, optional Gram penalty on old replay."""
        cls = list(tasks[t])
        remap = {int(c): i for i, c in enumerate(cls)}
        ld = DataLoader(
            Subset(tr_aug, FIT[t].tolist()),
            batch_size=BS,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        head = nn.Linear(d, len(cls)).to(DEV)
        opt = torch.optim.AdamW(lp + list(head.parameters()), lr=lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs * len(ld), 1)
        )
        ce = nn.CrossEntropyLoss()

        # reference geometry of OLD replay, in the CURRENT frame, detached. Snapshotting it
        # here (not at t=0) is what makes the penalty forbid FURTHER distortion rather than
        # demand a return to phi_0 -- the latter would just re-freeze the backbone, which is
        # the structural cap that closed the anti-drift conic family.
        B_ref, Xrep = None, None
        old = [int(c) for tk in tasks[:t] for c in tk]
        if use_gram and old:
            rows = np.concatenate([rep_idx[c] for c in old])
            Xrep = torch.cat(
                [
                    x
                    for x, _ in DataLoader(
                        Subset(tr_ev, rows.tolist()), batch_size=256, num_workers=8
                    )
                ],
                0,
            )
            model.eval()
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"
            ):
                B_ref = torch.cat(
                    [
                        model(Xrep[i : i + 256].to(DEV)).float()
                        for i in range(0, len(Xrep), 256)
                    ],
                    0,
                )
            B_ref = Fn.normalize(B_ref, dim=1)

        gl = 0.0
        for ep in range(epochs):
            model.train()
            for x, y in ld:
                x = x.to(DEV, non_blocking=True)
                yy = torch.tensor([remap[int(v)] for v in y], device=DEV)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"
                ):
                    loss = ce(head(model(x)).float(), yy)
                if B_ref is not None:
                    ridx = torch.randint(0, B_ref.shape[0], (min(GB, B_ref.shape[0]),))
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"
                    ):
                        A = Fn.normalize(model(Xrep[ridx].to(DEV)).float(), dim=1)
                    with torch.no_grad():
                        Gref = B_ref[ridx] @ B_ref[ridx].T
                    l = Fn.mse_loss(A @ A.T, Gref)
                    loss = loss + lam_g * l
                    gl = float(l)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lp + list(head.parameters()), GRAD_CLIP)
                opt.step()
                sched.step()
        del head, opt
        return gl

    # ---- gauge diagnostics need the task-0 cohort's birth geometry
    cohort = [int(c) for c in tasks[0]]
    birth = {}
    hist = {
        k: [] for k in ("ranpac", "ncm", "eps", "rigid_deg", "gram_corr", "gram_loss")
    }

    for t in range(T):
        gl = 0.0
        if arm == "fs" and t == 0:
            gl = train_task(0, EPOCHS_T0, LR, use_gram=False)
        elif arm.startswith("cont"):
            gl = train_task(
                t,
                EPOCHS_T0 if t == 0 else EPOCHS_T,
                LR if t == 0 else LR_T,
                use_gram=(lam_g > 0 and t > 0),
            )
        # arm == "frozen": never trains

        seen = np.concatenate(tasks[: t + 1])
        nval = sum(len(v) for v in VAL[: t + 1])
        vix = VAL_ALL[:nval]
        tei = np.where(np.isin(yte, seen))[0]

        Zte = F.extract(model, te_ev)[tei]
        Zv = F.extract(model, tr_ev, vix)
        if recompute:
            # rebuild from replay(old) + this task's FIT rows -- the method's own storage.
            # Extract ONCE and reuse for both the RanPAC fit and the NCM prototypes; a second
            # F.extract on the same rows is a full forward pass over them for nothing.
            rows = (
                np.concatenate(
                    [rep_idx[int(c)] for tk in tasks[:t] for c in tk] + [FIT[t]]
                )
                if t
                else FIT[t]
            )
            Zfit, yfit = F.extract(model, tr_ev, rows), ytr[rows]
            rp.reset()
            rp.add(Zfit, yfit)
        else:
            rp.add(F.extract(model, tr_ev, FIT[t]), ytr[FIT[t]])
            rows = np.concatenate([rep_idx[int(c)] for c in seen])
            Zfit, yfit = F.extract(model, tr_ev, rows), ytr[rows]

        Lt, _ = rp.fit_predict(Zv, ytr[vix], Zte, seen)
        hist["ranpac"].append(ncm_acc_from_logits(Lt, seen, yte[tei]))
        hist["ncm"].append(ncm(Zfit, yfit, Zte, yte[tei], seen))

        # ---- gauge diagnostics on the task-0 cohort
        Ztr_coh = F.extract(model, tr_ev, np.concatenate([rep_idx[c] for c in cohort]))
        ylab = np.concatenate([np.full(len(rep_idx[c]), c) for c in cohort])
        mu = {
            c: unit(unit(Ztr_coh[ylab == c]).mean(0, keepdims=True))[0] for c in cohort
        }
        if t == 0:
            birth = {"rows": Ztr_coh.copy(), "mu": dict(mu)}
            hist["eps"].append(0.0)
            hist["rigid_deg"].append(0.0)
            hist["gram_corr"].append(1.0)
        else:
            A_ = unit(Ztr_coh)
            B_ = unit(birth["rows"])
            U, _, Vt = np.linalg.svd(A_.T @ B_)
            R = U @ Vt
            eps = float(np.mean([deg(birth["mu"][c] @ mu[c]) for c in cohort]))
            res = float(
                np.mean(
                    [deg(birth["mu"][c] @ unit((mu[c] @ R)[None])[0]) for c in cohort]
                )
            )
            M0 = np.stack([birth["mu"][c] for c in cohort])
            M1 = np.stack([mu[c] for c in cohort])
            off = ~np.eye(len(cohort), dtype=bool)
            hist["eps"].append(eps)
            hist["rigid_deg"].append(res)
            hist["gram_corr"].append(
                float(np.corrcoef((M0 @ M0.T)[off], (M1 @ M1.T)[off])[0, 1])
            )
        hist["gram_loss"].append(gl)
        rg = (
            100 * (1 - hist["rigid_deg"][-1] / hist["eps"][-1])
            if hist["eps"][-1] > 1.0
            else float("nan")
        )
        log(
            f"    [{arm}] t={t} seen={len(seen):3d}  ranpac {hist['ranpac'][-1]*100:6.2f}  "
            f"ncm {hist['ncm'][-1]*100:6.2f}  eps {hist['eps'][-1]:4.0f}d  "
            f"rigid {rg:4.0f}%  gram {hist['gram_corr'][-1]:.2f}"
        )

    del model
    if lora:
        del lp
    torch.cuda.empty_cache()
    out = {k: v for k, v in hist.items()}
    for r in ("ranpac", "ncm"):
        out[f"{r}_A_last"] = hist[r][-1]
        out[f"{r}_A_avg"] = float(np.mean(hist[r]))
    out["_meta"] = {
        "arm": arm,
        "lam_gram": lam_g,
        "recompute_ranpac": recompute,
        "epochs": [EPOCHS_T0, EPOCHS_T],
        "m_replay": M_REPLAY,
        "order": CO.mode(),
        "model": MODEL,
    }
    return out


# ------------------------------------------------------------------ driver
if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in _DS:
        for T in _TS:
            for seed in _SEEDS:
                for arm in ARMS:
                    key = (
                        f"{ds}|{T}|{seed}|{arm}|e{EPOCHS_T0}-{EPOCHS_T}"
                        f"|lr{LR:g}-{LR_T:g}|m{M_REPLAY}|M{M_RP}"
                        f"{CO.order_tag()}|v1"
                    )
                    if key in allres:
                        log(f"skip {key}")
                        continue
                    log(f"=== {key}")
                    t_ = time.time()
                    allres[key] = run_arm(arm, ds, T, seed)
                    allres[key]["_meta"]["secs"] = round(time.time() - t_, 1)
                    json.dump(allres, open(OUT, "w"), indent=2)
                    r = allres[key]
                    log(
                        f"    DONE {arm}: ranpac {r['ranpac_A_last']*100:.2f}/"
                        f"{r['ranpac_A_avg']*100:.2f}  ncm {r['ncm_A_last']*100:.2f}/"
                        f"{r['ncm_A_avg']*100:.2f}  ({time.time()-t_:.0f}s)"
                    )

    # ------------------------------------------------------------------ summary
    W = 96
    print("\n" + "=" * W)
    print(
        "EXP69 -- Gram-preserving CONTINUAL adaptation + RanPAC, on a benchmark WITH headroom"
    )
    print("=" * W)
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        cells.setdefault((p[0], int(p[1])), {}).setdefault(p[3], []).append(v)

    # published / our own reference points, ViT-B/16-IN21k, PILOT order, 3 seeds
    REF = {
        ("IMAGENETR", 10): {
            "GR-LoRA": (82.09, 86.20),
            "our FE (5-member+cone)": (82.58, 87.01),
            "our base q32|ranpac": (80.76, 85.49),
        }
    }
    for (ds, T), arms in sorted(cells.items()):
        print(f"\n{ds} T={T}")
        print(
            f"  {'arm':<14}{'n':>3}{'RanPAC A-Last':>15}{'RanPAC A-Avg':>14}"
            f"{'NCM A-Last':>12}{'rigidity%':>11}{'gram':>7}"
        )
        for arm in ("frozen", "fs", "cont0", "cont50", "cont50_accum"):
            if arm not in arms:
                continue
            vs = arms[arm]
            rl = np.mean([v["ranpac_A_last"] for v in vs]) * 100
            ra = np.mean([v["ranpac_A_avg"] for v in vs]) * 100
            nl = np.mean([v["ncm_A_last"] for v in vs]) * 100
            e, r_ = np.mean([v["eps"][-1] for v in vs]), np.mean(
                [v["rigid_deg"][-1] for v in vs]
            )
            rg = 100 * (1 - r_ / e) if e > 1.0 else float("nan")
            g = np.mean([v["gram_corr"][-1] for v in vs])
            print(
                f"  {arm:<14}{len(vs):>3}{rl:>15.2f}{ra:>14.2f}{nl:>12.2f}{rg:>11.1f}{g:>7.2f}"
            )
        for nm, (a, b) in REF.get((ds, T), {}).items():
            print(f"  {'['+nm+']':<14}{'':>3}{a:>15.2f}{b:>14.2f}")

        # ---- the pre-registered tests
        def gm(arm, f="ranpac_A_last"):
            return (
                (np.mean([v[f] for v in arms[arm]]) * 100)
                if arm in arms
                else float("nan")
            )

        fs, c0, c50 = gm("fs"), gm("cont0"), gm("cont50")
        print(f"\n  PRE-REGISTERED")
        print(
            f"    primary   cont50 - fs    = {c50-fs:+.2f}   "
            f"(ALIVE >= +1.00, DEAD < +0.30)"
        )
        print(
            f"    necessary cont50 - cont0 = {c50-c0:+.2f}   "
            f"(the PENALTY must be the mechanism, >= +0.50)"
        )
        v = (
            "ALIVE -- safe continual adaptation beats first-session-only"
            if (c50 - fs) >= 1.0 and (c50 - c0) >= 0.5
            else (
                "DEAD -- first-session-only was the right call"
                if (c50 - fs) < 0.3
                else "AMBIGUOUS / mechanism unclear"
            )
        )
        print(f"    -> {v}")

    print("\n" + "-" * W)
    print(
        """HOW TO READ THIS
  1. `fs` is our published base, not a straw man. Beating it is the entire point; losing to it
     says first-session-only adaptation was correct and the Gram line does not rescue it.
  2. cont0 vs cont50 is what assigns CREDIT. cont50 > cont0 means the penalty did it; a tie
     means plain continual adaptation did it and the gauge derivation earns nothing.
  3. cont50_accum exposes frame mixing: RanPAC's ReLU lift is NOT O(d)-invariant, so
     accumulating G/C while the backbone rotates is invalid even when Gram is preserved. A
     large accum-vs-recompute gap is a result about the READ-OUT, not the adaptation.
  4. NCM vs RanPAC localises any failure. The penalty was validated with NCM; if it lifts NCM
     and not RanPAC, the problem is the non-invariant random-ReLU lift, not the penalty.
  5. rigidity and gram must MOVE in cont50 vs cont0. If they do not, the penalty is not biting
     on this dataset and a null tells us nothing about the hypothesis."""
    )
    print("=" * W)
    log(f"wrote {OUT}")
