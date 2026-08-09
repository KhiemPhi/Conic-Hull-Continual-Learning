#!/usr/bin/env python3
"""exp51_conic_loss_lab.py -- a bench for the UNTRIED conic-loss levers, with live progress.

WHY THIS FILE EXISTS
    exp48 answered "does an episodic conic loss do anything" with one configuration. The
    answer was: almost nothing in the `final` bracket (+0.32, inside the 0.34 seed range)
    and a lot in `drift` (+5.80 cone, +3.75 RanPAC). Six different training objectives have
    now landed in a 0.86-point band on CUB200 `final` cone-rp:

        sub -0.26 | ce_conic -0.40 | ce -0.72 | ce(e40-0) -0.72 | conic -0.79 | ce s1 -1.12

    with NO ordering by how conic the objective is -- `sub` (the non-conic control) is the
    best cell and `conic` (the purely conic one) is near the worst. Read that as: the
    read-out gap is not a property of the features, and the `final` bracket is the wrong
    place to look. THIS FILE DEFAULTS TO MEASURING `drift`, where the loss demonstrably
    does something.

    Everything here is a knob that exp48 hardcoded. Nothing here is a new objective.

THE FOUR LEVERS, in the order I would try them

  1. BANK=1  CROSS-STAGE NEGATIVES.  The single biggest untried lever.
        exp48's episode draws P classes from the CURRENT TASK ONLY, so the loss solves an
        8-way problem while the reader solves a 200-way one across all tasks -- and it
        never once sees the cross-stage confusion that actually degrades. This keeps each
        finished class's oPCA ray set (RAYS, not images -- still replay-free, zero storage
        growth in image terms) and appends BANK_C of them as extra logit columns. The bank
        cones are constants w.r.t. phi, but cone_energy still backprops through the QUERY
        argument (dE/dq = 2 V^T w*), so the gradient pushes current-task queries AWAY from
        past-class cones. That is the term exp48 structurally could not express.
        Bank rays are stored in the RAW space and re-whitened at use time with the current
        whitener, exactly as the read-out does (`A[c] @ Wh_inv` then `un(A[c] @ Wh)`) --
        storing them whitened would silently freeze them into a stale metric.

  2. MATCH_R=1  RAY-COUNT MATCHING.
        Training builds R_EP=4 rays; the read-out builds rays_for(n) = clip(n/5, 8, 128),
        which is 8 on CUB and 21-27 on ImageNet-R. So the loss optimises phi for a 4-ray
        cone that is never used. MATCH_R sets R_EP = rays_for(K_ROW-scaled rows) and raises
        N_SUP/K_ROW to keep >= 2 support rows per ray -- using raw support rows as rays is
        the `random generators` arm exp26 measured at 54.85 against k-means 74.25.
        NOTE this changes the batch size, so it is NOT a clean v1-vs-v2 style ablation.

  3. BATCHES=n  STEP BUDGET.
        CUB gets 6 batches/epoch, so a 15-epoch stage is 90 optimizer steps and a whole
        10-stage run is 1050. The loss may simply be undertrained and no existing number
        would reveal it. BATCHES overrides len(idx)//(P_CLS*K_ROW) directly, buying steps
        without touching the epoch schedule or the LR curve.

  4. TAU / LAM  never swept, either of them.
        The conic score is a norm of a projection of a unit vector, so it lives in [0,1];
        TAU=0.1 caps the logits at 10 and the 8-way CE saturates early. LAM=1 was a guess.

WHAT IT PRINTS
    A tqdm bar per stage over epochs, postfix = total / ce / conic loss and the EPISODE
    query accuracy (free -- it is already computed in the loss). At the end of every stage
    (or every PROBE_EVERY epochs) a PROBE: cone and NCM accuracy over all classes seen so
    far, using the read-out's own oPCA construction on a subsample. The probe is a proxy
    for watching, NOT a headline -- it subsamples and it has no RanPAC column. The headline
    is the full exp48 `evaluate` run once at the end, which produces exactly the same
    fields as every exp48 cell so the numbers drop straight into the existing tables.

READ IN THIS ORDER
    1. cone_A_last on `drift` vs the matched exp48 control.
         CUB200 s0:  ce 49.15   ce_conic 54.95      <- beat 54.95
    2. cone-rp on `drift`.  ce -7.08, ce_conic -5.02. Both readers moving means better
       features; only the cone moving means a reader-specific effect.
    3. epi_acc during training. If it is already ~1.0 by epoch 5 the episode is too easy
       and the loss has stopped producing gradient -- raise P_CLS or BANK_C, not epochs.
    4. gap_differential. -0.0020 (ce) -> +0.0090 (ce_conic). This is the one mechanism
       indicator that has ever moved the right way; negm and erank have not.

USAGE
    source ~/venvs/ml_env/bin/activate

    # smoke, ~5 min, writes to its own JSON
    DS=CUB200 T=2 EPOCHS_T0=1 EPOCHS_T=1 SUFFIX=_smoke python -u exp51_conic_loss_lab.py

    # lever 1 alone, the one worth running first (~80 min on CUB200)
    DS=CUB200 T=10 SEED=0 BANK=1 python -u exp51_conic_loss_lab.py

    # control for it -- identical file, bank off; this reproduces exp48's ce_conic2
    DS=CUB200 T=10 SEED=0 BANK=0 python -u exp51_conic_loss_lab.py

    # levers stacked
    DS=CUB200 T=10 SEED=0 BANK=1 MATCH_R=1 BATCHES=24 TAU=0.05 python -u exp51_conic_loss_lab.py

    Run cells SEQUENTIALLY. Concurrency on this box is measured strictly worse: four
    streams doubled IMAGENETA's per-task time from 206s to 414s. See run_exp48_grid.sh.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tqdm.auto import tqdm

import exp39_cone_construction as X                                    # noqa: E402
import fsa_train as F                                                  # noqa: E402
from backbone import freeze_non_lora, get_lora_params, load_backbone   # noqa: E402
# exp48 owns the pieces that must not diverge: the Danskin-exact cone energy, the
# oPCA episode generators, and `evaluate`. Importing rather than copying keeps every
# number here directly comparable to an exp48 cell. It reads the same env vars we do.
import exp48_conic_feature_loss as E48                                 # noqa: E402

T0 = time.time()


def log(m):
    tqdm.write(f"[{time.time()-T0:7.1f}s] {m}")


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
TAG = F.TAG
un = F.un

DS = os.environ.get("DS", "CUB200")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
PROTOCOLS = os.environ.get("PROTOCOL", "drift,final").split(",")
EPOCHS_T0 = int(os.environ.get("EPOCHS_T0", 40))
EPOCHS_T = int(os.environ.get("EPOCHS_T", 15))
LR = float(os.environ.get("LR", 3e-4))
LAM = float(os.environ.get("LAM", 1.0))
P_CLS = int(os.environ.get("P_CLS", 8))
K_ROW = int(os.environ.get("K_ROW", 12))
N_SUP = int(os.environ.get("N_SUP", 8))
R_EP = int(os.environ.get("R_EP", 4))
TAU = float(os.environ.get("TAU", 0.1))
GAMMA = float(os.environ.get("GAMMA", 0.5))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
WHITEN_EVERY = int(os.environ.get("WHITEN_EVERY", 20))
# --- the levers -----------------------------------------------------------------
BANK = int(os.environ.get("BANK", 1))          # 1 = cross-stage negatives on
BANK_C = int(os.environ.get("BANK_C", 24))     # past-class cones per episode
MATCH_R = int(os.environ.get("MATCH_R", 0))    # 1 = R_EP := rays_for(...), N_SUP/K_ROW up
BATCHES = int(os.environ.get("BATCHES", 0))    # 0 = auto (len(idx)//(P_CLS*K_ROW))
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", 0))   # 0 = end of stage only
PROBE_FIT = int(os.environ.get("PROBE_FIT", 10))      # train rows/class for the probe
PROBE_TEST = int(os.environ.get("PROBE_TEST", 1500))  # test rows sampled once per run
OUT = os.path.join(REPO, f"exp51_conic_loss_lab{os.environ.get('SUFFIX','')}_{TAG}.json")

if MATCH_R:
    # >=2 support rows per ray, and >=4 queries left over.
    R_EP = max(R_EP, E48.rays_for(K_ROW))
    N_SUP = max(N_SUP, 2 * R_EP)
    K_ROW = max(K_ROW, N_SUP + 4)


def cfg_key():
    return (f"{DS}|{T}|{SEED}|bank{BANK}x{BANK_C}|mr{MATCH_R}|b{BATCHES}"
            f"|e{EPOCHS_T0}-{EPOCHS_T}|lam{LAM:g}_t{TAU:g}"
            f"|P{P_CLS}K{K_ROW}s{N_SUP}R{R_EP}|g{GAMMA:g}|v1")


# ------------------------------------------------------------------ episode
def episode(fw, y, gen, bank_w):
    """One-draw support/query split (exact complement), oPCA generators matched to the
    read-out, plus BANK past-class cones as extra NEGATIVE-ONLY logit columns.

    bank_w: (n, R, d) whitened+normalised frozen ray sets, or None. They carry no gradient
    in V, but cone_energy still differentiates the QUERY, so the loss can push queries out
    of past-class cones -- which is the whole point of the bank."""
    splits, gmap = [], []
    for c in [int(v) for v in y.unique()]:
        idx = (y == c).nonzero(as_tuple=True)[0]
        if len(idx) < N_SUP + 1:
            continue
        perm = idx[torch.randperm(len(idx), generator=gen, device=fw.device)]
        splits.append((c, perm[:N_SUP], perm[N_SUP:]))
        gmap.append(c)
    if not splits:
        return None, None, None
    cols = []
    for c, sup, _ in splits:
        rays = E48._gens_opca(fw[sup], fw[y != c], min(R_EP, len(sup)), gen)
        cols.append(E48.cone_energy(rays, fw).clamp(min=1e-8).sqrt())
    n_own = len(cols)
    if bank_w is not None:
        for rays in bank_w:                      # constants in V, live in q
            cols.append(E48.cone_energy(rays, fw).clamp(min=1e-8).sqrt())
    is_q = torch.zeros(len(fw), dtype=torch.bool, device=fw.device)
    for _, _, qry in splits:
        is_q[qry] = True
    return torch.stack(cols, 1) / TAU, is_q, (gmap, n_own)


def make_bank(Zt, yt, classes, Wh, Wh_inv, rng):
    """Read-out-identical cones for finished classes, stored in the RAW space so they can
    be re-whitened later under whatever metric is current. Foreign material is the other
    classes of the same task -- the same set legally available when the class was born."""
    out = {}
    for c in classes:
        r = np.where(yt == c)[0]
        if len(r) < 2:
            continue
        oth = np.where((yt != c) & np.isin(yt, classes))[0]
        if len(oth) > 2000:
            oth = oth[rng.choice(len(oth), 2000, replace=False)]
        out[int(c)] = X.b_opca(un(Zt[r] @ Wh), un(Zt[oth] @ Wh),
                               E48.rays_for(len(r)), int(c), GAMMA) @ Wh_inv
    return out


# ------------------------------------------------------------------ probe
def probe(model, tr_ev, te_ev, ytr, yte, seen, fit_idx, test_idx):
    """Cheap read-out-shaped health check: cone and NCM over the classes seen so far.
    Subsampled, no RanPAC column -- for WATCHING, not for quoting."""
    model.eval()
    Ztr = un(F.extract(model, tr_ev, fit_idx))
    Zte = un(F.extract(model, te_ev, test_idx))
    ytr_f, yte_f = ytr[fit_idx], yte[test_idx]
    d = Ztr.shape[1]
    S = np.zeros((d, d), np.float64)
    n = 0
    for c in seen:
        r = np.where(ytr_f == c)[0]
        if len(r) < 2:
            continue
        Xc = Ztr[r] - Ztr[r].mean(0)
        S += Xc.T @ Xc
        n += len(Xc)
    S = S / max(n, 1)
    S = S + SHRINK * np.trace(S) / d * np.eye(d)
    Wh = np.linalg.cholesky(np.linalg.inv(S)).astype(np.float32)
    Wh_inv = np.linalg.inv(Wh).astype(np.float32)
    Qw = un(Zte @ Wh)
    Sc = np.full((len(test_idx), int(max(seen)) + 1), -np.inf, np.float32)
    Sn = np.full_like(Sc, -np.inf)
    for c in seen:
        r = np.where(ytr_f == c)[0]
        if len(r) < 2:
            continue
        oth = np.where(ytr_f != c)[0]
        A = X.b_opca(un(Ztr[r] @ Wh), un(Ztr[oth] @ Wh),
                     E48.rays_for(len(r)), int(c), GAMMA) @ Wh_inv
        Sc[:, c] = X.cone_score(un(A @ Wh), Qw)
        Sn[:, c] = Qw @ un(Ztr[r].mean(0, keepdims=True) @ Wh)[0]
    model.train()
    keep = np.isin(yte_f, list(seen))
    return (float((Sc[keep].argmax(1) == yte_f[keep]).mean()),
            float((Sn[keep].argmax(1) == yte_f[keep]).mean()))


# ------------------------------------------------------------------ train
def train():
    tag = cfg_key().replace("|", "_").replace("/", "_")
    cache = os.path.join(REPO, f"exp51_feats_{tag}_{TAG}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"cached {tag}")
        return z["Ftr"], z["Fte"], z["Ftr_on"], z["Fte_on"], json.loads(str(z["hist"]))

    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = F.get_data(DS)
    cpt = n_cls // T
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng0 = np.random.default_rng(SEED)
    order = rng0.permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    model = load_backbone(F.MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))
    d = model.num_features
    ce = nn.CrossEntropyLoss()
    gen = torch.Generator(device=DEV).manual_seed(SEED)

    Sacc = torch.zeros(d, d, device=DEV, dtype=torch.float64)   # cumulative, undecayed
    nacc = 0
    Wh = torch.eye(d, device=DEV)
    ON_tr = np.zeros((len(ytr), d), np.float32)
    ON_te = np.zeros((len(yte), d), np.float32)
    bank_raw, hist, step = {}, [], 0

    # probe subsamples, fixed once so the probe curve is comparable across stages
    test_idx = np.sort(rng0.choice(len(yte), min(PROBE_TEST, len(yte)), replace=False))
    fit_idx = np.sort(np.concatenate(
        [np.where(ytr == c)[0][:PROBE_FIT] for c in range(n_cls)]))

    log(f"cfg {cfg_key()}")
    log(f"levers: BANK={BANK}x{BANK_C}  MATCH_R={MATCH_R} -> P{P_CLS} K{K_ROW} "
        f"sup{N_SUP} R{R_EP}   BATCHES={BATCHES or 'auto'}  TAU={TAU}  LAM={LAM}")

    for t in range(T):
        idx = np.where(np.isin(ytr, tasks[t]))[0]
        remap = {int(c): i for i, c in enumerate(tasks[t])}
        head = nn.Linear(d, cpt).to(DEV)
        params = lp + list(head.parameters())
        ep = EPOCHS_T0 if t == 0 else EPOCHS_T
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(ep, 1))
        by_c = {int(c): idx[ytr[idx] == int(c)] for c in tasks[t]}
        by_c = {c: v for c, v in by_c.items() if len(v) >= N_SUP + 1}
        assert by_c, f"no class in task {t} has >= {N_SUP+1} rows"
        nb = BATCHES or max(len(idx) // (P_CLS * K_ROW), 1)

        bar = tqdm(range(ep), desc=f"stage {t}/{T-1}", unit="ep", dynamic_ncols=True)
        for e in bar:
            model.train()
            rng = np.random.default_rng(1000 * t + e)
            agg = np.zeros(4)
            for _ in tqdm(range(nb), desc=f"  ep {e}", leave=False,
                          unit="b", dynamic_ncols=True):
                cs = rng.choice(list(by_c), min(P_CLS, len(by_c)), replace=False)
                rows = np.concatenate([
                    by_c[c][rng.choice(len(by_c[c]), min(K_ROW, len(by_c[c])),
                                       replace=len(by_c[c]) < K_ROW)] for c in cs])
                xb = torch.stack([tr_aug[int(i)][0] for i in rows]).to(DEV)
                yb = torch.tensor([int(ytr[i]) for i in rows], device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = model(xb).float()

                with torch.no_grad():
                    S = torch.zeros(d, d, device=DEV, dtype=torch.float64)
                    for c in yb.unique():
                        z = f[yb == c].double()
                        z = z - z.mean(0, keepdim=True)
                        S += z.T @ z
                    Sacc += S
                    nacc += max(len(f), 1)
                    if step % WHITEN_EVERY == 0:
                        Sg = Sacc / nacc
                        Sr = Sg + SHRINK * torch.trace(Sg) / d * torch.eye(
                            d, device=DEV, dtype=torch.float64)
                        Wh = torch.linalg.cholesky(
                            torch.linalg.inv(Sr)).float().contiguous()
                step += 1

                l_ce = ce(head(f), torch.tensor(
                    [remap[int(v)] for v in yb.tolist()], device=DEV))
                fw = Fn.normalize(f @ Wh, dim=1)
                bank_w = None
                if BANK and bank_raw:
                    pick = rng.choice(list(bank_raw), min(BANK_C, len(bank_raw)),
                                      replace=False)
                    bank_w = [Fn.normalize(
                        torch.as_tensor(bank_raw[int(c)], device=DEV) @ Wh, dim=1)
                        for c in pick]
                lg, is_q, meta = episode(fw, yb, gen, bank_w)
                l_cn = torch.zeros((), device=DEV)
                qacc = 0.0
                if lg is not None and bool(is_q.any()):
                    gmap, n_own = meta
                    loc = {c: i for i, c in enumerate(gmap)}
                    tgt = torch.tensor([loc[int(v)] for v in yb.tolist()], device=DEV)
                    l_cn = ce(lg[is_q], tgt[is_q])
                    qacc = float((lg[is_q].argmax(1) == tgt[is_q]).float().mean())
                loss = l_ce + LAM * l_cn

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                agg += [float(loss), float(l_ce), float(l_cn), qacc]

            agg /= nb
            sch.step()
            bar.set_postfix(loss=f"{agg[0]:.3f}", ce=f"{agg[1]:.3f}",
                            conic=f"{agg[2]:.3f}", qacc=f"{agg[3]:.3f}",
                            bank=len(bank_raw))
            if PROBE_EVERY and (e + 1) % PROBE_EVERY == 0 and e + 1 < ep:
                seen = np.concatenate(tasks[:t + 1])
                c_a, n_a = probe(model, tr_ev, te_ev, ytr, yte, seen, fit_idx, test_idx)
                hist.append(dict(stage=t, epoch=e, cone=c_a, ncm=n_a, **{
                    k: float(v) for k, v in zip(("loss", "ce", "conic", "qacc"), agg)}))
                log(f"  probe s{t} e{e}: cone {100*c_a:.2f}  ncm {100*n_a:.2f}")
        bar.close()

        rows = np.where(np.isin(ytr, tasks[t]))[0]
        cols = np.where(np.isin(yte, tasks[t]))[0]
        ON_tr[rows] = F.extract(model, tr_ev, rows)
        ON_te[cols] = F.extract(model, te_ev, cols)

        # Bank this task's cones from the features that exist NOW -- the birth-stage
        # features, which is exactly what a replay-free system would have kept.
        if BANK:
            Zt = un(ON_tr[rows])
            Sg = (Sacc / max(nacc, 1)).cpu().numpy()
            Sg = Sg + SHRINK * np.trace(Sg) / d * np.eye(d)
            W = np.linalg.cholesky(np.linalg.inv(Sg)).astype(np.float32)
            bank_raw.update(make_bank(Zt, ytr[rows], tasks[t], W,
                                      np.linalg.inv(W).astype(np.float32), rng0))

        seen = np.concatenate(tasks[:t + 1])
        c_a, n_a = probe(model, tr_ev, te_ev, ytr, yte, seen, fit_idx, test_idx)
        hist.append(dict(stage=t, epoch=ep - 1, cone=c_a, ncm=n_a, end_of_stage=True))
        log(f"stage {t} done | probe over {len(seen)} classes: "
            f"cone {100*c_a:.2f}  ncm {100*n_a:.2f}  bank {len(bank_raw)}")

    Ftr, Fte = F.extract(model, tr_ev), F.extract(model, te_ev)
    del model
    torch.cuda.empty_cache()
    np.savez(cache, Ftr=Ftr, Fte=Fte, Ftr_on=ON_tr, Fte_on=ON_te,
             hist=json.dumps(hist))
    log(f"trained {tag}")
    return Ftr, Fte, ON_tr, ON_te, hist


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    _, _, ytr, _, yte, n_cls = F.get_data(DS)
    Ftr, Fte, ON_tr, ON_te, hist = train()
    for proto in PROTOCOLS:
        key = f"{cfg_key()}|{proto}"
        if key in allres:
            log(f"skip {key}")
            continue
        log(f"=== {key}")
        a, b = {"final": (Ftr, Fte), "drift": (ON_tr, Fte)}[proto]
        r = E48.evaluate(a, b, ytr, yte, n_cls)
        r["probe_hist"] = hist
        allres[key] = r
        json.dump(allres, open(OUT, "w"), indent=2)

    W = 96
    print("\n" + "=" * W)
    print(f"EXP51 -- conic-loss lab ({DS}, T={T}, seed {SEED})")
    print(cfg_key())
    print("=" * W)
    print(f"\n  {'proto':8}{'cone':>8}{'ranpac':>9}{'cone-rp':>9}{'sub':>8}"
          f"{'negm':>8}{'gapdiff':>10}{'erank':>8}")
    for proto in PROTOCOLS:
        r = allres.get(f"{cfg_key()}|{proto}")
        if not r:
            continue
        print(f"  {proto:8}{r['cone_A_last']*100:>8.2f}{r['ranpac_A_last']*100:>9.2f}"
              f"{(r['cone_A_last']-r['ranpac_A_last'])*100:>+9.2f}"
              f"{r['sub_A_last']*100:>8.2f}{r['negmass_own']*100:>7.1f}%"
              f"{r['gap_differential']:>+10.4f}{r['erank_mean']:>8.1f}")
    print("\n" + "-" * W)
    print("COMPARE AGAINST THE MATCHED exp48 CONTROLS, NOT THE FSA BAR (CUB200 s0):")
    print("   drift   ce 49.15 / rp 56.23 (-7.08)   ce_conic 54.95 / rp 59.98 (-5.02)")
    print("   final   ce 89.01 / rp 89.73 (-0.72)   ce_conic 89.33 / rp 89.73 (-0.40)")
    print("`drift` IS THE HEADLINE HERE. Six objectives now sit in a 0.86-point band on")
    print("   `final` cone-rp with no ordering by how conic they are; `final` has stopped")
    print("   discriminating between training objectives and `drift` has not.")
    print("If BANK moved cone AND ranpac together it improved the features; if it moved")
    print("   cone alone it is reader-specific. Both are results; they are not the same one.")
    print("=" * W)
    print(f"wrote {OUT}")
