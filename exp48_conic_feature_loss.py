#!/usr/bin/env python3
"""exp48_conic_feature_loss.py — train phi FOR the conic reader, across all T tasks.

THE CONFOUND THIS EXISTS TO TEST
    Every read-out comparison in this project (and in the CIL literature) scores different
    heads on features trained with a linear softmax. CE optimises LINEAR SEPARABILITY,
    which is exactly what a ridge consumes and which says nothing about angular or conic
    structure. So the comparison has been tilted toward the ridge by the feature objective
    from the start.

    That a feature space can favour one reader over the other is already measured, not
    speculated: RanPAC's random ReLU lift is worth +9.26 to the ridge and -0.07 to the cone
    (exp33/exp38), because it doubles mean pairwise cosine 0.207 -> 0.429. Two readers,
    same features, opposite signs.

    THE 2x2 IS THE EXPERIMENT:
                          conic reader     ridge (RanPAC)
        CE-trained            80.44            80.41        (exp49, measured)
        conic-trained           ?                ?
    The publishable outcome is bottom-left > bottom-right while the top row ties, which
    would mean read-out choice and feature objective are NOT separable. If the conic loss
    lifts both arms equally it is merely better features and there is no result. That is
    what `ranpac_same` measures and it is the only column that matters.

THE DEFECT BEING TARGETED, measured
    exp45: at the method's own atoms, 43.7% of the unconstrained least-squares coefficient
    mass on the TRUE class is negative -- a query from class c is not a non-negative
    combination of c's own generators. The constraint binds on 100% of query-class pairs.
    exp29: within-class coverage predicts per-class error (rho -0.585); between-class
    overlap does not (+0.037). The head wants each class conically spanned by few
    directions; nothing in CE asks for that.

THE LOSS: THE READER'S OWN SCORE AS THE LOGIT
    Class-balanced batches of P classes x K rows. Split each class into support and query,
    build the class cone from the SUPPORT, score every query against all P cones, and apply
    CE over {s_c(q)/tau} with s_c = ||Pi_C_c q||^2. This trains the actual decision rule
    rather than proxies for it, and it directly asks that R support rows conically span the
    class -- the property exp29 measured as binding.

    An earlier draft of this file used three hand-designed terms (own-rows-in, foreign-
    rows-out, drift-linearity). That was the wrong shape: pushing scores up and down
    independently does not optimise the MARGIN, and errors are made of margin. It also had
    a vacuous term -- "sample z from cone C_o and require z stays in C_o" has no dependence
    on phi, since z is built from V_o and is inside by construction.

GRADIENTS: EXACT, WITHOUT DIFFERENTIATING THE SOLVER
    ||Pi_C q||^2 = max_{w>=0} ( 2 w^T V q - w^T V V^T w ).  Solve for w* under no_grad, then
    evaluate that quadratic with w* DETACHED and V, q attached:
        E = 2 <proj, q> - ||proj||^2,      proj = w*.detach() @ V
    Its value is ||Pi q||^2, and by Danskin its gradients are exact in BOTH arguments:
        dE/dq = 2 V^T w*          dE/dV = 2 w* q^T - 2 w* proj^T
    NOTE the second term. Using the shorter form <q, proj> alone gives the right gradient
    for q but a WRONG one for V (it drops -2 w* proj^T), which matters here because the
    generators are functions of the support features and therefore of phi.

TRAIN AND TEST IN THE SAME GEOMETRY
    Deployment scores in the whitened space and the metric is 73% of the total gain
    (paper ledger: +6.65 of +9.05). Computing the loss in the raw normalised space would
    throw most of that away. We maintain an EMA of the pooled within-class scatter and
    apply its Cholesky whitener -- DETACHED, refreshed every WHITEN_EVERY steps -- inside
    the loss, so the training geometry matches the read-out geometry.

ARMS
    ce        plain CE. The exp16 objective, extended to all T tasks. Baseline.
    conic     episodic conic CE, above.
    sub       episodic SUBSPACE CE: identical episodes, identical support split, but the
              support is orthonormalised by QR and scored ||B_c^T q||. This is exp31's
              never-run objective and it is the TRAINING-SIDE mirror of exp38's constraint
              suite: conic - sub isolates non-negativity in the loss, exactly as cone - sub
              isolated it in the read-out. Given exp45 found non-negativity worth only
              +0.60 and only at discriminative atoms, these two are expected to land close;
              that expectation is why the control runs alongside rather than after.
    ce_conic  CE + LAM * conic. Hedge against the conic loss alone being too weak a
              training signal early, when P classes give few negatives.
    ce_conic2 ce_conic with three defects of the v1 episode repaired. Same objective, same
              batch geometry, same step count -- so ce_conic2 - ce_conic attributes the
              difference to the episode, not to a change of budget.
                1. SUPPORT/QUERY SPLIT. v1 drew two independent permutations, so the query
                   set was not the complement of the support set: at K_ROW=12, N_SUP=8,
                   2.66 of the 4 query rows per class were also support rows (measured).
                   Those rows are averaged into their own class's generators and are inside
                   the cone by construction, so ~67% of the query signal was near-vacuous
                   -- the same defect the discarded "sample z from C_o" term had. v2 draws
                   once and takes the exact complement.
                2. GENERATORS. v1 built rays as R_EP arbitrary interleaved group means of
                   the support, in the full whitened space, blind to other classes. That is
                   approximately the PRE-exp39 cone. The read-out this loss exists to serve
                   uses b_opca: k-means inside the top-R eigenspace of S_c - GAMMA*S_F. v2
                   matches it (see _gens_opca). Ray selection is the largest measured lever
                   in this project (spa 36.15 vs kmeans 74.25 at R=4, exp26), so training
                   for the wrong ray set is not a detail.
                3. WHITENER. v1 used an EMA at 0.95, which over ~90 steps/stage has an
                   effective window of ~20 steps and therefore tracks the CURRENT task. The
                   read-out uses a pooled scatter accumulated over every class seen so far
                   and never decayed. v2 accumulates. The docstring's own premise is that
                   train and test share a geometry; an EMA does not.
              NOT changed, deliberately, so the comparison stays attributable: ray COUNT
              (still R_EP, vs rays_for(n) at read-out), and the absence of cross-stage
              negatives (episodes still draw only from the current task).

WHERE TO LOOK
    CUB-200, not ImageNet-R. exp49 found the cone ties on CIFAR-100 / ImageNet-R /
    ImageNet-A and loses -1.05 on CUB-200 (0/6 cells) -- the only genuine deficit, and the
    one place with a full point of headroom rather than a 0.03 tie to defend. CUB-200 has
    27 fit rows/class, so with 8-24 rays the cone is fitting noise; compact, positively
    spanned classes should help there most.

LEADING INDICATORS -- READ BEFORE THE ACCURACY COLUMN
    negmass_own      43.7%   -> should fall toward 0
    gap_differential -0.0101 -> should turn POSITIVE
    erank            --      -> should DROP, and the R_c optimum with it
    erank_t0 vs erank_rest: the loss sees cpt classes and serves n_cls. If rank drops only
    for task-0 classes, it overfitted the first session and cannot help.
    If accuracy moves and none of these do, whatever happened is not the stated mechanism.

PROTOCOLS THAT BRACKET THE TRUTH
    final   extract everything with the final model (equivalent to perfect transport).
            OPTIMISTIC. Answers "are the features better for a cone?" in isolation.
    drift   class rays built from BIRTH-STAGE train features, queries extracted with the
            FINAL model. No transport. PESSIMISTIC, and the honest deployment bracket:
            the rays are as stale as the stage that wrote them, the query is whatever the
            deployed network currently emits.
    online  BROKEN -- DO NOT QUOTE. LABEL LEAKAGE. `ON_te[cols]` at line ~328 selects test
            rows by their TRUE label's task and extracts them with that task's model, so
            scoring a query requires already knowing its class in order to pick its feature
            extractor. It also compares scores across classes whose features came from
            DIFFERENT networks, which is exp19's "which LoRA produced this vector" confound
            turned into a free discriminant. Measured on CUB200 s0 ce: 97.34 A-Last against
            89.01 for `final` -- a supposedly PESSIMISTIC protocol reading 8.3 points ABOVE
            the optimistic one, which is the tell. Kept only so the number in the JSON has
            an explanation attached; use `drift`.

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=CUB200 T=10 SEED=0 python -u exp48_conic_feature_loss.py
    DS=IMAGENETR T=10 SEED=0 ARMS=ce,conic python -u exp48_conic_feature_loss.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Subset

# exp19_dataset_hull parses T and SEED as scalars at import time; keep them scalar here.
import exp39_cone_construction as X          # noqa: E402
import fsa_train as F                        # noqa: E402
from backbone import freeze_non_lora, get_lora_params, load_backbone  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
TAG = F.TAG
DS = os.environ.get("DS", "CUB200")
T = int(os.environ.get("T", 10))
SEED = int(os.environ.get("SEED", 0))
ARMS = os.environ.get("ARMS", "ce,conic,sub,ce_conic").split(",")
PROTOCOLS = os.environ.get("PROTOCOL", "final").split(",")
EPOCHS_T0 = int(os.environ.get("EPOCHS_T0", 40))
EPOCHS_T = int(os.environ.get("EPOCHS_T", 15))
LR = float(os.environ.get("LR", 3e-4))
LAM = float(os.environ.get("LAM", 1.0))          # weight on the conic term in ce_conic
P_CLS = int(os.environ.get("P_CLS", 8))          # classes per episode
K_ROW = int(os.environ.get("K_ROW", 12))         # rows per class per episode
N_SUP = int(os.environ.get("N_SUP", 8))          # support rows; the rest are queries
R_EP = int(os.environ.get("R_EP", 4))            # generators per class in an episode
TAU = float(os.environ.get("TAU", 0.1))
NNLS_TRAIN = int(os.environ.get("NNLS_TRAIN", 30))
WHITEN_EVERY = int(os.environ.get("WHITEN_EVERY", 20))
EMA = float(os.environ.get("EMA", 0.95))
GAMMA = float(os.environ.get("GAMMA", 0.5))
KVAL = float(os.environ.get("KVAL", 5))
KM_ITERS = int(os.environ.get("KM_ITERS", 5))    # Lloyd steps in the v2 episode k-means
RMIN = int(os.environ.get("RMIN", 8))
RMAX = int(os.environ.get("RMAX", 128))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
# SUFFIX exists so several (dataset, seed) cells can run CONCURRENTLY. Each process loads
# the whole result dict at startup and rewrites it after every key, so two processes
# sharing one file silently drop each other's cells -- last writer wins with a stale dict.
# One file per stream, merged afterwards. The feature caches are keyed per cell and are
# written once, so they are safe to share.
OUT = os.path.join(REPO,
                   f"exp48_conic_feature_loss{os.environ.get('SUFFIX', '')}_{TAG}.json")
assert all(a in ("ce", "conic", "sub", "ce_conic", "ce_conic2")
           for a in ARMS), f"bad arm in {ARMS}"
un = X.un


def rays_for(n):
    return int(np.clip(n / KVAL, RMIN, RMAX))


# ------------------------------------------------------------------ conic machinery
def nnls_w(V, Q, iters):
    """FISTA for min_{w>=0} ||Q - w V||^2, batched over rows of Q. no_grad only."""
    Vt = V.t()
    L = torch.linalg.matrix_norm(V @ Vt, 2).clamp(min=1e-6)
    w = torch.zeros(Q.shape[0], V.shape[0], device=Q.device, dtype=Q.dtype)
    z, tk = w.clone(), 1.0
    QV, VV = Q @ Vt, V @ Vt
    for _ in range(iters):
        wn = torch.clamp(z - (z @ VV - QV) / L, min=0.0)
        tn = 0.5 * (1 + (1 + 4 * tk * tk) ** 0.5)
        z = wn + ((tk - 1) / tn) * (wn - w)
        w, tk = wn, tn
    return w


def cone_energy(V, Q, iters=NNLS_TRAIN):
    """||Pi_C(V) q||^2 with EXACT Danskin gradients in both V and Q. See module docstring:
    the -||proj||^2 term is required, dropping it gives a wrong dE/dV."""
    with torch.no_grad():
        w = nnls_w(V.detach(), Q.detach(), iters)
    proj = w @ V
    return 2.0 * (proj * Q).sum(1) - (proj * proj).sum(1)


def _gens_opca(sup, foreign, R, gen):
    """v2 generators, matched to the read-out's b_opca: k-means inside the top-R eigenspace
    of S_c - GAMMA*S_F, lifted back to the whitened space.

    DIFFERENTIABILITY, by the same trick cone_energy uses for the NNLS weights: the
    subspace V and the cluster ASSIGNMENT are chosen under no_grad and detached -- they are
    geometry choices, like the whitener -- and the centroids are then recomputed as means
    of the assigned support rows with the gradient attached. So dV/dphi flows through the
    averaging, exactly as dE/dV flows through w*.detach() @ V. The docstring's claim that
    "k-means is not differentiable" is true of the assignment and false of the centroid,
    and only the centroid needs a gradient.

    The eigenproblem is solved inside the support's OWN SPAN (rank <= N_SUP). That is exact
    for every direction a centroid can occupy -- centroids are means of support rows, so
    they lie in that span -- and turns a 768x768 eigh into an N_SUP x N_SUP one.
    """
    with torch.no_grad():
        Xs = sup.detach()
        Q, _ = torch.linalg.qr(Xs.T)                              # (d, ns) span basis
        Zs = Xs @ Q
        M = (Zs.T @ Zs) / len(Zs)
        if len(foreign):
            Zf = foreign.detach() @ Q
            M = M - GAMMA * (Zf.T @ Zf) / len(Zf)
        k = int(min(R, M.shape[0]))
        V = Q @ torch.linalg.eigh(M)[1].flip(-1)[:, :k]           # (d, k), descending
        Z = Fn.normalize(Xs @ V, dim=1)
        C = Z[torch.randperm(len(Z), generator=gen, device=Z.device)[:k]].clone()
        for _ in range(KM_ITERS):                                 # spherical k-means
            a = (Z @ C.T).argmax(1)
            for j in range(k):
                m = a == j
                if bool(m.any()):
                    C[j] = Fn.normalize(Z[m].mean(0), dim=0)
        assign = (Z @ C.T).argmax(1)
    g = [sup[assign == j].mean(0) for j in range(k) if bool((assign == j).any())]
    return Fn.normalize(torch.stack(g), dim=1)


def episode_logits(fw, y, kind, gen):
    """Split each class into support/query, build the class object from the SUPPORT, score
    every row against every class object. Returns (logits over local class index, query
    mask, local->global label map).

    kind: "sub" / "cone" reproduce the v1 episode BIT FOR BIT, including its two-draw
    support/query split -- the v1 arms have cached features and must stay reproducible.
    "cone2" is the repaired episode: one draw, exact complement, b_opca-matched rays."""
    legacy = kind in ("sub", "cone")
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

    objs = []
    for c, sup, _ in splits:
        if kind == "sub":
            B, _ = torch.linalg.qr(fw[sup].T)                 # (d, N_SUP) orthonormal
            objs.append(("sub", B))
        elif kind == "cone":
            # v1: partition the support into R_EP interleaved groups and average each.
            # Kept verbatim -- see the ce_conic2 note in the module docstring for why this
            # is the wrong ray set and what replaces it.
            g = torch.stack([fw[sup[i::R_EP]].mean(0) for i in range(min(R_EP, len(sup)))])
            objs.append(("cone", Fn.normalize(g, dim=1)))
        else:
            # Foreign material is every row in the episode from another class -- the same
            # legal set the read-out uses at birth (other classes of the current task).
            objs.append(("cone", _gens_opca(fw[sup], fw[y != c],
                                            min(R_EP, len(sup)), gen)))

    is_q = torch.zeros(len(fw), dtype=torch.bool, device=fw.device)
    if legacy:
        for c in gmap:            # v1's SECOND, independent draw -- the ~67% query leak
            idx = (y == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(len(idx), generator=gen, device=fw.device)]
            is_q[perm[N_SUP:]] = True
    else:
        for _, _, qry in splits:
            is_q[qry] = True

    cols = []
    for kindc, O in objs:
        cols.append(torch.linalg.norm(fw @ O, dim=1) if kindc == "sub"
                    else cone_energy(O, fw).clamp(min=1e-8).sqrt())
    return torch.stack(cols, 1) / TAU, is_q, gmap


def snapshot(model):
    import copy
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ------------------------------------------------------------------ training
def train(arm):
    v2 = arm.endswith("2")
    tag = (f"{DS}_T{T}_s{SEED}_{arm}_e{EPOCHS_T0}-{EPOCHS_T}_lr{LR:g}_lam{LAM:g}"
           f"_P{P_CLS}K{K_ROW}s{N_SUP}R{R_EP}_t{TAU:g}_w{WHITEN_EVERY}")
    # v2's episode depends on GAMMA and KM_ITERS; v1's does not. Suffix only the v2 arms so
    # every existing v1 cache still hits.
    tag += f"_g{GAMMA:g}km{KM_ITERS}" if v2 else ""
    cache = os.path.join(REPO, f"exp48_feats_{tag}_{TAG}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        log(f"  cached {tag}")
        return z["Ftr"], z["Fte"], z["Ftr_on"], z["Fte_on"]

    tr_aug, tr_ev, ytr, te_ev, yte, n_cls = F.get_data(DS)
    cpt = n_cls // T
    torch.manual_seed(SEED); np.random.seed(SEED)
    order = np.random.default_rng(SEED).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    model = load_backbone(F.MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))
    d = model.num_features
    ce = nn.CrossEntropyLoss()
    gen = torch.Generator(device=DEV).manual_seed(SEED)

    Sig = torch.eye(d, device=DEV, dtype=torch.float64)      # EMA within-class scatter
    Sacc = torch.zeros(d, d, device=DEV, dtype=torch.float64)  # v2: cumulative, undecayed
    nacc = 0
    Wh = torch.eye(d, device=DEV)                            # its Cholesky whitener
    ON_tr = np.zeros((len(ytr), d), np.float32)
    ON_te = np.zeros((len(yte), d), np.float32)
    step = 0

    for t in range(T):
        idx = np.where(np.isin(ytr, tasks[t]))[0]
        remap = {int(c): i for i, c in enumerate(tasks[t])}
        head = nn.Linear(d, cpt).to(DEV)
        params = lp + list(head.parameters())
        ep = EPOCHS_T0 if t == 0 else EPOCHS_T
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)

        # class-balanced sampler: a per-class scatter estimated from whatever a random
        # batch happens to contain is useless, and the episode needs N_SUP+1 rows/class.
        by_c = {int(c): idx[ytr[idx] == int(c)] for c in tasks[t]}
        by_c = {c: v for c, v in by_c.items() if len(v) >= N_SUP + 1}
        assert by_c, f"no class in task {t} has >= {N_SUP+1} rows"
        n_ep_batches = max(len(idx) // (P_CLS * K_ROW), 1)

        for e in range(ep):
            model.train()
            rng = np.random.default_rng(1000 * t + e)
            for _ in range(n_ep_batches):
                cs = rng.choice(list(by_c), min(P_CLS, len(by_c)), replace=False)
                rows = np.concatenate([
                    by_c[c][rng.choice(len(by_c[c]), min(K_ROW, len(by_c[c])),
                                       replace=len(by_c[c]) < K_ROW)] for c in cs])
                xb = torch.stack([tr_aug[int(i)][0] for i in rows]).to(DEV)
                yb = torch.tensor([int(ytr[i]) for i in rows], device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = model(xb).float()

                # EMA of the pooled within-class scatter, and its whitener. Detached: this
                # is the read-out's geometry, not a thing the loss should game.
                with torch.no_grad():
                    S = torch.zeros(d, d, device=DEV, dtype=torch.float64)
                    for c in yb.unique():
                        z = f[yb == c].double()
                        z = z - z.mean(0, keepdim=True)
                        S += z.T @ z
                    if v2:
                        # The read-out pools scatter over every class seen so far and never
                        # decays it (see evaluate: `scatter += Xc.T @ Xc`). Match that.
                        Sacc += S
                        nacc += max(len(f), 1)
                        Sig = Sacc / nacc
                    else:
                        S = S / max(len(f), 1)
                        Sig = EMA * Sig + (1 - EMA) * S
                    if step % WHITEN_EVERY == 0:
                        Sr = Sig + SHRINK * torch.trace(Sig) / d * torch.eye(
                            d, device=DEV, dtype=torch.float64)
                        Wh = torch.linalg.cholesky(
                            torch.linalg.inv(Sr)).float().contiguous()
                step += 1

                loss = torch.zeros((), device=DEV)
                if arm in ("ce", "ce_conic", "ce_conic2"):
                    loss = loss + ce(head(f), torch.tensor(
                        [remap[int(v)] for v in yb.tolist()], device=DEV))
                if arm in ("conic", "sub", "ce_conic", "ce_conic2"):
                    fw = Fn.normalize(f @ Wh, dim=1)
                    kind = "sub" if arm == "sub" else ("cone2" if v2 else "cone")
                    lg, is_q, gmap = episode_logits(fw, yb, kind, gen)
                    if lg is not None and bool(is_q.any()):
                        loc = {c: i for i, c in enumerate(gmap)}
                        tgt = torch.tensor([loc[int(v)] for v in yb.tolist()], device=DEV)
                        lam = LAM if arm in ("ce_conic", "ce_conic2") else 1.0
                        loss = loss + lam * ce(lg[is_q], tgt[is_q])

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            sch.step()

        rows = np.where(np.isin(ytr, tasks[t]))[0]
        cols = np.where(np.isin(yte, tasks[t]))[0]
        ON_tr[rows] = F.extract(model, tr_ev, rows)
        ON_te[cols] = F.extract(model, te_ev, cols)
        log(f"  [{arm}] task {t} done")

    Ftr, Fte = F.extract(model, tr_ev), F.extract(model, te_ev)
    del model
    torch.cuda.empty_cache()
    np.savez(cache, Ftr=Ftr, Fte=Fte, Ftr_on=ON_tr, Fte_on=ON_te)
    log(f"  trained {tag}")
    return Ftr, Fte, ON_tr, ON_te


# ------------------------------------------------------------------ evaluation
def evaluate(Ftr, Fte, ytr, yte, n_cls):
    """exp49's staged replay (oPCA g=0.5, self-consistent negatives) plus RanPAC ON THE
    SAME FEATURES, plus the exp45 diagnostics at the final state."""
    Ztr, Zte = un(Ftr), un(Fte)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(SEED).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT = []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        FIT.append(ix[pm[max(int(0.1 * len(ix)), 1):]])

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A, accs = {}, []
    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        S_ = scatter / max(n_scat, 1)
        S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)
        rng = np.random.default_rng(1234 + t)
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
            past = [A[o] for o in A if o not in tasks[t]]
            Fr = np.concatenate([Ztr[oth]] + past, 0)
            if len(Fr) > F_MAX:
                Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
            A[c] = X.b_opca(un(Ztr[r] @ Wh), un(Fr @ Wh), rays_for(len(r)),
                            int(c), GAMMA) @ Wh_inv
        seen = np.concatenate(tasks[:t + 1])
        tei = np.where(np.isin(yte, seen))[0]
        Qw = un(Zte[tei] @ Wh)
        St = np.full((len(tei), n_cls), -np.inf, np.float32)
        for c in seen:
            if c in A:
                St[:, c] = X.cone_score(un(A[c] @ Wh), Qw)
        accs.append(float((np.asarray(seen)[St[:, seen].argmax(1)] == yte[tei]).mean()))

    seen = np.asarray(sorted(A))
    Qw = un(Zte @ Wh)
    nm, er = [], []
    Sc = np.full((len(yte), n_cls), -np.inf, np.float32)
    Ss = np.full((len(yte), n_cls), -np.inf, np.float32)
    for c in seen:
        Ac = un(A[c] @ Wh)
        Sc[:, c] = X.cone_score(Ac, Qw)
        U, s, _ = np.linalg.svd(Ac.T, full_matrices=False)
        B = U[:, s > max(s[0], 1e-12) * 1e-6]
        Ss[:, c] = np.linalg.norm(Qw @ B, axis=1)
        Wls = np.linalg.lstsq(Ac.T, Qw.T, rcond=None)[0].T
        nm.append(np.clip(-Wls, 0, None).sum(1) / (np.abs(Wls).sum(1) + 1e-12))
        rr = np.concatenate([f[ytr[f] == c] for f in FIT])
        Xw = un(Ztr[rr] @ Wh)
        ev = np.linalg.eigvalsh((Xw.T @ Xw) / len(Xw))
        er.append(float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12)))
    GAP = np.clip(Ss[:, seen] - Sc[:, seen], 0, None)
    col = {int(c): j for j, c in enumerate(seen)}
    # A class with fewer than 2 fit rows never gets a cone, so it is absent from `seen`.
    # IMAGENETA has such classes (~30 rows/class before the 10% val split, and the split is
    # not stratified), which is why every IMAGENETA cell died here with KeyError. The staged
    # accuracy above already tolerates this -- those columns stay -inf -- but the
    # diagnostics index BY class, so score them on the test rows that have a cone instead
    # of crashing on the first row that does not.
    keep = np.array([int(y) in col for y in yte])
    rows = np.arange(len(yte))[keep]
    tcol = np.array([col[int(y)] for y in yte[keep]])
    NM = np.stack(nm, 1)
    rng = np.random.default_rng(0)
    fc = rng.integers(0, len(seen), size=(len(rows), 8))
    fix = fc == tcol[:, None]
    fc[fix] = (fc[fix] + 1) % len(seen)
    ranpac = F.replay(Ftr, ytr, Fte, yte, T, SEED, n_cls)
    t0 = set(int(c) for c in tasks[0])
    return {
        "cone_A_last": accs[-1], "cone_A_avg": float(np.mean(accs)), "cone_accs": accs,
        "ranpac_A_last": ranpac[-1], "ranpac_A_avg": float(np.mean(ranpac)),
        "sub_A_last": float((seen[Ss[:, seen].argmax(1)] == yte).mean()),
        "gap_own": float(GAP[rows, tcol].mean()),
        "gap_foreign": float(GAP[rows[:, None], fc].mean()),
        "gap_differential": float(GAP[rows[:, None], fc].mean() - GAP[rows, tcol].mean()),
        "negmass_own": float(NM[rows, tcol].mean()),
        "erank_mean": float(np.mean(er)),
        "erank_t0": float(np.mean([er[j] for j, c in enumerate(seen) if int(c) in t0])),
        "erank_rest": float(np.mean([er[j] for j, c in enumerate(seen) if int(c) not in t0])),
        "mean_rays": float(np.mean([len(A[c]) for c in seen])),
    }


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    _, _, ytr, _, yte, n_cls = F.get_data(DS)
    bar = F.bar_for(DS, T, SEED)
    for arm in ARMS:
        Ftr, Fte, ON_tr, ON_te = train(arm)
        for proto in PROTOCOLS:
            key = (f"{DS}|{T}|{SEED}|{arm}|{proto}|e{EPOCHS_T0}-{EPOCHS_T}|lam{LAM:g}"
                   f"|P{P_CLS}K{K_ROW}s{N_SUP}R{R_EP}_t{TAU:g}|g{GAMMA:g}_k{KVAL:g}m{RMIN}|v2")
            if key in allres:
                log(f"skip {key}"); continue
            log(f"=== {key}")
            # `drift` is the honest pessimistic bracket: stale rays, current queries.
            # `online` also staleness-freezes the QUERIES per class, which needs the test
            # label to pick the extractor -- see the docstring. It stays reachable only
            # because a value for it is already in the JSON.
            a, b = {"final": (Ftr, Fte),
                    "drift": (ON_tr, Fte),
                    "online": (ON_tr, ON_te)}[proto]
            allres[key] = evaluate(a, b, ytr, yte, n_cls)
            json.dump(allres, open(OUT, "w"), indent=2)

    W = 96
    print("\n" + "=" * W)
    print(f"EXP48 — training phi FOR the conic reader ({DS}, T={T}, seed {SEED})")
    print("=" * W)
    if bar:
        print(f"\nFSA bar (exp16 A_plus + RanPAC): {bar['A_last']*100:.2f} / "
              f"{bar['A_avg']*100:.2f}")
    print(f"\n  {'arm|proto':<20}{'cone':>8}{'ranpac':>9}{'cone-rp':>9}{'sub':>8}"
          f"{'negm':>8}{'gapdiff':>10}{'erank':>8}{'er_t0':>8}{'er_rest':>9}")
    for key, r in sorted(allres.items()):
        p = key.split("|")
        print(f"  {p[3]+'|'+p[4]:<20}{r['cone_A_last']*100:>8.2f}"
              f"{r['ranpac_A_last']*100:>9.2f}"
              f"{(r['cone_A_last']-r['ranpac_A_last'])*100:>+9.2f}"
              f"{r['sub_A_last']*100:>8.2f}{r['negmass_own']*100:>7.1f}%"
              f"{r['gap_differential']:>+10.4f}{r['erank_mean']:>8.1f}"
              f"{r['erank_t0']:>8.1f}{r['erank_rest']:>9.1f}")
    print("\n" + "-" * W)
    print("`cone-rp` IS THE HEADLINE AND NOTHING ELSE IS. Both columns come from the SAME")
    print("   features, so it is the only comparison the recipe cannot inflate. Comparing")
    print("   a trained arm against the FSA bar is the mistake this project has made three")
    print("   times; the bar line above is context, not a comparator.")
    print("MECHANISM, cheaper than the headline: negm should fall from 43.7%, gapdiff should")
    print("   turn POSITIVE from -0.0101, erank should DROP. Accuracy moving while these do")
    print("   not means the stated mechanism is not what happened.")
    print("`conic` vs `sub` is the training-side test of non-negativity, mirroring exp38's")
    print("   read-out-side cone vs sub (-0.10 at k-means atoms, +0.60 at oPCA atoms).")
    print("TRANSFER: er_t0 vs er_rest. The loss sees cpt classes and serves n_cls; if rank")
    print("   drops only for task-0 classes it overfitted the first session.")
    print("=" * W)
    print(f"wrote {OUT}")
