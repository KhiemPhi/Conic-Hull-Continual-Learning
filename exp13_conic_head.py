#!/usr/bin/env python3
"""exp13_conic_head.py — exhaustive conic-hull head search on the A_plus lr3e-4 feature space.

THE BAR
    A_plus (augreg_in21k, ep40, lr3e-4, AUG=1, SEED=1) + RanPAC M_RP=10000, accum protocol:

        A-Last 0.8090    A-Avg 0.8609        <- every variant here must beat this

WHAT THE GATE ALREADY RULED OUT
    Measured on these exact features, training on ALL 24000 train rows at once (no
    incremental constraint at all):

        NCM (768-d)              0.7262
        ridge 768-d, ALL data    0.7640
        MLP head, ALL data       0.7975
        linear probe, ALL data   0.8055
        RanPAC, CIL protocol     0.8090   <- BEATS every offline head on raw 768-d

    So the best achievable head on the raw 768-d space is ~0.806, BELOW the bar. Any cone
    that classifies in 768-d is bounded by that and cannot win. Group A below is included
    only to confirm that bound empirically and to complete the sweep -- do not expect a win
    there. The live hypotheses are Groups B/C/D.

WHY A CONE MIGHT WIN HERE WHEN EVERY PRIOR CONIC APPLICATION FAILED
    RanPAC's feature map is h = ReLU(f P), so h lives in the NON-NEGATIVE ORTHANT of
    R^M_RP. A conic hull is the natural geometric object of that space, and a conic
    (non-negative) readout is its natural classifier. Every previous conic attempt in this
    repo operated either in weight space (covariance cone -- dead weight, removed) or in the
    signed 768-d feature space (ties NCM). exp1 tested "does a cone need a natively
    non-negative space?" by MAKING one with ReLU and got a worse result -- but that built a
    new space; here the non-negative space is already present and already beating every
    alternative. Cones inside RanPAC's orthant is the one conic angle never tested.

FAIRNESS RULES
    * Every hyperparameter (ridge lambda, fusion beta, rerank k) is selected on the SAME
      held-out 10% validation carve-out RanPAC uses. Never on test. A cone variant that
      picks beta on test would beat the bar for free.
    * Cones are fitted ONLY on the 90% train split used to build G/C -- never on the val
      rows, or the selection above is contaminated.
    * Classes are fitted at their BIRTH stage and never refitted. Valid because A_plus
      freezes the backbone after task 0, so features never drift. This is also what makes
      the sweep cheap: a query's score under class c's cone is constant across stages, so
      the full (n_query x 200) score matrix is computed ONCE and sliced per stage.

VARIANTS
  A  cone as classifier in 768-d           (reference; bounded at ~0.806, expect a loss)
       a_score a_nnlsres a_residcov a_angular a_combined a_klocal a_multi
  B  cone as classifier in RanPAC h-space  (the novel angle)
       b_score b_nnlsres
  C  conic (non-negative) READOUT          (W >= 0 instead of unconstrained ridge)
       c_nnls_h c_nnls_768
  D  cone FUSED with RanPAC                (most likely to actually lift)
       d_add  d_concat  d_rerank

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u exp13_conic_head.py                      # everything (~45-75 min)
    VARIANTS=d_add,d_rerank,b_score python -u exp13_conic_head.py
    N_RAYS=24 M_CONE=2000 python -u exp13_conic_head.py
"""
import json
import os
import time

import numpy as np
import torch

from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


DEV = "cuda"
SEED = int(os.environ.get("SEED", 1))
N_TASKS, CPT, N_CLS = 10, 20, 200
M_RP = int(os.environ.get("MRP", 10000))
M_CONE = int(os.environ.get("M_CONE", 2000))   # h-dims used for Group B (cost control)
N_RAYS = int(os.environ.get("N_RAYS", 24))
N_SUB = int(os.environ.get("N_SUB", 4))        # sub-cones per class for a_multi
K_LOCAL = int(os.environ.get("K_LOCAL", 8))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
TOPKS = [2, 3, 5, 10]
FEATS = os.environ.get("FEATS", "exp12_feats_augreg_in21k_ep40_lr0.0003_aug1_s1.npz")
BAR_LAST, BAR_AVG = 0.8090, 0.8609

ALL_VARIANTS = ["a_score", "a_nnlsres", "a_residcov", "a_angular", "a_combined",
                "a_klocal", "a_multi", "b_score", "b_nnlsres",
                "c_nnls_h", "c_nnls_768", "d_add", "d_concat", "d_rerank"]
WANT = [v for v in os.environ.get("VARIANTS", ",".join(ALL_VARIANTS)).split(",")
        if v in ALL_VARIANTS]


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ data
def load():
    z = np.load(FEATS)
    from datasets import load_dataset
    d = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
    w = np.array(d["wnid"])
    lab = np.searchsorted(np.array(sorted(set(w.tolist()))), w)
    p = np.random.default_rng(1993).permutation(len(lab))
    n = int(0.8 * len(lab))
    return (un(z["Ftr"]).astype(np.float32), lab[p[:n]],
            un(z["Fte"]).astype(np.float32), lab[p[n:]])


Ztr, ytr, Zte, yte = load()
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]
log(f"A_plus feats: train {Ztr.shape} test {Zte.shape} | seed {SEED} | M_RP {M_RP}")

# Reproduce exp12's per-stage 90/10 split EXACTLY (rng(t), nv = 10%), so the val rows used
# for hyperparameter selection are the same rows RanPAC used and the comparison is paired.
FIT_IDX, VAL_IDX = [], []
for t in range(N_TASKS):
    idx = np.where(np.isin(ytr, TASKS[t]))[0]
    pm = np.random.default_rng(t).permutation(len(idx))
    nv = max(int(0.1 * len(idx)), 1)
    VAL_IDX.append(idx[pm[:nv]])
    FIT_IDX.append(idx[pm[nv:]])
VAL_ALL = np.concatenate(VAL_IDX)
log(f"fit rows {sum(len(f) for f in FIT_IDX)}  val rows {len(VAL_ALL)}")

# ------------------------------------------------------------------ RanPAC space
P_RP = torch.randn(Ztr.shape[1], M_RP,
                   generator=torch.Generator().manual_seed(0)).to(DEV)


def H(Z, bs=4096):
    out = []
    for i in range(0, len(Z), bs):
        x = torch.tensor(Z[i:i + bs], device=DEV, dtype=torch.float32)
        out.append(torch.relu(x @ P_RP))
    return torch.cat(out)


Htr, Hte = H(Ztr), H(Zte)
Hval = Htr[torch.as_tensor(VAL_ALL, device=DEV)]
log(f"RanPAC h-space built: {tuple(Htr.shape)} (non-negative orthant)")


# ------------------------------------------------------------------ cone fitting
MIN_N = 8          # ConicHull's hybrid path runs NearestNeighbors(n_neighbors=6) internally


def _fit_one(X, n_rays=N_RAYS, k_local=None):
    """Fit ONE ConicHull with per-class-safe settings.

    build_class_conic_hulls applies a single global pca_dim to every class, which crashes on
    ImageNet-R: class sizes range from ~28 to ~300 fit rows and PCA(64) needs n_samples>=64.
    Same object, same fit, just clamped per class:
        pca_dim <= min(n, d)      (sklearn PCA with svd_solver='full')
        n_rays  <= n-2            (SPA cannot select more directions than points)
    """
    n, d = X.shape
    pdim = int(min(64, n, d))
    return ConicHull(n_rays=int(np.clip(n_rays, 2, max(n - 2, 2))),
                     use_pca=pdim < d, pca_dim=pdim, k_local=k_local,
                     ray_diversity="hybrid").fit(X)


def fit_cones(feat_getter, tag, n_rays=N_RAYS, k_local=None):
    """One hull per class, fitted on that class's FIT rows only (never val)."""
    hulls, skipped, sizes = {}, 0, []
    for t in range(N_TASKS):
        for c in TASKS[t]:
            rows = FIT_IDX[t][ytr[FIT_IDX[t]] == c]
            if len(rows) < MIN_N:
                skipped += 1
                continue
            sizes.append(len(rows))
            hulls[str(int(c))] = _fit_one(feat_getter(rows), n_rays, k_local)
    log(f"  fitted {len(hulls)} hulls [{tag}] n_rays={n_rays} k_local={k_local} "
        f"| class rows min {min(sizes)} med {int(np.median(sizes))} max {max(sizes)}"
        + (f" | SKIPPED {skipped} classes with <{MIN_N} rows" if skipped else ""))
    return hulls


def score_matrix(hulls, queries, method):
    """(n_query, N_CLS) score matrix; unseen/unfitted classes get -inf."""
    S = np.full((len(queries), N_CLS), -np.inf, dtype=np.float32)
    for name, hull in hulls.items():
        S[:, int(name)] = getattr(hull, method)(queries)
    return S


def multi_score_matrix(feat_getter, queries):
    """a_multi: N_SUB k-means sub-cones per class, score = max over sub-cones."""
    from sklearn.cluster import KMeans
    S = np.full((len(queries), N_CLS), -np.inf, dtype=np.float32)
    for t in range(N_TASKS):
        for c in TASKS[t]:
            rows = FIT_IDX[t][ytr[FIT_IDX[t]] == c]
            X = feat_getter(rows)
            if len(X) < MIN_N:
                continue
            k = min(N_SUB, max(1, len(X) // MIN_N))
            lab = (KMeans(n_clusters=k, n_init=4, random_state=int(c)).fit_predict(X)
                   if k > 1 else np.zeros(len(X), int))
            subs = [X[lab == j] for j in range(k) if (lab == j).sum() >= MIN_N] or [X]
            sc = np.stack([_fit_one(Xs).score(queries) for Xs in subs])
            S[:, int(c)] = sc.max(0)
    return S


# ------------------------------------------------------------------ heads
def ridge_W(G, C, lam):
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=DEV,
                                                  dtype=torch.float64), C)


def nnls_W(G, C, lam, iters=300):
    """Conic (non-negative) readout: min_W>=0  tr(W'GW) - 2tr(W'C) + lam||W||^2.

    This is the conic constraint applied to the CLASSIFIER rather than to the features:
    each class score becomes a NON-NEGATIVE combination of ReLU random features, i.e. the
    score vector is required to live in the dual cone of the feature orthant. Projected
    gradient with the Lipschitz step 1/||G+lam I||_2; W is tiny (M x 200) so this is cheap.
    """
    Gf = (G + lam * torch.eye(G.shape[0], device=DEV, dtype=torch.float64)).float()
    Cf = C.float()
    L = torch.linalg.matrix_norm(Gf, ord=2)
    step = 1.0 / float(L)
    W = torch.clamp(torch.linalg.solve(Gf.double(), C).float(), min=0)
    Y, tk = W.clone(), 1.0
    for _ in range(iters):                                   # FISTA
        Wn = torch.clamp(Y - step * (Gf @ Y - Cf), min=0)
        tn = 0.5 * (1 + (1 + 4 * tk * tk) ** 0.5)
        Y = Wn + ((tk - 1) / tn) * (Wn - W)
        W, tk = Wn, tn
    return W.double()


def acc(logits, y, seen_t):
    """Argmax restricted to seen classes, as in exp8's solve_eval."""
    pred = seen_t[logits[:, seen_t].argmax(1)].cpu().numpy()
    return float((pred == y).mean())


def zs(A, seen):
    """Row-wise z-score over the SEEN columns only, so RanPAC logits and cone scores are on
    a common scale before fusion.

    Must be restricted to `seen`: score_matrix() fills unfitted/unseen classes with -inf, and
    z-scoring across the full 200 columns would make every row nan and silently destroy the
    fused logit.
    """
    # A large finite floor, NOT -inf: BETAS contains 0.0 and 0.0 * -inf = nan, which would
    # poison the unseen columns. Harmless today (argmax is taken over `seen` only) but it
    # masks real nans behind a RuntimeWarning.
    B = np.full_like(A, -1e9, dtype=np.float64)
    sub = A[:, seen].astype(np.float64)
    sub = np.nan_to_num(sub, neginf=np.nanmin(sub[np.isfinite(sub)]) if
                        np.isfinite(sub).any() else 0.0)
    m, s = sub.mean(1, keepdims=True), sub.std(1, keepdims=True) + 1e-8
    B[:, seen] = (sub - m) / s
    return B


# ------------------------------------------------------------------ build score matrices
log("=== building cone score matrices (once; valid at every stage because phi is frozen)")
CONES, SCORES = {}, {}


def get768(rows):
    return Ztr[rows]


def getH(rows):
    return Htr[torch.as_tensor(rows, device=DEV), :M_CONE].cpu().numpy()


need_a = any(v.startswith("a_") for v in WANT) or any(v.startswith("d_") for v in WANT)
if need_a:
    h768 = fit_cones(get768, "768-d")
    Zte_q, Zval_q = Zte, Ztr[VAL_ALL]
    for meth, key in [("score", "a_score"), ("score_nnls_residual", "a_nnlsres"),
                      ("score_residual_coverage", "a_residcov"),
                      ("score_angular", "a_angular"), ("score_combined", "a_combined")]:
        if key in WANT or (key == "a_score" and any(v.startswith("d_") for v in WANT)):
            SCORES[key] = (score_matrix(h768, Zval_q, meth),
                           score_matrix(h768, Zte_q, meth))
            log(f"  scored {key}")
if "a_klocal" in WANT:
    hk = fit_cones(get768, "768-d k_local", k_local=K_LOCAL)
    SCORES["a_klocal"] = (score_matrix(hk, Ztr[VAL_ALL], "score"),
                          score_matrix(hk, Zte, "score"))
    log("  scored a_klocal")
if "a_multi" in WANT:
    SCORES["a_multi"] = (multi_score_matrix(get768, Ztr[VAL_ALL]),
                         multi_score_matrix(get768, Zte))
    log("  scored a_multi")
if any(v in WANT for v in ("b_score", "b_nnlsres")):
    hH = fit_cones(getH, f"h-space {M_CONE}-d orthant")
    HteC = Hte[:, :M_CONE].cpu().numpy()
    HvalC = Hval[:, :M_CONE].cpu().numpy()
    for meth, key in [("score", "b_score"), ("score_nnls_residual", "b_nnlsres")]:
        if key in WANT:
            SCORES[key] = (score_matrix(hH, HvalC, meth), score_matrix(hH, HteC, meth))
            log(f"  scored {key}")

# ------------------------------------------------------------------ stage replay
log("=== stage replay")
Gacc = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
Cacc = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
G7 = torch.zeros(768, 768, device=DEV, dtype=torch.float64)
C7 = torch.zeros(768, N_CLS, device=DEV, dtype=torch.float64)
res = {v: [] for v in WANT}
res["ranpac"] = []
nval_seen = 0

for t in range(N_TASKS):
    fi = FIT_IDX[t]
    hf = Htr[torch.as_tensor(fi, device=DEV)].double()
    Y = torch.zeros(len(fi), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(fi)), torch.as_tensor(ytr[fi], device=DEV)] = 1.0
    Gacc += hf.T @ hf
    Cacc += hf.T @ Y
    zf = torch.tensor(Ztr[fi], device=DEV, dtype=torch.float64)
    G7 += zf.T @ zf
    C7 += zf.T @ Y
    del hf

    seen = np.concatenate(TASKS[:t + 1])
    seen_t = torch.as_tensor(seen, device=DEV)
    te = np.where(np.isin(yte, seen))[0]
    nval_seen += len(VAL_IDX[t])
    vsl = slice(0, nval_seen)                     # VAL_ALL is task-ordered
    yv = ytr[VAL_ALL[vsl]]
    yt = yte[te]
    te_t = torch.as_tensor(te, device=DEV)

    # ---- RanPAC reference (lambda on val) -------------------------------------
    bestW, bestv = None, -1
    for lam in LAMBDAS:
        W = ridge_W(Gacc, Cacc, lam)
        a = acc(Hval[vsl].double() @ W, yv, seen_t)
        if a > bestv:
            bestv, bestW = a, W
    Lte = (Hte[te_t].double() @ bestW)
    Lval = (Hval[vsl].double() @ bestW)
    res["ranpac"].append(acc(Lte, yt, seen_t))

    Lte_np, Lval_np = Lte.cpu().numpy(), Lval.cpu().numpy()

    for v in WANT:
        if v in SCORES:                            # A/B: cone score IS the logit
            Sv, St = SCORES[v]
            res[v].append(acc(torch.tensor(St[te], device=DEV), yt, seen_t))
        elif v == "c_nnls_h":
            bw, bv = None, -1
            for lam in LAMBDAS:
                W = nnls_W(Gacc, Cacc, lam)
                a = acc(Hval[vsl].double() @ W, yv, seen_t)
                if a > bv:
                    bv, bw = a, W
            res[v].append(acc(Hte[te_t].double() @ bw, yt, seen_t))
        elif v == "c_nnls_768":
            bw, bv = None, -1
            for lam in [1e-2, 1e-1, 1, 10]:
                W = nnls_W(G7, C7, lam)
                zv = torch.tensor(Ztr[VAL_ALL[vsl]], device=DEV, dtype=torch.float64)
                a = acc(zv @ W, yv, seen_t)
                if a > bv:
                    bv, bw = a, W
            zt = torch.tensor(Zte[te], device=DEV, dtype=torch.float64)
            res[v].append(acc(zt @ bw, yt, seen_t))
        elif v == "d_add":                         # fuse: ranpac + beta * cone, beta on val
            Sv, St = SCORES["a_score"]
            zLv, zSv = zs(Lval_np, seen), zs(Sv[vsl], seen)
            zLt, zSt = zs(Lte_np, seen), zs(St[te], seen)
            bb, bv = 0.0, -1
            for b in BETAS:
                a = acc(torch.tensor(zLv + b * zSv, device=DEV), yv, seen_t)
                if a > bv:
                    bv, bb = a, b
            res[v].append(acc(torch.tensor(zLt + bb * zSt, device=DEV), yt, seen_t))
        elif v == "d_concat":                      # ridge on [h , cone scores]
            Sv, St = SCORES["a_score"]
            Sf = np.nan_to_num(score_matrix(h768, Ztr[fi], "score"), neginf=0.0)
            aug = torch.tensor(Sf, device=DEV, dtype=torch.float64)
            hf2 = torch.cat([Htr[torch.as_tensor(fi, device=DEV)].double(), aug], 1)
            if t == 0:
                Ga = torch.zeros(M_RP + N_CLS, M_RP + N_CLS, device=DEV, dtype=torch.float64)
                Ca = torch.zeros(M_RP + N_CLS, N_CLS, device=DEV, dtype=torch.float64)
                globals()["_Ga"], globals()["_Ca"] = Ga, Ca
            _Ga.add_(hf2.T @ hf2); _Ca.add_(hf2.T @ Y)
            bw, bv = None, -1
            hv = torch.cat([Hval[vsl].double(),
                            torch.tensor(np.nan_to_num(Sv[vsl], neginf=0.0),
                                         device=DEV, dtype=torch.float64)], 1)
            for lam in LAMBDAS:
                W = ridge_W(_Ga, _Ca, lam)
                a = acc(hv @ W, yv, seen_t)
                if a > bv:
                    bv, bw = a, W
            ht = torch.cat([Hte[te_t].double(),
                            torch.tensor(np.nan_to_num(St[te], neginf=0.0),
                                         device=DEV, dtype=torch.float64)], 1)
            res[v].append(acc(ht @ bw, yt, seen_t))
        elif v == "d_rerank":                      # cone reranks RanPAC's top-k
            Sv, St = SCORES["a_score"]

            def rr(L, S, k, y):
                top = np.argsort(-L, 1)[:, :k]
                pick = np.take_along_axis(S, top, 1).argmax(1)
                return float((np.take_along_axis(top, pick[:, None], 1)[:, 0] == y).mean())
            Lv_m = np.full((len(yv), N_CLS), -np.inf); Lv_m[:, seen] = Lval_np[:, seen]
            Lt_m = np.full((len(yt), N_CLS), -np.inf); Lt_m[:, seen] = Lte_np[:, seen]
            bk, bv = 1, acc(torch.tensor(Lval_np, device=DEV), yv, seen_t)
            for k in TOPKS:
                a = rr(Lv_m, Sv[vsl], k, yv)
                if a > bv:
                    bv, bk = a, k
            res[v].append(res["ranpac"][-1] if bk == 1 else rr(Lt_m, St[te], bk, yt))
    log(f"  t={t} seen={len(seen):3d} ranpac {res['ranpac'][-1]:.4f} | " +
        " ".join(f"{v} {res[v][-1]:.4f}" for v in WANT))

# ------------------------------------------------------------------ report
print("\n" + "=" * 92)
print(f"EXP13 — conic-hull heads on A_plus lr3e-4 features (ImageNet-R, seed {SEED})")
print(f"BAR: RanPAC A-Last {BAR_LAST:.4f}  A-Avg {BAR_AVG:.4f}     "
      f"| offline ceiling on raw 768-d ~0.8055")
print("=" * 92)
print(f"{'variant':<14}{'A-Last':>9}{'A-Avg':>9}{'dLast':>9}{'dAvg':>9}   group")
GRP = {"a": "A 768-d classifier (bounded ~0.806)", "b": "B h-space orthant cone",
       "c": "C conic non-negative readout", "d": "D fused with RanPAC", "r": "reference"}
rows = [("ranpac", res["ranpac"])] + [(v, res[v]) for v in WANT]
for v, a in sorted(rows, key=lambda x: -x[1][-1]):
    star = "  <== BEATS BAR" if (a[-1] > BAR_LAST or float(np.mean(a)) > BAR_AVG) else ""
    print(f"{v:<14}{a[-1]:>9.4f}{float(np.mean(a)):>9.4f}{a[-1]-BAR_LAST:>+9.4f}"
          f"{float(np.mean(a))-BAR_AVG:>+9.4f}   {GRP[v[0] if v[0] in GRP else 'r']}{star}")
print("=" * 92)
json.dump({v: a for v, a in rows}, open(f"exp13_conic_head_s{SEED}.json", "w"), indent=2)
print(f"wrote exp13_conic_head_s{SEED}.json")
