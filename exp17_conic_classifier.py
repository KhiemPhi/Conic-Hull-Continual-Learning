#!/usr/bin/env python3
"""exp17_conic_classifier.py — conic hulls as the CLASSIFIER, on any exp16 A_plus feature set.

WHAT THIS ANSWERS
    "Can a conic-hull classifier built on the saved A_plus features beat A-Last?"
    exp13 wrote these variants but only the fusion arm was ever executed; exp14 measured the
    hull as a RERANKER (-7.7 pts on a two-element candidate set) and the offline ceiling on
    raw 768-d ImageNet-R features (0.8055, BELOW the 0.8090 bar). This runs the hull as a
    standalone classifier, with real numbers, on any of the 36 exp16 feature caches.

WHY CIFAR-100 IS THE INTERESTING CELL, NOT IMAGENET-R
    Every prior hull test here ran on ImageNet-R, where a class has only ~108 fit rows in
    768 dimensions -- SPA is choosing ~24 extreme rays from a badly under-sampled class, so
    "the hull is a poor estimate" and "the hull is the wrong object" are confounded.
    CIFAR-100 has ~450 fit rows/class and A_plus is already SOTA there (92.55 vs GR-LoRA
    91.97). ImageNet-A is the opposite extreme (~24 rows/class) and should be the worst case.
    Running all three separates sample-starvation from model-misfit.

PROTOCOL
    Identical CIL replay to exp16: same class order rng(SEED).permutation, same per-task 10%
    val carve-out, RanPAC recomputed here as the in-run bar so every number is paired.
    Hulls are fitted ONLY on the 90% fit rows and only at their class's birth stage -- valid
    forever because A_plus freezes the backbone, which is also why the (n_query x n_cls)
    score matrices are computed once and sliced per stage.
    Any fusion weight is chosen on the val carve-out, never on test.

VARIANTS
    ncm              nearest class mean            (floor)
    ranpac           the bar, recomputed in-run    (should match exp16's cell)
    cone_score       ConicHull.score               (cosine to NNLS reconstruction)
    cone_nnlsres     score_nnls_residual
    cone_residcov    score_residual_coverage
    cone_angular     score_angular
    cone_combined    score_combined
    cone_klocal      score with k_local=K_LOCAL    (shrink-wrapped local hull)
    cone_multi       N_SUB k-means sub-cones/class, max over sub-cones
    cone_fuse        zs(ranpac) + beta*zs(best cone), beta on val (beta=0 in grid, so this
                     cannot lose on val -- it is the "does the cone add ANYTHING" arm)

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=CIFAR100 T=10 SEED=0 python -u exp17_conic_classifier.py
    DS=CIFAR100,IMAGENETR,IMAGENETA T=10 SEED=0 python -u exp17_conic_classifier.py
    DS=CIFAR100 T=10 SEED=0 N_RAYS=48 python -u exp17_conic_classifier.py
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
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
DSETS = os.environ.get("DS", "CIFAR100").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]
N_RAYS = int(os.environ.get("N_RAYS", 24))
K_LOCAL = int(os.environ.get("K_LOCAL", 8))
N_SUB = int(os.environ.get("N_SUB", 4))
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_N = 8            # ConicHull's hybrid path runs NearestNeighbors(n_neighbors=6) inside
SPLIT_SEED = 1993
REPO = os.path.dirname(os.path.abspath(__file__))
BARS = {}
_bp = os.path.join(REPO, f"exp16_full_table_{MODEL.split('.')[-1]}.json")
if os.path.exists(_bp):
    BARS = json.load(open(_bp))

VARIANTS = ["ncm", "ranpac", "cone_score", "cone_nnlsres", "cone_residcov", "cone_angular",
            "cone_combined", "cone_klocal", "cone_multi", "cone_fuse"]
WANT = [v for v in os.environ.get("VARIANTS", ",".join(VARIANTS)).split(",") if v in VARIANTS]
SCORE_METHOD = {"cone_score": "score", "cone_nnlsres": "score_nnls_residual",
                "cone_residcov": "score_residual_coverage", "cone_angular": "score_angular",
                "cone_combined": "score_combined"}


def un(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ labels (no images)
def get_labels(name):
    from datasets import load_dataset
    if name == "CIFAR100":
        from torchvision import datasets as tvd
        tr = tvd.CIFAR100(os.path.join(REPO, "data"), train=True, download=False)
        te = tvd.CIFAR100(os.path.join(REPO, "data"), train=False, download=False)
        return np.array(tr.targets), np.array(te.targets), 100
    if name == "CUB200":
        dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
        ytr, yte = np.array(dd["train"]["label"]), np.array(dd["test"]["label"])
        return ytr, yte, int(max(ytr.max(), yte.max())) + 1
    if name == "IMAGENETR":
        d = load_dataset("axiong/imagenet-r", cache_dir=os.path.join(REPO, "data/hf"))["test"]
        w = np.array(d["wnid"])
        lab = np.searchsorted(np.array(sorted(set(w.tolist()))), w)
    elif name == "IMAGENETA":
        d = load_dataset("barkermrl/imagenet-a",
                         cache_dir=os.path.join(REPO, "data/hf"))["train"]
        lab = np.array(d["label"])
    else:
        raise ValueError(name)
    p = np.random.default_rng(SPLIT_SEED).permutation(len(lab))
    n = int(0.8 * len(lab))
    return lab[p[:n]], lab[p[n:]], int(lab.max()) + 1


# ------------------------------------------------------------------ hulls (YOUR ConicHull)
def fit_one(X, n_rays=N_RAYS, k_local=None):
    """pca_dim/n_rays clamped per class: sklearn PCA needs n_components <= n_samples and
    class sizes here range ~24 (ImageNet-A) to ~450 (CIFAR-100)."""
    n, d = X.shape
    pdim = int(min(64, n, d))
    return ConicHull(n_rays=int(np.clip(n_rays, 2, max(n - 2, 2))), use_pca=pdim < d,
                     pca_dim=pdim, k_local=k_local, ray_diversity="hybrid").fit(X)


def score_mat(hulls, Q, method, n_cls):
    S = np.full((len(Q), n_cls), -np.inf, np.float32)
    for c, h in hulls.items():
        S[:, c] = getattr(h, method)(Q)
    return S


def multi_mat(Ztr, ytr, fit_rows, Q, n_cls):
    from sklearn.cluster import KMeans
    S = np.full((len(Q), n_cls), -np.inf, np.float32)
    for c in range(n_cls):
        rows = fit_rows[ytr[fit_rows] == c]
        if len(rows) < MIN_N:
            continue
        X = Ztr[rows]
        k = min(N_SUB, max(1, len(X) // MIN_N))
        lab = (KMeans(n_clusters=k, n_init=4, random_state=c).fit_predict(X) if k > 1
               else np.zeros(len(X), int))
        subs = [X[lab == j] for j in range(k) if (lab == j).sum() >= MIN_N] or [X]
        S[:, c] = np.stack([fit_one(Xs).score(Q) for Xs in subs]).max(0)
    return S


def zs(A, seen):
    """Row z-score over SEEN columns only; unseen get a finite floor (0*-inf would be nan)."""
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


# ------------------------------------------------------------------ one cell
def run_cell(ds, T, seed):
    f = os.path.join(REPO, f"exp16_feats_{ds}_T{T}_s{seed}_ep40_lr0.0003_aug1_"
                           f"{MODEL.split('.')[-1]}.npz")
    if not os.path.exists(f):
        log(f"  MISSING {f}")
        return None
    z = np.load(f)
    ytr, yte, n_cls = get_labels(ds)
    Ztr, Zte = un(z["Ftr"]).astype(np.float32), un(z["Fte"]).astype(np.float32)
    assert len(Ztr) == len(ytr) and len(Zte) == len(yte), \
        f"feature/label mismatch {ds}: {Ztr.shape} vs {ytr.shape}"
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    FIT, VAL = [], []
    for t in range(T):
        idx = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(idx))
        nv = max(int(0.1 * len(idx)), 1)
        VAL.append(idx[pm[:nv]]); FIT.append(idx[pm[nv:]])
    FIT_ALL, VAL_ALL = np.concatenate(FIT), np.concatenate(VAL)
    sizes = [int((ytr[FIT_ALL] == c).sum()) for c in range(n_cls)]
    log(f"  {ds} T={T} s={seed}: {n_cls} cls, fit rows/class min {min(sizes)} "
        f"med {int(np.median(sizes))} max {max(sizes)}")

    # ---- hulls + score matrices (once; A_plus froze the backbone so they never change) ----
    Q_val, Q_te = Ztr[VAL_ALL], Zte
    S = {}
    need_plain = [v for v in WANT if v in SCORE_METHOD] + \
                 (["cone_score"] if "cone_fuse" in WANT else [])
    if need_plain:
        hulls = {c: fit_one(Ztr[FIT_ALL[ytr[FIT_ALL] == c]])
                 for c in range(n_cls) if sizes[c] >= MIN_N}
        log(f"    fitted {len(hulls)} hulls (n_rays={N_RAYS})")
        for v in set(need_plain):
            S[v] = (score_mat(hulls, Q_val, SCORE_METHOD[v], n_cls),
                    score_mat(hulls, Q_te, SCORE_METHOD[v], n_cls))
            log(f"    scored {v}")
    if "cone_klocal" in WANT:
        hk = {c: fit_one(Ztr[FIT_ALL[ytr[FIT_ALL] == c]], k_local=K_LOCAL)
              for c in range(n_cls) if sizes[c] >= MIN_N}
        S["cone_klocal"] = (score_mat(hk, Q_val, "score", n_cls),
                            score_mat(hk, Q_te, "score", n_cls))
        log("    scored cone_klocal")
    if "cone_multi" in WANT:
        S["cone_multi"] = (multi_mat(Ztr, ytr, FIT_ALL, Q_val, n_cls),
                           multi_mat(Ztr, ytr, FIT_ALL, Q_te, n_cls))
        log("    scored cone_multi")

    # ---- NCM ----
    mu = np.zeros((n_cls, Ztr.shape[1]), np.float32)
    for c in range(n_cls):
        r = FIT_ALL[ytr[FIT_ALL] == c]
        if len(r):
            mu[c] = Ztr[r].mean(0)
    mu = un(mu)
    NCM_val, NCM_te = Q_val @ mu.T, Q_te @ mu.T

    # ---- RanPAC (the in-run bar) ----
    P = torch.randn(Ztr.shape[1], M_RP,
                    generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.tensor(X[i:i + bs], device=DEV,
                                             dtype=torch.float32) @ P)

    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def logits(X, W):
        return torch.cat([(h.double() @ W) for _, h in _H(X)]).cpu().numpy()

    res = {v: [] for v in WANT}
    nval = 0
    for t in range(T):
        for i, h in _H(Ztr[FIT[t]]):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval += len(VAL[t])
        vsl = slice(0, nval)
        yv, tei = ytr[VAL_ALL[vsl]], np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(L, y, rows=None):
            Ls = L if rows is None else L[rows]
            return float((np.asarray(seen)[Ls[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            W = torch.linalg.solve(G + lam * eye, C)
            a = acc(logits(Q_val[vsl], W), yv)
            if a > best:
                best, bw = a, W
        Lv, Lt = logits(Q_val[vsl], bw), logits(Q_te, bw)

        for v in WANT:
            if v == "ranpac":
                res[v].append(acc(Lt, yt, tei))
            elif v == "ncm":
                res[v].append(acc(NCM_te, yt, tei))
            elif v == "cone_fuse":
                sv, st = S["cone_score"]
                zLv, zSv = zs(Lv, seen), zs(sv[vsl], seen)
                zLt, zSt = zs(Lt[tei], seen), zs(st[tei], seen)
                bb, bv = 0.0, -1
                for b in BETAS:
                    a = acc(zLv + b * zSv, yv)
                    if a > bv:
                        bv, bb = a, b
                res[v].append(acc(zLt + bb * zSt, yt))
            else:
                res[v].append(acc(S[v][1], yt, tei))
    del G, C, P, eye
    torch.cuda.empty_cache()
    return {v: {"A_last": a[-1], "A_avg": float(np.mean(a)), "accs": a}
            for v, a in res.items()}


# ------------------------------------------------------------------ main
OUT = os.path.join(REPO, f"exp17_conic_classifier_{MODEL.split('.')[-1]}.json")
allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            key = f"{ds}|{T}|{seed}"
            if key in allres and all(v in allres[key] for v in WANT):
                log(f"skip {key} (done)")
                continue
            log(f"=== {key}")
            r = run_cell(ds, T, seed)
            if r:
                allres.setdefault(key, {}).update(r)
                json.dump(allres, open(OUT, "w"), indent=2)

W = 100
print("\n" + "=" * W)
print(f"EXP17 — conic hull as CLASSIFIER on A_plus features ({MODEL})")
print("=" * W)
for key in [f"{ds}|{T}|{s}" for ds in DSETS for T in TS for s in SEEDS]:
    if key not in allres:
        continue
    r = allres[key]
    bar = BARS.get(key)
    print(f"\n--- {key}" + (f"   [exp16 A_plus: A-Last {bar['A_last']*100:.2f} "
                            f"A-Avg {bar['A_avg']*100:.2f}]" if bar else ""))
    print(f"{'variant':<16}{'A-Last':>9}{'A-Avg':>9}{'vs ranpac':>11}")
    rp = r.get("ranpac", {}).get("A_last")
    for v, d in sorted(r.items(), key=lambda kv: -kv[1]["A_last"]):
        dl = f"{(d['A_last']-rp)*100:>+11.2f}" if rp is not None else " " * 11
        print(f"{v:<16}{d['A_last']*100:>9.2f}{d['A_avg']*100:>9.2f}{dl}")
print("\n" + "-" * W)
print("'ranpac' here is recomputed in-run and should match the exp16 A_plus cell; if it")
print("does not, the replay is broken and nothing else on that line is meaningful.")
print("cone_fuse has beta=0 in its grid, so it can only lose by failing to transfer val->test.")
print("=" * W)
print(f"wrote {OUT}")
