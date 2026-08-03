"""
crux_routing.py — CAN WE ROUTE? the one number that decides the per-stage cascade.

Setup: each stage s owns 20 classes and has an EXACT snapshotted adapter phi_s.
A test image x of class c belongs to stage s* = stage_of(c).  For every stage s we
featurise x with phi_s and compute a TYPICALITY / MEMBERSHIP score T_s(x), then route
to argmax_s T_s(x).  The question is simply: how often is that s*?

Why routing is possible at all: the class->stage split is random, so "stage" carries no
semantic content -- but adapter_s was TRAINED on stage-s classes, so it should produce
more in-distribution features for them.  That is an adapter-level OOD signal, and
detecting it does not require solving the 200-way problem.

The bar:  end-to-end ~= routing_acc x within_stage_acc,  within_stage_acc ~ 0.8413
          beat A+ (0.7858)  =>  routing_acc > 0.7858/0.8413 = 93.4%
          current implied routing (no-oracle 0.6200 / 0.8413) ~ 74%

Statistics compared (each = max over the stage's classes, i.e. "does ANY class here claim it?"):
  centroid      max cosine to class prototypes                     (1-point)
  multiproto    max cosine to K k-means centroids                  (equal-budget control)
  cone_full     max hull geo_residual in full 768-d                (CURRENT: degenerate,
                                                                    K=10 rays span <=10 of 768 dims
                                                                    so nothing is ever "inside")
  cone_sub      PER-CLASS PCA to d', hull with K > d' rays there,   <-- THE PROPOSED FIX
                score = in-subspace hull membership x subspace alignment
                (membership becomes dimensionally meaningful)
  maha          per-class Mahalanobis, shared within-stage covariance
  knn           negative distance to k-th nearest stage-s train feature
  featnorm      ||phi_s(x)|| — pure adapter-level typicality, no class model at all
  argmax_class  route via the globally best-scoring class (centroid) — the implicit
                router used by the no-oracle numbers

Each is reported raw and z-CALIBRATED per stage on held-out in-stage data (scores from
different stages are otherwise not commensurable).

Run:  python -u crux_routing.py            # first run trains + caches (~40 min)
      D_SUB=32 K_SUB=60 python -u crux_routing.py    # re-runs from cache in ~2 min
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from backbone import load_backbone, freeze_non_lora, get_lora_params
from conic_hull import ConicHull

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k"
N_TASKS, CPT = 10, 20
EPOCHS, LR, BS = 10, 1e-4, 128
N_RAYS_FULL = 10                                   # rays for the 768-d hull (as before)
D_SUB = int(os.environ.get("D_SUB", 24))           # per-class subspace dimension
K_SUB = int(os.environ.get("K_SUB", 40))           # rays inside that subspace (K > d')
CAL_FRAC = 0.15                                    # held-out per class for calibration
KNN_K = 5
CACHE = "crux_routing_cache.npz"
ADAPTERS = "crux_routing_adapters.pt"

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)


class HFWrap(Dataset):
    def __init__(self, ds, idx, labels):
        self.ds, self.idx, self.labels = ds, np.asarray(idx), np.asarray(labels)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return TF(img), int(self.labels[i])


from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"]); _cl = np.array(sorted(set(_w.tolist())))
_lab = np.searchsorted(_cl, _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
TRAIN = HFWrap(_ds, _p[:_n], _lab[_p[:_n]]);  TR_Y = _lab[_p[:_n]]
TEST  = HFWrap(_ds, _p[_n:], _lab[_p[_n:]]);  TE_Y = _lab[_p[_n:]]
N_CLS = len(_cl)
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]
STAGE_OF = np.empty(N_CLS, dtype=int)
for s, cs in enumerate(TASKS):
    STAGE_OF[cs] = s
TRUE_STAGE = STAGE_OF[TE_Y]
print(f"[ImageNet-R] test {len(TE_Y)} | {N_TASKS} stages x {CPT} | "
      f"d'={D_SUB} K_sub={K_SUB}")


@torch.no_grad()
def extract(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ==================== phase 1: adapters + per-frame features (cached) ====================
STAGE_TR_IDX = [np.where(np.isin(TR_Y, TASKS[s]))[0] for s in range(N_TASKS)]

if os.path.exists(CACHE):
    print(f"[cache] loading {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    F_te = {s: z[f"te{s}"] for s in range(N_TASKS)}
    F_tr = {s: z[f"tr{s}"] for s in range(N_TASKS)}
else:
    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lora_params = list(get_lora_params(model))
    lora_names = set(n for n, p in model.named_parameters() if p.requires_grad)
    adapters = []
    for t in range(N_TASKS):
        cls = np.asarray(TASKS[t]); remap = {int(c): i for i, c in enumerate(cls)}
        loader = DataLoader(Subset(TRAIN, STAGE_TR_IDX[t].tolist()), batch_size=BS,
                            shuffle=True, num_workers=8, pin_memory=True)
        head = nn.Linear(768, CPT).to(DEV)
        opt = torch.optim.AdamW(lora_params + list(head.parameters()), lr=LR,
                                weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        ce = nn.CrossEntropyLoss()
        for ep in range(EPOCHS):
            model.train(); ok = tot = 0
            for x, lab in loader:
                x = x.to(DEV, non_blocking=True)
                y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = head(model(x).float()); loss = ce(lg, y)
                opt.zero_grad(); loss.backward(); opt.step()
                ok += int((lg.argmax(1) == y).sum()); tot += len(y)
            sch.step()
        adapters.append({n: p.detach().cpu().clone()
                         for n, p in model.named_parameters() if n in lora_names})
        print(f"  [stage {t}] trainacc {ok/tot:.3f}")
    torch.save(adapters, ADAPTERS)

    F_te, F_tr = {}, {}
    for s in range(N_TASKS):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in adapters[s]:
                    p.copy_(adapters[s][n].to(DEV))
        F_te[s] = extract(model, TEST, np.arange(len(TE_Y)))
        F_tr[s] = extract(model, TRAIN, STAGE_TR_IDX[s])   # stage-s own classes only
        print(f"  [featurise] stage {s} done")
    np.savez(CACHE, **{f"te{s}": F_te[s] for s in range(N_TASKS)},
             **{f"tr{s}": F_tr[s] for s in range(N_TASKS)})
    del model; torch.cuda.empty_cache()


# ==================== phase 2: per-stage class models ====================
class SubHull:
    """Per-class PCA -> conic hull INSIDE that subspace (K > d'), so 'inside the cone'
    is dimensionally meaningful. Score = hull membership x subspace alignment."""

    def __init__(self, X, d_sub, k):
        Xn = un(X)
        d = int(min(d_sub, max(len(Xn) - 1, 2), Xn.shape[1]))
        self.pca = PCA(n_components=d).fit(Xn)
        Z = self.pca.transform(Xn)
        self.hull = ConicHull(n_rays=int(min(k, len(Z))), use_pca=False,
                              ray_diversity="hybrid").fit(Z)

    def score(self, Q):
        Qn = un(Q)
        Z = self.pca.transform(Qn)
        recon = self.pca.inverse_transform(Z)
        align = np.clip((un(recon) * Qn).sum(1), 0, 1)     # how much of q lies in-subspace
        memb = self.hull.score_all(Z)["geo_residual"]      # inside-ness within the subspace
        return memb * align


print("\n=== building per-stage class models in their birth frames ===")
models = []
for s in range(N_TASKS):
    idx = STAGE_TR_IDX[s]
    y_s = TR_Y[idx]
    F_s = F_tr[s]
    fit_i, cal_i = [], []
    for c in TASKS[s]:
        loc = np.where(y_s == c)[0]
        ncal = max(int(CAL_FRAC * len(loc)), 2)
        cal_i.append(loc[:ncal]); fit_i.append(loc[ncal:])
    m = dict(cls=np.asarray(TASKS[s]), proto={}, multi={}, hull={}, sub={}, mu={},
             fitF={}, cal_idx=np.concatenate(cal_i), cal_y=y_s[np.concatenate(cal_i)])
    allfit = []
    for c, fi in zip(TASKS[s], fit_i):
        X = un(F_s[fi]); c = int(c)
        m["proto"][c] = un(X.mean(0, keepdims=True))[0]
        k = int(min(N_RAYS_FULL, max(len(X) // 4, 1)))
        m["multi"][c] = un(KMeans(n_clusters=k, n_init=4, random_state=0)
                           .fit(X).cluster_centers_)
        m["hull"][c] = ConicHull(n_rays=int(min(N_RAYS_FULL, len(X))), use_pca=True,
                                 pca_dim=int(min(64, max(len(X) - 1, 2)))).fit(X)
        m["sub"][c] = SubHull(X, D_SUB, K_SUB)
        m["mu"][c] = X.mean(0)
        m["fitF"][c] = X
        allfit.append(X)
    A = np.concatenate(allfit)
    Sig = np.cov(A.T) + 1e-3 * np.eye(A.shape[1])
    m["Prec"] = np.linalg.inv(Sig)
    m["train_all"] = A
    models.append(m)
    print(f"  stage {s}: {len(TASKS[s])} classes, {len(A)} fit / "
          f"{len(m['cal_idx'])} cal samples")


# ==================== phase 3: statistics ====================
ALL_STATS = ["centroid", "multiproto", "cone_full", "cone_sub", "maha", "knn", "featnorm"]
STATS = [s for s in os.environ.get("STATS", ",".join(ALL_STATS)).split(",") if s]


def stats_for(s, Q, tag="", verbose=False):
    """Routing statistics for queries Q already in stage-s frame. -> dict name->(n,)"""
    import time
    m = models[s]
    Qn = un(Q)
    out = {}

    def tick(name, fn):
        if name not in STATS and name != "_argmax_cls":
            return
        t0 = time.time()
        out[name] = fn()
        if verbose:
            print(f"      {tag}{name:<12} {time.time()-t0:6.2f}s", flush=True)

    P = np.stack([m["proto"][int(c)] for c in m["cls"]])
    cos = Qn @ P.T
    out["_argmax_cls"] = cos                                   # for the argmax_class router
    tick("centroid", lambda: cos.max(1))
    tick("multiproto", lambda: np.stack(
        [np.max(Qn @ m["multi"][int(c)].T, 1) for c in m["cls"]], 1).max(1))
    tick("cone_full", lambda: np.stack(
        [m["hull"][int(c)].score_all(Qn)["geo_residual"] for c in m["cls"]], 1).max(1))
    tick("cone_sub", lambda: np.stack(
        [m["sub"][int(c)].score(Qn) for c in m["cls"]], 1).max(1))

    def _maha():
        # (q-mu)^T P (q-mu) = q^T P q - 2 q^T P mu + mu^T P mu, shared P => one matmul
        # for ALL classes (the naive per-class einsum was ~9.5 s/class = 32 min total).
        MU = np.stack([m["mu"][int(c)] for c in m["cls"]])
        qP = Qn @ m["Prec"]
        t1 = (qP * Qn).sum(1)[:, None]
        t2 = qP @ MU.T
        t3 = ((MU @ m["Prec"]) * MU).sum(1)[None, :]
        return (-(t1 - 2 * t2 + t3)).max(1)
    tick("maha", _maha)

    def _knn():
        T = torch.tensor(m["train_all"], device=DEV, dtype=torch.float32)
        ds = []
        for i in range(0, len(Qn), 1024):
            q = torch.tensor(Qn[i:i + 1024], device=DEV, dtype=torch.float32)
            ds.append((-torch.kthvalue(torch.cdist(q, T), KNN_K, dim=1)
                       .values).cpu().numpy())
        return np.concatenate(ds)
    tick("knn", _knn)
    tick("featnorm", lambda: np.linalg.norm(Q, axis=1))
    return out


N_TEST = int(os.environ.get("N_TEST", 0))          # >0 => subsample test for a fast smoke run
if N_TEST and N_TEST < len(TE_Y):
    sub = np.random.default_rng(0).choice(len(TE_Y), N_TEST, replace=False)
    TE_Y = TE_Y[sub]; TRUE_STAGE = TRUE_STAGE[sub]
    F_te = {s: F_te[s][sub] for s in range(N_TASKS)}
    print(f"[smoke] subsampled test set to {N_TEST}")

print(f"\n=== scoring test set under every stage  (stats: {','.join(STATS)}) ===", flush=True)
import time as _t
S_te, S_cal = {}, {}
for s in range(N_TASKS):
    t0 = _t.time()
    print(f"  stage {s}/{N_TASKS-1}: test ({len(TE_Y)} q)", flush=True)
    S_te[s] = stats_for(s, F_te[s], tag="test  ", verbose=True)
    S_cal[s] = stats_for(s, F_tr[s][models[s]["cal_idx"]], tag="cal   ", verbose=True)
    print(f"  stage {s} scored in {_t.time()-t0:.1f}s", flush=True)


# ==================== phase 4: routing accuracy ====================
def routing_acc(name):
    raw = np.stack([S_te[s][name] for s in range(N_TASKS)], 1)          # (n_test, 10)
    mu = np.array([S_cal[s][name].mean() for s in range(N_TASKS)])
    sd = np.array([S_cal[s][name].std() + 1e-8 for s in range(N_TASKS)])
    cal = (raw - mu) / sd
    a_raw = float((raw.argmax(1) == TRUE_STAGE).mean())
    a_cal = float((cal.argmax(1) == TRUE_STAGE).mean())
    top2 = float(np.mean([TRUE_STAGE[i] in np.argsort(-cal[i])[:2]
                          for i in range(len(TRUE_STAGE))]))
    return a_raw, a_cal, top2


# router that just takes the globally best class (what the no-oracle numbers used)
gl = np.concatenate([S_te[s]["_argmax_cls"] for s in range(N_TASKS)], 1)
gl_cls = np.concatenate([models[s]["cls"] for s in range(N_TASKS)])
a_argmax = float((STAGE_OF[gl_cls[gl.argmax(1)]] == TRUE_STAGE).mean())

# within-stage accuracy under oracle routing (centroid), for the end-to-end estimate
hit = tot = 0
for s in range(N_TASKS):
    own = np.where(TRUE_STAGE == s)[0]
    P = np.stack([models[s]["proto"][int(c)] for c in models[s]["cls"]])
    pred = models[s]["cls"][(un(F_te[s][own]) @ P.T).argmax(1)]
    hit += int((pred == TE_Y[own]).sum()); tot += len(own)
WITHIN = hit / tot
NEED = 0.7858 / WITHIN

print("\n" + "=" * 92)
print("STAGE-ROUTING DIAGNOSTIC — Split-ImageNet-R, 10-way stage ID (chance = 0.100)")
print("=" * 92)
print(f"within-stage acc (oracle route, centroid) = {WITHIN:.4f}")
print(f"=> to beat A+ (0.7858) need routing > {NEED:.4f}")
print("-" * 92)
print(f"{'statistic':>14} {'raw':>9} {'calibrated':>11} {'top-2':>9} "
      f"{'end-to-end':>11} {'vs A+':>8}")
rows = []
for nm in STATS:
    r, c, t2 = routing_acc(nm)
    e2e = c * WITHIN
    rows.append((nm, r, c, t2, e2e))
    print(f"{nm:>14} {r:>9.4f} {c:>11.4f} {t2:>9.4f} {e2e:>11.4f} {e2e-0.7858:>+8.4f}")
print(f"{'argmax_class':>14} {a_argmax:>9.4f} {'-':>11} {'-':>9} "
      f"{a_argmax*WITHIN:>11.4f} {a_argmax*WITHIN-0.7858:>+8.4f}")
print("-" * 92)
best = max(rows, key=lambda r: r[2])
print(f"BEST: {best[0]}  routing {best[2]:.4f}  ->  end-to-end {best[4]:.4f}")
R = {r[0]: r[2] for r in rows}


def _delta(a, b, msg):
    if a in R and b in R:
        print(f"{a} - {b} = {R[a]-R[b]:+.4f}   {msg}")


_delta("cone_sub", "cone_full", "<- did making membership dimensionally meaningful help?")
_delta("cone_sub", "centroid", "<- does the hull boundary beat a prototype at ROUTING?")
print(f"\nVERDICT: routing must exceed {NEED:.3f}. If nothing does, the per-stage cascade")
print("cannot clear the bar no matter how it is wired.")
print("=" * 92)
np.save("crux_routing_results.npy",
        {r[0]: dict(raw=r[1], cal=r[2], top2=r[3], e2e=r[4]) for r in rows},
        allow_pickle=True)
