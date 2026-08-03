"""
crux_memory.py — DOES THE CONE WIN ON MEMORY?  (exemplar selection, matched budget)

Claim under test:  hull-selected exemplars at budget B match random at budget 2B
                   ("same representation, less storage").

Realistic CIL setup (this is the wrinkle that makes it non-trivial):
  * class c is born at stage s.  You choose its B exemplars THEN, using only phi_s.
  * later, at stage 9, those B images are re-featurised with phi_9 to rebuild the class's
    statistics.  So the selection must be robust to the frame moving underneath it.

Mechanism prediction (from crux_stagecone):
  * SPA extreme rays are OUTLIERS -> bad for estimating a MEAN
        (max_ray_sim 0.7095 vs k-means multiproto 0.8548 = -14.5)
  * but they COVER THE EXTENT -> potentially good for estimating a SECOND MOMENT
  * RanPAC's head is driven by the Gram (2nd moment); NCM by the mean (1st moment)
  => predict: hull HURTS NCM, HELPS RanPAC.  Reporting both separates mechanism from
     bottom line; if hull wins on RanPAC while losing on NCM, the story is confirmed.

Selection rules (all return indices into the class's own training images):
  random | herding (iCaRL) | kmedoid | hull_spa | hull_hybrid (half medoid, half rays)
  + oracle_hull: selected in phi_9 instead of phi_s, to price the frame-mismatch.

Run:  python -u crux_memory.py            (needs crux_routing_cache.npz + adapters)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform
from sklearn.cluster import KMeans

from backbone import load_backbone, freeze_non_lora
from conic_hull import ConicHull

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT = 10, 20
BUDGETS = [5, 10, 20]
M_RP = 2000                       # smaller than 10k so even B=5 (1000 samples) is workable
LAMBDAS = [1e1, 1e2, 1e3]
CACHE = "crux_routing_cache.npz"
ADAPTERS = "crux_routing_adapters.pt"
X9_CACHE = "crux_memory_x9.npy"

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
STAGE_TR_IDX = [np.where(np.isin(TR_Y, TASKS[s]))[0] for s in range(N_TASKS)]


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
def deg(c): return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ---------------- features ----------------
z = np.load(CACHE, allow_pickle=True)
F_tr_own = {s: z[f"tr{s}"] for s in range(N_TASKS)}     # stage-s classes in frame s
F_te9 = z["te9"]                                         # test in final frame

if os.path.exists(X9_CACHE):
    X9 = np.load(X9_CACHE)
else:
    print("=== extracting ALL train in final frame phi_9 ===")
    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    ad = torch.load(ADAPTERS, map_location="cpu")[9]
    with torch.no_grad():
        for n_, p in model.named_parameters():
            if n_ in ad:
                p.copy_(ad[n_].to(DEV))
    model.eval()
    loader = DataLoader(TRAIN, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    X9 = np.concatenate(out, 0)
    np.save(X9_CACHE, X9)
    del model; torch.cuda.empty_cache()
print(f"[features] X9 {X9.shape} | test {F_te9.shape}")

# per-class: rows of its own training set, its selection-frame feats, its phi_9 feats
CLS_ROWS, SEL_F, EVAL_F = {}, {}, {}
for s in range(N_TASKS):
    ys = TR_Y[STAGE_TR_IDX[s]]
    for c in TASKS[s]:
        loc = np.where(ys == c)[0]
        rows = STAGE_TR_IDX[s][loc]
        CLS_ROWS[int(c)] = rows
        SEL_F[int(c)] = un(F_tr_own[s][loc])       # phi_s  (what you have at birth)
        EVAL_F[int(c)] = un(X9[rows])              # phi_9  (where it is later used)


# ---------------- selection rules ----------------
def sel_random(X, B, seed=0):
    r = np.random.default_rng(seed)
    return r.choice(len(X), min(B, len(X)), replace=False)


def sel_herding(X, B):
    Xn = un(X); mu = un(Xn.mean(0, keepdims=True))[0]
    picked, cur = [], np.zeros(Xn.shape[1])
    for k in range(min(B, len(Xn))):
        sc = ((cur + Xn) / (k + 1)) @ mu
        if picked:
            sc[np.array(picked)] = -np.inf
        i = int(np.argmax(sc)); picked.append(i); cur = cur + Xn[i]
    return np.array(picked)


def sel_kmedoid(X, B):
    Xn = un(X); k = int(min(B, len(Xn)))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(Xn)
    return np.unique([int(np.argmin(np.linalg.norm(Xn - cc, axis=1)))
                      for cc in km.cluster_centers_])


def sel_hull(X, B):
    n = len(X)
    h = ConicHull(n_rays=int(min(B, n)), use_pca=True,
                  pca_dim=int(min(64, max(n - 1, 2)))).fit(un(X))
    return np.unique(np.asarray(h.extreme_rays_index))


def sel_hybrid(X, B):
    a = sel_kmedoid(X, max(B // 2, 1))
    b = sel_hull(X, B)
    out = list(a)
    for i in b:
        if len(out) >= B:
            break
        if i not in out:
            out.append(int(i))
    return np.array(out)


RULES = {"random": lambda X, B: sel_random(X, B),
         "herding": sel_herding,
         "kmedoid": sel_kmedoid,
         "hull_spa": sel_hull,
         "hull_hybrid": sel_hybrid}


# ---------------- heads ----------------
P_RP = torch.randn(768, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)


def ranpac(Ztr, ytr, Zte, yte):
    def H(Z):
        return torch.relu(torch.tensor(un(Z), device=DEV, dtype=torch.float32) @ P_RP).double()
    h = H(Ztr)
    Y = torch.zeros(len(ytr), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(ytr)), torch.tensor(ytr, device=DEV)] = 1.0
    G, C = h.T @ h, h.T @ Y
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    ht = H(Zte)
    best = 0.0
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, C)
        pred = (ht @ W).argmax(1).cpu().numpy()
        best = max(best, float((pred == yte).mean()))
    return best


def ncm(Ztr, ytr, Zte, yte):
    P = un(np.stack([un(Ztr[ytr == c]).mean(0) for c in range(N_CLS)]))
    return float((np.argmax(un(Zte) @ P.T, axis=1) == yte).mean())


# ---------------- run ----------------
print("\n=== references (ALL training data, frame phi_9) ===")
full_ncm = ncm(X9, TR_Y, F_te9, TE_Y)
full_rp = ranpac(X9, TR_Y, F_te9, TE_Y)
print(f"  full-data  NCM {full_ncm:.4f} | RanPAC {full_rp:.4f}   "
      f"({len(TR_Y)} samples, {len(TR_Y)/N_CLS:.0f}/class)")

rows = []
for B in BUDGETS:
    for rname, rule in RULES.items():
        Z, y, pdeg, spread = [], [], [], []
        for c in range(N_CLS):
            Xsel, Xev = SEL_F[c], EVAL_F[c]
            idx = rule(Xsel, B)                      # chosen in phi_s
            sub = Xev[idx]                           # used in phi_9
            Z.append(sub); y.append(np.full(len(sub), c))
            pdeg.append(deg(float(un(sub.mean(0, keepdims=True))[0]
                                  @ un(Xev.mean(0, keepdims=True))[0])))
            # trace(cov) == sum of per-dim variances (avoids building a 768x768 matrix)
            spread.append(float(sub.var(0, ddof=1).sum() /
                                (Xev.var(0, ddof=1).sum() + 1e-12)) if len(sub) > 1 else 0.0)
        Z = np.concatenate(Z); y = np.concatenate(y)
        a_ncm, a_rp = ncm(Z, y, F_te9, TE_Y), ranpac(Z, y, F_te9, TE_Y)
        rows.append(dict(B=B, rule=rname, n=len(Z), ncm=a_ncm, rp=a_rp,
                         proto_deg=float(np.mean(pdeg)), spread=float(np.mean(spread))))
        print(f"  B={B:>2} {rname:>12} n={len(Z):>5} | NCM {a_ncm:.4f} | RanPAC {a_rp:.4f}"
              f" | proto-err {np.mean(pdeg):5.2f}deg | spread {np.mean(spread):.2f}",
              flush=True)

# frame-mismatch price: select using phi_9 itself (cheating)
orc = []
for c in range(N_CLS):
    idx = sel_hull(EVAL_F[c], 10)
    orc.append((EVAL_F[c][idx], np.full(len(idx), c)))
Zo = np.concatenate([a for a, _ in orc]); yo = np.concatenate([b for _, b in orc])
o_ncm, o_rp = ncm(Zo, yo, F_te9, TE_Y), ranpac(Zo, yo, F_te9, TE_Y)

np.save("crux_memory_results.npy", rows, allow_pickle=True)
print("\n" + "=" * 94)
print("EXEMPLAR SELECTION — does the cone buy storage efficiency?  (frame phi_9, ImageNet-R)")
print("=" * 94)
print(f"{'budget':>7} {'rule':>13} {'NCM':>8} {'RanPAC':>8} {'proto-err':>10} {'spread':>7}")
for B in BUDGETS:
    for r in [x for x in rows if x["B"] == B]:
        print(f"{B:>7} {r['rule']:>13} {r['ncm']:>8.4f} {r['rp']:>8.4f} "
              f"{r['proto_deg']:>9.2f}d {r['spread']:>7.2f}")
    print()
print(f"{'ALL DATA':>7} {'(~96/class)':>13} {full_ncm:>8.4f} {full_rp:>8.4f}")
print(f"{'oracle':>7} {'hull@10 in phi9':>13} {o_ncm:>8.4f} {o_rp:>8.4f}"
      "   <- price of selecting in the OLD frame")
print("-" * 94)
g = {(r["B"], r["rule"]): r for r in rows}
print("THE CLAIM — hull at half the storage vs random at full:")
for B in [5, 10]:
    h, rr = g[(B, "hull_spa")], g[(2 * B, "random")]
    print(f"  hull@{B:<2} vs random@{2*B:<2} :  NCM {h['ncm']-rr['ncm']:+.4f}   "
          f"RanPAC {h['rp']-rr['rp']:+.4f}")
print("\nMECHANISM — at matched budget, hull vs kmedoid:")
for B in BUDGETS:
    h, k = g[(B, "hull_spa")], g[(B, "kmedoid")]
    print(f"  B={B:<2}: NCM {h['ncm']-k['ncm']:+.4f} (expect NEGATIVE: rays are outliers) | "
          f"RanPAC {h['rp']-k['rp']:+.4f} (expect POSITIVE if coverage helps the Gram)")
print("=" * 94)
