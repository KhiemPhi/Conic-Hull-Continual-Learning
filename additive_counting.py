"""
additive_counting.py
---------------------
Solidify + generalize the cone niche (non-negative additive QUANTITY decoding).

Adds to coco_counting.py:
  #1 calibrated MAE (per-category scale fit) alongside scale-free Spearman
  #2 RAW (magnitude-preserving) vs L2-NORMALIZED features — the "additive
     magnitude" test. Sum-pooling ≡ mean-pooling up to a global scale, so what
     matters is whether keeping the feature NORM (which encodes "how much
     content") helps the additive decoder. Cached feats are raw, so this is free.
  #3 --dataset arg (COCO | CPPE5 | ...) to test a second HF detection set.

Decoders (predict per-category count from an image embedding):
  ridge(test-opt) : linear regression, alpha swept & best-on-test (favours baseline)
  nnls-cone       : x ≈ Σ aᵢ·atomᵢ (aᵢ≥0, kNN); count_c = Σ aᵢ·Count[i,c]
  cos-knn         : cosine-locality control (weighted-mean of neighbour counts)

    HF_HUB_OFFLINE=1 python -u additive_counting.py --dataset COCO --seeds 3
    HF_HUB_OFFLINE=1 python -u additive_counting.py --dataset CPPE5 --seeds 5
"""
import argparse, io, os, json
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DATA_DIR, OUT_DIR = "./data", "./additive_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_PRESENT = 20
ALPHAS = [1, 10, 100, 1000]

# name -> (repo, splits-to-pool). None splits => all available splits pooled.
DATASETS = {
    "COCO":  ("detection-datasets/coco", ["val"]),
    "CPPE5": ("rishitdagli/cppe-5", None),
}


# ── data ─────────────────────────────────────────────────────────────────────
def load_ds(name):
    from datasets import load_dataset, concatenate_datasets
    repo, splits = DATASETS[name]
    cache = os.path.join(DATA_DIR, "hf")
    if splits is None:
        dd = load_dataset(repo, cache_dir=cache)
        return concatenate_datasets([dd[s] for s in dd])
    parts = [load_dataset(repo, split=s, cache_dir=cache) for s in splits]
    from datasets import concatenate_datasets as cat
    return cat(parts) if len(parts) > 1 else parts[0]


def build_counts(ds):
    print(f"[count] reading 'objects' column ({len(ds)} rows, no image decode)...",
          flush=True)
    objs = ds["objects"]
    cats = [list(o["category"]) for o in objs]
    n = max((max(c) + 1 for c in cats if c), default=0)
    C = np.zeros((len(cats), n), np.float32)
    for i, c in enumerate(cats):
        for k in c:
            C[i, int(k)] += 1.0
    print(f"[count] {C.shape}, mean instances/img {C.sum(1).mean():.2f}", flush=True)
    return C


class _Imgs(Dataset):
    def __init__(self, ds, pre): self.ds, self.pre = ds, pre
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img = self.ds[i]["image"]
        if not hasattr(img, "mode"):
            from PIL import Image; img = Image.open(io.BytesIO(img["bytes"]))
        return self.pre(img.convert("RGB"))


@torch.no_grad()
def extract(name, ds):
    path = os.path.join(OUT_DIR, f"feats_{name}.npz")
    # reuse the older COCO cache if present
    legacy = "./coco_law_out/coco_clip_feats.npz"
    if name == "COCO" and not os.path.exists(path) and os.path.exists(legacy):
        path = legacy
    if os.path.exists(path):
        d = np.load(path); return d["cls"].astype(np.float32), d["patch"].astype(np.float32)
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.to(DEVICE).eval(); model.visual.output_tokens = True
    CLS, PATCH = [], []
    for x in tqdm(DataLoader(_Imgs(ds, pre), batch_size=128, num_workers=8),
                  desc=f"extract {name}"):
        x = x.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled, tok = model.visual(x)
        if tok.shape[1] % 2 == 1:
            tok = tok[:, 1:]
        CLS.append(pooled.float().cpu().numpy()); PATCH.append(tok.float().mean(1).cpu().numpy())
    cls, patch = np.concatenate(CLS), np.concatenate(PATCH)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"feats_{name}.npz"), cls=cls, patch=patch)
    return cls.astype(np.float32), patch.astype(np.float32)


def _unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


# ── decoders ─────────────────────────────────────────────────────────────────
def nnls_cone(Xtr, Ctr, Xte, k=100, chunk=512):
    A = torch.tensor(Xtr, device=DEVICE)
    out = np.zeros((len(Xte), Ctr.shape[1]), np.float32)
    for i in range(0, len(Xte), chunk):
        Q = torch.tensor(Xte[i:i+chunk], device=DEVICE)
        idx = (Q @ A.T).topk(k, dim=1).indices.cpu().numpy()
        for j in range(len(idx)):
            a, _ = nnls(Xtr[idx[j]].T, Xte[i+j]); out[i+j] = a @ Ctr[idx[j]]
    return out


def cos_knn(Xtr, Ctr, Xte, k=100, chunk=512):
    A = torch.tensor(_unit(Xtr), device=DEVICE); Ct = torch.tensor(Ctr, device=DEVICE)
    Xn = _unit(Xte); out = np.zeros((len(Xte), Ctr.shape[1]), np.float32)
    for i in range(0, len(Xn), chunk):
        Q = torch.tensor(Xn[i:i+chunk], device=DEVICE)
        vals, idx = (Q @ A.T).topk(k, dim=1); w = vals.clamp_min(0)
        out[i:i+chunk] = (torch.einsum("bk,bkc->bc", w, Ct[idx]) /
                          (w.sum(1, keepdim=True)+1e-8)).cpu().numpy()
    return out


def best_ridge(Xtr, Ctr, Xte, Cte):
    best_rho, best_P = -1, None
    for a in ALPHAS:
        r = Ridge(alpha=a); r.fit(Xtr, Ctr); P = np.clip(r.predict(Xte), 0, None)
        rho = spearman_present(P, Cte)[0]
        if rho > best_rho: best_rho, best_P = rho, P
    return best_P


# ── metrics ──────────────────────────────────────────────────────────────────
def spearman_present(P, C, tmin=1):
    rs = []
    for c in range(C.shape[1]):
        m = C[:, c] >= tmin
        if m.sum() >= MIN_PRESENT and len(np.unique(C[m, c])) > 1:
            rho = spearmanr(P[m, c], C[m, c]).correlation
            if not np.isnan(rho): rs.append(rho)
    return (float(np.mean(rs)) if rs else float("nan")), len(rs)


def mae_present(P, C):
    """Per-category scale-corrected MAE on present images (symmetric across methods)."""
    es = []
    for c in range(C.shape[1]):
        m = C[:, c] >= 1
        if m.sum() >= MIN_PRESENT:
            s = C[m, c].sum() / (P[m, c].sum() + 1e-8)   # global scale per cat
            es.append(np.mean(np.abs(s * P[m, c] - C[m, c])))
    return float(np.mean(es)) if es else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="COCO", choices=list(DATASETS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[load] {args.dataset}", flush=True)
    ds = load_ds(args.dataset); Cnt = build_counts(ds)
    cls, patch = extract(args.dataset, ds)
    print(f"[feat] cls {cls.shape} patch {patch.shape}", flush=True)

    variants = {"CLS-norm": _unit(cls), "CLS-raw": cls,
                "Patch-norm": _unit(patch), "Patch-raw": patch}
    res = {}
    for vname, F in variants.items():
        agg = {d: {"rho": [], "mae": []} for d in ("ridge", "nnls-cone", "cos-knn")}
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed); perm = rng.permutation(len(Cnt))
            n_te = int(len(Cnt) * 0.4); te, tr = perm[:n_te], perm[n_te:]
            Xtr, Xte, Ctr, Cte = F[tr], F[te], Cnt[tr], Cnt[te]
            P = {"ridge": best_ridge(Xtr, Ctr, Xte, Cte),
                 "nnls-cone": nnls_cone(Xtr, Ctr, Xte, k=args.k),
                 "cos-knn": cos_knn(Xtr, Ctr, Xte, k=args.k)}
            for d in P:
                agg[d]["rho"].append(spearman_present(P[d], Cte)[0])
                agg[d]["mae"].append(mae_present(P[d], Cte))
        res[vname] = {d: {"rho": float(np.mean(agg[d]["rho"])),
                          "rho_std": float(np.std(agg[d]["rho"])),
                          "mae": float(np.mean(agg[d]["mae"]))} for d in agg}
        rc, rr = res[vname]["nnls-cone"], res[vname]["ridge"]
        print(f"[{vname:11s}] cone ρ {rc['rho']:.3f}±{rc['rho_std']:.3f} MAE {rc['mae']:.3f} "
              f"| ridge ρ {rr['rho']:.3f} MAE {rr['mae']:.3f} "
              f"| cos-knn ρ {res[vname]['cos-knn']['rho']:.3f} "
              f"|| Δρ(cone−ridge) {rc['rho']-rr['rho']:+.3f}  ΔMAE {rr['mae']-rc['mae']:+.3f}",
              flush=True)

    with open(os.path.join(OUT_DIR, f"results_{args.dataset}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[done] {OUT_DIR}/results_{args.dataset}.json", flush=True)


if __name__ == "__main__":
    main()
