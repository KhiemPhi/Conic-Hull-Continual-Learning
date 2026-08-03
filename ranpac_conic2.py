"""
ranpac_conic2.py
----------------
Aid A, CIL-VALID incremental basis. The Gram needs a FIXED projection, so we fit the
extreme-ray basis on the BASE SESSION only (available at t=0), freeze it (exactly like
RanPAC's random W), then accumulate the decorrelated-prototype classifier over
base + increments (forgetting-free). Protocol: B50 (50 base classes + 5 x 10).

Compares projection bases, swept over dim M:
  random          : RanPAC's data-agnostic W (fixed)
  extremeray-base : SPA extreme rays fit on the BASE SESSION, frozen (CIL-valid, ours)
  extremeray-all  : rays fit on all-class train (idealized upper bound, reference)

Datasets: CIFAR100 (cached feats) and ImageNet-R (extract). Metric: avg-inc accuracy.

    HF_HUB_OFFLINE=1 python -u ranpac_conic2.py --dataset CIFAR100
    HF_HUB_OFFLINE=1 python -u ranpac_conic2.py --dataset ImageNet-R
"""
import argparse, io, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import timm
from conic_hull import ConicHull

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./ranpac_out"
MODEL = "vit_base_patch16_224"
LAM = 1e3


@torch.no_grad()
def feats_cifar():
    path = os.path.join(OUT, "cifar100_feats.npz")
    d = np.load(path)
    return d["ftr"], d["ytr"], d["fte"], d["yte"]


class _HF(Dataset):
    def __init__(self, ds, tf, ik, lk): self.ds, self.tf, self.ik, self.lk = ds, tf, ik, lk
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[i]; img = r[self.ik]
        if not hasattr(img, "mode"):
            from PIL import Image; img = Image.open(io.BytesIO(img["bytes"]))
        return self.tf(img.convert("RGB")), int(r[self.lk])


@torch.no_grad()
def feats_imagenetr():
    path = os.path.join(OUT, "imagenetr_feats.npz")
    if os.path.exists(path):
        d = np.load(path); return d["ftr"], d["ytr"], d["fte"], d["yte"]
    from datasets import load_dataset
    dd = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")
    ds = dd[list(dd)[0]]
    cols = ds.column_names
    ik = next(c for c in ("image", "img", "Image") if c in cols)
    lk = next(c for c in ("label", "labels", "fine_label", "class") if c in cols)
    cfg = timm.data.resolve_data_config({}, model=timm.create_model(MODEL))
    tf = timm.data.create_transform(**cfg, is_training=False)
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(DEVICE).eval()
    F, Y = [], []
    for x, y in tqdm(DataLoader(_HF(ds, tf, ik, lk), batch_size=256, num_workers=8), desc="imr"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F.append(model(x.to(DEVICE)).float().cpu().numpy())
        Y.append(np.asarray(y))
    F = np.concatenate(F).astype(np.float32); Y = np.concatenate(Y)
    # per-class 50/50 split
    rng = np.random.default_rng(0); tr_idx, te_idx = [], []
    for c in np.unique(Y):
        idx = np.where(Y == c)[0]; rng.shuffle(idx); h = len(idx)//2
        tr_idx += idx[:h].tolist(); te_idx += idx[h:].tolist()
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, ftr=F[tr_idx], ytr=Y[tr_idx], fte=F[te_idx], yte=Y[te_idx])
    return F[tr_idx], Y[tr_idx], F[te_idx], Y[te_idx]


def ranpac_b50(proj, Ftr, ytr, Fte, yte, order, base, inc, M):
    """Forgetting-free RanPAC over base session + increments (fixed proj)."""
    G = np.zeros((M, M), np.float32); csum, ccount = {}, {}; accs = []
    sessions = [order[:base]] + [order[base+i*inc:base+(i+1)*inc]
                                 for i in range((len(order)-base)//inc)]
    seen = []
    for cls in sessions:
        seen += list(cls)
        m = np.isin(ytr, cls); phi = proj(Ftr[m]); yt = ytr[m]
        G += phi.T @ phi
        for c in cls:
            csum[c] = phi[yt == c].sum(0); ccount[c] = (yt == c).sum()
        Ginv = np.linalg.inv(G + LAM * np.eye(M, dtype=np.float32))
        labels = sorted(seen)
        Wc = np.stack([Ginv @ (csum[c] / ccount[c]) for c in labels])
        mt = np.isin(yte, seen); phite = proj(Fte[mt])
        pred = np.array(labels)[(phite @ Wc.T).argmax(1)]
        accs.append(float((pred == yte[mt]).mean()))
    return float(np.mean(accs) * 100), float(accs[-1] * 100)


def fit_rays(F, Mmax, sub=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), min(sub, len(F)), replace=False)
    ch = ConicHull(n_rays=Mmax, use_pca=True, pca_dim=64, ray_diversity="hybrid")
    ch.fit(F[idx]); return ch.extreme_rays_.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR100", choices=["CIFAR100", "ImageNet-R"])
    ap.add_argument("--Ms", type=int, nargs="+", default=[256, 512, 1024, 2048])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = feats_cifar() if args.dataset == "CIFAR100" else feats_imagenetr()
    D = Ftr.shape[1]; ncls = int(max(ytr.max(), yte.max()) + 1)
    base = ncls // 2; inc = max(10, ncls // 10)
    order = np.random.default_rng(1993).permutation(ncls)
    print(f"[data] {args.dataset} train {Ftr.shape} classes {ncls} | B{base} + {inc}-incs", flush=True)

    Mmax = max(args.Ms); rng = np.random.default_rng(0)
    base_cls = order[:base]
    rays_base = fit_rays(Ftr[np.isin(ytr, base_cls)], Mmax)      # CIL-valid: base only
    rays_all = fit_rays(Ftr, Mmax)                                # idealized reference
    print(f"[rays] base {rays_base.shape[0]}  all {rays_all.shape[0]}", flush=True)

    res = {}
    for M in args.Ms:
        Wr = (rng.standard_normal((D, M)) / np.sqrt(D)).astype(np.float32)
        rnd = ranpac_b50(lambda X: np.maximum(X @ Wr, 0), Ftr, ytr, Fte, yte, order, base, inc, M)
        Rb = rays_base[:M]; Ra = rays_all[:M]
        rb = ranpac_b50(lambda X: np.maximum(X @ Rb.T, 0), Ftr, ytr, Fte, yte, order, base, inc, M)
        ra = ranpac_b50(lambda X: np.maximum(X @ Ra.T, 0), Ftr, ytr, Fte, yte, order, base, inc, M)
        res[M] = {"random": rnd[0], "ray_base": rb[0], "ray_all": ra[0]}
        print(f"  [M={M:>4}] random {rnd[0]:.1f} | ray-base(CIL) {rb[0]:.1f} "
              f"(Δ {rb[0]-rnd[0]:+.1f}) | ray-all(ideal) {ra[0]:.1f}", flush=True)

    with open(os.path.join(OUT, f"aidA_incr_{args.dataset}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== {args.dataset} avg-inc acc ===\n| M | random | ray-base(CIL) | Δ | ray-all(ideal) |\n|--:|--:|--:|--:|--:|")
    for M, r in res.items():
        print(f"| {M} | {r['random']:.1f} | {r['ray_base']:.1f} | {r['ray_base']-r['random']:+.1f} | {r['ray_all']:.1f} |")


if __name__ == "__main__":
    main()
