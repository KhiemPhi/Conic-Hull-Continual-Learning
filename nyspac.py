"""
nyspac.py
---------
NysPAC: beat RanPAC on ACCURACY at matched dimension by replacing its random-features
lift with a NYSTROM kernel feature map on CONIC extreme-ray landmarks.

RanPAC uses h=ReLU(Wx) = random features (data-agnostic kernel approx). Nystrom features
K_MM^{-1/2} k(x,L) with data landmarks L beat random features at the same rank M when the
kernel spectrum decays fast (Yang et al. NeurIPS'12) -- true for frozen ViT feats. The
conic hull supplies the landmarks: SPA extreme rays are a diverse, coverage-maximizing,
CIL-valid landmark set (Aid A showed base-session rays generalize).

Compared at matched M, CIFAR-100 B50 (forgetting-free RanPAC decorrelated prototypes):
  random-ReLU        : RanPAC baseline
  rbf-ray-nystrom    : Nystrom map, extreme-ray landmarks (OURS)
  rbf-kmeans-nystrom : Nystrom map, k-means landmarks (is it conic or just Nystrom?)
  rbf-ray-raw        : k(x,rays) w/o K_MM correction (is it the Nystrom correction?)

    HF_HUB_OFFLINE=1 python -u nyspac.py --Ms 1024 2048
"""
import argparse, os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from conic_hull import ConicHull

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./ranpac_out"
LAM = 1e3


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def feats():
    d = np.load(os.path.join(OUT, "cifar100_feats.npz"))
    return unit(d["ftr"]), d["ytr"], unit(d["fte"]), d["yte"]


def rbf(X, L, gamma, chunk=8192):
    """exp(-gamma ||x-l||^2); X,L unit-normalized. Returns (N, M) on CPU float32."""
    Lt = torch.tensor(L, device=DEVICE); ln = (Lt**2).sum(1)
    out = np.empty((len(X), len(L)), np.float32)
    for i in range(0, len(X), chunk):
        Xt = torch.tensor(X[i:i+chunk], device=DEVICE)
        d2 = (Xt**2).sum(1, keepdim=True) + ln[None, :] - 2 * Xt @ Lt.T
        out[i:i+chunk] = torch.exp(-gamma * d2.clamp_min(0)).cpu().numpy()
    return out


def nystrom_correction(L, gamma):
    K = rbf(L, L, gamma).astype(np.float64)
    w, V = np.linalg.eigh(K + 1e-6 * np.eye(len(L)))
    w = np.clip(w, 1e-8, None)
    return (V @ np.diag(w**-0.5) @ V.T).astype(np.float32)     # K_MM^{-1/2}


def median_gamma(X, n=2000, seed=0):
    rng = np.random.default_rng(seed); s = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    D2 = ((s[:, None, :] - s[None, :, :])**2).sum(-1)
    return 1.0 / (np.median(D2[D2 > 0]) + 1e-8)


def ranpac_b50(proj, Ftr, ytr, Fte, yte, order, base, inc, M):
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--Ms", type=int, nargs="+", default=[1024, 2048])
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = feats()
    D = Ftr.shape[1]; ncls = 100; base = 50; inc = 10
    order = np.random.default_rng(1993).permutation(ncls)
    base_cls = order[:base]; Fbase = Ftr[np.isin(ytr, base_cls)]
    gamma = median_gamma(Fbase)
    print(f"[nyspac] CIFAR100 B{base}+{inc}, gamma={gamma:.3f}, feats {Ftr.shape}", flush=True)

    Mmax = max(args.Ms); rng = np.random.default_rng(0)
    rays = ConicHull(n_rays=Mmax, use_pca=True, pca_dim=64, ray_diversity="hybrid").fit(Fbase).extreme_rays_.astype(np.float32)
    km = MiniBatchKMeans(Mmax, batch_size=4096, n_init=3, random_state=0).fit(Fbase)
    kmL = unit(km.cluster_centers_.astype(np.float32))
    print(f"[landmarks] rays {rays.shape[0]}  kmeans {kmL.shape[0]}", flush=True)

    res = {}
    for M in args.Ms:
        Wr = (rng.standard_normal((D, M)) / np.sqrt(D)).astype(np.float32)
        R, KM = rays[:M], kmL[:M]
        Cr = nystrom_correction(R, gamma); Ck = nystrom_correction(KM, gamma)
        variants = {
            "random-ReLU":        lambda X: np.maximum(X @ Wr, 0),
            "rbf-ray-nystrom":    lambda X: rbf(X, R, gamma) @ Cr,
            "rbf-kmeans-nystrom": lambda X: rbf(X, KM, gamma) @ Ck,
            "rbf-ray-raw":        lambda X: rbf(X, R, gamma),
        }
        res[M] = {}
        for name, proj in variants.items():
            a, l = ranpac_b50(proj, Ftr, ytr, Fte, yte, order, base, inc, M)
            res[M][name] = {"avg": a, "last": l}
            print(f"  [M={M} {name:20s}] avg {a:.1f}  last {l:.1f}", flush=True)
        base_ = res[M]["random-ReLU"]["avg"]
        print(f"    Δ ray-nystrom − RanPAC: {res[M]['rbf-ray-nystrom']['avg']-base_:+.1f}", flush=True)

    with open(os.path.join(OUT, "nyspac.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== NysPAC (CIFAR-100 B50, avg-inc acc) ===\n| M | random-ReLU | rbf-ray-nystrom | rbf-kmeans-nystrom | rbf-ray-raw |\n|--:|--:|--:|--:|--:|")
    for M, r in res.items():
        print(f"| {M} | {r['random-ReLU']['avg']:.1f} | {r['rbf-ray-nystrom']['avg']:.1f} | "
              f"{r['rbf-kmeans-nystrom']['avg']:.1f} | {r['rbf-ray-raw']['avg']:.1f} |")


if __name__ == "__main__":
    main()
