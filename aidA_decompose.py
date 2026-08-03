"""
aidA_decompose.py
-----------------
WHAT made Aid A good? Its extreme-ray projection tangles three things: (a) data-adaptive
directions, (b) denoised into a PCA subspace, (c) EXTREME/corner directions. This isolates
which one drives the low-M win over random, by swapping ONLY the projection basis W in
RanPAC's exact pipeline (ranpac_b50), all bases fit on the BASE session (CIL-valid), frozen.

Projection bases (W: D x M ; proj = ReLU(X @ W)):
  random     : Gaussian / sqrt(D)                         (RanPAC baseline)
  datasample : M random base feature vectors (unit)       (data-adaptive, TYPICAL points)
  kmeans     : M k-means centroids of base feats (unit)   (data-adaptive, CENTRAL)
  pca        : top principal components, +/- signs        (variance / DENOISING directions)
  extremeray : SPA extreme rays (Aid A)                    (data-adaptive, EXTREME corners)

Attribution:
  pca ~= extremeray > random          -> win is DENOISING/variance-subspace (cone not special)
  kmeans/datasample ~= extremeray     -> win is DATA-ADAPTIVITY (cone not special)
  extremeray > all others at low M    -> win is EXTREMENESS  => NOVEL cone-central finding

    HF_HUB_OFFLINE=1 python -u aidA_decompose.py --dataset CIFAR100
"""
import argparse, os, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from ranpac_conic2 import ranpac_b50, feats_cifar, feats_imagenetr, fit_rays

OUT = "./ranpac_out"
BASES = ["random", "datasample", "kmeans", "pca", "extremeray"]


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def basis_random(D, M, rng):
    return (rng.standard_normal((D, M)) / np.sqrt(D)).astype(np.float32)


def basis_datasample(Fbase, M, rng):
    idx = rng.choice(len(Fbase), M, replace=len(Fbase) < M)
    return unit(Fbase[idx]).T.astype(np.float32)


def basis_kmeans(Fbase, M, seed):
    km = MiniBatchKMeans(n_clusters=M, random_state=seed, n_init=3,
                         batch_size=2048, max_iter=200).fit(unit(Fbase))
    return unit(km.cluster_centers_).T.astype(np.float32)


def basis_pca(Fbase, M):
    k = min((M + 1) // 2, Fbase.shape[1], len(Fbase))
    comps = PCA(n_components=k).fit(unit(Fbase)).components_           # (k,D), unit
    W = np.concatenate([comps, -comps], 0)[:M]                        # +/- to fill M
    return unit(W).T.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR100", choices=["CIFAR100", "ImageNet-R"])
    ap.add_argument("--Ms", type=int, nargs="+", default=[128, 256, 512, 1024])
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = feats_cifar() if args.dataset == "CIFAR100" else feats_imagenetr()
    D = Ftr.shape[1]; ncls = int(max(ytr.max(), yte.max()) + 1)
    base = ncls // 2; inc = max(10, ncls // 10)
    order = np.random.default_rng(1993).permutation(ncls)             # SAME order as Aid A
    base_cls = order[:base]; Fbase = Ftr[np.isin(ytr, base_cls)]
    print(f"[data] {args.dataset} base {Fbase.shape} classes {ncls} | B{base}+{inc}-incs", flush=True)

    Mmax = max(args.Ms); rng = np.random.default_rng(0)
    rays = fit_rays(Fbase, Mmax)                                      # base-only, CIL-valid
    print(f"[fit] extreme rays {rays.shape[0]} (want up to {Mmax})", flush=True)
    # precompute the expensive data-adaptive bases at Mmax where possible
    W_cache = {}

    def make_W(basis, M):
        if basis == "random":     return basis_random(D, M, np.random.default_rng(0))
        if basis == "datasample": return basis_datasample(Fbase, M, np.random.default_rng(1))
        if basis == "kmeans":     return basis_kmeans(Fbase, M, 2)
        if basis == "pca":        return basis_pca(Fbase, M)
        if basis == "extremeray": return rays[:M].T.astype(np.float32)

    res = {}
    for M in args.Ms:
        res[M] = {}
        for basis in BASES:
            W = make_W(basis, M)
            if W.shape[1] < M:
                print(f"    [warn] {basis} gave {W.shape[1]}<{M} dirs", flush=True)
            avg, _ = ranpac_b50(lambda X: np.maximum(X @ W, 0),
                                Ftr, ytr, Fte, yte, order, base, inc, W.shape[1])
            res[M][basis] = avg
        r = res[M]
        deltas = "  ".join(f"{b}:{r[b]:.1f}({r[b]-r['random']:+.1f})"
                           for b in BASES if b != "random")
        print(f"  [M={M:>4}] random:{r['random']:.1f}  {deltas}", flush=True)

    with open(os.path.join(OUT, f"aidA_decompose_{args.dataset}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== {args.dataset} avg-inc acc by projection basis (Δ vs random) ===")
    print("  M    | " + "  ".join(f"{b:>10s}" for b in BASES))
    for M in args.Ms:
        print(f"  {M:<4d} | " + "  ".join(f"{res[M][b]:10.1f}" for b in BASES))
    print("\n  Δ vs random:")
    print("  M    | " + "  ".join(f"{b:>10s}" for b in BASES if b != "random"))
    for M in args.Ms:
        print(f"  {M:<4d} | " + "  ".join(f"{res[M][b]-res[M]['random']:+10.1f}"
                                          for b in BASES if b != "random"))


if __name__ == "__main__":
    main()
