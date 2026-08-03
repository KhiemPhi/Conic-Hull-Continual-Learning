"""Robustness check for the COCO-counting cone signal: 5 seeds, TEST-OPTIMAL ridge
(alpha swept and best picked ON TEST — deliberately favours the baseline, making
the cone claim conservative). If nnls-cone still beats it across seeds, real niche."""
import numpy as np
from sklearn.linear_model import Ridge
import coco_multilabel_law as M
import coco_counting as CC

ALPHAS = [1, 10, 100, 1000]


def best_ridge_rho(Xtr, Ctr, Xte, Cte, nonneg=False):
    best = -1
    for a in ALPHAS:
        r = Ridge(alpha=a, positive=nonneg); r.fit(Xtr, Ctr)
        P = np.clip(r.predict(Xte), 0, None)
        rho, _ = CC.per_cat_spearman(P, Cte, present_min=1)
        best = max(best, rho)
    return best


def main():
    import sys, time
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    feats_arg = sys.argv[2] if len(sys.argv) > 2 else "CLS"
    print("[1/3] loading COCO dataset (labels only)...", flush=True)
    ds = M.load_coco()
    print(f"[1/3] dataset loaded: {len(ds)} rows", flush=True)
    Cnt = CC.build_count_matrix(ds)
    print(f"[2/3] loading cached CLIP features from {M.CACHE}...", flush=True)
    d = np.load(M.CACHE)
    print(f"[2/3] features loaded: cls {d['cls'].shape} patch {d['patch'].shape}",
          flush=True)
    all_feats = {"CLS": d["cls"], "Patch": d["patch"]}
    print(f"[3/3] scoring {n_seeds} seeds x {feats_arg} "
          "(nnls-cone is the slow part, ~30-60s/seed)...", flush=True)
    for fname in feats_arg.split(","):
        raw = all_feats[fname]
        F = M._unit(raw.astype(np.float32))
        cone, ridge_best, ridge_nn, knn = [], [], [], []
        for seed in range(n_seeds):
            _t = time.time()
            rng = np.random.default_rng(seed); perm = rng.permutation(len(Cnt))
            n_te = int(len(Cnt) * 0.4); te, tr = perm[:n_te], perm[n_te:]
            Xtr, Xte, Ctr, Cte = F[tr], F[te], Cnt[tr], Cnt[te]
            cone.append(CC.per_cat_spearman(
                CC.nnls_cone_count(Xtr, Ctr, Xte, k=100), Cte)[0])
            knn.append(CC.per_cat_spearman(
                CC.cos_knn_count(Xtr, Ctr, Xte, k=100), Cte)[0])
            ridge_best.append(best_ridge_rho(Xtr, Ctr, Xte, Cte, nonneg=False))
            ridge_nn.append(best_ridge_rho(Xtr, Ctr, Xte, Cte, nonneg=True))
            print(f"    [{fname} seed {seed}] cone {cone[-1]:.3f}  "
                  f"ridge {ridge_best[-1]:.3f}  ({time.time()-_t:.0f}s)", flush=True)
        cone, ridge_best = np.array(cone), np.array(ridge_best)
        knn, ridge_nn = np.array(knn), np.array(ridge_nn)
        print(f"\n== {fname} (5 seeds) ==")
        print(f"  nnls-cone       {cone.mean():.3f} ± {cone.std():.3f}")
        print(f"  ridge(test-opt) {ridge_best.mean():.3f} ± {ridge_best.std():.3f}")
        print(f"  nnls-ridge(opt) {ridge_nn.mean():.3f} ± {ridge_nn.std():.3f}")
        print(f"  cos-knn         {knn.mean():.3f} ± {knn.std():.3f}")
        dv = cone - ridge_best
        print(f"  Δ cone − ridge(test-opt): mean {dv.mean():+.3f} ± {dv.std():.3f} "
              f"| per-seed {np.round(dv,3)}")


if __name__ == "__main__":
    main()
