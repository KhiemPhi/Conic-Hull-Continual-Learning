"""
ranpac_conic.py
---------------
Can the conic hull AID RanPAC (not replace its classifier)?

Aid A — extreme-ray projection vs random projection.
  RanPAC lifts frozen features by h=ReLU(Wx) with RANDOM W, then a Gram-decorrelated
  prototype classifier. Swap W for the data's SPA EXTREME RAYS (data-adaptive, training-
  free, expandable), keep the decorrelation + forgetting-free accumulation. Sweep dim M;
  does extreme-ray projection match/beat random, especially at LOW M (efficiency)?

Aid B — NNLS coverage residual as an OPEN-SET reject (a capability RanPAC lacks).
  Seen = 50 classes (build RanPAC + per-class ConicHull); novel = 50 held-out classes.
  Reject-score = hull coverage residual; compare AUROC vs RanPAC max-score & NCM-distance.

Frozen ViT-B/16 features on CIFAR-100 (cached). Aid A uses 10-task CIL avg-inc accuracy.

    HF_HUB_OFFLINE=1 python -u ranpac_conic.py
"""
import os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import timm
from conic_hull import ConicHull

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./ranpac_out"
MODEL = "vit_base_patch16_224"
LAM = 1e3


@torch.no_grad()
def extract_all():
    path = os.path.join(OUT, "cifar100_feats.npz")
    if os.path.exists(path):
        d = np.load(path); return d["ftr"], d["ytr"], d["fte"], d["yte"]
    cfg = timm.data.resolve_data_config({}, model=timm.create_model(MODEL))
    tf = timm.data.create_transform(**cfg, is_training=False)
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(DEVICE).eval()
    out = {}
    for split, tr in (("tr", True), ("te", False)):
        ds = CIFAR100("./data", train=tr, download=False, transform=tf)
        F, Y = [], []
        for x, y in DataLoader(ds, batch_size=256, num_workers=8, pin_memory=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                F.append(model(x.to(DEVICE)).float().cpu().numpy())
            Y.append(np.asarray(y))
        out[f"f{split}"] = np.concatenate(F).astype(np.float32); out[f"y{split}"] = np.concatenate(Y)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, ftr=out["ftr"], ytr=out["ytr"], fte=out["fte"], yte=out["yte"])
    return out["ftr"], out["ytr"], out["fte"], out["yte"]


def ranpac_cil(proj, Ftr, ytr, Fte, yte, order, tasks, M):
    """Forgetting-free RanPAC accumulation with a given projection. Returns avg-inc, last."""
    tsz = len(order) // tasks
    G = np.zeros((M, M), np.float32); csum, ccount = {}, {}; seen = []; accs = []
    for t in range(tasks):
        cls = order[t*tsz:(t+1)*tsz]; seen += list(cls)
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


def aid_A(Ftr, ytr, Fte, yte):
    D = Ftr.shape[1]
    order = np.random.default_rng(1993).permutation(100)
    rng = np.random.default_rng(0)
    # fit SPA extreme rays ONCE at max M, slice for smaller M (SPA orders by extremity)
    Mmax = 2048
    sub = rng.choice(len(Ftr), 20000, replace=False)
    ch = ConicHull(n_rays=Mmax, use_pca=True, pca_dim=64, ray_diversity="hybrid")
    ch.fit(Ftr[sub]); rays = ch.extreme_rays_.astype(np.float32)     # (Mmax, D)
    print(f"[aidA] fit {rays.shape[0]} extreme rays", flush=True)
    res = {}
    for M in (256, 512, 1024, 2048):
        Wr = (rng.standard_normal((D, M)) / np.sqrt(D)).astype(np.float32)
        rnd = ranpac_cil(lambda X: np.maximum(X @ Wr, 0), Ftr, ytr, Fte, yte, order, 10, M)
        R = rays[:M]
        ray = ranpac_cil(lambda X: np.maximum(X @ R.T, 0), Ftr, ytr, Fte, yte, order, 10, M)
        res[M] = {"random_avg": rnd[0], "random_last": rnd[1],
                  "extremeray_avg": ray[0], "extremeray_last": ray[1]}
        print(f"  [aidA M={M:>4}] random avg {rnd[0]:.1f} | extreme-ray avg {ray[0]:.1f} "
              f"(Δ {ray[0]-rnd[0]:+.1f})", flush=True)
    return res


def aid_B(Ftr, ytr, Fte, yte, M=4096):
    order = np.random.default_rng(7).permutation(100)
    seen = order[:50];
    tr_m = np.isin(ytr, seen)
    Ftr_s, ytr_s = Ftr[tr_m], ytr[tr_m]
    # RanPAC (random) on seen
    D = Ftr.shape[1]; rng = np.random.default_rng(0)
    W = (rng.standard_normal((D, M)) / np.sqrt(D)).astype(np.float32)
    proj = lambda X: np.maximum(X @ W, 0)
    phi = proj(Ftr_s); G = phi.T @ phi
    Ginv = np.linalg.inv(G + LAM * np.eye(M, dtype=np.float32))
    labels = sorted(seen.tolist())
    protoR = np.stack([Ginv @ (phi[ytr_s == c].mean(0)) for c in labels])
    # NCM prototypes (raw feature space)
    Fn = Ftr_s / (np.linalg.norm(Ftr_s, axis=1, keepdims=True) + 1e-8)
    ncm = np.stack([Fn[ytr_s == c].mean(0) for c in labels])
    ncm /= (np.linalg.norm(ncm, axis=1, keepdims=True) + 1e-8)
    # per-seen-class ConicHull
    cones = []
    for c in labels:
        Xc = Fn[ytr_s == c]; k = min(24, len(Xc))
        h = ConicHull(n_rays=k, use_pca=False, ray_diversity="hybrid")
        h.fit(Xc) if len(Xc) >= 12 else setattr(h, "extreme_rays_", Xc)
        cones.append(h)

    # test: all classes; novel = not in seen
    novel = ~np.isin(yte, seen)
    y = novel.astype(int)                       # 1 = should be rejected
    Fte_n = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    # reject scores (higher = more novel)
    rej = {}
    rej["RanPAC-maxscore"] = -(proj(Fte) @ protoR.T).max(1)
    rej["NCM-dist"] = -(torch.tensor(Fte_n, device=DEVICE) @ torch.tensor(ncm, device=DEVICE).T).max(1).values.cpu().numpy()
    best = np.full(len(Fte), -np.inf, np.float32)
    for h in cones:
        best = np.maximum(best, h.score_nnls_residual(Fte_n))   # idness (higher=in-dist)
    rej["hull-residual"] = -best
    res = {k: float(roc_auc_score(y, v)) for k, v in rej.items()}
    print("[aidB] open-set AUROC (detect novel): " +
          "  ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = extract_all()
    print(f"[data] train {Ftr.shape} test {Fte.shape}", flush=True)
    out = {"aidA": aid_A(Ftr, ytr, Fte, yte), "aidB": aid_B(Ftr, ytr, Fte, yte)}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== Aid A (avg-inc acc vs M) ===\n| M | random | extreme-ray | Δ |\n|--:|--:|--:|--:|")
    for M, r in out["aidA"].items():
        print(f"| {M} | {r['random_avg']:.1f} | {r['extremeray_avg']:.1f} | "
              f"{r['extremeray_avg']-r['random_avg']:+.1f} |")
    print("\n=== Aid B (open-set AUROC) ===")
    for k, v in out["aidB"].items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
