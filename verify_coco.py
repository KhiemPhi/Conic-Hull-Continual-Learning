"""Verify the COCO NNLS result: is the win real under a STRONG linear probe, and
is it the nonneg reconstruction (NNLS) or just locality (cosine-kNN)?"""
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
import coco_multilabel_law as M

DEVICE = M.DEVICE
BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
KEYS = ["1", "2", "3", "4", "5+"]


def strong_linear(Xtr, Ytr, Xte, C=1.0):
    S = np.full((len(Xte), Ytr.shape[1]), -1.0)
    for c in range(Ytr.shape[1]):
        y = Ytr[:, c]
        if 0 < y.sum() < len(y):
            lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            lr.fit(Xtr, y)
            S[:, c] = lr.decision_function(Xte)
    return S


def cosine_knn(Xtr, Ytr, Xte, k=100, chunk=512):
    A = torch.tensor(Xtr, device=DEVICE)
    S = np.zeros((len(Xte), Ytr.shape[1]), np.float32)
    Yt = torch.tensor(Ytr, device=DEVICE)
    for i in range(0, len(Xte), chunk):
        Q = torch.tensor(Xte[i:i + chunk], device=DEVICE)
        sims = Q @ A.T
        vals, idx = sims.topk(k, dim=1)                 # (b,k) cosine weights
        # weighted label vote: sum_j vals_j * Y[idx_j]
        S[i:i + chunk] = torch.einsum("bk,bkc->bc", vals,
                                      Yt[idx]).cpu().numpy()
    return S


def by_n(S, Yte, nlab):
    out = {}
    for (lo, hi), k in zip(BINS, KEYS):
        m = (nlab >= lo) & (nlab <= hi)
        out[k] = M.mAP(S[m], Yte[m]) if m.sum() > 5 else float("nan")
    return out


def main():
    ds = M.load_coco(); Y = M.build_label_matrix(ds)
    d = np.load(M.CACHE)
    n_lab = Y.sum(1).astype(int)
    rng = np.random.default_rng(0); perm = rng.permutation(len(Y))
    n_te = int(len(Y) * 0.4); te, tr = perm[:n_te], perm[n_te:]
    Ytr, Yte, nlab_te = Y[tr], Y[te], n_lab[te]

    for fname, raw in (("CLS", d["cls"]), ("Patch", d["patch"])):
        F = M._unit(raw.astype(np.float32)); Xtr, Xte = F[tr], F[te]
        print(f"\n===== {fname} =====")
        decs = {
            "linear(C=1)":  strong_linear(Xtr, Ytr, Xte, C=1.0),
            "linear(C=10)": strong_linear(Xtr, Ytr, Xte, C=10.0),
            "cos-knn":      cosine_knn(Xtr, Ytr, Xte, k=100),
            "nnls-cone":    M.knn_nnls(Xtr, Ytr, Xte, k=100),
            "ncm-multi":    M.ncm_multi(Xtr, Ytr, Xte, n_protos=4),
        }
        base = None
        for name, S in decs.items():
            ov = M.mAP(S, Yte); bn = by_n(S, Yte, nlab_te)
            if name == "linear(C=1)":
                base = bn; base_ov = ov
            dov = "" if base is None or name == "linear(C=1)" else f"  (Δ {ov-base_ov:+.3f})"
            print(f"  {name:13s} overall {ov:.3f}{dov} | " +
                  "  ".join(f"n{k}:{bn[k]:.3f}" for k in KEYS))
        # delta shape vs strong linear
        for name in ("nnls-cone", "cos-knn", "ncm-multi"):
            bn = by_n(decs[name], Yte, nlab_te)
            print(f"    Δ {name}-lin(C1): " +
                  "  ".join(f"n{k}:{bn[k]-base[k]:+.3f}" for k in KEYS))


if __name__ == "__main__":
    main()
