import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

EPS = 1e-9

def unit(X):
    """Row-normalize to the unit sphere. Angles only — we quotient out magnitude."""
    X = np.atleast_2d(X)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)

def angle_deg(a, b):
    c = np.clip(np.dot(unit(a).ravel(), unit(b).ravel()), -1, 1)
    return np.degrees(np.arccos(c))

# ---------- CONE ----------
def spa(X, m):
    """Successive Projection Algorithm: pick m anchor rows that conically generate
    the rest. This is the separable-NMF / X-RAY anchor finder — generators ARE data."""
    R = X.astype(float).copy()
    idx = []
    for _ in range(m):
        j = int(np.argmax((R ** 2).sum(1)))
        idx.append(j)
        u = R[j] / (np.linalg.norm(R[j]) + EPS)
        R = R - np.outer(R @ u, u)          # deflate: project onto complement of u
    return idx

def cone_fit(X, m):
    """Return generators W (d, m) as extreme-ray data points."""
    X = unit(X)
    return X[spa(X, m)].T

def cone_residual_deg(W, X):
    """Angle from each x to its NNLS projection onto Conic(W). 90° = orthogonal/outside."""
    X = unit(X)
    out = np.empty(len(X))
    for i, x in enumerate(X):
        a, _ = nnls(W, x)
        p = W @ a
        np_ = np.linalg.norm(p)
        out[i] = 90.0 if np_ < EPS else np.degrees(
            np.arccos(np.clip(np.dot(x, p) / np_, -1, 1)))
    return out

# ---------- vMF MIXTURE (the strong adversary) ----------
def spherical_kmeans(X, m, iters=100, seed=0):
    """m directional prototypes. Matched to the cone: m means (m*d params) vs
    m generators (m*d params). We even let vMF keep its extra kappa/weights for free."""
    X = unit(X); rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), m, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(m):
            pts = X[assign == j]
            if len(pts):
                C[j] = pts.sum(0) / (np.linalg.norm(pts.sum(0)) + EPS)
    return C

def vmf_distance_deg(C, X):
    """Angle to the NEAREST prototype. This is the fair centroid analogue of the
    cone residual: no convex fill between prototypes — that's the whole point."""
    X = unit(X)
    cos = np.clip(X @ unit(C).T, -1, 1)
    return np.degrees(np.arccos(cos.max(1)))

def auroc(dist_pos, dist_ood):
    """Higher distance => more OOD. pos = should-be-inside, ood = should-be-outside."""
    y = np.r_[np.zeros(len(dist_pos)), np.ones(len(dist_ood))]
    return roc_auc_score(y, np.r_[dist_pos, dist_ood])
