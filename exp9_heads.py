"""
exp9_heads.py — IS THE RanPAC HEAD THE BOTTLENECK?

Head quality is entirely separable from feature quality: fix the features, swap the head.
This is the cheapest measurement left (no backbone training when FEATS=frozen) and it probes
a component we have NEVER varied -- both our method and the CEILING use the same RanPAC head,
so improving it raises BOTH, i.e. it moves the absolute number the targets are stated in.

Known on IN21k:  frozen  NCM 0.5293 -> RanPAC 0.6902 (+16.1)
                 joint   NCM 0.8120 -> RanPAC 0.8498 (+3.8)
So the head is worth a lot, but is 0.8498 the best a head can do on those features?

Heads compared (identical features, identical train/test split):
  ncm          nearest class mean                              (1st-order only)
  ridge        RanPAC: RP -> ReLU -> (G+lam I)^-1 C            (the incumbent)
  ridge_bal    same, inverse-frequency class weighting         <- ImageNet-R is 51..430/class
                                                                  and C accumulates RAW counts,
                                                                  biasing toward frequent classes
  lda          shrunk LDA on the raw features                  (uses between/within scatter)
  fecam        per-class shrunk Mahalanobis                    (per-class 2nd order)
  probe        CE-trained linear head on the raw features      <- the OBJECTIVE test:
                                                                  ridge regresses to one-hot,
                                                                  which is not classification
  probe_rp     CE-trained linear head on ReLU(RP x)            (same map, better objective)

Reading it:
  probe >> ridge      -> the least-squares-to-one-hot objective is costing accuracy everywhere;
                         the ceiling moves and the targets get easier
  ridge_bal > ridge   -> free win, class imbalance was being ignored
  all within +-0.005  -> the head is NOT the bottleneck; stop thinking about it

Run:  python -u exp9_heads.py                 # frozen features, ~5 min, no training
      FEATS=joint python -u exp9_heads.py     # joint-adapted features (trains once, ~45 min)
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
FEATS = os.environ.get("FEATS", "frozen")        # frozen | joint
EPOCHS = int(os.environ.get("EPOCHS", 40))       # only for FEATS=joint
AUG = int(os.environ.get("AUG", 1))
M_RP = int(os.environ.get("M_RP", 10000))
LAMBDAS = [1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
PROBE_EPOCHS = int(os.environ.get("PROBE_EPOCHS", 60))
VAL_FRAC = 0.10

T0 = time.time()


def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
TF_EVAL = create_transform(**_cfg, is_training=False)
TF_TRAIN = (create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                             re_prob=0.25, scale=(0.7, 1.0), hflip=0.5) if AUG else TF_EVAL)


class HFWrap(Dataset):
    def __init__(self, ds, idx, labels, tf):
        self.ds, self.idx, self.labels, self.tf = ds, np.asarray(idx), np.asarray(labels), tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(self.labels[i])


from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"]); _cl = np.array(sorted(set(_w.tolist())))
_lab = np.searchsorted(_cl, _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
TRAIN_AUG = HFWrap(_ds, _p[:_n], _lab[_p[:_n]], TF_TRAIN)
TRAIN = HFWrap(_ds, _p[:_n], _lab[_p[:_n]], TF_EVAL); TR_Y = _lab[_p[:_n]]
TEST = HFWrap(_ds, _p[_n:], _lab[_p[_n:]], TF_EVAL); TE_Y = _lab[_p[_n:]]
N_CLS = len(_cl)
cnts = np.bincount(TR_Y, minlength=N_CLS)
log(f"[ImageNet-R] train {len(TR_Y)} test {len(TE_Y)} classes {N_CLS} | "
    f"per-class {cnts.min()}..{cnts.max()} (imbalance {cnts.max()/cnts.min():.1f}x)")


@torch.no_grad()
def extract(model, ds):
    model.eval()
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in DataLoader(ds, batch_size=256, num_workers=8, pin_memory=True):
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ------------------------------- features -------------------------------
if FEATS == "frozen":
    m = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV)
else:
    log(f"joint fine-tune ({EPOCHS} ep, aug={AUG}) -- the ceiling configuration")
    m = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(m)
    head = nn.Linear(768, N_CLS).to(DEV)
    opt = torch.optim.AdamW(list(get_lora_params(m)) + list(head.parameters()),
                            lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    ld = DataLoader(TRAIN_AUG, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    for ep in range(EPOCHS):
        m.train(); ok = tot = 0
        for x, y in ld:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(m(x).float()); loss = ce(lg, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ok += int((lg.argmax(1) == y).sum()); tot += len(y)
        sch.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            log(f"  joint ep {ep+1}/{EPOCHS} train {ok/tot:.4f}")
    del head

Ftr, Fte = extract(m, TRAIN), extract(m, TEST)
del m; torch.cuda.empty_cache()
log(f"features {Ftr.shape} / {Fte.shape}")

perm = np.random.default_rng(0).permutation(len(Ftr))
nval = int(VAL_FRAC * len(Ftr))
VI, TI = perm[:nval], perm[nval:]          # val split for lambda selection


# ------------------------------- heads -------------------------------
def h_ncm(Xtr, ytr, Xte):
    P = un(np.stack([un(Xtr[ytr == c]).mean(0) for c in range(N_CLS)]))
    return np.argmax(un(Xte) @ P.T, 1)


P_RP = torch.randn(768, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)


def rp(Z, bs=4096):
    out = []
    for i in range(0, len(Z), bs):
        out.append(torch.relu(torch.tensor(un(Z[i:i + bs]), device=DEV,
                                           dtype=torch.float32) @ P_RP))
    return torch.cat(out)


def h_ridge(Xtr, ytr, Xte, balanced=False):
    """RanPAC. balanced=True weights each sample by 1/n_class -- ImageNet-R is 51..430 per
    class and C otherwise accumulates raw counts, biasing the solution to frequent classes."""
    H = rp(Xtr[TI]).double(); y = ytr[TI]
    Y = torch.zeros(len(y), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(y)), torch.tensor(y, device=DEV)] = 1.0
    if balanced:
        w = torch.tensor((1.0 / np.maximum(cnts, 1))[y], device=DEV, dtype=torch.float64)
        w = w / w.mean()
        G = H.T @ (w[:, None] * H); C = H.T @ (w[:, None] * Y)
    else:
        G = H.T @ H; C = H.T @ Y
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    Hv, Ht = rp(Xtr[VI]).double(), rp(Xte).double()
    best, bp = -1, None
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, C)
        a = float(((Hv @ W).argmax(1).cpu().numpy() == ytr[VI]).mean())
        if a > best:
            best, bp = a, (Ht @ W).argmax(1).cpu().numpy()
    return bp


def h_ridge_nonneg(Xtr, ytr, Xte, iters=400, balanced=False):
    """CONE AS A HEAD. RanPAC solves W = (G+lam I)^-1 C with W UNCONSTRAINED. Constrain it to
    the non-negative orthant:   W* = argmin_{W>=0} ||HW - Y||^2 + lam||W||^2.
    Since h = ReLU(RP x) >= 0, each class score becomes a CONIC COMBINATION of non-negative
    features -- the class occupies a cone in feature space -- but the solve stays JOINT and
    DISCRIMINATIVE, which is what every prior (one-vs-rest, generative) cone head got wrong.

    It is the SIGN CONSTRAINT ON THE READOUT, not a hull fitted to data -- the same structure
    as the only cone-adjacent result that ever survived controls (SAE monosemanticity).
    Exact additivity is preserved: G and C are untouched, only the solve changes.

    Projected gradient on the ridge objective, warm-started from the unconstrained solution.
    Honest expectation: a MORE constrained discriminative solve usually costs accuracy (it
    removes 'class c is anti-correlated with feature j'); it can only win if non-negativity
    happens to regularise well in ReLU space."""
    H = rp(Xtr[TI]).double(); y = ytr[TI]
    Y = torch.zeros(len(y), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(y)), torch.tensor(y, device=DEV)] = 1.0
    if balanced:
        w = torch.tensor((1.0 / np.maximum(cnts, 1))[y], device=DEV, dtype=torch.float64)
        w = w / w.mean()
        G = H.T @ (w[:, None] * H); C = H.T @ (w[:, None] * Y)
    else:
        G = H.T @ H; C = H.T @ Y
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    Hv, Ht = rp(Xtr[VI]).double(), rp(Xte).double()
    best, bp = -1, None
    for lam in LAMBDAS:
        A = G + lam * eye
        W = torch.linalg.solve(A, C).clamp(min=0)          # warm start = projected ridge
        L = torch.linalg.eigvalsh(A)[-1]                    # Lipschitz const of the gradient
        for _ in range(iters):                              # projected gradient descent
            W = (W - (A @ W - C) / L).clamp(min=0)
        a = float(((Hv @ W).argmax(1).cpu().numpy() == ytr[VI]).mean())
        if a > best:
            best, bp = a, (Ht @ W).argmax(1).cpu().numpy()
    return bp


# ======================= CONE HEADS (2)-(9) =========================================
K_RAYS = int(os.environ.get("K_RAYS", 10))
_HULLS = {}


def hulls(Xtr, ytr):
    """Per-class extreme rays, shared by the hull-based heads (2)(3)(7)."""
    if _HULLS:
        return _HULLS["R"], _HULLS["own"]
    from conic_hull import ConicHull
    R, own = [], []
    for c in range(N_CLS):
        Xc = un(Xtr[TI][ytr[TI] == c])
        h = ConicHull(n_rays=int(min(K_RAYS, len(Xc))), use_pca=True,
                      pca_dim=int(min(64, max(len(Xc) - 1, 2)))).fit(Xc)
        R.append(h.extreme_rays_); own.append(np.full(len(h.extreme_rays_), c))
    _HULLS["R"] = torch.tensor(np.concatenate(R), device=DEV, dtype=torch.float32)
    _HULLS["own"] = torch.tensor(np.concatenate(own), device=DEV)
    log(f"    built {len(_HULLS['own'])} rays over {N_CLS} classes")
    return _HULLS["R"], _HULLS["own"]


def nnls_pg(D, Q, iters=200):
    """min_{a>=0} ||a D - q||^2 for every row q of Q. D:(m,d) atoms, Q:(n,d)."""
    G = D @ D.T
    L = torch.linalg.eigvalsh(G.double())[-1].float().clamp(min=1e-6)
    B = Q @ D.T
    A = torch.zeros(len(Q), len(D), device=DEV)
    for _ in range(iters):
        A = (A - (A @ G - B) / L).clamp(min=0)
    return A


def h_cone_collab(Xtr, ytr, Xte):
    """(2) COLLABORATIVE NNLS. One joint non-negative solve over the CONCATENATED dictionary
    of every class's rays, then assign by which class-block carries the most weight. Makes the
    cones COMPETITIVE instead of scored one-vs-rest -- the actual defect in argmax-of-membership.
    Still a RECONSTRUCTION objective, not a discriminative one."""
    R, own = hulls(Xtr, ytr)
    Q = torch.tensor(un(Xte), device=DEV, dtype=torch.float32)
    A = nnls_pg(R, Q)
    S = torch.zeros(len(Q), N_CLS, device=DEV)
    S.index_add_(1, own, A)                       # total weight assigned to each class block
    return S.argmax(1).cpu().numpy()


def h_cone_feats(Xtr, ytr, Xte):
    """(3) CONE AS FEATURE MAP, ridge as head. h(x) = relu(<x, rays>) (non-negative, sparse),
    then the ordinary decorrelated ridge on top. Coverage-as-BASIS rather than as a classifier."""
    R, _ = hulls(Xtr, ytr)

    def feat(Z, bs=4096):
        out = []
        for i in range(0, len(Z), bs):
            q = torch.tensor(un(Z[i:i + bs]), device=DEV, dtype=torch.float32)
            out.append(torch.relu(q @ R.T))
        return torch.cat(out).double()

    Ht, y = feat(Xtr[TI]), ytr[TI]
    Y = torch.zeros(len(y), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(y)), torch.tensor(y, device=DEV)] = 1.0
    G, C = Ht.T @ Ht, Ht.T @ Y
    eye = torch.eye(G.shape[0], device=DEV, dtype=torch.float64)
    Hv, Hte = feat(Xtr[VI]), feat(Xte)
    best, bp = -1, None
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, C)
        a = float(((Hv @ W).argmax(1).cpu().numpy() == ytr[VI]).mean())
        if a > best:
            best, bp = a, (Hte @ W).argmax(1).cpu().numpy()
    return bp


def _constrained_ridge(Xtr, ytr, Xte, project, iters=400, extra_grad=None):
    """Shared projected-gradient solver for the constrained-ridge family (1)(4)(5)(6).
    `project` maps W -> the feasible set; `extra_grad` adds a penalty gradient."""
    H = rp(Xtr[TI]).double(); y = ytr[TI]
    Y = torch.zeros(len(y), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(y)), torch.tensor(y, device=DEV)] = 1.0
    G, C = H.T @ H, H.T @ Y
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)
    Hv, Ht = rp(Xtr[VI]).double(), rp(Xte).double()
    best, bp = -1, None
    for lam in LAMBDAS:
        A = G + lam * eye
        L = torch.linalg.eigvalsh(A)[-1]
        W = project(torch.linalg.solve(A, C))
        for _ in range(iters):
            g = A @ W - C
            if extra_grad is not None:
                g = g + extra_grad(W)
            W = project(W - g / L)
        a = float(((Hv @ W).argmax(1).cpu().numpy() == ytr[VI]).mean())
        if a > best:
            best, bp = a, (Ht @ W).argmax(1).cpu().numpy()
    return bp


def _proj_simplex(W):
    """Column-wise projection onto the probability simplex (Duchi et al.)."""
    S, _ = torch.sort(W, dim=0, descending=True)
    cs = S.cumsum(0) - 1.0
    ind = torch.arange(1, W.shape[0] + 1, device=W.device, dtype=W.dtype).unsqueeze(1)
    cond = S - cs / ind > 0
    rho = cond.double().cumsum(0).argmax(0)
    theta = cs.gather(0, rho.unsqueeze(0)) / (rho + 1).double()
    return (W - theta).clamp(min=0)


def h_ridge_nn_lowrank(Xtr, ytr, Xte, r=64, iters=300):
    """(4) NON-NEGATIVE LOW-RANK head: W = U V with U,V >= 0 -- literally NMF of the classifier.
    Parts-based and far fewer parameters; the RANK constraint may regularise where the sign
    constraint alone only removes capacity."""
    H = rp(Xtr[TI]).double(); y = ytr[TI]
    Y = torch.zeros(len(y), N_CLS, device=DEV, dtype=torch.float64)
    Y[torch.arange(len(y)), torch.tensor(y, device=DEV)] = 1.0
    G, C = H.T @ H, H.T @ Y
    Hv, Ht = rp(Xtr[VI]).double(), rp(Xte).double()
    best, bp = -1, None
    for lam in [1e2, 1e3, 1e4]:
        A = G + lam * torch.eye(M_RP, device=DEV, dtype=torch.float64)
        W0 = torch.linalg.solve(A, C).clamp(min=1e-6)
        U = W0[:, torch.randperm(N_CLS, device=DEV)[:r]].clone().clamp(min=1e-6)
        V = torch.rand(r, N_CLS, device=DEV, dtype=torch.float64) + 1e-6
        for _ in range(iters):                     # multiplicative NMF-style updates
            V = V * ((U.T @ C) / (U.T @ A @ U @ V + 1e-12)).clamp(0.1, 10)
            U = U * ((C @ V.T) / (A @ U @ (V @ V.T) + 1e-12)).clamp(0.1, 10)
        W = U @ V
        a = float(((Hv @ W).argmax(1).cpu().numpy() == ytr[VI]).mean())
        if a > best:
            best, bp = a, (Ht @ W).argmax(1).cpu().numpy()
    return bp


def h_ridge_simplex(Xtr, ytr, Xte):
    """(5) SIMPLEX head: W_c >= 0 AND sums to 1 -- each class is a CONVEX combination of the
    random-feature dictionary. Scale-free, so no class wins by having larger weights."""
    return _constrained_ridge(Xtr, ytr, Xte, _proj_simplex)


def h_ridge_nn_excl(Xtr, ytr, Xte, mu=None):
    """(6) NON-NEGATIVE + CLASS-EXCLUSIVE: W >= 0 plus a penalty on <W_c, W_c'> for c != c',
    so classes claim DISJOINT feature sets. Conic + disjoint = each class owns a sub-cone,
    which is the geometric intuition that motivated hulls -- but kept discriminative."""
    mu = float(os.environ.get("EXCL_MU", 1.0)) if mu is None else mu

    def eg(W):
        S = W.T @ W
        S = S - torch.diag(torch.diagonal(S))       # off-diagonal only
        return 2.0 * mu * (W @ S)
    return _constrained_ridge(Xtr, ytr, Xte, lambda W: W.clamp(min=0), extra_grad=eg)


def h_cone_polar(Xtr, ytr, Xte):
    """(7) POLAR/DUAL-CONE score: sum_k relu(<x, r_ck>) -- total positive alignment with the
    class's rays, i.e. how far x is from that class's POLAR cone. Expected to be weak: with
    K << D the polar of a K-ray cone is enormous and uninformative (the same degeneracy that
    made hull membership vacuous)."""
    R, own = hulls(Xtr, ytr)
    Q = torch.tensor(un(Xte), device=DEV, dtype=torch.float32)
    S = torch.zeros(len(Q), N_CLS, device=DEV)
    S.index_add_(1, own, torch.relu(Q @ R.T))
    return S.argmax(1).cpu().numpy()


def h_cone_soc(Xtr, ytr, Xte):
    """(8) SECOND-ORDER (Lorentz) CONE per class: axis mu_c + aperture. Score = <x,mu_c> - b_c
    with b_c the class's calibrated cos-aperture. A true SOCP is ~1000x the cost and reverts to
    one-vs-rest; this is the same geometry at NCM cost, and the per-class bias is the point."""
    X, y = un(Xtr[TI]), ytr[TI]
    MU = un(np.stack([X[y == c].mean(0) for c in range(N_CLS)]))
    b = np.array([np.percentile(un(X[y == c]) @ MU[c], 10) for c in range(N_CLS)])
    return np.argmax(un(Xte) @ MU.T - b[None, :], 1)


def h_cone_copositive(Xtr, ytr, Xte, r=4, epochs=40):
    """(9) COPOSITIVE head (INNER APPROXIMATION -- exact copositive programming is NP-hard).
    Quadratic score on non-negative features: s_c(h) = h^T M_c h with M_c = L_c L_c^T + diag(n_c),
    L free and n_c >= 0. {PSD + nonneg} is the standard inner approximation of the copositive
    cone, and h = relu(.) >= 0 is exactly the orthant those matrices are copositive on.
    Trained by SGD (no closed form). Uses the raw 768-d features: M_RP^2 per class is infeasible."""
    A = torch.tensor(un(Xtr), device=DEV, dtype=torch.float32).relu_()
    B = torch.tensor(un(Xte), device=DEV, dtype=torch.float32).relu_()
    d = A.shape[1]
    L = nn.Parameter(torch.randn(N_CLS, d, r, device=DEV) * 0.02)
    n = nn.Parameter(torch.zeros(N_CLS, d, device=DEV))
    opt = torch.optim.AdamW([L, n], lr=1e-2, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    yt = torch.tensor(ytr, device=DEV); idx = torch.tensor(TI, device=DEV)

    def score(Z):
        q = torch.einsum('nd,cdr->ncr', Z, L)          # (n, C, r)
        return q.pow(2).sum(-1) + Z.pow(2) @ n.clamp(min=0).T
    for _ in range(epochs):
        p = idx[torch.randperm(len(idx), device=DEV)]
        for i in range(0, len(p), 512):
            b = p[i:i + 512]
            loss = ce(score(A[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.cat([score(B[i:i + 1024]).argmax(1)
                          for i in range(0, len(B), 1024)]).cpu().numpy()


def h_lda(Xtr, ytr, Xte, shrink=0.1):
    X = un(Xtr[TI]); y = ytr[TI]
    mus = np.stack([X[y == c].mean(0) for c in range(N_CLS)])
    Sw = np.zeros((768, 768))
    for c in range(N_CLS):
        d = X[y == c] - mus[c]; Sw += d.T @ d
    Sw /= len(X)
    Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / 768 * np.eye(768)
    P = np.linalg.solve(Sw, mus.T).T
    b = -0.5 * np.einsum('cd,cd->c', mus, P) + np.log(np.maximum(cnts, 1) / cnts.sum())
    return np.argmax(un(Xte) @ P.T + b, 1)


def h_fecam(Xtr, ytr, Xte, shrink=0.3):
    X = un(Xtr[TI]); y = ytr[TI]
    sc = []
    Q = torch.tensor(un(Xte), device=DEV, dtype=torch.float64)
    for c in range(N_CLS):
        Xc = X[y == c]; mu = Xc.mean(0); d = Xc - mu
        S = (d.T @ d) / max(len(Xc) - 1, 1)
        S = (1 - shrink) * S + shrink * np.trace(S) / 768 * np.eye(768)
        Pc = torch.tensor(np.linalg.inv(S), device=DEV, dtype=torch.float64)
        dq = Q - torch.tensor(mu, device=DEV, dtype=torch.float64)
        sc.append(-torch.einsum('nd,de,ne->n', dq, Pc, dq))
    return torch.stack(sc, 1).argmax(1).cpu().numpy()


def h_probe(Xtr, ytr, Xte, on_rp=False, epochs=PROBE_EPOCHS):
    """CE-trained linear head -- optimises the ACTUAL objective instead of least-squares
    to one-hot. CIL-invalid as a method (needs all data at once) but exactly right as a
    ceiling measurement: it prices the objective mismatch."""
    A = rp(Xtr).float() if on_rp else torch.tensor(un(Xtr), device=DEV)
    B = rp(Xte).float() if on_rp else torch.tensor(un(Xte), device=DEV)
    W = nn.Linear(A.shape[1], N_CLS).to(DEV)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    yt = torch.tensor(ytr, device=DEV)
    idx = torch.tensor(TI, device=DEV)
    for _ in range(epochs):
        p = idx[torch.randperm(len(idx), device=DEV)]
        for i in range(0, len(p), 1024):
            b = p[i:i + 1024]
            loss = ce(W(A[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad():
        return W(B).argmax(1).cpu().numpy()


HEADS = {
    "ncm":          lambda: h_ncm(Ftr, TR_Y, Fte),
    "ridge":        lambda: h_ridge(Ftr, TR_Y, Fte, balanced=False),
    "ridge_bal":    lambda: h_ridge(Ftr, TR_Y, Fte, balanced=True),
    # ---- CONE HEADS (1)-(9) ----
    "c1_nonneg":     lambda: h_ridge_nonneg(Ftr, TR_Y, Fte),      # non-negative ridge
    "c2_collab":     lambda: h_cone_collab(Ftr, TR_Y, Fte),       # joint NNLS over all rays
    "c3_conefeat":   lambda: h_cone_feats(Ftr, TR_Y, Fte),        # rays as basis + ridge
    "c4_nn_lowrank": lambda: h_ridge_nn_lowrank(Ftr, TR_Y, Fte),  # W = UV, U,V >= 0
    "c5_simplex":    lambda: h_ridge_simplex(Ftr, TR_Y, Fte),     # W >= 0, columns sum to 1
    "c6_nn_excl":    lambda: h_ridge_nn_excl(Ftr, TR_Y, Fte),     # W >= 0 + disjointness
    "c7_polar":      lambda: h_cone_polar(Ftr, TR_Y, Fte),        # dual/polar cone score
    "c8_soc":        lambda: h_cone_soc(Ftr, TR_Y, Fte),          # Lorentz cone / cap
    "c9_copositive": lambda: h_cone_copositive(Ftr, TR_Y, Fte),   # PSD+nonneg inner approx
    "lda":       lambda: h_lda(Ftr, TR_Y, Fte),
    "fecam":     lambda: h_fecam(Ftr, TR_Y, Fte),
    "probe":     lambda: h_probe(Ftr, TR_Y, Fte, on_rp=False),
    "probe_rp":  lambda: h_probe(Ftr, TR_Y, Fte, on_rp=True),
}
WANT = [h for h in os.environ.get("HEADS", ",".join(HEADS)).split(",") if h in HEADS]

res = {}
for h in WANT:
    t0 = time.time()
    try:
        res[h] = float((HEADS[h]() == TE_Y).mean())
        log(f"  {h:>10} {res[h]:.4f}   [{time.time()-t0:.0f}s]")
    except Exception as e:
        log(f"  {h:>10} FAILED: {type(e).__name__}: {str(e)[:90]}")

np.save(f"exp9_heads_{FEATS}_{MODEL.split('.')[-1]}.npy", res, allow_pickle=True)
print("\n" + "=" * 78)
print(f"EXP9 — head comparison on FIXED {FEATS} features ({MODEL})")
print("=" * 78)
base = res.get("ridge")
for h, a in sorted(res.items(), key=lambda kv: -kv[1]):
    d = f"{a-base:+.4f}" if base else "     -"
    print(f"{h:>12} {a:>8.4f}   vs ridge {d}")
print("-" * 78)
if base and "probe" in res:
    dp = res["probe"] - base
    print(f"OBJECTIVE test (probe - ridge): {dp:+.4f}")
    print("  >> +0.01  -> least-squares-to-one-hot is costing accuracy EVERYWHERE; a better")
    print("              head raises the ceiling too and the 0.82/0.84 targets get easier")
    print("  ~  0      -> the head is not the bottleneck; stop here")
if base and "ridge_bal" in res:
    print(f"IMBALANCE test (ridge_bal - ridge): {res['ridge_bal']-base:+.4f}  "
          f"(free if positive)")
cone_heads = {k: v for k, v in res.items() if k.startswith("c")}
if base and cone_heads:
    print("\nCONE HEADS vs ridge:")
    for k, v in sorted(cone_heads.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>15} {v:>8.4f}  {v-base:+.4f}")
    bk, bv = max(cone_heads.items(), key=lambda kv: kv[1])
    print(f"  best cone head: {bk} {bv:.4f}  vs ridge {bv-base:+.4f}")
    print("  NOTE (1)(4)(5)(6) are CONSTRAINED versions of the same discriminative solve, so")
    print("  they can only win by REGULARISING -- the unconstrained ridge puts ~50% of its")
    print("  mass on negative weights, so the constraint binds hard. (2)(3)(7)(8) are")
    print("  membership/basis scores, the family that has lost every prior head-to-head.")
print("=" * 78)
