"""
exp1_nonneg_native.py — DOES CONIC GEOMETRY NEED A NATIVELY NON-NEGATIVE SPACE?

The unifying hypothesis behind every cone result so far:

    cones fail when non-negativity is IMPOSED on a signed space,
    cones win when non-negativity is NATIVE to the representation.

Evidence it would explain: all failures (ViT/CLIP class features, transport, routing,
exemplar selection, projection basis) were on SIGNED features; the one ~10-sigma win
(SAE monosemanticity) was on codes that are non-negative BY CONSTRUCTION.

Test: build several representations OF THE SAME DATA that differ only in whether the space
is signed or non-negative, and run the identical cone-vs-multiprototype comparison on each.

  signed_raw     raw ViT / CLIP features                       (signed)
  signed_proj    X @ W                                          (signed)   <-- CONTROL
  nonneg_relu    ReLU(X @ W)                                    (non-negative)
                 ^ differs from signed_proj ONLY by the ReLU  => isolates non-negativity
                   from dimensionality / projection effects.  This is the key contrast.
  nonneg_sae     Top-K sparse autoencoder codes                 (non-negative, sparse)
  nonneg_nmf     NMF parts of shifted features                  (non-negative, parts-based)
  nonneg_abs     |X|                                            (non-negative but destroys
                 structure -- a placebo: non-negativity alone must NOT be sufficient)

Metric: cone - multiproto at MATCHED budget (K rays vs K k-means centroids).
Prediction: strongly negative on signed_*, and materially better (>=0) on nonneg_relu /
nonneg_sae / nonneg_nmf, with nonneg_abs staying bad.

Run:  python -u exp1_nonneg_native.py
      FEATS=clip python -u exp1_nonneg_native.py
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from conic_hull import ConicHull

T_START = time.time()


def log(msg):
    print(f"[{time.time()-T_START:7.1f}s] {msg}", flush=True)

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEATS = os.environ.get("FEATS", "vit")            # vit | clip
K_RAYS = int(os.environ.get("K_RAYS", 10))
N_CLS_USE = int(os.environ.get("N_CLS", 100))
PROJ_DIM = int(os.environ.get("PROJ_DIM", 1024))
SAE_DIM = int(os.environ.get("SAE_DIM", 1024))
SAE_TOPK = int(os.environ.get("SAE_TOPK", 32))
NMF_DIM = int(os.environ.get("NMF_DIM", 256))
# --- speed knobs (the run is dominated by 100 hulls x N_test NNLS solves per rep) ---
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 0))   # cap train samples/class (0 = all)
N_TEST = int(os.environ.get("N_TEST", 0))             # cap test samples      (0 = all)
ONLY = [r for r in os.environ.get("ONLY", "").split(",") if r]   # subset of reps to run

src = ("ranpac_out/cifar100_feats.npz" if FEATS == "vit" else "splice_out/cifar100_clip.npz")
z = np.load(src)
Xtr, ytr, Xte, yte = z["ftr"], z["ytr"], z["fte"], z["yte"]
keep = ytr < N_CLS_USE
Xtr, ytr = Xtr[keep], ytr[keep]
keep = yte < N_CLS_USE
Xte, yte = Xte[keep], yte[keep]
if N_PER_CLASS:
    sel = np.concatenate([np.where(ytr == c)[0][:N_PER_CLASS] for c in np.unique(ytr)])
    Xtr, ytr = Xtr[sel], ytr[sel]
if N_TEST and N_TEST < len(yte):
    sel = np.random.default_rng(0).choice(len(yte), N_TEST, replace=False)
    Xte, yte = Xte[sel], yte[sel]
log(f"[{FEATS}] {src}  train {Xtr.shape} test {Xte.shape} classes {N_CLS_USE}")
log(f"cost driver: {N_CLS_USE} hulls x {len(yte)} queries per representation")


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ------------------------- representations -------------------------
rng = np.random.default_rng(SEED)
W = rng.standard_normal((Xtr.shape[1], PROJ_DIM)).astype(np.float32) / np.sqrt(PROJ_DIM)


def sae_codes(Xtr, Xte, d=SAE_DIM, k=SAE_TOPK, epochs=8):
    """Minimal Top-K sparse autoencoder -> non-negative, sparse codes (the archetypal
    natively-non-negative space; this is where the one cone win lives)."""
    D = Xtr.shape[1]
    enc, dec = nn.Linear(D, d).to(DEV), nn.Linear(d, D, bias=False).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    Xt = torch.tensor(un(Xtr), device=DEV)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=DEV)
        tot = 0.0
        for i in range(0, len(Xt), 512):
            b = Xt[perm[i:i + 512]]
            h = torch.relu(enc(b))
            thr = h.topk(k, dim=1).values[:, -1:]           # Top-K sparsify
            h = h * (h >= thr)
            loss = ((dec(h) - b) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        print(f"    sae epoch {ep+1}/{epochs} recon {tot/(len(Xt)/512):.4f}", flush=True)

    def code(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                h = torch.relu(enc(torch.tensor(un(X[i:i + 2048]), device=DEV)))
                thr = h.topk(k, dim=1).values[:, -1:]
                out.append((h * (h >= thr)).cpu().numpy())
        return np.concatenate(out)
    return code(Xtr), code(Xte)


log("=== building paired representations of the SAME data ===")
WANT = ONLY or ["signed_raw", "signed_proj", "nonneg_relu", "nonneg_abs",
                "nonneg_sae", "nonneg_nmf"]
log(f"  representations to build: {WANT}")
REPS = {}
if "signed_raw" in WANT:
    REPS["signed_raw"] = (Xtr, Xte, "signed"); log("  signed_raw   built")
if "signed_proj" in WANT:
    REPS["signed_proj"] = (Xtr @ W, Xte @ W, "signed"); log("  signed_proj  built")
if "nonneg_relu" in WANT:
    REPS["nonneg_relu"] = (np.maximum(Xtr @ W, 0), np.maximum(Xte @ W, 0), "non-negative")
    log("  nonneg_relu  built")
if "nonneg_abs" in WANT:
    REPS["nonneg_abs"] = (np.abs(Xtr), np.abs(Xte), "non-negative (placebo)")
    log("  nonneg_abs   built")
if "nonneg_sae" in WANT:
    log(f"  nonneg_sae   training Top-{SAE_TOPK} SAE (d={SAE_DIM}) ...")
    sa, sb = sae_codes(Xtr, Xte)
    REPS["nonneg_sae"] = (sa, sb, "non-negative")
    log(f"  nonneg_sae   built  (mean nnz/sample {(sa>0).sum(1).mean():.1f})")
if "nonneg_nmf" in WANT:
    # NMF on the full 50k x 768 matrix is the slowest build step by far.
    log(f"  nonneg_nmf   fitting NMF(k={NMF_DIM}) on {Xtr.shape} — slowest build step ...")
    shift = Xtr.min()
    nm = NMF(n_components=NMF_DIM, init="nndsvd", max_iter=200, tol=1e-3)
    ntr = nm.fit_transform(Xtr - shift)
    log(f"  nonneg_nmf   fit done ({nm.n_iter_} iters), transforming test ...")
    REPS["nonneg_nmf"] = (ntr, nm.transform(Xte - shift), "non-negative")
    log("  nonneg_nmf   built")


# ------------------------- classifiers at matched budget -------------------------
KEYS = ["cosine", "geo_residual", "angular_margin", "max_ray_sim", "blended"]


def evaluate(Ztr, ytr, Zte, yte, K=K_RAYS, tag=""):
    cls = np.unique(ytr)
    t0 = time.time()
    ncm, multi, cone = [], [], []
    for i, c in enumerate(cls):
        Xc = un(Ztr[ytr == c]); n = len(Xc)
        ncm.append(un(Xc.mean(0, keepdims=True))[0])
        k = int(min(K, max(n // 4, 1)))
        multi.append(un(KMeans(n_clusters=k, n_init=4, random_state=0)
                        .fit(Xc).cluster_centers_))
        cone.append(ConicHull(n_rays=int(min(K, n)), use_pca=True,
                              pca_dim=int(min(64, max(n - 1, 2)))).fit(Xc))
        if (i + 1) % 10 == 0 or i == len(cls) - 1:
            el = time.time() - t0
            print(f"      {tag}fit {i+1:>3}/{len(cls)} classes  {el:6.1f}s "
                  f"(eta {el/(i+1)*(len(cls)-i-1):5.1f}s)", flush=True)

    Q = un(Zte)
    a_ncm = float((cls[np.argmax(Q @ np.stack(ncm).T, 1)] == yte).mean())
    a_mp = float((cls[np.argmax(np.stack([np.max(Q @ m.T, 1) for m in multi], 1), 1)]
                  == yte).mean())
    print(f"      {tag}NCM {a_ncm:.4f} | MP {a_mp:.4f}  -> scoring {len(cone)} hulls "
          f"x {len(Q)} queries", flush=True)

    # ONE score_all per hull: it already returns every key. (Calling it per key was a
    # 5x waste -- each call runs a fresh NNLS solve over all queries.)
    t1 = time.time()
    per_hull = []
    for i, h in enumerate(cone):
        sa = h.score_all(Q)
        per_hull.append({k: sa[k] for k in KEYS})
        if (i + 1) % 10 == 0 or i == len(cone) - 1:
            el = time.time() - t1
            print(f"      {tag}score {i+1:>3}/{len(cone)} hulls  {el:6.1f}s "
                  f"(eta {el/(i+1)*(len(cone)-i-1):5.1f}s)", flush=True)

    best_key, a_cone, per_key = None, -1.0, {}
    for key in KEYS:
        S = np.stack([ph[key] for ph in per_hull], 1)
        acc = float((cls[S.argmax(1)] == yte).mean())
        per_key[key] = acc
        if acc > a_cone:
            a_cone, best_key = acc, key
    print(f"      {tag}cone per-key: "
          + "  ".join(f"{k}={per_key[k]:.4f}" for k in KEYS), flush=True)
    return a_ncm, a_mp, a_cone, best_key, per_key


log(f"=== cone vs multi-prototype at matched budget K={K_RAYS} ===")
rows = []
for ri, (name, (Ztr, Zte, kind)) in enumerate(REPS.items()):
    log(f"--- [{ri+1}/{len(REPS)}] {name}  ({kind}, dim={Ztr.shape[1]}) ---")
    a_ncm, a_mp, a_cone, bk, pk = evaluate(Ztr, ytr, Zte, yte, tag=f"{name}: ")
    frac_pos = float((Ztr > 0).mean())
    rows.append(dict(rep=name, kind=kind, dim=Ztr.shape[1], frac_pos=frac_pos,
                     ncm=a_ncm, mp=a_mp, cone=a_cone, best=bk))
    log(f"  DONE {name:>13} [{kind:>22}] dim={Ztr.shape[1]:>5} pos={frac_pos:.2f} | "
        f"NCM {a_ncm:.4f} MP {a_mp:.4f} CONE {a_cone:.4f} ({bk}) | "
        f"cone-MP {a_cone-a_mp:+.4f}")

np.save(f"exp1_results_{FEATS}.npy", rows, allow_pickle=True)
print("\n" + "=" * 96)
print(f"EXP1 — native vs imposed non-negativity ({FEATS}, CIFAR-100, K={K_RAYS})")
print("=" * 96)
print(f"{'representation':>14} {'kind':>22} {'frac>0':>7} {'cone-MP':>9} {'cone-NCM':>9}")
for r in rows:
    print(f"{r['rep']:>14} {r['kind']:>22} {r['frac_pos']:>7.2f} "
          f"{r['cone']-r['mp']:>+9.4f} {r['cone']-r['ncm']:>+9.4f}")
print("-" * 96)
g = {r["rep"]: r for r in rows}
gap = {k: v["cone"] - v["mp"] for k, v in g.items()}
print("THE KEY CONTRAST (identical projection, differ only by the ReLU):")
if "signed_proj" in gap and "nonneg_relu" in gap:
    print(f"  signed_proj  cone-MP = {gap['signed_proj']:+.4f}")
    print(f"  nonneg_relu  cone-MP = {gap['nonneg_relu']:+.4f}")
    print(f"  => non-negativity alone is worth "
          f"{gap['nonneg_relu']-gap['signed_proj']:+.4f}")
else:
    print("  (need both signed_proj and nonneg_relu — rerun without ONLY=)")
print("\nPLACEBO CHECK: nonneg_abs must stay BAD, else 'any non-negativity' explains it:")
if "nonneg_abs" in gap:
    print(f"  nonneg_abs   cone-MP = {gap['nonneg_abs']:+.4f}")
print("\nHYPOTHESIS SUPPORTED IF: signed_* strongly negative, nonneg_relu/sae/nmf >= 0,")
print("and the placebo stays negative. That is a mechanism-backed law, and it turns every")
print("prior cone negative into supporting evidence.")
print("=" * 96)
