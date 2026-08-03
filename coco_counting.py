"""
coco_counting.py
----------------
Iteration toward a cone niche: use the conic hull as a CODE, not a scorer.

Task = per-category instance COUNTING on COCO (an unbounded, non-negative,
ADDITIVE target — the cone's native output type; softmax's simplex and prototype
assignment structurally cannot represent it).

Decoders predict a per-category count from a CLIP embedding:
  * ridge      : linear regression (unconstrained) — strong baseline
  * nnls-ridge : NON-NEGATIVE linear regression (fair: baseline also non-neg)
  * cos-knn    : locality control — weighted-mean of neighbours' counts
  * nnls-cone  : the conic code — x ≈ Σ aᵢ·atomᵢ (aᵢ≥0, k-NN); count_c = Σ aᵢ·Count[i,c]
                 (counts propagate ADDITIVELY through the nonneg conic coefficients)

Metric = per-category Spearman ρ between predicted and true count, over test
images where the category is PRESENT (count≥1) — pure "given it's here, how many".
Scale-free, so all decoders are compared fairly. Stratified by true count to test
whether the cone's additive-code edge GROWS with quantity.

Uses cached CLIP features from coco_multilabel_law.py.

    HF_HUB_OFFLINE=1 python -u coco_counting.py
"""
import json
import os

import numpy as np
import torch
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import coco_multilabel_law as M

OUT_DIR = "./coco_count_out"
DEVICE = M.DEVICE
MIN_PRESENT = 20            # categories need >=20 present test imgs to score


def build_count_matrix(ds):
    # Access ONLY the 'objects' column — row-wise iteration (`for r in ds`) would
    # decode every image (slow hang); column access is lazy and skips the Image.
    print(f"[count] reading 'objects' column for {len(ds)} rows "
          "(no image decode)...", flush=True)
    objs = ds["objects"]
    cats = [list(o["category"]) for o in objs]
    n_classes = max((max(c) + 1 for c in cats if c), default=0)
    Cnt = np.zeros((len(cats), n_classes), np.float32)
    for i, c in enumerate(cats):
        for k in c:
            Cnt[i, int(k)] += 1.0
    print(f"[count] built count matrix {Cnt.shape}", flush=True)
    return Cnt


def ridge(Xtr, Ctr, Xte, nonneg=False):
    r = Ridge(alpha=10.0, positive=nonneg)
    r.fit(Xtr, Ctr)
    return np.clip(r.predict(Xte), 0, None)


def cos_knn_count(Xtr, Ctr, Xte, k=100, chunk=512):
    A = torch.tensor(Xtr, device=DEVICE)
    Ct = torch.tensor(Ctr, device=DEVICE)
    out = np.zeros((len(Xte), Ctr.shape[1]), np.float32)
    for i in range(0, len(Xte), chunk):
        Q = torch.tensor(Xte[i:i + chunk], device=DEVICE)
        vals, idx = (Q @ A.T).topk(k, dim=1)
        w = vals.clamp_min(0)
        num = torch.einsum("bk,bkc->bc", w, Ct[idx])
        out[i:i + chunk] = (num / (w.sum(1, keepdim=True) + 1e-8)).cpu().numpy()
    return out


def nnls_cone_count(Xtr, Ctr, Xte, k=100, chunk=512):
    A = torch.tensor(Xtr, device=DEVICE)
    out = np.zeros((len(Xte), Ctr.shape[1]), np.float32)
    for i in range(0, len(Xte), chunk):
        Q = torch.tensor(Xte[i:i + chunk], device=DEVICE)
        idx = (Q @ A.T).topk(k, dim=1).indices.cpu().numpy()
        Qn = Xte[i:i + chunk]
        for j in range(len(Qn)):
            nn_idx = idx[j]
            a, _ = nnls(Xtr[nn_idx].T, Qn[j])
            out[i + j] = a @ Ctr[nn_idx]          # additive count propagation
    return out


def per_cat_spearman(P, Cnt, present_min=1):
    """Mean Spearman over categories, on imgs where true count>=present_min."""
    rs = []
    for c in range(Cnt.shape[1]):
        m = Cnt[:, c] >= present_min
        if m.sum() >= MIN_PRESENT and len(np.unique(Cnt[m, c])) > 1:
            rho = spearmanr(P[m, c], Cnt[m, c]).correlation
            if not np.isnan(rho):
                rs.append(rho)
    return float(np.mean(rs)) if rs else float("nan"), len(rs)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = M.load_coco()
    Cnt = build_count_matrix(ds)
    d = np.load(M.CACHE)
    tot = Cnt.sum(1)
    print(f"[coco] {len(Cnt)} imgs, {Cnt.shape[1]} cats | instances/img mean "
          f"{tot.mean():.2f} max {int(tot.max())} | max single-cat count "
          f"{int(Cnt.max())}")

    rng = np.random.default_rng(0); perm = rng.permutation(len(Cnt))
    n_te = int(len(Cnt) * 0.4); te, tr = perm[:n_te], perm[n_te:]
    Ctr, Cte = Cnt[tr], Cnt[te]

    results = {}
    for fname, raw in (("CLS", d["cls"]), ("Patch", d["patch"])):
        F = M._unit(raw.astype(np.float32)); Xtr, Xte = F[tr], F[te]
        decs = {
            "ridge":      ridge(Xtr, Ctr, Xte, nonneg=False),
            "nnls-ridge": ridge(Xtr, Ctr, Xte, nonneg=True),
            "cos-knn":    cos_knn_count(Xtr, Ctr, Xte, k=100),
            "nnls-cone":  nnls_cone_count(Xtr, Ctr, Xte, k=100),
        }
        results[fname] = {}
        print(f"\n===== {fname} =====")
        for name, P in decs.items():
            overall, ncat = per_cat_spearman(P, Cte, present_min=1)
            # stratify: among present imgs, does edge grow with count? use count>=t
            by_t = {}
            for t in (1, 2, 3):
                rho, _ = per_cat_spearman(P, Cte, present_min=t)
                by_t[f">={t}"] = rho
            results[fname][name] = {"spearman_present": overall, "n_cats": ncat,
                                    "by_min_count": by_t}
            print(f"  {name:11s} ρ(count|present) {overall:.3f} ({ncat} cats) | "
                  + "  ".join(f"{k}:{v:.3f}" for k, v in by_t.items()))
        base = results[fname]["ridge"]["spearman_present"]
        nnb = results[fname]["nnls-ridge"]["spearman_present"]
        print(f"    Δ nnls-cone − ridge      {results[fname]['nnls-cone']['spearman_present']-base:+.3f}")
        print(f"    Δ nnls-cone − nnls-ridge {results[fname]['nnls-cone']['spearman_present']-nnb:+.3f}")
        print(f"    Δ nnls-cone − cos-knn    {results[fname]['nnls-cone']['spearman_present']-results[fname]['cos-knn']['spearman_present']:+.3f}")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
