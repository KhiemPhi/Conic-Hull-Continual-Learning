"""
cone_vs_multiproto_ood.py
-------------------------
Does a filled cone beat a *multi-prototype* NCM on real data — in the one regime
theory says it can: multimodal ID classes at a matched, small parameter budget?

Construction (real-data proxy for "filled continuum, not blobs"):
  ID class = a MULTIMODAL super-class = several fine classes merged into one label.
    * CIFAR100-super : the 20 official semantic super-classes (5 fine modes each)
    * <ds>-mergeK    : random-merge K fine classes -> ceil(C/K) super-classes
  OOD = every OTHER dataset's test split (held out).

Matched-budget bake-off (each ID super-class gets m vectors either way):
  * cone       : ConicHull with n_rays = m   (ONE filled conic region / class)
  * multiproto : m spherical-k-means prototypes / class (nearest-prototype angle)
  * (m=1 multiproto == single-centroid NCM, included as the reference)

Prediction: the cone's fill spans the modes with few extreme rays, so it should
help at SMALL m (budget < #modes) where m discrete prototypes can't cover every
mode — and lose once m >= #modes (a prototype lands on each mode and the cone's
inter-mode fill only admits OOD).  We sweep m to find the crossover, if any.

Reuses the cached CLIP test features from cone_geometry.py (no re-extraction).

Usage
-----
    python -u cone_vs_multiproto_ood.py                       # CIFAR100-super
    python -u cone_vs_multiproto_ood.py --id CUB200 --merge-k 10
    python -u cone_vs_multiproto_ood.py --budgets 1 2 3 5 8
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from conic_hull import ConicHull
import demo_joint_floor as djf   # _CIFAR100_COARSE, merge_labels

CACHE_DIR = "./cone_geom_out/cache"
OUT_DIR = "./cone_multiproto_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(ds):
    path = os.path.join(CACHE_DIR, f"feats_{ds}_clip.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    f = d["feats"].astype(np.float32)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8), d["labels"]


def available():
    return sorted(f[len("feats_"):-len("_clip.npz")] for f in os.listdir(CACHE_DIR)
                  if f.startswith("feats_") and f.endswith("_clip.npz"))


def make_superclasses(ds, labels, merge_k):
    """Return (super_labels, n_super, n_modes_per_super_est)."""
    if ds == "CIFAR100" and merge_k == 0:
        y = djf._CIFAR100_COARSE[labels]
        return y, 20, 5
    # random merge K fine -> super
    n_fine = int(labels.max()) + 1
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_fine)
    fine2super = np.empty(n_fine, dtype=int)
    for s, chunk in enumerate(np.array_split(perm, int(np.ceil(n_fine / merge_k)))):
        fine2super[chunk] = s
    return fine2super[labels], int(fine2super.max()) + 1, merge_k


def split_sq(feats, labels, max_support=200, seed=0):
    rng = np.random.default_rng(seed)
    s, q = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        ss = idx[:h][:max_support]
        s.append(ss)
        q.append(idx[h:])
    return np.concatenate(s), np.concatenate(q)


def cap(F, n, seed=0):
    if n and len(F) > n:
        return F[np.random.default_rng(seed).choice(len(F), n, replace=False)]
    return F


# ── scorers (ID-ness: higher = in-distribution) ──────────────────────────────
def spherical_kmeans(Xn, m, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    m = min(m, len(Xn))
    C = Xn[rng.choice(len(Xn), m, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(Xn @ C.T, axis=1)
        newC = C.copy()
        for j in range(m):
            pts = Xn[a == j]
            if len(pts):
                s = pts.sum(0)
                newC[j] = s / (np.linalg.norm(s) + 1e-8)
        if np.allclose(newC, C):
            break
        C = newC
    return C


def fit_multiproto(Fn, y, m):
    protos = []
    for c in np.unique(y):
        protos.append(spherical_kmeans(Fn[y == c], m))
    return np.concatenate(protos)                      # (<= C*m, D) unit


def idness_multiproto(protos, Fn, chunk=8192):
    P = torch.tensor(protos, device=DEVICE)
    out = np.empty(len(Fn), np.float32)
    for i in range(0, len(Fn), chunk):
        Q = torch.tensor(Fn[i:i + chunk], device=DEVICE)
        out[i:i + chunk] = (Q @ P.T).max(1).values.cpu().numpy()
    return out


def fit_cones(Fn, y, m):
    cones = []
    for c in np.unique(y):
        Xc = Fn[y == c]
        k = int(min(m, len(Xc)))
        ch = ConicHull(n_rays=k, use_pca=False, ray_diversity="hybrid")
        if len(Xc) < 12 or k < 2:
            ch.extreme_rays_ = Xc[:max(k, 1)]
            ch.extreme_rays_index = np.arange(max(k, 1))
        else:
            ch.fit(Xc)
        cones.append(ch)
    return cones


def idness_cone(cones, Fn):
    best = np.full(len(Fn), -np.inf, np.float32)
    for ch in cones:
        best = np.maximum(best, ch.score_nnls_residual(Fn))
    return best


def auroc_fpr(idn_id, idn_ood):
    y = np.r_[np.ones(len(idn_id)), np.zeros(len(idn_ood))]
    auc = roc_auc_score(y, np.r_[idn_id, idn_ood])
    thr = np.quantile(idn_id, 0.05)
    fpr = float(np.mean(idn_ood >= thr))
    return float(auc), fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="CIFAR100")
    ap.add_argument("--merge-k", type=int, default=0,
                    help="0 = CIFAR100 official 20 superclasses; else random-merge K fine")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--n-query-cap", type=int, default=2000)
    ap.add_argument("--n-ood-cap", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    avail = available()
    f_id = load_clip(args.id)
    if f_id is None:
        raise SystemExit(f"no CLIP cache for {args.id}; have {avail}")
    F, yfine = f_id
    ysuper, n_super, n_modes = make_superclasses(args.id, yfine, args.merge_k)
    print(f"[id] {args.id}  fine={int(yfine.max())+1} -> super={n_super} "
          f"(~{n_modes} modes/super)  feats {F.shape}")

    s_i, q_i = split_sq(F, ysuper)
    Fs, ys = F[s_i], ysuper[s_i]
    Fq = cap(F[q_i], args.n_query_cap)

    ood_feats = []
    for ood in avail:
        if ood == args.id:
            continue
        ood_feats.append((ood, cap(load_clip(ood)[0], args.n_ood_cap, seed=1)))
    print(f"[ood] {len(ood_feats)} datasets: {[o for o,_ in ood_feats]}")

    results = {}
    tag = f"{args.id}" + ("_super" if args.merge_k == 0 and args.id == "CIFAR100"
                          else f"_merge{args.merge_k}")
    for m in args.budgets:
        cones = fit_cones(Fs, ys, m)
        protos = fit_multiproto(Fs, ys, m)
        idc_id = idness_cone(cones, Fq)
        idp_id = idness_multiproto(protos, Fq)
        cone_a, cone_f, mp_a, mp_f = [], [], [], []
        for ood, Fo in ood_feats:
            ca, cf = auroc_fpr(idc_id, idness_cone(cones, Fo))
            pa, pf = auroc_fpr(idp_id, idness_multiproto(protos, Fo))
            cone_a.append(ca); cone_f.append(cf); mp_a.append(pa); mp_f.append(pf)
        r = dict(cone_auroc=float(np.mean(cone_a)), mp_auroc=float(np.mean(mp_a)),
                 cone_fpr=float(np.mean(cone_f)), mp_fpr=float(np.mean(mp_f)),
                 n_protos=int(len(protos)))
        r["delta_auroc"] = r["cone_auroc"] - r["mp_auroc"]
        results[m] = r
        print(f"[m={m}] protos={r['n_protos']:>3} | AUROC cone {r['cone_auroc']:.4f} "
              f"multiproto {r['mp_auroc']:.4f} (Δ {r['delta_auroc']:+.4f}) | "
              f"FPR95 cone {r['cone_fpr']:.3f} mp {r['mp_fpr']:.3f}")

    _report(results, args, tag, n_super, n_modes)


def _report(results, args, tag, n_super, n_modes):
    with open(os.path.join(OUT_DIR, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = [f"# Cone vs multi-prototype NCM on multimodal ID ({tag})\n",
             f"ID = {args.id} merged into {n_super} super-classes (~{n_modes} modes each). "
             "OOD = other datasets. Matched budget: cone n_rays = m vs m prototypes/class. "
             "Mean over OOD sets.\n",
             "| budget m | AUROC cone | AUROC multiproto | Δ(cone−mp) | FPR95 cone | FPR95 mp |",
             "|--:|--:|--:|--:|--:|--:|"]
    for m, r in results.items():
        lines.append(f"| {m} | {r['cone_auroc']:.4f} | {r['mp_auroc']:.4f} | "
                     f"{r['delta_auroc']:+.4f} | {r['cone_fpr']:.3f} | {r['mp_fpr']:.3f} |")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, f"report_{tag}.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"[done] wrote {OUT_DIR}/report_{tag}.md")


if __name__ == "__main__":
    main()
