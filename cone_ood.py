"""
cone_ood.py
-----------
Cone OOD detection on CLIP embeddings: can a per-class conic hull tell its own
dataset (ID) from a held-out *different* dataset (OOD)?

Score (per the request):  s_OOD(x) = min_c r_c(x)
where r_c(x) is the unit-sphere NNLS residual of x against class c's cone.  A
sample that fits *some* class cone well (small min residual) is ID; a sample no
cone can explain (large min residual) is OOD.  We rank by ID-ness

    idness_cone(x) = max_c  ConicHull_c.score_nnls_residual(x)   # = 1 - min_c r_c/2

so AUROC(OOD-positive) uses  -idness  as the OOD score.

Why CLIP: the geometry sweep (cone_geometry.py) shows CLIP class cones are 100%
acute on every dataset (half-angles 22-46°) — the regime where a positive cone
result is most plausible.

Baselines (the honest comparison — prior work found the cone only *ties* NCM at
OOD routing because a cone is a larger admissible region):
  * NCM  : idness = max_c cos(x, centroid_c)
  * kNN  : idness = max over ID support exemplars of cos(x, exemplar)

Protocol
--------
Reuses the cached CLIP test features from cone_geometry.py (no re-extraction).
For each ID dataset: per-class seeded 50/50 support/query split.  Cones/centroids/
exemplars are fit on SUPPORT.  ID eval = held-out QUERY half.  OOD eval = every
other dataset's test features (capped).  AUROC per (ID, OOD) pair, plus each ID
vs its pooled OOD.

Usage
-----
    python -u cone_ood.py                          # all cached CLIP datasets
    python -u cone_ood.py --id-datasets CIFAR100 CUB200
    python -u cone_ood.py --n-rays 24 --n-ood-cap 1500
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from conic_hull import ConicHull

CACHE_DIR = "./cone_geom_out/cache"   # reuse cone_geometry.py feature cache
OUT_DIR = "./cone_ood_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_feats(ds):
    """Return (feats [N,D] float32 L2-normalised, labels [N]) for a CLIP cache."""
    path = os.path.join(CACHE_DIR, f"feats_{ds}_clip.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    feats = d["feats"].astype(np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats, d["labels"]


def available_datasets():
    if not os.path.isdir(CACHE_DIR):
        return []
    return sorted(f[len("feats_"):-len("_clip.npz")]
                  for f in os.listdir(CACHE_DIR)
                  if f.startswith("feats_") and f.endswith("_clip.npz"))


def split_support_query(feats, labels, max_support, seed=0):
    """Per-class 50/50 support/query split, support capped at max_support."""
    rng = np.random.default_rng(seed)
    sup_idx, qry_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        s = idx[:h]
        if max_support and len(s) > max_support:
            s = s[:max_support]
        sup_idx.append(s)
        qry_idx.append(idx[h:])
    return np.concatenate(sup_idx), np.concatenate(qry_idx)


def cap(feats, n, seed=0):
    if n and len(feats) > n:
        rng = np.random.default_rng(seed)
        return feats[rng.choice(len(feats), n, replace=False)]
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Fit ID class models on the support split
# ─────────────────────────────────────────────────────────────────────────────
def fit_id_models(sup_feats, sup_labels, n_rays):
    """Return (cones list, centroids [C,D], exemplar_bank [M,D])."""
    cones, centroids = [], []
    classes = np.unique(sup_labels)
    for c in classes:
        Xc = sup_feats[sup_labels == c]
        k = int(min(n_rays, len(Xc)))
        ch = ConicHull(n_rays=k, use_pca=False, ray_diversity="hybrid")
        # SPA/FPS ray selection needs a kNN graph (≥ ~12 pts). Tiny classes
        # just use their own normalised points as extreme rays.
        if len(Xc) < 12:
            Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
            ch.extreme_rays_ = Xn
            ch.extreme_rays_index = np.arange(len(Xc))
        else:
            ch.fit(Xc)
        cones.append(ch)
        mu = Xc.mean(0)
        centroids.append(mu / (np.linalg.norm(mu) + 1e-8))
    return cones, np.stack(centroids), sup_feats  # bank = all support


# ─────────────────────────────────────────────────────────────────────────────
# ID-ness scores (higher = more in-distribution)
# ─────────────────────────────────────────────────────────────────────────────
def idness_cone(cones, queries):
    """max_c score_nnls_residual = 1 - (min_c r_c)/2.  Higher = more ID."""
    best = np.full(len(queries), -np.inf, dtype=np.float32)
    for ch in cones:
        s = ch.score_nnls_residual(queries)          # (N,) in [0,1], GPU FISTA
        best = np.maximum(best, s)
    return best


def idness_ncm(centroids, queries):
    """max_c cos(x, centroid_c).  queries already L2-normalised."""
    C = torch.tensor(centroids, device=DEVICE)
    Q = torch.tensor(queries, device=DEVICE)
    return (Q @ C.T).max(dim=1).values.cpu().numpy()


def idness_knn(bank, queries, chunk=4096):
    """max over ID support exemplars of cos(x, exemplar)."""
    B = torch.tensor(bank, device=DEVICE)
    out = np.empty(len(queries), dtype=np.float32)
    for i in range(0, len(queries), chunk):
        Q = torch.tensor(queries[i:i + chunk], device=DEVICE)
        out[i:i + chunk] = (Q @ B.T).max(dim=1).values.cpu().numpy()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    avail = available_datasets()
    ap.add_argument("--id-datasets", nargs="+", default=avail)
    ap.add_argument("--n-rays", type=int, default=24)
    ap.add_argument("--max-support-per-class", type=int, default=150)
    ap.add_argument("--n-query-cap", type=int, default=2000,
                    help="cap on ID query samples used for AUROC")
    ap.add_argument("--n-ood-cap", type=int, default=1500,
                    help="cap per OOD dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[device] {DEVICE}  cached CLIP datasets: {avail}")
    print(f"[config] n_rays={args.n_rays} max_sup={args.max_support_per_class} "
          f"query_cap={args.n_query_cap} ood_cap={args.n_ood_cap}")

    # Preload every dataset's CLIP test features once.
    feats = {ds: load_clip_feats(ds) for ds in avail}
    feats = {k: v for k, v in feats.items() if v is not None}

    methods = ["cone", "ncm", "knn"]
    # results[method][id][ood] = AUROC ; results[method][id]["POOLED"] = AUROC
    results = {m: {} for m in methods}
    summary = {}

    for idd in args.id_datasets:
        if idd not in feats:
            print(f"[skip] {idd}: no CLIP cache")
            continue
        t0 = time.time()
        f_id, y_id = feats[idd]
        sup_i, qry_i = split_support_query(f_id, y_id, args.max_support_per_class,
                                           seed=args.seed)
        sup_f, sup_y = f_id[sup_i], y_id[sup_i]
        qry_f = cap(f_id[qry_i], args.n_query_cap, seed=args.seed)

        cones, centroids, bank = fit_id_models(sup_f, sup_y, args.n_rays)
        C = len(cones)

        # ID-ness of the ID query half
        idq = {
            "cone": idness_cone(cones, qry_f),
            "ncm": idness_ncm(centroids, qry_f),
            "knn": idness_knn(bank, qry_f),
        }

        pooled_scores = {m: [] for m in methods}
        pooled_labels = []
        for ood in feats:
            if ood == idd:
                continue
            f_ood = cap(feats[ood][0], args.n_ood_cap, seed=args.seed + 1)
            ido = {
                "cone": idness_cone(cones, f_ood),
                "ncm": idness_ncm(centroids, f_ood),
                "knn": idness_knn(bank, f_ood),
            }
            # AUROC per pair: OOD is positive class, score = -idness (OOD-ness)
            y = np.r_[np.zeros(len(qry_f)), np.ones(len(f_ood))]
            for m in methods:
                score = -np.r_[idq[m], ido[m]]
                auc = roc_auc_score(y, score)
                results[m].setdefault(idd, {})[ood] = float(auc)
                pooled_scores[m].append(-ido[m])
            pooled_labels.append(np.ones(len(f_ood)))

        # pooled: ID query vs all OOD combined
        y_pool = np.r_[np.zeros(len(qry_f)), np.concatenate(pooled_labels)]
        for m in methods:
            score = np.r_[-idq[m], np.concatenate(pooled_scores[m])]
            results[m][idd]["POOLED"] = float(roc_auc_score(y_pool, score))

        summary[idd] = {
            "n_classes": int(C),
            "n_query": int(len(qry_f)),
            "cone_pooled": results["cone"][idd]["POOLED"],
            "ncm_pooled": results["ncm"][idd]["POOLED"],
            "knn_pooled": results["knn"][idd]["POOLED"],
            "cone_minus_ncm": results["cone"][idd]["POOLED"] - results["ncm"][idd]["POOLED"],
        }
        print(f"[{idd}] C={C} q={len(qry_f)} | POOLED AUROC  "
              f"cone {summary[idd]['cone_pooled']:.4f}  "
              f"ncm {summary[idd]['ncm_pooled']:.4f}  "
              f"knn {summary[idd]['knn_pooled']:.4f}  "
              f"(cone-ncm {summary[idd]['cone_minus_ncm']:+.4f})  "
              f"[{time.time()-t0:.1f}s]")

    _write_report(results, summary, args)


def _write_report(results, summary, args):
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"results": results, "summary": summary,
                   "config": vars(args)}, f, indent=2)

    ids = list(summary)
    lines = ["# Cone OOD detection on CLIP (min_c r_c(x))\n",
             "Score = `min_c r_c(x)` (unit-sphere NNLS residual to nearest class "
             "cone). AUROC with OOD as the positive class; ID = held-out query "
             "half of the same dataset. Higher = better ID/OOD separation.\n"]

    # headline table
    lines.append("## Pooled OOD (each ID vs all other datasets)\n")
    lines.append("| ID dataset | C | cone | NCM | kNN | cone−NCM |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for idd in ids:
        s = summary[idd]
        lines.append(f"| {idd} | {s['n_classes']} | {s['cone_pooled']:.4f} | "
                     f"{s['ncm_pooled']:.4f} | {s['knn_pooled']:.4f} | "
                     f"{s['cone_minus_ncm']:+.4f} |")
    if ids:
        mc = np.mean([summary[i]["cone_pooled"] for i in ids])
        mn = np.mean([summary[i]["ncm_pooled"] for i in ids])
        mk = np.mean([summary[i]["knn_pooled"] for i in ids])
        lines.append(f"| **mean** | | **{mc:.4f}** | **{mn:.4f}** | **{mk:.4f}** | "
                     f"**{mc-mn:+.4f}** |")

    # full pairwise cone matrix
    lines.append("\n## Pairwise cone AUROC (rows = ID, cols = OOD)\n")
    header = "| ID \\\\ OOD | " + " | ".join(ids) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ids) + 1))
    for idd in ids:
        row = [idd]
        for ood in ids:
            if ood == idd:
                row.append("—")
            else:
                row.append(f"{results['cone'].get(idd, {}).get(ood, float('nan')):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print(f"\n{report}")
    print(f"[done] wrote {OUT_DIR}/report.md and results.json")


if __name__ == "__main__":
    main()
