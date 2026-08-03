"""
shift_detect.py
---------------
"Am I seeing a NEW dataset?" -- population-level distribution-shift detection from a
trained detector, as a function of batch size k. This is the COVERAGE question (does a
SET open new territory), not the per-sample OOD question (which density already wins,
and where the cone is redundant -- corr 0.9, see ood_hybrid.py).

Trained detector = fit on the ID support half:
  cone : per-class ConicHull coverage (union of class cones)   -> YOUR conic hull
  maha : per-class shared-cov Gaussian (density)               -> baseline
  mmd  : RBF kernel two-sample test vs a fixed ID reference    -> the population baseline

Batch score (higher = more shift):
  cone = -mean_i idness_cone(x_i)        (mean support-residual over the batch)
  maha = -mean_i idness_maha(x_i)        (mean Mahalanobis over the batch)
  mmd  = biased MMD^2(batch, ID-reference), RBF, median-heuristic bandwidth

cone/maha batch scores are means of PER-SAMPLE idness -> precomputed once, then averaged
over sampled batches (fast). MMD is a set statistic -> computed per batch.

Test: for each ID dataset, draw B same-dist batches (k from ID query half) and B
new-dataset batches (k from a randomly chosen OTHER dataset). Detection AUROC
(new=positive) as a function of k in {1,2,5,10,20,50}. Report min-k to reach AUROC 0.95.

WIN condition (cone-native): cone reaches reliable detection at SMALLER k than MMD /
batch-Mahalanobis (early detection = the efficiency thesis applied to shift detection).
NULL: all three converge; corr 0.9 predicts no early-k edge -> closes the line honestly.

    HF_HUB_OFFLINE=1 python -u shift_detect.py
"""
import os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import cone_ood as CO
from ood_hybrid import unit, fit_gauss, idness_maha

DEVICE = CO.DEVICE
OUT = "./shift_detect_out"
K_LIST = [1, 2, 5, 10, 20, 50]
B = 200                       # batches per class (ID / OOD) per k
R_REF = 500                   # MMD reference size
POOL_CAP = 3000               # cap per feature pool for speed
METHODS = ["cone", "maha", "mmd"]


def median_sigma(ref):
    with torch.no_grad():
        d = torch.cdist(ref, ref)
        m = d[d > 0]
        return float(m.median()) if len(m) else 1.0


def mmd2(Xg, ref, Kyy_mean, inv2s2):
    """biased RBF MMD^2(batch, reference)."""
    Kxx = torch.exp(-torch.cdist(Xg, Xg).pow(2) * inv2s2).mean()
    Kxy = torch.exp(-torch.cdist(Xg, ref).pow(2) * inv2s2).mean()
    return float(Kxx + Kyy_mean - 2 * Kxy)


def batch_indices(n, k, B, rng):
    return [rng.choice(n, min(k, n), replace=(k > n)) for _ in range(B)]


def main():
    os.makedirs(OUT, exist_ok=True)
    avail = CO.available_datasets()
    feats = {d: CO.load_clip_feats(d) for d in avail}
    feats = {k: v for k, v in feats.items() if v is not None}
    print(f"[data] {list(feats)}  device={DEVICE}", flush=True)

    # results[idd][method][k] = AUROC
    results = {}
    for idd in feats:
        rng = np.random.default_rng(0)
        f_id, y_id = feats[idd]
        si, qi = CO.split_support_query(f_id, y_id, 150, seed=0)
        sup_f, sup_y = f_id[si], y_id[si]
        classes = np.unique(sup_y)

        # --- trained detectors on the support half ---
        cones, _, _ = CO.fit_id_models(sup_f, sup_y, 24)
        mus, prec = fit_gauss(unit(sup_f), sup_y, classes)
        ref_np = CO.cap(unit(sup_f), R_REF, seed=1)
        ref = torch.tensor(ref_np, device=DEVICE)
        sigma = median_sigma(ref); inv2s2 = 1.0 / (2 * sigma * sigma + 1e-8)
        Kyy_mean = torch.exp(-torch.cdist(ref, ref).pow(2) * inv2s2).mean()

        # --- candidate pools (ID query + each OOD) with precomputed per-sample idness ---
        def make_pool(F):
            Fu = unit(CO.cap(F, POOL_CAP, seed=2))
            return dict(F=Fu, Fg=torch.tensor(Fu, device=DEVICE),
                        c=CO.idness_cone(cones, Fu),
                        m=idness_maha(mus, prec, Fu))
        idpool = make_pool(f_id[qi])
        oodpools = [make_pool(feats[o][0]) for o in feats if o != idd]

        # --- sweep k ---
        results[idd] = {m: {} for m in METHODS}
        for k in K_LIST:
            sc = {m: ([], []) for m in METHODS}     # (id_scores, ood_scores)
            # ID batches
            for idx in batch_indices(len(idpool["F"]), k, B, rng):
                sc["cone"][0].append(-idpool["c"][idx].mean())
                sc["maha"][0].append(-idpool["m"][idx].mean())
                sc["mmd"][0].append(mmd2(idpool["Fg"][idx], ref, Kyy_mean, inv2s2))
            # OOD batches: each from a randomly chosen other dataset
            for _ in range(B):
                p = oodpools[rng.integers(len(oodpools))]
                idx = rng.choice(len(p["F"]), min(k, len(p["F"])), replace=(k > len(p["F"])))
                sc["cone"][1].append(-p["c"][idx].mean())
                sc["maha"][1].append(-p["m"][idx].mean())
                sc["mmd"][1].append(mmd2(p["Fg"][idx], ref, Kyy_mean, inv2s2))
            y = np.r_[np.zeros(B), np.ones(B)]
            for m in METHODS:
                s = np.r_[np.array(sc[m][0]), np.array(sc[m][1])]
                results[idd][m][k] = float(roc_auc_score(y, s))
        line = " | ".join(
            f"{m}:" + ",".join(f"{results[idd][m][k]:.2f}" for k in K_LIST) for m in METHODS)
        print(f"[{idd:14s}] AUROC@k({K_LIST})  {line}", flush=True)

    # --- aggregate ---
    ids = list(results)
    agg = {m: {k: float(np.mean([results[i][m][k] for i in ids])) for k in K_LIST}
           for m in METHODS}

    def min_k(m):
        for k in K_LIST:
            if agg[m][k] >= 0.95:
                return k
        return None

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"per_dataset": results, "mean": agg,
                   "min_k_0.95": {m: min_k(m) for m in METHODS}}, f, indent=2)

    print("\n=== detection AUROC vs batch size k (mean over ID datasets) ===")
    print("  k    | " + "  ".join(f"{m:>6s}" for m in METHODS))
    for k in K_LIST:
        print(f"  {k:<4d} | " + "  ".join(f"{agg[m][k]:6.3f}" for m in METHODS))
    print("\nmin-k to reach mean AUROC >= 0.95:")
    for m in METHODS:
        mk = min_k(m)
        print(f"  {m:6s}: k={mk if mk else '>50'}")


if __name__ == "__main__":
    main()
