"""
demo_stage_oracle.py
--------------------
STAGE-ORACLE FLOOR on FROZEN features — the "split into stages + route" algorithm.

Continual learning with N_STAGES experts forgets nothing IF an oracle tells you which
stage a query belongs to (Task-IL ceiling).  The whole game is then the oracle: a
router that maps a query to its stage.  This demo measures, on frozen ViT features (no
training yet), how good a CONE is as that router vs the two point baselines:

    NCM-router          : 1 mean per stage          (single-prototype point)
    multiproto-router   : K centroids per stage      (capacity-matched point)
    cone-router         : 1 (mixture-)cone per stage (the conic region)

Protocol (closed-set routing; every test sample belongs to exactly one stage):
    1. Split the C classes into N_STAGES near-equal stages (random; the incremental
       order doesn't matter on FROZEN features — features are stationary, so an offline
       split is identical to building stage-by-stage).
    2. WITHIN-STAGE HEAD is SHARED across all routers: nearest fine-class centroid among
       the routed stage's classes.  So the ONLY thing compared is the router.
    3. score(total) = route to a stage, then classify within it.  Reported against:
         - ORACLE      : route with the TRUE stage → within-stage head  (the ceiling)
         - flat NCM / flat multiproto : no routing, C-way (the floor)
    4. Soft top-m routing: keep the top-m stages, fine-rank over their union of classes
       (turns "pick the exact stage" into "is it in my top-m" — the error-propagation fix).

Decomposition this makes visible:   Acc_total ≈ Acc_route × Acc_within .

Reuses demo_joint_floor for feature loading / transform / cone building (build_hulls,
so RAY_METHOD="mixture" + N_CONES works here too) and its multiproto baseline.

    python -u demo_stage_oracle.py
"""

import demo_joint_floor as djf
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as _l2n

# ── knobs ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "vit_base_patch16_224.orig_in21k"
# Loop over every dataset you have a cache for; a missing/undownloaded one is skipped
# (it just prints a warning and moves on) so one absent set can't abort the sweep.
DATASETS = [
    # "CIFAR100",
    # "FGVCAircraft",
    # "Flowers102",
    # "OxfordIIITPet",
    # "Food101",
    "CUB200",
    "StanfordCars",
]
N_STAGES = 10  # number of experts/stages (the router is an N_STAGES-way problem)
STAGE_SEED = 0  # which random class→stage partition

# router / cone construction (forwarded to demo_joint_floor globals) ──────────────
RAY_METHOD = "spa"  # "spa" (one cone/stage) | "mixture" (union of N_CONES cones)
N_RAYS = 50  # rays per stage cone  (and K for the multiproto-router — capacity-matched)
N_CONES = 5  # mixture: sub-cones per stage   (RAY_METHOD="mixture")
MIXTURE_SUB_METHOD = "spa"
USE_PCA = False
PCA_DIM = 128

# separability transform fit on TRAIN (pca/none keep modes intact → cone-friendly)
TRANSFORM = "none"
WHITEN_RIDGE = 1.0
WHITEN_ALPHA = 0.25
PCA_KEEP = 512
LDA_SHRINK = "auto"

# cone routing score schemes to try (higher = more in-stage); headline = best by total
SCORE_NAMES = ["cosine", "angular_margin", "blended", "max_ray_sim"]
TOPK = [3, 6, 9]  # soft top-m routing depths to report

BATCH_SIZE = 64
DATA_DIR = "./data"


def _push_globals(dataset):
    """Drive demo_joint_floor's module globals so its get_features / build_hulls /
    fit_feature_transform behave exactly as in the joint demo (same cache, same cones).
    """
    djf.DATASET = dataset
    djf.MODEL_NAME = MODEL_NAME
    djf.BATCH_SIZE = BATCH_SIZE
    djf.DATA_DIR = DATA_DIR
    djf.RAY_METHOD = RAY_METHOD
    djf.N_RAYS = N_RAYS
    djf.N_CONES = N_CONES
    djf.MIXTURE_SUB_METHOD = MIXTURE_SUB_METHOD
    djf.USE_PCA = USE_PCA
    djf.PCA_DIM = PCA_DIM


def _unit(F):
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


# ── routers: each returns a (N_test, S) stage-score matrix (higher = more in-stage) ──
def ncm_router(Ftr_t, stage_tr, Fte_t, stages):
    """1 mean per stage → cosine route score."""
    scent = np.stack([Ftr_t[stage_tr == s].mean(axis=0) for s in stages])
    return _unit(Fte_t) @ _l2n(scent, axis=1).T


def multiproto_router(Ftr_t, stage_tr, Fte_t, stages, k, seed=0):
    """K k-means centroids per stage → route score = max cosine to ANY of the stage's
    centroids (the union-of-caps point detector; capacity-matched to the cone's rays).
    """
    Q = _unit(Fte_t)
    route = np.empty((len(Fte_t), len(stages)), dtype=np.float32)
    for si, s in enumerate(stages):
        Xn = _l2n(np.asarray(Ftr_t[stage_tr == s], np.float64), axis=1)
        kk = min(k, len(Xn))
        C = (
            _l2n(
                KMeans(n_clusters=kk, n_init=1, max_iter=50, random_state=seed)
                .fit(Xn)
                .cluster_centers_,
                axis=1,
            )
            if kk < len(Xn)
            else Xn
        )
        route[:, si] = (Q @ C.T).max(axis=1)
    return route


def cone_router_scores(Ftr_t, stage_tr, Fte_t, stages):
    """1 (mixture-)cone per stage via demo_joint_floor.build_hulls.  Returns a dict
    {scheme: (N,S) route matrix} for every SCORE_NAMES scheme (one score_all per stage,
    reused across schemes)."""
    feature_dict = {str(s): Ftr_t[stage_tr == s] for s in stages}
    hulls = djf.build_hulls(feature_dict)
    per = [hulls[str(s)].score_all(Fte_t) for s in stages]  # S dicts of (N,) arrays
    return {sch: np.stack([p[sch] for p in per], axis=1) for sch in SCORE_NAMES}


# ── route → within-stage classify (within head SHARED across routers) ───────────
def _hard(route, fine_cos, groups, stage_of_class, yte):
    """Top-1 route → nearest fine centroid inside the routed stage.  Returns
    (total_acc, route_acc)."""
    g = route.argmax(axis=1)
    pred = np.empty(len(yte), dtype=np.int64)
    for s, cls in groups.items():
        m = g == s
        if m.any():
            pred[m] = cls[fine_cos[m][:, cls].argmax(axis=1)]  # argmax within stage
    return float((pred == yte).mean()), float((g == stage_of_class[yte]).mean())


def _topk(route, fine_cos, stage_of_class, yte, k):
    """Soft route: fine-rank over the union of classes from the top-k stages."""
    S = route.shape[1]
    k = min(k, S)
    topg = np.argsort(-route, axis=1)[:, :k]  # (N,k) stage ids
    allowed = np.zeros(fine_cos.shape, dtype=bool)  # (N,C)
    sc = stage_of_class[None, :]  # (1,C) class→stage
    for j in range(k):
        allowed |= sc == topg[:, j][:, None]
    pred = np.where(allowed, fine_cos, -np.inf).argmax(axis=1)
    return float((pred == yte).mean())


def stage_oracle_eval(Ftr, ytr, Fte, yte):
    """One dataset: split into N_STAGES, build the shared within-stage head + the three
    routers, and report oracle / flat baselines / per-router (hard + soft) accuracy."""
    C = int(max(int(ytr.max()), int(yte.max()))) + 1
    stage_of_class = djf._partition_groups(C, N_STAGES, STAGE_SEED)  # (C,) class→stage
    stages = list(range(int(stage_of_class.max()) + 1))
    S = len(stages)
    groups = {s: np.where(stage_of_class == s)[0] for s in stages}  # stage→class ids

    T = djf.fit_feature_transform(
        Ftr,
        ytr,
        TRANSFORM,
        ridge=WHITEN_RIDGE,
        lda_shrink=LDA_SHRINK,
        alpha=WHITEN_ALPHA,
        k=PCA_KEEP,
    )
    Ftr_t, Fte_t = T(Ftr), T(Fte)
    stage_tr, ys = stage_of_class[ytr], stage_of_class[yte]

    # shared within-stage head: fine-class centroids (NCM within stage)
    fcent = np.stack([_l2n(Ftr_t[ytr == c].mean(0)[None], axis=1)[0] for c in range(C)])
    fine_cos = _unit(Fte_t) @ fcent.T  # (N,C) cos to every fine centroid

    # floor (no routing) and ceiling (true-stage routing)
    flat_ncm = float((fine_cos.argmax(1) == yte).mean())
    flat_mp, _ = djf.multiproto_accuracy(Ftr_t, ytr, Fte_t, yte, list(range(C)), N_RAYS)
    # ORACLE: build a perfect (N,S) route from the true stage, reuse _hard's within head
    true_route = np.full((len(yte), S), -1.0, np.float32)
    true_route[np.arange(len(yte)), ys] = 1.0
    oracle, _ = _hard(true_route, fine_cos, groups, stage_of_class, yte)

    # routers → stage-score matrices
    routers = {
        "NCM": ncm_router(Ftr_t, stage_tr, Fte_t, stages),
        f"multiproto(k={N_RAYS})": multiproto_router(
            Ftr_t, stage_tr, Fte_t, stages, N_RAYS, STAGE_SEED
        ),
    }
    cone_mats = cone_router_scores(Ftr_t, stage_tr, Fte_t, stages)
    # pick the cone scheme with the best HARD total as the cone-router headline
    cone_hard = {
        sch: _hard(mat, fine_cos, groups, stage_of_class, yte)
        for sch, mat in cone_mats.items()
    }
    best_sch = max(cone_hard, key=lambda s: cone_hard[s][0])
    routers[f"cone:{best_sch}"] = cone_mats[best_sch]

    rows = {}
    for name, route in routers.items():
        total, racc = _hard(route, fine_cos, groups, stage_of_class, yte)
        soft = {k: _topk(route, fine_cos, stage_of_class, yte, k) for k in TOPK}
        rows[name] = dict(total=total, route=racc, soft=soft)

    return dict(
        C=C,
        S=S,
        chance=1.0 / C,
        flat_ncm=flat_ncm,
        flat_mp=flat_mp,
        oracle=oracle,
        rows=rows,
        cone_schemes={s: cone_hard[s] for s in cone_hard},
        best_sch=best_sch,
    )


def _print_report(name, r):
    print("\n" + "=" * 68)
    print(
        f"STAGE ORACLE — {name}  ({r['C']} classes / {r['S']} stages, "
        f"transform={TRANSFORM}, router={RAY_METHOD}×{N_RAYS}"
        + (f"/{N_CONES}cones" if RAY_METHOD == "mixture" else "")
        + f", chance={r['chance']:.4f})"
    )
    print("=" * 68)
    print(f"  {'flat NCM (no routing, C-way)':<34} {r['flat_ncm']:.4f}   <- floor")
    print(f"  {'flat multiproto (no routing)':<34} {r['flat_mp']:.4f}")
    print(f"  {'ORACLE (true stage → within)':<34} {r['oracle']:.4f}   <- ceiling")
    print("  " + "-" * 60)
    print(
        f"  {'router':<24} {'total':>7} {'route':>7}   "
        + "  ".join(f"top{k}" for k in TOPK)
    )
    for nm, d in r["rows"].items():
        soft = "  ".join(f"{d['soft'][k]:.4f}" for k in TOPK)
        print(f"  {nm:<24} {d['total']:>7.4f} {d['route']:>7.4f}   {soft}")
    # verdict: cone vs the capacity-matched point router (route accuracy is the oracle)
    cone_nm = f"cone:{r['best_sch']}"
    mp_nm = next(n for n in r["rows"] if n.startswith("multiproto"))
    d_route = r["rows"][cone_nm]["route"] - r["rows"][mp_nm]["route"]
    d_total = r["rows"][cone_nm]["total"] - r["rows"][mp_nm]["total"]
    verdict = (
        "cone routes better"
        if d_route > 0.003
        else "multiproto routes better" if d_route < -0.003 else "tied"
    )
    print(
        f"\n  cone − multiproto:  route {d_route:+.4f}  total {d_total:+.4f}  ({verdict})"
        f"   [oracle headroom {r['oracle'] - r['rows'][cone_nm]['total']:+.4f}]"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] no CUDA — ViT-B feature extraction on CPU will be slow.")

    summary = []
    for name in DATASETS:
        _push_globals(name)
        try:
            Ftr, ytr, Fte, yte = djf.get_features(device)
        except Exception as e:  # missing cache / failed download → skip, keep sweeping
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        djf.check_feature_health(Ftr, Fte, ytr, yte, name)
        r = stage_oracle_eval(Ftr, ytr, Fte, yte)
        _print_report(name, r)
        cone_nm = f"cone:{r['best_sch']}"
        mp_nm = next(n for n in r["rows"] if n.startswith("multiproto"))
        summary.append((name, r, cone_nm, mp_nm))

    if len(summary) > 1:
        print("\n" + "=" * 68)
        print("SUMMARY — total accuracy (hard top-1 route → within-stage head)")
        print("=" * 68)
        print(
            f"  {'dataset':<14} {'oracle':>7} {'NCM-r':>7} {'mp-r':>7} {'cone-r':>7} "
            f"{'Δroute':>7}"
        )
        for name, r, cone_nm, mp_nm in summary:
            print(
                f"  {name:<14} {r['oracle']:>7.4f} {r['rows']['NCM']['total']:>7.4f} "
                f"{r['rows'][mp_nm]['total']:>7.4f} {r['rows'][cone_nm]['total']:>7.4f} "
                f"{r['rows'][cone_nm]['route'] - r['rows'][mp_nm]['route']:>+7.4f}"
            )


if __name__ == "__main__":
    main()
