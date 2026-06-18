"""
demo_drift_transport.py
-----------------------
DRIFT-MEMORY test on FROZEN features — the cone's last defensible niche.

Every STATIC test says points (multiproto) >= cones, with a structural reason (the
cone dilutes and degenerates to multiproto when its rays are well placed).  The ONE
regime untested is DYNAMIC: under representation drift, when you CANNOT recompute the
representation (old data gone), does the cone's SHAPE survive transport better than a
set of free-floating points?  That is the only place the cone could earn its keep, and
it is what "transported skeleton hulls" is built for.

We simulate drift on cached features so we can answer this BEFORE real training:

    birth space  : the cached features (optionally a separability transform).
    current space: birth space rotated by a random orthogonal R (magnitude = DRIFT
                   angle), optionally + anisotropic shear (non-rigid drift).
    All classes are "born" in birth space (reps built on FULL birth data); test data
    is observed in current space; reps must be TRANSPORTED birth->current to be usable.
    The transport map is fit from a few per-class REPLAY anchors (birth/current pairs)
    via incremental.fit_drift_map_from_pairs — so the map carries realistic error.

Fairness note: a PERFECT map just rotates cone and multiproto identically -> reduces to
the static comparison (multiproto wins, tautology).  The cone can only win when the map
is IMPERFECT (few anchors) and (a) its shape degrades more gracefully OR (b) its extreme
rays are better-spread anchors for fitting the map.  So we inject map error (few anchors,
swept) and expose ANCHOR_SELECT (random|extreme|central) to probe (b).

For each primitive {mean, multiproto-k, cone-k} we report current-space accuracy under:
    oracle      : rep rebuilt on FULL current-space data         (perfect-knowledge ceiling)
    stale       : birth rep used as-is, no transport             (ignore-drift floor)
    transported : birth rep transported via the estimated map    (THE method)
    dynamic     : rep rebuilt from the few replay anchors only    (re-fit baseline)

Decisive question:  transported-cone  vs  transported-multiproto  across drift levels.
If the cone doesn't win here, the cone-as-classifier/router/memory line is closed.

Reuses demo_joint_floor (features / transform / build_hulls / multiproto) and
incremental (fit_drift_map_from_pairs, _fit_pca_on_features).

    python -u demo_drift_transport.py
"""

import demo_joint_floor as djf
import numpy as np
import torch
from incremental import _fit_pca_on_features, fit_drift_map_from_pairs
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as _l2n

# ── knobs ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "vit_base_patch16_224.orig_in21k"
DATASETS = ["CUB200", "StanfordCars"]
CLASS_LIMIT = 0  # keep only first N classes (0 = all); small = faster sweeps

# drift simulation ───────────────────────────────────────────────────────────────
DRIFT_ANGLES = [0.25, 0.5]  # geodesic rotation in radians; mean
#   cos(birth,current) ≈ cos(angle): 0.5→0.88, 1.0→0.54, 1.5→0.07, 2.0→−0.42 (swept)
# Non-rigid drift: anisotropic per-axis scaling on top of the rotation.  This is the
# DECISIVE setting — a rigid (shear=0) rotation is exactly what a procrustes map undoes,
# so transported→oracle losslessly and cone vs multiproto just reproduce their static
# gap.  shear>0 makes the orthogonal map STRUCTURALLY unable to fully fit (residual
# persists even with many anchors), so "whose structure survives an imperfect map?"
# becomes a real question.  Set 0.0 to recover the rigid sanity check.
DRIFT_SHEAR = 0.2
DRIFT_SEED = 0

# replay anchors used to FIT the transport map ───────────────────────────────────
N_ANCHORS_PER_CLASS = 20  # few → realistic map error (the regime where shape matters)
ANCHOR_SELECT = "random"  # "random" | "extreme" (far from class mean) | "central"
ANCHOR_SEED = 0

# transport map (matches incremental._fit_transport conventions) ──────────────────
PAIR_METHOD = "procrustes"  # "procrustes" (rigid) | "ridge_affine" (allows shear)
TRANSPORT_RIDGE = 1e-3
TRANSPORT_PCA = (
    32  # fit map in a low-rank PCA subspace (robust for few anchors); 0=full-D
)

# representations ─────────────────────────────────────────────────────────────────
RAY_METHOD = "spa"  # "spa" | "mixture"
# cone rays per class AND k for multiproto (capacity-matched).  Keep WELL BELOW the
# min samples/class or multiproto degenerates to 1-NN (every sample its own centroid):
# CUB/Cars have ~30-40/class, so 50 was secretly 1-NN.  10 is a real multi-prototype set.
N_RAYS = 10
N_CONES = 5
MIXTURE_SUB_METHOD = "spa"
USE_PCA = False
PCA_DIM = 128
SCORE_NAMES = ["cosine", "angular_margin", "blended", "max_ray_sim"]

# birth-space separability transform (none keeps it clean; pca is cone-friendly) ──
TRANSFORM = "none"
WHITEN_RIDGE = 1.0
WHITEN_ALPHA = 0.25
PCA_KEEP = 512
LDA_SHRINK = "auto"

BATCH_SIZE = 64
DATA_DIR = "./data"


def _push_globals(dataset):
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


# ── drift simulation ─────────────────────────────────────────────────────────────
def make_drift(D, angle, rng, shear=0.0):
    """Drift = rotation by EXACTLY `angle` radians in D//2 random orthogonal planes, so
    every feature direction turns by `angle` → mean cos(birth, current) ≈ cos(angle).
    Optionally times anisotropic scaling (`shear`) for non-rigid drift.  Current-space
    feature = birth_feature @ R.T.

    Built as Q·B·Qᵀ with Q a random orthonormal basis and B block-diagonal 2×2 rotations
    by `angle`.  A Frobenius-normalised expm generator (the old version) instead spread a
    fixed budget over all ~D/2 planes → per-direction rotation ≈ angle/√(D/2), negligible
    for large D (mean cos stayed ~0.996 even at angle=2).  Here `angle` is a real geodesic
    rotation: cos(0.5)≈0.88, cos(1.0)≈0.54, cos(1.5)≈0.07, cos(2.0)≈−0.42."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))  # random orthonormal basis
    c, s = np.cos(angle), np.sin(angle)
    B = np.eye(D, dtype=np.float64)
    for i in range(0, D - 1, 2):  # rotate each random plane by `angle`
        B[i, i] = c
        B[i, i + 1] = -s
        B[i + 1, i] = s
        B[i + 1, i + 1] = c
    R = Q @ B @ Q.T
    if shear > 0:
        sc = np.exp(rng.standard_normal(D) * shear)  # log-normal per-axis scale
        R = R * sc[None, :]
    return R.astype(np.float32)


# ── transport (mirrors incremental._fit_transport) ───────────────────────────────
def fit_transport(Xo, Xn, method, ridge, pca_components):
    """Return transport(rays): birth-space → current-space, fit from paired anchors.
    PCA-subspace variant fits the map in a low-rank basis (well-conditioned for few
    pairs) and applies it only in-subspace; the orthogonal complement is preserved."""
    Xo = _l2n(np.asarray(Xo, np.float64), axis=1)
    Xn = _l2n(np.asarray(Xn, np.float64), axis=1)
    sw = np.ones(len(Xo))
    if pca_components:
        k = min(pca_components, len(Xo) - 1)
        mean, comps = _fit_pca_on_features({"_": np.vstack([Xo, Xn])}, n_components=k)
        O, N = (Xo - mean) @ comps.T, (Xn - mean) @ comps.T
        fit = fit_drift_map_from_pairs(
            O, N, method=method, ridge=ridge, sample_weights=sw
        )
        A, mo, mn = fit["A"].astype(np.float64), fit["mu_old"], fit["mu_new"]

        def transport(rays):
            c = np.asarray(rays, np.float64) - mean
            par = c @ comps.T
            perp = c - par @ comps
            return mean + ((par - mo) @ A.T + mn) @ comps + perp

        return transport, fit["residual"]

    fit = fit_drift_map_from_pairs(
        Xo, Xn, method=method, ridge=ridge, sample_weights=sw
    )
    A, mo, mn = fit["A"].astype(np.float64), fit["mu_old"], fit["mu_new"]
    return (lambda rays: (np.asarray(rays, np.float64) - mo) @ A.T + mn), fit[
        "residual"
    ]


# ── anchor selection ─────────────────────────────────────────────────────────────
def pick_anchors(Fb, y, classes, n_per, mode, rng):
    """Per class, choose n_per training indices as transport anchors.  'extreme' picks
    the points FARTHEST from the class mean (the well-spread set a cone's extreme rays
    live on); 'central' the nearest; 'random' uniform.  Probes whether spread anchors
    (the cone's structural byproduct) fit a better drift map."""
    idx_all = []
    for c in classes:
        ci = np.where(y == c)[0]
        if len(ci) <= n_per:
            idx_all.append(ci)
            continue
        if mode == "random":
            sel = rng.choice(ci, n_per, replace=False)
        else:
            mu = _l2n(Fb[ci].mean(0)[None], axis=1)[0]
            d = _l2n(Fb[ci], axis=1) @ mu  # cosine to class mean
            order = np.argsort(d)  # ascending: low cos = far/extreme
            sel = ci[order[:n_per]] if mode == "extreme" else ci[order[-n_per:]]
        idx_all.append(sel)
    return np.concatenate(idx_all)


# ── representations & classifiers ────────────────────────────────────────────────
def class_centroids(F, y, classes, k, seed):
    cents, owner = [], []
    for c in classes:
        Xn = _l2n(np.asarray(F[y == c], np.float64), axis=1)
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
        ).astype(np.float32)
        cents.append(C)
        owner.append(np.full(len(C), c, dtype=np.int64))
    return np.concatenate(cents), np.concatenate(owner)


def acc_cents(cents, owner, Fte, yte):
    pred = owner[(_l2n(Fte, axis=1) @ _l2n(cents, axis=1).T).argmax(1)]
    return float((pred == yte).mean())


def acc_cone(hulls, classes, Fte, yte):
    per = [hulls[str(c)].score_all(Fte) for c in classes]
    best = 0.0
    for s in SCORE_NAMES:
        pred = np.array(classes)[np.stack([p[s] for p in per], 1).argmax(1)]
        best = max(best, float((pred == yte).mean()))
    return best


def cone_set_rays(hull_src, transform_fn=None):
    """Copy a source hull's rays (optionally transported) into a fresh ConicHull."""
    rays = hull_src.extreme_rays_
    if transform_fn is not None:
        rays = _l2n(transform_fn(rays), axis=1)
    return djf._rays_to_hull(rays, k_local=hull_src.k_local)


def run_drift(Fb_tr, ytr, Fb_te, yte, classes):
    """Sweep DRIFT_ANGLES; for each, report oracle/stale/transported/dynamic accuracy
    for mean, multiproto, cone.  Birth space = Fb_*, current = birth @ R.T."""
    rng = np.random.default_rng(DRIFT_SEED)
    arng = np.random.default_rng(ANCHOR_SEED)
    D = Fb_tr.shape[1]
    anchor_idx = pick_anchors(
        Fb_tr, ytr, classes, N_ANCHORS_PER_CLASS, ANCHOR_SELECT, arng
    )

    min_per = min(int((ytr == c).sum()) for c in classes)
    if N_RAYS >= min_per:  # multiproto would degenerate to 1-NN → broken control
        print(
            f"[warn] N_RAYS={N_RAYS} >= min samples/class={min_per}: multiproto "
            f"degenerates to 1-NN (every sample its own centroid). Lower N_RAYS."
        )

    # birth-space reps (built ONCE on full birth data; reused every angle) ─────────
    b_means = np.stack(
        [_l2n(Fb_tr[ytr == c].mean(0)[None], axis=1)[0] for c in classes]
    )
    b_cents, b_owner = class_centroids(Fb_tr, ytr, classes, N_RAYS, 0)
    b_hulls = djf.build_hulls({str(c): Fb_tr[ytr == c] for c in classes})

    rows = []
    for ang in DRIFT_ANGLES:
        R = (
            make_drift(D, ang, rng, DRIFT_SHEAR)
            if ang > 0
            else np.eye(D, dtype=np.float32)
        )
        Fc_te = Fb_te @ R.T  # current-space test
        Fc_tr = Fb_tr @ R.T  # current-space train (oracle/dynamic/anchors)

        transport, resid = fit_transport(
            Fb_tr[anchor_idx],
            Fc_tr[anchor_idx],
            PAIR_METHOD,
            TRANSPORT_RIDGE,
            TRANSPORT_PCA,
        )

        # ── MEAN ──────────────────────────────────────────────────────────────────
        m_oracle = acc_cents(
            *(
                np.stack(
                    [_l2n(Fc_tr[ytr == c].mean(0)[None], axis=1)[0] for c in classes]
                ),
                np.array(classes),
            ),
            Fc_te,
            yte,
        )
        m_stale = acc_cents(b_means, np.array(classes), Fc_te, yte)
        m_transp = acc_cents(
            _l2n(transport(b_means), axis=1), np.array(classes), Fc_te, yte
        )
        dm = np.stack(
            [
                _l2n(Fc_tr[anchor_idx][ytr[anchor_idx] == c].mean(0)[None], axis=1)[0]
                for c in classes
            ]
        )
        m_dyn = acc_cents(dm, np.array(classes), Fc_te, yte)

        # ── MULTIPROTO ──────────────────────────────────────────────────────────────
        oc, oo = class_centroids(Fc_tr, ytr, classes, N_RAYS, 0)
        p_oracle = acc_cents(oc, oo, Fc_te, yte)
        p_stale = acc_cents(b_cents, b_owner, Fc_te, yte)
        p_transp = acc_cents(_l2n(transport(b_cents), axis=1), b_owner, Fc_te, yte)
        dc, do = class_centroids(Fc_tr[anchor_idx], ytr[anchor_idx], classes, N_RAYS, 0)
        p_dyn = acc_cents(dc, do, Fc_te, yte)

        # ── CONE ───────────────────────────────────────────────────────────────────
        c_oracle = acc_cone(
            djf.build_hulls({str(c): Fc_tr[ytr == c] for c in classes}),
            classes,
            Fc_te,
            yte,
        )
        c_stale = acc_cone(b_hulls, classes, Fc_te, yte)
        t_hulls = {str(c): cone_set_rays(b_hulls[str(c)], transport) for c in classes}
        c_transp = acc_cone(t_hulls, classes, Fc_te, yte)
        d_hulls = djf.build_hulls(
            {str(c): Fc_tr[anchor_idx][ytr[anchor_idx] == c] for c in classes}
        )
        c_dyn = acc_cone(d_hulls, classes, Fc_te, yte)

        rows.append(
            dict(
                angle=ang,
                resid=resid,
                mean=(m_oracle, m_stale, m_transp, m_dyn),
                multiproto=(p_oracle, p_stale, p_transp, p_dyn),
                cone=(c_oracle, c_stale, c_transp, c_dyn),
            )
        )
    return rows


def _print(name, rows, n_cls):
    print("\n" + "=" * 76)
    print(
        f"DRIFT-MEMORY — {name}  ({n_cls} classes, transform={TRANSFORM}, "
        f"reps={RAY_METHOD}×{N_RAYS}, anchors={N_ANCHORS_PER_CLASS}/cls·{ANCHOR_SELECT}, "
        f"map={PAIR_METHOD}/pca{TRANSPORT_PCA})"
    )
    print("=" * 76)
    for r in rows:
        print(
            f"\n  drift angle = {r['angle']:.2f}   (map in-sample resid = {r['resid']:.3f})"
        )
        print(
            f"    {'primitive':<12} {'oracle':>8} {'stale':>8} {'transp':>8} {'dynamic':>8}"
        )
        for p in ("mean", "multiproto", "cone"):
            o, s, t, d = r[p]
            print(f"    {p:<12} {o:>8.4f} {s:>8.4f} {t:>8.4f} {d:>8.4f}")
        ct, pt = r["cone"][2], r["multiproto"][2]
        cd, _ = r["cone"][3], r["multiproto"][3]
        verdict = (
            "cone transports better"
            if ct - pt > 0.003
            else "multiproto transports better" if ct - pt < -0.003 else "tied"
        )
        print(
            f"    → transported cone − multiproto: {ct - pt:+.4f} ({verdict})  |  "
            f"transported − dynamic (cone): {ct - cd:+.4f}"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "[warning] no CUDA — feature extraction on CPU is slow (cache hits are fine)."
        )

    for name in DATASETS:
        _push_globals(name)
        try:
            Ftr, ytr, Fte, yte = djf.get_features(device)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        T = djf.fit_feature_transform(
            Ftr,
            ytr,
            TRANSFORM,
            ridge=WHITEN_RIDGE,
            lda_shrink=LDA_SHRINK,
            alpha=WHITEN_ALPHA,
            k=PCA_KEEP,
        )
        Ftr, Fte = T(Ftr), T(Fte)
        classes = sorted(np.unique(ytr).tolist())
        if CLASS_LIMIT:
            classes = classes[:CLASS_LIMIT]
            m_tr, m_te = np.isin(ytr, classes), np.isin(yte, classes)
            Ftr, ytr, Fte, yte = Ftr[m_tr], ytr[m_tr], Fte[m_te], yte[m_te]
        rows = run_drift(Ftr, ytr, Fte, yte, classes)
        _print(name, rows, len(classes))


if __name__ == "__main__":
    main()
