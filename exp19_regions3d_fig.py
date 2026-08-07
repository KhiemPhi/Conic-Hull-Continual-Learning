#!/usr/bin/env python3
"""exp19_regions3d_fig.py — the 3-D version of exp19_regions_fig.py.

Same construction, one dimension up: descriptors are fitted in the FULL 768-d space, a
64^3 grid is laid out in the PC1/PC2/PC3 volume, every voxel is mapped BACK to 768-d
(mu + g.PC, renormalised to the unit sphere) and scored by the real descriptor.

The accept region is drawn as a CONTOUR STACK: the threshold isoline computed on each of
eight horizontal slices through the volume and drawn at its own height. That is a
mesh-free isosurface (scikit-image is unavailable offline, so marching cubes is out), and
unlike a point-sampled shell it does not occlude the data underneath it.
Voxels far from any real feature are dropped, so the shell is clipped to the data envelope
rather than extrapolated into a region no image ever occupies.

Rows = the dataset being described.  Columns = the method.  All four clouds in every panel,
so you can see whether a shape sits on its own data or swallows everyone else's.

USAGE  source ~/venvs/ml_env/bin/activate && python -u exp19_regions3d_fig.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import exp19_dataset_hull as E

REPO = os.path.dirname(os.path.abspath(__file__))
DS = ["CIFAR100", "IMAGENETR", "IMAGENETA", "CUB200"]
COL = {"CIFAR100": "#2a78d6", "IMAGENETR": "#eb6834",
       "IMAGENETA": "#1baf7a", "CUB200": "#4a3aa7"}
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#c9c8c3", "#fcfcfb"
R, NG, N_SCAT = 64, 64, 700
N_SLICE = 9                         # horizontal slices in the contour stack
ROWS = ["CIFAR100", "CUB200"]
COLS = [("cone", "conic hull (V-rep)"), ("facet", "facet cone (H-rep)"),
        ("subspace", "subspace tube"), ("maha", "Mahalanobis (ellipsoid)")]
ELEV, AZIM = 20, -58

plt.rcParams.update({"font.size": 8.5, "text.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ data
D = {}
for ds in DS:
    Ztr, Zte = E.adapted_features(ds)
    ytr, yte, _ = E.get_labels(ds)
    p = rng.permutation(len(Ztr))
    ncal = max(int(0.1 * len(p)), 50)
    D[ds] = dict(fit=Ztr[p[ncal:][:6000]], yfit=ytr[p[ncal:][:6000]], cal=Ztr[p[:ncal]],
                 te=Zte[rng.permutation(len(Zte))[:N_SCAT]])

POOL = np.concatenate([D[d]["fit"] for d in DS])
mu = POOL.mean(0)
PC = np.linalg.svd(POOL - mu, full_matrices=False)[2][:3]        # the drawing volume


def to3d(X):
    return (X - mu) @ PC.T


P3 = {d: to3d(D[d]["te"]) for d in DS}
ALL3 = np.concatenate([P3[d] for d in DS])
lo3, hi3 = ALL3.min(0), ALL3.max(0)
pad = 0.07 * (hi3 - lo3)
lo3, hi3 = lo3 - pad, hi3 + pad
ax_ = [np.linspace(lo3[i], hi3[i], NG) for i in range(3)]
gx, gy, gz = np.meshgrid(*ax_, indexing="ij")
G3 = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
NEAR = cKDTree(ALL3).query(G3)[0] < 0.085 * np.linalg.norm(hi3 - lo3)
print(f"grid {NG}^3 = {len(G3)},  supported by data: {NEAR.sum()}")
G3n = G3[NEAR]


def score_grid(f, chunk=32768):
    """Score only the data-supported voxels, mapping each chunk back to 768-d on the fly
    (the full grid at 768 floats/row would be ~800 MB)."""
    out = []
    for i in range(0, len(G3n), chunk):
        out.append(f(E.un(mu + G3n[i:i + chunk] @ PC).astype(np.float32)))
    return np.concatenate(out)


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(15.2, 9.4))
gs = fig.add_gridspec(2, 4, wspace=0.02, hspace=0.06,
                      left=0.01, right=0.99, top=0.845, bottom=0.01)

for r, host in enumerate(ROWS):
    for c, (meth, nice) in enumerate(COLS):
        ax = fig.add_subplot(gs[r, c], projection="3d")
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 0.9)); ax.margins(0)
        f = E.FIT[meth](D[host]["fit"], D[host]["yfit"], R, np.random.default_rng(0))
        thr = np.quantile(f(D[host]["cal"]), 0.20)      # region holds 80% of own rows
        full = np.full(NG ** 3, np.nan)
        full[NEAR] = score_grid(f)
        full = full.reshape(NG, NG, NG)
        gxx, gyy = np.meshgrid(ax_[0], ax_[1], indexing="ij")
        for k in np.linspace(4, NG - 5, N_SLICE).astype(int):        # the contour stack
            sl = full[:, :, k]
            if np.isfinite(sl).sum() < 40 or np.nanmin(sl) > thr or np.nanmax(sl) < thr:
                continue
            ax.contourf(gxx, gyy, sl, levels=[thr, np.nanmax(sl) + 1e-9], zdir="z",
                        offset=ax_[2][k], colors=[COL[host]], alpha=.10)
            ax.contour(gxx, gyy, sl, levels=[thr], zdir="z", offset=ax_[2][k],
                       colors=[COL[host]], linewidths=1.5, alpha=.85, linestyles="solid")
        for d in DS:                                    # clouds LAST so nothing hides them
            ax.scatter(*P3[d].T, s=2.6, c=COL[d], alpha=.42, linewidths=0,
                       depthshade=False, zorder=6)
        ax.set_xlim(lo3[0], hi3[0]); ax.set_ylim(lo3[1], hi3[1]); ax.set_zlim(lo3[2], hi3[2])
        if r == 0:
            ax.set_title(nice, fontsize=10.2, color=INK, pad=-30)
        ax.text2D(0.02, 0.06, f"describing {host}", transform=ax.transAxes,
                  color=COL[host], fontsize=9.2, fontweight="bold")
        print(f"  {host:10s} {meth:9s} thr {thr:.4f}")

for i, ds in enumerate(DS):
    fig.text(0.055 + i * 0.135, 0.885, "●  " + ds, color=COL[ds], fontsize=10.5,
             fontweight="bold", va="center")
fig.suptitle("The same accept regions in 3-D — contour stack of each score, "
             "on the real features", fontsize=13.5, color=INK, y=0.975)
fig.text(0.5, 0.935, "stacked isolines at the 80%-coverage threshold of the named dataset's "
                     "own score   ·   fitted in full 768-d, drawn in the PC1–3 volume, "
                     "clipped to the data envelope",
         ha="center", color=INK2, fontsize=8.6)
out = os.path.join(REPO, "exp19_regions3d.png")
fig.savefig(out, dpi=170)
print("wrote", out)
