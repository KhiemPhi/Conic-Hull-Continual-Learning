#!/usr/bin/env python3
"""exp19_regions_fig.py — the four datasets, and the region each method's score carves out.

WHAT IS DRAWN
    Scatter  = real exp16 test features, one colour per dataset.
    Outlines = the ACTUAL accept region of each descriptor, one per dataset, in that
               dataset's colour, at the threshold containing 80% of its own held-out rows.
               So four shapes per panel: if a method can tell the datasets apart, its four
               regions sit on their own clouds and do not swallow the others.

HOW THE 2-D IS HONEST
    Descriptors are fitted in the FULL 768-d space on real rows -- nothing is fitted in 2-d.
    The plane is PC1/PC2 of the pooled fit rows. Each pixel is mapped BACK to 768-d
    (mean + g1*pc1 + g2*pc2, renormalised to the unit sphere) and scored by the real
    descriptor, so every outline is a genuine planar slice through the 768-d score field.
    It is a slice, not a projection of the region: a shape can look like it covers a cloud
    here and miss it in the 766 directions that are not drawn. The AUROC in each title is
    the full-dimensional number and is what actually adjudicates.

USAGE  source ~/venvs/ml_env/bin/activate && python -u exp19_regions_fig.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exp19_dataset_hull as E

REPO = os.path.dirname(os.path.abspath(__file__))
DS = ["CIFAR100", "IMAGENETR", "IMAGENETA", "CUB200"]
COL = {"CIFAR100": "#2a78d6", "IMAGENETR": "#eb6834",      # validated all-pairs, light
       "IMAGENETA": "#1baf7a", "CUB200": "#4a3aa7"}
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#c9c8c3", "#fcfcfb"
R = 64
NG = 200                     # grid resolution
N_SCAT = 900                 # points plotted per dataset
PANELS = [("mean", "cap  (1 stored vector)"), ("cone", "conic hull  (V-rep)"),
          ("facet", "facet cone  (H-rep)"), ("subspace", "subspace tube"),
          ("kmeans", "k-means  (union of caps)"), ("maha", "Mahalanobis  (ellipsoid)")]

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ data
D = {}
for ds in DS:
    F = E.adapted_features(ds)
    ytr, yte, ncls = E.get_labels(ds)
    Ztr, Zte = F
    p = rng.permutation(len(Ztr))
    ncal = max(int(0.1 * len(p)), 50)
    fit = p[ncal:][:6000]
    D[ds] = dict(fit=Ztr[fit], yfit=ytr[fit], cal=Ztr[p[:ncal]],
                 te=Zte[rng.permutation(len(Zte))[:N_SCAT]])
    print(f"{ds}: fit {len(fit)} cal {ncal}")

POOL = np.concatenate([D[d]["fit"] for d in DS])
mu = POOL.mean(0)
PC = np.linalg.svd(POOL - mu, full_matrices=False)[2][:2]        # the drawing plane


def to2d(X):
    return (X - mu) @ PC.T


# grid -> back to 768-d -> renormalise (features live on the unit sphere)
G2 = np.concatenate([to2d(D[d]["te"]) for d in DS])
pad = 0.06 * (G2.max(0) - G2.min(0))
x0, y0 = G2.min(0) - pad
x1, y1 = G2.max(0) + pad
gx, gy = np.meshgrid(np.linspace(x0, x1, NG), np.linspace(y0, y1, NG))
GRID = E.un(mu + np.c_[gx.ravel(), gy.ravel()] @ PC).astype(np.float32)

# Only draw where the slice is supported by real rows. A grid pixel far from every feature
# is a synthetic point on a 2-plane that no image ever produced; contouring it invents
# shape out of extrapolation. Masking keeps the outlines honest AND readable.
from scipy.spatial import cKDTree
_tree = cKDTree(G2)
MASK = (_tree.query(np.c_[gx.ravel(), gy.ravel()])[0]
        > 0.085 * np.hypot(x1 - x0, y1 - y0)).reshape(NG, NG)

# ------------------------------------------------------------------ figure
fig, AX = plt.subplots(2, 3, figsize=(13.4, 8.6))
fig.subplots_adjust(left=.035, right=.988, top=.815, bottom=.075, wspace=.10, hspace=.20)

for ax, (meth, nice) in zip(AX.ravel(), PANELS):
    aurocs = []
    for ds in DS:
        g = np.random.default_rng(0)
        f = E.FIT[meth](D[ds]["fit"], D[ds]["yfit"], R, g)
        thr = np.quantile(f(D[ds]["cal"]), 0.20)                 # region holds 80% of own
        Zg = f(GRID).reshape(NG, NG).astype(np.float64)
        Zg[MASK] = np.nan
        ax.contourf(gx, gy, Zg, levels=[thr, np.nanmax(Zg) + 1e-9],
                    colors=[COL[ds]], alpha=.16)
        ax.contour(gx, gy, Zg, levels=[thr], colors=[COL[ds]], linewidths=2.2,
                   linestyles="solid")
        idv = f(D[ds]["te"])
        oov = np.concatenate([f(D[e]["te"]) for e in DS if e != ds])
        from sklearn.metrics import roc_auc_score
        aurocs.append(roc_auc_score(np.r_[np.ones(len(idv)), np.zeros(len(oov))],
                                    np.r_[idv, oov]))
    for ds in DS:                                                # clouds on top of fills
        P2 = to2d(D[ds]["te"])
        ax.scatter(*P2.T, s=3.0, c=COL[ds], alpha=.40, linewidths=0, zorder=4)
    ax.set_title(f"{nice}\nmean 1-vs-rest AUROC (768-d)  {np.mean(aurocs):.3f}",
                 fontsize=9.6, color=INK, linespacing=1.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)

# direct labels (relief rule: aqua is below 3:1 on this surface)
for i, ds in enumerate(DS):
    fig.text(0.045 + i * 0.150, 0.880, "●  " + ds, color=COL[ds], fontsize=10.5,
             fontweight="bold", va="center")
fig.suptitle("The region each score actually carves out, drawn on the real features",
             fontsize=13, color=INK, y=0.972)
fig.text(0.5, 0.934, "outline = that dataset's accept region at the threshold holding 80% "
                     "of its own held-out rows   ·   descriptors fitted in full 768-d; "
                     "the plane is a slice, not a projection",
         ha="center", color=INK2, fontsize=8.6)
out = os.path.join(REPO, "exp19_regions.png")
fig.savefig(out, dpi=175)
print("wrote", out)
