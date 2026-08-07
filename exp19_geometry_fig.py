#!/usr/bin/env python3
"""exp19_geometry_fig.py — render the geometry exp19 measured.

TOP ROW is exact, not a cartoon: on S^2 the three cores are literally visualisable, and
every method in the table is cos(geodesic distance to a core set) -- they differ ONLY in
the core. cap (1 point) -> tube (great circle) -> tube around a spherical simplex.
The simplex core is a SUBSET of the great-circle core, which is what non-negativity does.

BOTTOM ROW is real CUB200 exp16 features under exp19's P3 split (fit on classes 0-99,
ID = test rows of 0-99, OOD = test rows of 100-199), so the claim "signal is radial, not
angular" is shown from data rather than asserted.

USAGE  source ~/venvs/ml_env/bin/activate && python -u exp19_geometry_fig.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from conic_hull import ConicHull

REPO = os.path.dirname(os.path.abspath(__file__))
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated slots 1,2,3 (all-pairs, light)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#c9c8c3"
SURF = "#fcfcfb"
R_FIT = 64

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def un(A):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- top row: the cores
def fib_sphere(n=24000):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


ELEV, AZIM, TUBE = 16.0, 38.0, 30.0
VIEW = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
                 np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
                 np.sin(np.radians(ELEV))])


def arc(a0, a1, n=800):
    """Sub-arc of the SAME great circle (the xy-plane) -- so the three cores nest:
    point  subset  arc  subset  full circle, which is exactly mean / cone / subspace."""
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.c_[np.cos(t), np.sin(t), np.zeros(n)]


def draw_sphere(ax, P, core, title, sub, colour):
    """Region = every point within TUBE degrees of `core`; core drawn on top in ink."""
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    u, v = np.mgrid[0:2 * np.pi:90j, 0:np.pi:46j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color="#eeede9", shade=False, linewidth=0, antialiased=True, zorder=0)
    inside = (P @ core.T).max(1) >= np.cos(np.radians(TUBE))
    q = P[inside & (P @ VIEW > 0.05)] * 1.012
    ax.scatter(*q.T, s=2.4, c=colour, alpha=0.62, linewidths=0, zorder=3)
    cv = core[core @ VIEW > 0.05] * 1.08
    if len(cv) > 1:
        ax.plot(*cv.T, color=INK, lw=2.8, zorder=5, solid_capstyle="round")
    else:
        ax.scatter(*cv.T, s=150, c=INK, zorder=10, edgecolors=SURF, linewidths=1.4)
    # caption goes in the TITLE, not under the axes -- a 3d axes fills its box and
    # anything at y<=0.05 lands behind the sphere.
    ax.set_title(f"{title}\n{sub}", color=INK, fontsize=9.4, pad=4, linespacing=1.7)
    ax.set_xlim(-.60, .60); ax.set_ylim(-.60, .60); ax.set_zlim(-.60, .60)


# ---------------------------------------------------------------- bottom row: real data
def load_p3(ds="CUB200"):
    from datasets import load_dataset
    z = np.load(os.path.join(
        REPO, f"exp16_feats_{ds}_T10_s0_ep40_lr0.0003_aug1_augreg_in21k.npz"))
    dd = load_dataset("Donghyun99/cub-200-2011", cache_dir=os.path.join(REPO, "data/hf"))
    ytr, yte = np.array(dd["train"]["label"]), np.array(dd["test"]["label"])
    Ztr, Zte = un(z["Ftr"]).astype(np.float32), un(z["Fte"]).astype(np.float32)
    seen = np.arange(100)
    return Ztr[np.isin(ytr, seen)], Zte[np.isin(yte, seen)], Zte[~np.isin(yte, seen)]


X, ID, OOD = load_p3()
B = np.linalg.svd(X, full_matrices=False)[2][:R_FIT]


def radial_angular(Q):
    Z = Q @ B.T
    e = np.linalg.norm(Z, axis=1)                          # cos angle to the subspace
    U = Z / (e[:, None] + 1e-12)
    return e, U


e_fit, U_fit = radial_angular(X)
lo, hi = np.quantile(U_fit, 0.01, 0), np.quantile(U_fit, 0.99, 0)
mid, half = (lo + hi) / 2, (hi - lo) / 2 + 1e-8


def stats(Q):
    e, U = radial_angular(Q)
    return e, np.linalg.norm((U - mid) / half, axis=1) / np.sqrt(R_FIT)


e_id, a_id = stats(ID)
e_od, a_od = stats(OOD)
lab = np.r_[np.ones(len(ID)), np.zeros(len(OOD))]
auc_rad = roc_auc_score(lab, np.r_[e_id, e_od])
auc_ang = roc_auc_score(lab, -np.r_[a_id, a_od])
hull = ConicHull(n_rays=R_FIT, use_pca=True, pca_dim=3 * R_FIT + 16,
                 ray_diversity="hybrid").fit(X)
c_id, c_od = hull.score(ID), hull.score(OOD)
auc_cone = roc_auc_score(lab, np.r_[c_id, c_od])
print(f"AUROC radial {auc_rad:.4f}  angular {auc_ang:.4f}  cone {auc_cone:.4f}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(11.6, 7.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.16, wspace=0.26,
                      left=0.055, right=0.975, top=0.845, bottom=0.085)

P = fib_sphere()
# Same great circle, same tube radius, same camera in all three -- only the CORE shrinks.
draw_sphere(fig.add_subplot(gs[0, 0], projection="3d"), P, arc(52, 52, 1),
            "core = 1 point  →  cap", "`mean`      P3 0.704", S3)
draw_sphere(fig.add_subplot(gs[0, 1], projection="3d"), P, arc(4, 100),
            "core = arc of that circle  →  carved tube", "`cone` (hull)      P3 0.732", S2)
draw_sphere(fig.add_subplot(gs[0, 2], projection="3d"), P, arc(0, 360),
            "core = the whole great circle  →  tube", "`subspace`      P3 0.887", S1)

# (d) the money plot -----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
sl = np.random.default_rng(0).permutation(len(e_od))[:len(e_id)]
ax.scatter(e_od[sl], a_od[sl], s=3.5, c=S2, alpha=0.30, linewidths=0)
ax.scatter(e_id, a_id, s=3.5, c=S1, alpha=0.30, linewidths=0)
ax.axvline(np.median(e_id), color=S1, lw=1.2, ls="--", alpha=.9)
ax.axvline(np.median(e_od), color=S2, lw=1.2, ls="--", alpha=.9)
ax.set_xlabel("radial:  ‖Bq‖ = cos∠(q, subspace)")
ax.set_ylabel("angular:  position within subspace")
ax.set_title(f"separation is radial only\nAUROC  radial {auc_rad:.3f}   "
             f"angular {auc_ang:.3f}", fontsize=9, color=INK)
ax.text(0.03, 0.95, "seen (ID)", color=S1, transform=ax.transAxes, fontsize=8.5,
        va="top", fontweight="bold")
ax.text(0.03, 0.87, "unseen (OOD)", color=S2, transform=ax.transAxes, fontsize=8.5,
        va="top", fontweight="bold")
ax.grid(alpha=.18, lw=.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (e) score distributions ------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(0.55, 1.0, 70)
ax.hist(c_id, bins=bins, color=S2, alpha=.42, label=None, lw=0)
ax.hist(c_od, bins=bins, histtype="step", color=S2, lw=1.6)
ax.hist(e_id, bins=bins, color=S1, alpha=.42, lw=0)
ax.hist(e_od, bins=bins, histtype="step", color=S1, lw=1.6)
ax.set_xlabel("score  (filled = seen / ID,  outline = unseen / OOD)")
ax.set_ylabel("count")
ax.set_title("the hull's ID/OOD overlap;\nthe tube's separate", fontsize=9, color=INK)
ax.text(0.42, 0.93, f"cone  AUROC {auc_cone:.3f}", color=S2, transform=ax.transAxes,
        fontsize=8.5, fontweight="bold")
ax.text(0.42, 0.85, f"subspace  AUROC {auc_rad:.3f}", color=S1, transform=ax.transAxes,
        fontsize=8.5, fontweight="bold")
ax.grid(alpha=.18, lw=.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (f) budget conversion --------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
cells = json.load(open(os.path.join(REPO, "exp19_dataset_hull_augreg_in21k.json"))
                  )["adapted|T10|s0"]["cells"]
Rs = [8, 24, 64, 128, 256]
for name, col, lb in [("subspace", S1, "subspace (tube)"), ("facet", S3, "facet (H-rep)"),
                      ("cone", S2, "cone (V-rep hull)")]:
    y = [cells[f"{name}|{r}"]["p3_mean"] for r in Rs]
    ax.plot(Rs, y, "-o", color=col, lw=2.0, ms=5.5, mew=0, label=lb)
    ax.annotate(lb, (Rs[-1], y[-1]), textcoords="offset points", xytext=(-4, 8),
                ha="right", color=col, fontsize=8.5, fontweight="bold")
ax.set_xscale("log", base=2); ax.set_xticks(Rs)
ax.set_xticklabels([str(r) for r in Rs])
ax.set_xlabel("budget R  (stored 768-d vectors)")
ax.set_ylabel("P3 near-OOD AUROC  (mean of 4 datasets)")
ax.set_title("V-rep grows where it should shrink\n(Δ +0.039 vs H-rep +0.220)",
             fontsize=9, color=INK)
ax.grid(alpha=.18, lw=.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.suptitle("Every method is  cos(geodesic distance to a core set)  —  they differ only "
             "in the core", fontsize=12.5, color=INK, y=0.985)
fig.text(0.5, 0.945, "top: same great circle, same tube radius, same camera in all three — only the "
                     "CORE shrinks  (P3 = mean of 4 datasets)   ·   bottom: CUB200 features, exp19 P3 split",
         ha="center", color=INK2, fontsize=8.6)
out = os.path.join(REPO, "exp19_geometry.png")
fig.savefig(out, dpi=190)
print("wrote", out)
