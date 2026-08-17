#!/usr/bin/env python3
"""make_paper2_figs.py -- the two figures for the NeurReps extended abstract.

Regenerated from the crux_* npz files, never hand-drawn, for the same reason results_paper2.txt
is generated: a figure edited by hand drifts from the arrays silently.

FIG 1 -- THE GAUGE CLAIM (Result 2). Task-0 cohort across a ten-task sequence:
    left panel  stale vs one-Procrustes-transport vs current-frame oracle, 3-seed mean with
                min-max band. The claim is that transport TRACKS oracle while stale decays.
    right panel eps (total drift angle) and the residual left after removing one global
                rotation. The claim is that eps SATURATES and the residual stays a small
                fraction of it -- i.e. the drift explores a bounded, mostly-rotational orbit.

FIG 2 -- AFFINE-NESS AND THE INTERVENTION (Results 1 and 4).
    left panel  reconstruction cosine to phi_0 by transport hypothesis class
                (identity / orthogonal / linear / MLP). The claim is SATURATION at `linear`:
                the MLP buys nothing, so the drift is affine at this measurement budget.
    right panel lambda=0 vs lambda=50 on the three quantities the gauge reading predicts a
                Gram penalty should move -- rigidity, Gram correlation, and the feature-quality
                oracle -- with the frozen-backbone baseline drawn as the honest reference.

RIGIDITY IS DERIVED, NOT STORED: rigidity% = 100*(1 - rigid_deg/eps), exactly as
crux_sweep.py:185 prints it. It is UNDEFINED at t=0, where eps is 0.007-0.012 deg and
rigid_deg is 0-0.006 deg -- both numerical noise, so the ratio is noise/noise and the scripts'
inline max(eps,1e-6) guard turns it into an arbitrary finite number (it printed 71.6%). t=0 is
therefore dropped from any rigidity curve rather than plotted.

USAGE
    source ~/venvs/ml_env/bin/activate
    python -u make_paper2_figs.py        # writes fig1_gauge.pdf/.png, fig2_affine_gram.pdf/.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.4,
    }
)
C = {
    "stale": "#c44e52",
    "transport": "#4c72b0",
    "oracle": "#55a868",
    "eps": "#8172b2",
    "resid": "#937860",
    "frozen": "#666666",
}


def band(ax, x, Y, color, label, marker="o", per_seed=True):
    """Thick 3-seed mean plus thin per-seed traces.

    A min-max envelope was tried first and rejected: with three arms on one panel the envelopes
    overlapped almost completely and hid the actual claim (transport indistinguishable from
    oracle, stale clearly below). Thin traces keep the seed spread visible without burying it.
    """
    if per_seed:
        for y in Y:
            ax.plot(x, y, "-", color=color, alpha=0.25, linewidth=0.6)
    m = Y.mean(0)
    ax.plot(x, m, marker=marker, ms=2.8, color=color, label=label, zorder=3)
    return m


def rigidity(eps, rd):
    """Undefined before the backbone has adapted -- see module docstring."""
    return np.where(eps < 1.0, np.nan, 100.0 * (1.0 - rd / np.maximum(eps, 1e-9)))


def fig1():
    S = [np.load(os.path.join(REPO, f"crux_sweep_hist_s{s}.npz")) for s in SEEDS]
    t = np.arange(len(S[0]["eps"]))
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.5), constrained_layout=True)

    st = np.stack([z["coh_stale"] for z in S]) * 100
    tr = np.stack([z["coh_transport"] for z in S]) * 100
    oc = np.stack([z["coh_oracle"] for z in S]) * 100
    band(ax[0], t, oc, C["oracle"], "oracle (refit in current frame)", "s")
    band(ax[0], t, tr, C["transport"], "one Procrustes map", "o")
    band(ax[0], t, st, C["stale"], "stale prototypes", "v")
    ax[0].set_xlabel("task")
    ax[0].set_ylabel("task-0 cohort accuracy (%)")
    ax[0].set_title("(a) a single rotation tracks the oracle at every depth")
    ax[0].legend(loc="lower left", frameon=False)
    gap = np.abs(tr.mean(0) - oc.mean(0)).max()
    ax[0].annotate(
        f"|transport $-$ oracle|\n$\\leq$ {gap:.2f} pts at all {len(t)} stages",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=6.6,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
    )

    ep = np.stack([z["eps"] for z in S])
    rd = np.stack([z["rigid_deg"] for z in S])
    band(ax[1], t, ep, C["eps"], r"total drift $\varepsilon$", "o")
    band(ax[1], t, rd, C["resid"], "residual after one rotation", "d")
    ax[1].set_xlabel("task")
    ax[1].set_ylabel("angle (deg)")
    ax[1].set_title("(b) the drift saturates and stays rotational")
    ax[1].legend(loc="upper left", frameon=False)
    rg = rigidity(ep.mean(0), rd.mean(0))
    ax[1].annotate(
        f"rigidity {np.nanmin(rg):.0f}-{np.nanmax(rg):.0f}% of drift\n"
        rf"removed by ONE rotation",
        xy=(0.97, 0.34),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=6.6,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
    )

    for e in ("pdf", "png"):
        fig.savefig(os.path.join(REPO, f"fig1_gauge.{e}"))
    plt.close(fig)
    print(
        f"fig1_gauge: max |transport-oracle| = {gap:.4f} pts; "
        f"eps {ep.mean(0)[1]:.1f} -> {ep.mean(0)[-1]:.1f} deg; "
        f"rigidity {np.nanmin(rg):.1f}-{np.nanmax(rg):.1f}%"
    )


def fig2():
    T = [np.load(os.path.join(REPO, f"crux_transport_s{s}.npz")) for s in SEEDS]
    R = {
        l: [
            np.load(os.path.join(REPO, f"crux_relational_lam{l}_s{s}.npz"))
            for s in SEEDS
        ]
        for l in (0, 50)
    }
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.5), constrained_layout=True)

    # ---- (a) transport hypothesis classes. The npz stores (recon_cos, NCM) PAIRS per variant;
    # averaging across both axes yields a number that is neither (an error made 2026-08-14).
    cls = ["identity", "orthogonal", "linear", "mlp"]
    probe = T[0]["lam=0|identity"]
    assert np.shape(probe) == (
        2,
    ), f"expected (recon_cos, NCM) pairs, got {np.shape(probe)}"
    rc = np.array([[t[f"lam=0|{c}"][0] for c in cls] for t in T])
    x = np.arange(len(cls))
    ax[0].fill_between(
        x, rc.min(0), rc.max(0), color=C["transport"], alpha=0.16, linewidth=0
    )
    ax[0].plot(x, rc.mean(0), "o-", ms=3.4, color=C["transport"])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["identity\n(no map)", "orthogonal", "linear", "MLP"])
    ax[0].set_ylabel(r"reconstruction $\cos$ to $\phi_0$")
    ax[0].set_title("(a) the drift is affine: MLP buys nothing")
    d = rc.mean(0)
    ax[0].annotate(
        rf"identity $=\cos({np.degrees(np.arccos(d[0])):.0f}\degree)$",
        xy=(0.03, d[0]),
        xycoords=("axes fraction", "data"),
        fontsize=6.6,
        va="bottom",
    )
    ax[0].annotate(
        f"MLP - linear = {d[3]-d[2]:+.3f}",
        xy=(0.97, 0.12),
        xycoords="axes fraction",
        ha="right",
        fontsize=6.6,
    )

    # ---- (b) the intervention. Rigidity/Gram on the left axis (both 0-100 scales), the
    # feature-quality oracle on the right with the frozen baseline as the honest reference.
    def fin(l, key):
        return np.array([z[key][-1] for z in R[l]])

    rig = {l: rigidity(fin(l, "eps"), fin(l, "rigid_deg")) for l in (0, 50)}
    gc = {l: fin(l, "gram_corr") * 100 for l in (0, 50)}
    orc = {l: fin(l, "seen_oracle") * 100 for l in (0, 50)}
    frozen = fin(0, "seen_frozen").mean() * 100

    xs = np.array([0, 1])
    w = 0.3
    ax[1].bar(
        xs - w / 2,
        [rig[0].mean(), rig[50].mean()],
        w,
        color=C["resid"],
        label="rigidity (%)",
    )
    ax[1].bar(
        xs + w / 2,
        [gc[0].mean(), gc[50].mean()],
        w,
        color=C["eps"],
        label="Gram corr (x100)",
    )
    for i, l in enumerate((0, 50)):
        ax[1].plot([xs[i] - w / 2] * 3, rig[l], ".", color="k", ms=2.2)
        ax[1].plot([xs[i] + w / 2] * 3, gc[l], ".", color="k", ms=2.2)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([r"$\lambda=0$", r"$\lambda=50$"])
    ax[1].set_ylabel("rigidity / Gram corr")
    ax[1].set_ylim(0, 132)
    ax[1].set_title("(b) the Gram penalty moves what the geometry predicts")

    a2 = ax[1].twinx()
    a2.spines["top"].set_visible(False)
    a2.errorbar(
        xs,
        [orc[0].mean(), orc[50].mean()],
        yerr=[
            [orc[l].mean() - orc[l].min() for l in (0, 50)],
            [orc[l].max() - orc[l].mean() for l in (0, 50)],
        ],
        fmt="s-",
        ms=3.4,
        color=C["oracle"],
        capsize=2,
        label="oracle (feature quality)",
    )
    a2.axhline(frozen, ls="--", lw=1.0, color=C["frozen"])
    a2.annotate(
        f"frozen backbone {frozen:.1f}",
        xy=(0.02, frozen),
        xycoords=("axes fraction", "data"),
        fontsize=6.6,
        va="bottom",
        color=C["frozen"],
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0),
    )
    a2.set_ylabel("accuracy (%)")
    a2.set_ylim(70, 88)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        frameon=False,
        ncol=3,
        fontsize=6.2,
        columnspacing=1.0,
        handlelength=1.4,
        bbox_to_anchor=(0.5, 1.02),
    )

    for e in ("pdf", "png"):
        fig.savefig(os.path.join(REPO, f"fig2_affine_gram.{e}"))
    plt.close(fig)
    print(
        f"fig2_affine_gram: recon {dict(zip(cls, np.round(d,4)))}; "
        f"rigidity {rig[0].mean():.1f}->{rig[50].mean():.1f}%; "
        f"oracle {orc[0].mean():.3f}->{orc[50].mean():.3f} vs frozen {frozen:.3f}"
    )


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote fig1_gauge.{pdf,png} and fig2_affine_gram.{pdf,png}")
