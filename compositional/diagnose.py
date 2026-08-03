"""Disambiguate 'bug vs finding' for the synthetic control.

Q1 (mechanism): with ORACLE generators W=G, does the cone give ~noise° residual on
    seen/unseen and beat vMF on AUROC(unseen vs ood)?  If yes → mechanism is sound.
Q2 (fit): does SPA recover the true parts?  We measure, per recovered ray, the max
    cosine to any true part.  Anchor-free data (k>=2 always) should give POOR recovery;
    injecting k=1 pure-part 'anchors' should FIX it.
Q3 (regime): sweep k (parts-per-sample).  The cone's structural edge should appear only
    where m prototypes can't tile the mixture simplex (high k, m=P).
"""
import numpy as np
from itertools import combinations
from conic_lib import (unit, spa, cone_fit, cone_residual_deg, spherical_kmeans,
                       vmf_distance_deg, auroc)
from synthetic_control import make_parts, gen


def recovery_quality(W, G):
    """W: (d,m) recovered rays; G: (P,d) true parts. Return median over recovered
    rays of the max cosine to any true part (1.0 = perfect part recovery)."""
    Wc = unit(W.T)            # (m,d)
    Gc = unit(G)              # (P,d)
    cos = Wc @ Gc.T           # (m,P)
    return float(np.median(cos.max(1)))


def one(P=12, d=128, k=3, m=12, n_per=200, noise=0.05, seed=0, verbose=True):
    G = make_parts(P, d, seed)
    G_ood = make_parts(4, d, seed + 999)
    all_c = list(combinations(range(P), k))
    rng = np.random.default_rng(seed); rng.shuffle(all_c)
    cut = len(all_c) // 2
    seen, unseen = all_c[:cut], all_c[cut:]

    X_seen = gen(G, seen, n_per, noise, seed + 1)
    X_unseen = gen(G, unseen, n_per, noise, seed + 2)
    X_ood = gen(G_ood, list(combinations(range(4), 2)), n_per, noise, seed + 3)

    # --- three cone fits ---
    W_spa = cone_fit(X_seen, m)                                   # as in harness
    W_oracle = unit(G).T                                          # true parts (m=P)
    # anchor-augmented: inject pure single-part samples (k=1) then SPA
    X_anchor = np.vstack([X_seen, gen(G, [(i,) for i in range(P)], 30, noise, seed + 7)])
    W_aug = cone_fit(X_anchor, m)

    C = spherical_kmeans(X_seen, m, seed=seed)                    # matched vMF

    def cone_line(name, W):
        rs = cone_residual_deg(W, X_seen)
        ru = cone_residual_deg(W, X_unseen)
        ro = cone_residual_deg(W, X_ood)
        rec = recovery_quality(W, G)
        if verbose:
            print(f"  CONE[{name:7s}] seen {np.median(rs):5.1f}° unseen {np.median(ru):5.1f}° "
                  f"ood {np.median(ro):5.1f}°  gap {np.median(ru)-np.median(rs):+4.1f}°  "
                  f"AUROC {auroc(ru, ro):.3f}  part-recovery(cos) {rec:.3f}")
        return auroc(ru, ro), np.median(ru) - np.median(rs)

    if verbose:
        print(f"\nP={P} k={k} m={m} d={d}  seen={len(seen)} unseen={len(unseen)}")
    a_spa, g_spa = cone_line("spa", W_spa)
    a_aug, g_aug = cone_line("spa+anch", W_aug)
    a_orc, g_orc = cone_line("oracle", W_oracle)

    vs = vmf_distance_deg(C, X_seen); vu = vmf_distance_deg(C, X_unseen); vo = vmf_distance_deg(C, X_ood)
    a_vmf = auroc(vu, vo); g_vmf = np.median(vu) - np.median(vs)
    if verbose:
        print(f"  vMF            seen {np.median(vs):5.1f}° unseen {np.median(vu):5.1f}° "
              f"ood {np.median(vo):5.1f}°  gap {g_vmf:+4.1f}°  AUROC {a_vmf:.3f}")
    return dict(a_spa=a_spa, a_aug=a_aug, a_orc=a_orc, a_vmf=a_vmf,
                g_spa=g_spa, g_aug=g_aug, g_orc=g_orc, g_vmf=g_vmf)


if __name__ == "__main__":
    print("="*80)
    print("Q1/Q2 — mechanism & fit at the ORIGINAL regime (P=12,k=3), m=P=12")
    one(P=12, k=3, m=12)

    print("\n" + "="*80)
    print("Q3 — regime sweep: does the cone's edge appear where m can't tile the simplex?")
    print("(cone uses oracle parts, m=P; vMF gets the same m=P prototypes)")
    for P, k in [(8, 2), (8, 4), (8, 6), (12, 3), (12, 6), (12, 9)]:
        r = one(P=P, k=k, m=P, verbose=False)
        edge_auc = r["a_orc"] - r["a_vmf"]
        edge_gap = r["g_vmf"] - r["g_orc"]   # positive => cone generalizes better than vMF
        print(f"  P={P:2d} k={k}: oracle-cone AUROC {r['a_orc']:.3f} vs vMF {r['a_vmf']:.3f} "
              f"(edge {edge_auc:+.3f}) | gen-gap vMF {r['g_vmf']:+4.1f}° cone {r['g_orc']:+4.1f}° "
              f"(cone tighter by {edge_gap:+4.1f}°)")
