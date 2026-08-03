import numpy as np
from itertools import combinations
from conic_lib import unit, cone_fit, cone_residual_deg, spherical_kmeans, vmf_distance_deg, auroc

def make_parts(P, d, seed):
    rng = np.random.default_rng(seed)
    return unit(np.abs(rng.normal(size=(P, d))))     # positive orthant => acute pointed cone

def gen(parts_matrix, combos, n_per, noise, seed):
    rng = np.random.default_rng(seed)
    X = []
    for parts in combos:
        G = parts_matrix[list(parts)]
        a = rng.random((n_per, len(parts)))           # nonneg coeffs => conic combination
        pts = a @ G + noise * rng.normal(size=(n_per, parts_matrix.shape[1]))
        X.append(pts)
    return unit(np.vstack(X))

def run(P=12, d=128, k=3, m=12, n_per=200, noise=0.05, seed=0):
    G       = make_parts(P, d, seed)                  # true extreme rays
    G_ood   = make_parts(4, d, seed + 999)            # DISJOINT parts => hard OOD (same process)
    all_c   = list(combinations(range(P), k))
    rng     = np.random.default_rng(seed); rng.shuffle(all_c)
    cut     = len(all_c) // 2
    seen, unseen = all_c[:cut], all_c[cut:]

    X_seen   = gen(G,     seen,                       n_per, noise, seed + 1)
    X_unseen = gen(G,     unseen,                     n_per, noise, seed + 2)   # novel combos, same parts
    X_ood    = gen(G_ood, list(combinations(range(4), 2)), n_per, noise, seed + 3)

    W = cone_fit(X_seen, m)
    C = spherical_kmeans(X_seen, m, seed=seed)

    r = dict(cone_seen=cone_residual_deg(W, X_seen),   cone_unseen=cone_residual_deg(W, X_unseen),
             cone_ood=cone_residual_deg(W, X_ood),
             vmf_seen=vmf_distance_deg(C, X_seen),      vmf_unseen=vmf_distance_deg(C, X_unseen),
             vmf_ood=vmf_distance_deg(C, X_ood))
    med = {k_: np.median(v) for k_, v in r.items()}

    print(f"P={P} k={k} m={m} d={d}  seen={len(seen)} unseen={len(unseen)} combos")
    print(f"  CONE  seen {med['cone_seen']:5.1f}° | unseen {med['cone_unseen']:5.1f}° | ood {med['cone_ood']:5.1f}°  "
          f"gen-gap {med['cone_unseen']-med['cone_seen']:+5.1f}°  AUROC(unseen vs ood) {auroc(r['cone_unseen'], r['cone_ood']):.3f}")
    print(f"  vMF   seen {med['vmf_seen']:5.1f}° | unseen {med['vmf_unseen']:5.1f}° | ood {med['vmf_ood']:5.1f}°  "
          f"gen-gap {med['vmf_unseen']-med['vmf_seen']:+5.1f}°  AUROC(unseen vs ood) {auroc(r['vmf_unseen'], r['vmf_ood']):.3f}")
    return r

if __name__ == "__main__":
    for m in (6, 12, 24):        # matched-param bake-off: same m for both models
        run(m=m)
