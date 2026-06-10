"""
test_collaborative_scoring.py
──────────────────────────────
Ablation: per-class (independent) NNLS residuals vs. joint collaborative NNLS scoring.

Two feature generation modes:
  "random" : Gaussian-perturbed mean directions on the unit sphere (basic baseline)
  "conic"  : features as nonneg linear combinations of class atoms (proper conic structure)

The conic mode is where the SRC-style collaborative scoring is expected to help: each
query genuinely lies in the convex cone of its class's atoms, so the joint NNLS should
concentrate weights on the true class's dictionary entries.

Usage:
    cd /workspace/conic_hull_improved
    python test_collaborative_scoring.py
"""

import sys, time
import numpy as np
from sklearn.preprocessing import normalize

sys.path.insert(0, "/workspace/conic_hull_improved")
from conic_hull import ConicHull, build_class_conic_hulls, collaborative_nnls_scores


# ─────────────────────────────────────────────────────────────────────────────
# Feature generation
# ─────────────────────────────────────────────────────────────────────────────

def make_random_features(
    n_classes, samples_per_cls, dim, angular_spread, inter_class_sep, rng
):
    """
    Gaussian-perturbed mean directions: simple baseline.
    Classes occupy overlapping directional caps when sep is small.
    """
    raw_means = rng.standard_normal((n_classes, dim))
    class_means = normalize(raw_means, axis=1)

    # Gram-Schmidt separation
    if inter_class_sep > 0.0:
        ortho = []
        for i, m in enumerate(class_means):
            v = m.copy()
            for o in ortho:
                v -= (v @ o) * o
            n = np.linalg.norm(v)
            if n > 1e-6:
                v /= n
            ortho.append(v)
            alpha = min(1.0, inter_class_sep / (np.pi / 2))
            blended = (1 - alpha) * m + alpha * v
            blended /= np.linalg.norm(blended) + 1e-12
            class_means[i] = blended
        class_means = normalize(class_means, axis=1)

    feature_dict = {}
    for c in range(n_classes):
        noise = rng.standard_normal((samples_per_cls, dim)) * angular_spread
        feats = normalize(class_means[c] + noise, axis=1).astype(np.float32)
        feature_dict[str(c)] = feats
    return feature_dict, class_means


def make_conic_features(
    n_classes, samples_per_cls, dim, n_atoms, noise_scale, overlap_atoms, rng
):
    """
    Proper conic features: each sample is a nonneg linear combination of
    class-specific atoms (extreme rays) plus small noise.

    This guarantees each query genuinely lies near the convex cone of its class.
    Cross-class confusion is controlled via `overlap_atoms` (shared atoms).

    Parameters
    ----------
    n_atoms       : extreme rays per class (conic hull support)
    overlap_atoms : number of atoms shared with the adjacent class (0 = independent)
    """
    # Generate orthonormal atom sets per class
    class_atoms = []
    for c in range(n_classes):
        raw = rng.standard_normal((n_atoms + overlap_atoms, dim))
        Q, _ = np.linalg.qr(raw.T, mode="reduced")  # (dim, n_atoms+overlap)
        atoms = Q.T[:n_atoms]                         # (n_atoms, dim) orthonormal rows
        class_atoms.append(atoms.astype(np.float32))

    # Introduce overlap: replace some atoms of class c with atoms from class c-1
    if overlap_atoms > 0:
        for c in range(1, n_classes):
            n_own = n_atoms - overlap_atoms
            class_atoms[c][:n_own]  = class_atoms[c][:n_own]      # keep own
            class_atoms[c][n_own:]  = class_atoms[c - 1][n_own:]  # borrow last `overlap` from prev

    # Generate training samples as nonneg linear combos of atoms + noise
    feature_dict = {}
    for c in range(n_classes):
        atoms = class_atoms[c]                                         # (n_atoms, dim)
        weights = rng.exponential(1.0, (samples_per_cls, n_atoms))     # nonneg weights
        feats = weights @ atoms                                         # (N, dim)
        feats += rng.standard_normal(feats.shape) * noise_scale
        feature_dict[str(c)] = normalize(feats, axis=1).astype(np.float32)

    return feature_dict, class_atoms


def make_conic_test_queries(class_atoms, samples_per_cls, noise_scale, rng):
    """Test queries drawn from the same conic distribution as training."""
    n_atoms = class_atoms[0].shape[0]
    parts, labels = [], []
    for c, atoms in enumerate(class_atoms):
        weights = rng.exponential(1.0, (samples_per_cls, n_atoms))
        feats = weights @ atoms
        feats += rng.standard_normal(feats.shape) * noise_scale
        parts.append(normalize(feats, axis=1).astype(np.float32))
        labels.extend([c] * samples_per_cls)
    return np.vstack(parts), np.array(labels)


def make_random_test_queries(class_means, samples_per_cls, angular_spread, rng):
    parts, labels = [], []
    for c, mean in enumerate(class_means):
        noise = rng.standard_normal((samples_per_cls, len(mean))) * angular_spread
        feats = normalize(mean + noise, axis=1).astype(np.float32)
        parts.append(feats)
        labels.extend([c] * samples_per_cls)
    return np.vstack(parts), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def independent_accuracy(hulls, queries, true_labels, score_key="cosine"):
    all_classes = list(hulls.keys())
    per_class   = [hulls[name].score_all(queries) for name in all_classes]
    mat         = np.stack([pc[score_key] for pc in per_class], axis=1)
    preds       = np.array([int(all_classes[i]) for i in np.argmax(mat, axis=1)])
    return float((preds == true_labels).mean())


def independent_margin_accuracy(hulls, queries, true_labels, score_key="cosine"):
    """Relative scoring: best - runner-up margin.  Same argmax, confirms it's equivalent."""
    all_classes = list(hulls.keys())
    per_class   = [hulls[name].score_all(queries) for name in all_classes]
    mat         = np.stack([pc[score_key] for pc in per_class], axis=1)   # (N, C)
    # Margin = col_score - 2nd_best_score (does not change argmax)
    sorted_m    = np.sort(mat, axis=1)[:, ::-1]
    runner_up   = sorted_m[:, 1:2]
    margin_mat  = mat - runner_up
    preds       = np.array([int(all_classes[i]) for i in np.argmax(margin_mat, axis=1)])
    return float((preds == true_labels).mean())


def run_collab(hulls, queries, true_labels, lam=0.0):
    scores_dict, class_names = collaborative_nnls_scores(queries, hulls, lasso_lambda=lam)
    results = {}
    for sk in ["collab_energy", "collab_residual", "collab_margin"]:
        mat   = scores_dict[sk]
        preds = np.array([int(class_names[i]) for i in np.argmax(mat, axis=1)])
        results[sk] = float((preds == true_labels).mean())
    return results


def print_results(indep, collab_all, label, total_queries, n_classes, n_rays):
    best_indep = max(indep.values())
    print(f"\n  {label}")
    print(f"  {'─' * 58}")
    print(f"  {'Scheme':<36} {'Acc':>6}  {'vs best-indep':>12}")
    print(f"  {'─' * 58}")
    print("  [Independent per-class NNLS]")
    for k, v in indep.items():
        tag = " ◄" if v == best_indep else ""
        print(f"  {k:<36} {v*100:>5.1f}%{tag}")
    print("  [Collaborative joint NNLS]")
    all_collab = {k: v for d in collab_all for k, v in d.items()}
    best_collab = max(all_collab.values()) if all_collab else 0.0
    for k, v in all_collab.items():
        delta = v - best_indep
        sign  = "+" if delta >= 0 else ""
        tag   = " ◄" if v == best_collab else ""
        print(f"  {k:<36} {v*100:>5.1f}%  ({sign}{delta*100:.1f}pp){tag}")
    lift = best_collab - best_indep
    print(f"\n  Lift: {lift*100:+.1f}pp  "
          f"({'WIN' if lift > 0.5e-2 else 'NEUTRAL' if abs(lift) <= 0.5e-2 else 'LOSS'})")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runners
# ─────────────────────────────────────────────────────────────────────────────

def run_random_scenario(name, n_classes, train_n, test_n, dim,
                         spread, sep, n_rays, seed=0):
    print(f"\n{'━'*65}")
    print(f"  [RANDOM] {name}")
    print(f"  {n_classes} classes · train={train_n} test={test_n} · "
          f"dim={dim} spread={spread:.2f} sep={sep:.2f} rays={n_rays}")
    print(f"{'━'*65}")
    rng = np.random.default_rng(seed)
    fd, means = make_random_features(n_classes, train_n, dim, spread, sep, rng)
    hulls = build_class_conic_hulls(fd, n_rays=n_rays, use_pca=False, min_samples=3)
    queries, labels = make_random_test_queries(means, test_n, spread, rng)

    indep = {
        "cosine":         independent_accuracy(hulls, queries, labels, "cosine"),
        "angular_margin": independent_accuracy(hulls, queries, labels, "angular_margin"),
        "blended":        independent_accuracy(hulls, queries, labels, "blended"),
    }
    collab_all = [
        {f"energy(λ={lam})":   run_collab(hulls, queries, labels, lam)["collab_energy"],
         f"residual(λ={lam})": run_collab(hulls, queries, labels, lam)["collab_residual"]}
        for lam in [0.0, 0.01, 0.1]
    ]
    print_results(indep, collab_all, "Results", len(queries), n_classes, n_rays)


def run_conic_scenario(name, n_classes, train_n, test_n, dim,
                        n_atoms, noise, overlap, n_rays, seed=0):
    print(f"\n{'━'*65}")
    print(f"  [CONIC]  {name}")
    print(f"  {n_classes} classes · train={train_n} test={test_n} · "
          f"dim={dim} atoms/class={n_atoms} noise={noise:.2f} "
          f"overlap={overlap} rays={n_rays}")
    print(f"{'━'*65}")
    rng = np.random.default_rng(seed)
    fd, class_atoms = make_conic_features(
        n_classes, train_n, dim, n_atoms, noise, overlap, rng
    )
    hulls = build_class_conic_hulls(fd, n_rays=n_rays, use_pca=False, min_samples=3)
    queries, labels = make_conic_test_queries(class_atoms, test_n, noise, rng)

    indep = {
        "cosine":         independent_accuracy(hulls, queries, labels, "cosine"),
        "angular_margin": independent_accuracy(hulls, queries, labels, "angular_margin"),
        "blended":        independent_accuracy(hulls, queries, labels, "blended"),
    }
    collab_all = [
        {f"energy(λ={lam})":   run_collab(hulls, queries, labels, lam)["collab_energy"],
         f"residual(λ={lam})": run_collab(hulls, queries, labels, lam)["collab_residual"],
         f"margin(λ={lam})":   run_collab(hulls, queries, labels, lam)["collab_margin"]}
        for lam in [0.0, 0.01, 0.1]
    ]
    print_results(indep, collab_all, "Results", len(queries), n_classes, n_rays)

    # ── sparsity diagnostic ────────────────────────────────────────────────
    scores_dict, class_names = collaborative_nnls_scores(queries[:30], hulls, lasso_lambda=0.1)
    energy_mat = scores_dict["collab_energy"]   # (30, C)
    nonzero_cls = (energy_mat > 1e-4).sum(axis=1).mean()
    correct_energy = energy_mat[np.arange(30), labels[:30]].mean()
    total_energy   = energy_mat.sum(axis=1).mean() + 1e-8
    print(f"\n  [Diagnostic λ=0.1] avg classes with nonzero energy: {nonzero_cls:.1f} / {n_classes}")
    print(f"  [Diagnostic λ=0.1] avg fraction of energy in true class: "
          f"{correct_energy / total_energy * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Collaborative Scoring Ablation")
    print("  Joint NNLS vs Independent Per-Class NNLS")
    print("=" * 65)
    print("""
  Theory:
    Independent NNLS: C separate ‖R_c w_c - q‖² solves.  When cones
    overlap, neighbor classes achieve similar reconstruction cost, making
    the residuals miscalibrated relative to each other.

    Collaborative NNLS: one joint solve over D = [R_1; ...; R_C].
    Cross-class competition inside the program suppresses "neighbor
    reconstructs me almost as well" — the solver allocates weight to
    whichever class's dictionary best explains each query.

    For this to work, features must have *conic structure* (queries lie
    near the convex cone of their class), not just a directional mean.
  """)

    # ── Random baseline scenarios ──────────────────────────────────────────
    print("\n" + "━"*65)
    print("  PART 1 — Random features (Gaussian-perturbed means)")
    print("  Expected: collaborative ~ independent (no conic structure)")
    print("━"*65)

    run_random_scenario(
        "Well-separated", n_classes=10, train_n=60, test_n=40,
        dim=64, spread=0.15, sep=0.8, n_rays=8, seed=0,
    )
    run_random_scenario(
        "Correlated means", n_classes=10, train_n=60, test_n=40,
        dim=64, spread=0.30, sep=0.1, n_rays=8, seed=1,
    )

    # ── Conic structure scenarios ──────────────────────────────────────────
    print("\n" + "━"*65)
    print("  PART 2 — Proper conic features (nonneg linear combos of atoms)")
    print("  Expected: collaborative improves when cones share atoms (overlap>0)")
    print("━"*65)

    # No overlap: independent should work
    run_conic_scenario(
        "No atom overlap", n_classes=8, train_n=80, test_n=50,
        dim=64, n_atoms=6, noise=0.05, overlap=0, n_rays=6, seed=10,
    )
    # Moderate overlap: collaborative should help
    run_conic_scenario(
        "Moderate overlap (2/6 atoms shared)", n_classes=8, train_n=80, test_n=50,
        dim=64, n_atoms=6, noise=0.05, overlap=2, n_rays=6, seed=11,
    )
    # Heavy overlap: hardest case
    run_conic_scenario(
        "Heavy overlap (4/6 atoms shared)", n_classes=8, train_n=80, test_n=50,
        dim=64, n_atoms=6, noise=0.10, overlap=4, n_rays=6, seed=12,
    )

    # ── Scaling test ───────────────────────────────────────────────────────
    print("\n" + "━"*65)
    print("  PART 3 — Scaling: more classes (realistic continual learning)")
    print("━"*65)

    run_conic_scenario(
        "20 classes, moderate overlap", n_classes=20, train_n=50, test_n=30,
        dim=128, n_atoms=8, noise=0.08, overlap=3, n_rays=8, seed=20,
    )

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)
