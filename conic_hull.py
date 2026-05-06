"""
conic_hull.py
-------------
ConicHull — SPA (Successive Projection Algorithm) based extreme-ray finder.
build_class_conic_hulls — fits one hull per class.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm


class ConicHull:
    """
    Approximate conic hull using the Successive Projection Algorithm (SPA).

    A conic hull C(V) = { Σ λᵢ vᵢ | λᵢ ≥ 0 } is represented compactly by
    a set of *extreme rays* — the most directionally diverse unit vectors in
    the dataset — found via SPA.

    SPA is preferred over ConvexHull for high-dimensional ViT embeddings
    because it is O(n · m · d) instead of exponential in d.

    Parameters
    ----------
    n_rays  : number of extreme rays to extract (capped by dataset size)
    use_pca : reduce dimensions with PCA before SPA (speeds up projection maths)
    pca_dim : PCA target dimensionality
    """

    def __init__(self, n_rays: int = 50, use_pca: bool = True, pca_dim: int = 64):
        self.n_rays     = n_rays
        self.use_pca    = use_pca
        self.pca_dim    = pca_dim
        self.extreme_rays_: Optional[np.ndarray] = None
        self.pca_:          Optional[PCA]        = None

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "ConicHull":
        """
        Find the extreme rays from feature matrix X (N, D).

        Steps
        -----
        1. L2-normalise rows  →  project onto the unit sphere.
        2. Optional PCA reduction for speed.
        3. Run SPA on the reduced space.
        4. Store the *original-space* normalised vectors as extreme rays.
        """
        X_norm = normalize(X, axis=1)

        if self.use_pca and X_norm.shape[1] > self.pca_dim:
            self.pca_ = PCA(n_components=self.pca_dim)
            X_proc    = self.pca_.fit_transform(X_norm)
        else:
            X_proc = X_norm

        indices            = self._robust_density_spa(X_proc, self.n_rays)
        self.extreme_rays_ = X_norm[indices]
        self.extreme_rays_index = indices
        return self

    def _spa(self, X: np.ndarray, m: int) -> np.ndarray:
        """
        Successive Projection Algorithm.
        Finds m extreme rays of the conic hull of rows of X.
        X: (N, D) — N feature vectors in D-dimensional space
        m: number of extreme rays to select
        """
        indices = []
        resid   = X.copy().astype(np.float64)

        for _ in range(m):
            # 1. Find point with largest residual norm
            norms = np.linalg.norm(resid, axis=1)   # (N,) ← use norm not norm²
            idx   = int(np.argmax(norms))
            indices.append(idx)

            # 2. Project out the direction of the selected point
            u      = resid[idx]                              # (D,)
            u_norm = u / (np.linalg.norm(u) + 1e-12)        # unit vector (D,)
            proj   = resid @ u_norm                          # (N,) scalar projections
            resid  = resid - np.outer(proj, u_norm)          # (N, D) subtract component

        return np.array(indices)
    
    def _robust_density_spa(self, X: np.ndarray, m: int, k_neighbors: int = 5, outlier_percentile: int = 95) -> np.ndarray:
        """
        Robust Successive Projection Algorithm.
        Filters out isolated outliers before finding the extreme rays of the data manifold.
        
        X: (N, D) — N feature vectors
        m: number of images to select for the replay buffer (e.g., 20)
        k_neighbors: number of neighbors to evaluate local density
        outlier_percentile: threshold to define what constitutes an outlier
        """
        N, D = X.shape
        from sklearn.neighbors import NearestNeighbors
        
        # -------------------   --------------------------------------
        # STEP 1: Outlier Detection via Local Density
        # ---------------------------------------------------------
        # Find the distance to the k-nearest neighbors for every point
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Ignore the first column (distance to itself, which is 0)
        # Calculate the mean distance to its local neighborhood
        mean_local_dist = distances[:, 1:].mean(axis=1)
        
        # Determine the cutoff threshold. 
        # E.g., if outlier_percentile=95, the 5% most isolated points are marked as outliers.
        threshold = np.percentile(mean_local_dist, outlier_percentile)
        
        # Boolean mask: True if the point is an inlier, False if it's an outlier
        valid_inlier_mask = mean_local_dist <= threshold

        # ---------------------------------------------------------
        # STEP 2: Masked Successive Projection
        # ---------------------------------------------------------
        indices = []
        resid = X.copy().astype(np.float64)

        for _ in range(m):
            # Calculate residual norms
            norms = np.linalg.norm(resid, axis=1)
            
            # Apply the outlier mask: force outlier norms to -1 so they are never picked
            norms[~valid_inlier_mask] = -1.0
            
            # Force already-selected indices to -1 so we don't pick duplicates
            for idx in indices:
                norms[idx] = -1.0
                
            # 1. Find valid inlier with the largest residual norm
            best_idx = int(np.argmax(norms))
            indices.append(best_idx)

            # 2. Project out the direction of the selected point
            u = resid[best_idx]
            u_norm = u / (np.linalg.norm(u) + 1e-12)
            proj = resid @ u_norm
            resid = resid - np.outer(proj, u_norm)

        return np.array(indices)

    # ── inference ─────────────────────────────────────────────────────────────

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        """
        Approximate each query as a non-negative combination of extreme rays.
        Solves  min ‖R w − q‖²  s.t. w ≥ 0  (batched projected gradient on GPU).

        Falls back to CPU scipy NNLS if torch/CUDA is unavailable.

        Returns
        -------
        np.ndarray of shape (N_queries, n_rays) — the NNLS weight vectors
        """
        from sklearn.preprocessing import normalize

        queries_norm = normalize(queries, axis=1)          # (N, D)
        R            = self.extreme_rays_.T                # (D, K)

        # ── try GPU path ──────────────────────────────────────────────────────────
        if self._torch_cuda_available():
            return self._reconstruct_gpu(queries_norm, R)

        # ── CPU fallback (your original scipy NNLS) ───────────────────────────────
        return self._reconstruct_cpu(queries_norm, R)


    # ─── helpers ──────────────────────────────────────────────────────────────────

    def _torch_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


    def _reconstruct_gpu(self,
        queries_norm: np.ndarray,
        R:            np.ndarray,
        lr:           float = 1e-2,
        max_iter:     int   = 2_000,
        tol:          float = 1e-7,
        batch_size:   int   = 4_096,
    ) -> np.ndarray:
        """
        Batched projected gradient descent on GPU.

        min_{W ≥ 0}  ‖W R^T − Q‖²_F

        Gradient w.r.t W:  2 (W R^T − Q) R   → shape (N, K)
        Projection:        clamp(W, min=0)

        Armijo-style step size is estimated once on a small probe batch
        so lr is self-tuning regardless of the ray matrix conditioning.
        """
        import torch

        device = torch.device("cuda")
        dtype  = torch.float32

        R_t  = torch.tensor(R.T, dtype=dtype, device=device)   # (K, D)
        Q_all = torch.tensor(queries_norm, dtype=dtype, device=device)  # (N, D)

        # ── auto step-size via Lipschitz constant: L = λ_max(R R^T) ─────────────
        # power iteration is fast and avoids a full SVD
        with torch.no_grad():
            RRt   = R_t @ R_t.T                         # (K, K)
            v     = torch.randn(RRt.shape[0], device=device, dtype=dtype)
            v     = v / v.norm()
            for _ in range(50):                          # ~50 iters is enough
                v = RRt @ v
                v = v / v.norm()
            L  = (v @ RRt @ v).item()                   # largest eigenvalue
            lr = 1.0 / (L + 1e-8)

        N, K    = Q_all.shape[0], R_t.shape[0]
        W_out   = torch.zeros(N, K, dtype=dtype)        # collect results on CPU

        for start in range(0, N, batch_size):
            Q = Q_all[start : start + batch_size]       # (B, D)
            B = Q.shape[0]

            W = torch.zeros(B, K, dtype=dtype, device=device, requires_grad=False)

            prev_loss = float("inf")
            for _ in range(max_iter):
                residual = W @ R_t - Q                  # (B, D)
                grad     = residual @ R_t.T             # (B, K)  = dL/dW
                W        = (W - lr * grad).clamp_(min=0)

                # convergence check every 100 steps (cheap)
                if _ % 100 == 0:
                    loss = residual.pow(2).sum().item()
                    if abs(prev_loss - loss) < tol * (1 + abs(prev_loss)):
                        break
                    prev_loss = loss

            W_out[start : start + batch_size] = W.cpu()

        return W_out.numpy()


    def _reconstruct_cpu(self,
        queries_norm: np.ndarray,
        R:            np.ndarray,
    ) -> np.ndarray:
        """Original per-sample scipy NNLS (CPU fallback)."""
        from scipy.optimize import nnls

        max_iter = R.shape[1] * 10
        weights  = []
        for q in queries_norm:
            try:
                w, _ = nnls(R, q, maxiter=max_iter)
            except RuntimeError:
                R_jitter = R + 1e-9 * np.random.standard_normal(R.shape)
                w, _     = nnls(R_jitter, q, maxiter=max_iter * 2)
            weights.append(w)
        return np.array(weights)

    def score(self, queries: np.ndarray) -> np.ndarray:
        """
        Conic angular similarity score.
        1.0 = query is perfectly inside/on the cone.
        Lower values indicate the query lies outside the cone.

        Returns
        -------
        np.ndarray of shape (N_queries,)  — scores in (-1, 1]
        """
        weights       = self.reconstruct(queries)
        reconstructed = weights @ self.extreme_rays_
        q_n           = normalize(queries,       axis=1)
        r_n           = normalize(reconstructed, axis=1)
        return np.sum(q_n * r_n, axis=1)

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        if self.extreme_rays_ is None:
            return {"fitted": False}
        return {
            "fitted":        True,
            "n_extreme_rays": len(self.extreme_rays_),
            "ray_dim":        self.extreme_rays_.shape[1],
            "pca_used":       self.pca_ is not None,
        }

    def print_extreme_rays(
        self,
        label:    str           = "",
        max_rays: Optional[int] = None,
        max_dims: int           = 12,
    ) -> None:
        """Pretty-print the extreme rays for quick inspection."""
        if self.extreme_rays_ is None:
            print("[ConicHull] Not fitted — call fit() first.")
            return

        rays    = self.extreme_rays_
        K, D    = rays.shape
        shown   = rays if max_rays is None else rays[:max_rays]
        title   = f"Extreme Rays — {label}" if label else "Extreme Rays"
        width   = 70

        cos_all = np.clip(rays @ rays.T, -1.0, 1.0)
        np.fill_diagonal(cos_all, np.nan)
        mean_ang = np.degrees(np.arccos(np.nanmean(cos_all)))

        print("=" * width)
        print(f"  {title}")
        print(f"  rays={K}  dim={D}  pca={self.pca_ is not None}  "
              f"mean_angle={mean_ang:.1f}°")
        print("-" * width)
        print(f"  {'idx':>4}  {'‖v‖':>6}  {'mean':>7}  {'std':>7}  values (first {max_dims} dims)")
        print("-" * width)

        for i, ray in enumerate(shown):
            vals = "  ".join(f"{x:+.4f}" for x in ray[:max_dims])
            if D > max_dims:
                vals += f"  …({D - max_dims} more)"
            print(f"  {i:>4}  {np.linalg.norm(ray):>6.4f}  "
                  f"{ray.mean():>+7.4f}  {ray.std():>7.4f}  {vals}")

        if K > len(shown):
            print(f"  … {K - len(shown)} more rays not shown …")
        print("=" * width)


# ── per-class builder ─────────────────────────────────────────────────────────

def build_class_conic_hulls(
    feature_dict: Dict[str, np.ndarray],
    n_rays:       int  = 50,
    use_pca:      bool = True,
    pca_dim:      int  = 64,
    min_samples:  int  = 3,
) -> Dict[str, ConicHull]:
    """
    Fit a ConicHull for every class in `feature_dict`.

    Parameters
    ----------
    feature_dict : { class_name: np.ndarray (N, D) }
    n_rays       : max extreme rays per class
    use_pca      : apply PCA before SPA
    pca_dim      : PCA target dim (only used when use_pca=True)
    min_samples  : skip classes with fewer samples

    Returns
    -------
    { class_name: ConicHull }
    """
    sample_key = next(iter(feature_dict))
    input_dim  = feature_dict[sample_key].shape[1]

    # print("=" * 50)
    # print("CONIC HULL CONFIGURATION")
    # print(f"  Classes          : {len(feature_dict)}")
    # print(f"  Input dim        : {input_dim}")
    # print(f"  Max rays / class : {n_rays}")
    # print(f"  Min samples      : {min_samples}")
    # print(f"  Use PCA          : {use_pca}" +
    #       (f"  →  dim {pca_dim}" if use_pca else ""))
    # print("=" * 50)

    class_hulls: Dict[str, ConicHull] = {}

    pbar = tqdm(feature_dict.items(), desc="Building conic hulls",
                total=len(feature_dict), unit="class", dynamic_ncols=True)

    for class_name, vectors in pbar:
        pbar.set_postfix({"class": class_name})
        n = len(vectors)

        if n < min_samples:
            tqdm.write(f"  [skip] '{class_name}': {n} samples < {min_samples}")
            continue

        current_rays = min(n_rays, n)
        hull = ConicHull(n_rays=current_rays, use_pca=use_pca, pca_dim=pca_dim)
        hull.fit(vectors)
        class_hulls[class_name] = hull

        n_found = len(hull.extreme_rays_)
        if n_found < n_rays:
            tqdm.write(f"  [note] '{class_name}': {n_found} rays extracted")

    print(f"\nBuilt hulls for {len(class_hulls)}/{len(feature_dict)} classes.")
    return class_hulls
