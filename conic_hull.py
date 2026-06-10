"""
conic_hull.py
-------------
ConicHull — SPA (Successive Projection Algorithm) based extreme-ray finder.
build_class_conic_hulls — fits one hull per class.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
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
    k_local : if set, each query is reconstructed using only its k_local nearest
              extreme rays (by cosine similarity) rather than all rays.  This
              "shrink-wraps" the hull locally so that out-of-class queries that
              fall in the global hull's interior but outside any local patch are
              penalised correctly.  None = use all rays (original behaviour).
    """

    def __init__(self, n_rays: int = 50, use_pca: bool = True, pca_dim: int = 64,
                 k_local: Optional[int] = None,
                 ray_diversity: str = "hybrid",
                 spa_oversample: int = 3):
        self.n_rays          = n_rays
        self.use_pca         = use_pca
        self.pca_dim         = pca_dim
        self.k_local         = k_local
        self.ray_diversity   = ray_diversity   # "spa" | "fps" | "hybrid"
        self.spa_oversample  = spa_oversample  # multiplier for hybrid/fps candidate pool
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
        3. Select extreme rays via the configured diversity strategy.
        4. Store the *original-space* normalised vectors as extreme rays.

        ray_diversity
        -------------
        "spa"    : original robust-density SPA (may produce packed rays)
        "fps"    : farthest-point sampling on inliers (maximum spread, no
                   extremity guarantee)
        "hybrid" : (default) oversample with SPA, then FPS on candidates.
                   Combines extremity (SPA) with angular diversity (FPS).
        """
        X_norm = normalize(X, axis=1)

        if self.use_pca and X_norm.shape[1] > self.pca_dim:
            self.pca_ = PCA(n_components=self.pca_dim)
            X_proc    = self.pca_.fit_transform(X_norm)
        else:
            X_proc = X_norm

        mode = (self.ray_diversity or "hybrid").lower()
        if mode == "spa":
            indices = self._robust_density_spa(X_proc, self.n_rays)
        elif mode == "fps":
            indices = self._fps_inlier(X_proc, self.n_rays)
        else:  # hybrid
            indices = self._hybrid_spa_fps(X_proc, self.n_rays, self.spa_oversample)

        self.extreme_rays_      = X_norm[indices]
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
    
    def _robust_density_spa(self, X: np.ndarray, m: int, k_neighbors: int = 1, outlier_percentile: int = 95) -> np.ndarray:
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

    # ── farthest-point sampling on inliers ────────────────────────────────────

    def _fps_inlier(
        self,
        X: np.ndarray,
        m: int,
        k_neighbors: int = 5,
        outlier_percentile: int = 95,
    ) -> np.ndarray:
        """
        Farthest-point sampling restricted to inlier points.

        Picks m unit vectors with maximum minimum geodesic separation.
        All points are required to be L2-normalised before calling.
        """
        from sklearn.neighbors import NearestNeighbors

        N = X.shape[0]
        m = min(m, N)

        # Outlier filter
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, N)).fit(X)
        dists, _ = nbrs.kneighbors(X)
        mean_dist = dists[:, 1:].mean(axis=1)
        thr = np.percentile(mean_dist, outlier_percentile)
        valid = np.where(mean_dist <= thr)[0]

        if len(valid) == 0:
            valid = np.arange(N)

        X_v = X[valid]
        centroid = X_v.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        local_sel = [int(np.argmax(X_v @ centroid))]
        min_ang = np.full(len(X_v), np.inf, dtype=np.float64)

        for _ in range(m - 1):
            s = X_v[local_sel[-1]]
            cos = np.clip(X_v @ s, -1.0, 1.0)
            np.minimum(min_ang, np.arccos(cos), out=min_ang)
            min_ang[np.array(local_sel)] = -1.0
            nxt = int(np.argmax(min_ang))
            if min_ang[nxt] <= 1e-8:
                break
            local_sel.append(nxt)

        return valid[np.array(local_sel, dtype=int)]

    def _hybrid_spa_fps(
        self,
        X: np.ndarray,
        m: int,
        oversample: int = 3,
        k_neighbors: int = 5,
        outlier_percentile: int = 95,
    ) -> np.ndarray:
        """
        Hybrid SPA → FPS: addresses packed rays without losing extremity.

        Two-stage selection:
        1. Run robust SPA to collect up to ``oversample × m`` extreme-ray
           *candidates* — these are guaranteed to be at actual data extremes.
        2. Apply farthest-point sampling on those candidates to pick the
           final m rays with maximum angular diversity.

        Why this works
        --------------
        Pure SPA can produce a cluster of rays all pointing in nearly the same
        direction when the data distribution is anisotropic (one long axis,
        several short axes).  SPA exhausts the long axis quickly and then
        selects several nearly-identical vectors for the remaining slots.

        FPS on the SPA candidates forces maximum angular spread *within* the
        set of genuine extreme directions, so the hull covers the full class
        cone rather than over-sampling a single peak.
        """
        n_cand = min(m * oversample, X.shape[0])
        cand_global = self._robust_density_spa(X, n_cand, k_neighbors, outlier_percentile)

        if len(cand_global) <= m:
            return cand_global

        X_cand = X[cand_global]
        centroid = X_cand.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        local_sel = [int(np.argmax(X_cand @ centroid))]
        min_ang = np.full(len(X_cand), np.inf, dtype=np.float64)

        for _ in range(m - 1):
            s = X_cand[local_sel[-1]]
            cos = np.clip(X_cand @ s, -1.0, 1.0)
            np.minimum(min_ang, np.arccos(cos), out=min_ang)
            min_ang[np.array(local_sel)] = -1.0
            nxt = int(np.argmax(min_ang))
            if min_ang[nxt] <= 1e-8:
                break
            local_sel.append(nxt)

        return cand_global[np.array(local_sel, dtype=int)]

    # ── inference ─────────────────────────────────────────────────────────────

    def reconstruct(self, queries: np.ndarray,
                    k_local: Optional[int] = None) -> np.ndarray:
        """
        Approximate each query as a non-negative combination of extreme rays.
        Solves  min ‖R w − q‖²  s.t. w ≥ 0  (batched projected gradient on GPU).

        Parameters
        ----------
        k_local : override self.k_local for this call.  When set, each query
                  uses only its k_local nearest rays (cosine similarity) so the
                  reconstruction is local rather than global.

        Falls back to CPU scipy NNLS if torch/CUDA is unavailable.

        Returns
        -------
        np.ndarray of shape (N_queries, n_rays) — the NNLS weight vectors
        """
        from sklearn.preprocessing import normalize

        queries_norm = normalize(queries, axis=1)          # (N, D)
        R            = self.extreme_rays_.T                # (D, K)
        k            = k_local if k_local is not None else self.k_local

        # ── try GPU path ──────────────────────────────────────────────────────────
        if self._torch_cuda_available():
            return self._reconstruct_gpu_fast(queries_norm, R, k_local=k)

        # ── CPU fallback (your original scipy NNLS) ───────────────────────────────
        return self._reconstruct_cpu(queries_norm, R, k_local=k)


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
    
    def _reconstruct_gpu_fast(self,
        queries_norm: np.ndarray,
        R:            np.ndarray,
        max_iter:     int            = 500,
        tol:          float          = 1e-7,
        batch_size:   int            = 4_096,
        k_local:      Optional[int]  = None,
    ) -> np.ndarray:
        import torch
        import math

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype  = torch.float32

        R_t   = torch.tensor(R.T, dtype=dtype, device=device)   # (K, D)
        Q_all = torch.tensor(queries_norm, dtype=dtype, device=device)  # (N, D)

        H = R_t @ R_t.T                                          # (K, K)

        with torch.no_grad():
            v = torch.randn(H.shape[0], device=device, dtype=dtype)
            v = v / v.norm()
            for _ in range(50):
                v = H @ v
                v = v / v.norm()
            L  = (v @ H @ v).item()
            lr = 1.0 / (L + 1e-8)

        N, K  = Q_all.shape[0], R_t.shape[0]
        W_out = torch.zeros(N, K, dtype=dtype)

        # clamp k_local to valid range
        k = min(k_local, K) if k_local is not None else None

        for start in range(0, N, batch_size):
            Q = Q_all[start : start + batch_size]   # (B, D)
            B = Q.shape[0]

            # C = cosine similarities between queries and rays (both unit-normalised)
            C = Q @ R_t.T                            # (B, K)

            # ── local-k mask: each query only uses its k nearest rays ─────────────
            if k is not None:
                top_idx = torch.topk(C, k, dim=1).indices   # (B, k)
                mask    = torch.zeros(B, K, dtype=dtype, device=device)
                mask.scatter_(1, top_idx, 1.0)               # (B, K) binary
            else:
                mask = None

            W = torch.zeros(B, K, dtype=dtype, device=device)
            Y = torch.zeros(B, K, dtype=dtype, device=device)
            t = 1.0

            for i in range(max_iter):
                grad   = Y @ H - C
                W_next = (Y - lr * grad).clamp_(min=0)
                if mask is not None:
                    W_next = W_next * mask              # zero out non-local rays

                t_next   = (1.0 + math.sqrt(1.0 + 4.0 * t**2)) / 2.0
                momentum = (t - 1.0) / t_next
                Y = W_next + momentum * (W_next - W)
                W = W_next
                t = t_next

                if i % 50 == 0:
                    with torch.no_grad():
                        W_change = torch.norm(W_next - W) / (torch.norm(W) + 1e-8)
                        if W_change < tol:
                            break

            W_out[start : start + batch_size] = W.cpu()

        return W_out.numpy()


    def _reconstruct_cpu(self,
        queries_norm: np.ndarray,
        R:            np.ndarray,
        k_local:      Optional[int] = None,
    ) -> np.ndarray:
        """Per-sample scipy NNLS (CPU fallback).  Supports local-k restriction."""
        from scipy.optimize import nnls

        K        = R.shape[1]
        k        = min(k_local, K) if k_local is not None else None
        max_iter = K * 10
        weights  = []

        for q in queries_norm:
            if k is not None:
                sims    = R.T @ q                              # (K,) cosine sims
                top_idx = np.argpartition(sims, -k)[-k:]      # k largest indices
                R_local = R[:, top_idx]                        # (D, k)
                try:
                    w_local, _ = nnls(R_local, q, maxiter=k * 10)
                except RuntimeError:
                    w_local, _ = nnls(
                        R_local + 1e-9 * np.random.standard_normal(R_local.shape),
                        q, maxiter=k * 20,
                    )
                w       = np.zeros(K)
                w[top_idx] = w_local
            else:
                try:
                    w, _ = nnls(R, q, maxiter=max_iter)
                except RuntimeError:
                    w, _ = nnls(
                        R + 1e-9 * np.random.standard_normal(R.shape),
                        q, maxiter=max_iter * 2,
                    )
            weights.append(w)

        return np.array(weights)

    def score(self, queries: np.ndarray) -> np.ndarray:
        """
        Conic angular similarity score (cosine, original scheme).
        1.0 = query is perfectly inside/on the cone.

        Returns np.ndarray of shape (N_queries,) in (-1, 1].
        """
        weights       = self.reconstruct(queries)
        reconstructed = weights @ self.extreme_rays_
        q_n           = normalize(queries,       axis=1)
        r_n           = normalize(reconstructed, axis=1)
        return np.sum(q_n * r_n, axis=1)

    # ── alternative scoring schemes ───────────────────────────────────────────

    def score_residual_coverage(self, queries: np.ndarray) -> np.ndarray:
        """
        Cosine fidelity weighted by reconstruction coverage.

        Penalises under-reconstruction: a query outside the cone yields a
        small ‖reconstructed‖ relative to ‖query‖, reducing the score.

        Returns np.ndarray in [0, 1] (roughly).
        """
        weights       = self.reconstruct(queries)
        reconstructed = weights @ self.extreme_rays_
        q_n           = normalize(queries,       axis=1)
        r_n           = normalize(reconstructed, axis=1)
        cos_sim       = np.clip(np.sum(q_n * r_n, axis=1), -1.0, 1.0)
        q_norm        = np.linalg.norm(queries,       axis=1)
        r_norm        = np.linalg.norm(reconstructed, axis=1)
        coverage      = np.clip(r_norm / (q_norm + 1e-8), 0.0, 1.0)
        return cos_sim * coverage

    def score_angular(self, queries: np.ndarray) -> np.ndarray:
        """
        Signed angular distance to the cone boundary.

        Maps angular deviation linearly: 0° → +1, 90° → 0, 180° → -1.
        Consistent with the ε < γ/2 angular guarantee in CHCL theory.

        Returns np.ndarray in [-1, 1].
        """
        weights       = self.reconstruct(queries)
        reconstructed = weights @ self.extreme_rays_
        q_n           = normalize(queries,       axis=1)
        r_n           = normalize(reconstructed, axis=1)
        cos_sim       = np.clip(np.sum(q_n * r_n, axis=1), -1.0, 1.0)
        angle         = np.arccos(cos_sim)               # [0, π]
        return 1.0 - (2.0 * angle / np.pi)               # maps [0, π] → [1, -1]

    def score_nnls_residual(self, queries: np.ndarray) -> np.ndarray:
        """
        Unit-sphere NNLS residual norm, inverted to a similarity score.

        Directly measures the NNLS objective on the unit sphere.
        Avoids arccos instability and is numerically stable when
        the reconstruction is near zero.

        Returns np.ndarray in [0, 1].
        """
        queries_norm  = normalize(queries, axis=1)
        weights       = self.reconstruct(queries)
        reconstructed = weights @ self.extreme_rays_
        r_n           = normalize(reconstructed, axis=1)
        residuals     = np.linalg.norm(queries_norm - r_n, axis=1)  # [0, 2]
        return 1.0 - residuals / 2.0                                 # → [0, 1]

    def score_combined(self, queries: np.ndarray) -> np.ndarray:
        """
        Angular fidelity modulated by weight-entropy uniformity.

        Queries deep inside the cone are explained by many extreme rays
        (high entropy → high uniformity bonus). Boundary queries are
        explained by one or two rays (low entropy → smaller bonus).

        Returns np.ndarray in [0, 1] (roughly).
        """
        weights       = self.reconstruct(queries)
        reconstructed = weights @ self.extreme_rays_
        q_n           = normalize(queries,       axis=1)
        r_n           = normalize(reconstructed, axis=1)
        cos_sim       = np.clip(np.sum(q_n * r_n, axis=1), -1.0, 1.0)
        w_sum         = weights.sum(axis=1) + 1e-8
        w_prob        = weights / w_sum[:, None]
        entropy       = -np.sum(w_prob * np.log(w_prob + 1e-10), axis=1)
        max_ent       = np.log(weights.shape[1] + 1e-10)
        uniformity    = entropy / (max_ent + 1e-8)        # [0, 1]
        return cos_sim * (0.5 + 0.5 * uniformity)

    def score_all(self, queries: np.ndarray) -> dict:
        weights       = self.reconstruct(queries)           # (N, K)
        reconstructed = weights @ self.extreme_rays_        # (N, D)

        q_norm = np.linalg.norm(queries,       axis=1, keepdims=True)  # (N,1)
        r_norm = np.linalg.norm(reconstructed, axis=1, keepdims=True)

        q_n = queries       / (q_norm + 1e-8)
        r_n = reconstructed / (r_norm + 1e-8)

        cos_sim = np.clip(np.sum(q_n * r_n, axis=1), -1.0, 1.0)

       # 2. Nearest Ray Score (Continuous tie-breaker)
        # self.extreme_rays_ shape: (K, D)
        rays_norm = self.extreme_rays_ / (np.linalg.norm(self.extreme_rays_, axis=1, keepdims=True) + 1e-8)
        
        # (N, D) @ (D, K) -> (N, K)
        ray_similarities = q_n @ rays_norm.T
        max_ray_sim = np.max(ray_similarities, axis=1) # (N,)

        

        # --- 1. True geometric NNLS residual (unnormalized space) ---
        # How well does the cone actually reconstruct the ray direction?
        # Residual is measured against the *unit* query, not the normalized reconstructed.
        # This penalizes cases where reconstruction lands in a different direction entirely.
        raw_residual = np.linalg.norm(q_n - r_n, axis=1)   # in [0, 2]
        geo_residual = 1.0 - raw_residual / 2.0             # in [0, 1]

        # --- 2. Cone margin score ---
        # A query is "deep inside" the cone if each supporting ray has non-trivial weight.
        # Gini impurity of weights: high = concentrated on few rays (boundary), 
        # low = spread (interior). Invert so interior → higher score.
        w_sum  = weights.sum(axis=1, keepdims=True) + 1e-8
        w_prob = weights / w_sum                            # (N, K)
        # Normalized Herfindahl: 1 = one ray dominates (boundary), 1/K = uniform (interior)
        hhi        = np.sum(w_prob ** 2, axis=1)            # (N,)
        K          = weights.shape[1]
        hhi_norm   = (hhi - 1.0/K) / (1.0 - 1.0/K + 1e-8) # 0=interior, 1=single ray
        margin     = 1.0 - hhi_norm                         # 1=interior, 0=boundary

        # --- 3. Angular + margin combined (the principled version) ---
        # This rewards being both angularly close AND deep inside the cone
        angular_margin = cos_sim * (0.3 + 0.7 * margin)

        # --- 4. Reconstruction fidelity in full space ---
        # Unlike geo_residual (direction only), this also captures scale alignment.
        # Useful if your queries have meaningful magnitude variation.
        r_norm_sq = r_norm.squeeze() / (q_norm.squeeze() + 1e-8)  # scale ratio
        scale_pen  = np.exp(-np.abs(np.log(r_norm_sq + 1e-8)))     # 1 if scale matches
        full_fidelity = geo_residual * scale_pen

        # --- 5. SPA-aware score: use support size as confidence ---
        # Queries needing fewer rays to reconstruct → more likely truly inside cone
        support_size = (weights > 1e-6).sum(axis=1).astype(float)   # (N,)
        support_norm = 1.0 - (support_size - 1.0) / (K - 1.0 + 1e-8)  # 1=1 ray, 0=all rays
        sparse_score  = cos_sim * (0.5 + 0.5 * support_norm)

        # 1. Global: Cosine similarity to the centroid of the rays
        centroid = np.mean(self.extreme_rays_, axis=0)
        
        
        # 2. Local: Mean cosine similarity to the top-k nearest extreme rays
        # (Assuming self.extreme_rays_ are L2 normalized)
        all_sims = queries @ self.extreme_rays_.T  # (N, num_rays)
        top_k_sims = np.sort(all_sims, axis=1)[:, -7:] # Get top k for each query
        local_sim = np.mean(top_k_sims, axis=1)
        
        # Blend them (alpha can be tuned on a validation set, often 0.5 is a good start)
        score_blend = 0.5 * cos_sim + 0.5 * local_sim

        # 1. Soft-Max Ray Similarity (Density-Aware)
        # Temperature controls how many nearby rays contribute. Lower = sharper focus.
        temp = 0.05 
        # Shift similarities to avoid vanishing gradients/underflow
        shifted_sims = ray_similarities - np.max(ray_similarities, axis=1, keepdims=True)
        exp_sims = np.exp(shifted_sims / temp)
        # Weighted average of similarities
        soft_ray_sim = np.sum(exp_sims * ray_similarities, axis=1) / (np.sum(exp_sims, axis=1) + 1e-8)

        # 2. Weight-Norm Sparsity Ratio (Reconstruction Effort)
        # L1 norm / L2 norm. 
        # If perfectly 1 ray is used: L1/L2 = 1.0
        # If uniformly K rays are used: L1/L2 = sqrt(K)
        l1_norm = np.linalg.norm(weights, ord=1, axis=1)
        l2_norm = np.linalg.norm(weights, ord=2, axis=1)
        # We invert it so higher score = more efficient (sparse) reconstruction
        weight_efficiency = l2_norm / (l1_norm + 1e-8) 

        # 3. Blended Soft-Density Score (The new standard)
        # Rewards points that are geometrically inside the cone AND sit in a dense region of extreme rays
        soft_blended = 0.5 * cos_sim + 0.5 * soft_ray_sim

        return {
            "cosine":          cos_sim,
            "geo_residual":    geo_residual,        # replaces your broken nnls_residual
            "margin":          margin,              # pure interiority signal
            "angular_margin":  angular_margin,      # best general-purpose candidate
            "full_fidelity":   score_blend,       # if scale matters
            "sparse_support":  sparse_score,        # if sparse reconstruction = confidence
            "max_ray_sim": max_ray_sim,
            "blended": soft_blended
        }
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


# ── Kernel Conic Hull ─────────────────────────────────────────────────────────

from typing import Optional
import numpy as np
from sklearn.preprocessing import normalize


class KernelConicHull:
    """
    Conic hull with the kernel trick applied to both SPA and NNLS.

    Instead of dot products in the original D-dimensional space, all
    geometry is computed through a kernel K(x, y), implicitly lifting
    vectors into a (potentially infinite-dimensional) feature space where
    classes have far more room to spread out.

    Supported kernels
    -----------------
    "rbf"    : K(x,y) = exp(−γ ‖x − y‖²)
    "poly"   : K(x,y) = (γ x·y + c₀)^d
    "vmf"    : peaked vMF mixture — tight cones
    "spread" : chordal kernel + farthest-point rays + geometric-mean scoring.
               Selects evenly spaced extreme rays and scores queries by how
               well they align with *all* rays, not just the nearest one.

    Kernel SPA
    ----------
    Runs SPA on the Gram matrix K[i,j] = K(xᵢ, xⱼ) via rank-1 Gram–Schmidt
    deflation — no explicit feature vectors needed.

    Kernel NNLS
    -----------
    Solves   min_{w≥0}  wᵀ G w − 2 wᵀ k_{qR} + K(q,q)
    where G[i,j] = K(rᵢ,rⱼ) and k_{qR}[i] = K(q,rᵢ).
    Uses FISTA (projected gradient with Nesterov momentum).
    """

    def __init__(
        self,
        n_rays: int   = 50,
        kernel: str   = "rbf",
        gamma:  float = 5.0,
        degree: int   = 3,
        coef0:  float = 1.0,
        ridge:  float = 1e-6,   # regularization on G_ to handle near-singular cases
        kappa: float = 50.0
    ):
        self.n_rays  = n_rays
        self.kernel  = kernel
        self.gamma   = gamma
        self.degree  = degree
        self.coef0   = coef0
        self.ridge   = ridge
        self.kappa = kappa

        self.extreme_rays_:      Optional[np.ndarray] = None  # (K, D) normalized
        self.extreme_rays_index: Optional[np.ndarray] = None
        self.G_:                 Optional[np.ndarray] = None  # (K, K) Gram matrix of rays
        self._lip_:              Optional[float]      = None  # cached Lipschitz constant

    # ── kernel evaluation ──────────────────────────────────────────────────────

    def _score_vmf(self, queries: np.ndarray) -> np.ndarray:
        queries_norm = normalize(queries, axis=1)
        K_qR = self._K(queries_norm, self.extreme_rays_)   # (N, K), values in [0, 1]
        
        # log-mean-exp; max value is 0 (when query matches a ray exactly)
        log_K   = np.log(K_qR + 1e-30)                     # (N, K), values in (−∞, 0]
        max_log = log_K.max(axis=1, keepdims=True)         # (N, 1)
        lme     = max_log.squeeze() + np.log(
                    np.exp(log_K - max_log).mean(axis=1)
                )                                        # (N,) in (−∞, 0]
        
        # exp(lme) is already in [0, 1] — no normalization needed
        return -np.exp(lme)                                 # (N,)

    def _log_Z(self) -> float:
        from scipy.special import ive
        D, nu = self.extreme_rays_.shape[1], self.extreme_rays_.shape[1] / 2.0 - 1.0
        return (
            (D / 2.0) * np.log(2.0 * np.pi)
            + np.log(ive(nu, self.kappa) + 1e-300)
            + self.kappa
            - nu * np.log(self.kappa + 1e-300)
    )

    def _K(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Return kernel matrix of shape (|X|, |Y|).  Both X and Y must be L2-normalized.

        Kernels
        -------
        "rbf"  : K(x,y) = exp(−γ ‖x−y‖²)
                Standard RBF. On unit sphere: ‖x−y‖² = 2(1−x·y).

        "poly" : K(x,y) = (γ x·y + c₀)^d
                Polynomial kernel.

        "vmf"  : K(x,y) = exp(κ · x·y) / Z(κ)
                Von Mises-Fisher kernel. Treats each ray as a vMF distribution
                on the sphere with concentration κ. Scoring becomes
                log-mean-exp over rays rather than NNLS reconstruction.
                Unlike RBF, this kernel has a direct probabilistic interpretation:
                K(x,y) is proportional to the vMF likelihood of x given center y.
                κ controls hull sharpness:
                κ → 0   : uniform/flat (no discrimination)
                κ = 10  : moderate, smooth hull boundaries  (recommended start)
                κ → ∞   : collapses to nearest-ray argmax
                Z(κ) is the vMF normalizing constant (see below). When comparing
                scores within a single class it cancels out, but it matters for
                cross-class margin scoring so it is computed explicitly.
        """
        if self.kernel == "rbf":
            sq = np.clip(2.0 - 2.0 * (X @ Y.T), 0.0, None)
            return np.exp(-self.gamma * sq)

        elif self.kernel == "poly":
            return (self.gamma * (X @ Y.T) + self.coef0) ** self.degree

        elif self.kernel == "vmf":
            # ── vMF kernel, bounded form ─────────────────────────────────────
            # K(x, y) = exp(κ · (x·y − 1))
            #
            # This is the vMF density up to a multiplicative constant exp(κ)/Z(κ),
            # rescaled so that K(x, x) = 1 for unit-normalized inputs.
            #
            # Why drop Z(κ):
            #   • Z(κ, D) for high D is tiny (log Z ≈ −D for moderate κ), making
            #     the unnormalized density exp(κ·x·y)/Z(κ) overflow float64.
            #   • Z is a global constant per (κ, D) — it cancels in every meaningful
            #     comparison: within-class log-mean-exp, cross-class margin scores,
            #     and ratio-based membership probabilities.
            #
            # Why subtract κ inside the exp:
            #   • Equivalent to dividing through by exp(κ), the kernel's max value.
            #   • Keeps the exponent in (−2κ, 0], so exp() never overflows
            #     regardless of κ or embedding dimension.
            #   • K(x, x) = exp(0) = 1   (perfect self-match)
            #     K(x,−x) = exp(−2κ)     (antipodal, ≈ 0 for large κ)
            cos_sim = X @ Y.T                                  # (|X|, |Y|) in [−1, 1]
            return np.exp(self.kappa * (cos_sim - 1.0)).astype(np.float32)

        elif self.kernel == "spread":
            # Chordal affinity on the unit sphere: K = ((1 + cos) / 2)^p
            # p = spread_power (stored in gamma).  p → 0 gives flatter, more
            # uniform angular support; p = 1 is linear chordal distance.
            cos_sim = np.clip(X @ Y.T, -1.0, 1.0)
            aff     = (1.0 + cos_sim) * 0.5
            power   = max(float(self.gamma), 1e-3)
            return np.power(aff, power).astype(np.float32)

        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'")

    @staticmethod
    def _farthest_point_spherical(X: np.ndarray, m: int) -> np.ndarray:
        """
        Farthest-point sampling on the unit sphere.

        Picks ``m`` indices whose corresponding unit vectors maximise the
        minimum geodesic separation — producing evenly spread extreme rays
        instead of the correlated picks from peaked kernel SPA.
        """
        N = X.shape[0]
        m = min(m, N)
        if m <= 0:
            return np.array([], dtype=int)

        centroid = normalize(X.mean(axis=0, keepdims=True), axis=1)[0]
        start    = int(np.argmax(X @ centroid))
        selected = [start]
        min_dist = np.full(N, np.inf, dtype=np.float64)

        for _ in range(m - 1):
            s      = X[selected[-1]]
            cos    = np.clip(X @ s, -1.0, 1.0)
            ang    = np.arccos(cos)
            min_dist = np.minimum(min_dist, ang)
            min_dist[selected] = -1.0
            nxt = int(np.argmax(min_dist))
            if min_dist[nxt] <= 1e-8:
                break
            selected.append(nxt)

        return np.array(selected, dtype=int)

    def _spread_affinity(self, cos_sim: np.ndarray) -> np.ndarray:
        """Per-pair chordal affinity in [0, 1] for the spread kernel."""
        aff   = (1.0 + np.clip(cos_sim, -1.0, 1.0)) * 0.5
        power = max(float(self.gamma), 1e-3)
        return np.power(aff, power)

    def _score_spread(self, queries: np.ndarray) -> np.ndarray:
        queries_norm = normalize(queries, axis=1)
        cos_sim      = queries_norm @ self.extreme_rays_.T
        aff          = self._spread_affinity(cos_sim)
        # Geometric mean: query must align with every ray, not just one peak.
        return np.exp(np.log(aff + 1e-8).mean(axis=1)).astype(np.float32)

    def _score_all_spread(self, queries: np.ndarray) -> dict:
        """
        Even mixture scoring for the spread kernel.

        membership  : geometric mean of ray affinities (broad cone interior)
        min_affinity: worst ray — penalises tight packing onto one direction
        weights     : normalised affinities (soft uniform allocation)
        """
        queries_norm = normalize(queries, axis=1)
        cos_sim      = queries_norm @ self.extreme_rays_.T              # (N, K)
        aff          = self._spread_affinity(cos_sim)                   # (N, K)
        K            = aff.shape[1]
        eps          = 1e-8

        log_aff      = np.log(aff + eps)
        membership   = np.exp(log_aff.mean(axis=1))                     # geom mean
        min_aff      = aff.min(axis=1)
        mean_aff     = aff.mean(axis=1)

        weights      = aff / (aff.sum(axis=1, keepdims=True) + eps)
        entropy      = -(weights * np.log(weights + eps)).sum(axis=1)
        effective_rays = np.exp(entropy)
        sparsity     = 1.0 - (effective_rays - 1.0) / max(K - 1, 1)

        residuals    = np.sqrt(np.clip(1.0 - membership ** 2, 0.0, 1.0))
        recon_sq     = 1.0 - membership ** 2

        return {
            "cosine":            membership,
            "kernel_residual":   residuals,
            "weights":           weights,
            "support_sparsity":  sparsity,
            "reconstruction_sq": recon_sq,
            "angular_margin":    min_aff,
            "max_cosine":        cos_sim.max(axis=1),
            "mean_affinity":     mean_aff,
            "min_affinity":      min_aff,
            "effective_rays":    effective_rays,
            "blended":           0.5 * membership + 0.5 * min_aff,
            "max_ray_sim":       mean_aff,
        }

    # ── fitting ────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "KernelConicHull":
        # FIX #1 + #4: normalize first and use X_norm consistently everywhere
        X_norm  = normalize(X, axis=1)
        if self.kernel == "spread":
            indices = self._farthest_point_spherical(X_norm, self.n_rays)
            K_full  = self._K(X_norm, X_norm)
        else:
            K_full  = self._K(X_norm, X_norm)                        # (N, N)
            indices = self._kernel_spa(K_full, self.n_rays)

        self.extreme_rays_       = X_norm[indices]               # (K, D) — normalized
        self.extreme_rays_index  = indices
        # FIX #8: add ridge regularization to G_ for numerical stability
        G_raw   = K_full[np.ix_(indices, indices)].copy()        # (K, K)
        self.G_ = G_raw + self.ridge * np.eye(len(indices))

        # FIX #5: cache Lipschitz constant at fit time, not at every inference call
        self._lip_ = self._compute_lipschitz(self.G_)

        return self

    def _compute_lipschitz(self, G: np.ndarray, n_iter: int = 200) -> float:
        """Largest eigenvalue of G via power iteration."""
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G_t = torch.tensor(G, dtype=torch.float32, device=device)
        v   = torch.randn(G_t.shape[0], device=device, dtype=torch.float32)
        v   = v / v.norm()
        with torch.no_grad():
            for _ in range(n_iter):
                v = G_t @ v
                L = v.norm()
                v = v / (L + 1e-12)
        return L.item()

    def _kernel_spa(self, K: np.ndarray, m: int) -> np.ndarray:
        """
        Kernel SPA via Gram-matrix rank-1 deflation.

        The diagonal of the residual Gram matrix K_resid tracks ‖φ(xᵢ)_resid‖²
        in the implicit feature space.  At each step we pick the largest-norm
        residual point, then deflate K_resid to remove that direction.
        """
        N        = K.shape[0]
        K_resid  = K.copy().astype(np.float64)
        indices: list = []
        # FIX #6: maintain a boolean mask instead of O(m) inner loop
        excluded = np.zeros(N, dtype=bool)

        for _ in range(min(m, N)):
            diag           = K_resid.diagonal().copy()
            diag[excluded] = -np.inf
            i              = int(np.argmax(diag))
            if K_resid[i, i] < 1e-10:
                break
            indices.append(i)
            excluded[i] = True

            col   = K_resid[:, i].copy()
            denom = K_resid[i, i]
            K_resid -= np.outer(col, col) / denom

        return np.array(indices)

    # ── inference ──────────────────────────────────────────────────────────────

    def _kernel_nnls(
        self,
        queries:    np.ndarray,
        max_iter:   int   = 500,
        tol:        float = 1e-7,
        batch_size: int   = 4_096,
    ) -> tuple:
        """
        FISTA-based kernel NNLS on GPU (PyTorch CUDA).

        Objective per query q:
            min_{w≥0}  wᵀ G w − 2 wᵀ k_{qR} + K(q,q)

        Falls back to CPU if CUDA is unavailable.

        Returns
        -------
        weights   : (N, K)  non-negative NNLS weights
        residuals : (N,)    √(kernel-space squared residual), normalised by √K(q,q)
        """
        import torch

        if self.extreme_rays_ is None or self.G_ is None:
            raise RuntimeError("KernelConicHull is not fitted. Call fit() first.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype  = torch.float32

        # FIX #1: extreme_rays_ are already normalized at fit() time; just normalize queries
        queries_norm = normalize(queries, axis=1)

        # Compute cross-kernel between queries and extreme rays (both normalized)
        K_qR_np = self._K(queries_norm, self.extreme_rays_).astype(np.float32)  # (N, K)

        # K(q,q): self-similarity of each query in kernel space
        # For RBF on unit sphere K(q,q) = exp(0) = 1; for poly it may differ
        K_qq_np = self._K(queries_norm, queries_norm).diagonal().astype(np.float32)  # (N,)

        G_t = torch.tensor(self.G_, dtype=dtype, device=device)  # (K, K)
        lr  = 1.0 / (self._lip_ + 1e-8)                          # FIX #5: use cached L

        N, K    = K_qR_np.shape
        W_out   = np.zeros((N, K), dtype=np.float32)
        res_out = np.zeros(N,      dtype=np.float32)

        for start in range(0, N, batch_size):
            K_qR = torch.tensor(K_qR_np[start : start + batch_size],
                                dtype=dtype, device=device)      # (B, K)
            K_qq = torch.tensor(K_qq_np[start : start + batch_size],
                                dtype=dtype, device=device)      # (B,)
            B    = K_qR.shape[0]

            W = torch.zeros(B, K, dtype=dtype, device=device)
            Y = torch.zeros(B, K, dtype=dtype, device=device)
            t = 1.0

            for i in range(max_iter):
                grad   = Y @ G_t - K_qR                          # (B, K)
                W_next = (Y - lr * grad).clamp_(min=0.0)         # (B, K)  project ≥ 0

                # FIX #3: convergence check AFTER computing W_next but BEFORE
                # updating Y — check on the primal iterates, not the momentum point.
                # Also avoid double-assignment: don't touch W/Y inside the if block.
                if i % 50 == 0:
                    with torch.no_grad():
                        denom = W.norm() + 1e-8
                        delta = (W_next - W).norm() / denom
                        if delta.item() < tol:
                            W = W_next
                            break

                t_next   = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
                momentum = (t - 1.0) / t_next
                Y = W_next + momentum * (W_next - W)
                W = W_next
                t = t_next

            # Compute kernel-space squared residual: K(q,q) - 2 w·k_qR + w·G·w
            with torch.no_grad():
                Gw     = W @ G_t                                  # (B, K)
                res_sq = (
                    K_qq
                    - 2.0 * (W * K_qR).sum(1)
                    + (Gw * W).sum(1)
                ).clamp_(min=0.0)                                  # numerical guard

                # FIX #2: normalise residual by sqrt(K(q,q)) so score is in [0,1]
                # regardless of kernel parameterization (RBF gives K_qq=1, poly doesn't)
                res_norm = res_sq.sqrt() / (K_qq.sqrt() + 1e-8)

            W_out[start : start + batch_size]  = W.cpu().numpy()
            res_out[start : start + batch_size] = res_norm.cpu().numpy()

        return W_out, res_out

    # ── scoring ────────────────────────────────────────────────────────────────

    def score(self, queries: np.ndarray) -> np.ndarray:
        """
        Membership score in [0, 1].  1 = query lies inside the conic hull;
        lower values indicate increasing distance from the hull in kernel space.

        Routing
        -------
        "vmf"        : log-mean-exp over rays — bypasses NNLS entirely,
                    geometrically correct for spherical vMF mixture.
        "rbf"/"poly" : NNLS residual-based score — original behaviour.
        """
        if self.kernel == "vmf":
            return self._score_vmf(queries)
        if self.kernel == "spread":
            return self._score_spread(queries)
        else:
            return self._score_nnls(queries)
        
    def rotate_into_null_space(
        self,
        old_rays:           np.ndarray,
        variance_threshold: float = 0.95,
        min_norm:           float = 1e-8,
    ) -> "KernelConicHull":
        """
        Project this hull's extreme rays into the null space of old-class directions,
        then recompute the Gram matrix so that _kernel_nnls stays consistent.

        Algorithm
        ---------
        1. Thin SVD of the stacked old-class ray matrix:
               R_old = U S Vt      Vt ∈ ℝ^{r×D}
           Keep the top-k right singular vectors that capture
           `variance_threshold` of the squared-SV energy.
           These span the "occupied" subspace.

        2. Null-space projection of each extreme ray:
               r_proj = r − (r · V_old^T) V_old
           Rays that lie entirely in the old subspace collapse to near-zero
           and are discarded (norm < min_norm).

        3. Re-normalise surviving rays back to the unit sphere.

        4. Recompute G_ = K(R_new, R_new) + ridge·I and the cached Lipschitz
           constant so that _kernel_nnls runs on the updated geometry.

        Parameters
        ----------
        old_rays           : (M, D) L2-normalized extreme rays from all old classes.
        variance_threshold : fraction of squared-SV energy to include in the old
                             subspace.  0.95 removes the dominant 95% of old
                             directions; lower values apply a milder projection.
        min_norm           : rays whose post-projection norm falls below this
                             threshold are removed (they were nearly collinear
                             with the old subspace).

        Returns
        -------
        self  (mutated in-place; fluent interface for chaining)
        """
        if self.extreme_rays_ is None:
            return self

        D = old_rays.shape[1]
        if D != self.extreme_rays_.shape[1]:
            raise ValueError(
                f"Dimension mismatch: old_rays has D={D}, "
                f"hull has D={self.extreme_rays_.shape[1]}"
            )

        # ── 1. Truncated SVD basis of the old-class subspace ──────────────────
        _, S, Vt   = np.linalg.svd(old_rays, full_matrices=False)   # Vt: (r, D)
        cumvar     = np.cumsum(S ** 2) / (S ** 2).sum()
        k          = min(int(np.searchsorted(cumvar, variance_threshold)) + 1, len(S))
        V_old      = Vt[:k]                                          # (k, D) orthonormal rows
        

        # ── 2. Project out the old subspace from every extreme ray ────────────
        # r_proj = r  −  (r Vᵀ_old) V_old   (subtract old-subspace component)
        proj_coefs = self.extreme_rays_ @ V_old.T                    # (K, k)
        rays_proj  = self.extreme_rays_ - proj_coefs @ V_old         # (K, D)

        # ── 3. Discard collapsed rays; re-normalize survivors ─────────────────
        norms      = np.linalg.norm(rays_proj, axis=1)               # (K,)
        valid      = norms >= min_norm
        if valid.sum() == 0:
            return self                                               # nothing survives; keep original

        rays_proj  = rays_proj[valid]
        rays_new   = rays_proj / norms[valid, np.newaxis]            # (K', D) unit-norm

        # ── 4. Recompute Gram matrix and Lipschitz constant ───────────────────
        G_raw      = self._K(rays_new, rays_new)                     # (K', K')
        self.G_    = G_raw + self.ridge * np.eye(len(rays_new))
        self._lip_ = self._compute_lipschitz(self.G_)

        # ── 5. Commit updated rays ────────────────────────────────────────────
        if self.extreme_rays_index is not None:
            self.extreme_rays_index = self.extreme_rays_index[valid]
        self.extreme_rays_ = rays_new

        return self


    def score_all(self, queries: np.ndarray) -> dict:
        """
        Return multiple geometric/probabilistic scores per query.

        Routing
        -------
        "vmf"        : log-mean-exp mixture scoring + per-ray attention weights.
                    No NNLS — uses the bounded kernel K(x,y) = exp(κ(x·y−1)).
        "rbf"/"poly" : NNLS residual scoring with reconstruction weights.

        Common keys (all kernels)
        -------------------------
        cosine             : (N,) in [0, 1]   primary membership score
        kernel_residual    : (N,) ≥ 0         normalised distance in kernel space
        weights            : (N, K) ≥ 0       per-ray weights (NNLS or softmax attention)
        support_sparsity   : (N,) in [0, 1]   fraction of rays with non-trivial weight
        reconstruction_sq  : (N,) ≥ 0         unnormalized squared residual
        angular_margin     : (N,) in [0, 1]   alias of cosine for downstream compatibility

        vMF-only keys
        -------------
        log_mixture        : (N,) in (−∞, 0]  raw log-mean-exp score
        max_cosine         : (N,) in [−1, 1]  cosine to nearest ray (κ→∞ limit)
        effective_rays     : (N,) in [1, K]   exp(entropy of attention) — 1 = single ray
                                            dominates, K = uniform spread (OOD signal)
        """
        if self.kernel == "vmf":
            return self._score_all_vmf(queries)
        if self.kernel == "spread":
            return self._score_all_spread(queries)
        return self._score_all_nnls(queries)


    def _score_all_nnls(self, queries: np.ndarray) -> dict:
        """Original NNLS-based scoring for rbf/poly kernels."""
        weights, residuals = self._kernel_nnls(queries)

        membership = np.clip(1.0 - residuals, 0.0, 1.0)
        sparsity   = (weights > 1e-4).mean(axis=1)               # (N,)

        # FIX (carried over): use _K_diag to avoid materialising (N,N) matrix
        queries_norm = normalize(queries, axis=1)
        K_qq         = self._K_diag(queries_norm)                # (N,)
        recon_sq     = (residuals ** 2) * K_qq

        return {
            "cosine":            membership,
            "kernel_residual":   residuals,
            "weights":           weights,
            "support_sparsity":  sparsity,
            "reconstruction_sq": recon_sq,
            "angular_margin":    membership,
            "max_ray_sim": membership,
            "blended": membership
        }


    def _score_all_vmf(self, queries: np.ndarray) -> dict:
        """
        vMF mixture scoring without NNLS.

        Computes everything from a single (N, K) cosine-similarity matrix:
        • log-mean-exp → membership score
        • softmax weights = posterior responsibilities of each ray for each query
        • entropy of those weights → effective ray count → spreading / OOD signal
        """
        queries_norm = normalize(queries, axis=1)                 # (N, D)
        K            = self.extreme_rays_.shape[0]
        eps          = 1e-30

        # Cosine similarities (in [-1, 1])
        cos_sim = queries_norm @ self.extreme_rays_.T             # (N, K)

        # ── log-mean-exp on the bounded vMF kernel ──────────────────────────────
        # log K(q, r) = κ · (cos − 1)  ∈ (−∞, 0]
        # Using log-K analytically avoids the −log(eps) ≈ −69 floor that comes
        # from log(K_qR + eps), which would distort LME when many rays are far.
        log_K   = self.kappa * (cos_sim - 1.0)                    # (N, K), ≤ 0
        max_log = log_K.max(axis=1, keepdims=True)                # (N, 1)
        lme     = (max_log.squeeze(axis=1)
                + np.log(np.exp(log_K - max_log).mean(axis=1)))  # (N,) ≤ 0

        membership = np.exp(lme)                                  # (N,) in [0, 1]
        max_cos    = cos_sim.max(axis=1)                          # (N,) in [-1, 1]

        # ── soft attention weights = posterior over rays ────────────────────────
        # softmax(log_K) gives the responsibility of each ray for each query.
        # Equivalent to: which ray "explains" this query best, with smoothing.
        log_softmax = log_K - (max_log + np.log(
                        np.exp(log_K - max_log).sum(axis=1, keepdims=True)
                    ))                                          # (N, K)
        weights     = np.exp(log_softmax)                         # (N, K) sums to 1

        # ── effective ray count via entropy ─────────────────────────────────────
        # H = −Σ wᵢ log wᵢ;  effective_rays = exp(H) ∈ [1, K]
        # Low value → confident, single dominant ray.
        # High value → spread across many rays, possible OOD or boundary case.
        entropy        = -(weights * (log_softmax)).sum(axis=1)   # (N,) in [0, log K]
        effective_rays = np.exp(entropy)                          # (N,) in [1, K]
        sparsity       = 1.0 - (effective_rays - 1.0) / max(K - 1, 1)  # (N,) in [0, 1]

        # ── kernel residual analogue ────────────────────────────────────────────
        # For vMF mixture: residual = √(1 − membership²) lives in [0, 1].
        # This makes residual = 0 when membership = 1 (perfect match) and
        # residual → 1 as membership → 0, matching the NNLS convention.
        residuals = np.sqrt(np.clip(1.0 - membership ** 2, 0.0, 1.0))
        recon_sq  = 1.0 - membership ** 2                         # squared form

        return {
            # Common keys (matching NNLS interface)
            "cosine":            membership,
            "kernel_residual":   residuals,
            "weights":           weights,
            "support_sparsity":  sparsity,
            "reconstruction_sq": recon_sq,
            "angular_margin":    membership,
            # vMF-specific keys
            "log_mixture":       lme,
            "max_cosine":        max_cos,
            "effective_rays":    effective_rays,
            "blended": membership, 
            "max_ray_sim": membership
        }
    def summary(self) -> dict:
        if self.extreme_rays_ is None:
            return {"fitted": False}
        return {
            "fitted":         True,
            "kernel":         self.kernel,
            "gamma":          self.gamma,
            "n_extreme_rays": len(self.extreme_rays_),
            "lipschitz":      self._lip_,
            "ridge":          self.ridge,
        }
# ── per-class kernel hull builder ─────────────────────────────────────────────

def build_class_kernel_conic_hulls(
    feature_dict: Dict[str, np.ndarray],
    n_rays:       int   = 50,
    kernel:       str   = "rbf",
    gamma:        float = 5.0,
    min_samples:  int   = 3,
) -> Dict[str, KernelConicHull]:
    """Fit a KernelConicHull for every class in feature_dict."""
    hulls: Dict[str, KernelConicHull] = {}
    pbar = tqdm(feature_dict.items(), desc="Building kernel hulls",
                total=len(feature_dict), unit="class", dynamic_ncols=True)

    for class_name, vectors in pbar:
        pbar.set_postfix({"class": class_name})
        if len(vectors) < min_samples:
            continue
        hull = KernelConicHull(
            n_rays=min(n_rays, len(vectors)),
            kernel=kernel,
            gamma=gamma,
            kappa=gamma if kernel == "vmf" else 50.0,
        )
        hull.fit(vectors)
        hulls[class_name] = hull

    print(f"\nBuilt kernel hulls for {len(hulls)}/{len(feature_dict)} classes.")
    return hulls


# ── per-class builder ─────────────────────────────────────────────────────────

# ── Shifted-Origin Conic Hull ─────────────────────────────────────────────────

class TaskOriginRegistry:
    """
    Manages per-task anchor points on the unit hypersphere.

    Each task is assigned a unit vector o_t.  Features are scored by shifting
    q → normalize(q − o_t) before running NNLS, giving the hull fine-grained
    angular resolution near o_t and naturally compressing off-task queries.

    Strategies
    ----------
    "orthogonal" (default)
        Random unit vectors iteratively Gram-Schmidt orthogonalised against
        all previous origins.  Guarantees 90° separation between every pair
        of task anchors; supports up to D tasks in D dimensions.
    "learned"
        o_t = L2-normalized mean of task t's training features.
        Places the anchor exactly at the class centroid — maximises resolution
        right where the new classes cluster, at the cost of requiring features
        before registration.
    "axis"
        o_t = standard basis vector e_t.  Identical to orthogonal for the
        first D tasks but burns one dedicated coordinate per task.
    "random"
        Independent random unit vectors (no orthogonality guarantee).
    """

    def __init__(
        self,
        dim:      int,
        strategy: str = "orthogonal",
        seed:     int = 42,
    ):
        self.dim      = dim
        self.strategy = strategy
        self.rng      = np.random.default_rng(seed)
        self.origins: Dict[int, np.ndarray] = {}
        self._basis:  list = []             # accumulated for Gram-Schmidt

    def register_task(
        self,
        task_id:  int,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute and store the origin for task_id; return it."""
        if self.strategy == "axis":
            if task_id >= self.dim:
                raise ValueError(
                    f"axis strategy exhausted: task_id={task_id} >= dim={self.dim}"
                )
            o = np.zeros(self.dim, dtype=np.float32)
            o[task_id] = 1.0

        elif self.strategy == "learned":
            if features is None:
                raise ValueError("learned strategy requires features array")
            o = normalize(features, axis=1).mean(axis=0).astype(np.float32)
            o = o / (np.linalg.norm(o) + 1e-12)

        elif self.strategy == "random":
            o = self.rng.standard_normal(self.dim).astype(np.float32)
            o = o / np.linalg.norm(o)

        elif self.strategy == "orthogonal":
            o = self.rng.standard_normal(self.dim).astype(np.float64)
            for prev in self._basis:
                o -= (o @ prev) * prev
            norm = np.linalg.norm(o)
            if norm < 1e-8:
                raise RuntimeError(
                    f"Orthogonal origins exhausted at task {task_id} "
                    f"(dim={self.dim} — increase embedding dim or switch strategy)."
                )
            o = (o / norm).astype(np.float32)
            self._basis.append(o.astype(np.float64))

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        self.origins[task_id] = o
        return o

    def get_origin(self, task_id: int) -> np.ndarray:
        if task_id not in self.origins:
            raise KeyError(f"Task {task_id} not registered.")
        return self.origins[task_id]

    def separation(self, task_a: int, task_b: int) -> float:
        """Angular separation (radians) between two task origins."""
        oa = self.origins[task_a]
        ob = self.origins[task_b]
        return float(np.arccos(np.clip(float(oa @ ob), -1.0, 1.0)))


class ShiftedConicHull:
    """
    Conic hull fitted in a coordinate frame shifted to a task-specific origin.

    The shift  q' = normalize(q − o_t)  is the first-order approximation of
    the Riemannian log map at o_t on the unit sphere.  It gives the inner
    ConicHull fine-grained angular resolution around the task anchor while
    naturally compressing features far from o_t toward a single antipodal
    cluster, where they will score low.

    The public interface mirrors ConicHull exactly so it is a drop-in
    replacement inside HullManager.evaluate_all_scores.

    Parameters
    ----------
    origin      : (D,) anchor point for this hull's task (will be unit-normed).
    n_rays      : number of extreme rays for the inner ConicHull.
    renormalize : if True (default) L2-normalize after shifting.
    """

    def __init__(
        self,
        origin:      np.ndarray,
        n_rays:      int           = 50,
        use_pca:     bool          = True,
        pca_dim:     int           = 64,
        k_local:     Optional[int] = None,
        renormalize: bool          = True,
    ):
        o = np.asarray(origin, dtype=np.float32)
        self.origin_     = o / (np.linalg.norm(o) + 1e-12)
        self.renormalize = renormalize
        self._hull       = ConicHull(n_rays=n_rays, use_pca=use_pca,
                                     pca_dim=pca_dim, k_local=k_local)

    # ── internal shift ────────────────────────────────────────────────────────

    def _shift(self, X_norm: np.ndarray) -> np.ndarray:
        """(N, D) unit vectors → shifted (and optionally renormalized) vectors."""
        shifted = X_norm - self.origin_[np.newaxis, :]          # (N, D)
        if self.renormalize:
            norms   = np.linalg.norm(shifted, axis=1, keepdims=True)
            shifted = shifted / np.maximum(norms, 1e-8)
        return shifted

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "ShiftedConicHull":
        self._hull.fit(self._shift(normalize(X, axis=1)))
        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        return self._hull.reconstruct(self._shift(normalize(queries, axis=1)))

    def score(self, queries: np.ndarray) -> np.ndarray:
        return self._hull.score(self._shift(normalize(queries, axis=1)))

    def score_all(self, queries: np.ndarray) -> dict:
        return self._hull.score_all(self._shift(normalize(queries, axis=1)))

    # ── property pass-throughs (for NSR loss / null-space rotation) ───────────

    @property
    def extreme_rays_(self) -> Optional[np.ndarray]:
        return self._hull.extreme_rays_

    @extreme_rays_.setter
    def extreme_rays_(self, v: np.ndarray) -> None:
        self._hull.extreme_rays_ = v

    @property
    def extreme_rays_index(self) -> Optional[np.ndarray]:
        return self._hull.extreme_rays_index

    @property
    def pca_(self):
        return self._hull.pca_

    def summary(self) -> dict:
        s = self._hull.summary()
        s["origin_norm"] = float(np.linalg.norm(self.origin_))
        return s


def build_shifted_conic_hulls(
    feature_dict: Dict[str, np.ndarray],
    origin:       np.ndarray,
    n_rays:       int           = 50,
    use_pca:      bool          = True,
    pca_dim:      int           = 64,
    min_samples:  int           = 3,
    k_local:      Optional[int] = None,
) -> "Dict[str, ShiftedConicHull]":
    """Fit one ShiftedConicHull per class in feature_dict, all sharing the same origin."""
    hulls: Dict[str, ShiftedConicHull] = {}
    pbar = tqdm(feature_dict.items(), desc="Building shifted hulls",
                total=len(feature_dict), unit="class", dynamic_ncols=True)

    for class_name, vectors in pbar:
        pbar.set_postfix({"class": class_name})
        if len(vectors) < min_samples:
            continue
        hull = ShiftedConicHull(
            origin=origin,
            n_rays=min(n_rays, len(vectors)),
            use_pca=use_pca,
            pca_dim=pca_dim,
            k_local=k_local,
        )
        hull.fit(vectors)
        hulls[class_name] = hull

    print(f"\nBuilt shifted hulls for {len(hulls)}/{len(feature_dict)} classes.")
    return hulls


# ── Pre-allocated spherical regions for budgeted hull construction ────────────

@dataclass
class ClassRegion:
    """
    A pre-allocated angular cap on the unit sphere for one class.

    All extreme rays for this class must lie inside the cap centred at
    ``anchor`` with cos(x, anchor) >= ``cos_cap``.  The class also inherits
    a stage-level cap around ``stage_origin``.
    """
    class_id:        int
    stage_id:        int
    local_class_idx: int
    anchor:          np.ndarray    # (D,) unit vector
    max_rays:        int
    cos_cap:         float         # cos(class_cap) — inner region
    stage_origin:    np.ndarray    # (D,) unit vector
    stage_cos_cap:   float         # cos(stage_cap) — outer region


def _gram_schmidt_origins(dim: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``count`` mutually orthogonal unit vectors in R^dim."""
    origins = []
    basis   = []
    for _ in range(count):
        v = rng.standard_normal(dim).astype(np.float64)
        for b in basis:
            v -= (v @ b) * b
        n = np.linalg.norm(v)
        if n < 1e-8:
            raise RuntimeError(
                f"Cannot place {count} orthogonal origins in dim={dim}; "
                "increase embedding dim or reduce num_stages."
            )
        v = v / n
        origins.append(v.astype(np.float32))
        basis.append(v)
    return np.stack(origins, axis=0)


def _tangent_basis(origin: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the tangent space at ``origin`` on the sphere. (D-1, D)"""
    o = origin.astype(np.float64)
    o = o / (np.linalg.norm(o) + 1e-12)
    D = len(o)
    tangents = []
    for i in range(D):
        t = np.eye(D)[i] - (np.eye(D)[i] @ o) * o
        if np.linalg.norm(t) > 1e-6:
            tangents.append(t)
    if not tangents:
        t = np.array([1.0] + [0.0] * (D - 1))
        t = t - (t @ o) * o
        tangents = [t]
    B = np.stack(tangents, axis=0)
    Q, _ = np.linalg.qr(B.T, mode="reduced")
    return Q.T.astype(np.float32)


def _spread_class_anchors(
    stage_origin: np.ndarray,
    num_classes:  int,
    spread_rad:   float,
) -> np.ndarray:
    """
    Place ``num_classes`` unit-norm anchors on a small circle around ``stage_origin``.

    The circle lies in the tangent plane at ``stage_origin`` with angular
    radius ``spread_rad`` from the stage pole.
    """
    o  = stage_origin.astype(np.float64)
    o  = o / (np.linalg.norm(o) + 1e-12)
    Tb = _tangent_basis(o.astype(np.float32)).astype(np.float64)
    e1 = Tb[0]
    e2 = Tb[1] if Tb.shape[0] > 1 else np.cross(o, e1)

    cos_s = np.cos(spread_rad)
    sin_s = np.sin(spread_rad)
    anchors = []
    for i in range(num_classes):
        phi = 2.0 * np.pi * i / num_classes
        direction = np.cos(phi) * e1 + np.sin(phi) * e2
        direction /= np.linalg.norm(direction) + 1e-12
        a = cos_s * o + sin_s * direction
        anchors.append((a / np.linalg.norm(a)).astype(np.float32))
    return np.stack(anchors, axis=0)


def project_to_spherical_cap(
    vectors:  np.ndarray,
    anchor:   np.ndarray,
    cos_cap:  float,
) -> np.ndarray:
    """Project rows of ``vectors`` onto the spherical cap cos(x, anchor) >= cos_cap."""
    X = normalize(vectors.astype(np.float64), axis=1)
    a = anchor.astype(np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    cos_cap = float(np.clip(cos_cap, -1.0, 1.0))
    sin_cap = np.sqrt(max(0.0, 1.0 - cos_cap ** 2))

    cos_xa = X @ a
    out    = X.copy()
    outside = cos_xa < cos_cap
    if outside.any():
        t = X[outside] - cos_xa[outside, np.newaxis] * a
        t_norm = np.linalg.norm(t, axis=1, keepdims=True)
        t = np.where(t_norm > 1e-8, t / np.maximum(t_norm, 1e-8), _tangent_basis(a.astype(np.float32))[0])
        out[outside] = cos_cap * a + sin_cap * t
        out[outside] = normalize(out[outside], axis=1)
    return out.astype(np.float32)


def project_to_class_region(vectors: np.ndarray, region: ClassRegion) -> np.ndarray:
    """Apply stage cap then class cap."""
    X = project_to_spherical_cap(vectors, region.stage_origin, region.stage_cos_cap)
    X = project_to_spherical_cap(X, region.anchor, region.cos_cap)
    return X


class SphericalRegionRegistry:
    """
    Pre-allocate angular regions and ray budgets across stages and classes.

    Example (CIFAR-100, 10 stages × 10 classes, budget 400)
    ---------------------------------------------------------
    • 400 / 100 = 4 rays per class globally
    • Each stage receives one orthogonal pole on the unit sphere
    • Within a stage, 10 class anchors are spread on a small circle around
      the stage pole; each class may use at most ``rays_per_class`` rays
    """

    def __init__(
        self,
        dim:               int,
        total_classes:     int,
        num_stages:        int,
        classes_per_stage: int,
        total_ray_budget:  int   = 400,
        stage_cap_deg:     float = 35.0,
        class_cap_deg:     float = 12.0,
        class_spread_deg:  float = 8.0,
        seed:              int   = 42,
    ):
        if num_stages * classes_per_stage > total_classes:
            raise ValueError(
                f"num_stages ({num_stages}) × classes_per_stage ({classes_per_stage}) "
                f"exceeds total_classes ({total_classes})."
            )
        self.dim               = dim
        self.total_classes     = total_classes
        self.num_stages        = num_stages
        self.classes_per_stage = classes_per_stage
        self.total_ray_budget  = total_ray_budget
        self.stage_cap_deg     = stage_cap_deg
        self.class_cap_deg     = class_cap_deg
        self.class_spread_deg  = class_spread_deg
        self.rng               = np.random.default_rng(seed)

        self.rays_per_class = max(1, total_ray_budget // total_classes)
        self.stage_ray_budget = self.rays_per_class * classes_per_stage
        self.regions: Dict[str, ClassRegion] = {}
        self._stage_origins: Dict[int, np.ndarray] = {}
        self._preallocate()

    def _preallocate(self) -> None:
        stage_origins = _gram_schmidt_origins(self.dim, self.num_stages, self.rng)
        stage_cos     = np.cos(np.deg2rad(self.stage_cap_deg))
        class_cos     = np.cos(np.deg2rad(self.class_cap_deg))
        spread_rad    = np.deg2rad(self.class_spread_deg)

        for stage_id in range(self.num_stages):
            origin  = stage_origins[stage_id]
            anchors = _spread_class_anchors(
                origin, self.classes_per_stage, spread_rad
            )
            self._stage_origins[stage_id] = origin
            for local_idx in range(self.classes_per_stage):
                class_id = stage_id * self.classes_per_stage + local_idx
                self.regions[str(class_id)] = ClassRegion(
                    class_id=class_id,
                    stage_id=stage_id,
                    local_class_idx=local_idx,
                    anchor=anchors[local_idx],
                    max_rays=self.rays_per_class,
                    cos_cap=class_cos,
                    stage_origin=origin,
                    stage_cos_cap=stage_cos,
                )

    def get_region(self, class_id: Union[int, str]) -> ClassRegion:
        key = str(int(class_id))
        if key not in self.regions:
            raise KeyError(f"Class {class_id} has no pre-allocated region.")
        return self.regions[key]

    def rays_for_class(self, class_id: Union[int, str]) -> int:
        return self.get_region(class_id).max_rays

    def stage_for_class(self, class_id: Union[int, str]) -> int:
        return self.get_region(class_id).stage_id

    def summary(self) -> str:
        lines = [
            f"SphericalRegionRegistry: {self.total_ray_budget} rays / "
            f"{self.total_classes} classes → {self.rays_per_class} rays/class",
            f"  stages={self.num_stages}, stage budget={self.stage_ray_budget}, "
            f"stage cap={self.stage_cap_deg:.1f}°, class cap={self.class_cap_deg:.1f}°",
        ]
        return "\n".join(lines)


class RegionConstrainedConicHull:
    """
    Conic hull whose extreme rays are confined to a pre-allocated class region.

    Fitting projects training samples into the class cap before SPA, caps the
    number of rays at ``region.max_rays``, and re-projects the final rays.
    """

    def __init__(
        self,
        region:   ClassRegion,
        use_pca:  bool          = False,
        pca_dim:  int           = 64,
        k_local:  Optional[int] = None,
    ):
        self.region = region
        self._hull  = ConicHull(
            n_rays=region.max_rays,
            use_pca=use_pca,
            pca_dim=pca_dim,
            k_local=k_local,
        )

    def fit(self, X: np.ndarray) -> "RegionConstrainedConicHull":
        if len(X) < 3:
            return self
        X_region = project_to_class_region(X, self.region)
        n_rays   = min(self.region.max_rays, len(X_region))
        self._hull.n_rays = n_rays
        self._hull.fit(X_region)
        if self._hull.extreme_rays_ is not None:
            self._hull.extreme_rays_ = project_to_class_region(
                self._hull.extreme_rays_, self.region
            )
        return self

    def reconstruct(self, queries: np.ndarray) -> np.ndarray:
        Q = project_to_class_region(normalize(queries, axis=1), self.region)
        return self._hull.reconstruct(Q)

    def score(self, queries: np.ndarray) -> np.ndarray:
        Q = normalize(queries, axis=1)
        in_stage = (Q @ self.region.stage_origin) >= self.region.stage_cos_cap
        in_class = (Q @ self.region.anchor) >= self.region.cos_cap
        valid    = in_stage & in_class
        scores   = np.full(len(Q), -np.inf, dtype=np.float32)
        if valid.any():
            scores[valid] = self._hull.score(Q[valid])
        return scores

    def score_all(self, queries: np.ndarray) -> dict:
        Q = normalize(queries, axis=1)
        in_stage = (Q @ self.region.stage_origin) >= self.region.stage_cos_cap
        in_class = (Q @ self.region.anchor) >= self.region.cos_cap
        valid    = in_stage & in_class
        out      = {k: np.full(len(Q), -np.inf if k == "cosine" else 0.0, dtype=np.float32)
                    for k in ("cosine", "angular_margin", "blended", "max_ray_sim")}
        if valid.any():
            sub = self._hull.score_all(Q[valid])
            for k in out:
                if k in sub:
                    out[k][valid] = sub[k]
        out["in_region"] = valid.astype(np.float32)
        return out

    @property
    def extreme_rays_(self) -> Optional[np.ndarray]:
        return self._hull.extreme_rays_

    @extreme_rays_.setter
    def extreme_rays_(self, v: np.ndarray) -> None:
        self._hull.extreme_rays_ = v

    @property
    def extreme_rays_index(self) -> Optional[np.ndarray]:
        return self._hull.extreme_rays_index

    @property
    def pca_(self):
        return self._hull.pca_


def build_region_conic_hulls(
    feature_dict:       Dict[str, np.ndarray],
    region_registry:    SphericalRegionRegistry,
    use_pca:            bool = False,
    pca_dim:            int  = 64,
    min_samples:        int  = 3,
    k_local:            Optional[int] = None,
) -> Dict[str, RegionConstrainedConicHull]:
    """Fit one region-constrained hull per class using pre-allocated caps/budgets."""
    hulls: Dict[str, RegionConstrainedConicHull] = {}
    pbar = tqdm(feature_dict.items(), desc="Building region hulls",
                total=len(feature_dict), unit="class", dynamic_ncols=True)

    for class_name, vectors in pbar:
        pbar.set_postfix({"class": class_name, "rays": region_registry.rays_for_class(class_name)})
        if len(vectors) < min_samples:
            continue
        region = region_registry.get_region(class_name)
        hull   = RegionConstrainedConicHull(
            region=region, use_pca=use_pca, pca_dim=pca_dim, k_local=k_local
        )
        hull.fit(vectors)
        hulls[class_name] = hull

    print(f"\nBuilt region hulls for {len(hulls)}/{len(feature_dict)} classes "
          f"({region_registry.rays_per_class} rays/class cap).")
    return hulls


# ── per-class builder ─────────────────────────────────────────────────────────

def build_class_conic_hulls(
    feature_dict:   Dict[str, np.ndarray],
    n_rays:         int            = 50,
    use_pca:        bool           = True,
    pca_dim:        int            = 64,
    min_samples:    int            = 3,
    k_local:        Optional[int]  = None,
    ray_diversity:  str            = "hybrid",
    spa_oversample: int            = 3,
) -> Dict[str, ConicHull]:
    """
    Fit a ConicHull for every class in `feature_dict`.

    Parameters
    ----------
    feature_dict   : { class_name: np.ndarray (N, D) }
    n_rays         : max extreme rays per class
    use_pca        : apply PCA before SPA
    pca_dim        : PCA target dim (only used when use_pca=True)
    min_samples    : skip classes with fewer samples
    k_local        : if set, scoring uses only the k_local nearest rays per query
    ray_diversity  : "spa" | "fps" | "hybrid" — ray selection strategy
    spa_oversample : oversample factor for hybrid strategy

    Returns
    -------
    { class_name: ConicHull }
    """
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
        hull = ConicHull(
            n_rays=current_rays,
            use_pca=use_pca,
            pca_dim=pca_dim,
            k_local=k_local,
            ray_diversity=ray_diversity,
            spa_oversample=spa_oversample,
        )
        hull.fit(vectors)
        class_hulls[class_name] = hull

    print(f"\nBuilt hulls for {len(class_hulls)}/{len(feature_dict)} classes.")
    return class_hulls

def null_space_repulsion_loss(
    features:            "torch.Tensor",
    old_hulls:           Dict[str, any],
    new_hull:            Optional[any] = None,
    old_centroids:       Optional["torch.Tensor"] = None,
    normalize_feats:     bool          = True,
    weight_repel:        float         = 1.0,
    weight_anchor:       float         = 1.0,
    energy_margin:       float         = 0.0,
    variance_threshold:  float         = 0.95,
    max_subspace_rank:   Optional[int] = None,
    min_ray_norm:        float         = 1e-8,
    repel_only:          bool          = False,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    from typing import Dict, Optional
    """
    Null-Space Repulsion + Conic Anchor Loss.

    Repulsion prefers a low-rank old-class centroid basis (C_old directions) so
    the target stays achievable when many extreme rays would otherwise span
    nearly all of feature space.  Falls back to a capped SVD basis over old
    rays with a random-projection baseline subtracted so the hinge does not
    saturate at 1.0.  Anchor rays are projected into the repulsion null space
    before alignment so the two terms do not fight each other.
    """
    if not old_hulls and new_hull is None and old_centroids is None:
        return (features * 0.0).sum()

    device = features.device
    dtype  = features.dtype
    feat_dim = features.shape[1]

    f = F.normalize(features, p=2, dim=1) if normalize_feats else features  # (N, D)

    # ── 1. Build a low-rank old-class basis ───────────────────────────────────
    V_old = None
    repel_baseline = 0.0

    if old_centroids is not None and old_centroids.numel() > 0:
        # Centroid repulsion: same scale as orthogonal_centroid_loss (optimisable).
        V_old = F.normalize(old_centroids.to(device=device, dtype=dtype), p=2, dim=1).detach()
        repel_baseline = 0.0
    else:
        all_old_rays = []
        for hull in old_hulls.values():
            rays = hull.extreme_rays_
            if rays is not None and len(rays) > 0:
                all_old_rays.append(torch.tensor(rays, dtype=dtype, device=device))

        if all_old_rays:
            R_old = F.normalize(torch.cat(all_old_rays, dim=0), p=2, dim=1).detach()
            _, S, Vh = torch.linalg.svd(R_old, full_matrices=False)
            cumvar   = torch.cumsum(S.pow(2), dim=0) / S.pow(2).sum()
            rank     = min(
                int(torch.searchsorted(
                    cumvar,
                    torch.tensor(variance_threshold, device=cumvar.device),
                ).item()) + 1,
                len(S),
            )
            if max_subspace_rank is not None:
                rank = min(rank, max_subspace_rank)
            rank = min(rank, feat_dim - 1) if feat_dim > 1 else 0
            if rank > 0:
                V_old = Vh[:rank].detach()
                # Expected energy for a random unit vector onto an rank-r subspace.
                repel_baseline = rank / feat_dim

    # ── 2. Repulsion ──────────────────────────────────────────────────────────
    if V_old is not None and V_old.shape[0] > 0:
        if old_centroids is not None:
            # Per-class squared cosines — bounded and easy to drive toward 0.
            l_repel = (f @ V_old.T).pow(2).sum(dim=1).mean()
        else:
            proj_energy = (f @ V_old.T).pow(2).sum(dim=1)
            l_repel = (proj_energy - repel_baseline - energy_margin).clamp(min=0.0).mean()
    else:
        l_repel = (features * 0.0).sum()

    if repel_only:
        return weight_repel * l_repel

    # ── 3. Anchor: align with null-space hull rays ─────────────────────────────
    if new_hull is not None and new_hull.extreme_rays_ is not None \
            and len(new_hull.extreme_rays_) > 0:

        R_new = F.normalize(
            torch.tensor(new_hull.extreme_rays_, dtype=dtype, device=device),
            p=2, dim=1,
        ).detach()

        if V_old is not None and V_old.shape[0] > 0:
            proj_coefs = R_new @ V_old.T
            R_new      = R_new - proj_coefs @ V_old
            norms = R_new.norm(dim=1)
            valid = norms >= min_ray_norm
            if valid.any():
                R_new = F.normalize(R_new[valid], p=2, dim=1)
            else:
                R_new = None

        if R_new is not None and len(R_new) > 0:
            cos_sq         = (f @ R_new.T).pow(2)
            best_alignment = cos_sq.max(dim=1).values
            l_anchor       = (1.0 - best_alignment).mean()
        else:
            l_anchor = (features * 0.0).sum()
    else:
        l_anchor = (features * 0.0).sum()

    return weight_repel * l_repel + weight_anchor * l_anchor


def covariance_collapse_penalty(
    centered:    "torch.Tensor",
    eps:         float = 1e-4,
    temperature: float = 2.0,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    """
    Non-negative penalty that is high when a batch has collapsed and ≈0 when spread.

    Uses softplus on the negative spread margin so the value is always ≥ 0 and
    minimising it still encourages larger covariance volume.
    """
    N, D = centered.shape
    if N < 2:
        return centered.new_zeros(())

    if N <= D:
        G = centered @ centered.T / max(N - 1, 1)
        G = G + eps * torch.eye(N, device=centered.device, dtype=centered.dtype)
        L = torch.linalg.cholesky(G)
        log_det = 2.0 * torch.log(L.diagonal()).sum()
        dim = N
    else:
        Sigma = (centered.T @ centered) / max(N - 1, 1)
        Sigma = Sigma + eps * torch.eye(D, device=centered.device, dtype=centered.dtype)
        L = torch.linalg.cholesky(Sigma)
        log_det = 2.0 * torch.log(L.diagonal()).sum()
        dim = D

    baseline     = dim * torch.log(
        torch.tensor(eps, device=centered.device, dtype=centered.dtype)
    )
    spread_margin = (log_det - baseline) / dim
    return F.softplus(-spread_margin / temperature)


def normalized_covariance_logdet(
    centered: "torch.Tensor",
    eps:      float = 1e-4,
) -> "torch.Tensor":
    """Deprecated alias — use :func:`covariance_collapse_penalty` instead."""
    return covariance_collapse_penalty(centered, eps=eps)


def null_space_spread_loss(
    features:        "torch.Tensor",
    old_centroids:   Optional["torch.Tensor"] = None,
    labels:          Optional["torch.Tensor"] = None,
    weight_volume:   float = 1.0,
    weight_inter:    float = 0.5,
    eps:             float = 1e-4,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    from typing import Optional
    """
    Expand features into the available null space instead of collapsing on rays.

    Two complementary terms:
      • Volume — log-det on residuals after projecting out old-class centroids.
                 Maximises the ellipsoid volume new features occupy in free directions.
      • Inter-class — penalises squared cosine between per-class batch centroids,
                      pushing new-class cones apart from each other.
    """
    if features.shape[0] < 2:
        return (features * 0.0).sum()

    device, dtype = features.device, features.dtype
    loss = features.new_zeros(())

    # ── 1. Null-space volume (use raw features; magnitude carries spread info) ─
    residual = features
    if old_centroids is not None and old_centroids.numel() > 0:
        C        = F.normalize(old_centroids.to(device=device, dtype=dtype), p=2, dim=1)
        residual = features - (features @ C.T) @ C

    if weight_volume > 0.0 and residual.pow(2).sum() > eps:
        Z = residual - residual.mean(dim=0, keepdim=True)
        loss = loss + weight_volume * covariance_collapse_penalty(Z, eps)

    # ── 2. Inter-class centroid repulsion (within the current batch) ───────────
    if weight_inter > 0.0 and labels is not None:
        cls_ids = labels.unique()
        if len(cls_ids) >= 2:
            cents = []
            for cid in cls_ids:
                mask = labels == cid
                if mask.sum() >= 1:
                    c = F.normalize(features[mask].mean(dim=0, keepdim=True), p=2, dim=1)
                    cents.append(c)
            if len(cents) >= 2:
                C_mat = torch.cat(cents, dim=0)
                sims  = C_mat @ C_mat.T
                off   = ~torch.eye(len(cents), dtype=torch.bool, device=device)
                loss  = loss + weight_inter * sims[off].pow(2).mean()

    return loss


def build_orthogonal_stage_poles(
    num_stages: int,
    dim:        int,
    seed:       int = 0,
) -> np.ndarray:
    """Gram–Schmidt stage poles on the unit sphere, shape (num_stages, dim)."""
    rng = np.random.default_rng(seed)
    return _gram_schmidt_origins(dim, num_stages, rng)


def stage_confinement_loss(
    features:             "torch.Tensor",
    current_stage_idx:    int,
    stage_poles:          "torch.Tensor",
    cos_stage_cap:        float,
    cos_inter_max:        float = 0.35,
    weight_in:            float = 1.0,
    weight_out:           float = 1.0,
    repel_stage_indices:  Optional[list] = None,
) -> "torch.Tensor":
    """
    Keep features inside the current stage's spherical cap.

    L_in  — hinge when cos(z, pole_cur) < cos_stage_cap (too far from own pole).
    L_out — penalise leakage toward ``repel_stage_indices`` (prior stages only).

    At stage 0 use ``repel_stage_indices=[]`` so only L_in is active: compact the
    first task into its cap without repelling toward unoccupied future poles.
    """
    import torch
    import torch.nn.functional as F

    if features.shape[0] == 0 or stage_poles.shape[0] == 0:
        return features.new_zeros(())

    z = F.normalize(features, p=2, dim=1)
    poles = F.normalize(stage_poles.to(device=z.device, dtype=z.dtype), p=2, dim=1)
    cos_mat = z @ poles.T                                          # (N, S)

    cur = int(current_stage_idx)
    if cur < 0 or cur >= poles.shape[0]:
        return features.new_zeros(())

    cos_own = cos_mat[:, cur]
    cap     = features.new_tensor(float(cos_stage_cap))
    l_in    = F.relu(cap - cos_own).pow(2).mean() if weight_in > 0 else features.new_zeros(())

    l_out = features.new_zeros(())
    if weight_out > 0:
        if repel_stage_indices is None:
            repel_stage_indices = [i for i in range(poles.shape[0]) if i != cur]
        if repel_stage_indices:
            inter = features.new_tensor(float(cos_inter_max))
            idx   = torch.tensor(repel_stage_indices, device=z.device, dtype=torch.long)
            leak  = F.relu(cos_mat[:, idx] - inter).pow(2).mean()
            l_out = leak

    return weight_in * l_in + weight_out * l_out


def stage_confinement_loss_labeled(
    features:          "torch.Tensor",
    stage_indices:     "torch.Tensor",
    stage_poles:       "torch.Tensor",
    cos_stage_cap:     float,
    weight_in:         float = 1.0,
) -> "torch.Tensor":
    """
    L_in only: keep each sample inside its *own* stage cap (from labels).

    Used for replay batches where samples belong to different prior stages.
    """
    import torch
    import torch.nn.functional as F

    if features.shape[0] == 0 or stage_poles.shape[0] == 0:
        return features.new_zeros(())

    valid = stage_indices >= 0
    if not valid.any():
        return features.new_zeros(())

    z     = F.normalize(features[valid], p=2, dim=1)
    poles = F.normalize(
        stage_poles.to(device=z.device, dtype=z.dtype), p=2, dim=1,
    )
    s_idx = stage_indices[valid].long()
    cos_own = (z * poles[s_idx]).sum(dim=1)
    cap     = features.new_tensor(float(cos_stage_cap))
    l_in    = F.relu(cap - cos_own).pow(2).mean()
    return float(weight_in) * l_in


# ── Per-stage plane projection for staged hull scoring ────────────────────────

def _build_planes_from_stage_points(
    stage_points:         Dict[int, np.ndarray],
    variance_threshold:   float = 0.95,
    orthogonal_poles:     bool  = True,
) -> Dict[int, np.ndarray]:
    """
    Build orthonormal row-bases B_s from per-stage point clouds (rays or features).

    Each plane = orthogonalised stage pole + tangent SVD of points ⊥ pole.
    """
    stage_ids = sorted(stage_points.keys())
    if not stage_ids:
        return {}

    raw_poles = []
    for s in stage_ids:
        P    = normalize(np.asarray(stage_points[s], dtype=np.float64), axis=1)
        pole = normalize(P.mean(axis=0, keepdims=True), axis=1)[0]
        raw_poles.append(pole)

    poles: Dict[int, np.ndarray] = {}
    if orthogonal_poles and len(raw_poles) > 1:
        basis: list = []
        for s, p in zip(stage_ids, raw_poles):
            v = p.copy()
            for b in basis:
                v -= (v @ b) * b
            n = np.linalg.norm(v)
            if n < 1e-8:
                poles[s] = p.astype(np.float32)
            else:
                v = (v / n).astype(np.float32)
                poles[s] = v
                basis.append(v.astype(np.float64))
    else:
        for s, p in zip(stage_ids, raw_poles):
            poles[s] = p.astype(np.float32)

    plane_bases: Dict[int, np.ndarray] = {}
    for stage_idx in stage_ids:
        P = normalize(np.asarray(stage_points[stage_idx], dtype=np.float64), axis=1)
        pole = poles[stage_idx].astype(np.float64)

        tangent = P - (P @ pole)[:, np.newaxis] * pole
        norms   = np.linalg.norm(tangent, axis=1)
        tangent = tangent[norms > 1e-6]

        rows = [pole]
        if len(tangent) >= 2:
            _, S, Vt = np.linalg.svd(tangent, full_matrices=False)
            cumvar   = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12)
            k_tan    = min(
                int(np.searchsorted(cumvar, variance_threshold)) + 1,
                len(S),
                P.shape[1] - 1,
            )
            for i in range(k_tan):
                rows.append(Vt[i])

        B = np.stack(rows, axis=0)
        Q, _ = np.linalg.qr(B.T, mode="reduced")
        plane_bases[stage_idx] = Q.T.astype(np.float32)

    return plane_bases


def build_stage_plane_bases(
    hulls:                Dict[str, any],
    stage_class_map:      Dict[int, int],
    variance_threshold:   float = 0.95,
    orthogonal_poles:     bool  = True,
) -> Dict[int, np.ndarray]:
    """
    Build scoring planes from frozen hull extreme rays (build-time geometry).
    """
    from collections import defaultdict

    stage_rays: Dict[int, list] = defaultdict(list)
    for cls_name, hull in hulls.items():
        if hull.extreme_rays_ is None or len(hull.extreme_rays_) == 0:
            continue
        stage_idx = stage_class_map.get(int(cls_name), -1)
        if stage_idx < 0:
            continue
        stage_rays[stage_idx].append(hull.extreme_rays_)

    stage_points = {
        s: np.vstack(rays) for s, rays in stage_rays.items() if rays
    }
    return _build_planes_from_stage_points(
        stage_points, variance_threshold, orthogonal_poles,
    )


def build_stage_plane_bases_from_features(
    feature_dict:         Dict[str, np.ndarray],
    stage_class_map:      Dict[int, int],
    variance_threshold:   float = 0.95,
    orthogonal_poles:     bool  = True,
    max_per_class:        Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Build scoring planes from *current* backbone features (drift-aligned).

    Groups per-class feature matrices by stage, optionally subsamples each
    class to ``max_per_class`` rows, then calls ``_build_planes_from_stage_points``.
    """
    from collections import defaultdict

    stage_feats: Dict[int, list] = defaultdict(list)
    for cls_name, feats in feature_dict.items():
        if feats is None or len(feats) == 0:
            continue
        stage_idx = stage_class_map.get(int(cls_name), -1)
        if stage_idx < 0:
            continue
        F = np.asarray(feats, dtype=np.float64)
        if max_per_class is not None and len(F) > max_per_class:
            idx = np.linspace(0, len(F) - 1, max_per_class, dtype=int)
            F = F[idx]
        stage_feats[stage_idx].append(F)

    stage_points = {
        s: np.vstack(parts) for s, parts in stage_feats.items() if parts
    }
    return _build_planes_from_stage_points(
        stage_points, variance_threshold, orthogonal_poles,
    )


def build_stage_plane_bases_blended(
    hulls:                Dict[str, any],
    feature_dict:         Dict[str, np.ndarray],
    stage_class_map:      Dict[int, int],
    variance_threshold:   float = 0.95,
    orthogonal_poles:     bool  = True,
    max_per_class:        Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Planes from hull rays + current features stacked per stage."""
    from collections import defaultdict

    stage_parts: Dict[int, list] = defaultdict(list)
    for cls_name, hull in hulls.items():
        if hull.extreme_rays_ is None or len(hull.extreme_rays_) == 0:
            continue
        stage_idx = stage_class_map.get(int(cls_name), -1)
        if stage_idx < 0:
            continue
        stage_parts[stage_idx].append(hull.extreme_rays_)

    for cls_name, feats in feature_dict.items():
        if feats is None or len(feats) == 0:
            continue
        stage_idx = stage_class_map.get(int(cls_name), -1)
        if stage_idx < 0:
            continue
        F = np.asarray(feats, dtype=np.float64)
        if max_per_class is not None and len(F) > max_per_class:
            idx = np.linspace(0, len(F) - 1, max_per_class, dtype=int)
            F = F[idx]
        stage_parts[stage_idx].append(F)

    stage_points = {
        s: np.vstack(parts) for s, parts in stage_parts.items() if parts
    }
    return _build_planes_from_stage_points(
        stage_points, variance_threshold, orthogonal_poles,
    )


def project_to_stage_plane(
    X:           np.ndarray,
    B:           np.ndarray,
    renormalize: bool = True,
) -> np.ndarray:
    """Project rows of X onto the plane spanned by orthonormal rows of B."""
    X     = np.asarray(X, dtype=np.float64)
    coeff = X @ B.T
    proj  = coeff @ B
    if renormalize:
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        proj  = proj / np.maximum(norms, 1e-8)
    return proj.astype(np.float32)


def _score_projected_hull_batch(
    queries: np.ndarray,
    rays:    np.ndarray,
) -> Dict[str, np.ndarray]:
    """Standard cosine-style scores in an already-projected subspace."""
    Q = normalize(queries, axis=1)
    R = normalize(rays, axis=1)
    cos_mat = Q @ R.T
    cosine  = cos_mat.max(axis=1)
    mean_c  = cos_mat.mean(axis=1)
    return {
        "cosine":         cosine,
        "angular_margin": cosine,
        "blended":        0.5 * cosine + 0.5 * mean_c,
        "max_ray_sim":    cosine,
    }


def stack_hull_extreme_rays(hulls: Dict[str, any]) -> Optional[np.ndarray]:
    """Stack L2-normalised extreme rays from a hull dict; None if empty."""
    ray_list = [
        h.extreme_rays_
        for h in hulls.values()
        if h.extreme_rays_ is not None and len(h.extreme_rays_) > 0
    ]
    if not ray_list:
        return None
    return normalize(np.vstack(ray_list), axis=1)


def project_features_into_null_space(
    features:             np.ndarray,
    old_rays:             np.ndarray,
    variance_threshold:   float = 0.95,
    max_subspace_rank:    Optional[int] = None,
    min_norm:             float = 1e-8,
    renormalize:          bool  = True,
) -> np.ndarray:
    """
    Project rows of ``features`` into the null space of the old-class subspace
    spanned by ``old_rays``, then optionally re-normalise back to the unit sphere.
    """
    if old_rays is None or len(old_rays) == 0:
        return features

    R_old = normalize(old_rays.astype(np.float64), axis=1)
    _, S, Vt = np.linalg.svd(R_old, full_matrices=False)
    cumvar   = np.cumsum(S ** 2) / (S ** 2).sum()
    rank     = min(int(np.searchsorted(cumvar, variance_threshold)) + 1, len(S))
    if max_subspace_rank is not None:
        rank = min(rank, max_subspace_rank)
    if rank <= 0:
        return features

    V_old    = Vt[:rank]
    X        = features.astype(np.float64)
    proj     = (X @ V_old.T) @ V_old
    residual = X - proj

    if not renormalize:
        return residual.astype(np.float32)

    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.maximum(norms, min_norm)
    return (residual / norms).astype(np.float32)


def project_new_class_features(
    feature_dict:         Dict[str, np.ndarray],
    new_class_keys:       set,
    old_hulls:            Dict[str, any],
    variance_threshold:   float = 0.95,
    max_subspace_rank:    Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Return a copy of ``feature_dict`` with new-class vectors null-space projected."""
    old_rays = stack_hull_extreme_rays(old_hulls)
    if old_rays is None or not new_class_keys:
        return feature_dict

    out = dict(feature_dict)
    for cls_str in new_class_keys:
        if cls_str in out:
            out[cls_str] = project_features_into_null_space(
                out[cls_str],
                old_rays,
                variance_threshold=variance_threshold,
                max_subspace_rank=max_subspace_rank,
            )
    return out


def rotate_kernel_hulls_into_null_space(
    new_hulls:          Dict[str, KernelConicHull],
    old_hulls:          Dict[str, KernelConicHull],
    variance_threshold: float = 0.95,
    min_norm:           float = 1e-8,
    verbose:            bool  = True,
) -> int:
    """
    Rotate every hull in `new_hulls` into the null space of the old-class subspace.

    Stacks all extreme rays from `old_hulls` into a single (M, D) matrix, then
    calls hull.rotate_into_null_space() on each hull in `new_hulls` in-place.

    The null space is computed once (shared SVD) so the cost is O(M·D²) for the
    SVD plus O(K·D) per hull for the projection — far cheaper than refitting.

    Parameters
    ----------
    new_hulls           : hulls whose rays will be projected (mutated in-place).
                          Typically the freshly-fit hulls for the current stage.
    old_hulls           : hulls from all previous stages; their rays define the
                          subspace to be removed.
    variance_threshold  : fraction of squared-SV energy kept when building the
                          old-class basis (passed through to rotate_into_null_space).
    min_norm            : rays below this post-projection norm are discarded.
    verbose             : print a summary line when done.

    Returns
    -------
    Number of hulls that were successfully rotated.
    """
    if not old_hulls:
        return 0

    old_ray_list = [
        h.extreme_rays_
        for h in old_hulls.values()
        if h.extreme_rays_ is not None and len(h.extreme_rays_) > 0
    ]
    if not old_ray_list:
        return 0

    all_old_rays = np.vstack(old_ray_list)   # (M, D)

    n_rotated = 0
    for hull in new_hulls.values():
        if hull.extreme_rays_ is None:
            continue
        prev_k = len(hull.extreme_rays_)
        hull.rotate_into_null_space(all_old_rays, variance_threshold, min_norm)
        n_rotated += 1
        if verbose:
            new_k = len(hull.extreme_rays_)
            if new_k < prev_k:
                print(f"  [NullRotation] {prev_k} → {new_k} rays "
                      f"(discarded {prev_k - new_k} collapsed rays)")

    if verbose:
        print(f"  [NullRotation] Rotated {n_rotated}/{len(new_hulls)} kernel hulls "
              f"into null space of {len(old_hulls)} old classes "
              f"({all_old_rays.shape[0]} total old rays, "
              f"variance_threshold={variance_threshold}).")
    return n_rotated


# ─────────────────────────────────────────────────────────────────────────────
# Layer-stratified conic hull classifier
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_index_for_stage(stage_idx: int, num_stages: int, num_layers: int,
                               strategy: str = "linear") -> int:
    """
    Map a continual-learning stage index to a backbone block index.

    Strategies
    ----------
    "linear"  : stages spread evenly across layers (0 → layer 0,
                last stage → last layer).
    "last"    : every stage uses the final layer (same as standard).
    "first"   : every stage uses the first layer.
    "cyclic"  : stage_idx % num_layers.
    """
    if num_layers == 0:
        return 0
    if strategy == "linear":
        if num_stages <= 1:
            return num_layers - 1
        return round(stage_idx / (num_stages - 1) * (num_layers - 1))
    if strategy == "last":
        return num_layers - 1
    if strategy == "first":
        return 0
    if strategy == "cyclic":
        return stage_idx % num_layers
    raise ValueError(f"Unknown layer_strategy: {strategy!r}")


class LayerFeatureExtractor:
    """
    Registers forward hooks on backbone transformer blocks so that after a
    forward pass the hidden states from specific layers are available.

    Designed for timm ViT models where blocks live in ``backbone.blocks``,
    but will also try ``backbone.layers`` and ``backbone.encoder.layers``.
    For each registered block the hook stores the CLS token
    (``out[:, 0, :]``) for 3-D outputs or the raw output otherwise.

    Usage
    -----
    extractor = LayerFeatureExtractor(backbone, [0, 4, 8, 11])
    backbone(batch)                       # triggers hooks
    feats = extractor[4]                  # (B, D) tensor, on CPU
    extractor.clear()                     # reset cache between batches
    extractor.remove()                    # unregister all hooks
    """

    def __init__(self, backbone, layer_indices: list):
        self.backbone      = backbone
        self.layer_indices = list(layer_indices)
        self._cache: Dict[int, any] = {}
        self._handles: list = []
        self._register()

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_blocks(self):
        for attr in ("blocks", "layers", "encoder.layers"):
            obj = self.backbone
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise AttributeError(
            "Cannot locate transformer blocks on backbone. "
            "Expected attribute: .blocks, .layers, or .encoder.layers"
        )

    def _register(self):
        blocks = self._get_blocks()
        for idx in self.layer_indices:
            def _hook(module, inp, out, _idx=idx):
                if out.dim() == 3:          # (B, tokens, D) — take CLS token
                    self._cache[_idx] = out[:, 0, :].detach().cpu()
                else:
                    self._cache[_idx] = out.detach().cpu()
            self._handles.append(blocks[idx].register_forward_hook(_hook))

    # ── public API ────────────────────────────────────────────────────────────

    def __getitem__(self, layer_idx: int):
        """Return cached (B, D) CPU tensor for *layer_idx*, or None."""
        return self._cache.get(layer_idx)

    def clear(self):
        """Reset the feature cache (call before each forward pass)."""
        self._cache.clear()

    def remove(self):
        """Unregister all hooks and clear the cache."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._cache.clear()

    @property
    def num_backbone_layers(self) -> int:
        return len(self._get_blocks())


class LayeredConicHullClassifier:
    """
    Continual-learning conic hull classifier where stage *k* builds its
    hulls from backbone layer ``layer_indices[k]`` rather than always using
    the final representation.

    Motivation
    ----------
    Earlier layers capture coarser, more transferable structure; later layers
    encode task-specific fine-grained patterns.  Anchoring each stage's hull
    to the layer that was "active" when that stage was trained reduces
    inter-task representation drift and makes the cascade more robust.

    Inference: regressive cascade
    -----------------------------
    Stages are tested in order 0 → K.  The first stage whose best class-hull
    cosine score ≥ ``threshold`` claims the sample.  If no stage exceeds the
    threshold the globally highest-scoring (stage, class) pair wins.

    Parameters
    ----------
    n_rays      : extreme rays per class hull.
    use_pca     : apply PCA before SPA (set False when layer dim is already
                  moderate, e.g. 192 for ViT-Tiny blocks).
    pca_dim     : PCA target dim (ignored when use_pca=False).
    min_samples : skip classes with fewer training samples than this.
    """

    def __init__(self, n_rays: int = 50, use_pca: bool = False,
                 pca_dim: int = 64, min_samples: int = 3):
        self.n_rays      = n_rays
        self.use_pca     = use_pca
        self.pca_dim     = pca_dim
        self.min_samples = min_samples
        self.stage_layer_map: Dict[int, int]              = {}
        self.stage_hulls:     Dict[int, Dict[str, ConicHull]] = {}

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit_stage(self, stage_idx: int, layer_idx: int,
                  feature_dict: Dict[str, np.ndarray]) -> int:
        """
        Fit per-class ConicHulls for one stage from layer *layer_idx* features.

        Parameters
        ----------
        stage_idx    : sequential stage identifier (0, 1, 2, …)
        layer_idx    : which backbone block these features came from
        feature_dict : {class_str: (N, D) float32 array of L2-normalised feats}

        Returns
        -------
        Number of class hulls fitted.
        """
        self.stage_layer_map[stage_idx] = layer_idx
        hulls: Dict[str, ConicHull] = {}
        for cls_str, feats in feature_dict.items():
            if len(feats) < self.min_samples:
                continue
            hull = KernelConicHull(
                n_rays = 50,
                kernel = "rbf",
                gamma = 5.0,               
            )
            
            # hull = ConicHull(
            #     n_rays  = min(self.n_rays, len(feats)),
            #     use_pca = self.use_pca,
            #     pca_dim = self.pca_dim,
            # )
            hull.fit(feats)
            hulls[cls_str] = hull
        self.stage_hulls[stage_idx] = hulls
        return len(hulls)

    # ── inference ─────────────────────────────────────────────────────────────

    def _best_in_stage(self, stage_idx: int,
                       feat_vec: np.ndarray) -> tuple:
        """(best_class_str, best_score) for a single (D,) feature vector."""
        hulls = self.stage_hulls.get(stage_idx, {})
        if not hulls:
            return "", -np.inf
        best_cls, best_score = "", -np.inf
        query = feat_vec[np.newaxis, :]          # (1, D)
        for cls_str, hull in hulls.items():
            s = float(hull.score(query)[0])
            if s > best_score:
                best_score, best_cls = s, cls_str
        return best_cls, best_score

    def predict_one(self, layer_cache: Dict[int, np.ndarray],
                    threshold: float = 0.0) -> tuple:
        """
        Cascade prediction for a single sample.

        Parameters
        ----------
        layer_cache : {layer_idx: (D,) L2-normalised feature vector}
        threshold   : minimum cosine score for a stage to claim the sample.
                      With threshold=0.0 the first stage always wins (use a
                      positive value such as 0.3–0.6 to activate the cascade).

        Returns
        -------
        (predicted_class_str, winning_stage_idx, winning_score)
        """
        global_best: tuple = ("", -1, -np.inf)
        for stage_idx in sorted(self.stage_hulls):
            layer_idx = self.stage_layer_map[stage_idx]
            feat_vec  = layer_cache.get(layer_idx)
            if feat_vec is None:
                continue
            cls, score = self._best_in_stage(stage_idx, feat_vec)
            if score > global_best[2]:
                global_best = (cls, stage_idx, score)
            if score >= threshold:
                return cls, stage_idx, score
        return global_best

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate_cascade(self, backbone, test_loader,
                         extractor: "LayerFeatureExtractor",
                         device, threshold: float = 0.0) -> Dict[str, float]:
        """
        Run cascade inference over a DataLoader and return accuracy metrics.

        Returns
        -------
        dict with keys:
          "accuracy"          – fraction of correctly classified samples
          "avg_winning_stage" – mean stage index that claimed the sample
                                (low = earlier stages dominate)
        """
        import torch
        import torch.nn.functional as F

        backbone.eval()
        correct        = 0
        total          = 0
        winning_stages: list = []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                extractor.clear()
                backbone(imgs)                   # triggers hooks

                # Collect normalised (B, D) arrays per layer
                layer_arrays: Dict[int, np.ndarray] = {}
                for li in extractor.layer_indices:
                    t = extractor[li]
                    if t is not None:
                        layer_arrays[li] = F.normalize(t, p=2, dim=1).numpy()

                for b_idx, lbl in enumerate(labels.tolist()):
                    sample_cache = {li: arr[b_idx]
                                    for li, arr in layer_arrays.items()}
                    pred_cls, win_stage, _ = self.predict_one(
                        sample_cache, threshold
                    )
                    if pred_cls == str(lbl):
                        correct += 1
                    winning_stages.append(win_stage)
                    total += 1

        return {
            "accuracy":          correct / total if total > 0 else 0.0,
            "avg_winning_stage": float(np.mean(winning_stages))
                                 if winning_stages else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collaborative scoring (Sparse Representation Classification style)
# ─────────────────────────────────────────────────────────────────────────────

def _joint_nnls_fista(
    queries_norm: np.ndarray,    # (N, D) float32, L2-normalized
    R_joint:      np.ndarray,    # (K_total, D) float32, joint ray dictionary
    lasso_lambda: float = 0.0,   # L1 coefficient (0 = pure NNLS)
    max_iter:     int   = 500,
    tol:          float = 1e-6,
    batch_size:   int   = 4096,
) -> np.ndarray:
    """
    FISTA solver for joint NNLS/LASSO over concatenated class dictionary.

    Solves per query batch:
        min_{W ≥ 0}  ‖W R_joint - Q‖²_F / 2  +  lasso_lambda · ‖W‖₁

    The proximal operator of (λ‖·‖₁ + I_{·≥0}) is max(z - λ, 0), so the
    LASSO+NN constraint collapses to a single soft-threshold-then-clamp step.

    Uses GPU (PyTorch CUDA) when available; falls back to CPU otherwise.

    Returns W of shape (N, K_total).
    """
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except ImportError:
        _cuda = False

    if _cuda:
        return _joint_nnls_fista_torch(
            queries_norm, R_joint, lasso_lambda, max_iter, tol, batch_size,
        )

    # ── CPU fallback: per-sample scipy NNLS (lasso_lambda ignored on CPU) ────
    from scipy.optimize import nnls as _nnls

    N, K = queries_norm.shape[0], R_joint.shape[0]
    W_out = np.zeros((N, K), dtype=np.float32)
    A = R_joint.T.astype(np.float64)           # (D, K) for scipy NNLS
    for i, q in enumerate(queries_norm):
        w, _ = _nnls(A, q.astype(np.float64))
        W_out[i] = w.astype(np.float32)
    return W_out


def _joint_nnls_fista_torch(
    queries_norm: np.ndarray,
    R_joint:      np.ndarray,
    lasso_lambda: float,
    max_iter:     int,
    tol:          float,
    batch_size:   int,
) -> np.ndarray:
    import math
    import torch

    device = torch.device("cuda")
    dtype  = torch.float32

    R = torch.tensor(R_joint, dtype=dtype, device=device)       # (K, D)
    Q_all = torch.tensor(queries_norm, dtype=dtype, device=device)  # (N, D)

    # Lipschitz constant of gradient = λ_max(R R^T)
    # Use power iteration (cheap vs full SVD for large K)
    with torch.no_grad():
        v = torch.randn(R.shape[0], device=device, dtype=dtype)
        v = v / v.norm()
        for _ in range(60):
            u = R @ (R.T @ v)      # R R^T v without materialising R R^T
            L_val = u.norm().item()
            if L_val < 1e-10:
                break
            v = u / L_val
    lr = 1.0 / (L_val + 1e-8)

    N, K = queries_norm.shape[0], R_joint.shape[0]
    lam  = float(lasso_lambda)
    W_out = torch.zeros(N, K, dtype=dtype)          # collect on CPU

    for start in range(0, N, batch_size):
        Q = Q_all[start : start + batch_size]        # (B, D)
        B = Q.shape[0]

        # C[i, k] = q_i · r_k  (dot-products: query vs ray)
        C = Q @ R.T                                   # (B, K)

        W = torch.zeros(B, K, dtype=dtype, device=device)
        Y = torch.zeros(B, K, dtype=dtype, device=device)
        t = 1.0

        for i in range(max_iter):
            # Gradient of ½‖W R - Q‖²_F  w.r.t. W:  W R R^T - Q R^T = (W R - Q) R^T
            recon  = Y @ R                            # (B, D)
            grad   = (recon - Q) @ R.T               # (B, K)

            # Proximal step: soft-threshold then clamp to [0, ∞)
            W_next = (Y - lr * grad - lr * lam).clamp_(min=0.0)

            t_next   = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            momentum = (t - 1.0) / t_next
            Y = W_next + momentum * (W_next - W)
            W = W_next
            t = t_next

            if i % 50 == 0 and i > 0:
                with torch.no_grad():
                    delta = (W_next - W).norm() / (W.norm() + 1e-8)
                    if delta.item() < tol:
                        break

        W_out[start : start + batch_size] = W.cpu()

    return W_out.numpy()


def collaborative_nnls_scores(
    queries:       np.ndarray,
    hulls:         Dict[str, object],
    lasso_lambda:  float                   = 0.0,
    temperature:   Optional[Dict[str, float]] = None,
    max_iter:      int                     = 500,
    tol:           float                   = 1e-6,
    batch_size:    int                     = 4096,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Joint nonneg lasso over all class dictionaries (Sparse-Representation-style).

    Instead of C independent per-class NNLS solves, concatenate every class's
    extreme rays into one dictionary D = [R_1; R_2; ...; R_C] and solve *one*
    joint program:

        min_{w ≥ 0}  ‖D w − q‖²  +  lasso_lambda · ‖w‖₁

    Cross-class competition inside the single program suppresses the
    "neighbor class reconstructs query almost as well" failure that biases
    independent per-class residuals when cones are geometrically correlated.

    Classification signals computed from the joint solution w*:
    ─────────────────────────────────────────────────────────
    collab_energy    : Σ_{k ∈ class c} w*_k  (L1 attribution — how much of
                       the solution budget did class c's rays absorb?)
    collab_residual  : 1 − ‖q − R_c w*_c‖ / 2  (partial residual — how
                       well does class c's own rays reconstruct q alone,
                       using only the jointly-solved weights?)
    collab_margin    : energy_c − 2nd_best_energy  (relative gap; suppresses
                       scale variation across queries)

    Parameters
    ----------
    queries       : (N, D) feature vectors (need not be L2-normalized).
    hulls         : ordered dict {class_name: ConicHull-like object}.
    lasso_lambda  : L1 penalty weight (0 = pure NNLS).  A small value such
                    as 1e-3 promotes sparser, cleaner class attribution.
    temperature   : per-class temperature τ_c; each class's energy score is
                    divided by τ_c before argmax.  Set τ_c > 1 to cool down
                    (suppress) class c; τ_c < 1 to amplify it.  None = no
                    calibration.
    max_iter, tol, batch_size : passed to the FISTA solver.

    Returns
    -------
    scores     : dict {scheme_name: (N, C) float32 ndarray}
    class_names: list of C class names, in column order.
    """
    class_names = list(hulls.keys())
    C = len(class_names)
    N = len(queries)
    queries_norm = normalize(queries.astype(np.float32), axis=1)  # (N, D)

    # Build joint dictionary and record per-class slice boundaries
    ray_blocks: list = []
    boundaries: list = []     # (start, end) index into joint dictionary rows
    offset = 0
    for name in class_names:
        hull = hulls[name]
        rays = getattr(hull, "extreme_rays_", None)
        if rays is None or len(rays) == 0:
            boundaries.append((offset, offset))
            continue
        R = np.asarray(rays, dtype=np.float32)
        ray_blocks.append(R)
        boundaries.append((offset, offset + len(R)))
        offset += len(R)

    if not ray_blocks:
        zeros = np.zeros((N, C), dtype=np.float32)
        return {
            "collab_energy":   zeros,
            "collab_residual": zeros,
            "collab_margin":   zeros,
        }, class_names

    R_joint = np.vstack(ray_blocks)   # (K_total, D)

    # One joint FISTA NNLS solve for all queries
    W = _joint_nnls_fista(
        queries_norm, R_joint, lasso_lambda, max_iter, tol, batch_size,
    )   # (N, K_total)

    # ── per-class attributed scores ──────────────────────────────────────────
    energy      = np.zeros((N, C), dtype=np.float32)
    partial_res = np.zeros((N, C), dtype=np.float32)

    for c_idx, (name, (s, e)) in enumerate(zip(class_names, boundaries)):
        if s == e:
            continue
        W_c   = W[:, s:e]             # (N, K_c)
        R_c   = R_joint[s:e]          # (K_c, D)

        # Attributed energy: total nonneg weight flowing to class c's rays
        energy[:, c_idx] = W_c.sum(axis=1)

        # Partial residual: reconstruct query using ONLY class c's rays & weights
        recon_c = W_c @ R_c                                      # (N, D)
        res     = np.linalg.norm(queries_norm - recon_c, axis=1) # (N,) ∈ [0, 2]
        partial_res[:, c_idx] = 1.0 - res * 0.5                  # ∈ [0, 1]

    # ── per-class temperature calibration ────────────────────────────────────
    if temperature is not None:
        for c_idx, name in enumerate(class_names):
            T = float(temperature.get(name, 1.0))
            if T > 0.0 and T != 1.0:
                energy[:, c_idx]      /= T
                partial_res[:, c_idx] /= T

    # ── relative margin: energy gap over runner-up ───────────────────────────
    # Sort columns descending; margin[i, c] = energy[i, c] − 2nd_best[i]
    # This penalises tied or ambiguous cases without changing the argmax winner.
    sorted_e = np.sort(energy, axis=1)[:, ::-1]          # (N, C) descending
    runner_up = sorted_e[:, 1:2] if C > 1 else np.zeros((N, 1), dtype=np.float32)
    collab_margin = energy - runner_up                    # (N, C)

    return {
        "collab_energy":   energy,
        "collab_residual": partial_res,
        "collab_margin":   collab_margin,
    }, class_names