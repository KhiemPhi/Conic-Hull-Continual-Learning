import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from conic_hull import ConicHull, build_class_conic_hulls
import timm

from features import get_default_transform
from backbone import load_backbone, get_lora_params
import numpy as np
import os


import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


import math
import random 
from collections import Counter, defaultdict

class HullManager:
    def __init__(self, n_rays=50, use_pca=True, pca_dim=64):
        self.n_rays = n_rays
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.static_hulls = {}       # Frozen at birth
        self.extreme_ray_images = {} # cls_str -> list of CPU image tensors

    def fit_new_classes(self, feature_dict):
        """Fits and freezes hulls for classes seen for the first time."""
        new_hulls = build_class_conic_hulls(
            feature_dict,
            n_rays=self.n_rays,
            use_pca=self.use_pca,
            pca_dim=self.pca_dim
        )
        for cls, hull in new_hulls.items():
            if cls not in self.static_hulls:
                self.static_hulls[cls] = hull

    def get_dynamic_hulls(self, feature_dict):
        """Fits hulls to the current representations (Dynamic)."""
        return build_class_conic_hulls(
            feature_dict,
            n_rays=self.n_rays,
            use_pca=self.use_pca,
            pca_dim=self.pca_dim
        )

    def evaluate(self, backbone, test_loader, hulls, device):
        """Classification rule: argmax of hull.score()"""
        backbone.eval()
        correct, total = 0, 0
        all_classes = list(hulls.keys())
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                feats = backbone(imgs.to(device)).cpu().numpy()
                # (N_queries, N_classes)
                scores = np.stack([hulls[name].score(feats) for name in all_classes], axis=1)
                
                best_idx = np.argmax(scores, axis=1)
                preds = np.array([int(all_classes[i]) for i in best_idx])
                correct += (preds == labels.numpy()).sum()
                total += labels.size(0)
        return correct / total

class ReplayBuffer:
    def __init__(self, max_total_size: int, max_classes: int = 100):
        self.max_total_size = max_total_size
        self.max_classes = max_classes
        self.buffer = defaultdict(list) # Now stores tuples: {class_id: [(image_tensor, emb_tensor), ...]}
        self.all_classes = []
       
        self.per_class_cap = int(max_total_size // max_classes)
        

    def add(self, images: torch.Tensor, embeddings: torch.Tensor, labels: torch.Tensor):
        """Adds images and their embeddings to the buffer, ensuring class balance."""
        images_cpu = images.detach().cpu()
        embeddings_cpu = embeddings.detach().cpu()
        labels_cpu = labels.detach().cpu()
        
        for img, emb, lbl in zip(images_cpu, embeddings_cpu, labels_cpu):
            class_id = lbl.item()
            if class_id not in self.buffer:
                self.all_classes.append(class_id)
            
            # Store as a paired tuple
            self.buffer[class_id].append((img, emb, lbl))
            
            # Keep a per-class cap to prevent memory explosion
           
            if len(self.buffer[class_id]) > self.per_class_cap:
                self.buffer[class_id].pop(random.randrange(len(self.buffer[class_id])))

    def sample(self, batch_size: int, strategy: str = "balanced"):
        """Modular sampling gateway. Returns (images, embeddings, labels)."""
        if not self.buffer:
            return None, None, None

        if strategy == "balanced":
            return self._sample_balanced(batch_size)
        elif strategy == "uniform":
            return self._sample_uniform(batch_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _sample_balanced(self, batch_size: int):
        """Ensures every class seen so far is represented equally in the batch."""
        n_classes = len(self.all_classes)
        samples_per_class = max(1, batch_size // n_classes)

        batch_imgs = []
        batch_embs = []
        batch_lbls = []

        for class_id in self.all_classes:
            class_pool = self.buffer[class_id]
            # Sample with replacement if class_pool is smaller than requested
            indices = torch.randint(0, len(class_pool), (samples_per_class,))
            for idx in indices:
                img, emb, lbl = class_pool[idx]
                batch_imgs.append(img)
                batch_embs.append(emb)
                batch_lbls.append(lbl)

        return torch.stack(batch_imgs), torch.stack(batch_embs), torch.stack(batch_lbls)

    def _sample_uniform(self, batch_size: int):
        """Standard random sampling (biased toward tasks with more samples)."""
        all_pairs = [pair for cls_list in self.buffer.values() for pair in cls_list]
        selected = random.sample(all_pairs, min(batch_size, len(all_pairs)))

        batch_imgs = [pair[0] for pair in selected]
        batch_embs = [pair[1] for pair in selected]
        batch_lbls = [pair[2] for pair in selected]

        return torch.stack(batch_imgs), torch.stack(batch_embs), torch.stack(batch_lbls)

    def get_all(self):
        """
        Returns all stored images, embeddings, and labels as unified tensors.
        Useful for building the SPA Conic Hulls without re-running the backbone.
        """
        all_images = []
        all_embeddings = []
        all_labels = []

        # Ensure we iterate through classes in a consistent order
        for class_id in sorted(self.all_classes):
            class_pool = self.buffer[class_id]
            if not class_pool:
                continue
                
            for img, emb, lbl in class_pool:
                all_images.append(img)
                all_embeddings.append(emb)
                all_labels.append(lbl)

        if not all_images:
            return torch.empty(0), torch.empty(0), torch.empty(0)

        # Return as tensors
        # memory_images: (N_total, C, H, W)
        # memory_embeddings: (N_total, Feature_Dim)
        # memory_labels: (N_total,)
        return torch.stack(all_images), torch.stack(all_embeddings), torch.tensor(all_labels)

class IncrementalLinearHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, 0) # Start empty

    def add_classes(self, num_new_classes: int, device: torch.device):
        old_fc = self.fc
        old_out = old_fc.out_features
        new_out = old_out + num_new_classes
        
        # Create new linear layer
        new_fc = nn.Linear(self.in_features, new_out)
        
        # Copy old weights to the new layer to preserve previous knowledge
        if old_out > 0:
            with torch.no_grad():
                new_fc.weight[:old_out] = old_fc.weight
                new_fc.bias[:old_out] = old_fc.bias
        
        self.fc = new_fc

    def forward(self, x):
        return self.fc(F.normalize(x, p=2, dim=1)) # Normalize features for a fair comparison

class FixedConicHead(nn.Module):
    """
    Step 2 & 3: Pre-allocates all future class cones as frozen vectors.
    """
    def __init__(self, in_features: int, total_classes: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.s = s
        self.m = m
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # Pre-allocate frozen cones for ALL stages
        fixed_weights = torch.randn(total_classes, in_features)
        fixed_weights = F.normalize(fixed_weights, p=2, dim=1)
        self.register_buffer('weight', fixed_weights)

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), self.weight)
        
        if labels is None:
            return cosine * self.s

        # Apply ArcFace margin to force embeddings strictly inside the cone
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

def classify_by_conic_hull(features, head, margin_threshold=None):
    """
    Classifies features based on geometric membership in the conic hulls.
    """
    if margin_threshold is None:
        margin_threshold = head.m  # Use the training margin as the radius

    # 1. Normalize and find angles to all centers
    features_norm = F.normalize(features, p=2, dim=1)
    # Cosine similarity -> Angle in radians
    cosines = F.linear(features_norm, head.weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cosines)

    # 2. Determine membership: Angle must be <= Radius
    # Find the closest cone for every sample
    min_angles, preds = angles.min(dim=1)
    
    # 3. Apply the Hull Constraint
    # If the closest angle is still larger than the margin, it's not in the hull
    is_inside_any_hull = min_angles <= margin_threshold
    
    # Optional: Mark orphans as -1 or a special 'unknown' class
    final_preds = torch.where(is_inside_any_hull, preds, torch.tensor(-1).to(preds.device))
    
    return final_preds, min_angles

class IncrementalConicHead(nn.Module):
    def __init__(self, in_features: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.s = s
        self.m = m
        
        # Pre-compute constants for the ArcFace margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # Initialize an empty buffer for weights
        self.register_buffer('weight', torch.empty(0, in_features))

    def add_classes(self, num_new_classes: int, device: torch.device):
        # Radius of each cone is m. For no overlap, distance must be > 2*m
        min_separation = 2 * self.m 
        new_centers = []

        while len(new_centers) < num_new_classes:
            # 1. Propose a new random center
            proposal = torch.randn(1, self.in_features).to(device)
            proposal = F.normalize(proposal, p=2, dim=1)

            # 2. Check against ALL existing centers (if any)
            if self.weight.numel() > 0:
                # Calculate cosines with existing weights
                cosines = F.linear(proposal, self.weight)
                # Find the max cosine (smallest angle)
                max_cos = cosines.max().item()
                # Convert to angle
                min_angle = math.acos(max_cos)
                
                if min_angle < min_separation:
                    continue # Too close! Re-roll.
            
            # 3. Check against other centers currently being added in this batch
            if len(new_centers) > 0:
                current_new = torch.cat(new_centers, dim=0)
                cosines = F.linear(proposal, current_new)
                if math.acos(cosines.max().item()) < min_separation:
                    continue # Too close to another new class! Re-roll.

            new_centers.append(proposal)

        # Append to the frozen buffer
        new_weights = torch.cat(new_centers, dim=0)
        if self.weight.numel() == 0:
            self.weight = new_weights
        else:
            self.weight = torch.cat([self.weight, new_weights], dim=0)

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # Normalize features to the unit hypersphere
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Calculate cosine similarity (dot product) with all existing centers
        cosine = F.linear(features_norm, self.weight)
        
        if labels is None:
            return cosine * self.s

        # Apply ArcFace margin logic
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for the current batch labels
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Inject the margin only into the target class logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

class IncrementalMLPHead(nn.Module):
    """
    Two-layer MLP classifier head that grows its output dimension as new classes
    arrive, keeping old neuron weights frozen for ablation studies.

    Architecture
    ------------
    Input (D)  →  Linear(D, hidden_dim)  →  GELU  →  LayerNorm  →  Linear(hidden_dim, C)

    The hidden projection is fixed at init and never grows; only the final
    output layer expands.  Old output weights and biases are preserved exactly
    when new classes are appended so the head retains its prior knowledge even
    without replay.

    Parameters
    ----------
    in_features  : backbone feature dimension
    hidden_dim   : width of the hidden layer (default 512)
    dropout      : dropout probability applied after GELU (0 = disabled)
    """

    def __init__(self, in_features: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim  = hidden_dim

        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout),
        )
        # Start with zero output classes; expanded by add_classes()
        self.fc = nn.Linear(hidden_dim, 0)

    def add_classes(self, num_new_classes: int, device: torch.device) -> None:
        """Grow the output layer by *num_new_classes*, preserving old weights."""
        old_fc  = self.fc
        old_out = old_fc.out_features
        new_out = old_out + num_new_classes

        new_fc = nn.Linear(self.hidden_dim, new_out).to(device)

        if old_out > 0:
            with torch.no_grad():
                new_fc.weight[:old_out] = old_fc.weight
                new_fc.bias[:old_out]   = old_fc.bias

        self.fc = new_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)   # L2-normalise input, consistent with other heads
        h = self.projection(x)
        return self.fc(h)

    # Expose a .weight property so log_barrier_margin_loss and
    # update_head_weights_analytically can access classifier directions uniformly.
    # We return the *effective* weight matrix W = fc.weight @ projection[0].weight
    # (shape: num_classes × in_features), normalised row-wise.
    @property
    def weight(self) -> torch.Tensor:
        W_proj = self.projection[0].weight          # (hidden_dim, in_features)
        W_out  = self.fc.weight                     # (num_classes, hidden_dim)
        W_eff  = W_out @ W_proj                     # (num_classes, in_features)
        return F.normalize(W_eff, p=2, dim=1)


# ─── Analytical Classifier Weight Correction ─────────────────────────────────

class ClassStatisticsStore:
    """
    Maintains per-class feature statistics (mean and covariance) for analytical
    correction of classifier weights after backbone drift.

    Usage pattern
    -------------
    1. After each stage, call ``update(class_id, features)`` for every old class
       using the *current* backbone to get a consistent snapshot.
    2. Before the next stage's training, call ``snapshot()`` to freeze those stats.
    3. After training, call ``update(...)`` again with the *new* backbone to get
       post-drift stats.  Pass both dicts to ``estimate_affine_drift``.
    """

    def __init__(self):
        self.stats: Dict[int, Dict] = {}  # class_id -> {mean, cov, n}

    def update(self, class_id: int, features: np.ndarray) -> None:
        """Compute and store mean / cov from a feature matrix of shape (N, D)."""
        n, d = features.shape
        mean = features.mean(axis=0)
        cov  = np.cov(features.T, ddof=1) if n > 1 else np.zeros((d, d))
        self.stats[class_id] = {"mean": mean, "cov": cov, "n": n}

    def update_from_hull(self, class_id: int, hull) -> None:
        """Use a hull's stored extreme rays as lightweight class statistics."""
        if hull.extreme_rays_ is not None:
            self.update(class_id, hull.extreme_rays_)

    def snapshot(self) -> Dict:
        """Return a deep copy of current statistics (safe to mutate independently)."""
        return {
            k: {kk: (v.copy() if isinstance(v, np.ndarray) else v)
                for kk, v in vv.items()}
            for k, vv in self.stats.items()
        }

    def get_class_ids(self) -> List[int]:
        return list(self.stats.keys())


def estimate_affine_drift(
    old_stats: Dict,
    new_stats: Dict,
    method: str = "procrustes",
) -> np.ndarray:
    """
    Estimates the linear transformation **A** that maps the old feature space to
    the new one after backbone fine-tuning.

    Procrustes (default)
    --------------------
    Finds the best-fit orthogonal A minimising ||M_new - M_old A^T||_F, where
    M_old and M_new are (C × D) matrices of per-class means.

        M_old^T M_new = U S V^T   (thin SVD)
        A             = V U^T      (D × D, orthogonal)

    For orthogonal A the inverse-transpose equals A itself, so the weight
    update w_new = A^{-T} w_old reduces to a simple rotation.

    Covariance
    ----------
    Finds A such that  A Σ_old A^T = Σ_new  using pooled Cholesky factors:

        A = L_new @ L_old^{-1}   where  Σ = L L^T

    Falls back to an eigendecomposition-based symmetric square root when
    the Cholesky factorisation fails (near-singular covariance).

    Parameters
    ----------
    old_stats : {class_id: {"mean": ndarray(D,), "cov": ndarray(D,D), "n": int}}
    new_stats : same structure, computed after backbone training
    method    : "procrustes" | "covariance"

    Returns
    -------
    A : (D, D) ndarray — estimated feature-space drift matrix
    """
    common = sorted(set(old_stats.keys()) & set(new_stats.keys()))
    if len(common) < 2:
        raise ValueError(
            f"estimate_affine_drift needs ≥2 shared classes; found {len(common)}."
        )

    D = old_stats[common[0]]["mean"].shape[0]

    if method == "procrustes":
        M_old = np.stack([old_stats[c]["mean"] for c in common])   # (C, D)
        M_new = np.stack([new_stats[c]["mean"] for c in common])   # (C, D)

        # Centre both mean matrices so the solution is invariant to translation
        M_old_c = M_old - M_old.mean(axis=0, keepdims=True)
        M_new_c = M_new - M_new.mean(axis=0, keepdims=True)

        # Orthogonal Procrustes: M_old^T M_new = U S V^T  =>  A = V U^T
        U, _S, Vt = np.linalg.svd(M_old_c.T @ M_new_c, full_matrices=False)
        A = Vt.T @ U.T                                              # (D, D)
        return A

    elif method == "covariance":
        # Build pooled covariances (sample-count weighted)
        Sigma_old = np.zeros((D, D))
        Sigma_new = np.zeros((D, D))
        total_n   = 0

        for c in common:
            n          = old_stats[c]["n"]
            Sigma_old += n * old_stats[c]["cov"]
            Sigma_new += n * new_stats[c]["cov"]
            total_n   += n

        Sigma_old /= total_n
        Sigma_new /= total_n

        # Tikhonov regularisation for numerical stability
        reg        = 1e-5 * np.eye(D)
        Sigma_old += reg
        Sigma_new += reg

        try:
            L_old = np.linalg.cholesky(Sigma_old)
            L_new = np.linalg.cholesky(Sigma_new)
            # A L_old = L_new  =>  A = L_new @ L_old^{-1}
            A = np.linalg.solve(L_old.T, L_new.T).T               # (D, D)
        except np.linalg.LinAlgError:
            # Symmetric matrix square-root fallback via eigendecomposition
            eps = 1e-8
            vals_o, vecs_o = np.linalg.eigh(Sigma_old)
            vals_n, vecs_n = np.linalg.eigh(Sigma_new)
            sqrt_o_inv = (vecs_o
                          @ np.diag(1.0 / np.sqrt(np.maximum(vals_o, eps)))
                          @ vecs_o.T)
            sqrt_n     = (vecs_n
                          @ np.diag(np.sqrt(np.maximum(vals_n, eps)))
                          @ vecs_n.T)
            A = sqrt_n @ sqrt_o_inv

        return A

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'procrustes' or 'covariance'."
        )


def rotate_conic_hulls(
    old_stats: Dict,
    new_stats: Dict,
    A_global: np.ndarray,
    boundary_margin: float = 0.05,
) -> np.ndarray:
    """
    Refines a global drift matrix A into a boundary-aware rotation via polar
    decomposition and class-conditional angular width scaling.

    **Motivation.**  A global affine A minimises Euclidean MSE on class means but
    may "squash" inter-class angles in high dimensions, allowing conic hulls to
    bleed into each other after the transformation.  This function replaces the
    raw A with a corrected matrix that (a) uses only the rotation component of A
    so inter-class angles are isometrically preserved, and (b) applies a
    centripetal contraction proportional to how much wider the new cones are
    relative to the old ones, pulling class boundaries away from one another.

    **Algorithm**

    1.  *Angular width* per class is estimated as the square root of the largest
        eigenvalue of the class covariance divided by the mean norm — a proxy for
        the half-angle subtended by the conic hull:

            w_c = sqrt( λ_max(Σ_c) ) / ( ‖μ_c‖ + ε )

    2.  A per-class *scale factor* compares old and new widths:

            s_c = min( 1, (w_old_c / w_new_c) × (1 − margin) )

        When the new cone is wider (w_new > w_old) the factor is < 1,
        applying centripetal pressure.  When it is narrower, s_c = 1
        (no expansion allowed — only contraction is protective).

    3.  The global A is polar-decomposed:

            A = U Σ V^T  →  R = U V^T    (pure rotation / isometry)

        R preserves all L2 distances and inter-class angles by construction.

    4.  The protected matrix returned is:

            A_protected = R × mean(s_c)

    Parameters
    ----------
    old_stats       : per-class stats dict before backbone training
                      { class_id: {"mean": (D,), "cov": (D,D), "n": int} }
    new_stats       : same structure, after backbone training
    A_global        : (D, D) drift matrix from ``estimate_affine_drift``
    boundary_margin : safety margin subtracted from the scale factor,
                      preventing exact-boundary cases.  0.05 = 5% buffer.

    Returns
    -------
    A_protected : (D, D) ndarray — rotation-only transformation with
                  boundary-contraction scaling applied
    """
    common = sorted(set(old_stats.keys()) & set(new_stats.keys()))

    def _angular_width(stats: Dict, class_id: int) -> float:
        mu  = stats[class_id]["mean"]
        cov = stats[class_id]["cov"]
        lam_max = float(np.linalg.eigvalsh(cov)[-1])          # largest eigenvalue
        return np.sqrt(max(lam_max, 0.0)) / (np.linalg.norm(mu) + 1e-8)

    scales = []
    for c in common:
        w_old = _angular_width(old_stats, c)
        w_new = _angular_width(new_stats, c)
        if w_new > 1e-12:
            s = min(1.0, (w_old / w_new) * (1.0 - boundary_margin))
        else:
            s = 1.0
        scales.append(s)

    s_factor = float(np.mean(scales))

    # Polar decomposition: A = U diag(S) V^T  →  R = U V^T (isometric part)
    U, _, Vt = np.linalg.svd(A_global)
    R = U @ Vt                                                  # (D, D) rotation

    return R * s_factor


def rotate_hulls(
    hulls: Dict[str, "ConicHull"], 
    old_stats: Dict, 
    new_stats: Dict, 
    A: np.ndarray,
    shrinkage: float = 0.95
) -> None:
    # 1. Calculate the Global Translation (Bias)
    common = list(set(old_stats.keys()) & set(new_stats.keys()))
    mu_old_gen = np.mean([old_stats[c]["mean"] for c in common], axis=0)
    mu_new_gen = np.mean([new_stats[c]["mean"] for c in common], axis=0)
    
    for class_name, hull in hulls.items():
        if hull.extreme_rays_ is None:
            continue

        # 2. De-bias, Rotate, and Re-bias
        # This ensures the 'origin' of the conic hull matches the new space
        centered_rays = hull.extreme_rays_ - mu_old_gen
        rotated = centered_rays @ A.T
        shifted = rotated + mu_new_gen
        
        # 3. Boundary Protection (Shrinkage)
        # We push the rays slightly closer to the class mean to prevent overlap
        class_mu_new = new_stats[class_name]["mean"]
        # Pull rays toward the mean by 'shrinkage' amount
        protected_rays = class_mu_new + shrinkage * (shifted - class_mu_new)

        # 4. Final Re-normalization
        norms = np.linalg.norm(protected_rays, axis=1, keepdims=True)
        hull.extreme_rays_ = protected_rays / np.maximum(norms, 1e-12)
        
        # 5. Wipe PCA (It's safer to re-fit or ignore it after non-rigid drift)
        hull.pca_ = None


def update_head_weights_analytically(
    head: nn.Module,
    A: np.ndarray,
    old_class_ids: List[int],
    device: torch.device,
    renormalize: bool = True,
    magnitude_preserving: bool = False,
) -> None:
    """
    Corrects old classifier-head weights to account for feature-space drift.

    **Math.**  If features transform as  x_new ≈ A x_old,  the equivalent
    decision hyperplane satisfies:

        w_new = A^{-T} w_old

    *Derivation.*  The linear score  w^T x  must be invariant:
        w_new^T (A x_old) = w_old^T x_old
        (A^T w_new)       = w_old
        w_new             = A^{-T} w_old

    For orthogonal A (Procrustes output): A^{-T} = A, so the update is a
    pure rotation with no matrix inversion required.

    **Magnitude-preserving mode.**  When A contains a contraction (features
    cluster more tightly), A^{-T} is an expansion.  Without correction, the
    adapted old weights grow in norm and dominate the softmax purely through
    logit volume, not angular advantage.  Setting ``magnitude_preserving=True``
    decouples direction from scale:

        w_final = (w_adapted / ‖w_adapted‖) × ‖w_old‖

    This keeps the directional benefit of A^{-T} while holding the confidence
    level of each old class at its pre-drift value.

    Parameters
    ----------
    head                 : Module with a ``.weight`` tensor (num_classes, D).
    A                    : (D, D) drift matrix from ``estimate_affine_drift``.
    old_class_ids        : Indices of old-class weights to update.
    device               : Computation device.
    renormalize          : Project updated weight onto the unit sphere.
                           Use True for ArcFace heads, False for plain linear.
                           Ignored when ``magnitude_preserving=True``.
    magnitude_preserving : If True, restore ‖w_old‖ after applying A^{-T},
                           decoupling the rotation from the expansion/contraction
                           induced by A.  Recommended when the drift matrix is
                           ill-conditioned or when old and new logit scales diverge.
    """
    A_inv_T   = np.linalg.inv(A).T                                # (D, D)
    A_inv_T_t = torch.tensor(A_inv_T, dtype=head.weight.dtype, device=device)

    with torch.no_grad():
        for class_id in old_class_ids:
            if class_id >= head.weight.shape[0]:
                continue
            w_old     = head.weight[class_id]                      # (D,)
            norm_old  = w_old.norm()
            w_adapted = A_inv_T_t @ w_old                          # (D,)

            if magnitude_preserving and w_adapted.norm() > 1e-8:
                # Use direction of A^{-T} w_old, scale of w_old
                w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0) * norm_old
            elif renormalize:
                w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0)
            else:
                w_new = w_adapted

            head.weight[class_id] = w_new


def update_head_weights_orthogonal_projected(
    head: nn.Module,
    A: np.ndarray,
    old_class_ids: List[int],
    new_class_means: np.ndarray,
    device: torch.device,
    magnitude_preserving: bool = False,
) -> None:
    """
    A^{-T} drift correction with Gram-Schmidt projection onto the orthogonal
    complement of the new-class subspace.

    **Motivation.**  The plain A^{-T} update preserves relative margins among
    old classes but is blind to where new classes have landed.  The adapted old
    weights may rotate directly into the new cones, causing overlapping logits.

    **Math.**
    Let U_new ∈ R^{D×k} be an orthonormal basis for the subspace spanned by the
    new-class mean feature vectors (obtained via thin SVD of the mean matrix).

        P_perp = I − U_new U_new^T       (projector onto orthogonal complement)

    The update proceeds in three steps:

        w_adapted = A^{-T} w_old          (drift correction)
        w_proj    = P_perp w_adapted      (zero out component in new-class directions)
        w_final   = rescale(w_proj)       (see magnitude_preserving below)

    If the projection collapses a vector (‖w_proj‖ < ε), w_adapted is kept as
    a safe fallback.

    Parameters
    ----------
    head                 : classifier module with a ``.weight`` tensor (num_classes, D)
    A                    : (D, D) drift matrix from ``estimate_affine_drift``
    old_class_ids        : indices of old-class weights to update
    new_class_means      : (n_new, D) mean feature vectors of the newly learned classes
    device               : computation device
    magnitude_preserving : If False (default), normalise w_proj to the unit sphere
                           (correct for ArcFace heads).
                           If True, restore ‖w_old‖ after projection to prevent the
                           expansion of A^{-T} from inflating old-class logit confidence.
    """
    D = new_class_means.shape[1]

    _, _, Vt   = np.linalg.svd(new_class_means, full_matrices=False)  # Vt: (k, D)
    U_new      = Vt.T                                                   # (D, k)
    P_perp     = np.eye(D) - U_new @ U_new.T                           # (D, D)
    A_inv_T    = np.linalg.inv(A).T                                     # (D, D)

    P_perp_t   = torch.tensor(P_perp,  dtype=head.weight.dtype, device=device)
    A_inv_T_t  = torch.tensor(A_inv_T, dtype=head.weight.dtype, device=device)

    with torch.no_grad():
        for class_id in old_class_ids:
            if class_id >= head.weight.shape[0]:
                continue
            w_old     = head.weight[class_id]                           # (D,)
            norm_old  = w_old.norm()
            w_adapted = A_inv_T_t @ w_old                               # (D,)
            w_proj    = P_perp_t  @ w_adapted                           # (D,)

            w_base = w_proj if w_proj.norm() > 1e-8 else w_adapted
            if magnitude_preserving:
                w_final = F.normalize(w_base.unsqueeze(0), p=2, dim=1).squeeze(0) * norm_old
            else:
                w_final = F.normalize(w_base.unsqueeze(0), p=2, dim=1).squeeze(0)

            head.weight[class_id] = w_final


def _extract_class_stats_from_buffer(
    backbone: nn.Module,
    replay_buffer: "ReplayBuffer",
    class_ids: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> Dict:
    """
    Helper: pass all replay-buffer images for the given classes through
    *backbone* and return a stats dict suitable for ``estimate_affine_drift``.
    """
    stats: Dict[int, Dict] = {}
    backbone.eval()
    with torch.no_grad():
        for cls_id in class_ids:
            if cls_id not in replay_buffer.buffer or not replay_buffer.buffer[cls_id]:
                continue
            class_imgs = [item[0] for item in replay_buffer.buffer[cls_id]]
            all_feats  = []
            for start in range(0, len(class_imgs), batch_size):
                chunk = torch.stack(class_imgs[start : start + batch_size]).to(device)
                feats = backbone(chunk).cpu().numpy()
                all_feats.append(feats)
            all_feats = np.concatenate(all_feats, axis=0)
            n, d      = all_feats.shape
            mean      = all_feats.mean(axis=0)
            cov       = np.cov(all_feats.T, ddof=1) if n > 1 else np.zeros((d, d))
            stats[cls_id] = {"mean": mean, "cov": cov, "n": n}
    return stats


def log_barrier_margin_loss(
    x_new: torch.Tensor,
    x_old: torch.Tensor,
    labels: torch.Tensor,
    head_weight: torch.Tensor,
    num_seen_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Logarithmic Barrier Margin-Preservation Loss.

    Replaces the isotropic L2 distillation penalty with a direction-aware
    barrier that fires only when features drift *toward* a decision boundary,
    and is silent when drift moves features *deeper into* their own cone.

    **Math.**
    Let w_i be the classifier weight for the true class i and w_k for each
    competing class k.  The geometric margin is:

        m_k(x) = (w_i - w_k)^T x  =  s_i(x) - s_k(x)

    where s_j(x) = w_j^T x is the logit for class j.

    The loss penalises any *shrinkage* of that margin:

        L = mean_{n} mean_{k≠i_n} ReLU( -log( M_new_k / M_old_k ) )

    where  M = max(m_k(x), ε)  and  ε prevents log(0).

    **Properties**
    - Zero when drift is safe (M_new ≥ M_old, ratio ≥ 1, log ≥ 0).
    - Grows as M_new → 0, enforcing an impassable geometric wall at the
      decision boundary without the need to tune an absolute threshold.
    - Scale-invariant: only the *ratio* of margins matters, so it works
      naturally on the unit hypersphere where features live.
    - Handles pre-existing margin violations gracefully: when M_old ≤ ε
      (the old feature was already misclassified), the clamped denominator
      makes the ratio ≥ 1 for any positive M_new, yielding zero loss.

    Parameters
    ----------
    x_new            : (N, D) — new-backbone features, *with* grad
    x_old            : (N, D) — old-backbone features, detached (no grad)
    labels           : (N,)  — true class indices (integers)
    head_weight      : (C, D) — classifier weight matrix (normalized).
                       Gradients w.r.t. this tensor are not needed;
                       the loss is purely a function of x_new through
                       the logit computation.
    num_seen_classes : only consider classes 0..num_seen_classes-1 as
                       competing boundaries.  Unseen future classes have
                       random weights that would inject false gradients.
    eps              : small floor for margin values before taking log.
    """
    N = x_new.shape[0]
    W = F.normalize(head_weight[:num_seen_classes].detach(), p=2, dim=1)  # (C', D)

    # Raw cosine logits using the frozen classifier directions
    x_new_n = F.normalize(x_new,        p=2, dim=1)   # (N, D), retains grad
    x_old_n = F.normalize(x_old.detach(), p=2, dim=1)  # (N, D), no grad

    logits_new = x_new_n @ W.T   # (N, C')
    logits_old = x_old_n @ W.T   # (N, C')

    # True-class scores: (N, 1)
    lbl  = labels.view(-1, 1).long()
    s_i_new = logits_new.gather(1, lbl)   # (N, 1)
    s_i_old = logits_old.gather(1, lbl)   # (N, 1)

    # Per-class margins:  m_k(x) = s_i - s_k  → (N, C')
    margins_new = s_i_new - logits_new
    margins_old = s_i_old - logits_old

    # Mask out the k == i diagonal (margin = 0 by construction)
    diag_mask = torch.zeros(N, W.shape[0], dtype=torch.bool, device=x_new.device)
    diag_mask.scatter_(1, lbl, True)
    margins_new = margins_new.masked_fill(diag_mask, float("inf"))  # skip in mean
    margins_old = margins_old.masked_fill(diag_mask, float("inf"))

    # Clamp denominators (old margins) away from zero
    M_old = margins_old.detach().clamp(min=eps)

    # Clamp numerators so log is always defined; barrier fires when M_new << M_old
    M_new = margins_new.clamp(min=eps)

    # log-ratio: positive → safe, negative → shrinking margin
    log_ratio = torch.log(M_new) - torch.log(M_old)

    # Barrier: penalise only shrinkage; exclude the diagonal (inf entries)
    valid = ~diag_mask
    barrier = F.relu(-log_ratio)                    # (N, C')
    loss = barrier[valid].mean()                    # scalar

    return loss


def logdet_volume_penalty(
    features: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Log-Determinant Volume Penalty.

    Penalises feature-space collapse by encouraging the empirical covariance
    of the current mini-batch to maintain geometric volume.

    **Math.**
    For a batch of features X ∈ ℝ^{N×D}, the empirical covariance is:

        Σ = (1/(N-1)) · (X - μ)^T (X - μ)     μ = col-wise mean

    The regulariser added to the total loss is:

        L_vol = -log det(Σ + ε I)
              = -Σ_i log(λ_i + ε)              (λ_i = eigenvalues of Σ)

    Minimising L_vol (i.e. maximising log-det) maximises the volume of the
    ellipsoid defined by Σ, preventing the backbone from projecting all
    features onto a low-dimensional subspace.

    **Implementation via log-sum of Cholesky diagonal.**
    For a PD matrix  Σ + εI = L Lᵀ  we have:

        log det(Σ + εI) = 2 · Σ_i log L_{ii}

    This avoids a full eigendecomposition and is numerically stable because
    the Cholesky factor always exists after the εI regularisation.

    When N ≤ D the sample covariance is rank-deficient.  We then work in the
    transposed (gram-matrix) space and use the identity:

        log det(Σ + εI) ≈ log det((1/(N-1)) · Z Zᵀ + εI)   Z = X - μ

    which has at most N-1 non-zero eigenvalues and is cheaper to factorise.

    Parameters
    ----------
    features : (N, D) — raw (unnormalised) backbone outputs for the batch.
               Do **not** L2-normalise before calling; volume is a property
               of the full feature magnitude, not just direction.
    eps      : ridge added to Σ before log-det for numerical stability.
               A value of 1e-4 works well for ViT features (scale ~O(1)).

    Returns
    -------
    Scalar tensor — the negative log-determinant (lower = more volume).
    Gradient flows through features via the Cholesky factorisation.
    """
    N, D = features.shape
    if N < 2:
        return features.new_zeros(1).squeeze()

    mu    = features.mean(dim=0, keepdim=True)          # (1, D)
    Z     = features - mu                               # (N, D), centred

    if N <= D:
        # Work in sample space: gram matrix is (N, N), cheaper for N << D
        G   = Z @ Z.T / (N - 1)                        # (N, N)
        G   = G + eps * torch.eye(N, device=features.device, dtype=features.dtype)
        L   = torch.linalg.cholesky(G)
        # log det(G) = 2 * sum(log diag(L))
        log_det = 2.0 * torch.log(L.diagonal()).sum()
    else:
        # Full covariance: (D, D)
        Sigma = (Z.T @ Z) / (N - 1)                    # (D, D)
        Sigma = Sigma + eps * torch.eye(D, device=features.device, dtype=features.dtype)
        L     = torch.linalg.cholesky(Sigma)
        log_det = 2.0 * torch.log(L.diagonal()).sum()

    return -log_det                                     # negate: minimise → maximise volume


def build_old_subspace_projector(
    static_hulls: Dict,
    device: torch.device,
    variance_threshold: float = 0.95,
) -> torch.Tensor:
    """
    Build P_old = U_old U_old^T — the orthogonal projector onto the subspace
    spanned by all stored old-class extreme rays.

    SVD on the stacked ray matrix (N_rays_total, D) gives right singular
    vectors (rows of Vt) that form an orthonormal basis for the row space.
    We keep only the top-k vectors that cover variance_threshold of the total
    squared-singular-value energy, trading off completeness for efficiency.

    Returns a (D, D) float32 tensor on device.  Computed once per stage; the
    per-step cost of L_route is then a single (N, D) x (D, D) matmul.
    """
    all_rays = np.vstack([h.extreme_rays_ for h in static_hulls.values()])  # (N, D)
    _, S, Vt = np.linalg.svd(all_rays, full_matrices=False)                 # Vt: (r, D)

    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(cumvar, variance_threshold)) + 1
    k = min(k, len(S))

    U_old   = Vt[:k].T          # (D, k) — orthonormal column basis
    P_old   = U_old @ U_old.T   # (D, D) — idempotent symmetric projector
    return torch.tensor(P_old, dtype=torch.float32, device=device)


def orthogonal_conic_routing_loss(
    backbone:            nn.Module,
    old_backbone:        nn.Module,
    P_old:               torch.Tensor,
    extreme_ray_images:  Dict,
    new_features:        torch.Tensor,
    device:              torch.device,
    n_rays_sample:       int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonal Conic Routing Loss.

    L_lock  — Angular Rigidity.
    Penalises only the *angular* shift of stored old extreme rays, ignoring
    magnitude changes that are geometrically harmless for conic membership.
    Cost is cosine distance between old-backbone and new-backbone outputs on
    the same extreme-ray images:

        L_lock = mean_{x in X_old} (1 - cos(x_new, x_old))

    L_route — Subspace Routing.
    Forces new-class features to grow in dimensions orthogonal to the old
    cones by penalising the squared norm of their projection onto P_old:

        L_route = mean_{z in Z_new} ||P_old z||^2

    Since P_old is idempotent, ||P_old z||^2 = z^T P_old z.

    Parameters
    ----------
    n_rays_sample : max extreme-ray images sampled per step for L_lock.
                    With 150 rays x 90 classes = 13,500 candidates, a budget
                    of 128 keeps the extra forward passes manageable.

    Returns
    -------
    (l_lock, l_route) — scalar tensors, both differentiable w.r.t. backbone
    """
    # ── L_lock: Angular Rigidity ──────────────────────────────────────────────
    all_ray_imgs = [img for imgs in extreme_ray_images.values() for img in imgs]
    l_lock = new_features.new_zeros(())

    if all_ray_imgs and old_backbone is not None:
        n_sample = min(n_rays_sample, len(all_ray_imgs))
        perm     = torch.randperm(len(all_ray_imgs))[:n_sample]
        sampled  = torch.stack([all_ray_imgs[i] for i in perm]).to(device)

        with torch.no_grad():
            x_old = F.normalize(old_backbone(sampled), p=2, dim=1)

        backbone.eval()
        x_new = F.normalize(backbone(sampled), p=2, dim=1)
        backbone.train()

        l_lock = (1.0 - (x_new * x_old).sum(dim=1)).mean()

    # ── L_route: Subspace Routing ─────────────────────────────────────────────
    z       = F.normalize(new_features, p=2, dim=1)   # (N, D)
    proj    = z @ P_old                                # (N, D) — P_old symmetric
    l_route = proj.pow(2).sum(dim=1).mean()

    return l_lock, l_route


def get_incremental_dataloaders(dataset_name="CIFAR100", classes_per_stage=10, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name == "CIFAR100":
        train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        total_classes = 100
    else:
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        total_classes = 10
        
    num_stages = total_classes // classes_per_stage
    stages = []
    
    train_targets = torch.tensor(train_ds.targets)
    test_targets = torch.tensor(test_ds.targets)
    
    for i in range(num_stages):
        start_class = i * classes_per_stage
        end_class = start_class + classes_per_stage
        
        idx_train = torch.where((train_targets >= start_class) & (train_targets < end_class))[0]
        idx_test = torch.where((test_targets >= start_class) & (test_targets < end_class))[0]
        
        stages.append({
            "stage_id": i,
            "classes": list(range(start_class, end_class)),
            "train_loader": DataLoader(Subset(train_ds, idx_train), batch_size=batch_size, shuffle=True),
            "test_loader": DataLoader(Subset(test_ds, idx_test), batch_size=batch_size, shuffle=False)
        })
        
    return stages, total_classes, len(train_ds)

def evaluate_stage(backbone, head, old_backbone, test_loader, device):
    """Evaluates accuracy and average feature drift on a given test loader."""
    backbone.eval()
    head.eval()
    correct, total = 0, 0
    total_drift, drift_batches = 0.0, 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            features = backbone(imgs)

            if isinstance(head, (FixedConicHead, IncrementalConicHead)):
                logits = head(features, labels=None)  # Eval mode: raw cosines
            else:
                logits = head(features)
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Calculate Feature Drift if we have an older model to compare against
            if old_backbone is not None:
                old_features = old_backbone(imgs)
                f_new = F.normalize(features, p=2, dim=1)
                f_old = F.normalize(old_features, p=2, dim=1)
                # Euclidean distance between normalized vectors
                drift = F.pairwise_distance(f_new, f_old, p=2).mean().item()
                total_drift += drift
                drift_batches += 1
                
    acc = correct / total if total > 0 else 0.0
    avg_drift = total_drift / drift_batches if drift_batches > 0 else 0.0
    return acc, avg_drift


def train_incremental_pipeline_lwf(
    dataset_name="CIFAR100", 
    classes_per_stage=10, 
    epochs_per_stage=10, 
    alpha=0.1,  
    distill_weight=10.0, 
    min_delta=0.001,
    patience=3, 
    batch_size=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stages, total_classes, total_data_size = get_incremental_dataloaders(dataset_name, classes_per_stage, batch_size)
    # ==========================================
    #           DATASET DEBUGGING 
    # ==========================================
    print("\n" + "="*50)
    print("   [Debug] Data Distribution Per Stage")
    print("="*50)

    for stage_idx, stage in enumerate(stages):
        train_loader = stage["train_loader"]
        test_loader = stage["test_loader"]
        stage_classes = stage["classes"]
        
        # Tally up the exact number of examples per class in the train set
        train_counts = Counter()
        for _, labels in train_loader:
            train_counts.update(labels.tolist())
            
        # Tally up the exact number of examples per class in the test set
        test_counts = Counter()
        for _, labels in test_loader:
            test_counts.update(labels.tolist())
            
        print(f"\n  --- Stage {stage_idx} | Classes: {stage_classes} ---")
        
        # Print the breakdown per class
        for cls in sorted(stage_classes):
            n_train = train_counts.get(cls, 0)
            n_test = test_counts.get(cls, 0)
            print(f"    Class {cls:>3} -> Train: {n_train:>4} examples | Test: {n_test:>4} examples")

    print("\n" + "="*50 + "\n")
        
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
    feat_dim = backbone.num_features
    head = FixedConicHead(in_features=feat_dim, total_classes=total_classes, m=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    old_backbone = None
    
    # Trackers for Continual Learning Metrics
    num_stages = len(stages)
    acc_matrix = np.zeros((num_stages, num_stages))
    drift_matrix = np.zeros((num_stages, num_stages))

    for param in backbone.patch_embed.parameters():
        param.requires_grad = False
        
    for i in range(10): # Freeze first 10 blocks
        for param in backbone.blocks[i].parameters():
            param.requires_grad = False
    
    
    for current_stage_idx, stage in enumerate(stages):
        print(f"\n{'='*50}")
        print(f"=== Training Stage {current_stage_idx} (Classes {stage['classes'][0]} to {stage['classes'][-1]}) ===")
        print(f"{'='*50}")
        
        train_loader = stage["train_loader"]
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-4)
        
        # Early stopping trackers
        best_loss = float('inf')
        patience_counter = 0
        
        # Set tqdm total to max_epochs, but we will break out of it early
        epoch_iterator = tqdm(range(epochs_per_stage), desc=f"Stage {current_stage_idx}", unit="ep")
        
        for epoch in epoch_iterator:
            backbone.train()
            head.train()
            total_loss, total_cls, total_dist = 0, 0, 0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                features = backbone(imgs)
                logits = head(features, labels)
                loss_cls = criterion(logits, labels)
                
                loss_distill = torch.tensor(0.0).to(device)
                if old_backbone is not None:
                    with torch.no_grad():
                        old_features = old_backbone(imgs)
                    f_new = F.normalize(features, p=2, dim=1)
                    f_old = F.normalize(old_features, p=2, dim=1)
                    drift = F.pairwise_distance(f_new, f_old, p=2)
                    loss_distill = torch.clamp(drift - alpha, min=0.0).mean()

                loss = loss_cls + (distill_weight * loss_distill)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_cls += loss_cls.item()
                total_dist += loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0

            avg_loss = total_loss / len(train_loader)
            
            # Update the progress bar suffix
            epoch_iterator.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Cls": f"{total_cls/len(train_loader):.4f}",
                "Dist": f"{total_dist/len(train_loader):.5f}"
            })

            # --- Early Stopping Check ---
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
                # If you wanted to save the absolute best weights, you would do `torch.save` here
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                tqdm.write(f"  -> Early stopping triggered at epoch {epoch+1} (Best Loss: {best_loss:.4f})")
                epoch_iterator.close() # Close the progress bar cleanly
                break
        
        print(f"\n--- Evaluation after Stage {current_stage_idx} ---")
        
        # Evaluate on all stages seen so far
        for eval_stage_idx in range(current_stage_idx + 1):
            test_loader = stages[eval_stage_idx]["test_loader"]
            acc, avg_drift = evaluate_stage(backbone, head, old_backbone, test_loader, device)
            
            acc_matrix[current_stage_idx, eval_stage_idx] = acc
            drift_matrix[current_stage_idx, eval_stage_idx] = avg_drift
            
            stage_type = "NEW" if eval_stage_idx == current_stage_idx else "OLD"
            if stage_type == "OLD":
                print(f"  [Task {eval_stage_idx} - {stage_type}] Acc: {acc:.2%} | Feature Drift from prev: {avg_drift:.4f}")
            else: 
                print(f"  [Task {eval_stage_idx} - {stage_type}] Acc: {acc:.2%}")
            
            
        # Compute Metrics
        current_avg_acc = np.mean(acc_matrix[current_stage_idx, :current_stage_idx + 1])
        
        # Calculate Forgetting (Max accuracy achieved in the past minus current accuracy)
        forgetting = 0.0
        if current_stage_idx > 0:
            forgetting_per_task = []
            for j in range(current_stage_idx):
                best_past_acc = np.max(acc_matrix[:current_stage_idx, j])
                current_acc = acc_matrix[current_stage_idx, j]
                forgetting_per_task.append(best_past_acc - current_acc)
            forgetting = np.mean(forgetting_per_task)
            
        print(f"\n  -> Average Accuracy (Tasks 0-{current_stage_idx}): {current_avg_acc:.2%}")
        if current_stage_idx > 0:
            print(f"  -> Average Forgetting: {forgetting:.2%}")

        # Update teacher model for the next stage
        old_backbone = copy.deepcopy(backbone)
        old_backbone.eval()
        for param in old_backbone.parameters():
            param.requires_grad = False
            
    print("\n=== Final CL Performance Matrix ===")
    print("Rows: Model state after Task i. Columns: Evaluated on Task j.")
    print(np.round(acc_matrix, 3))
    
    return backbone, head, acc_matrix

def evaluate_spa_conic_hulls(backbone, test_loader, memory_images, memory_labels, device, n_rays=50):
    """
    Evaluates accuracy by fitting your SPA ConicHulls to replay data
    and classifying test samples based on the highest hull score.
    """
    backbone.eval()
    
    # 1. Extract features from memory to build the hulls
    feature_dict = {}
    print(f"  -> Extracting features for {len(memory_images)} replay images...")
    with torch.no_grad():
        # Process in batches to avoid VRAM overflow
        mem_tensor = torch.stack(memory_images).to(device)
        mem_features = backbone(mem_tensor).cpu().numpy()
        
        for i, lbl in enumerate(memory_labels):
            lbl_str = str(lbl)
            if lbl_str not in feature_dict:
                feature_dict[lbl_str] = []
            feature_dict[lbl_str].append(mem_features[i])
    
    feature_dict = {k: np.array(v) for k, v in feature_dict.items()}

    # 2. Fit one ConicHull per class using your provided builder
    # This uses SPA to find the extreme rays
    class_hulls = build_class_conic_hulls(feature_dict, n_rays=n_rays, use_pca=False)

    # 3. Classify test samples
    correct, total = 0, 0
    all_class_names = list(class_hulls.keys())
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing hulls"):
            test_feats = backbone(imgs.to(device)).cpu().numpy()
            
            # Scores matrix: (N_queries, N_classes)
            # Each entry is the conic angular similarity score
            scores = np.zeros((test_feats.shape[0], len(all_class_names)))
            
            for idx, name in enumerate(all_class_names):
                scores[:, idx] = class_hulls[name].score(test_feats)
            
            # Prediction is the class with the highest hull membership score
            best_idx = np.argmax(scores, axis=1)
            preds = np.array([int(all_class_names[i]) for i in best_idx])
            
            correct += (preds == labels.numpy()).sum()
            total += labels.size(0)
            
    return correct / total



def calculate_distillation_loss(distill_type, new_features, old_features, alpha=0.1, pointwise_weight=0.2, predictor=None, kl_divergence=True):
    """
    Modular distillation function.
    - 'basic': Uses your existing L2 drift + Relational MSE.
    - 'ssl': Uses Teacher-Student Predictive Cosine Similarity.
    """
    if distill_type == "ssl":
        # 1. Pass student features through the predictor
        predicted_features = predictor(new_features)
        
        # 2. Maximize cosine similarity (we do 1.0 - sim so perfect match = 0 loss)
        loss = F.mse_loss(predicted_features, old_features)
        return loss
        
    elif distill_type == "basic":
        # Your original pointwise + relational distillation
        f_new = F.normalize(new_features, p=2, dim=1)
        f_old = F.normalize(old_features, p=2, dim=1)
        
        sim_matrix_new = torch.mm(f_new, f_new.T) 
        sim_matrix_old = torch.mm(f_old, f_old.T)
        
        drift = F.pairwise_distance(f_new, f_old, p=2)
        loss_pointwise = torch.clamp(drift, min=0.0).mean()
        if not kl_divergence:
            loss_distill_relational = F.mse_loss(sim_matrix_new, sim_matrix_old)
        else: 
            temp = 0.05
            sim_matrix_new *= temp
            sim_matrix_old *= temp 
            # Convert to probabilities using Softmax
            p_new = F.log_softmax(sim_matrix_new, dim=1)
            p_old = F.softmax(sim_matrix_old, dim=1)
            loss_distill_relational = F.kl_div(p_new, p_old, reduction='batchmean') * (temp ** 2)
      
        return (pointwise_weight * loss_pointwise) + loss_distill_relational
        
    else:
        raise ValueError(f"Unknown distill_type: {distill_type}")


def train_incremental_pipeline_replay(
    dataset_name="CIFAR100",
    model_name="vit_tiny_patch16_224",
    classes_per_stage=10,
    epochs_per_stage=10,
    alpha=0.1,
    distill_weight=10.0,
    min_delta=0.001,
    patience=3,
    batch_size=128,
    reserved_space=True,
    conic_hull_margin=0.7,
    conic_hull_n_rays=150,
    learning_rate=1e-4,
    pointwise_distill_weight=0.2,
    memory_budget=0.04,                # 4% of the total dataset
    use_analytical_head_update=True,   # Analytically correct old class weights after drift
    drift_method="covariance",         # "procrustes" | "covariance"
    head_update_method="affine",             # "affine" | "ortho_projected"
    head_update_magnitude_preserving=False,  # decouple A^{-T} rotation from scale expansion
    use_log_barrier_distill=False,      # Replace L2 KD with log-barrier margin loss
    barrier_weight=1.0,                # λ in  L = L_cls + λ * L_barrier
    head_type="conic",                 # "conic" | "linear" | "mlp"  (ablation)
    mlp_hidden_dim=512,                # hidden width when head_type="mlp"
    mlp_dropout=0.0,                   # dropout probability when head_type="mlp"
    use_logdet_penalty=False,          # add log-det volume penalty to prevent collapse
    logdet_weight=1e-3,                # λ_vol in  L = L_cls + λ_vol * L_vol
    logdet_eps=1e-4,                   # ridge ε added to Σ for numerical stability
    track_superclass_confusion=False,  # run superclass confusion analysis after each stage
    superclass_confusion_top_n=15,     # how many confused pairs to print per stage
    superclass_confusion_plot=False,   # render plots per stage (can be slow)
    use_ortho_routing=False,           # enable Orthogonal Conic Routing Loss
    lambda_lock=1.0,                   # weight for L_lock (angular rigidity)
    lambda_route=1.0,                  # weight for L_route (subspace routing)
    n_lock_rays_sample=128,            # extreme-ray images sampled per step for L_lock
    ortho_svd_variance=0.95,           # fraction of variance kept in P_old SVD truncation
    use_conic_hull_rotation=False,     # refine A via polar decomp + angular width scaling
    hull_rotation_boundary_margin=0.05,  # centripetal safety margin for cone scaling
    rotate_static_hulls=False,         # apply A to frozen static hull extreme rays each stage
    rotate_dynamic_hulls=False,        # apply A to freshly re-fit dynamic hull extreme rays
    blocks_freeze=12,
    lora_rank=0,                       # > 0 → inject LoRA into ViT attention layers
    lora_alpha=1.0,                    # LoRA scaling factor α (effective scale = α/rank)
    lora_target_modules=None,          # layer suffixes to adapt, e.g. ["attn.qkv", "attn.proj"]
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stages, total_classes, total_data_size = get_incremental_dataloaders(dataset_name, classes_per_stage, batch_size)

    # # ==========================================
    # #           DATASET DEBUGGING 
    # # ==========================================
    # print("\n" + "="*50)
    # print("   [Debug] Data Distribution Per Stage")
    # print("="*50)

    # for stage_idx, stage in enumerate(stages):
    #     train_loader = stage["train_loader"]
    #     test_loader = stage["test_loader"]
    #     stage_classes = stage["classes"]
        
    #     # Tally up the exact number of examples per class in the train set
    #     train_counts = Counter()
    #     for _, labels in train_loader:
    #         train_counts.update(labels.tolist())
            
    #     # Tally up the exact number of examples per class in the test set
    #     test_counts = Counter()
    #     for _, labels in test_loader:
    #         test_counts.update(labels.tolist())
            
    #     print(f"\n  --- Stage {stage_idx} | Classes: {stage_classes} ---")
        
    #     # Print the breakdown per class
    #     for cls in sorted(stage_classes):
    #         n_train = train_counts.get(cls, 0)
    #         n_test = test_counts.get(cls, 0)
    #         print(f"    Class {cls:>3} -> Train: {n_train:>4} examples | Test: {n_test:>4} examples")

    # print("\n" + "="*50 + "\n")
    
    backbone = load_backbone(
        model_name, pretrained=True, num_classes=0, device=str(device),
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
    )
    feat_dim = backbone.num_features

    if head_type == "conic":
        if reserved_space:
            head = FixedConicHead(in_features=feat_dim, total_classes=total_classes, m=conic_hull_margin).to(device)
        else:
            head = IncrementalConicHead(in_features=feat_dim, m=conic_hull_margin).to(device)
    elif head_type == "linear":
        head = IncrementalLinearHead(in_features=feat_dim).to(device)
    elif head_type == "mlp":
        head = IncrementalMLPHead(in_features=feat_dim, hidden_dim=mlp_hidden_dim, dropout=mlp_dropout).to(device)
    else:
        raise ValueError(f"Unknown head_type '{head_type}'. Choose 'conic', 'linear', or 'mlp'.")

    print(f"  Head: {head.__class__.__name__}")

    criterion = nn.CrossEntropyLoss()
    old_backbone = None
    P_old: Optional[torch.Tensor] = None   # built after each stage for ortho routing

    # Trackers for Continual Learning Metrics
    num_stages = len(stages)
    acc_matrix = np.zeros((num_stages, num_stages))
    acc_matrix_conic_hull_static = np.zeros((num_stages, num_stages)) # NEW: Track conic hull accuracy separately
    acc_matrix_conic_hull_dynamic = np.zeros((num_stages, num_stages)) # NEW: Track dynamic hull accuracy separately
    drift_matrix = np.zeros((num_stages, num_stages))

    for name, param in backbone.patch_embed.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    for i in range(blocks_freeze): # Freeze first 10 blocks
        for name, param in backbone.blocks[i].named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    
    
    memory_limit = memory_budget * total_data_size
    replay_buffer = ReplayBuffer(max_total_size=memory_limit, max_classes=total_classes)
    sampling_type = "balanced" # or "uniform"
    

    hull_mgr = HullManager(n_rays=conic_hull_n_rays, use_pca=False)

    # Per-stage superclass confusion history: list of dicts returned by
    # analyze_superclass_confusion, one entry per stage.
    superclass_confusion_history: List[Dict] = []

    # Analytical head-update state.  _prev_stats holds per-class statistics
    # computed with the backbone *before* the current stage's training so they
    # can be compared against post-training statistics to estimate feature drift.
    _prev_stats: Optional[Dict] = None

    for current_stage_idx, stage in enumerate(stages):
        print(f"\n{'='*50}")
        print(f"=== Training Stage {current_stage_idx} (Classes {stage['classes'][0]} to {stage['classes'][-1]}) ===")
        print(f"{'='*50}")
        num_new_classes = len(stage["classes"])
        if hasattr(head, "add_classes"):
            head.add_classes(num_new_classes, device)
        
        train_loader = stage["train_loader"]
        # Collect only parameters that require gradients.
        # When LoRA is active the original linear weights inside LoRALinear are
        # frozen; block-level freezing still suppresses adapters in frozen blocks.
        trainable_params = [p for p in backbone.parameters() if p.requires_grad]
        if isinstance(head, (IncrementalLinearHead, IncrementalMLPHead)):
            trainable_params += list(head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        
        
        # Early stopping trackers
        best_loss = float('inf')
        patience_counter = 0
        
        # Set tqdm total to max_epochs, but we will break out of it early
        epoch_iterator = tqdm(range(epochs_per_stage), desc=f"Stage {current_stage_idx}", unit="ep")
        
        for epoch in epoch_iterator:
            backbone.train()
            head.train()
            total_loss, total_cls, total_dist, total_barrier, total_logdet = 0, 0, 0, 0, 0
            total_lock, total_route = 0, 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                features = backbone(imgs)

                # Conic heads accept labels for ArcFace margin; others do not
                if isinstance(head, (FixedConicHead, IncrementalConicHead)):
                    logits = head(features, labels)
                else:
                    logits = head(features)

                loss_cls = criterion(logits, labels)

                # ── Log-Det Volume Penalty ────────────────────────────────────
                loss_logdet = torch.tensor(0.0).to(device)
                if use_logdet_penalty:
                    raw_logdet  = logdet_volume_penalty(features, eps=logdet_eps)
                    loss_logdet = logdet_weight * raw_logdet
                    total_logdet += raw_logdet.item()

                loss_distill = torch.tensor(0.0).to(device)

                if old_backbone is not None and len(replay_buffer.all_classes) > 0:
                    # Sample a balanced batch from the replay buffer (now includes labels)
                    mem_imgs, mem_embs_stored, mem_lbls = replay_buffer.sample(
                        batch_size, strategy=sampling_type
                    )
                    mem_imgs = mem_imgs.to(device)
                    mem_lbls = mem_lbls.to(device)

                    # Teacher features (frozen old backbone)
                    with torch.no_grad():
                        old_mem_features = old_backbone(mem_imgs)

                    # Student features (backbone being trained)
                    # eval() removes dropout stochasticity for a fair comparison
                    backbone.eval()
                    new_mem_features = backbone(mem_imgs)
                    backbone.train()

                    if use_log_barrier_distill:
                        # ── Log-Barrier Margin-Preservation Loss ─────────────────
                        # Penalises any shrinkage of each old feature's angular
                        # margin to every competing class boundary.  Zero cost for
                        # drift that moves features *deeper* into their own cone.
                        num_seen = len(replay_buffer.all_classes)
                        raw_barrier = log_barrier_margin_loss(
                            x_new=new_mem_features,
                            x_old=old_mem_features,
                            labels=mem_lbls,
                            head_weight=head.weight,
                            num_seen_classes=num_seen,
                        )
                        loss_distill = barrier_weight * raw_barrier
                        total_barrier += raw_barrier.item()
                    else:
                        # ── Legacy isotropic L2 + relational distillation ─────────
                        loss_distill = distill_weight * (
                            calculate_distillation_loss(
                                distill_type="basic",
                                new_features=new_mem_features,
                                old_features=mem_embs_stored.to(device),
                                alpha=alpha,
                                pointwise_weight=pointwise_distill_weight,
                            )
                            + calculate_distillation_loss(
                                distill_type="basic",
                                new_features=new_mem_features,
                                old_features=old_mem_features,
                                alpha=alpha,
                                pointwise_weight=1-pointwise_distill_weight,
                            )
                        )

                # ── Orthogonal Conic Routing Loss ─────────────────────────────
                loss_lock  = features.new_zeros(())
                loss_route = features.new_zeros(())
                if use_ortho_routing and P_old is not None:
                    raw_lock, raw_route = orthogonal_conic_routing_loss(
                        backbone=backbone,
                        old_backbone=old_backbone,
                        P_old=P_old,
                        extreme_ray_images=hull_mgr.extreme_ray_images,
                        new_features=features,
                        device=device,
                        n_rays_sample=n_lock_rays_sample,
                    )
                    loss_lock  = lambda_lock  * raw_lock
                    loss_route = lambda_route * raw_route
                    total_lock  += raw_lock.item()
                    total_route += raw_route.item()

                loss = loss_cls + loss_distill + loss_logdet + loss_lock + loss_route

                loss.backward()
                optimizer.step()

                total_loss   += loss.item()
                total_cls    += loss_cls.item()
                total_dist   += loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0

            avg_loss    = total_loss    / len(train_loader)
            avg_cls     = total_cls     / len(train_loader)
            avg_dist    = total_dist    / len(train_loader)
            avg_barrier = total_barrier / len(train_loader)
            avg_logdet  = total_logdet  / len(train_loader)
            avg_lock    = total_lock    / len(train_loader)
            avg_route   = total_route   / len(train_loader)

            # Update the progress bar suffix
            postfix = {
                "Loss":   f"{avg_loss:.4f}",
                "Cls":    f"{avg_cls:.4f}",
                "Dist":   f"{avg_dist:.5f}",
            }
            if use_log_barrier_distill:
                postfix["Barrier"] = f"{avg_barrier:.5f}"
            if use_logdet_penalty:
                postfix["LogDet"] = f"{avg_logdet:.3f}"
            if use_ortho_routing:
                postfix["Lock"]  = f"{avg_lock:.5f}"
                postfix["Route"] = f"{avg_route:.5f}"
            epoch_iterator.set_postfix(postfix)

            # --- Early Stopping Check ---
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
                # If you wanted to save the absolute best weights, you would do `torch.save` here
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                tqdm.write(f"  -> Early stopping triggered at epoch {epoch+1} (Best Loss: {best_loss:.4f})")
                epoch_iterator.close() # Close the progress bar cleanly
                break
        
        # ── Analytical Classifier Head Correction ────────────────────────────────
        # After backbone training the feature space has drifted.  We estimate the
        # linear transformation A : x_old → x_new from per-class statistics
        # (stored before training) and correct old classifier weights analytically:
        #
        #   w_new = A^{-T} w_old
        #
        # This avoids the need to fine-tune the frozen conic head on replay data.
        _stage_A: Optional[np.ndarray] = None   # A persists for hull rotation after this block
        if use_analytical_head_update and _prev_stats is not None and len(_prev_stats) >= 2:
            print("  -> Analytical head correction: estimating feature-space drift...")
            old_class_ids = list(_prev_stats.keys())

            # Re-extract features for old classes with the *updated* backbone.
            # The replay buffer still contains only old-class images at this point
            # (new-class images are added in the next block), so this is exact.
            new_stats = _extract_class_stats_from_buffer(
                backbone, replay_buffer, old_class_ids, device, batch_size
            )

            if len(new_stats) >= 2:
                try:
                    A = estimate_affine_drift(_prev_stats, new_stats, method=drift_method)

                    if use_conic_hull_rotation:
                        A = rotate_conic_hulls(
                            _prev_stats, new_stats, A,
                            boundary_margin=hull_rotation_boundary_margin,
                        )
                        print(f"  -> Applied conic hull rotation (boundary_margin={hull_rotation_boundary_margin}).")

                    _stage_A = A  # expose A for hull rotation after this try block

                    if rotate_static_hulls:
                        rotate_hulls(hull_mgr.static_hulls, _prev_stats, new_stats, A)
                        print(f"  -> Rotated {len(hull_mgr.static_hulls)} static hull(s) into new feature space.")

                    if head_update_method == "ortho_projected":
                        # Compute new-class mean feature vectors for subspace projection.
                        # One forward pass through the current stage's training data.
                        backbone.eval()
                        new_mean_acc: Dict[int, np.ndarray] = {}
                        new_mean_cnt: Dict[int, int]        = {}
                        with torch.no_grad():
                            for imgs, labels in stage["train_loader"]:
                                feats = backbone(imgs.to(device)).cpu().numpy()
                                for feat, lbl in zip(feats, labels.tolist()):
                                    if lbl not in new_mean_acc:
                                        new_mean_acc[lbl] = feat.copy()
                                        new_mean_cnt[lbl] = 1
                                    else:
                                        new_mean_acc[lbl] += feat
                                        new_mean_cnt[lbl] += 1
                        new_class_means = np.stack([
                            new_mean_acc[c] / new_mean_cnt[c]
                            for c in sorted(new_mean_acc)
                        ])                                              # (n_new, D)
                        update_head_weights_orthogonal_projected(
                            head, A, list(new_stats.keys()),
                            new_class_means, device,
                            magnitude_preserving=head_update_magnitude_preserving,
                        )
                        print(
                            f"  -> Applied A^{{-T}} + P_perp correction to "
                            f"{len(new_stats)} old classes "
                            f"(new subspace dim={new_class_means.shape[0]})."
                        )
                    else:
                        update_head_weights_analytically(
                            head, A, list(new_stats.keys()), device,
                            renormalize=not head_update_magnitude_preserving,
                            magnitude_preserving=head_update_magnitude_preserving,
                        )
                        print(
                            f"  -> Applied A^{{-T}} weight correction to "
                            f"{len(new_stats)} old classes (method={drift_method})."
                        )
                except Exception as exc:
                    print(f"  [WARNING] Analytical head update skipped: {exc}")
            else:
                print(f"  [WARNING] Only {len(new_stats)} classes in buffer; skipping drift estimation.")

        # ── Fit hulls on full training data; commit only extreme rays to buffer ──────
        #
        # Step 1: Pass ALL new-class training images through the backbone and
        # collect them in memory.  We deliberately do NOT add them to the buffer
        # yet — the buffer will receive only the SPA-selected extreme rays.
        print(f"  -> Extracting new-class features for hull fitting...")
        backbone.eval()
        new_cls_images: Dict[int, list] = defaultdict(list)  # cls_id -> [img_cpu, ...]
        new_cls_feats:  Dict[int, list] = defaultdict(list)  # cls_id -> [feat_np, ...]

        with torch.no_grad():
            for imgs, labels in stage["train_loader"]:
                feats_np = backbone(imgs.to(device)).cpu().numpy()
                imgs_cpu = imgs.cpu()
                for img, feat, lbl in zip(imgs_cpu, feats_np, labels.tolist()):
                    new_cls_images[lbl].append(img)
                    new_cls_feats[lbl].append(feat)

        # Step 2: Fit static hulls using ALL training data for maximum hull quality.
        new_feature_dict = {
            str(cls_id): np.stack(feats_list)
            for cls_id, feats_list in new_cls_feats.items()
        }
        hull_mgr.fit_new_classes(new_feature_dict)

        # Step 3: Write ONLY the SPA-selected extreme ray images into the buffer.
        # Old-class entries stay untouched; each new class gets exactly n_rays slots.
        print(f"  -> Populating replay buffer with extreme ray images only...")
        for cls_id, feats_list in new_cls_feats.items():
            cls_str = str(cls_id)
            if cls_str not in hull_mgr.static_hulls:
                continue
            hull    = hull_mgr.static_hulls[cls_str]
            # SPA returns rays in decreasing order of extremeness; truncate to the
            # per-class memory budget so the most extreme rays are kept first.
           
            indices = hull.extreme_rays_index[:replay_buffer.per_class_cap]

            if cls_id not in replay_buffer.buffer:
                replay_buffer.all_classes.append(cls_id)

            ray_tuples = [
                (new_cls_images[cls_id][i],
                 torch.from_numpy(feats_list[i]),
                 torch.tensor(cls_id))
                for i in indices if i < len(new_cls_images[cls_id])
            ]
            replay_buffer.buffer[cls_id] = ray_tuples

            if use_ortho_routing:
                hull_mgr.extreme_ray_images[cls_str] = [
                    new_cls_images[cls_id][i]
                    for i in indices if i < len(new_cls_images[cls_id])
                ]

        print("\n  [Debug] Replay Buffer (extreme rays only):")
        total_samples = sum(len(v) for v in replay_buffer.buffer.values())
        print(f"  [Debug] Total: {total_samples} rays  "
              f"({len(replay_buffer.all_classes)} classes)")
        class_counts = [f"Class {c}: {len(replay_buffer.buffer[c])}"
                        for c in sorted(replay_buffer.all_classes)]
        for i in range(0, len(class_counts), 5):
            print("    " + ", ".join(class_counts[i:i+5]))
        print("=" * 44 + "\n")

        # Step 4: Re-extract features from the buffer (extreme rays, current backbone)
        # for dynamic hull fitting and drift-estimation statistics.
        feature_dict = {}
        with torch.no_grad():
            for cls_id in replay_buffer.all_classes:
                class_imgs = [item[0] for item in replay_buffer.buffer[cls_id]]
                if not class_imgs:
                    continue
                feats = backbone(torch.stack(class_imgs).to(device)).cpu().numpy()
                feature_dict[str(cls_id)] = feats

        # ── Snapshot per-class statistics for the *next* stage's drift estimation ─
        if use_analytical_head_update:
            _prev_stats = {}
            for cls_id_str, feats in feature_dict.items():
                cls_id = int(cls_id_str)
                n, d   = feats.shape
                mean   = feats.mean(axis=0)
                cov    = np.cov(feats.T, ddof=1) if n > 1 else np.zeros((d, d))
                _prev_stats[cls_id] = {"mean": mean, "cov": cov, "n": n}
            print(f"  -> Stored statistics for {len(_prev_stats)} classes "
                  f"(used for drift estimation in next stage).")

        # Step 5: Fit dynamic hulls from the current-backbone extreme-ray features.
        dynamic_hulls = hull_mgr.get_dynamic_hulls(feature_dict)

        # if rotate_dynamic_hulls and _stage_A is not None:
            
        #     rotate_hulls(dynamic_hulls, _prev_stats)
        #     print(f"  -> Rotated {len(dynamic_hulls)} dynamic hull(s) into new feature space.")

        # ── Orthogonal Conic Routing: rebuild projector ───────────────────────────
        if use_ortho_routing:
            P_old = build_old_subspace_projector(
                hull_mgr.static_hulls, device, ortho_svd_variance
            )
            n_total_rays = sum(len(v) for v in hull_mgr.extreme_ray_images.values())
            print(f"  -> [OrthoRouting] P_old built from {len(hull_mgr.static_hulls)} classes, "
                  f"{n_total_rays} extreme-ray images stored.")

        print(f"\n--- Evaluation after Stage {current_stage_idx} ---")
        
        # Evaluate on all stages seen so far
        for eval_stage_idx in range(current_stage_idx + 1):
            test_loader = stages[eval_stage_idx]["test_loader"]
            acc, avg_drift = evaluate_stage(backbone, head, old_backbone, test_loader, device)
            
            acc_dynamic_hull = hull_mgr.evaluate(backbone, test_loader, dynamic_hulls, device)
            acc_static_hull = hull_mgr.evaluate(backbone, test_loader, hull_mgr.static_hulls, device)
            
            acc_matrix[current_stage_idx, eval_stage_idx] = acc
            drift_matrix[current_stage_idx, eval_stage_idx] = avg_drift
            acc_matrix_conic_hull_static[current_stage_idx, eval_stage_idx] = acc_static_hull
            acc_matrix_conic_hull_dynamic[current_stage_idx, eval_stage_idx] = acc_dynamic_hull

            stage_type = "NEW" if eval_stage_idx == current_stage_idx else "OLD"
            if stage_type == "OLD":
                print(f"  [Task {eval_stage_idx} - {stage_type}] Acc: {acc:.2%} | Feature Drift from prev: {avg_drift:.4f} | Static Hull Replay Acc: {acc_static_hull:.2%} | Dynamic Hull Replay Acc: {acc_dynamic_hull:.2%}")
            else: 
                print(f"  [Task {eval_stage_idx} - {stage_type}] Acc: {acc:.2%} | Static Hull Replay Acc: {acc_static_hull:.2%} | Dynamic Hull Replay Acc: {acc_dynamic_hull:.2%}")
        
        # Compute Metrics
        current_avg_acc = np.mean(acc_matrix[current_stage_idx, :current_stage_idx + 1])
        current_avg_acc_hull_static = np.mean(acc_matrix_conic_hull_static[current_stage_idx, :current_stage_idx + 1])
        current_avg_acc_hull_dynamic = np.mean(acc_matrix_conic_hull_dynamic[current_stage_idx, :current_stage_idx + 1])

        # Calculate Forgetting (Max accuracy achieved in the past minus current accuracy)
        forgetting = 0.0
        forgetting_static = 0.0
        forgetting_dynamic = 0.0
        if current_stage_idx > 0:
            forgetting_per_task         = []
            forgetting_per_task_static  = []
            forgetting_per_task_dynamic = []
            for j in range(current_stage_idx):
                forgetting_per_task.append(
                    np.max(acc_matrix[:current_stage_idx, j])
                    - acc_matrix[current_stage_idx, j]
                )
                forgetting_per_task_static.append(
                    np.max(acc_matrix_conic_hull_static[:current_stage_idx, j])
                    - acc_matrix_conic_hull_static[current_stage_idx, j]
                )
                forgetting_per_task_dynamic.append(
                    np.max(acc_matrix_conic_hull_dynamic[:current_stage_idx, j])
                    - acc_matrix_conic_hull_dynamic[current_stage_idx, j]
                )
            forgetting         = np.mean(forgetting_per_task)
            forgetting_static  = np.mean(forgetting_per_task_static)
            forgetting_dynamic = np.mean(forgetting_per_task_dynamic)

        print(f"\n  -> Average Accuracy (Tasks 0-{current_stage_idx}): {current_avg_acc:.2%}")
        print(f"  -> Average Static Hull Accuracy (Tasks 0-{current_stage_idx}): {current_avg_acc_hull_static:.2%}")
        print(f"  -> Average Dynamic Hull Accuracy (Tasks 0-{current_stage_idx}): {current_avg_acc_hull_dynamic:.2%}")

        if current_stage_idx > 0:
            print(f"  -> Average Forgetting (Linear):       {forgetting:.2%}")
            print(f"  -> Average Forgetting (Static Hull):  {forgetting_static:.2%}")
            print(f"  -> Average Forgetting (Dynamic Hull): {forgetting_dynamic:.2%}")

        # ── Per-stage superclass confusion analysis ───────────────────────────
        if track_superclass_confusion:
            from analysis import analyze_superclass_confusion
            from torch.utils.data import ConcatDataset, DataLoader as _DL

            # Build a combined loader covering all classes seen so far
            seen_loaders = [stages[j]["test_loader"] for j in range(current_stage_idx + 1)]
            combined_dataset = ConcatDataset([dl.dataset for dl in seen_loaders])
            combined_loader  = _DL(
                combined_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=getattr(seen_loaders[0], "num_workers", 0),
            )

            seen_ids = []
            for j in range(current_stage_idx + 1):
                seen_ids.extend(stages[j]["classes"])

            print(f"\n  [Superclass Confusion] Stage {current_stage_idx} "
                  f"— {len(seen_ids)} classes seen so far")
            # Pass static hulls so the analysis runs for both the head classifier
            # and the conic hull classifier side-by-side.
            confusion_result = analyze_superclass_confusion(
                backbone=backbone,
                head=head,
                test_loader=combined_loader,
                device=device,
                seen_class_ids=seen_ids,
                top_n_pairs=superclass_confusion_top_n,
                plot=superclass_confusion_plot,
                hulls=hull_mgr.static_hulls,
            )
            confusion_result["stage"] = current_stage_idx
            superclass_confusion_history.append(confusion_result)

        # Update teacher model for the next stage
        old_backbone = copy.deepcopy(backbone)
        old_backbone.eval()
        for param in old_backbone.parameters():
            param.requires_grad = False
            
    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n=== Final CL Performance Matrix ===")
    print("Rows: Model state after Task i. Columns: Evaluated on Task j.")
    print(np.round(acc_matrix, 3))
    print("\n=== Final CL Conic Hull Performance Matrix (Static) ===")
    print(np.round(acc_matrix_conic_hull_static, 3))
    print("\n=== Final CL Conic Hull Performance Matrix (Dynamic) ===")
    print(np.round(acc_matrix_conic_hull_dynamic, 3))

    # Compute final average forgetting across all tasks (last row only)
    num_stages = len(acc_matrix)
    def _avg_forgetting(mat):
        rows = []
        for j in range(num_stages - 1):
            best = np.max(mat[:num_stages - 1, j])
            rows.append(best - mat[num_stages - 1, j])
        return float(np.mean(rows)) if rows else 0.0

    final_forgetting         = _avg_forgetting(acc_matrix)
    final_forgetting_static  = _avg_forgetting(acc_matrix_conic_hull_static)
    final_forgetting_dynamic = _avg_forgetting(acc_matrix_conic_hull_dynamic)

    print(f"\n=== Final Average Forgetting ===")
    print(f"  Linear Head:   {final_forgetting:.2%}")
    print(f"  Static Hull:   {final_forgetting_static:.2%}")
    print(f"  Dynamic Hull:  {final_forgetting_dynamic:.2%}")

    return (
        backbone, head,
        acc_matrix, acc_matrix_conic_hull_static, acc_matrix_conic_hull_dynamic,
        drift_matrix,
        final_forgetting, final_forgetting_static, final_forgetting_dynamic,
        superclass_confusion_history,
    )


def train_incremental_pipeline_benchmark(
    dataset_name="CIFAR100", 
    classes_per_stage=10, 
    epochs_per_stage=10, 
    alpha=0.1,  
    distill_weight=10.0, 
    min_delta=0.001,
    patience=3, 
    batch_size=128,
    memory_size_per_stage=200
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stages, total_classes = get_incremental_dataloaders(dataset_name, classes_per_stage, batch_size)
    
    # 1. Standard Backbone + Standard Linear Head
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
    feat_dim = backbone.num_features
    head = IncrementalLinearHead(in_features=feat_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    old_backbone = None
    
    num_stages = len(stages)
    acc_matrix = np.zeros((num_stages, num_stages))
    acc_matrix_conic_hull = np.zeros((num_stages, num_stages)) # NEW: Track conic hull accuracy separately  
    memory_images = []
    memory_labels = []  # Track labels for replay samples for conic hull evaluation

    # Freeze Backbone layers same as Conic to be fair
    for param in backbone.patch_embed.parameters(): param.requires_grad = False
    for i in range(10): 
        for param in backbone.blocks[i].parameters(): param.requires_grad = False
    
    for current_stage_idx, stage in enumerate(stages):
        print(f"\n{'='*50}\n=== BENCHMARK: Stage {current_stage_idx} (MLP Head) ===\n{'='*50}")
        
        # Expand the Linear Layer for new classes
        head.add_classes(len(stage["classes"]), device)
        head.to(device)
        
        train_loader = stage["train_loader"]
        # Optimizer must now include BOTH backbone and the trainable head
        optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=1e-4)
        
        best_loss = float('inf')
        patience_counter = 0
        epoch_iterator = tqdm(range(epochs_per_stage), desc=f"Stage {current_stage_idx}", unit="ep")
        
        for epoch in epoch_iterator:
            backbone.train()
            head.train()
            total_loss = 0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                features = backbone(imgs)
                logits = head(features) # Standard Linear Logits
                loss_cls = criterion(logits, labels)
                
                # Replay Distillation
                loss_distill = torch.tensor(0.0).to(device)
                if old_backbone is not None and len(memory_images) > 0:
                    mem_idx = torch.randint(0, len(memory_images), (imgs.size(0),))
                    mem_batch = torch.stack([memory_images[i] for i in mem_idx]).to(device)
                    
                    with torch.no_grad():
                        old_mem_features = old_backbone(mem_batch)
                    new_mem_features = backbone(mem_batch)
                    
                    # Same L2 Drift Distillation for fairness
                    drift = F.pairwise_distance(F.normalize(new_mem_features), F.normalize(old_mem_features))
                    loss_distill = torch.clamp(drift - alpha, min=0.0).mean()

                loss = loss_cls + (distill_weight * loss_distill)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            epoch_iterator.set_postfix({"Loss": f"{avg_loss:.4f}"})

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience: break
        
        # --- AFTER THE STAGE COMPLETES: Update Memory Buffer ---
        # Randomly select a subset of the current stage's data to save for the future
        print(f"  -> Updating Hull Replay Buffer...")
        saved_count = 0
        # Use the train_loader but extract labels too
        for imgs, labels in stage["train_loader"]:
            for i in range(imgs.size(0)):
                memory_images.append(imgs[i].cpu())
                memory_labels.append(labels[i].item()) # Save the label!
                saved_count += 1
                if saved_count >= memory_size_per_stage: break
            if saved_count >= memory_size_per_stage: break
            
        print(f"\n--- Evaluation after Stage {current_stage_idx} ---")
        
        # Evaluation
        for eval_stage_idx in range(current_stage_idx + 1):
            test_loader = stages[eval_stage_idx]["test_loader"]
            backbone.eval(); head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = head(backbone(imgs))
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
            
            acc = correct / total
            acc_matrix[current_stage_idx, eval_stage_idx] = acc
            
            # Conic Hull Evaluation
            acc_hull = evaluate_spa_conic_hulls(backbone, test_loader, memory_images, memory_labels, device)
            acc_matrix_conic_hull[current_stage_idx, eval_stage_idx] = acc_hull
            
            print(f"  [Task {eval_stage_idx}] Acc: {acc:.2%}, Hull Acc: {acc_hull:.2%}")
            
        old_backbone = copy.deepcopy(backbone)

    return acc_matrix, acc_matrix_conic_hull