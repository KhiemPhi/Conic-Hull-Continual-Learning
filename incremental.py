import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from conic_hull import (ConicHull, build_class_conic_hulls,
                        KernelConicHull, build_class_kernel_conic_hulls,
                        null_space_repulsion_loss, null_space_spread_loss,
                        covariance_collapse_penalty,
                        project_new_class_features, stack_hull_extreme_rays,
                        build_stage_plane_bases, project_to_stage_plane,
                        build_stage_plane_bases_from_features,
                        build_stage_plane_bases_blended,
                        build_orthogonal_stage_poles,
                        stage_confinement_loss,
                        stage_confinement_loss_labeled,
                        project_to_spherical_cap,
                        _score_projected_hull_batch,
                        rotate_kernel_hulls_into_null_space,
                        TaskOriginRegistry, ShiftedConicHull,
                        build_shifted_conic_hulls,
                        SphericalRegionRegistry, build_region_conic_hulls,
                        project_to_class_region,
                        LayerFeatureExtractor, LayeredConicHullClassifier,
                        get_layer_index_for_stage,
                        collaborative_nnls_scores)
import types
import timm

from features import get_default_transform
from cone_anchor import (
    ConeAnchorMemory, init_new_direction, expand_head_with_directions,
)
from geometric_loss import build_training_rays, geometric_cone_loss, cos_from_deg
from backbone import (load_backbone, get_lora_params, advance_lora_task, GradientProjector,
                      FeatureExpansionHead, FeatureExpansionBackbone)
import numpy as np
from sklearn.preprocessing import normalize
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection

def _draw_cone_panel(
    ax,
    class_ids: list,
    class_colour_map: dict,
    coords_3d: list,
    n_cone_lines: int,
    title: str,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    zlabel: str = "Dim 3",
) -> None:
    """Shared helper: draw cone fans + centroid labels onto a 3D axis."""
    origin = np.zeros(3)
    for cls_id, rays_3d in zip(class_ids, coords_3d):
        rays_3d = rays_3d / np.maximum(
            np.linalg.norm(rays_3d, axis=1, keepdims=True), 1e-12
        )
        idx = np.linspace(0, len(rays_3d) - 1, min(n_cone_lines, len(rays_3d)), dtype=int)
        colour = class_colour_map[cls_id]
        for ray in rays_3d[idx]:
            ax.plot([origin[0], ray[0]], [origin[1], ray[1]], [origin[2], ray[2]],
                    color=colour, alpha=0.55, linewidth=0.8)
        centroid = rays_3d.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        ax.scatter(*centroid, color=colour, s=30, zorder=5)
        ax.text(centroid[0], centroid[1], centroid[2], f" {cls_id}", fontsize=6,
                color=colour)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=9)


def visualize_extreme_rays_3d(
    static_hulls: dict,
    stage_idx: int,
    stage_class_map: dict,
    n_cone_lines: int = 40,
    kernel_hulls: dict = None,
    kernel_gamma: float = 1.0,
) -> None:
    """
    PCA-project extreme rays to 3D and draw each class as a cone, coloured by stage.

    When kernel_hulls is provided a two-panel figure is saved:
      Left  — linear PCA of the standard SPA extreme rays.
      Right — Kernel PCA (RBF, same γ) of the kernel SPA extreme rays.
    """
    from sklearn.decomposition import PCA, KernelPCA

    # ── collect rays ─────────────────────────────────────────────────────────
    all_rays, class_ids = [], []
    for cls_str, hull in static_hulls.items():
        if hull.extreme_rays_ is None or len(hull.extreme_rays_) == 0:
            continue
        all_rays.append(hull.extreme_rays_)
        class_ids.append(int(cls_str))

    if not all_rays:
        print("  [VisualizeRays] No extreme rays available yet — skipping.")
        return

    # ── colour by stage ───────────────────────────────────────────────────────
    all_stages   = sorted(set(stage_class_map.values()))
    n_stages     = max(len(all_stages), 1)
    stage_colours = plt.cm.tab10(np.linspace(0, 1, n_stages, endpoint=False))
    stage_colour_map = {s: stage_colours[i] for i, s in enumerate(all_stages)}
    class_colour_map = {
        c: stage_colour_map[stage_class_map.get(c, 0)] for c in class_ids
    }

    # ── linear PCA projection ─────────────────────────────────────────────────
    stacked   = np.vstack(all_rays)
    pca       = PCA(n_components=3)
    pca.fit(stacked)
    explained_lin = pca.explained_variance_ratio_.sum()
    lin_coords    = [pca.transform(r) for r in all_rays]

    # ── kernel PCA projection (only when kernel_hulls provided) ───────────────
    k_coords, explained_kern, k_all_rays, k_class_ids = None, None, None, None
    if kernel_hulls is not None:
        k_all_rays, k_class_ids = [], []
        for cls_str in static_hulls:
            hull = kernel_hulls.get(cls_str)
            if hull is None or hull.extreme_rays_ is None:
                continue
            k_all_rays.append(hull.extreme_rays_)
            k_class_ids.append(int(cls_str))

        if k_all_rays:
            k_stacked = np.vstack(k_all_rays)
            kpca = KernelPCA(n_components=3, kernel="rbf", gamma=kernel_gamma)
            kpca.fit(k_stacked)
            k_coords = [kpca.transform(r) for r in k_all_rays]
            lambdas = kpca.eigenvalues_
            explained_kern = lambdas[:3].sum() / (lambdas.sum() + 1e-12)

    # ── figure layout ─────────────────────────────────────────────────────────
    n_panels = 2 if k_coords is not None else 1
    fig = plt.figure(figsize=(9 * n_panels, 7))

    ax_lin = fig.add_subplot(1, n_panels, 1, projection="3d")
    _draw_cone_panel(
        ax_lin, class_ids, class_colour_map, lin_coords, n_cone_lines,
        title=f"Linear PCA — Stage {stage_idx}  ({len(class_ids)} cls, {explained_lin:.1%} var)",
        xlabel="PC1", ylabel="PC2", zlabel="PC3",
    )

    if k_coords is not None:
        ax_kern = fig.add_subplot(1, 2, 2, projection="3d")
        _draw_cone_panel(
            ax_kern, k_class_ids, class_colour_map, k_coords, n_cone_lines,
            title=(f"Kernel PCA (RBF γ={kernel_gamma}) — Stage {stage_idx}  "
                   f"({len(k_class_ids)} cls, {explained_kern:.1%} var)"),
            xlabel="KPC1", ylabel="KPC2", zlabel="KPC3",
        )

    # Legend: one entry per stage.
    legend_handles = [
        plt.Line2D([0], [0], color=stage_colour_map[s], linewidth=2, label=f"Stage {s}")
        for s in all_stages
    ]
    fig.legend(handles=legend_handles, fontsize=8,
               loc="lower center", ncol=len(all_stages),
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = f"extreme_rays_stage{stage_idx}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VisualizeRays] Saved → {path}")


class HullManager:
    def __init__(
        self,
        n_rays: int = 50,
        use_pca: bool = True,
        pca_dim: int = 64,
        adaptive_rays: bool = False,
        total_classes: int = 0,
        ray_budget: Optional[int] = None,
        min_rays: int = 1,
        ray_diversity: str = "hybrid",
        spa_oversample: int = 3,
    ):
        self.n_rays = n_rays
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.adaptive_rays = adaptive_rays
        self.total_classes = total_classes
        self.ray_budget = ray_budget      # total ray budget across all classes; None → use D
        self.min_rays = min_rays          # floor on per-class ray count
        self.ray_diversity = ray_diversity
        self.spa_oversample = spa_oversample
        self.static_hulls = {}       # Frozen at birth
        self.extreme_ray_images = {} # cls_str -> list of CPU image tensors

    def _compute_n_rays(self, D: int) -> int:
        """
        Return the per-class ray count to use when building hulls.

        Fixed mode  (adaptive_rays=False): always returns self.n_rays.
        Adaptive mode (adaptive_rays=True):
            per_class = max(min_rays, ray_budget // total_classes)
            where ray_budget defaults to D (one full subspace worth of rays).
            self.n_rays acts as a hard upper cap.

        This keeps the combined ray matrix rank at most ray_budget ≤ D,
        preventing subspace saturation as the class count grows.
        """
        if not self.adaptive_rays or self.total_classes <= 0 or D <= 0:
            return self.n_rays
        budget = self.ray_budget if self.ray_budget is not None else D
        per_class = max(self.min_rays, budget // self.total_classes)
        result = min(per_class, self.n_rays)
        return result
    
    def orthogonalize_new_features(self, new_feature_dict, threshold=1e-4):
        """
        Projects new features into the null space of all past static hulls.
        This isolates the new conic hulls from intersecting with past tasks.
        
        Parameters
        ----------
        new_feature_dict : dict {class_id: np.ndarray of shape (N, D)}
        threshold        : float, cutoff for SVD singular values to determine rank
        
        Returns
        -------
        dict : The rotated/projected feature dictionary ready for fit_new_classes.
        """
        if not self.static_hulls:
            # Stage 0: Nothing to rotate away from yet
            return new_feature_dict 
            
        # 1. Collect all extreme rays from past static hulls to define the occupied subspace.
        # (Assuming your hull objects store their defining vectors in a '.rays' or '.vertices' attribute)
        past_vectors = []
        for hull in self.static_hulls.values():
            if hasattr(hull, 'rays'):
                past_vectors.append(hull.rays)
            elif hasattr(hull, 'vertices'):
                past_vectors.append(hull.vertices)
            else:
               past_vectors.append(hull.extreme_rays)
                
        # Stack into matrix of shape (total_past_rays, feature_dim)
        past_matrix = np.vstack(past_vectors) 
        feature_dim = past_matrix.shape[1]
        
        # 2. Perform SVD to find the basis of the occupied space
        # past_matrix = U * S * Vh. The rows of Vh form an orthonormal basis for the space.
        _, S, Vh = np.linalg.svd(past_matrix, full_matrices=False)
        
        # Determine the rank (how many dimensions are actually being used by past tasks)
        rank = np.sum(S > threshold)
        
        if rank >= feature_dim:
            print(f"[Warning] Past tasks occupy the entire {feature_dim}D space. No null space available for rotation.")
            return new_feature_dict
            
        # V_past contains the basis vectors of the occupied space
        V_past = Vh[:rank, :] # (rank, feature_dim)
        
        # 3. Construct the null space projection matrix
        # P_null = I - V_past^T * V_past
        identity = np.eye(feature_dim)
        P_null = identity - (V_past.T @ V_past) 
        
        # 4. Project and re-normalize the new features
        rotated_feature_dict = {}
        for cls, feats in new_feature_dict.items():
            # Project into the null space
            proj_feats = feats @ P_null.T
            
            # L2 Normalize to push the features back onto the unit hypersphere.
            # This maintains the exact intra-class cosine geometry.
            norms = np.linalg.norm(proj_feats, axis=1, keepdims=True)
            
            # Avoid division by zero for any outlier vectors that lived entirely in the past subspace
            norms[norms < 1e-9] = 1.0 
            
            rotated_feature_dict[cls] = proj_feats / norms
            
        print(f"    [HullManager] Projected new features into {feature_dim - rank}D null space.")
        return rotated_feature_dict

    def fit_new_classes(self, feature_dict, region_registry: Optional[SphericalRegionRegistry] = None):
        """Fits and freezes hulls for classes seen for the first time."""
        D = next(iter(feature_dict.values())).shape[1] if feature_dict else 0
        n_rays_effective = self._compute_n_rays(D)
        if self.adaptive_rays and D > 0:
            budget = self.ray_budget if self.ray_budget is not None else D
            print(
                f"  [HullManager] adaptive rays: budget={budget} / "
                f"total_classes={self.total_classes} → "
                f"{n_rays_effective} rays/class  (cap={self.n_rays})"
            )
        if region_registry is not None:
            new_hulls = build_region_conic_hulls(
                feature_dict,
                region_registry,
                use_pca=self.use_pca,
                pca_dim=self.pca_dim,
                k_local=10,
            )
        else:
            new_hulls = build_class_conic_hulls(
                feature_dict,
                n_rays=n_rays_effective,
                use_pca=self.use_pca,
                pca_dim=self.pca_dim,
                k_local=10,
                ray_diversity=self.ray_diversity,
                spa_oversample=self.spa_oversample,
            )
        for cls, hull in new_hulls.items():
            if cls not in self.static_hulls:
                self.static_hulls[cls] = hull

    def get_dynamic_hulls(self, feature_dict):
        """Fits hulls to the current representations (Dynamic)."""
        D = next(iter(feature_dict.values())).shape[1] if feature_dict else 0
        return build_class_conic_hulls(
            feature_dict,
            n_rays=self._compute_n_rays(D),
            use_pca=False,
            ray_diversity=self.ray_diversity,
            spa_oversample=self.spa_oversample,
        )

    def get_kernel_dynamic_hulls(self, feature_dict, kernel="spread", gamma=1.0):
        """Fits kernel hulls to the current representations (Dynamic)."""
        D = next(iter(feature_dict.values())).shape[1] if feature_dict else 0
        return build_class_kernel_conic_hulls(
            feature_dict,
            n_rays=self._compute_n_rays(D),
            kernel=kernel,
            gamma=gamma,
        )

    # Canonical ordering used everywhere for consistent printing
    SCORE_NAMES = ["cosine", "angular_margin", "blended", "max_ray_sim"]

    def evaluate(self, backbone, test_loader, hulls, device):
        """Classification rule: argmax of hull.score() (cosine, original scheme)."""
        backbone.eval()
        correct, total = 0, 0
        all_classes = list(hulls.keys())

        with torch.no_grad():
            for imgs, labels in test_loader:
                feats = backbone(imgs.to(device)).cpu().numpy()
                scores = np.stack([hulls[name].score(feats) for name in all_classes], axis=1)
                best_idx = np.argmax(scores, axis=1)
                preds = np.array([int(all_classes[i]) for i in best_idx])
                correct += (preds == labels.numpy()).sum()
                total += labels.size(0)
        return correct / total

    def evaluate_all_scores(self, backbone, test_loader, hulls, device):
        """
        Evaluate classification accuracy for all five scoring schemes in one pass.

        Each batch computes a single NNLS reconstruction per class hull (via
        hull.score_all), then argmax-classifies per scheme.

        Returns
        -------
        dict  { scheme_name: float accuracy }
        Keys: 'cosine', 'residual_coverage', 'angular', 'nnls_residual', 'combined'.
        """
        backbone.eval()
        all_classes = list(hulls.keys())
        correct = {s: 0 for s in self.SCORE_NAMES}
        total   = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                feats      = backbone(imgs.to(device)).cpu().numpy()
                labels_np  = labels.numpy()

                # One score_all call per class hull — single NNLS solve each
                per_class  = [hulls[name].score_all(feats) for name in all_classes]

                for s in self.SCORE_NAMES:
                    mat      = np.stack([pc[s] for pc in per_class], axis=1)  # (N, C)
                    best_idx = np.argmax(mat, axis=1)
                    preds    = np.array([int(all_classes[i]) for i in best_idx])
                    correct[s] += (preds == labels_np).sum()
                total += labels.size(0)

        return {s: correct[s] / total for s in self.SCORE_NAMES}

    # Collaborative scoring schemes (joint NNLS over all classes at once)
    COLLAB_SCORE_NAMES = ["collab_energy", "collab_residual", "collab_margin"]

    def evaluate_collaborative(
        self,
        backbone,
        test_loader,
        hulls: dict,
        device,
        lasso_lambda: float = 0.0,
        temperature: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Evaluate classification using collaborative (joint) NNLS scoring.

        Instead of C independent per-class NNLS solves, concatenate all class
        dictionaries into one joint problem and classify by attributed energy,
        partial residual, or energy margin over the runner-up.

        Parameters
        ----------
        hulls         : per-class ConicHull dict (typically static_hulls).
        lasso_lambda  : L1 regularization weight (0 = pure NNLS; try 1e-3
                        for cleaner class attribution when cones overlap).
        temperature   : per-class temperature calibration dict (optional).

        Returns
        -------
        dict {scheme_name: float accuracy}
        Keys: 'collab_energy', 'collab_residual', 'collab_margin'.
        """
        import numpy as np

        backbone.eval()
        all_classes  = list(hulls.keys())
        if not all_classes:
            return {s: 0.0 for s in self.COLLAB_SCORE_NAMES}

        correct = {s: 0 for s in self.COLLAB_SCORE_NAMES}
        total   = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                feats     = backbone(imgs.to(device)).cpu().numpy()
                labels_np = labels.numpy()

                # One joint NNLS solve for all classes
                scores_dict, class_names = collaborative_nnls_scores(
                    feats, hulls,
                    lasso_lambda=lasso_lambda,
                    temperature=temperature,
                )

                cls_to_int = {name: int(name) for name in class_names}

                for s in self.COLLAB_SCORE_NAMES:
                    mat      = scores_dict[s]                   # (N, C)
                    best_idx = np.argmax(mat, axis=1)           # (N,)
                    preds    = np.array([cls_to_int[class_names[i]] for i in best_idx])
                    correct[s] += int((preds == labels_np).sum())
                total += int(labels.size(0))

        return {s: correct[s] / total for s in self.COLLAB_SCORE_NAMES}

    def evaluate_ood_detection(
        self,
        backbone: "nn.Module",
        id_loader,
        ood_loader,
        hulls: dict,
        device,
        score_key: str = "cosine",
        calibrate_percentile: float = 5.0,
    ) -> Optional[dict]:
        """
        Binary OOD detection using max hull membership score as the signal.

        At stage k the caller passes:
          id_loader  — test data from stages 0..k   (in-distribution)
          ood_loader — test data from stages k+1..N (out-of-distribution)

        A threshold is calibrated as the `calibrate_percentile`-th percentile
        of the ID scores.  Samples scoring below this threshold are predicted OOD.

        Returns a dict with accuracy, F1, precision, recall, threshold, and
        confusion counts; or None if either loader is empty.
        """
        backbone.eval()
        all_classes = list(hulls.keys())
        if not all_classes:
            return None

        def _max_scores(loader) -> np.ndarray:
            parts: list = []
            with torch.no_grad():
                for imgs, _ in loader:
                    feats = backbone(imgs.to(device)).cpu().numpy()
                    per_class = [hulls[name].score_all(feats) for name in all_classes]
                    mat = np.stack([pc[score_key] for pc in per_class], axis=1)  # (N, C)
                    parts.append(mat.max(axis=1))                                # (N,)
            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

        id_scores  = _max_scores(id_loader)
        ood_scores = _max_scores(ood_loader)
        if len(id_scores) == 0 or len(ood_scores) == 0:
            return None

        threshold = float(np.percentile(id_scores, calibrate_percentile))

        # OOD label = 1, ID label = 0
        y_true = np.concatenate([
            np.zeros(len(id_scores),  dtype=int),
            np.ones(len(ood_scores), dtype=int),
        ])
        y_pred = np.concatenate([
            (id_scores  < threshold).astype(int),
            (ood_scores < threshold).astype(int),
        ])

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        n    = len(y_true)
        acc  = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "accuracy":       acc,
            "f1":             f1,
            "precision":      prec,
            "recall":         rec,
            "threshold":      threshold,
            "tp":             tp,
            "fp":             fp,
            "tn":             tn,
            "fn":             fn,
            "n_id":           len(id_scores),
            "n_ood":          len(ood_scores),
            "id_score_mean":  float(id_scores.mean()),
            "ood_score_mean": float(ood_scores.mean()),
            "score_key":      score_key,
        }

    def _build_stage_subspaces(
        self,
        hulls: dict,
        stage_class_map: dict,
        variance_threshold: float = 0.95,
    ) -> dict:
        """
        Build a projection matrix P = V Vᵀ for each stage from its extreme rays.

        Stacks all extreme rays across every class in a stage, runs thin SVD,
        and keeps the top-k right singular vectors that capture
        `variance_threshold` of the total squared-SV energy.

        Returns {stage_idx: P} where P is a (D, D) float64 ndarray.
        """
        stage_rays: dict = defaultdict(list)
        for cls_name, hull in hulls.items():
            if hull.extreme_rays_ is None:
                continue
            s = stage_class_map.get(int(cls_name), -1)
            stage_rays[s].append(hull.extreme_rays_)   # (n_rays, D)

        subspaces = {}
        for stage_idx, ray_list in stage_rays.items():
            R   = np.vstack(ray_list)                  # (total_rays, D)
            _, S, Vt = np.linalg.svd(R, full_matrices=False)   # Vt: (r, D)
            cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
            k   = min(int(np.searchsorted(cumvar, variance_threshold)) + 1, len(S))
            V   = Vt[:k].T                             # (D, k) orthonormal basis
            subspaces[stage_idx] = V @ V.T             # (D, D) projection matrix

        return subspaces

    @staticmethod
    def extract_class_features(
        backbone: "nn.Module",
        loader,
        device,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run backbone on loader; return {class_str: (N, D)} feature matrices."""
        backbone.eval()
        want = None if class_names is None else {str(c) for c in class_names}
        buckets: dict = defaultdict(list)

        with torch.no_grad():
            for imgs, labels in loader:
                feats = backbone(imgs.to(device)).cpu().numpy()
                for i, lbl in enumerate(labels.numpy().tolist()):
                    key = str(lbl)
                    if want is not None and key not in want:
                        continue
                    buckets[key].append(feats[i])

        return {k: np.asarray(v, dtype=np.float32) for k, v in buckets.items()}

    def _build_stage_planes(
        self,
        hulls: dict,
        stage_class_map: dict,
        plane_source: str,
        subspace_variance_threshold: float,
        backbone: Optional["nn.Module"] = None,
        calibration_loader=None,
        device=None,
        calibration_max_per_class: Optional[int] = 64,
    ) -> Dict[int, np.ndarray]:
        """
        Build per-stage plane bases.

        plane_source
        ------------
        ``features`` : current backbone features (drift-aligned; recommended)
        ``hulls``    : frozen extreme-ray geometry at hull-build time
        ``both``     : stack hull rays + calibration features per stage
        """
        source = (plane_source or "features").lower()
        if source == "hulls":
            return build_stage_plane_bases(
                hulls, stage_class_map, subspace_variance_threshold,
            )

        if calibration_loader is None:
            raise ValueError(
                "calibration_loader is required when plane_source is "
                f"'{source}' (pass train data seen so far)."
            )
        if backbone is None or device is None:
            raise ValueError("backbone and device required for feature-based planes")

        feature_dict = self.extract_class_features(
            backbone, calibration_loader, device,
            class_names=list(hulls.keys()),
        )
        if source == "both":
            return build_stage_plane_bases_blended(
                hulls, feature_dict, stage_class_map,
                subspace_variance_threshold,
                max_per_class=calibration_max_per_class,
            )
        return build_stage_plane_bases_from_features(
            feature_dict, stage_class_map, subspace_variance_threshold,
            max_per_class=calibration_max_per_class,
        )

    @staticmethod
    def _build_stage_centroids(
        feature_dict:    Dict[str, np.ndarray],
        stage_class_map: Dict[int, int],
    ) -> Tuple[List[int], np.ndarray]:
        """Per-stage mean direction from calibration features. Returns (stage_ids, C)."""
        stage_parts: dict = defaultdict(list)
        for cls_name, feats in feature_dict.items():
            s = stage_class_map.get(int(cls_name), -1)
            if s < 0:
                continue
            stage_parts[s].append(np.asarray(feats, dtype=np.float64))

        ordered = sorted(stage_parts.keys())
        cents   = []
        for s in ordered:
            P = normalize(np.vstack(stage_parts[s]), axis=1)
            cents.append(normalize(P.mean(axis=0, keepdims=True), axis=1)[0])
        return ordered, np.stack(cents, axis=0).astype(np.float32)

    @staticmethod
    def align_features_to_hull_space(
        features: np.ndarray,
        A_inv_T:  np.ndarray,
        mu_old:   np.ndarray,
        mu_new:   np.ndarray,
    ) -> np.ndarray:
        """Map current-backbone features into hull-native space at build time."""
        X = np.asarray(features, dtype=np.float64)
        aligned = (X - mu_new) @ A_inv_T + mu_old
        return normalize(aligned, axis=1).astype(np.float32)

    @staticmethod
    def _drift_map_for_scoring(
        drift_maps: Optional[Dict[int, dict]],
        stage_idx: int,
        cls_name: Optional[str] = None,
    ) -> Optional[dict]:
        if not drift_maps or stage_idx not in drift_maps:
            return None
        dm = drift_maps[stage_idx]
        if dm.get("mode") == "per_class" and cls_name is not None:
            return dm.get("classes", {}).get(cls_name)
        if "A_inv_T" in dm:
            return dm
        return None

    def build_stage_drift_maps(
        self,
        backbone: "nn.Module",
        replay_buffer: "ReplayBuffer",
        device,
        stage_to_classes: dict,
        stage_stats_snapshots: Dict[int, Dict],
        ordered_stages: list,
        *,
        hulls: Optional[Dict[str, "ConicHull"]] = None,
        stage_feature_snapshots: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
        align_mode: str = "dense_rays",
        drift_method: str = "procrustes",
        pair_method: str = "procrustes",
        ridge: float = 1e-3,
        ray_weight: float = 3.0,
        batch_size: int = 64,
    ) -> Dict[int, dict]:
        """
        Per-stage drift maps: current features → hull build-time coordinates.

        ``align_mode``
          ``centroid``   — class-mean Procrustes (legacy)
          ``dense``      — all replay pairs per stage
          ``dense_rays`` — replay + weighted extreme-ray anchors (default)
          ``per_class``  — separate map per class (strongest for scoring)
        """
        mode = align_mode.lower()
        drift_maps: Dict[int, dict] = {}

        for stage_idx in ordered_stages:
            class_ids = [int(c) for c in stage_to_classes[stage_idx]]
            cls_names = [str(c) for c in class_ids]

            if mode == "centroid":
                old_stats = stage_stats_snapshots.get(stage_idx)
                if not old_stats or len(old_stats) < 2:
                    continue
                new_stats = _extract_class_stats_from_buffer(
                    backbone, replay_buffer, class_ids, device, batch_size,
                )
                if len(new_stats) < 2:
                    continue
                try:
                    A = estimate_affine_drift(old_stats, new_stats, method=drift_method)
                except ValueError:
                    continue
                common = sorted(set(old_stats.keys()) & set(new_stats.keys()))
                mu_old = np.mean([old_stats[c]["mean"] for c in common], axis=0)
                mu_new = np.mean([new_stats[c]["mean"] for c in common], axis=0)
                try:
                    A_inv_T = np.linalg.inv(A).T
                except np.linalg.LinAlgError:
                    A_inv_T = np.linalg.pinv(A).T
                drift_maps[stage_idx] = {
                    "mode":    "stage",
                    "A":       A.astype(np.float32),
                    "A_inv_T": A_inv_T.astype(np.float32),
                    "mu_old":  mu_old.astype(np.float32),
                    "mu_new":  mu_new.astype(np.float32),
                    "n_pairs": len(common),
                    "method":  drift_method,
                }
                continue

            feat_snap = (stage_feature_snapshots or {}).get(stage_idx)
            if not feat_snap:
                continue

            if mode == "per_class":
                class_maps: Dict[str, dict] = {}
                for cls_id, cls_name in zip(class_ids, cls_names):
                    X_old, X_new, w = _collect_drift_alignment_pairs(
                        backbone, replay_buffer, [cls_id], feat_snap, device,
                        hulls=hulls, ray_weight=ray_weight, batch_size=batch_size,
                    )
                    if X_old.shape[0] < 2:
                        continue
                    try:
                        class_maps[cls_name] = fit_drift_map_from_pairs(
                            X_old, X_new, method=pair_method, ridge=ridge,
                            sample_weights=w,
                        )
                    except ValueError:
                        continue
                if len(class_maps) >= 2:
                    drift_maps[stage_idx] = {
                        "mode":    "per_class",
                        "classes": class_maps,
                    }
                continue

            use_rays = mode == "dense_rays"
            X_old, X_new, w = _collect_drift_alignment_pairs(
                backbone, replay_buffer, class_ids, feat_snap, device,
                hulls=hulls if use_rays else None,
                ray_weight=ray_weight if use_rays else 0.0,
                batch_size=batch_size,
            )
            if X_old.shape[0] < 2:
                continue
            try:
                dm = fit_drift_map_from_pairs(
                    X_old, X_new, method=pair_method, ridge=ridge,
                    sample_weights=w,
                )
            except ValueError:
                continue
            dm["mode"] = "stage"
            drift_maps[stage_idx] = dm

        return drift_maps

    def _calibrate_stage_routing(
        self,
        backbone: "nn.Module",
        calibration_loader,
        device,
        hulls: dict,
        stage_class_map: dict,
        ordered_stages: list,
        stage_cols_map: dict,
        all_classes: list,
        stage_to_classes: dict,
        score_key: str,
        cascade_percentile: float = 10.0,
        drift_maps: Optional[Dict[int, dict]] = None,
    ) -> dict:
        """
        Estimate per-stage acceptance thresholds and cross-stage score biases
        from calibration data (current backbone, train features seen so far).

        Used by ``cascade`` and ``hull_max_cal`` routers to protect older stages
        when feature drift inflates scores on newer hulls.
        """
        backbone.eval()
        in_stage_max:  dict = {s: [] for s in ordered_stages}
        cross_stage_max: dict = {s: [] for s in ordered_stages}

        def _score_batch(feats_raw: np.ndarray) -> np.ndarray:
            N = feats_raw.shape[0]
            mat = np.full((N, len(all_classes)), -np.inf, dtype=np.float32)
            for stage_idx in ordered_stages:
                for cls_name in stage_to_classes[stage_idx]:
                    m = self._drift_map_for_scoring(drift_maps, stage_idx, cls_name)
                    if m is not None:
                        feats_s = self.align_features_to_hull_space(
                            feats_raw, m["A_inv_T"], m["mu_old"], m["mu_new"],
                        )
                    else:
                        feats_s = feats_raw
                    mat[:, all_classes.index(cls_name)] = (
                        hulls[cls_name].score_all(feats_s)[score_key]
                    )
            return mat

        with torch.no_grad():
            for imgs, labels in calibration_loader:
                feats = backbone(imgs.to(device)).cpu().numpy()
                mat   = _score_batch(feats)
                stage_max = np.stack(
                    [mat[:, stage_cols_map[s]].max(axis=1) for s in ordered_stages],
                    axis=1,
                )
                for i, lbl in enumerate(labels.numpy().tolist()):
                    true_s = stage_class_map.get(int(lbl), -1)
                    if true_s < 0:
                        continue
                    for si, s in enumerate(ordered_stages):
                        val = float(stage_max[i, si])
                        if s == true_s:
                            in_stage_max[s].append(val)
                        else:
                            cross_stage_max[s].append(val)

        thresholds = {}
        biases     = {}
        for s in ordered_stages:
            if in_stage_max[s]:
                thresholds[s] = float(np.percentile(in_stage_max[s], cascade_percentile))
            else:
                thresholds[s] = 0.5
            if cross_stage_max[s]:
                biases[s] = float(np.percentile(cross_stage_max[s], 75))
            else:
                biases[s] = 0.0
        return {"thresholds": thresholds, "biases": biases}

    def evaluate_staged(
        self,
        backbone: "nn.Module",
        test_loader,
        hulls: dict,
        device,
        stage_class_map: dict,
        score_key: str = "cosine",
        subspace_variance_threshold: float = 0.95,
        plane_scoring: bool = True,
        score_space: str = "full",
        stage_routing: str = "cascade",
        plane_source: str = "features",
        calibration_loader=None,
        calibration_max_per_class: int = 64,
        routing_cascade_percentile: float = 10.0,
        drift_align: bool = False,
        stage_stats_snapshots: Optional[Dict[int, Dict]] = None,
        stage_feature_snapshots: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
        replay_buffer: Optional["ReplayBuffer"] = None,
        drift_method: str = "procrustes",
        drift_align_mode: str = "dense_rays",
        drift_pair_method: str = "procrustes",
        drift_ridge: float = 1e-3,
        drift_ray_weight: float = 3.0,
        drift_routing_only: bool = True,
        score_hulls: Optional[dict] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Staged classification: route to a stage, then argmax hull score within it.

        ``hulls``
          Hull set used for **stage routing** and drift-map ray anchors (typically
          frozen static hulls).

        ``score_hulls``
          Hull set used for **within-stage class scoring** after routing (e.g.
          dynamic hulls re-fit on the current backbone).  Defaults to ``hulls``.

        ``drift_align``
          When True, per-stage drift maps align queries into each stage's
          hull-native space before **routing** (and scoring when
          ``drift_routing_only=False``).

        ``drift_routing_only``
          When True (default), drift maps are used only for routing /
          calibration; within-stage scoring uses raw features against
          ``score_hulls``.

        ``stage_routing``
          ``cascade``      — oldest-first with calibrated thresholds
          ``hull_max_cal`` — argmax per-stage max minus cross-stage bias
          ``hull_max``     — argmax raw per-stage max
        """
        backbone.eval()
        routing_hulls = hulls
        scoring_hulls = score_hulls if score_hulls is not None else hulls
        use_split_hulls = scoring_hulls is not routing_hulls
        all_classes = list(routing_hulls.keys())

        stage_to_classes: dict = defaultdict(list)
        for cls_name in all_classes:
            s = stage_class_map.get(int(cls_name), -1)
            stage_to_classes[s].append(cls_name)

        cls_to_idx = {name: i for i, name in enumerate(all_classes)}
        use_plane_scores = (
            score_space.lower() == "plane"
            or (score_space.lower() == "auto" and plane_scoring)
        )
        routing = stage_routing.lower()

        plane_bases: Dict[int, np.ndarray] = {}
        B_list: list = []
        rays_in_plane: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
        stage_centroid_stages: Optional[List[int]] = None
        stage_centroid_mat: Optional[np.ndarray] = None
        P_list: list = []
        ordered_stages: list = []

        plane_hulls = scoring_hulls if use_split_hulls else routing_hulls
        if use_plane_scores or routing in ("energy", "residual", "centroid"):
            plane_bases = self._build_stage_planes(
                plane_hulls, stage_class_map, plane_source,
                subspace_variance_threshold, backbone,
                calibration_loader, device, calibration_max_per_class,
            ) if use_plane_scores or routing in ("energy", "residual") else {}
            if not plane_bases and (use_plane_scores or routing in ("energy", "residual")):
                plane_bases = build_stage_plane_bases(
                    plane_hulls, stage_class_map, subspace_variance_threshold,
                )
            if plane_bases:
                ordered_stages = sorted(plane_bases.keys())
                B_list = [plane_bases[s] for s in ordered_stages]

            if routing == "centroid":
                if calibration_loader is not None:
                    cal_fd = self.extract_class_features(
                        backbone, calibration_loader, device,
                        class_names=list(routing_hulls.keys()),
                    )
                    stage_centroid_stages, stage_centroid_mat = (
                        self._build_stage_centroids(cal_fd, stage_class_map)
                    )
                elif plane_bases:
                    D = next(iter(routing_hulls.values())).extreme_rays_.shape[1]
                    poles = build_orthogonal_stage_poles(len(ordered_stages), D, seed=0)
                    stage_centroid_stages = ordered_stages
                    stage_centroid_mat = poles[: len(ordered_stages)]

            if use_plane_scores and plane_bases:
                for stage_idx in ordered_stages:
                    B = plane_bases[stage_idx]
                    for cls_name in stage_to_classes[stage_idx]:
                        hull = plane_hulls.get(cls_name)
                        if hull is None or hull.extreme_rays_ is None:
                            continue
                        rays_in_plane[stage_idx][cls_name] = project_to_stage_plane(
                            hull.extreme_rays_, B
                        )

        if not ordered_stages:
            ordered_stages = sorted(stage_to_classes.keys())
            if not use_plane_scores and routing == "residual":
                stage_subspaces = self._build_stage_subspaces(
                    plane_hulls, stage_class_map, subspace_variance_threshold
                )
                ordered_stages = sorted(stage_subspaces.keys())
                P_list = [stage_subspaces[s] for s in ordered_stages]

        stage_cols_map = {
            s: np.array([cls_to_idx[n] for n in stage_to_classes[s]])
            for s in ordered_stages
        }

        drift_maps: Optional[Dict[int, dict]] = None
        if drift_align and replay_buffer is not None:
            drift_maps = self.build_stage_drift_maps(
                backbone, replay_buffer, device,
                stage_to_classes, stage_stats_snapshots or {},
                ordered_stages,
                hulls=routing_hulls,
                stage_feature_snapshots=stage_feature_snapshots,
                align_mode=drift_align_mode,
                drift_method=drift_method,
                pair_method=drift_pair_method,
                ridge=drift_ridge,
                ray_weight=drift_ray_weight,
            )
            if verbose and use_split_hulls:
                print("    [StagedEval]  routing=static  scoring=dynamic")
            if verbose and drift_maps:
                parts = []
                for s in ordered_stages:
                    if s not in drift_maps:
                        continue
                    dm = drift_maps[s]
                    if dm.get("mode") == "per_class":
                        n_cls = len(dm.get("classes", {}))
                        res = np.mean([
                            c["residual"] for c in dm["classes"].values()
                            if "residual" in c
                        ]) if n_cls else 0.0
                        parts.append(f"s{s}:per_class({n_cls},res={res:.3f})")
                    else:
                        parts.append(
                            f"s{s}:{dm.get('n_pairs', '?')}pairs"
                            f"(res={dm.get('residual', 0.0):.3f})"
                        )
                print(
                    f"    [StagedEval drift]  {drift_align_mode}/{drift_pair_method}  "
                    f"{len(drift_maps)}/{len(ordered_stages)} stages  "
                    + "  ".join(parts)
                )
            elif verbose and drift_align:
                print("    [StagedEval drift]  no maps built — falling back to raw features")

        routing_cal: Optional[dict] = None
        if calibration_loader is not None and routing in ("cascade", "hull_max_cal"):
            routing_cal = self._calibrate_stage_routing(
                backbone, calibration_loader, device, routing_hulls, stage_class_map,
                ordered_stages, stage_cols_map, all_classes, stage_to_classes,
                score_key, cascade_percentile=routing_cascade_percentile,
                drift_maps=drift_maps,
            )
        elif routing == "cascade" and verbose:
            print("    [StagedEval] cascade routing needs calibration_loader; "
                  "falling back to hull_max")

        n_correct           = 0
        n_oracle_correct    = 0
        n_oracle_raw        = 0
        n_oracle_aligned    = 0
        n_total             = 0
        n_intra             = 0
        n_inter             = 0
        n_routing_correct   = 0
        n_within_stage      = 0
        n_within_correct    = 0
        n_routed_by_stage: dict       = defaultdict(int)
        n_oracle_by_stage: dict       = defaultdict(int)
        confusion_by_pred_stage: dict = defaultdict(int)

        def _score_mat(
            feats_raw: np.ndarray,
            hull_dict: dict,
            *,
            drift_aligned: bool = False,
        ) -> np.ndarray:
            N = feats_raw.shape[0]
            mat = np.full((N, len(all_classes)), -np.inf, dtype=np.float32)
            for stage_idx in ordered_stages:
                for cls_name in stage_to_classes[stage_idx]:
                    hull = hull_dict.get(cls_name)
                    if hull is None:
                        continue
                    if drift_aligned:
                        m = self._drift_map_for_scoring(drift_maps, stage_idx, cls_name)
                        if m is not None:
                            feats_s = self.align_features_to_hull_space(
                                feats_raw, m["A_inv_T"], m["mu_old"], m["mu_new"],
                            )
                        else:
                            feats_s = feats_raw
                    else:
                        feats_s = feats_raw
                    mat[:, cls_to_idx[cls_name]] = hull.score_all(feats_s)[score_key]
            return mat

        def _route_score_mat(feats_raw: np.ndarray) -> np.ndarray:
            if drift_maps:
                return _score_mat(
                    feats_raw, routing_hulls, drift_aligned=True,
                )
            return _score_mat(feats_raw, routing_hulls, drift_aligned=False)

        def _classify_score_mat(feats_raw: np.ndarray) -> np.ndarray:
            if drift_maps and not drift_routing_only:
                return _score_mat(
                    feats_raw, scoring_hulls, drift_aligned=True,
                )
            return _score_mat(feats_raw, scoring_hulls, drift_aligned=False)

        def _raw_classify_score_mat(feats_raw: np.ndarray) -> np.ndarray:
            return _score_mat(feats_raw, scoring_hulls, drift_aligned=False)

        def _aligned_classify_score_mat(feats_raw: np.ndarray) -> np.ndarray:
            return _score_mat(feats_raw, scoring_hulls, drift_aligned=True)

        def _plane_score_mat(feats_n: np.ndarray) -> np.ndarray:
            N = feats_n.shape[0]
            mat = np.full((N, len(all_classes)), -np.inf, dtype=np.float32)
            for stage_idx in ordered_stages:
                B = plane_bases[stage_idx]
                feats_p = project_to_stage_plane(feats_n, B)
                for cls_name in stage_to_classes[stage_idx]:
                    rays_s = rays_in_plane[stage_idx].get(cls_name)
                    if rays_s is None or len(rays_s) == 0:
                        continue
                    sc = _score_projected_hull_batch(feats_p, rays_s)[score_key]
                    mat[:, cls_to_idx[cls_name]] = sc
            return mat

        def _stage_max_scores(score_mat: np.ndarray) -> np.ndarray:
            return np.stack(
                [score_mat[:, stage_cols_map[s]].max(axis=1) for s in ordered_stages],
                axis=1,
            )

        def _route_stages(
            feats_n: np.ndarray,
            feats_raw: np.ndarray,
            score_mat: np.ndarray,
        ) -> np.ndarray:
            stage_scores = _stage_max_scores(score_mat)
            N = stage_scores.shape[0]

            if routing == "cascade" and routing_cal is not None:
                thr = routing_cal["thresholds"]
                assigned = np.empty(N, dtype=int)
                for n in range(N):
                    picked = ordered_stages[-1]
                    for si, s in enumerate(ordered_stages):
                        if stage_scores[n, si] >= thr[s]:
                            picked = s
                            break
                    assigned[n] = picked
                return assigned

            if routing == "hull_max_cal" and routing_cal is not None:
                bias = np.array(
                    [routing_cal["biases"][s] for s in ordered_stages],
                    dtype=np.float32,
                )
                adjusted = stage_scores - bias[np.newaxis, :]
                best_local = adjusted.argmax(axis=1)
                return np.array([ordered_stages[i] for i in best_local])

            if routing == "cascade":
                best_local = stage_scores.argmax(axis=1)
                return np.array([ordered_stages[i] for i in best_local])

            if routing == "hull_max":
                best_local = stage_scores.argmax(axis=1)
                return np.array([ordered_stages[i] for i in best_local])

            if routing == "centroid" and stage_centroid_mat is not None:
                cols = stage_centroid_stages or ordered_stages
                C = normalize(stage_centroid_mat, axis=1)
                best_local = (feats_n @ C.T).argmax(axis=1)
                return np.array([cols[i] for i in best_local])

            if routing == "energy" and B_list:
                energies = np.stack(
                    [np.sum((feats_n @ B.T) ** 2, axis=1) for B in B_list], axis=1,
                )
                best_local = energies.argmax(axis=1)
                return np.array([ordered_stages[i] for i in best_local])

            if routing == "residual" and B_list:
                residuals = np.stack(
                    [
                        np.linalg.norm(
                            feats_n - project_to_stage_plane(feats_n, B, False),
                            axis=1,
                        )
                        for B in B_list
                    ],
                    axis=1,
                )
                best_local = residuals.argmin(axis=1)
                return np.array([ordered_stages[i] for i in best_local])

            if P_list:
                residuals = np.stack(
                    [np.linalg.norm(feats_raw - feats_raw @ P, axis=1) for P in P_list],
                    axis=1,
                )
                best_local = residuals.argmin(axis=1)
                return np.array([ordered_stages[i] for i in best_local])

            best_local = stage_scores.argmax(axis=1)
            return np.array([ordered_stages[i] for i in best_local])

        def _preds_from_assignment(
            assignment: np.ndarray,
            score_mat: np.ndarray,
        ) -> np.ndarray:
            N = score_mat.shape[0]
            preds = np.zeros(N, dtype=int)
            for stage_idx in ordered_stages:
                mask = assignment == stage_idx
                if not mask.any():
                    continue
                stage_cls  = stage_to_classes[stage_idx]
                stage_cols = stage_cols_map[stage_idx]
                sub_scores = score_mat[mask][:, stage_cols]
                best_local = sub_scores.argmax(axis=1)
                preds[mask] = np.array([int(stage_cls[i]) for i in best_local])
            return preds

        with torch.no_grad():
            for imgs, labels in test_loader:
                feats     = backbone(imgs.to(device)).cpu().numpy()
                labels_np = labels.numpy()
                N         = feats.shape[0]
                feats_n   = normalize(feats, axis=1)

                true_stages = np.array(
                    [stage_class_map.get(int(lbl), -1) for lbl in labels_np]
                )

                route_mat = _route_score_mat(feats)
                score_mat = (
                    _plane_score_mat(feats_n)
                    if use_plane_scores and not drift_maps
                    else _classify_score_mat(feats)
                )

                assigned      = _route_stages(feats_n, feats, route_mat)
                oracle_assign = true_stages.copy()

                raw_mat = _raw_classify_score_mat(feats) if drift_maps else score_mat
                aligned_mat = (
                    _aligned_classify_score_mat(feats) if drift_maps else score_mat
                )

                for s in ordered_stages:
                    n_routed_by_stage[s] += int((assigned == s).sum())
                    n_oracle_by_stage[s] += int((oracle_assign == s).sum())

                n_routing_correct += int((assigned == true_stages).sum())

                preds        = _preds_from_assignment(assigned, score_mat)
                preds_oracle = _preds_from_assignment(oracle_assign, score_mat)
                if drift_maps:
                    preds_oracle_raw = _preds_from_assignment(oracle_assign, raw_mat)
                    preds_oracle_aligned = _preds_from_assignment(
                        oracle_assign, aligned_mat,
                    )
                    n_oracle_raw += int((preds_oracle_raw == labels_np).sum())
                    n_oracle_aligned += int((preds_oracle_aligned == labels_np).sum())

                routed_ok = assigned == true_stages
                n_within_stage += int(routed_ok.sum())
                n_within_correct += int((preds[routed_ok] == labels_np[routed_ok]).sum())

                n_correct        += int((preds == labels_np).sum())
                n_oracle_correct += int((preds_oracle == labels_np).sum())
                n_total          += N

                for true_lbl, pred_lbl in zip(labels_np.tolist(), preds.tolist()):
                    if true_lbl == pred_lbl:
                        continue
                    true_stage = stage_class_map.get(true_lbl, -1)
                    pred_stage = stage_class_map.get(pred_lbl, -1)
                    if true_stage == pred_stage:
                        n_intra += 1
                    else:
                        n_inter += 1
                    confusion_by_pred_stage[pred_stage] += 1

        n_wrong = n_total - n_correct
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        oracle_accuracy = n_oracle_correct / n_total if n_total > 0 else 0.0
        routing_accuracy = n_routing_correct / n_total if n_total > 0 else 0.0
        within_stage_accuracy = (
            n_within_correct / n_within_stage if n_within_stage > 0 else 0.0
        )
        routing_gap = oracle_accuracy - accuracy
        intra_rate = n_intra / n_wrong if n_wrong > 0 else 0.0
        inter_rate = n_inter / n_wrong if n_wrong > 0 else 0.0

        space_tag = "plane" if use_plane_scores else "full"
        route_tag = routing
        src_tag   = (plane_source or "features").lower() if use_plane_scores else space_tag
        mode_label = f"{route_tag}+{space_tag}" + (
            f"+{src_tag}" if use_plane_scores and plane_source else ""
        )
        if use_split_hulls:
            mode_label += "+score(dynamic)"
        if drift_maps:
            mode_label += f"+drift({drift_align_mode}"
            if drift_routing_only:
                mode_label += ",route"
            mode_label += ")"
        result = {
            "accuracy":                    accuracy,
            "oracle_accuracy":             oracle_accuracy,
            "routing_accuracy":            routing_accuracy,
            "within_stage_accuracy":       within_stage_accuracy,
            "routing_gap":                 routing_gap,
            "n_correct":                   n_correct,
            "n_oracle_correct":            n_oracle_correct,
            "n_routing_correct":           n_routing_correct,
            "n_within_stage":              n_within_stage,
            "n_wrong":                     n_wrong,
            "n_total":                     n_total,
            "n_routed_by_stage":           dict(n_routed_by_stage),
            "n_oracle_by_stage":           dict(n_oracle_by_stage),
            "intra_stage_confusion_rate":  intra_rate,
            "inter_stage_confusion_rate":  inter_rate,
            "confusion_by_pred_stage":     dict(confusion_by_pred_stage),
            "score_key":                   score_key,
            "subspace_variance_threshold": subspace_variance_threshold,
            "plane_scoring":               use_plane_scores,
            "score_space":                 space_tag,
            "plane_source":                src_tag,
            "stage_routing":               stage_routing,
            "drift_align":                 bool(drift_maps),
            "drift_align_mode":            drift_align_mode if drift_maps else None,
            "drift_routing_only":          drift_routing_only if drift_maps else None,
            "score_hulls_dynamic":         use_split_hulls,
            "n_drift_maps":                len(drift_maps or {}),
            "oracle_raw_accuracy":         (
                n_oracle_raw / n_total if drift_maps and n_total > 0 else None
            ),
            "oracle_aligned_accuracy":     (
                n_oracle_aligned / n_total if drift_maps and n_total > 0 else None
            ),
            "scoring_mode":                mode_label,
        }

        if verbose:
            routed_str = "  ".join(
                f"s{s}:{c}" for s, c in sorted(n_routed_by_stage.items())
            )
            print(
                f"    [StagedEval {mode_label}]  "
                f"acc={accuracy:.2%}  oracle={oracle_accuracy:.2%}  "
                f"route={routing_accuracy:.2%}  within={within_stage_accuracy:.2%}  "
                f"gap={routing_gap:+.2%}"
            )
            if drift_maps and n_total > 0:
                o_raw = n_oracle_raw / n_total
                o_aln = n_oracle_aligned / n_total
                print(
                    f"    [StagedEval drift-diag]  oracle_raw={o_raw:.2%}  "
                    f"oracle_aligned={o_aln:.2%}  "
                    f"(routing_only={'on' if drift_routing_only else 'off'})"
                )
            print(
                f"    [StagedEval {mode_label}]  wrong={n_wrong}  "
                f"intra={n_intra} ({intra_rate:.1%})  inter={n_inter} ({inter_rate:.1%})"
            )
            print(
                f"    [StagedEval {mode_label}]  routed=[{routed_str}]"
                + (
                    "  wrong-pred stages → "
                    + "  ".join(
                        f"s{s}:{c}" for s, c in sorted(confusion_by_pred_stage.items())
                    )
                    if confusion_by_pred_stage else ""
                )
            )

        return result

    def debug_hull_confusion(
        self,
        backbone: "nn.Module",
        test_loader,
        hulls: dict,
        device,
        current_stage_classes: list,
        stage_class_map: dict,
        score_key: str = "cosine",
        verbose: bool = True,
    ) -> dict:
        """
        Diagnose whether hull mispredictions come from within the same stage
        (intra-stage confusion) or from a past stage (inter-stage confusion).

        For each test sample the predicted class is found via argmax of hull
        scores.  A wrong prediction is labelled:
          - "intra_stage"  if pred_class belongs to current_stage_classes
          - "inter_stage"  if pred_class belongs to a different (past) stage

        Parameters
        ----------
        current_stage_classes : class IDs belonging to the stage being tested
        stage_class_map       : {class_id: stage_idx} for every known class
        score_key             : score from score_all() used for argmax

        Returns
        -------
        dict with keys: accuracy, n_correct, n_wrong, n_total,
            intra_stage_confusion_rate, inter_stage_confusion_rate,
            confusion_by_predicted_stage (stage_idx -> count),
            per_sample (list of per-example dicts for fine inspection).
        """
        backbone.eval()
        all_classes = list(hulls.keys())
        current_stage_set = set(current_stage_classes)

        n_correct = 0
        n_intra   = 0
        n_inter   = 0
        n_total   = 0
        confusion_by_predicted_stage: dict = defaultdict(int)
        per_sample = []

        with torch.no_grad():
            for imgs, labels in test_loader:
                feats     = backbone(imgs.to(device)).cpu().numpy()
                labels_np = labels.numpy()

                per_class = [hulls[name].score_all(feats) for name in all_classes]
                mat       = np.stack([pc[score_key] for pc in per_class], axis=1)
                best_idx  = np.argmax(mat, axis=1)
                preds     = np.array([int(all_classes[i]) for i in best_idx])

                for true_lbl, pred_lbl in zip(labels_np.tolist(), preds.tolist()):
                    true_stage = stage_class_map.get(true_lbl, -1)
                    pred_stage = stage_class_map.get(pred_lbl, -1)
                    correct    = true_lbl == pred_lbl

                    if correct:
                        n_correct += 1
                        confusion_type = "correct"
                    elif pred_lbl in current_stage_set:
                        n_intra += 1
                        confusion_type = "intra_stage"
                        confusion_by_predicted_stage[pred_stage] += 1
                    else:
                        n_inter += 1
                        confusion_type = "inter_stage"
                        confusion_by_predicted_stage[pred_stage] += 1

                    per_sample.append({
                        "true":           true_lbl,
                        "pred":           pred_lbl,
                        "true_stage":     true_stage,
                        "pred_stage":     pred_stage,
                        "correct":        correct,
                        "confusion_type": confusion_type,
                    })
                    n_total += 1

        n_wrong    = n_total - n_correct
        accuracy   = n_correct / n_total   if n_total > 0 else 0.0
        intra_rate = n_intra   / n_wrong   if n_wrong  > 0 else 0.0
        inter_rate = n_inter   / n_wrong   if n_wrong  > 0 else 0.0

        result = {
            "accuracy":                     accuracy,
            "n_correct":                    n_correct,
            "n_wrong":                      n_wrong,
            "n_total":                      n_total,
            "intra_stage_confusion_rate":   intra_rate,
            "inter_stage_confusion_rate":   inter_rate,
            "confusion_by_predicted_stage": dict(confusion_by_predicted_stage),
            "per_sample":                   per_sample,
        }

        if verbose:
            print(
                f"    [HullConfusion]  acc={accuracy:.2%}  wrong={n_wrong}  "
                f"intra={n_intra} ({intra_rate:.1%})  inter={n_inter} ({inter_rate:.1%})"
            )
            if confusion_by_predicted_stage:
                breakdown = "  ".join(
                    f"stage {s}: {c}"
                    for s, c in sorted(confusion_by_predicted_stage.items())
                )
                print(f"    [HullConfusion]  wrong-pred stage breakdown → {breakdown}")

        return result

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


def _select_replay_indices(
    hull,
    n_images:    int,
    target:      int,
    fill_to_cap: bool,
    features:    Optional[np.ndarray] = None,
    use_fps_fill: bool = True,
) -> list:
    """
    Choose replay-buffer indices for one class.

    Always includes hull extreme-ray indices first.  The fill strategy:

    ``use_fps_fill=True`` (default) and ``features`` provided:
        Fill remaining slots via farthest-point sampling (FPS) on the L2-
        normalised feature vectors — maximises angular coverage of the class
        distribution in the replay buffer, giving the dynamic hull better
        support when rebuilt later.

    Fallback (``use_fps_fill=False`` or no features):
        Evenly-spaced indices from the un-selected pool (original behaviour).
    """
    target = min(max(target, 0), n_images)
    if target == 0:
        return []

    if hull.extreme_rays_index is not None and len(hull.extreme_rays_index) > 0:
        selected = [
            int(i) for i in hull.extreme_rays_index
            if 0 <= int(i) < n_images
        ]
        selected = list(dict.fromkeys(selected))
    else:
        selected = []

    if not fill_to_cap:
        return selected[:target]

    if len(selected) >= target:
        return selected[:target]

    need  = target - len(selected)
    taken = set(selected)
    rest  = [i for i in range(n_images) if i not in taken]

    if not rest:
        return selected[:target]

    # ── FPS fill: maximise angular spread of stored exemplars ────────────────
    if use_fps_fill and features is not None and len(rest) > 1:
        try:
            from sklearn.preprocessing import normalize as _norm
            import numpy as _np

            F_rest = _norm(
                _np.asarray([features[i] for i in rest], dtype=np.float32),
                axis=1,
            )  # (|rest|, D)

            # Seed from the centroid of already-selected rays (if any exist)
            if selected:
                F_sel = _norm(
                    _np.asarray([features[i] for i in selected], dtype=np.float32),
                    axis=1,
                )
                seed_vec = F_sel.mean(axis=0)
            else:
                seed_vec = F_rest.mean(axis=0)
            seed_vec = seed_vec / (_np.linalg.norm(seed_vec) + 1e-12)

            local_sel = [int(_np.argmax(F_rest @ seed_vec))]
            min_ang = _np.full(len(F_rest), _np.inf, dtype=_np.float64)

            # Also account for distance from already-stored extreme rays
            if selected:
                for f in F_sel:
                    cos = _np.clip(F_rest @ f, -1.0, 1.0)
                    _np.minimum(min_ang, _np.arccos(cos), out=min_ang)

            for _ in range(min(need, len(F_rest)) - 1):
                s = F_rest[local_sel[-1]]
                cos = _np.clip(F_rest @ s, -1.0, 1.0)
                _np.minimum(min_ang, _np.arccos(cos), out=min_ang)
                min_ang[_np.array(local_sel)] = -1.0
                nxt = int(_np.argmax(min_ang))
                if min_ang[nxt] <= 1e-8:
                    break
                local_sel.append(nxt)

            fps_indices = [rest[j] for j in local_sel]
            selected.extend(fps_indices[:need])
            return selected[:target]
        except Exception:
            pass  # fall through to uniform fill on any error

    # ── uniform fill (original behaviour) ────────────────────────────────────
    step = max(1, len(rest) // need)
    selected.extend(rest[::step][:need])
    return selected[:target]


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


class ArcFaceHead(nn.Module):
    """
    Incremental ArcFace head with learnable, L2-normalised class-centre weights.

    During training pass `labels` to apply the additive angular margin; during
    inference omit `labels` to get raw scaled cosine logits.
    """
    def __init__(self, in_features: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(0, in_features))

    def add_classes(self, num_new_classes: int, device: torch.device):
        new_w = F.normalize(torch.randn(num_new_classes, self.in_features), p=2, dim=1).to(device)
        if self.weight.numel() == 0:
            self.weight = nn.Parameter(new_w)
        else:
            self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=0))

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features, p=2, dim=1),
                          F.normalize(self.weight, p=2, dim=1))
        if labels is None:
            return cosine * self.s
        sine = torch.sqrt(1.0 - cosine.pow(2)).clamp(0, 1)
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


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


def fit_drift_map_from_pairs(
    X_old: np.ndarray,
    X_new: np.ndarray,
    method: str = "procrustes",
    ridge: float = 1e-3,
    sample_weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Fit a drift map from paired feature rows (same images, old vs new backbone).

    Returns inverse map fields for ``align_features_to_hull_space``:
    x̃ = normalize((x − μ_new) @ A_inv_T + μ_old).
    """
    X_old = np.asarray(X_old, dtype=np.float64)
    X_new = np.asarray(X_new, dtype=np.float64)
    if X_old.shape != X_new.shape or X_old.ndim != 2 or X_old.shape[0] < 2:
        raise ValueError(
            f"fit_drift_map_from_pairs needs ≥2 paired rows; got {X_old.shape}."
        )

    w = np.ones(X_old.shape[0], dtype=np.float64)
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        w = np.maximum(w, 1e-8)
        w = w / w.sum()

    mu_old = (w[:, None] * X_old).sum(axis=0)
    mu_new = (w[:, None] * X_new).sum(axis=0)
    O = X_old - mu_old
    N = X_new - mu_new
    D = X_old.shape[1]

    if method == "procrustes":
        sw = np.sqrt(w)[:, None]
        U, _S, Vt = np.linalg.svd((sw * O).T @ (sw * N), full_matrices=False)
        A = Vt.T @ U.T
    elif method == "ridge_affine":
        gram = O.T @ (w[:, None] * O) + ridge * np.eye(D)
        cross = O.T @ (w[:, None] * N)
        A = np.linalg.solve(gram, cross).T
    else:
        raise ValueError(
            f"Unknown pair drift method '{method}'. "
            "Choose 'procrustes' or 'ridge_affine'."
        )

    try:
        A_inv_T = np.linalg.inv(A).T
    except np.linalg.LinAlgError:
        A_inv_T = np.linalg.pinv(A).T

    aligned = (X_new - mu_new) @ A_inv_T + mu_old
    aligned = normalize(aligned, axis=1)
    target  = normalize(X_old, axis=1)
    residual = float(1.0 - np.mean(np.sum(aligned * target, axis=1)))

    return {
        "A":         A.astype(np.float32),
        "A_inv_T":   A_inv_T.astype(np.float32),
        "mu_old":    mu_old.astype(np.float32),
        "mu_new":    mu_new.astype(np.float32),
        "n_pairs":   int(X_old.shape[0]),
        "residual":  residual,
        "method":    method,
    }


def _extract_paired_replay_features(
    backbone: nn.Module,
    replay_buffer: "ReplayBuffer",
    class_ids: List[int],
    feature_snapshot: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paired (old snapshot, current backbone) replay features per class."""
    X_old_parts: list = []
    X_new_parts: list = []
    backbone.eval()
    with torch.no_grad():
        for cls_id in class_ids:
            snap = feature_snapshot.get(cls_id)
            if snap is None or cls_id not in replay_buffer.buffer:
                continue
            class_imgs = [item[0] for item in replay_buffer.buffer[cls_id]]
            if not class_imgs:
                continue
            new_feats: list = []
            for start in range(0, len(class_imgs), batch_size):
                chunk = torch.stack(class_imgs[start : start + batch_size]).to(device)
                new_feats.append(backbone(chunk).cpu().numpy())
            new_feats = np.concatenate(new_feats, axis=0)
            n = min(len(snap), len(new_feats))
            if n < 1:
                continue
            X_old_parts.append(np.asarray(snap[:n], dtype=np.float64))
            X_new_parts.append(np.asarray(new_feats[:n], dtype=np.float64))
    if not X_old_parts:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.vstack(X_old_parts), np.vstack(X_new_parts)


def _collect_drift_alignment_pairs(
    backbone: nn.Module,
    replay_buffer: "ReplayBuffer",
    class_ids: List[int],
    feature_snapshot: Dict[int, np.ndarray],
    device: torch.device,
    hulls: Optional[Dict[str, "ConicHull"]] = None,
    ray_weight: float = 3.0,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Replay pairs plus weighted extreme-ray anchor pairs for a stage/class set.
    """
    X_old, X_new = _extract_paired_replay_features(
        backbone, replay_buffer, class_ids, feature_snapshot, device, batch_size,
    )
    weights: Optional[np.ndarray] = None
    if X_old.shape[0] > 0:
        weights = np.ones(X_old.shape[0], dtype=np.float64)

    if hulls is None or ray_weight <= 0:
        return X_old, X_new, weights

    ray_old: list = []
    ray_new: list = []
    backbone.eval()
    with torch.no_grad():
        for cls_id in class_ids:
            cls_name = str(cls_id)
            hull = hulls.get(cls_name)
            if hull is None or hull.extreme_rays_ is None:
                continue
            rays = hull.extreme_rays_
            idxs = getattr(hull, "extreme_rays_index", None)
            if cls_id not in replay_buffer.buffer or idxs is None:
                continue
            class_imgs = [item[0] for item in replay_buffer.buffer[cls_id]]
            if not class_imgs:
                continue
            all_new: list = []
            for start in range(0, len(class_imgs), batch_size):
                chunk = torch.stack(class_imgs[start : start + batch_size]).to(device)
                all_new.append(backbone(chunk).cpu().numpy())
            all_new = np.concatenate(all_new, axis=0)
            for i, ray_idx in enumerate(idxs):
                if ray_idx < 0 or ray_idx >= len(all_new):
                    continue
                ray_old.append(rays[i])
                ray_new.append(all_new[ray_idx])

    if ray_old:
        ro = normalize(np.asarray(ray_old, dtype=np.float64), axis=1)
        rn = normalize(np.asarray(ray_new, dtype=np.float64), axis=1)
        rw = np.full(len(ro), float(ray_weight), dtype=np.float64)
        if X_old.shape[0] == 0:
            return ro, rn, rw
        X_old = np.vstack([X_old, ro])
        X_new = np.vstack([X_new, rn])
        weights = np.concatenate([weights, rw])

    return X_old, X_new, weights


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
        
def translate_hulls(
    hulls:     Dict[str, "ConicHull"],
    old_stats: Dict,
    new_stats: Dict,
    alpha:     float = 0.2  # Momentum factor to handle 20-sample noise
) -> int:
    n_translated = 0
    all_deltas = []

    # 1. First, collect all available deltas to find the "Global Drift"
    for cls_id in old_stats:
        if cls_id in new_stats:
            all_deltas.append(new_stats[cls_id]["mean"] - old_stats[cls_id]["mean"])
    
    if not all_deltas:
        return 0
    
    # Global drift is much more stable than per-class drift for N=20
    global_delta = np.mean(all_deltas, axis=0)

    for cls_str, hull in hulls.items():
        if hull.extreme_rays_ is None:
            continue

        cls_id = int(cls_str)
        
        # Calculate Class-Specific Drift
        if cls_id in old_stats and cls_id in new_stats:
            local_delta = new_stats[cls_id]["mean"] - old_stats[cls_id]["mean"]
            # Blend Local (20%) and Global (80%) for maximum stability
            delta = 0.2 * local_delta + 0.8 * global_delta
        else:
            # If we don't have buffer for this class, use global drift as a best guess
            delta = global_delta

        # Apply Momentum-based update
        # This prevents the Hull from 'jumping' too far based on one bad batch
        translated = hull.extreme_rays_ + (alpha * delta)
        
        # Re-normalize to unit sphere
        norms = np.linalg.norm(translated, axis=1, keepdims=True)
        hull.extreme_rays_ = translated / np.maximum(norms, 1e-12)

        hull.pca_ = None 
        n_translated += 1

    return n_translated

def update_head_weights_analytically(
    head: nn.Module,
    A: np.ndarray,
    new_stats: Dict,
    old_class_ids: List[int],
    device: torch.device,
    renormalize: bool = True,
    magnitude_preserving: bool = False,
) -> None:
    """
    Updates old classifier-head weights. 
    Uses EXACT new centroids if available in the memory buffer (new_stats).
    Falls back to the Affine drift matrix A only for classes not in the buffer.
    """
    A_inv_T = np.linalg.inv(A).T
    A_inv_T_t = torch.tensor(A_inv_T, dtype=head.weight.dtype, device=device)

    with torch.no_grad():
        for class_id in old_class_ids:
            if class_id >= head.weight.shape[0]:
                continue
            
            w_old = head.weight[class_id]
            norm_old = w_old.norm()

            # --- THE FIX: Direct Centroid Assignment ---
            if class_id in new_stats:
                # We have the exact new mean, so bypass the linear 'A' assumption completely
                new_mean = torch.tensor(
                    new_stats[class_id]["mean"], 
                    dtype=head.weight.dtype, 
                    device=device
                )
                w_adapted = new_mean
                
                # If using magnitude preserving, we want the direction of the new mean, 
                # but the scale of the old weight.
                if magnitude_preserving and w_adapted.norm() > 1e-8:
                    w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0) * norm_old
                elif renormalize:
                    w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0)
                else:
                    w_new = w_adapted
                    
            # --- FALLBACK: Affine Transformation ---
            else:
                # We don't have buffered data for this class, so we must guess 
                # its new location using the global drift matrix A
                w_adapted = A_inv_T_t @ w_old

                if magnitude_preserving and w_adapted.norm() > 1e-8:
                    w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0) * norm_old
                elif renormalize:
                    w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0)
                else:
                    w_new = w_adapted

            # Apply the update
            head.weight[class_id] = w_new


# def update_head_weights_analytically(
#     head: nn.Module,
#     A: np.ndarray,
#     old_class_ids: List[int],
#     device: torch.device,
#     renormalize: bool = True,
#     magnitude_preserving: bool = False,
# ) -> None:
#     """
#     Corrects old classifier-head weights to account for feature-space drift.

#     **Math.**  If features transform as  x_new ≈ A x_old,  the equivalent
#     decision hyperplane satisfies:

#         w_new = A^{-T} w_old

#     *Derivation.*  The linear score  w^T x  must be invariant:
#         w_new^T (A x_old) = w_old^T x_old
#         (A^T w_new)       = w_old
#         w_new             = A^{-T} w_old

#     For orthogonal A (Procrustes output): A^{-T} = A, so the update is a
#     pure rotation with no matrix inversion required.

#     **Magnitude-preserving mode.**  When A contains a contraction (features
#     cluster more tightly), A^{-T} is an expansion.  Without correction, the
#     adapted old weights grow in norm and dominate the softmax purely through
#     logit volume, not angular advantage.  Setting ``magnitude_preserving=True``
#     decouples direction from scale:

#         w_final = (w_adapted / ‖w_adapted‖) × ‖w_old‖

#     This keeps the directional benefit of A^{-T} while holding the confidence
#     level of each old class at its pre-drift value.

#     Parameters
#     ----------
#     head                 : Module with a ``.weight`` tensor (num_classes, D).
#     A                    : (D, D) drift matrix from ``estimate_affine_drift``.
#     old_class_ids        : Indices of old-class weights to update.
#     device               : Computation device.
#     renormalize          : Project updated weight onto the unit sphere.
#                            Use True for ArcFace heads, False for plain linear.
#                            Ignored when ``magnitude_preserving=True``.
#     magnitude_preserving : If True, restore ‖w_old‖ after applying A^{-T},
#                            decoupling the rotation from the expansion/contraction
#                            induced by A.  Recommended when the drift matrix is
#                            ill-conditioned or when old and new logit scales diverge.
#     """
#     A_inv_T   = np.linalg.inv(A).T                                # (D, D)
#     A_inv_T_t = torch.tensor(A_inv_T, dtype=head.weight.dtype, device=device)

#     with torch.no_grad():
#         for class_id in old_class_ids:
#             if class_id >= head.weight.shape[0]:
#                 continue
#             w_old     = head.weight[class_id]                      # (D,)
#             norm_old  = w_old.norm()
#             w_adapted = A_inv_T_t @ w_old                          # (D,)

#             if magnitude_preserving and w_adapted.norm() > 1e-8:
#                 # Use direction of A^{-T} w_old, scale of w_old
#                 w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0) * norm_old
#             elif renormalize:
#                 w_new = F.normalize(w_adapted.unsqueeze(0), p=2, dim=1).squeeze(0)
#             else:
#                 w_new = w_adapted

#             head.weight[class_id] = w_new


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


def _fit_pca_on_features(
    feature_dict: Dict[str, np.ndarray],
    n_components = None
) -> tuple:
    """
    Fit PCA on all features in feature_dict (thin SVD).

    Parameters
    ----------
    feature_dict  : {cls_id: (N_i, D)} feature arrays per class
    n_components  : number of PCA dimensions to keep.  None (default) keeps
                    all valid components: min(N-1, D).

    Returns
    -------
    (pca_mean, pca_comps) where pca_mean is (D,) and pca_comps is (k, D)
    with k = min(n_components, N-1, D) and each row is a principal component
    (unit vector).
    """
    all_feats = np.concatenate(list(feature_dict.values()), axis=0)  # (N, D)
    pca_mean  = all_feats.mean(axis=0)
    centered  = all_feats - pca_mean
    # Thin SVD gives at most min(N, D) components; drop the last to avoid
    # the degenerate zero singular value when N ≤ D.
    _, _, Vt  = np.linalg.svd(centered, full_matrices=False)         # (k, D)
    k = min(all_feats.shape[0] - 1, all_feats.shape[1])
    if n_components is not None:
        k = min(k, n_components)
    return pca_mean, Vt[:k]


def _project_stats_to_pca(
    stats: Dict,
    pca_mean: np.ndarray,
    pca_comps: np.ndarray,
) -> Dict:
    """
    Project full-space per-class statistics into PCA space.

    Parameters
    ----------
    stats     : {cls_id: {"mean": (D,), "cov": (D,D), "n": int}}
    pca_mean  : (D,) global mean used when fitting the PCA
    pca_comps : (k, D) principal components (rows are unit vectors)

    Returns
    -------
    Projected stats with mean in (k,) and cov in (k, k).
    """
    projected = {}
    for cls_id, s in stats.items():
        mean_c   = s["mean"] - pca_mean            # centre in same frame
        mean_pca = pca_comps @ mean_c              # (k,)
        cov_pca  = pca_comps @ s["cov"] @ pca_comps.T  # (k, k)
        projected[cls_id] = {"mean": mean_pca, "cov": cov_pca, "n": s["n"]}
    return projected


def _lift_pca_drift(
    A_pca: np.ndarray,
    pca_comps: np.ndarray,
) -> np.ndarray:
    """
    Lift a (k, k) drift matrix estimated in PCA space to full (D, D) space.

    The PCA subspace contribution is  V^T A_pca V, where V = pca_comps^T.
    The orthogonal complement is left as the identity, so directions unseen
    by the PCA are assumed to be stationary.

    Parameters
    ----------
    A_pca     : (k, k) drift matrix estimated in PCA space
    pca_comps : (k, D) principal components used to project into PCA space

    Returns
    -------
    A_full : (D, D) drift matrix in original feature space
    """
    D      = pca_comps.shape[1]
    A_sub  = pca_comps.T @ A_pca @ pca_comps           # (D, D) — subspace part
    A_full = A_sub + (np.eye(D) - pca_comps.T @ pca_comps)  # identity on complement
    return A_full


def mahalanobis_cov_loss(
    new_features: torch.Tensor,
    old_features: torch.Tensor,
    labels: torch.Tensor,
    class_cov_inv: Dict[int, torch.Tensor],
    max_pairs_per_class: int = 64,
) -> torch.Tensor:
    """
    Covariance calibration loss (Eq. 10 from the GKEAL paper).

    For replay samples, compute pairwise Mahalanobis distances under the old
    covariance Σ^{t-1}_c.  The loss penalises the absolute difference in those
    distances between the old and new networks:

        L_cov = Σ_c Σ_{i,j} |dM(φ_new(xi), φ_new(xj), Σ_old_c)
                               - dM(φ_old(xi), φ_old(xj), Σ_old_c)|

    Using the Cholesky factor L (s.t. Σ = L Lᵀ):
        dM(x, y, Σ) = ‖L⁻¹(x − y)‖₂

    so the Mahalanobis distance reduces to a standard L2 norm in the whitened
    space, which is differentiable w.r.t. the embeddings.

    Parameters
    ----------
    new_features        : (N, D) embeddings from the current backbone
    old_features        : (N, D) embeddings from the frozen old backbone
    labels              : (N,) integer class IDs matching the replay batch
    class_cov_inv       : {class_id: L_inv (D, D) on device}
                          L_inv is the inverse lower-triangular Cholesky factor
                          of the old class covariance.
    max_pairs_per_class : cap on random pairs sampled per class per step.
    """
    device = new_features.device
    total = new_features.new_zeros(())
    n_terms = 0

    unique_classes = labels.unique()
    for c in unique_classes:
        c_int = int(c.item())
        if c_int not in class_cov_inv:
            continue
        L_inv = class_cov_inv[c_int]  # (D, D)

        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue

        # Generate pairs; cap to avoid O(N²) cost.
        n = idx.numel()
        if n * (n - 1) // 2 <= max_pairs_per_class:
            # All pairs
            ii, jj = torch.triu_indices(n, n, offset=1)
        else:
            # Random pairs
            rand = torch.randint(0, n, (max_pairs_per_class, 2), device=device)
            # Ensure i != j
            rand[:, 1] = (rand[:, 0] + 1 + torch.randint(0, n - 1, (max_pairs_per_class,), device=device)) % n
            ii, jj = rand[:, 0], rand[:, 1]

        fi_new = new_features[idx[ii]]  # (P, D)
        fj_new = new_features[idx[jj]]
        fi_old = old_features[idx[ii]]
        fj_old = old_features[idx[jj]]

        diff_new = fi_new - fj_new  # (P, D)
        diff_old = fi_old - fj_old

        # dM = ‖L_inv @ diff^T‖_2 = ‖diff @ L_inv^T‖_2  (row-wise)
        dm_new = (diff_new @ L_inv.T).norm(dim=1)  # (P,)
        dm_old = (diff_old @ L_inv.T).norm(dim=1)

        total = total + (dm_new - dm_old).abs().mean()
        n_terms += 1

    if n_terms == 0:
        return new_features.new_zeros(())
    return total / n_terms


def mean_shift_compensation(
    backbone_old: nn.Module,
    backbone_new: nn.Module,
    replay_buffer: "ReplayBuffer",
    old_stats: Dict,
    device: torch.device,
    batch_size: int = 64,
    sigma: float = 1.0,
) -> Dict[int, np.ndarray]:
    """
    Estimate the per-class mean shift due to backbone drift (Eq. 6–7).

    For each old class c with replay samples {xi}:
        wi = exp(−‖φ_old(xi) − μ_old_c‖² / 2σ²)
        Δ̂μ_c = Σ_i wi·(φ_new(xi) − φ_old(xi)) / Σ_i wi

    Samples near the old class centroid receive higher weight, making the
    estimate more robust to outliers.

    Returns a dict {class_id: estimated_mean_shift (D,)}.
    """
    backbone_old.eval()
    backbone_new.eval()
    deltas: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        for cls_id, stats in old_stats.items():
            if cls_id not in replay_buffer.buffer or not replay_buffer.buffer[cls_id]:
                continue

            mu_old = stats["mean"]          # (D,)
            imgs_cpu = [item[0] for item in replay_buffer.buffer[cls_id]]
            if not imgs_cpu:
                continue

            phi_old_list, phi_new_list = [], []
            for start in range(0, len(imgs_cpu), batch_size):
                batch = torch.stack(imgs_cpu[start:start + batch_size]).to(device)
                phi_old_list.append(backbone_old(batch).cpu().numpy())
                phi_new_list.append(backbone_new(batch).cpu().numpy())

            phi_old = np.concatenate(phi_old_list, axis=0)  # (N, D)
            phi_new = np.concatenate(phi_new_list, axis=0)

            diff_to_mean = phi_old - mu_old[None]           # (N, D)
            sq_dist = (diff_to_mean ** 2).sum(axis=1)       # (N,)
            weights = np.exp(-sq_dist / (2.0 * sigma ** 2)) # (N,)
            w_sum = weights.sum()
            if w_sum < 1e-12:
                continue

            per_sample_drift = phi_new - phi_old             # (N, D)
            delta = (weights[:, None] * per_sample_drift).sum(axis=0) / w_sum
            deltas[cls_id] = delta

    return deltas


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
    Log-Determinant Volume Penalty (non-negative, batch-size invariant).

    Returns a collapse penalty in [0, ∞): near 0 when the batch is spread out,
    higher when features occupy a low-volume subspace.  Safe to add to the
    total loss alongside CE / NSR without driving the sum negative.
    """
    if features.shape[0] < 2:
        return features.new_zeros(())
    Z = features - features.mean(dim=0, keepdim=True)
    return covariance_collapse_penalty(Z, eps)


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


def orthogonal_centroid_loss(
    features:      torch.Tensor,
    old_centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonal Centroid Regularization Loss.

    Penalises new-task features for having any component in the direction of
    old-class centroids.  By minimising the squared cosine similarity between
    each new feature and every frozen old centroid, the backbone is forced to
    route new representations into directions that are orthogonal to the old
    feature subspace — making conic hulls naturally non-overlapping without
    any post-hoc transformation.

    Math
    ----
    Let  z_i = f(x_i) / ‖f(x_i)‖  (unit-normalised new feature)
    and  μ_c                        (unit-normalised old class centroid).

        L_ortho = (1/N) Σ_i Σ_c (z_i · μ_c)²

    Each term is the squared cosine similarity — it is zero when z_i ⊥ μ_c
    and one when they are collinear.  Summing over all old classes encourages
    the new features to avoid every old subspace direction simultaneously.

    Parameters
    ----------
    features      : (N, D) raw backbone output for the current new-class batch.
                    L2-normalisation is applied internally.
    old_centroids : (C_old, D) L2-normalised old-class mean feature vectors,
                    one row per class.  Build once per stage from _prev_stats
                    and keep frozen on device.

    Returns
    -------
    Scalar tensor — mean squared cosine across all (sample, old-class) pairs.
    Gradient flows through features; old_centroids are treated as constants.
    """
    z    = F.normalize(features, p=2, dim=1)          # (N, D)
    sims = z @ old_centroids.T                         # (N, C_old) cosine similarities
    return sims.pow(2).sum(dim=1).mean()               # scalar


def get_incremental_dataloaders(dataset_name="CIFAR100", classes_per_stage=10, batch_size=128, class_order=None):
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

    if class_order is None:
        class_order = list(range(total_classes))

    num_stages = total_classes // classes_per_stage
    stages = []

    train_targets = torch.tensor(train_ds.targets)
    test_targets = torch.tensor(test_ds.targets)

    for i in range(num_stages):
        stage_classes = class_order[i * classes_per_stage : (i + 1) * classes_per_stage]
        cls_tensor = torch.tensor(stage_classes)

        idx_train = torch.where(torch.isin(train_targets, cls_tensor))[0]
        idx_test = torch.where(torch.isin(test_targets, cls_tensor))[0]

        stages.append({
            "stage_id": i,
            "classes": stage_classes,
            "train_loader": DataLoader(Subset(train_ds, idx_train), batch_size=batch_size, shuffle=True),
            "test_loader": DataLoader(Subset(test_ds, idx_test), batch_size=batch_size, shuffle=False)
        })

    return stages, total_classes, len(train_ds)

def evaluate_stage(backbone, head, old_backbone, test_loader, device, class_to_idx=None):
    """Evaluates accuracy and average feature drift on a given test loader."""
    backbone.eval()
    head.eval()
    correct, total = 0, 0
    total_drift, drift_batches = 0.0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            features = backbone(imgs)

            if isinstance(head, (FixedConicHead, IncrementalConicHead, ArcFaceHead)):
                logits = head(features, labels=None)  # Eval mode: raw cosines
            else:
                logits = head(features)
            preds = logits.argmax(dim=1)

            # When class order is shuffled the head assigns sequential indices
            # (0, 1, 2, …) to classes in the order they are introduced, which
            # differs from the raw dataset label values.  Remap ground-truth
            # labels to head indices before comparing.  FixedConicHead
            # preallocates one slot per class ID so no remapping is needed there.
            if class_to_idx is not None and not isinstance(head, FixedConicHead):
                gt = torch.tensor(
                    [class_to_idx[int(l)] for l in labels.tolist()],
                    device=device, dtype=labels.dtype,
                )
            else:
                gt = labels

            correct += (preds == gt).sum().item()
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



def _distill_pointwise_and_relational(
    f_new: torch.Tensor,
    f_old: torch.Tensor,
    alpha: float,
    pointwise_weight: float,
    kl_divergence: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared L2 anchor + relational batch-structure term."""
    sim_matrix_new = torch.mm(f_new, f_new.T)
    sim_matrix_old = torch.mm(f_old, f_old.T)

    drift = F.pairwise_distance(f_new, f_old, p=2)
    loss_pointwise = torch.clamp(drift - alpha, min=0.0).mean()
    if not kl_divergence:
        loss_relational = F.mse_loss(sim_matrix_new, sim_matrix_old)
    else:
        temp = 0.05
        sim_matrix_new = sim_matrix_new / temp
        sim_matrix_old = sim_matrix_old / temp
        p_new = F.log_softmax(sim_matrix_new, dim=1)
        p_old = F.softmax(sim_matrix_old, dim=1)
        loss_relational = F.kl_div(p_new, p_old, reduction='batchmean') * (temp ** 2)

    return pointwise_weight * loss_pointwise, loss_relational


def calculate_distillation_loss(
    distill_type,
    new_features,
    old_features,
    alpha=0.1,
    pointwise_weight=0.2,
    predictor=None,
    kl_divergence=True,
    centroids=None,
    centroid_outward_weight=0.25,
):
    """
    Modular distillation function.
    - 'basic': L2 drift + relational KL distillation.
    - 'ssl': Teacher-Student predictive cosine similarity.
    - 'centroid_aware': Only penalize outward drift from class centroid (legacy).
    - 'hybrid': L2 + relational + outward centroid hinge (recommended).

      hybrid = pointwise L2(f_new, f_old) + relational KL
               + centroid_outward_weight * mean(ReLU(cos_old − cos_new))

      The L2 term pins tangential drift; the outward term allows inward motion.
    """
    if distill_type == "ssl":
        predicted_features = predictor(new_features)
        return F.mse_loss(predicted_features, old_features)

    elif distill_type in ("basic", "hybrid", "centroid_aware"):
        f_new = F.normalize(new_features, p=2, dim=1)
        f_old = F.normalize(old_features, p=2, dim=1)

        if distill_type == "centroid_aware":
            if centroids is None:
                raise ValueError("distill_type='centroid_aware' requires centroids tensor")
            cos_old = torch.sum(f_old * centroids, dim=1)
            cos_new = torch.sum(f_new * centroids, dim=1)
            return torch.clamp(cos_old - cos_new, min=0.0).mean()

        loss_pointwise, loss_relational = _distill_pointwise_and_relational(
            f_new, f_old, alpha, pointwise_weight, kl_divergence,
        )
        if distill_type == "basic":
            return loss_pointwise + loss_relational

        # hybrid
        if centroids is None:
            return loss_pointwise + loss_relational
        cos_old = torch.sum(f_old * centroids, dim=1)
        cos_new = torch.sum(f_new * centroids, dim=1)
        loss_outward = torch.clamp(cos_old - cos_new, min=0.0).mean()
        return loss_pointwise + loss_relational + centroid_outward_weight * loss_outward

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
    adaptive_hull_rays=False,          # scale rays/class down as classes grow
    hull_ray_budget=None,              # total ray budget; None → D (feature dim); only used when adaptive_hull_rays=True
    ray_diversity="fps",            # "spa" | "fps" | "hybrid" — ray selection strategy
    spa_oversample=3,                  # oversample multiplier for hybrid strategy
    learning_rate=1e-4,
    pointwise_distill_weight=0.2,
    memory_budget=0.04,                # 4% of the total dataset
    disable_memory_no_distill=False,   # skip replay buffer when distill_weight==0 and no rehearsal
    replay_fill_to_cap=True,           # fill replay to per_class_cap, not just hull rays
    fps_replay_fill=True,              # use FPS to select fill samples (True) vs uniform spacing (False)
    replay_samples_per_class=None,     # optional override; default uses memory budget cap
    use_analytical_head_update=True,   # Analytically correct old class weights after drift
    drift_method="covariance",         # "procrustes" | "covariance"
    head_update_method="affine",             # "affine" | "ortho_projected"
    head_update_magnitude_preserving=False,  # decouple A^{-T} rotation from scale expansion
    use_log_barrier_distill=False,      # Replace L2 KD with log-barrier margin loss
    barrier_weight=1.0,                # λ in  L = L_cls + λ * L_barrier
    distill_mode="hybrid",             # "basic" | "centroid_aware" | "hybrid"
    use_centroid_aware_distill=False,  # legacy: True → distill_mode="hybrid"
    centroid_outward_weight=0.25,      # outward hinge weight in hybrid mode
    head_type="conic",                 # "conic" | "linear" | "mlp"  (ablation)
    mlp_hidden_dim=1024,                # hidden width when head_type="mlp"
    mlp_dropout=0.0,                   # dropout probability when head_type="mlp"
    use_logdet_penalty=False,          # add log-det volume penalty to prevent collapse
    logdet_weight=0.05,                # λ_vol in  L = L_cls + λ_vol * L_vol
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
    translate_static_hulls=False,      # translate static hulls by per-class mean drift Δμ
    blocks_freeze=12,
    lora_rank=0,                       # > 0 → inject LoRA into ViT attention layers
    lora_alpha=1.0,                    # LoRA scaling factor α (effective scale = α/rank)
    lora_target_modules=None,          # layer suffixes to adapt, e.g. ["attn.qkv", "attn.proj"]
    lora_config="task_shared",         # "task_shared" | "task_specific" | "hybrid"
    evaluate_task_incremental=False,   # also evaluate each stage only against its own stage's cones
    use_null_space_projection=False,   # project LoRA grads onto null space of old extreme rays
    null_space_variance_threshold=0.99,  # fraction of SV² energy kept when building P_⊥
    debug_hull_confusion=False,        # print intra/inter stage confusion breakdown each eval
    use_ortho_centroid_loss=False,     # penalise new features for cosine overlap with old centroids
    lambda_ortho=1.0,                  # weight λ for L_ortho
    use_rehearsal_cls_loss=False,      # apply CE to a mixed new+replay batch each step
    rehearsal_cls_weight=1.0,          # weight applied to the rehearsal CE term
    use_cone_anchor=False,             # replay-free: anchor + margin-hinge vs frozen cones
    lambda_cone_stab=1.0,              # λ_s · AnchorLoss
    lambda_cone_marg=0.5,              # λ_m · MarginHinge
    cone_margin_deg=35.0,              # γ* for margin hinge (degrees)
    cone_anchor_batch=128,             # anchor vertices sampled per step
    cone_n_rays=None,                  # K rays/class (default: conic_hull_n_rays)
    cone_init_candidates=4096,         # FPS candidates for new-class weight init
    training_loss="ce",                  # "ce" | "geometric" — classification objective
    lambda_geo_attr=1.0,                 # λ_a · L_attr (own-cone alignment)
    lambda_geo_rep=0.1,                  # λ_r · L_rep  (cross-class ray repulsion)
    lambda_geo_marg=0.5,                 # λ_m · L_marg (feature vs other cones)
    geo_margin_deg=35.0,                 # γ* shared by all three geometric terms
    geo_kernel="hinge_sq",               # "hinge_sq" | "softplus" | "squared"
    geo_softplus_beta=10.0,              # β for softplus kernel
    geo_rep_max_pairs=50_000,            # subsample cap for O(C²K²) repulsion pairs
    evaluate_staged_hulls=False,       # run staged hull evaluation each stage
    staged_subspace_variance=0.95,     # SVD variance threshold for stage subspace
    staged_plane_scoring=False,        # legacy alias: True → score_space="plane"
    staged_score_space="full",         # "full" (standard hull) | "plane" (projected)
    staged_plane_routing="cascade",    # "cascade" | "hull_max_cal" | "hull_max" | ...
    staged_routing_cascade_percentile=10.0,  # accept stage s if max score ≥ this cal percentile
    staged_drift_align=True,           # align queries to hull build-time space per stage
    staged_drift_align_mode="dense_rays",  # centroid|dense|dense_rays|per_class
    staged_drift_pair_method="procrustes", # procrustes|ridge_affine (pair fitting)
    staged_drift_ridge=1e-3,
    staged_drift_ray_weight=3.0,
    staged_drift_routing_only=True,    # align for routing only; raw features for class scoring
    staged_score_hulls="dynamic",        # "static" | "dynamic" — hulls for within-stage scoring
    staged_plane_source="features",    # "features" (drift-aligned) | "hulls" | "both"
    staged_calibration_max_per_class=64,
    use_stage_confinement_loss=True,   # L_in at stage 0; +L_out vs prior stages from stage 1
    lambda_stage_in=1.0,               # weight for own-stage cap hinge
    lambda_stage_out=0.5,              # weight for leaking into other stage poles
    stage_confinement_on_replay=True,  # mild L_in on replay toward own stage pole
    lambda_stage_replay=0.15,          # weight for replay stage confinement
    stage_cap_deg=35.0,                # angular radius of each stage cap (degrees)
    stage_inter_cos_max=0.35,          # max cos to other stage poles (~70° separation)
    project_hulls_to_stage_cap=True,   # snap new hull rays into cap at fit time
    visualize_extreme_rays=False,      # plot PCA-3D cone visualisation after each stage
    use_kernel_hull=False,             # also build KernelConicHull each stage
    kernel_hull_type="spread",          # "spread" | "vmf" | "rbf" | "poly"
    kernel_gamma=1.0,                  # vmf κ / spread power / rbf γ
    rotate_kernel_into_null_space=False,   # project new kernel hull rays ⊥ to old-class subspace
    kernel_null_variance_threshold=0.99,   # SVD energy fraction kept when building old subspace
    use_shifted_hull=False,                # shifted-origin conic hull (per-task anchor on sphere)
    shifted_hull_strategy="orthogonal",    # origin assignment: "orthogonal"|"learned"|"axis"|"random"
    shifted_hull_n_rays=50,                # extreme rays for each shifted hull
    use_region_hulls=False,                # pre-allocate spherical caps + ray budget per class
    total_ray_budget=400,                  # global cap on extreme rays across all classes
    region_stage_cap_deg=35.0,             # angular radius of each stage's region on the sphere
    region_class_cap_deg=12.0,             # angular radius of each class cap within its stage
    region_class_spread_deg=8.0,           # spread of class anchors around the stage pole
    use_nsr_loss=False,                # Null-Space Repulsion + Anchor Loss
    nsr_weight_repel=1.0,              # λ for the repulsion term (new features ⊥ old rays)
    nsr_weight_anchor=0.5,             # λ for the anchor term (new features ∥ new hull rays)
    nsr_margin=0.0,                    # hinge: only penalise projections above this cosine²
    nsr_n_rays=20,                     # extreme rays in the per-epoch dynamic new-class hull
    nsr_ray_sample=64,                 # max ray images re-forwarded per step for the hull term
    nsr_ray_repel_weight=0.25,         # extra repulsion on re-forwarded hull rays (anchor skipped)
    nsr_weight_spread=0.05,            # null-space volume + inter-class spread (0 = off)
    nsr_subspace_from_hulls=False,       # repel from capped hull-ray subspace (better for kernel hulls)
    project_kernel_features=False,      # null-project new-class features before kernel hull fit
    use_layered_hull=False,            # per-stage hull built from backbone layer N (regressive cascade)
    layered_hull_strategy="linear",    # how to map stage→layer: "linear"|"last"|"first"|"cyclic"
    layered_hull_n_rays=50,            # extreme rays per class in the layered hull
    layered_hull_threshold=0.3,        # cascade threshold: stage k claims sample if score≥threshold
    use_feature_expansion=False,       # insert transformer encoder after backbone to expand features
    expansion_dim=512,                 # output dimensionality N of the expansion head
    expansion_n_tokens=8,              # number of tokens to split features into (expansion_dim % n must == 0)
    expansion_n_heads=4,               # attention heads per transformer encoder layer
    expansion_n_layers=1,              # number of stacked transformer encoder layers
    expansion_dropout=0.0,             # dropout inside the expansion transformer
    shuffle_class_order=False,         # randomly permute which classes appear in each stage
    class_order_seed=None,             # seed for the class-order shuffle (None = random)
    memory_loss_enabled=True,          # False → keep replay buffer for hull building but skip distill/rehearsal losses and memory forward passes
    # ── Covariance calibration loss (Eq. 10) ────────────────────────────────
    use_cov_loss=False,                # add Mahalanobis covariance calibration loss during training
    lambda_cov=1.0,                    # weight λ for L_cov
    cov_loss_eps=1e-4,                 # ridge added to Σ before Cholesky for numerical stability
    cov_max_pairs=64,                  # max random pairs per class per step
    # ── Mean shift compensation (Eq. 6–7) ───────────────────────────────────
    use_mean_shift_comp=False,         # post-training: estimate weighted mean drift and patch head weights
    mean_shift_sigma=1.0,              # Gaussian kernel bandwidth σ for mean shift weights
    # ── OOD Detection evaluation ─────────────────────────────────────────────
    evaluate_ood_hull=False,           # run OOD detection eval at end of every stage
    ood_calibrate_percentile=5.0,      # Nth pct of ID scores used as detection threshold
    ood_score_key="cosine",            # hull score used for OOD detection
    # ── Collaborative scoring (joint NNLS over all class dictionaries) ───────
    evaluate_collaborative_scoring=False,  # add collab_energy/residual/margin to per-stage table
    collaborative_lasso_lambda=0.0,        # L1 penalty; 0 = pure NNLS, try 1e-2 for sparser attribution
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_order = None
    if shuffle_class_order:
        import random as _random
        rng = _random.Random(class_order_seed)
        _n_cls = 100 if dataset_name == "CIFAR100" else 10
        class_order = list(range(_n_cls))
        rng.shuffle(class_order)
        print(f"[shuffle_class_order] seed={class_order_seed}  order={class_order}")

    stages, total_classes, total_data_size = get_incremental_dataloaders(
        dataset_name, classes_per_stage, batch_size, class_order=class_order
    )

    # Maps every class_id to the stage index it was introduced in.
    # Used by debug_hull_confusion to label intra- vs inter-stage errors.
    stage_class_map: dict = {}
    for _s_idx, _s_data in enumerate(stages):
        for _cls_id in _s_data["classes"]:
            stage_class_map[_cls_id] = _s_idx

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
        lora_config=lora_config,
    )

    if use_feature_expansion:
        _expansion_head = FeatureExpansionHead(
            in_dim        = backbone.num_features,
            expansion_dim = expansion_dim,
            n_tokens      = expansion_n_tokens,
            n_heads       = expansion_n_heads,
            n_layers      = expansion_n_layers,
            dropout       = expansion_dropout,
        ).to(device)
        backbone = FeatureExpansionBackbone(backbone, _expansion_head)
        print(f"  [FeatureExpansion] {backbone._backbone.num_features}D → {expansion_dim}D "
              f"(tokens={expansion_n_tokens}, heads={expansion_n_heads}, "
              f"layers={expansion_n_layers})")

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
    elif head_type == "arcface":
        head = ArcFaceHead(in_features=feat_dim).to(device)
    else:
        raise ValueError(f"Unknown head_type '{head_type}'. Choose 'conic', 'linear', 'mlp', or 'arcface'.")

    print(f"  Head: {head.__class__.__name__}")

    criterion = nn.CrossEntropyLoss()
    old_backbone = None
    P_old: Optional[torch.Tensor] = None          # built after each stage for ortho routing
    grad_projector: Optional[GradientProjector] = None  # null-space LoRA grad projection

    # Trackers for Continual Learning Metrics
    num_stages = len(stages)
    acc_matrix = np.zeros((num_stages, num_stages))
    acc_matrix_conic_hull_static          = np.zeros((num_stages, num_stages))
    acc_matrix_conic_hull_dynamic         = np.zeros((num_stages, num_stages))
    acc_matrix_conic_hull_staged          = np.zeros((num_stages, num_stages))
    acc_matrix_conic_hull_angular_margin  = np.zeros((num_stages, num_stages))
    acc_matrix_kernel_static              = np.zeros((num_stages, num_stages))
    acc_matrix_kernel_dynamic             = np.zeros((num_stages, num_stages))
    acc_matrix_shifted_hull               = np.zeros((num_stages, num_stages))
    acc_matrix_layered_hull               = np.zeros((num_stages, num_stages))
    drift_matrix = np.zeros((num_stages, num_stages))

    for name, param in backbone.patch_embed.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    for i in range(blocks_freeze): # Freeze first 10 blocks
        for name, param in backbone.blocks[i].named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
    
    
    memory_limit = memory_budget * total_data_size

    _memory_enabled = not (
        disable_memory_no_distill
        and distill_weight == 0
        and not use_rehearsal_cls_loss
    )
    if not _memory_enabled:
        print("  [Memory] Replay buffer disabled (disable_memory_no_distill=True, "
              "distill_weight=0, no rehearsal CE).")

    replay_buffer = ReplayBuffer(max_total_size=memory_limit, max_classes=total_classes)
    sampling_type = "balanced" # or "uniform"
    

    hull_mgr = HullManager(
        n_rays=conic_hull_n_rays,
        use_pca=False,
        adaptive_rays=adaptive_hull_rays,
        total_classes=total_classes,
        ray_budget=hull_ray_budget,
        ray_diversity=ray_diversity,
        spa_oversample=spa_oversample,
    )
    if adaptive_hull_rays:
        print(
            f"  [HullManager] adaptive ray budget enabled: "
            f"budget={hull_ray_budget if hull_ray_budget is not None else 'D (auto)'} / "
            f"{total_classes} classes → "
            f"≤{conic_hull_n_rays} rays/class cap"
        )

    _cone_n_rays = cone_n_rays if cone_n_rays is not None else conic_hull_n_rays
    cone_memory: Optional[ConeAnchorMemory] = (
        ConeAnchorMemory(n_rays=_cone_n_rays) if use_cone_anchor else None
    )
    if use_cone_anchor:
        print(
            f"  [ConeAnchor] Replay-free stability  "
            f"λ_stab={lambda_cone_stab}  λ_marg={lambda_cone_marg}  "
            f"margin={cone_margin_deg:.0f}°  K={_cone_n_rays} rays/class"
        )
        if use_rehearsal_cls_loss or distill_weight > 0:
            print(
                "  [ConeAnchor] Replay/distill disabled during training "
                "(stability from anchor + margin losses)."
            )

    _training_loss = training_loss.lower().strip()
    if _training_loss not in ("ce", "geometric"):
        raise ValueError(f"training_loss must be 'ce' or 'geometric', got {training_loss!r}")
    if _training_loss == "geometric":
        print(
            f"  [GeoLoss] CE replaced  "
            f"λ_a={lambda_geo_attr}  λ_r={lambda_geo_rep}  λ_m={lambda_geo_marg}  "
            f"γ*={geo_margin_deg:.0f}°  κ={geo_kernel}"
        )
        if use_cone_anchor and lambda_cone_marg > 0:
            print(
                "  [GeoLoss] Note: lambda_cone_marg overlaps with L_marg; "
                "set lambda_cone_marg=0 when using geometric loss."
            )

    region_registry: Optional[SphericalRegionRegistry] = None
    if use_region_hulls:
        region_registry = SphericalRegionRegistry(
            dim=feat_dim,
            total_classes=total_classes,
            num_stages=num_stages,
            classes_per_stage=classes_per_stage,
            total_ray_budget=total_ray_budget,
            stage_cap_deg=region_stage_cap_deg,
            class_cap_deg=region_class_cap_deg,
            class_spread_deg=region_class_spread_deg,
        )
        print(f"  [RegionHull] {region_registry.summary()}")

    # Fixed orthogonal stage poles for confinement loss + optional hull projection.
    stage_poles_np = build_orthogonal_stage_poles(num_stages, feat_dim, seed=0)
    stage_poles_t  = torch.tensor(stage_poles_np, dtype=torch.float32, device=device)
    cos_stage_cap  = float(np.cos(np.deg2rad(stage_cap_deg)))
    if use_stage_confinement_loss:
        print(f"  [StageConfine] {num_stages} orthogonal poles, cap={stage_cap_deg:.0f}°, "
              f"λ_in={lambda_stage_in} λ_out={lambda_stage_out} "
              f"replay={'on' if stage_confinement_on_replay else 'off'}"
              f"{f' λ_replay={lambda_stage_replay}' if stage_confinement_on_replay else ''} "
              f"(stage 0: L_in only; stage k≥1: L_in + repel stages 0..k-1)")

    # Shifted-origin conic hulls: one task anchor per stage, hulls accumulate.
    task_origin_registry: Optional[TaskOriginRegistry] = None
    shifted_static_hulls: Dict[str, ShiftedConicHull]  = {}
    if use_shifted_hull:
        task_origin_registry = TaskOriginRegistry(
            dim=feat_dim, strategy=shifted_hull_strategy
        )
        print(f"  [ShiftedHull] TaskOriginRegistry ready "
              f"(dim={feat_dim}, strategy={shifted_hull_strategy})")

    # Layered conic hull: one classifier that accumulates per-stage hulls.
    layered_classifier: Optional[LayeredConicHullClassifier] = None
    _num_backbone_layers: int = 0
    _stage_to_layer: Dict[int, int] = {}
    if use_layered_hull:
        if hasattr(backbone, "blocks"):
            _num_backbone_layers = len(backbone.blocks)
        elif hasattr(backbone, "layers"):
            _num_backbone_layers = len(backbone.layers)
        if _num_backbone_layers > 0:
            layered_classifier = LayeredConicHullClassifier(
                n_rays=layered_hull_n_rays, use_pca=False
            )
            _stage_to_layer = {
                si: get_layer_index_for_stage(
                    si, num_stages, _num_backbone_layers, layered_hull_strategy
                )
                for si in range(num_stages)
            }
            print(f"  [LayeredHull] Ready: {_num_backbone_layers} backbone layers, "
                  f"strategy={layered_hull_strategy!r}, "
                  f"stage→layer map: {_stage_to_layer}")

    # Per-stage superclass confusion history: list of dicts returned by
    # analyze_superclass_confusion, one entry per stage.
    superclass_confusion_history: List[Dict] = []

    # OOD detection results: one entry per stage (None for the last stage where
    # there are no future classes to use as OOD samples).
    ood_detection_history: List[Optional[dict]] = []

    # Analytical head-update state.  _prev_stats holds per-class statistics
    # computed with the backbone *before* the current stage's training so they
    # can be compared against post-training statistics to estimate feature drift.
    # _prev_pca stores (pca_mean, pca_comps) fitted on all buffer features at
    # snapshot time so that drift can be estimated in the lower-dimensional
    # PCA subspace before being lifted back to the full feature space.
    _prev_stats: Optional[Dict] = None
    _prev_pca:   Optional[tuple] = None  # (pca_mean: (D,), pca_comps: (k, D))
    stage_stats_snapshots: Dict[int, Dict] = {}
    stage_feature_snapshots: Dict[int, Dict[int, np.ndarray]] = {}

    # Maps each original dataset class-ID to its sequential position in the
    # incremental head.  When classes are introduced in non-identity order
    # (shuffle_class_order=True), this differs from the raw label value.
    # FixedConicHead pre-allocates one slot per class-ID so it never needs
    # remapping; all other heads use this map.
    class_to_idx: Dict[int, int] = {}

    for current_stage_idx, stage in enumerate(stages):
        print(f"\n{'='*50}")
        print(f"=== Training Stage {current_stage_idx} (Classes {stage['classes'][0]} to {stage['classes'][-1]}) ===")
        print(f"{'='*50}")
        num_new_classes = len(stage["classes"])
        if (
            use_cone_anchor
            and cone_memory is not None
            and current_stage_idx > 0
            and cone_memory.num_classes() > 0
        ):
            _init_dirs = torch.stack([
                init_new_direction(
                    cone_memory.get_cones_dict(),
                    feat_dim,
                    device,
                    cone_init_candidates,
                )
                for _ in range(num_new_classes)
            ])
            expand_head_with_directions(head, num_new_classes, _init_dirs, device)
        elif hasattr(head, "add_classes"):
            head.add_classes(num_new_classes, device)

        # Register the new classes: each gets the next sequential head index.
        # For FixedConicHead this stays empty (no remapping ever needed there).
        if not isinstance(head, FixedConicHead):
            _base = len(class_to_idx)
            for _i, _cls_id in enumerate(stage["classes"]):
                class_to_idx[_cls_id] = _base + _i

        train_loader = stage["train_loader"]

        # Helper: remap a tensor of raw dataset class-IDs to sequential head
        # indices.  For FixedConicHead class_to_idx is always empty so this
        # returns the original labels unchanged, which is correct.
        def _remap_labels(lbl: torch.Tensor, _map: Dict[int, int] = class_to_idx) -> torch.Tensor:
            if not _map:
                return lbl
            return torch.tensor(
                [_map[int(v)] for v in lbl.tolist()],
                device=lbl.device, dtype=lbl.dtype,
            )

        # For task-specific / hybrid LoRA, freeze the previous task's adapters
        # and allocate a fresh set before building the optimizer.
        if current_stage_idx > 0 and lora_rank > 0 and lora_config in ("task_specific", "hybrid"):
            n_advanced = advance_lora_task(backbone)
            print(f"  [LoRA] Advanced to task {current_stage_idx} "
                  f"({n_advanced} layer(s) updated, config={lora_config})")

        # Collect only parameters that require gradients.
        # When LoRA is active the original linear weights inside LoRALinear are
        # frozen; block-level freezing still suppresses adapters in frozen blocks.
        trainable_params = [p for p in backbone.parameters() if p.requires_grad]
        if isinstance(head, (IncrementalLinearHead, IncrementalMLPHead, ArcFaceHead)):
            trainable_params += list(head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        
        
        # Early stopping trackers
        best_loss = float('inf')
        patience_counter = 0

        # Build centroid lookup for hybrid / centroid_aware distillation.
        _effective_distill_mode = distill_mode
        if use_centroid_aware_distill and distill_mode == "basic":
            _effective_distill_mode = "hybrid"
        centroid_lookup: Optional[torch.Tensor] = None
        if _effective_distill_mode in ("centroid_aware", "hybrid") and _prev_stats:
            max_cls = max(_prev_stats.keys()) + 1
            ctable  = torch.zeros(max_cls, feat_dim, dtype=torch.float32)
            for cid, stats in _prev_stats.items():
                ctable[cid] = torch.tensor(stats["mean"], dtype=torch.float32)
            centroid_lookup = F.normalize(ctable, p=2, dim=1).to(device)
            if current_stage_idx > 0:
                print(f"  [Distill] mode={_effective_distill_mode}  "
                      f"pointwise_w={pointwise_distill_weight}  "
                      f"outward_w={centroid_outward_weight}")

        # (C_old, D) matrix of L2-normalised old-class centroids.
        # Used by orthogonal_centroid_loss and NSR repulsion.
        old_centroid_matrix: Optional[torch.Tensor] = None
        if (use_ortho_centroid_loss or use_nsr_loss) and _prev_stats:
            _c_stack = np.stack([s["mean"] for s in _prev_stats.values()])  # (C_old, D)
            old_centroid_matrix = F.normalize(
                torch.tensor(_c_stack, dtype=torch.float32), p=2, dim=1
            ).to(device)
            if use_ortho_centroid_loss:
                print(f"  [OrthoLoss] Built centroid matrix: {old_centroid_matrix.shape} "
                      f"(λ={lambda_ortho})")
            if use_nsr_loss:
                print(f"  [NSR] Using {old_centroid_matrix.shape[0]}-class centroid "
                      f"basis for repulsion (avoids full-ray subspace saturation).")

        # ── Covariance calibration: precompute L_inv per old class ───────────────
        # L_inv is the inverse of the Cholesky factor of the old covariance matrix.
        # Kept on CPU here; moved to device inside mahalanobis_cov_loss at call time
        # (negligible cost: one small matrix per class, not per step).
        _cov_inv_tensors: Dict[int, torch.Tensor] = {}
        if use_cov_loss and _prev_stats and current_stage_idx > 0:
            _D = next(iter(_prev_stats.values()))["cov"].shape[0]
            _I = np.eye(_D)
            for _cid, _st in _prev_stats.items():
                _Sigma = _st["cov"] + cov_loss_eps * _I
                try:
                    _L = np.linalg.cholesky(_Sigma)
                    _L_inv = np.linalg.inv(_L)
                except np.linalg.LinAlgError:
                    # Fallback: diagonal whitening
                    _L_inv = np.diag(1.0 / (np.sqrt(np.diag(_Sigma)) + 1e-8))
                _cov_inv_tensors[_cid] = torch.tensor(
                    _L_inv, dtype=torch.float32, device=device
                )
            if _cov_inv_tensors:
                print(f"  [CovLoss] Precomputed L_inv for {len(_cov_inv_tensors)} classes "
                      f"(λ={lambda_cov}, max_pairs={cov_max_pairs})")

        # ── NSR: class set and per-epoch buffers ─────────────────────────────────
        _current_stage_cls_set = set(stage["classes"])
        # features/images accumulated this epoch to refit the new-class dynamic hull
        _nsr_feats_buf: Dict[int, list] = defaultdict(list)
        _nsr_imgs_buf:  Dict[int, list] = defaultdict(list)
        # extreme-ray images from the PREVIOUS epoch's dynamic hull; empty at stage start
        _nsr_ray_imgs: list = []

        seen_class_ids = sorted(
            c for si in range(current_stage_idx + 1) for c in stages[si]["classes"]
        )
        _cos_geo = cos_from_deg(geo_margin_deg)

        # Set tqdm total to max_epochs, but we will break out of it early
        epoch_iterator = tqdm(range(epochs_per_stage), desc=f"Stage {current_stage_idx}", unit="ep")

        for epoch in epoch_iterator:
            backbone.train()
            head.train()
            total_loss, total_cls, total_dist, total_barrier, total_logdet = 0, 0, 0, 0, 0
            total_lock, total_route, total_ortho, total_reh_cls, total_nsr, total_stage = 0, 0, 0, 0, 0, 0
            total_cone_stab, total_cone_marg = 0, 0
            total_cov = 0.0
            total_geo_attr, total_geo_rep, total_geo_marg = 0, 0, 0
            total_own_align, total_worst_other = 0.0, 0.0
            _geo_metric_n = 0

            _use_replay = not use_cone_anchor
            _cone_margin_rad = math.radians(cone_margin_deg)
            _cone_old_ids = (
                cone_memory.old_class_ids()
                if use_cone_anchor and cone_memory is not None
                else []
            )

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                N_new     = imgs.size(0)
                has_replay = (
                    _use_replay
                    and memory_loss_enabled
                    and old_backbone is not None
                    and len(replay_buffer.all_classes) > 0
                )

                # ── Pre-sample replay for mixed-batch CE (if enabled) ─────────
                # Done here so we can fold new + replay into a single forward
                # pass, avoiding a second backbone call later.
                _reh_imgs, _reh_lbls, _reh_new_feats = None, None, None
                if use_rehearsal_cls_loss and memory_loss_enabled and has_replay:
                    _reh_imgs, _, _reh_lbls_t = replay_buffer.sample(
                        batch_size // 2, strategy=sampling_type
                    )
                    _reh_imgs = _reh_imgs.to(device)
                    _reh_lbls = _reh_lbls_t.to(device)

                # ── Single forward pass (new only, or new + replay combined) ──
                if _reh_imgs is not None:
                    _combined = backbone(torch.cat([imgs, _reh_imgs], dim=0))
                    features       = _combined[:N_new]    # new-class features
                    _reh_new_feats = _combined[N_new:]    # replay features (new backbone)
                else:
                    features = backbone(imgs)

                # ── Accumulate new-class features for per-epoch NSR hull ──────
                if use_nsr_loss:
                    _f_np  = features.detach().cpu().numpy()
                    _i_cpu = imgs.cpu()
                    for _f, _img, _lbl in zip(_f_np, _i_cpu, labels.cpu().tolist()):
                        if _lbl in _current_stage_cls_set:
                            _nsr_feats_buf[_lbl].append(_f)
                            _nsr_imgs_buf[_lbl].append(_img)

                # ── Classification loss (CE or globally geometric cone loss) ──
                loss_reh_cls = features.new_zeros(())
                if _reh_new_feats is not None:
                    _cls_feats = torch.cat([features, _reh_new_feats], dim=0)
                    _cls_lbls  = torch.cat([labels,   _reh_lbls],      dim=0)
                else:
                    _cls_feats = features
                    _cls_lbls  = labels

                if _training_loss == "geometric":
                    _geo_rays = build_training_rays(
                        head, seen_class_ids, device,
                        cone_memory=cone_memory,
                        freeze_stored_rays=True,
                    )
                    loss_cls, _geo_parts, _geo_metrics = geometric_cone_loss(
                        _cls_feats,
                        _cls_lbls,
                        _geo_rays,
                        cos_gamma=_cos_geo,
                        lambda_attr=lambda_geo_attr,
                        lambda_rep=lambda_geo_rep,
                        lambda_marg=lambda_geo_marg,
                        kernel=geo_kernel,
                        beta=geo_softplus_beta,
                        max_rep_pairs=geo_rep_max_pairs,
                    )
                    total_geo_attr += _geo_parts["attr"].item()
                    total_geo_rep  += _geo_parts["rep"].item()
                    total_geo_marg += _geo_parts["marg"].item()
                    if _geo_metrics:
                        total_own_align   += _geo_metrics.get("own_align", 0.0)
                        total_worst_other += _geo_metrics.get("worst_other", 0.0)
                        _geo_metric_n += 1
                else:
                    # FixedConicHead uses raw class-IDs as slot indices directly;
                    # all other heads use sequential positions so remap first.
                    _head_lbls = (
                        _cls_lbls if isinstance(head, FixedConicHead)
                        else _remap_labels(_cls_lbls)
                    )
                    if isinstance(head, (FixedConicHead, IncrementalConicHead, ArcFaceHead)):
                        logits = head(_cls_feats, _head_lbls)
                    else:
                        logits = head(_cls_feats)
                    loss_cls = criterion(logits, _head_lbls)
                    if _reh_new_feats is not None:
                        loss_reh_cls = (rehearsal_cls_weight - 1.0) * loss_cls
                        total_reh_cls += loss_cls.item()

                # ── Log-Det Volume Penalty (new-class features only) ──────────
                loss_logdet = torch.tensor(0.0).to(device)
                if use_logdet_penalty:
                    raw_logdet  = logdet_volume_penalty(features, eps=logdet_eps)
                    loss_logdet = logdet_weight * raw_logdet
                    total_logdet += loss_logdet.item()

                # ── Distillation Loss ─────────────────────────────────────────
                loss_distill = torch.tensor(0.0).to(device)

                if has_replay:
                    # Reuse replay images already sampled for CE, or sample fresh.
                    if _reh_imgs is not None:
                        # Already have new-backbone features; just need the teacher.
                        mem_imgs_dist    = _reh_imgs
                        mem_lbls_dist    = _reh_lbls
                        new_mem_features = _reh_new_feats
                    else:
                        mem_imgs_dist, _, _mem_lbls_t = replay_buffer.sample(
                            batch_size, strategy=sampling_type
                        )
                        mem_imgs_dist = mem_imgs_dist.to(device)
                        mem_lbls_dist = _mem_lbls_t.to(device)
                        backbone.eval()
                        new_mem_features = backbone(mem_imgs_dist)
                        backbone.train()

                    with torch.no_grad():
                        old_mem_features = old_backbone(mem_imgs_dist)

                    if use_log_barrier_distill:
                        # ── Log-Barrier Margin-Preservation Loss ─────────────────
                        num_seen = len(replay_buffer.all_classes)
                        _barrier_lbls = (
                            mem_lbls_dist if isinstance(head, FixedConicHead)
                            else _remap_labels(mem_lbls_dist)
                        )
                        raw_barrier = log_barrier_margin_loss(
                            x_new=new_mem_features,
                            x_old=old_mem_features,
                            labels=_barrier_lbls,
                            head_weight=head.weight,
                            num_seen_classes=num_seen,
                        )
                        loss_distill = barrier_weight * raw_barrier
                        total_barrier += raw_barrier.item()

                    elif _effective_distill_mode in ("centroid_aware", "hybrid"):
                        batch_centroids = (
                            centroid_lookup[mem_lbls_dist]
                            if centroid_lookup is not None else None
                        )
                        raw_distill = calculate_distillation_loss(
                            distill_type=_effective_distill_mode,
                            new_features=new_mem_features,
                            old_features=old_mem_features,
                            alpha=alpha,
                            pointwise_weight=pointwise_distill_weight,
                            centroids=batch_centroids,
                            centroid_outward_weight=centroid_outward_weight,
                        )
                        loss_distill = distill_weight * raw_distill

                    else:
                        # ── Isotropic L2 + relational distillation ────────────────
                        raw_distill = calculate_distillation_loss(
                            distill_type="basic",
                            new_features=new_mem_features,
                            old_features=old_mem_features,
                            alpha=alpha,
                            pointwise_weight=pointwise_distill_weight,
                        )
                        loss_distill = distill_weight * raw_distill

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

                # ── Orthogonal Centroid Loss ──────────────────────────────────
                loss_ortho = features.new_zeros(())
                if use_ortho_centroid_loss and old_centroid_matrix is not None:
                    raw_ortho  = orthogonal_centroid_loss(features, old_centroid_matrix)
                    loss_ortho = lambda_ortho * raw_ortho
                    total_ortho += raw_ortho.item()

                # ── Null-Space Repulsion + Conic Anchor Loss ──────────────────
                # L_repel: pushes new features away from old-class extreme rays.
                # L_anchor: pulls new features toward their own dynamic hull rays
                #           (prevents collapse into a shared null-space).
                # The dynamic hull is refit at epoch-end from accumulated features;
                # its extreme-ray images are re-forwarded each step for live grads.
                loss_nsr = features.new_zeros(())
                if use_nsr_loss and hull_mgr.static_hulls:
                    # ── build a transient new-class hull from this step's extreme
                    # ray images (forwarded with gradient for the anchor term)
                    _nsr_new_hull = None
                    _new_cls_strs   = {str(c) for c in stage["classes"]}
                    _old_static     = {
                        k: v for k, v in hull_mgr.static_hulls.items()
                        if k not in _new_cls_strs
                    }
                    _use_hull_sub   = (
                        nsr_subspace_from_hulls and use_kernel_hull
                        and current_stage_idx > 0 and _old_static
                    )
                    _nsr_centroids  = None if _use_hull_sub else old_centroid_matrix
                    _nsr_max_rank   = (
                        max(len(_old_static) * 4, 1) if _use_hull_sub
                        else len(hull_mgr.static_hulls) * 3
                    )
                    if _nsr_ray_imgs:
                        _n_sample = min(nsr_ray_sample, len(_nsr_ray_imgs))
                        _perm     = torch.randperm(len(_nsr_ray_imgs))[:_n_sample]
                        _ray_imgs = torch.stack(
                            [_nsr_ray_imgs[i] for i in _perm]
                        ).to(device)
                        _ray_feats = backbone(_ray_imgs)   # gradient flows through backbone

                        # wrap in a lightweight hull-like object so null_space_repulsion_loss
                        # can read .extreme_rays_ without refitting
                        _nsr_new_hull = types.SimpleNamespace(
                            extreme_rays_=F.normalize(
                                _ray_feats.detach(), p=2, dim=1
                            ).cpu().numpy()
                        )

                    # batch: repulsion + anchor (main training signal)
                    loss_nsr = null_space_repulsion_loss(
                        features,
                        _old_static if _use_hull_sub else hull_mgr.static_hulls,
                        new_hull=_nsr_new_hull,
                        old_centroids=_nsr_centroids,
                        normalize_feats=True,
                        weight_repel=nsr_weight_repel,
                        weight_anchor=nsr_weight_anchor,
                        energy_margin=nsr_margin,
                        variance_threshold=kernel_null_variance_threshold,
                        max_subspace_rank=_nsr_max_rank,
                    )

                    # optional extra repulsion on hull-ray images (no anchor duplicate)
                    if _nsr_ray_imgs:
                        loss_nsr = loss_nsr + nsr_ray_repel_weight * null_space_repulsion_loss(
                            _ray_feats,
                            _old_static if _use_hull_sub else hull_mgr.static_hulls,
                            old_centroids=_nsr_centroids,
                            normalize_feats=True,
                            weight_repel=nsr_weight_repel,
                            energy_margin=nsr_margin,
                            variance_threshold=kernel_null_variance_threshold,
                            max_subspace_rank=_nsr_max_rank,
                            repel_only=True,
                        )

                    if nsr_weight_spread > 0.0:
                        loss_nsr = loss_nsr + nsr_weight_spread * null_space_spread_loss(
                            features,
                            old_centroids=old_centroid_matrix,
                            labels=labels,
                        )

                    total_nsr += loss_nsr.item()

                # ── Per-stage spherical confinement ───────────────────────────
                loss_stage = features.new_zeros(())
                if use_stage_confinement_loss:
                    _repel_stages = list(range(current_stage_idx))
                    loss_stage = stage_confinement_loss(
                        features,
                        current_stage_idx,
                        stage_poles_t,
                        cos_stage_cap,
                        cos_inter_max=stage_inter_cos_max,
                        weight_in=lambda_stage_in,
                        weight_out=lambda_stage_out if _repel_stages else 0.0,
                        repel_stage_indices=_repel_stages,
                    )
                    if (
                        stage_confinement_on_replay
                        and _reh_new_feats is not None
                        and current_stage_idx > 0
                    ):
                        _replay_stage_idx = torch.tensor(
                            [stage_class_map.get(int(l.item()), -1) for l in _reh_lbls],
                            device=device,
                            dtype=torch.long,
                        )
                        loss_stage = loss_stage + stage_confinement_loss_labeled(
                            _reh_new_feats,
                            _replay_stage_idx,
                            stage_poles_t,
                            cos_stage_cap,
                            weight_in=lambda_stage_replay,
                        )
                    total_stage += loss_stage.item()

                # ── Cone + anchor stability (replay-free alternative) ─────────
                loss_cone_stab = features.new_zeros(())
                loss_cone_marg = features.new_zeros(())
                if (
                    use_cone_anchor
                    and cone_memory is not None
                    and current_stage_idx > 0
                    and _cone_old_ids
                ):
                    loss_cone_stab = cone_memory.anchor_loss(
                        backbone, device, batch_size=cone_anchor_batch,
                    )
                    if _training_loss != "geometric":
                        loss_cone_marg = cone_memory.margin_loss(
                            F.normalize(features, p=2, dim=1),
                            old_class_ids=_cone_old_ids,
                            gamma_star=_cone_margin_rad,
                        )
                    total_cone_stab += loss_cone_stab.item()
                    total_cone_marg += loss_cone_marg.item()

                # ── Covariance Calibration Loss ───────────────────────────────
                # Penalises shifts in intra-class geometry by matching pairwise
                # Mahalanobis distances (under the old covariance) between the
                # old and new network on the replay batch.
                loss_cov = features.new_zeros(())
                if (use_cov_loss and _cov_inv_tensors and has_replay
                        and new_mem_features is not None
                        and old_mem_features is not None):
                    raw_cov = mahalanobis_cov_loss(
                        new_features=new_mem_features,
                        old_features=old_mem_features,
                        labels=mem_lbls_dist,
                        class_cov_inv=_cov_inv_tensors,
                        max_pairs_per_class=cov_max_pairs,
                    )
                    loss_cov = lambda_cov * raw_cov
                    total_cov += raw_cov.item()

                loss = (loss_cls + loss_reh_cls
                        + loss_distill + loss_logdet
                        + loss_lock + loss_route + loss_ortho
                        + loss_nsr + loss_stage
                        + lambda_cone_stab * loss_cone_stab
                        + lambda_cone_marg * loss_cone_marg
                        + loss_cov)

                loss.backward()
                if grad_projector is not None:
                    grad_projector.project_grads(backbone)
                # Freeze old-class centres: zero their gradients so ArcFace
                # margin only reshapes the new-class weights.
                if (isinstance(head, ArcFaceHead)
                        and current_stage_idx > 0
                        and head.weight.grad is not None):
                    num_old_classes = head.weight.shape[0] - num_new_classes
                    head.weight.grad[:num_old_classes].zero_()
                optimizer.step()

                total_loss    += loss.item()
                total_cls     += loss_cls.item()
                total_dist    += loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0

            # ── Refit per-epoch new-class hull; cache its extreme-ray images ──
            if use_nsr_loss:
                _nsr_ray_imgs_new: list = []
                for _cls_id, _feats_list in _nsr_feats_buf.items():
                    if len(_feats_list) < 3:
                        continue
                    _feats_arr  = np.stack(_feats_list)
                    _epoch_hull = ConicHull(
                        n_rays=min(nsr_n_rays, len(_feats_list)), use_pca=False
                    )
                    _epoch_hull.fit(_feats_arr)
                    _imgs_list = _nsr_imgs_buf[_cls_id]
                    for _idx in _epoch_hull.extreme_rays_index:
                        if _idx < len(_imgs_list):
                            _nsr_ray_imgs_new.append(_imgs_list[_idx])
                _nsr_ray_imgs  = _nsr_ray_imgs_new
                _nsr_feats_buf = defaultdict(list)
                _nsr_imgs_buf  = defaultdict(list)

            avg_loss     = total_loss     / len(train_loader)
            avg_cls      = total_cls      / len(train_loader)
            avg_dist     = total_dist     / len(train_loader)
            avg_barrier  = total_barrier  / len(train_loader)
            avg_logdet   = total_logdet   / len(train_loader)
            avg_lock     = total_lock     / len(train_loader)
            avg_route    = total_route    / len(train_loader)
            avg_ortho    = total_ortho    / len(train_loader)
            avg_reh_cls  = total_reh_cls  / len(train_loader)
            avg_nsr      = total_nsr      / len(train_loader)
            avg_stage    = total_stage    / len(train_loader)
            avg_cov      = total_cov      / len(train_loader)

            # Update the progress bar suffix
            postfix = {
                "Loss":   f"{avg_loss:.4f}",
                "Cls":    f"{avg_cls:.4f}",
                "Dist":   f"{avg_dist:.5f}",
            }
            if _training_loss == "geometric":
                n_steps = len(train_loader)
                postfix["Attr"] = f"{total_geo_attr / n_steps:.5f}"
                postfix["Rep"]  = f"{total_geo_rep / n_steps:.5f}"
                postfix["Marg"] = f"{total_geo_marg / n_steps:.5f}"
                if _geo_metric_n > 0:
                    postfix["Own"]   = f"{total_own_align / _geo_metric_n:.3f}"
                    postfix["Othr"]  = f"{total_worst_other / _geo_metric_n:.3f}"
            if use_rehearsal_cls_loss:
                postfix["RehCls"] = f"{avg_reh_cls:.4f}"
            if use_log_barrier_distill:
                postfix["Barrier"] = f"{avg_barrier:.5f}"
            if use_logdet_penalty:
                postfix["LogDet"] = f"{avg_logdet:.3f}"
            if use_ortho_routing:
                postfix["Lock"]  = f"{avg_lock:.5f}"
                postfix["Route"] = f"{avg_route:.5f}"
            if use_ortho_centroid_loss:
                postfix["Ortho"] = f"{avg_ortho:.5f}"
            if use_nsr_loss:
                postfix["NSR"] = f"{avg_nsr:.5f}"
            if use_stage_confinement_loss:
                postfix["Stage"] = f"{avg_stage:.5f}"
            if use_cov_loss:
                postfix["Cov"] = f"{avg_cov:.5f}"
            if use_cone_anchor:
                postfix["Stab"] = f"{total_cone_stab / len(train_loader):.5f}"
                if _training_loss != "geometric":
                    postfix["CMarg"] = f"{total_cone_marg / len(train_loader):.5f}"
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
            if use_cone_anchor and cone_memory is not None:
                new_stats = cone_memory.extract_class_stats(
                    backbone, old_class_ids, device, batch_size,
                )
            else:
                new_stats = _extract_class_stats_from_buffer(
                    backbone, replay_buffer, old_class_ids, device, batch_size
                )

            if len(new_stats) >= 2:
                try:
                    if _prev_pca is not None:
                        pca_mean, pca_comps = _prev_pca
                        _prev_stats_pca = _project_stats_to_pca(_prev_stats, pca_mean, pca_comps)
                        _new_stats_pca  = _project_stats_to_pca(new_stats,   pca_mean, pca_comps)
                        A_pca = estimate_affine_drift(_prev_stats_pca, _new_stats_pca, method=drift_method)
                        A = _lift_pca_drift(A_pca, pca_comps)
                        print(f"     (drift estimated in {pca_comps.shape[0]}-d PCA subspace, lifted to {pca_comps.shape[1]}-d)")
                    else:
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

                    if translate_static_hulls:
                        n_tr = translate_hulls(hull_mgr.static_hulls, _prev_stats, new_stats)
                        print(f"  -> Translated {n_tr} static hull(s) by per-class mean drift Δμ.")

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
                            head, A, new_stats, list(new_stats.keys()), device,
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

        # ── Mean Shift Compensation (Eq. 6–7) ────────────────────────────────────
        # Weighted post-training patch of head weights: for each old class, estimate
        # the centroid drift using replay samples weighted by proximity to μ_old,
        # then add the estimated drift directly to the head weight vector.
        # Runs independently of use_analytical_head_update (can stack or replace).
        if (
            use_mean_shift_comp
            and old_backbone is not None
            and _prev_stats is not None
            and current_stage_idx > 0
            and len(replay_buffer.buffer) > 0
        ):
            try:
                shift_deltas = mean_shift_compensation(
                    backbone_old=old_backbone,
                    backbone_new=backbone,
                    replay_buffer=replay_buffer,
                    old_stats=_prev_stats,
                    device=device,
                    batch_size=batch_size,
                    sigma=mean_shift_sigma,
                )
                n_patched = 0
                with torch.no_grad():
                    for cls_id, delta in shift_deltas.items():
                        if cls_id >= head.weight.shape[0]:
                            continue
                        head.weight[cls_id] += torch.tensor(
                            delta, dtype=head.weight.dtype, device=device
                        )
                        n_patched += 1
                print(f"  [MeanShift] Patched {n_patched} old class weights "
                      f"(σ={mean_shift_sigma}).")
            except Exception as exc:
                print(f"  [WARNING] Mean shift compensation skipped: {exc}")

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
        if (project_hulls_to_stage_cap and not use_region_hulls
                and current_stage_idx < len(stage_poles_np)):
            _pole = stage_poles_np[current_stage_idx]
            for _cls_str in new_feature_dict:
                new_feature_dict[_cls_str] = project_to_spherical_cap(
                    new_feature_dict[_cls_str], _pole, cos_stage_cap,
                )
            print(f"  [StageConfine] Projected {len(new_feature_dict)} class feature "
                  f"sets into stage-{current_stage_idx} cap ({stage_cap_deg:.0f}°)")

        hull_mgr.fit_new_classes(new_feature_dict, region_registry=region_registry)

        if (project_hulls_to_stage_cap and not use_region_hulls
                and current_stage_idx < len(stage_poles_np)):
            _pole = stage_poles_np[current_stage_idx]
            for _cls_str in (str(c) for c in stage["classes"]):
                if _cls_str not in hull_mgr.static_hulls:
                    continue
                _h = hull_mgr.static_hulls[_cls_str]
                if _h.extreme_rays_ is not None and len(_h.extreme_rays_) > 0:
                    _h.extreme_rays_ = project_to_spherical_cap(
                        _h.extreme_rays_, _pole, cos_stage_cap,
                    )

        # Shifted-origin conic hulls: register a task anchor, fit one hull per
        # new class in the shifted frame, then accumulate into shifted_static_hulls.
        if use_shifted_hull and task_origin_registry is not None:
            all_new_feats = np.vstack(list(new_feature_dict.values()))
            task_origin   = task_origin_registry.register_task(
                current_stage_idx,
                features=(all_new_feats
                          if shifted_hull_strategy == "learned" else None),
            )
            new_shifted = build_shifted_conic_hulls(
                new_feature_dict,
                origin   = task_origin,
                n_rays   = shifted_hull_n_rays,
                use_pca  = False,
            )
            shifted_static_hulls.update(new_shifted)
            if current_stage_idx > 0:
                seps = [
                    np.degrees(task_origin_registry.separation(j, current_stage_idx))
                    for j in range(current_stage_idx)
                ]
                print(f"  [ShiftedHull] Stage {current_stage_idx} anchor registered "
                      f"({len(new_shifted)} hulls).  "
                      f"Separation from prev stages: "
                      + ", ".join(f"s{j}:{s:.0f}°" for j, s in enumerate(seps)))

        # Layered conic hull: collect intermediate-layer features and fit this stage.
        if use_layered_hull and layered_classifier is not None and _num_backbone_layers > 0:
            import torch.nn.functional as _F
            layer_idx    = _stage_to_layer[current_stage_idx]
            _extractor   = LayerFeatureExtractor(backbone, [layer_idx])
            _layer_cls_feats: Dict[str, list] = {}
            backbone.eval()
            with torch.no_grad():
                for _imgs, _lbls in stage["train_loader"]:
                    _extractor.clear()
                    backbone(_imgs.to(device))
                    _lf = _extractor[layer_idx]
                    if _lf is not None:
                        _lf_np = _F.normalize(_lf, p=2, dim=1).cpu().numpy()
                        for _feat, _lbl in zip(_lf_np, _lbls.tolist()):
                            _layer_cls_feats.setdefault(str(_lbl), []).append(_feat)
            _extractor.remove()
            _layer_feature_dict = {k: np.stack(v) for k, v in _layer_cls_feats.items()}
            n_fitted = layered_classifier.fit_stage(
                current_stage_idx, layer_idx, _layer_feature_dict
            )
            print(f"  [LayeredHull] Stage {current_stage_idx} → backbone layer {layer_idx} "
                  f"({n_fitted} class hulls fitted)")

        # Kernel hulls: built from all classes seen so far each stage.
        if use_kernel_hull:
            new_cls_strs      = {str(c) for c in stage["classes"]}
            old_static_hulls  = {
                k: v for k, v in hull_mgr.static_hulls.items()
                if k not in new_cls_strs
            }
            _kernel_max_rank  = max(len(old_static_hulls) * 4, 1)

            all_feature_dict = {}
            for cls_id in replay_buffer.all_classes:
                cls_str = str(cls_id)
                if cls_str in hull_mgr.static_hulls:
                    all_feature_dict[cls_str] = hull_mgr.static_hulls[cls_str].extreme_rays_
            # include new classes not yet in replay buffer
            for cls_str, feats in new_feature_dict.items():
                if cls_str not in all_feature_dict:
                    all_feature_dict[cls_str] = feats

            if use_region_hulls and region_registry is not None:
                for cls_str in new_cls_strs:
                    if cls_str in all_feature_dict:
                        reg = region_registry.get_region(cls_str)
                        all_feature_dict[cls_str] = project_to_class_region(
                            all_feature_dict[cls_str], reg
                        )
                print(f"  [RegionProj] Snapped {len(new_cls_strs)} new-class feature sets "
                      f"to pre-allocated stage/class caps.")
            elif project_kernel_features and current_stage_idx > 0 and old_static_hulls:
                all_feature_dict = project_new_class_features(
                    all_feature_dict,
                    new_class_keys=new_cls_strs,
                    old_hulls=old_static_hulls,
                    variance_threshold=kernel_null_variance_threshold,
                    max_subspace_rank=_kernel_max_rank,
                )
                print(f"  [KernelNullProj] Projected {len(new_cls_strs)} new-class "
                      f"feature sets into null space of {len(old_static_hulls)} "
                      f"old classes (rank cap={_kernel_max_rank}).")

            _kernel_n_rays = (
                region_registry.rays_per_class
                if region_registry is not None else hull_mgr.n_rays
            )
            kernel_static_hulls = build_class_kernel_conic_hulls(
                all_feature_dict,
                n_rays=_kernel_n_rays,
                kernel=kernel_hull_type,
                gamma=kernel_gamma,
            )

            # Rotate new-stage kernel hulls into null space of all old-class rays.
            # Only the current-stage hulls are rotated; old hulls stay frozen.
            if rotate_kernel_into_null_space and current_stage_idx > 0:
                new_cls_strs  = {str(c) for c in stage["classes"]}
                old_k_hulls   = {k: v for k, v in kernel_static_hulls.items()
                                 if k not in new_cls_strs}
                new_k_hulls   = {k: v for k, v in kernel_static_hulls.items()
                                 if k in new_cls_strs}
                if old_k_hulls and new_k_hulls:
                    n_rot = rotate_kernel_hulls_into_null_space(
                        new_hulls=new_k_hulls,
                        old_hulls=old_k_hulls,
                        variance_threshold=kernel_null_variance_threshold,
                    )
                    print(f"  [KernelNullRot] Rotated {n_rot} new-class static kernel hull(s) "
                          f"(variance_threshold={kernel_null_variance_threshold}).")
        else:
            kernel_static_hulls = None

        if visualize_extreme_rays:
            visualize_extreme_rays_3d(
                hull_mgr.static_hulls,
                stage_idx=current_stage_idx,
                stage_class_map=stage_class_map,
                kernel_hulls=kernel_static_hulls,
                kernel_gamma=kernel_gamma,
            )

        # Step 3: Register cone + anchor memory (replay-free) or populate replay buffer.
        if use_cone_anchor and cone_memory is not None:
            cone_memory.register_stage(
                stage["classes"],
                dict(new_cls_images),
                dict(new_cls_feats),
            )
            cone_memory.sync_replay_buffer(replay_buffer)
            n_cone = sum(
                cone_memory.anchor_images[c].shape[0]
                for c in stage["classes"]
                if c in cone_memory.anchor_images
            )
            print(f"  [ConeAnchor] Registered {len(stage['classes'])} classes "
                  f"({n_cone} anchor vertices total, "
                  f"{cone_memory.num_classes()} classes in memory)")
        elif _memory_enabled:
            # Hull may use few extreme rays (e.g. 4 with region budget) while replay
            # can store more exemplars up to the memory-budget per-class cap.
            _replay_target = (
                replay_samples_per_class
                if replay_samples_per_class is not None
                else replay_buffer.per_class_cap
            )
            print(f"  -> Populating replay buffer "
                  f"(target={_replay_target}/class, fill_to_cap={replay_fill_to_cap}, "
                  f"hull rays independent)...")
            for cls_id, feats_list in new_cls_feats.items():
                cls_str = str(cls_id)
                if cls_str not in hull_mgr.static_hulls:
                    continue
                hull      = hull_mgr.static_hulls[cls_str]
                n_images  = len(new_cls_images[cls_id])
                cls_feats = np.stack(feats_list) if feats_list else None
                indices   = _select_replay_indices(
                    hull, n_images, _replay_target, replay_fill_to_cap,
                    features=cls_feats,
                    use_fps_fill=fps_replay_fill,
                )

                if cls_id not in replay_buffer.buffer:
                    replay_buffer.all_classes.append(cls_id)

                ray_tuples = [
                    (new_cls_images[cls_id][i],
                     torch.from_numpy(feats_list[i]),
                     torch.tensor(cls_id))
                    for i in indices
                ]
                replay_buffer.buffer[cls_id] = ray_tuples

                if use_ortho_routing:
                    _ray_idx = (
                        list(hull.extreme_rays_index)
                        if hull.extreme_rays_index is not None else indices
                    )
                    hull_mgr.extreme_ray_images[cls_str] = [
                        new_cls_images[cls_id][i]
                        for i in _ray_idx if i < n_images
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

        # Step 4: Re-extract features for dynamic hull fitting and drift statistics.
        if use_cone_anchor and cone_memory is not None:
            feature_dict = cone_memory.build_feature_dict_from_backbone(
                backbone, device, batch_size=batch_size,
            )
        else:
            feature_dict = {}
            if not memory_loss_enabled:
                # Use stored embeddings directly — avoids a backbone forward pass
                for cls_id in replay_buffer.all_classes:
                    class_embs = [item[1] for item in replay_buffer.buffer[cls_id]]
                    if not class_embs:
                        continue
                    feature_dict[str(cls_id)] = torch.stack(class_embs).numpy()
            else:
                with torch.no_grad():
                    for cls_id in replay_buffer.all_classes:
                        class_imgs = [item[0] for item in replay_buffer.buffer[cls_id]]
                        if not class_imgs:
                            continue
                        feats = backbone(torch.stack(class_imgs).to(device)).cpu().numpy()
                        feature_dict[str(cls_id)] = feats

        # ── Snapshot per-class statistics for the *next* stage's drift estimation ─
        _stage_snapshot: Dict[int, Dict] = {}
        _feat_snapshot: Dict[int, np.ndarray] = {}
        for cls_id_str, feats in feature_dict.items():
            cls_id = int(cls_id_str)
            if cls_id not in stage["classes"]:
                continue
            n, d   = feats.shape
            mean   = feats.mean(axis=0)
            cov    = np.cov(feats.T, ddof=1) if n > 1 else np.zeros((d, d))
            _stage_snapshot[cls_id] = {"mean": mean, "cov": cov, "n": n}
            _feat_snapshot[cls_id] = feats.astype(np.float32)
        if _stage_snapshot:
            stage_stats_snapshots[current_stage_idx] = _stage_snapshot
            stage_feature_snapshots[current_stage_idx] = _feat_snapshot
            print(f"  -> Stored stage-{current_stage_idx} feature snapshot "
                  f"({len(_stage_snapshot)} classes, {sum(v.shape[0] for v in _feat_snapshot.values())} "
                  f"{'anchor' if use_cone_anchor else 'replay'} vectors) for drift-aligned eval.")

        if use_analytical_head_update:
            _prev_stats = {}
            for cls_id_str, feats in feature_dict.items():
                cls_id = int(cls_id_str)
                n, d   = feats.shape
                mean   = feats.mean(axis=0)
                cov    = np.cov(feats.T, ddof=1) if n > 1 else np.zeros((d, d))
                _prev_stats[cls_id] = {"mean": mean, "cov": cov, "n": n}
            _prev_pca = _fit_pca_on_features(feature_dict)
            k_pca = _prev_pca[1].shape[0]
            print(f"  -> Stored statistics for {len(_prev_stats)} classes "
                  f"(used for drift estimation in next stage). "
                  f"PCA fitted: {k_pca} components on {sum(v.shape[0] for v in feature_dict.values())} vectors.")

        # Step 5: Fit dynamic hulls from the current-backbone extreme-ray features.
        dynamic_hulls = hull_mgr.get_dynamic_hulls(feature_dict)

        if use_kernel_hull:
            _kernel_feat_dict = feature_dict
            if project_kernel_features and current_stage_idx > 0:
                _new_cls_strs = {str(c) for c in stage["classes"]}
                _old_static   = {
                    k: v for k, v in hull_mgr.static_hulls.items()
                    if k not in _new_cls_strs
                }
                if _old_static:
                    _kernel_feat_dict = project_new_class_features(
                        feature_dict,
                        new_class_keys=_new_cls_strs,
                        old_hulls=_old_static,
                        variance_threshold=kernel_null_variance_threshold,
                        max_subspace_rank=max(len(_old_static) * 4, 1),
                    )
            kernel_dynamic_hulls = hull_mgr.get_kernel_dynamic_hulls(
                _kernel_feat_dict, kernel=kernel_hull_type, gamma=kernel_gamma
            )
            if rotate_kernel_into_null_space and current_stage_idx > 0:
                _new_cls_strs = {str(c) for c in stage["classes"]}
                _old_k        = {k: v for k, v in kernel_dynamic_hulls.items()
                                 if k not in _new_cls_strs}
                _new_k        = {k: v for k, v in kernel_dynamic_hulls.items()
                                 if k in _new_cls_strs}
                if _old_k and _new_k:
                    rotate_kernel_hulls_into_null_space(
                        new_hulls=_new_k,
                        old_hulls=_old_k,
                        variance_threshold=kernel_null_variance_threshold,
                        verbose=False,
                    )
        else:
            kernel_dynamic_hulls = None

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

        # ── Null-space gradient projection: rebuild P_⊥ for next stage ───────────
        if use_null_space_projection and lora_rank > 0:
            all_old_rays = np.vstack(
                [h.extreme_rays_ for h in hull_mgr.static_hulls.values()]
            )
            grad_projector = GradientProjector(
                all_old_rays, device, null_space_variance_threshold
            )

        print(f"\n--- Evaluation after Stage {current_stage_idx} ---")

        # Calibration data for drift-aligned stage planes (train sets seen so far).
        from torch.utils.data import ConcatDataset
        staged_calibration_loader = None
        if evaluate_staged_hulls and current_stage_idx > 0:
            cal_sets = [stages[j]["train_loader"].dataset for j in range(current_stage_idx + 1)]
            staged_calibration_loader = DataLoader(
                ConcatDataset(cal_sets),
                batch_size=batch_size,
                shuffle=False,
                num_workers=stages[0]["train_loader"].num_workers,
            )

        # Evaluate on all stages seen so far
        for eval_stage_idx in range(current_stage_idx + 1):
            test_loader = stages[eval_stage_idx]["test_loader"]
            acc, avg_drift = evaluate_stage(backbone, head, old_backbone, test_loader, device,
                                            class_to_idx=class_to_idx)

            static_scores  = hull_mgr.evaluate_all_scores(backbone, test_loader, hull_mgr.static_hulls, device)
            dynamic_scores = hull_mgr.evaluate_all_scores(backbone, test_loader, dynamic_hulls, device)

            if evaluate_collaborative_scoring:
                collab_static_scores = hull_mgr.evaluate_collaborative(
                    backbone, test_loader, hull_mgr.static_hulls, device,
                    lasso_lambda=collaborative_lasso_lambda,
                )
                collab_dynamic_scores = hull_mgr.evaluate_collaborative(
                    backbone, test_loader, dynamic_hulls, device,
                    lasso_lambda=collaborative_lasso_lambda,
                )
            else:
                collab_static_scores = collab_dynamic_scores = None

            # Primary metric for matrix tracking: cosine (preserves existing semantics)
            acc_static_hull              = static_scores["cosine"]
            acc_dynamic_hull             = dynamic_scores["cosine"]
            acc_static_angular_margin    = static_scores["angular_margin"]

            # Staged evaluation: cascade through stages in order
            acc_staged_hull = 0.0
            if evaluate_staged_hulls and current_stage_idx > 0:
                _score_space = "plane" if staged_plane_scoring else staged_score_space
                _staged_score_hulls = (
                    dynamic_hulls
                    if staged_score_hulls.lower() == "dynamic"
                    else hull_mgr.static_hulls
                )
                staged_result = hull_mgr.evaluate_staged(
                    backbone=backbone,
                    test_loader=test_loader,
                    hulls=hull_mgr.static_hulls,
                    score_hulls=_staged_score_hulls,
                    device=device,
                    stage_class_map=stage_class_map,
                    score_key="cosine",
                    subspace_variance_threshold=staged_subspace_variance,
                    plane_scoring=(_score_space == "plane"),
                    score_space=_score_space,
                    stage_routing=staged_plane_routing,
                    plane_source=staged_plane_source,
                    calibration_loader=staged_calibration_loader,
                    calibration_max_per_class=staged_calibration_max_per_class,
                    routing_cascade_percentile=staged_routing_cascade_percentile,
                    drift_align=staged_drift_align,
                    stage_stats_snapshots=stage_stats_snapshots,
                    stage_feature_snapshots=stage_feature_snapshots,
                    replay_buffer=replay_buffer,
                    drift_method=drift_method,
                    drift_align_mode=staged_drift_align_mode,
                    drift_pair_method=staged_drift_pair_method,
                    drift_ridge=staged_drift_ridge,
                    drift_ray_weight=staged_drift_ray_weight,
                    drift_routing_only=staged_drift_routing_only,
                    verbose=True,
                )
                acc_staged_hull = staged_result["accuracy"]

            # Kernel hull evaluation
            if use_kernel_hull and kernel_static_hulls and kernel_dynamic_hulls:
                kernel_static_scores  = hull_mgr.evaluate_all_scores(backbone, test_loader, kernel_static_hulls, device)
                kernel_dynamic_scores = hull_mgr.evaluate_all_scores(backbone, test_loader, kernel_dynamic_hulls, device)
                acc_kernel_static  = kernel_static_scores["cosine"]
                acc_kernel_dynamic = kernel_dynamic_scores["cosine"]
            else:
                acc_kernel_static  = 0.0
                acc_kernel_dynamic = 0.0

            # Shifted hull evaluation
            if use_shifted_hull and shifted_static_hulls:
                shifted_scores   = hull_mgr.evaluate_all_scores(backbone, test_loader, shifted_static_hulls, device)
                acc_shifted_hull = shifted_scores["cosine"]
            else:
                acc_shifted_hull = 0.0

            # Layered hull evaluation — regressive cascade across stages
            if use_layered_hull and layered_classifier is not None:
                _all_layer_idxs   = sorted(set(_stage_to_layer.values()))
                _eval_extractor   = LayerFeatureExtractor(backbone, _all_layer_idxs)
                _layered_result   = layered_classifier.evaluate_cascade(
                    backbone, test_loader, _eval_extractor, device,
                    threshold=layered_hull_threshold,
                )
                _eval_extractor.remove()
                acc_layered_hull      = _layered_result["accuracy"]
                _layered_avg_stage    = _layered_result["avg_winning_stage"]
            else:
                acc_layered_hull   = 0.0
                _layered_avg_stage = 0.0

            acc_matrix[current_stage_idx, eval_stage_idx] = acc
            drift_matrix[current_stage_idx, eval_stage_idx] = avg_drift
            acc_matrix_conic_hull_static[current_stage_idx, eval_stage_idx]         = acc_static_hull
            acc_matrix_conic_hull_dynamic[current_stage_idx, eval_stage_idx]        = acc_dynamic_hull
            acc_matrix_conic_hull_staged[current_stage_idx, eval_stage_idx]         = acc_staged_hull
            acc_matrix_conic_hull_angular_margin[current_stage_idx, eval_stage_idx] = acc_static_angular_margin
            acc_matrix_kernel_static[current_stage_idx, eval_stage_idx]             = acc_kernel_static
            acc_matrix_kernel_dynamic[current_stage_idx, eval_stage_idx]            = acc_kernel_dynamic
            acc_matrix_shifted_hull[current_stage_idx, eval_stage_idx]              = acc_shifted_hull
            acc_matrix_layered_hull[current_stage_idx, eval_stage_idx]             = acc_layered_hull

            stage_type = "NEW" if eval_stage_idx == current_stage_idx else "OLD"
            drift_str  = f" | Drift: {avg_drift:.4f}" if stage_type == "OLD" else ""
            print(f"  [Task {eval_stage_idx} - {stage_type}] Head: {acc:.2%}{drift_str}")

            # Scoring-scheme comparison table
            col_w = 20
            header = f"  {'Scoring':<{col_w}} {'Static':>8}  {'Dynamic':>8}"
            if evaluate_staged_hulls:
                header += f"  {'Staged':>8}"
            if use_kernel_hull:
                header += f"  {'KStatic':>8}  {'KDynamic':>9}"
            if use_shifted_hull:
                header += f"  {'Shifted':>8}"
            if use_layered_hull:
                header += f"  {'Layered':>8}"
            print(header)
            sep_w = col_w + 20
            if evaluate_staged_hulls:
                sep_w += 10
            if use_kernel_hull:
                sep_w += 21
            if use_shifted_hull:
                sep_w += 11
            if use_layered_hull:
                sep_w += 11
            print("  " + "-" * sep_w)
            for s in hull_mgr.SCORE_NAMES:
                marker = " *" if s == "cosine" else "  "
                row = f"  {s:<{col_w}} {static_scores[s]:>7.2%}   {dynamic_scores[s]:>7.2%}{marker}"
                if evaluate_staged_hulls and s == "cosine":
                    row += f"  {acc_staged_hull:>7.2%}"
                if use_kernel_hull and s == "cosine":
                    row += f"  {acc_kernel_static:>7.2%}   {acc_kernel_dynamic:>7.2%}"
                if use_shifted_hull and s == "cosine":
                    row += f"  {acc_shifted_hull:>7.2%}"
                if use_layered_hull and s == "cosine":
                    row += f"  {acc_layered_hull:>7.2%}  (s̄={_layered_avg_stage:.1f})"
                print(row)
            if collab_static_scores is not None:
                print("  " + "-" * sep_w)
                lam_str = f"λ={collaborative_lasso_lambda}" if collaborative_lasso_lambda else "λ=0"
                for s in hull_mgr.COLLAB_SCORE_NAMES:
                    label = f"{s}({lam_str})"
                    print(f"  {label:<{col_w}} {collab_static_scores[s]:>7.2%}   {collab_dynamic_scores[s]:>7.2%}  †")
                print(f"  {'':>{col_w}}  † collaborative: one joint NNLS over all {len(hull_mgr.static_hulls)} class dicts")

            if debug_hull_confusion:
                hull_mgr.debug_hull_confusion(
                    backbone=backbone,
                    test_loader=test_loader,
                    hulls=dynamic_hulls,
                    device=device,
                    current_stage_classes=stages[eval_stage_idx]["classes"],
                    stage_class_map=stage_class_map,
                    score_key="cosine",
                    verbose=True,
                )

        # ── OOD Detection evaluation ──────────────────────────────────────────
        # At stage k, use test data from stages 0..k as ID and stages k+1..N as OOD.
        _ood_result: Optional[dict] = None
        if evaluate_ood_hull and hull_mgr.static_hulls and current_stage_idx < num_stages - 1:
            from torch.utils.data import ConcatDataset as _CatDS
            _id_sets  = [stages[j]["test_loader"].dataset for j in range(current_stage_idx + 1)]
            _ood_sets = [stages[j]["test_loader"].dataset for j in range(current_stage_idx + 1, num_stages)]
            _nw = stages[0]["train_loader"].num_workers
            _id_ood_loader  = DataLoader(_CatDS(_id_sets),  batch_size=batch_size, shuffle=False, num_workers=_nw)
            _ood_ood_loader = DataLoader(_CatDS(_ood_sets), batch_size=batch_size, shuffle=False, num_workers=_nw)
            _ood_result = hull_mgr.evaluate_ood_detection(
                backbone=backbone,
                id_loader=_id_ood_loader,
                ood_loader=_ood_ood_loader,
                hulls=hull_mgr.static_hulls,
                device=device,
                score_key=ood_score_key,
                calibrate_percentile=ood_calibrate_percentile,
            )
            if _ood_result is not None:
                _remaining = num_stages - 1 - current_stage_idx
                print(
                    f"\n  [OOD Detection | Stage {current_stage_idx}] "
                    f"ID: {_ood_result['n_id']} samples ({current_stage_idx + 1} seen stage(s))  "
                    f"OOD: {_ood_result['n_ood']} samples ({_remaining} future stage(s))\n"
                    f"    Score: {ood_score_key}  "
                    f"Threshold (p{ood_calibrate_percentile:.0f} of ID): {_ood_result['threshold']:.4f}  "
                    f"ID mean: {_ood_result['id_score_mean']:.4f}  "
                    f"OOD mean: {_ood_result['ood_score_mean']:.4f}\n"
                    f"    Accuracy: {_ood_result['accuracy']:.2%}  "
                    f"F1: {_ood_result['f1']:.2%}  "
                    f"Precision: {_ood_result['precision']:.2%}  "
                    f"Recall: {_ood_result['recall']:.2%}\n"
                    f"    TP={_ood_result['tp']}  FP={_ood_result['fp']}  "
                    f"TN={_ood_result['tn']}  FN={_ood_result['fn']}"
                )
        ood_detection_history.append(_ood_result)

        # Compute Metrics
        current_avg_acc                     = np.mean(acc_matrix[current_stage_idx, :current_stage_idx + 1])
        current_avg_acc_hull_static         = np.mean(acc_matrix_conic_hull_static[current_stage_idx,          :current_stage_idx + 1])
        current_avg_acc_hull_dynamic        = np.mean(acc_matrix_conic_hull_dynamic[current_stage_idx,         :current_stage_idx + 1])
        current_avg_acc_hull_staged         = np.mean(acc_matrix_conic_hull_staged[current_stage_idx,          :current_stage_idx + 1])
        current_avg_acc_angular_margin      = np.mean(acc_matrix_conic_hull_angular_margin[current_stage_idx,  :current_stage_idx + 1])
        current_avg_acc_kernel_static       = np.mean(acc_matrix_kernel_static[current_stage_idx,              :current_stage_idx + 1])
        current_avg_acc_kernel_dynamic      = np.mean(acc_matrix_kernel_dynamic[current_stage_idx,             :current_stage_idx + 1])
        current_avg_acc_shifted_hull        = np.mean(acc_matrix_shifted_hull[current_stage_idx,               :current_stage_idx + 1])
        current_avg_acc_layered_hull        = np.mean(acc_matrix_layered_hull[current_stage_idx,               :current_stage_idx + 1])

        # Calculate Forgetting (Max accuracy achieved in the past minus current accuracy)
        forgetting = 0.0
        forgetting_static         = 0.0
        forgetting_dynamic        = 0.0
        forgetting_staged         = 0.0
        forgetting_angular_margin = 0.0
        forgetting_kernel_static  = 0.0
        forgetting_kernel_dynamic = 0.0
        forgetting_shifted_hull   = 0.0
        forgetting_layered_hull   = 0.0
        if current_stage_idx > 0:
            forgetting_per_task                = []
            forgetting_per_task_static         = []
            forgetting_per_task_dynamic        = []
            forgetting_per_task_staged         = []
            forgetting_per_task_angular_margin = []
            forgetting_per_task_kernel_static  = []
            forgetting_per_task_kernel_dynamic = []
            forgetting_per_task_shifted_hull   = []
            forgetting_per_task_layered_hull   = []
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
                forgetting_per_task_staged.append(
                    np.max(acc_matrix_conic_hull_staged[:current_stage_idx, j])
                    - acc_matrix_conic_hull_staged[current_stage_idx, j]
                )
                forgetting_per_task_angular_margin.append(
                    np.max(acc_matrix_conic_hull_angular_margin[:current_stage_idx, j])
                    - acc_matrix_conic_hull_angular_margin[current_stage_idx, j]
                )
                forgetting_per_task_kernel_static.append(
                    np.max(acc_matrix_kernel_static[:current_stage_idx, j])
                    - acc_matrix_kernel_static[current_stage_idx, j]
                )
                forgetting_per_task_kernel_dynamic.append(
                    np.max(acc_matrix_kernel_dynamic[:current_stage_idx, j])
                    - acc_matrix_kernel_dynamic[current_stage_idx, j]
                )
                forgetting_per_task_shifted_hull.append(
                    np.max(acc_matrix_shifted_hull[:current_stage_idx, j])
                    - acc_matrix_shifted_hull[current_stage_idx, j]
                )
                forgetting_per_task_layered_hull.append(
                    np.max(acc_matrix_layered_hull[:current_stage_idx, j])
                    - acc_matrix_layered_hull[current_stage_idx, j]
                )
            forgetting                = np.mean(forgetting_per_task)
            forgetting_static         = np.mean(forgetting_per_task_static)
            forgetting_dynamic        = np.mean(forgetting_per_task_dynamic)
            forgetting_staged         = np.mean(forgetting_per_task_staged)
            forgetting_angular_margin = np.mean(forgetting_per_task_angular_margin)
            forgetting_kernel_static  = np.mean(forgetting_per_task_kernel_static)
            forgetting_kernel_dynamic = np.mean(forgetting_per_task_kernel_dynamic)
            forgetting_shifted_hull   = np.mean(forgetting_per_task_shifted_hull)
            forgetting_layered_hull   = np.mean(forgetting_per_task_layered_hull)

        print(f"\n  -> Average Accuracy (Tasks 0-{current_stage_idx}): {current_avg_acc:.2%}")
        print(f"  -> Average Static Hull Accuracy    (Tasks 0-{current_stage_idx}): {current_avg_acc_hull_static:.2%}")
        print(f"  -> Average Dynamic Hull Accuracy   (Tasks 0-{current_stage_idx}): {current_avg_acc_hull_dynamic:.2%}")
        print(f"  -> Average Angular Margin Accuracy (Tasks 0-{current_stage_idx}): {current_avg_acc_angular_margin:.2%}")
        if evaluate_staged_hulls:
            print(f"  -> Average Staged Hull Accuracy    (Tasks 0-{current_stage_idx}): {current_avg_acc_hull_staged:.2%}")
        if use_kernel_hull:
            print(f"  -> Average Kernel Static Accuracy  (Tasks 0-{current_stage_idx}): {current_avg_acc_kernel_static:.2%}")
            print(f"  -> Average Kernel Dynamic Accuracy (Tasks 0-{current_stage_idx}): {current_avg_acc_kernel_dynamic:.2%}")
        if use_shifted_hull:
            print(f"  -> Average Shifted Hull Accuracy   (Tasks 0-{current_stage_idx}): {current_avg_acc_shifted_hull:.2%}")
        if use_layered_hull:
            print(f"  -> Average Layered Hull Accuracy   (Tasks 0-{current_stage_idx}): {current_avg_acc_layered_hull:.2%}")

        if current_stage_idx > 0:
            print(f"  -> Average Forgetting (Linear):         {forgetting:.2%}")
            print(f"  -> Average Forgetting (Static Hull):    {forgetting_static:.2%}")
            print(f"  -> Average Forgetting (Dynamic Hull):   {forgetting_dynamic:.2%}")
            print(f"  -> Average Forgetting (Angular Margin): {forgetting_angular_margin:.2%}")
            if evaluate_staged_hulls:
                print(f"  -> Average Forgetting (Staged Hull):    {forgetting_staged:.2%}")
            if use_kernel_hull:
                print(f"  -> Average Forgetting (Kernel Static):  {forgetting_kernel_static:.2%}")
                print(f"  -> Average Forgetting (Kernel Dynamic): {forgetting_kernel_dynamic:.2%}")
            if use_shifted_hull:
                print(f"  -> Average Forgetting (Shifted Hull):   {forgetting_shifted_hull:.2%}")
            if use_layered_hull:
                print(f"  -> Average Forgetting (Layered Hull):   {forgetting_layered_hull:.2%}")

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
    def _print_avg_stats(label: str, mat: np.ndarray) -> None:
        print(f"\n=== {label} ===")
        all_avg = []
        for stage_i in range(mat.shape[0]):
            stage_stats     = mat[stage_i, : stage_i + 1]
            avg_of_stage    = float(np.mean(stage_stats))
            all_avg.append(avg_of_stage)
            print(f"  Stage {stage_i} average: {avg_of_stage:.4f}")
        print(f"  Average of all stages: {float(np.mean(all_avg)):.4f}")
        print(f"  Final stage average:   {float(np.mean(mat[-1, :mat.shape[0]])):.4f}")

    _print_avg_stats("Head Classifier",           acc_matrix)
    _print_avg_stats("Static Hull (cosine)",       acc_matrix_conic_hull_static)
    _print_avg_stats("Dynamic Hull (cosine)",      acc_matrix_conic_hull_dynamic)
    _print_avg_stats("Static Hull (angular_margin)", acc_matrix_conic_hull_angular_margin)
    if evaluate_staged_hulls:
        _print_avg_stats("Staged Hull (cosine)", acc_matrix_conic_hull_staged)
    if use_kernel_hull:
        _print_avg_stats("Kernel Static Hull (cosine)",  acc_matrix_kernel_static)
        _print_avg_stats("Kernel Dynamic Hull (cosine)", acc_matrix_kernel_dynamic)
    if use_shifted_hull:
        _print_avg_stats("Shifted Hull (cosine)",        acc_matrix_shifted_hull)
    if use_layered_hull:
        _print_avg_stats("Layered Hull (cosine, cascade)", acc_matrix_layered_hull)

    # ── OOD Detection summary table ───────────────────────────────────────────
    if evaluate_ood_hull:
        valid_ood = [(si, r) for si, r in enumerate(ood_detection_history) if r is not None]
        if valid_ood:
            print(f"\n{'='*80}")
            print(f"=== OOD Detection Summary  (score={ood_score_key}, "
                  f"threshold=p{ood_calibrate_percentile:.0f} of ID) ===")
            print(f"{'='*80}")
            hdr = (f"  {'Stage':<7} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} "
                   f"{'Thr':>8} {'ID mean':>9} {'OOD mean':>10} {'n_ID':>7} {'n_OOD':>8}")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for si, r in valid_ood:
                print(
                    f"  {si:<7} {r['accuracy']:>8.2%} {r['f1']:>8.2%} "
                    f"{r['precision']:>8.2%} {r['recall']:>8.2%} "
                    f"{r['threshold']:>8.4f} {r['id_score_mean']:>9.4f} "
                    f"{r['ood_score_mean']:>10.4f} {r['n_id']:>7} {r['n_ood']:>8}"
                )
            if len(valid_ood) > 1:
                mean_acc = float(np.mean([r["accuracy"] for _, r in valid_ood]))
                mean_f1  = float(np.mean([r["f1"]       for _, r in valid_ood]))
                print("  " + "-" * (len(hdr) - 2))
                print(f"  {'mean':<7} {mean_acc:>8.2%} {mean_f1:>8.2%}")
        else:
            print("\n  [OOD Detection] No results (last stage has no future classes).")

    # Compute final average forgetting across all tasks (last row only)
    num_stages = len(acc_matrix)
    def _avg_forgetting(mat):
        rows = []
        for j in range(num_stages - 1):
            best = np.max(mat[:num_stages - 1, j])
            rows.append(best - mat[num_stages - 1, j])
        return float(np.mean(rows)) if rows else 0.0

    # final_forgetting                = _avg_forgetting(acc_matrix)
    # final_forgetting_static         = _avg_forgetting(acc_matrix_conic_hull_static)
    # final_forgetting_dynamic        = _avg_forgetting(acc_matrix_conic_hull_dynamic)
    # final_forgetting_staged         = _avg_forgetting(acc_matrix_conic_hull_staged)
    # final_forgetting_angular_margin = _avg_forgetting(acc_matrix_conic_hull_angular_margin)
    # final_forgetting_kernel_static  = _avg_forgetting(acc_matrix_kernel_static)
    # final_forgetting_kernel_dynamic = _avg_forgetting(acc_matrix_kernel_dynamic)
    # final_forgetting_shifted_hull   = _avg_forgetting(acc_matrix_shifted_hull)
    # final_forgetting_layered_hull   = _avg_forgetting(acc_matrix_layered_hull)

    # print(f"\n=== Final Average Forgetting ===")
    # print(f"  Linear Head:      {final_forgetting:.2%}")
    # print(f"  Static Hull:      {final_forgetting_static:.2%}")
    # print(f"  Dynamic Hull:     {final_forgetting_dynamic:.2%}")
    # print(f"  Angular Margin:   {final_forgetting_angular_margin:.2%}")
    # if evaluate_staged_hulls:
    #     print(f"  Staged Hull:      {final_forgetting_staged:.2%}")
    # if use_kernel_hull:
    #     print(f"  Kernel Static:    {final_forgetting_kernel_static:.2%}")
    #     print(f"  Kernel Dynamic:   {final_forgetting_kernel_dynamic:.2%}")
    # if use_shifted_hull:
    #     print(f"  Shifted Hull:     {final_forgetting_shifted_hull:.2%}")
    # if use_layered_hull:
    #     print(f"  Layered Hull:     {final_forgetting_layered_hull:.2%}")

    return (
        backbone, head,
        acc_matrix, acc_matrix_conic_hull_static, acc_matrix_conic_hull_dynamic,
        acc_matrix_conic_hull_staged,
        acc_matrix_conic_hull_angular_margin,
        acc_matrix_kernel_static, acc_matrix_kernel_dynamic,
        acc_matrix_shifted_hull,
        acc_matrix_layered_hull,
        drift_matrix,
        
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