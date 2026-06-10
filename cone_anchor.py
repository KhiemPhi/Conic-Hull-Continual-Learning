"""
Cone + anchor continual learning (replay-free stability).

State
-----
  Cones[c]   : (K, d) frozen extreme-ray directions per class
  Anchors[c] : K vertex images + frozen embeddings per class

Losses
------
  L_stab : MSE between current and frozen anchor embeddings
  L_marg : hinge repelling batch features from old-class cones
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conic_hull import ConicHull


def init_new_direction(
    cones: Dict[int, torch.Tensor],
    d: int,
    device: torch.device,
    n_candidates: int = 4096,
) -> torch.Tensor:
    """
    Farthest-point direction against all stored extreme rays (unit sphere).
    """
    if not cones:
        return F.normalize(torch.randn(d, device=device), dim=0)

    V_all = torch.cat(list(cones.values())).to(device)
    n_candidates = max(int(n_candidates), 1)
    cand = F.normalize(torch.randn(n_candidates, d, device=device), dim=1)
    max_cos = (cand @ V_all.T).max(dim=1).values
    return cand[max_cos.argmin()]


def margin_hinge_loss(
    z: torch.Tensor,
    cones: Dict[int, torch.Tensor],
    gamma_star: float,
) -> torch.Tensor:
    """
    Penalise features that enter within angular margin γ* of any old-class cone.

    Per old class c, take max cosine to its rays; violate if max_cos > cos(γ*).
    """
    if not cones:
        return z.new_zeros(())

    z = F.normalize(z, p=2, dim=1)
    cos_gamma = z.new_tensor(float(math.cos(gamma_star)))

    max_cos_parts: list = []
    for V in cones.values():
        V_n = F.normalize(V.to(device=z.device, dtype=z.dtype), p=2, dim=1)
        max_cos_parts.append((z @ V_n.T).max(dim=1).values)

    max_cos = torch.stack(max_cos_parts, dim=1)
    return F.relu(max_cos - cos_gamma).mean()


def anchor_loss(
    backbone: nn.Module,
    anchor_imgs: torch.Tensor,
    anchor_z_star: torch.Tensor,
) -> torch.Tensor:
    """L2 on the sphere between re-forwarded anchors and frozen embeddings."""
    z = F.normalize(backbone(anchor_imgs), p=2, dim=1)
    z_star = F.normalize(anchor_z_star, p=2, dim=1)
    return ((z - z_star) ** 2).sum(dim=1).mean()


class ConeAnchorMemory:
    """
    Stores frozen cones and vertex anchors for all seen classes.
    """

    def __init__(self, n_rays: int = 20):
        self.n_rays = int(n_rays)
        self.cones: Dict[int, torch.Tensor] = {}
        self.anchor_images: Dict[int, torch.Tensor] = {}
        self.anchor_z: Dict[int, torch.Tensor] = {}

    def num_classes(self) -> int:
        return len(self.cones)

    def get_cones_dict(self) -> Dict[int, torch.Tensor]:
        return self.cones

    def cones_on_device(self, device: torch.device) -> Dict[int, torch.Tensor]:
        return {c: V.to(device) for c, V in self.cones.items()}

    def old_class_ids(self) -> List[int]:
        return sorted(self.cones.keys())

    def register_class(
        self,
        cls_id: int,
        images: List[torch.Tensor],
        feats: np.ndarray,
        n_rays: Optional[int] = None,
    ) -> None:
        """
        Fit SPA extreme rays on class features; store vertex images + z*.
        """
        K = n_rays if n_rays is not None else self.n_rays
        feats = np.asarray(feats, dtype=np.float64)
        if feats.shape[0] == 0:
            return
        K = min(K, feats.shape[0])

        hull = ConicHull(n_rays=K, use_pca=False)
        hull.fit(feats)
        idxs = hull.extreme_rays_index
        if idxs is None or len(idxs) == 0:
            return

        rays = torch.tensor(hull.extreme_rays_, dtype=torch.float32)
        imgs = torch.stack([images[int(i)] for i in idxs])
        z_star = F.normalize(torch.tensor(feats[idxs], dtype=torch.float32), dim=1)

        self.cones[int(cls_id)] = rays
        self.anchor_images[int(cls_id)] = imgs.cpu()
        self.anchor_z[int(cls_id)] = z_star

    def register_stage(
        self,
        class_ids: List[int],
        images_by_class: Dict[int, list],
        feats_by_class: Dict[int, list],
    ) -> None:
        for cls_id in class_ids:
            if cls_id not in images_by_class or cls_id not in feats_by_class:
                continue
            imgs = images_by_class[cls_id]
            feats = np.stack(feats_by_class[cls_id])
            if len(imgs) != feats.shape[0]:
                n = min(len(imgs), feats.shape[0])
                imgs = imgs[:n]
                feats = feats[:n]
            self.register_class(cls_id, imgs, feats)

    def sample_anchors(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Uniform sample over all stored anchor vertices."""
        if not self.anchor_images:
            return None, None

        pools: list = []
        for cls_id in self.old_class_ids():
            n = self.anchor_images[cls_id].shape[0]
            for vi in range(n):
                pools.append((cls_id, vi))

        if not pools:
            return None, None

        batch_size = min(int(batch_size), len(pools))
        perm = torch.randperm(len(pools))[:batch_size].tolist()
        imgs: list = []
        z_stars: list = []
        for j in perm:
            cls_id, vi = pools[j]
            imgs.append(self.anchor_images[cls_id][vi])
            z_stars.append(self.anchor_z[cls_id][vi])
        return (
            torch.stack(imgs).to(device),
            torch.stack(z_stars).to(device),
        )

    def anchor_loss(
        self,
        backbone: nn.Module,
        device: torch.device,
        batch_size: int = 128,
    ) -> torch.Tensor:
        imgs, z_star = self.sample_anchors(batch_size, device)
        if imgs is None or z_star is None:
            return next(backbone.parameters()).new_zeros(())
        return anchor_loss(backbone, imgs, z_star)

    def margin_loss(
        self,
        z: torch.Tensor,
        old_class_ids: Optional[List[int]] = None,
        gamma_star: float = math.radians(35.0),
    ) -> torch.Tensor:
        if old_class_ids is None:
            cones = self.cones_on_device(z.device)
        else:
            cones = {
                c: self.cones[c].to(z.device)
                for c in old_class_ids
                if c in self.cones
            }
        return margin_hinge_loss(z, cones, gamma_star)

    def build_feature_dict(self) -> Dict[str, np.ndarray]:
        """Feature dict from frozen anchor embeddings (for drift / hull aux)."""
        out: Dict[str, np.ndarray] = {}
        for cls_id in self.old_class_ids():
            out[str(cls_id)] = self.anchor_z[cls_id].numpy()
        return out

    def build_feature_dict_from_backbone(
        self,
        backbone: nn.Module,
        device: torch.device,
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """Re-forward anchor vertex images through the current backbone."""
        out: Dict[str, np.ndarray] = {}
        backbone.eval()
        with torch.no_grad():
            for cls_id in self.old_class_ids():
                imgs = self.anchor_images[cls_id]
                if imgs.numel() == 0:
                    continue
                feats: list = []
                for start in range(0, imgs.shape[0], batch_size):
                    chunk = imgs[start : start + batch_size].to(device)
                    feats.append(backbone(chunk).cpu().numpy())
                out[str(cls_id)] = np.concatenate(feats, axis=0)
        return out

    def extract_class_stats(
        self,
        backbone: nn.Module,
        class_ids: List[int],
        device: torch.device,
        batch_size: int = 64,
    ) -> Dict[int, Dict]:
        """Per-class mean/cov from anchor vertices (for analytical head update)."""
        stats: Dict[int, Dict] = {}
        feat_dict = self.build_feature_dict_from_backbone(
            backbone, device, batch_size=batch_size,
        )
        for cls_id in class_ids:
            feats = feat_dict.get(str(cls_id))
            if feats is None or feats.shape[0] == 0:
                continue
            n, d = feats.shape
            mean = feats.mean(axis=0)
            cov = np.cov(feats.T, ddof=1) if n > 1 else np.zeros((d, d))
            stats[int(cls_id)] = {"mean": mean, "cov": cov, "n": n}
        return stats

    def sync_replay_buffer(self, replay_buffer) -> None:
        """
        Mirror anchor vertices into the replay buffer for drift-aligned eval.
        Training does not sample from the buffer when cone-anchor mode is on.
        """
        for cls_id in self.old_class_ids():
            imgs = self.anchor_images[cls_id]
            z_star = self.anchor_z[cls_id]
            if imgs.numel() == 0:
                continue
            if cls_id not in replay_buffer.buffer:
                replay_buffer.all_classes.append(cls_id)
            replay_buffer.buffer[cls_id] = [
                (imgs[i], z_star[i].clone(), torch.tensor(cls_id))
                for i in range(imgs.shape[0])
            ]


def _mlp_output_weights_from_directions(
    head: nn.Module,
    directions: torch.Tensor,
) -> torch.Tensor:
    """
    Map unit input-space directions to fc rows so W_out @ W_proj ≈ directions.
    """
    W_proj = head.projection[0].weight                      # (H, D)
    W_pinv = W_proj.T @ torch.linalg.inv(W_proj @ W_proj.T)  # (D, H)
    return directions @ W_pinv                              # (num_new, H)


def expand_head_with_directions(
    head: nn.Module,
    num_new: int,
    directions: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Append *num_new* cosine class weights initialized to unit directions.
    """
    directions = F.normalize(directions[:num_new].to(device), p=2, dim=1)
    cls_name = head.__class__.__name__

    if cls_name == "ArcFaceHead":
        head.add_classes(num_new, device)
        with torch.no_grad():
            head.weight[-num_new:] = directions
        return

    if cls_name == "IncrementalMLPHead":
        head.add_classes(num_new, device)
        with torch.no_grad():
            head.fc.weight[-num_new:] = _mlp_output_weights_from_directions(
                head, directions,
            )
            head.fc.bias[-num_new:].zero_()
        return

    if hasattr(head, "fc") and hasattr(head, "in_features"):
        head.add_classes(num_new, device)
        with torch.no_grad():
            if head.fc.weight.shape[1] == directions.shape[1]:
                head.fc.weight[-num_new:] = directions
            else:
                head.fc.weight[-num_new:].normal_(0, 0.01)
            if head.fc.bias is not None:
                head.fc.bias[-num_new:].zero_()
        return

    if cls_name == "IncrementalConicHead" or (
        hasattr(head, "weight")
        and not isinstance(getattr(head, "weight"), nn.Parameter)
        and not hasattr(head, "fc")
    ):
        if head.weight.numel() == 0:
            head.weight = directions
        else:
            head.weight = torch.cat([head.weight, directions], dim=0)
        return

    if isinstance(getattr(head, "weight", None), nn.Parameter):
        if head.weight.numel() == 0:
            head.weight = nn.Parameter(directions.clone())
        else:
            head.add_classes(num_new, device)
            with torch.no_grad():
                head.weight[-num_new:] = directions
        return

    if hasattr(head, "add_classes"):
        head.add_classes(num_new, device)
        return

    raise TypeError(f"Unsupported head type for cone-anchor init: {type(head)}")
