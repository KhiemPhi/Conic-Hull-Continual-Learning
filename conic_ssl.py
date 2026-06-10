"""
Conic self-supervised learning — core components.

Phase 0 state:
  online encoder φ_θ, target encoder φ_θ̄ (EMA), cone bank V, anchor store A.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conic_hull import ConicHull
from geometric_loss import cone_ray_repulsion_loss, cos_from_deg


class ConeStatus(IntEnum):
    RESERVED = 0
    ACTIVE = 1
    FROZEN = 2


@dataclass
class CapacityRecord:
    task_id: int
    d_eff: float
    packing_number: float
    gamma_t: float
    n_active: int
    n_frozen: int


@dataclass
class SSLStepMetrics:
    loss_total: float = 0.0
    loss_consist: float = 0.0
    loss_rep: float = 0.0
    loss_vol: float = 0.0
    loss_tight: float = 0.0
    loss_temp: float = 0.0
    loss_stab: float = 0.0
    loss_attr: float = 0.0
    own_align: float = 0.0
    worst_other: float = 0.0


class ProjectionHead(nn.Module):
    """Backbone features → L2-normalised sphere embedding."""

    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class OnlineEncoder(nn.Module):
    """φ_θ: backbone + projection head."""

    def __init__(self, backbone: nn.Module, proj_dim: int, feat_dim: Optional[int] = None):
        super().__init__()
        self.backbone = backbone
        if feat_dim is None:
            feat_dim = getattr(backbone, "num_features", None)
        if feat_dim is None:
            device = next(backbone.parameters()).device
            with torch.no_grad():
                feat_dim = backbone(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
        self._feat_dim = int(feat_dim)
        self.head = ProjectionHead(self._feat_dim, proj_dim)

    @property
    def num_features(self) -> int:
        return self._feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class EMAEncoder:
    """Target encoder φ_θ̄ with stop-gradient outputs."""

    def __init__(self, online: OnlineEncoder, decay: float = 0.996):
        self.decay = decay
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, online: OnlineEncoder) -> None:
        m = self.decay
        for p_t, p_o in zip(self.target.parameters(), online.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target(x)


class ConeBank(nn.Module):
    """
    Cone bank V = {V_c ∈ R^{K×d}} with status flags per cone.
    """

    def __init__(
        self,
        c_max: int,
        k: int,
        d: int,
        gamma_star: float,
        device: torch.device,
    ):
        super().__init__()
        self.c_max = int(c_max)
        self.k = int(k)
        self.d = int(d)
        self.gamma_star = float(gamma_star)
        self.cos_gamma = float(math.cos(gamma_star))

        init = F.normalize(torch.randn(c_max, k, d), dim=2)
        self.rays = nn.Parameter(init)
        self.register_buffer("status", torch.full((c_max,), ConeStatus.RESERVED, dtype=torch.long))

    def active_indices(self) -> List[int]:
        return torch.where(self.status == ConeStatus.ACTIVE)[0].tolist()

    def frozen_indices(self) -> List[int]:
        return torch.where(self.status == ConeStatus.FROZEN)[0].tolist()

    def usable_indices(self) -> List[int]:
        return (self.active_indices() + self.frozen_indices())

    def reserved_indices(self) -> List[int]:
        return torch.where(self.status == ConeStatus.RESERVED)[0].tolist()

    def n_reserved(self) -> int:
        return int((self.status == ConeStatus.RESERVED).sum())

    def rays_for_indices(self, indices: Sequence[int], detach_frozen: bool = True) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for c in indices:
            V = F.normalize(self.rays[c], p=2, dim=1)
            if detach_frozen and int(self.status[c]) == ConeStatus.FROZEN:
                V = V.detach()
            out[int(c)] = V
        return out

    def activate_cone(self, c: int, rays: torch.Tensor) -> None:
        assert int(self.status[c]) == ConeStatus.RESERVED
        with torch.no_grad():
            self.rays[c].copy_(F.normalize(rays.to(self.rays.device), p=2, dim=1))
            self.status[c] = ConeStatus.ACTIVE

    def freeze_cone(self, c: int) -> None:
        self.status[c] = ConeStatus.FROZEN

    def ensure_seed_cone(self) -> int:
        """Activate the first reserved cone if none are active yet."""
        active = self.active_indices()
        if active:
            return active[0]
        reserved = self.reserved_indices()
        if not reserved:
            raise RuntimeError("Cone bank full: no reserved slots for seed cone")
        c = reserved[0]
        rays = F.normalize(
            torch.randn(self.k, self.d, device=self.rays.device), dim=1,
        )
        self.activate_cone(c, rays)
        return c

    def reinit_reserved_farthest(self, c: int) -> None:
        """Re-initialise a dead cone via farthest-point against active cones."""
        active = self.active_indices()
        if not active:
            with torch.no_grad():
                self.rays[c].copy_(F.normalize(torch.randn(self.k, self.d, device=self.rays.device), dim=1))
            return
        V_all = torch.cat([F.normalize(self.rays[i], dim=1) for i in active], dim=0)
        cand = F.normalize(torch.randn(4096, self.d, device=self.rays.device), dim=1)
        best = cand[(cand @ V_all.T).max(dim=1).values.argmin()]
        with torch.no_grad():
            self.rays[c].copy_(best.unsqueeze(0).expand(self.k, -1))

    def trainable_parameters(self) -> List[nn.Parameter]:
        if self.active_indices():
            return [self.rays]
        return []

    def mask_inactive_grads(self) -> None:
        """Zero gradients on reserved / frozen cones; only ACTIVE slots update."""
        if self.rays.grad is None:
            return
        with torch.no_grad():
            for c in range(self.c_max):
                if int(self.status[c]) != ConeStatus.ACTIVE:
                    self.rays.grad[c].zero_()


class AnchorStore:
    """A = {(x_a, z_a*, c_a)} for stability replay."""

    def __init__(self):
        self.images: List[torch.Tensor] = []
        self.embeddings: List[torch.Tensor] = []
        self.cone_ids: List[int] = []

    def __len__(self) -> int:
        return len(self.images)

    def add(self, img: torch.Tensor, z_star: torch.Tensor, cone_id: int) -> None:
        self.images.append(img.cpu())
        self.embeddings.append(F.normalize(z_star.detach().cpu(), dim=0))
        self.cone_ids.append(int(cone_id))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.images:
            return None, None
        n = min(batch_size, len(self.images))
        idx = torch.randperm(len(self.images))[:n].tolist()
        imgs = torch.stack([self.images[i] for i in idx]).to(device)
        z_stars = torch.stack([self.embeddings[i] for i in idx]).to(device)
        return imgs, z_stars


class CapacityLog:
    def __init__(self):
        self.records: List[CapacityRecord] = []

    def append(self, record: CapacityRecord) -> None:
        self.records.append(record)


def default_gamma_star(c_max: int, scale: float = 0.7) -> float:
    """ETF angle scaled: arccos(-1/(C_max-1)) * scale."""
    if c_max <= 1:
        return math.pi / 2
    etf = math.acos(-1.0 / (c_max - 1))
    return etf * scale


def sinkhorn_normalize(Q: torch.Tensor, n_iters: int = 3, eps: float = 1e-8) -> torch.Tensor:
    """Doubly-stochastic normalisation on assignment matrix Q (B × C)."""
    P = Q.clamp(min=eps)
    for _ in range(n_iters):
        P = P / (P.sum(dim=0, keepdim=True) + eps)
        P = P / (P.sum(dim=1, keepdim=True) + eps)
    return P


def cone_nearest_scores(z: torch.Tensor, rays_by_cone: Dict[int, torch.Tensor]) -> torch.Tensor:
    """s_c(z) = max_k ⟨z, v_c^(k)⟩ for each cone. Returns (B, C_usable)."""
    if not rays_by_cone:
        return z.new_zeros(z.shape[0], 0)
    cone_ids = sorted(rays_by_cone.keys())
    scores = []
    for c in cone_ids:
        V = F.normalize(rays_by_cone[c].to(z.device), p=2, dim=1)
        scores.append((z @ V.T).max(dim=1).values.unsqueeze(1))
    return torch.cat(scores, dim=1)


def soft_cone_assignments(
    z: torch.Tensor,
    rays_by_cone: Dict[int, torch.Tensor],
    tau: float,
) -> Tuple[torch.Tensor, List[int]]:
    """Softmax over active/frozen cones. Returns (B, C) and cone id list."""
    cone_ids = sorted(rays_by_cone.keys())
    if not cone_ids:
        return z.new_zeros(z.shape[0], 0), []
    scores = cone_nearest_scores(z, rays_by_cone)
    return F.softmax(tau * scores, dim=1), cone_ids


def consistency_loss(
    z_online: torch.Tensor,
    z_target: torch.Tensor,
    rays_by_cone: Dict[int, torch.Tensor],
    tau: float,
    sinkhorn_iters: int = 3,
) -> torch.Tensor:
    """Cross-view cone assignment CE; falls back to cosine alignment with 0–1 cones."""
    if not rays_by_cone:
        return z_online.new_zeros(())
    if len(rays_by_cone) == 1:
        return (1.0 - (z_online * z_target).sum(dim=1)).mean()
    q_online, _ = soft_cone_assignments(z_online, rays_by_cone, tau)
    q_target, _ = soft_cone_assignments(z_target, rays_by_cone, tau)
    Q = sinkhorn_normalize(q_target, n_iters=sinkhorn_iters)
    return -(Q * (q_online.clamp(min=1e-8).log())).sum(dim=1).mean()


def volume_preservation_loss(
    z: torch.Tensor,
    eps: float = 1e-4,
    max_val: float = 3.0,
) -> torch.Tensor:
    """
    Encourage non-degenerate batch spread via mean log singular value.

    Full d×d log-det on L2-normalised features is ill-conditioned (rank ≤ B−1);
    SVD on the centred batch is stable and bounded.
    """
    B, d = z.shape
    if B < 2:
        return z.new_zeros(())
    centered = z - z.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centered).clamp(min=eps)
    k = min(B - 1, d)
    log_spread = sv[:k].log().mean()
    return (-log_spread).clamp(max=max_val)


def tightness_loss(z: torch.Tensor, rays_by_cone: Dict[int, torch.Tensor]) -> torch.Tensor:
    scores = cone_nearest_scores(z, rays_by_cone)
    if scores.numel() == 0:
        return z.new_zeros(())
    return -scores.max(dim=1).values.mean()


def temporal_distillation_loss(z_new: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
    return ((z_new - z_prev) ** 2).sum(dim=1).mean()


def anchor_stability_loss(z_anchor: torch.Tensor, z_star: torch.Tensor) -> torch.Tensor:
    return ((F.normalize(z_anchor, dim=1) - F.normalize(z_star, dim=1)) ** 2).sum(dim=1).mean()


def supervised_attraction_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    cone_ids: List[int],
    rays_by_cone: Dict[int, torch.Tensor],
    cos_gamma: float,
) -> torch.Tensor:
    """(cos γ* - max_k ⟨z, v_y^(k)⟩)_+² averaged over labelled batch."""
    if not cone_ids:
        return z.new_zeros(())
    id_to_col = {c: i for i, c in enumerate(cone_ids)}
    per_sample: list = []
    for i, y in enumerate(labels.tolist()):
        c = int(y)
        if c not in rays_by_cone:
            continue
        V = F.normalize(rays_by_cone[c].to(z.device), p=2, dim=1)
        cos_own = (z[i : i + 1] @ V.T).max()
        per_sample.append(F.relu(cos_gamma - cos_own).pow(2))
    if not per_sample:
        return z.new_zeros(())
    return torch.stack(per_sample).mean()


def spa_initialize_cone(feats: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """SPA extreme rays; returns (K, d) rays and indices into feats."""
    hull = ConicHull(n_rays=min(k, len(feats)), use_pca=False)
    hull.fit(feats)
    idx = hull.extreme_rays_index
    if idx is None or len(idx) == 0:
        idx = np.arange(min(k, len(feats)))
    rays = np.asarray(hull.extreme_rays_, dtype=np.float32)
    return rays, np.asarray(idx, dtype=np.int64)


def effective_dimensionality(feats: np.ndarray, eps: float = 1e-8) -> float:
    """Participation ratio from feature covariance eigenvalues."""
    if feats.shape[0] < 2:
        return float(feats.shape[1])
    X = feats - feats.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(feats.shape[0] - 1, 1)
    eig = np.linalg.eigvalsh(cov).clip(min=eps)
    return float((eig.sum() ** 2) / (np.square(eig).sum() + eps))


def packing_number(rays_by_cone: Dict[int, torch.Tensor], cos_gamma: float) -> float:
    """Fraction of cross-class ray pairs with cosine ≤ cos(γ*)."""
    if len(rays_by_cone) < 2:
        return 1.0
    parts, tags = [], []
    for c, V in rays_by_cone.items():
        V = F.normalize(V, dim=1)
        parts.append(V)
        tags.extend([c] * V.shape[0])
    V_all = torch.cat(parts, dim=0)
    tags_t = torch.tensor(tags)
    cos = V_all @ V_all.T
    n = V_all.shape[0]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    cross = tags_t.unsqueeze(0) != tags_t.unsqueeze(1)
    mask = upper & cross
    if not mask.any():
        return 1.0
    vals = cos[mask]
    return float((vals <= cos_gamma).float().mean().item())


def inter_cone_margin_gamma(rays_by_cone: Dict[int, torch.Tensor]) -> float:
    """Min angular separation (radians) between cross-class ray pairs."""
    if len(rays_by_cone) < 2:
        return math.pi
    parts, tags = [], []
    for c, V in rays_by_cone.items():
        parts.append(F.normalize(V, dim=1))
        tags.extend([c] * V.shape[0])
    V_all = torch.cat(parts, dim=0)
    tags_t = torch.tensor(tags)
    cos = V_all @ V_all.T
    n = V_all.shape[0]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    cross = tags_t.unsqueeze(0) != tags_t.unsqueeze(1)
    mask = upper & cross
    if not mask.any():
        return math.pi
    min_cos = float(cos[mask].max().item())
    return float(math.acos(max(min(min_cos, 1.0), -1.0)))


@dataclass
class SSLLossWeights:
    lambda_r: float = 0.05
    lambda_v: float = 0.05
    lambda_t: float = 0.05
    lambda_d: float = 1.0
    lambda_s: float = 1.0
    lambda_a: float = 1.0


def repulsion_margin_cos(n_active_cones: int, repulsion_margin_deg: float = 40.0) -> float:
    """
    Training repulsion target — softer than ETF-at-C_max, which is unreachable
    with few active cones and drives L_rep upward without bound.
    """
    if n_active_cones <= 1:
        return 1.0
    return cos_from_deg(repulsion_margin_deg)


def compute_ssl_losses(
    z_online: torch.Tensor,
    z_target: torch.Tensor,
    cone_bank: ConeBank,
    weights: SSLLossWeights,
    tau: float = 0.1,
    z_prev: Optional[torch.Tensor] = None,
    anchor_z: Optional[torch.Tensor] = None,
    anchor_z_star: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    label_to_cone: Optional[Dict[int, int]] = None,
    max_rep_pairs: int = 50_000,
    repulsion_margin_deg: float = 40.0,
) -> Tuple[torch.Tensor, SSLStepMetrics, Dict[str, torch.Tensor]]:
    """
    Full per-batch loss for Phase 1 / Phase 2 training.
    """
    usable = cone_bank.usable_indices()
    rays = cone_bank.rays_for_indices(usable, detach_frozen=True)

    l_consist = consistency_loss(z_online, z_target, rays, tau)
    cos_rep = repulsion_margin_cos(len(rays), repulsion_margin_deg)
    l_rep = cone_ray_repulsion_loss(rays, cos_rep, max_pairs=max_rep_pairs)
    l_vol = volume_preservation_loss(z_online)
    l_tight = tightness_loss(z_online, rays).clamp(min=-1.0, max=1.0)

    l_temp = z_online.new_zeros(())
    if z_prev is not None:
        l_temp = temporal_distillation_loss(z_online, z_prev)

    l_stab = z_online.new_zeros(())
    if anchor_z is not None and anchor_z_star is not None:
        l_stab = anchor_stability_loss(anchor_z, anchor_z_star)

    l_attr = z_online.new_zeros(())
    if labels is not None and label_to_cone is not None:
        mapped = {label_to_cone[lbl]: rays[label_to_cone[lbl]]
                  for lbl in label_to_cone if label_to_cone[lbl] in rays}
        l_attr = supervised_attraction_loss(
            z_online, labels, sorted(mapped.keys()), mapped, cone_bank.cos_gamma,
        )

    total = (
        l_consist
        + weights.lambda_r * l_rep
        + weights.lambda_v * l_vol
        + weights.lambda_t * l_tight
        + weights.lambda_d * l_temp
        + weights.lambda_s * l_stab
        + weights.lambda_a * l_attr
    )

    metrics = SSLStepMetrics(
        loss_consist=float(l_consist.item()),
        loss_rep=float(l_rep.item()),
        loss_vol=float(l_vol.item()),
        loss_tight=float(l_tight.item()),
        loss_temp=float(l_temp.item()),
        loss_stab=float(l_stab.item()),
        loss_attr=float(l_attr.item()),
    )
    parts = {
        "consist": l_consist, "rep": l_rep, "vol": l_vol, "tight": l_tight,
        "temp": l_temp, "stab": l_stab, "attr": l_attr,
    }
    return total, metrics, parts


@torch.no_grad()
def classify_nearest_vertex(
    encoder: OnlineEncoder,
    cone_bank: ConeBank,
    images: torch.Tensor,
) -> torch.Tensor:
    """argmax_c max_k ⟨z, v_c^(k)⟩ over frozen + active cones."""
    z = encoder(images)
    rays = cone_bank.rays_for_indices(cone_bank.usable_indices(), detach_frozen=True)
    cone_ids = sorted(rays.keys())
    if not cone_ids:
        return images.new_full((images.shape[0],), -1, dtype=torch.long)
    scores = cone_nearest_scores(z, rays)
    return torch.tensor(cone_ids, device=images.device)[scores.argmax(dim=1)]


@torch.no_grad()
def evaluate_cone_accuracy(
    encoder: OnlineEncoder,
    cone_bank: ConeBank,
    loader,
    device: torch.device,
    label_to_cone: Dict[int, int],
) -> float:
    """Linear-probe-style accuracy: label class must map to predicted cone."""
    cone_to_label = {v: k for k, v in label_to_cone.items()}
    correct = total = 0
    encoder.eval()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = classify_nearest_vertex(encoder, cone_bank, imgs)
        for p, y in zip(preds.tolist(), labels.tolist()):
            expected = label_to_cone.get(int(y))
            if expected is None:
                continue
            total += 1
            if int(p) == expected:
                correct += 1
    return correct / max(total, 1)
