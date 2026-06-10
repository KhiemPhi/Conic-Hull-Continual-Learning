"""
Globally geometrically-aware cone loss (CE replacement).

Three decoupled terms with a shared angular margin gamma*:

  L_attr : attract features to the nearest ray of their own class cone
  L_rep  : repel extreme rays of different classes (all pairs, every step)
  L_marg : repel batch features from rays of other classes

Set training_loss="geometric" in the incremental pipeline, or "ce" for vanilla CE.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_from_deg(deg: float) -> float:
    return float(math.cos(math.radians(deg)))


def repulsion_kernel(
    u: torch.Tensor,
    cos_gamma: float,
    kernel: str = "hinge_sq",
    beta: float = 10.0,
) -> torch.Tensor:
    """
    κ(u) with target margin cos(γ*): penalise u > cos_gamma.

    hinge_sq  : (relu(u - cos_gamma))^2
    softplus  : softplus(β (u - cos_gamma)) / β  → hinge_sq in the β→∞ limit
    squared   : (u - cos_gamma)^2  (always active)
    """
    if kernel == "hinge_sq":
        return F.relu(u - cos_gamma).pow(2)
    if kernel == "softplus":
        return F.softplus(beta * (u - cos_gamma)) / beta
    if kernel == "squared":
        return (u - cos_gamma).pow(2)
    raise ValueError(f"Unknown geometric kernel {kernel!r}")


def attraction_kernel(
    cos_own: torch.Tensor,
    cos_gamma: float,
    kernel: str = "hinge_sq",
    beta: float = 10.0,
) -> torch.Tensor:
    """Penalise cos_own < cos_gamma (feature too far from own cone)."""
    if kernel == "hinge_sq":
        return F.relu(cos_gamma - cos_own).pow(2)
    if kernel == "softplus":
        return F.softplus(beta * (cos_gamma - cos_own)) / beta
    if kernel == "squared":
        return (cos_gamma - cos_own).pow(2)
    raise ValueError(f"Unknown geometric kernel {kernel!r}")


def class_direction_from_head(
    head: nn.Module,
    class_id: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Unit direction for class *class_id* from a cosine / MLP head. Shape (d,)."""
    if not hasattr(head, "weight"):
        return None
    W = head.weight
    if not isinstance(W, torch.Tensor) or W.numel() == 0:
        return None
    if class_id < 0 or class_id >= W.shape[0]:
        return None
    return F.normalize(W[class_id].to(device), p=2, dim=0)


def build_training_rays(
    head: nn.Module,
    class_ids: List[int],
    device: torch.device,
    cone_memory=None,
    hull_rays: Optional[Dict[int, torch.Tensor]] = None,
    freeze_stored_rays: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Per-class extreme rays for the geometric loss.

    Priority: cone_memory → hull_rays → head direction (single ray).
    Stored rays are detached by default so only the backbone / head learn.
    """
    rays: Dict[int, torch.Tensor] = {}
    for c in class_ids:
        V: Optional[torch.Tensor] = None
        if cone_memory is not None and c in cone_memory.cones:
            V = cone_memory.cones[c]
        elif hull_rays is not None and c in hull_rays:
            V = hull_rays[c]
        else:
            w = class_direction_from_head(head, c, device)
            if w is not None:
                V = w.unsqueeze(0)

        if V is None or V.numel() == 0:
            continue
        V = F.normalize(V.to(device), p=2, dim=1)
        if freeze_stored_rays and (
            (cone_memory is not None and c in cone_memory.cones)
            or (hull_rays is not None and c in hull_rays)
        ):
            V = V.detach()
        rays[int(c)] = V
    return rays


def cone_attraction_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    rays_by_class: Dict[int, torch.Tensor],
    cos_gamma: float,
    kernel: str = "hinge_sq",
    beta: float = 10.0,
) -> torch.Tensor:
    """L_attr:  E[(cos γ* - max_k ⟨z, v_c^(k)⟩)_+^2] on the batch."""
    if not rays_by_class:
        return z.new_zeros(())

    z = F.normalize(z, p=2, dim=1)
    per_sample: list = []
    for i, c in enumerate(labels.tolist()):
        V = rays_by_class.get(int(c))
        if V is None:
            continue
        cos_own = (z[i : i + 1] @ V.T).max(dim=1).values.squeeze(0)
        per_sample.append(attraction_kernel(cos_own, cos_gamma, kernel, beta))

    if not per_sample:
        return z.new_zeros(())
    return torch.stack(per_sample).mean()


def cone_ray_repulsion_loss(
    rays_by_class: Dict[int, torch.Tensor],
    cos_gamma: float,
    kernel: str = "hinge_sq",
    beta: float = 10.0,
    max_pairs: Optional[int] = None,
) -> torch.Tensor:
    """
    L_rep: mean κ(⟨v_i, v_j⟩) over all cross-class ray pairs (i ≠ j, c_i ≠ c_j).
    """
    class_ids = sorted(rays_by_class.keys())
    if len(class_ids) < 2:
        ref = next(iter(rays_by_class.values()), None)
        return ref.new_zeros(()) if ref is not None else torch.zeros(())

    parts: list = []
    cls_tags: list = []
    for c in class_ids:
        V = F.normalize(rays_by_class[c], p=2, dim=1)
        parts.append(V)
        cls_tags.extend([c] * V.shape[0])

    V_all = torch.cat(parts, dim=0)
    cls_t = torch.tensor(cls_tags, device=V_all.device, dtype=torch.long)
    cos = V_all @ V_all.T
    n = V_all.shape[0]
    upper = torch.triu(torch.ones(n, n, device=V_all.device, dtype=torch.bool), diagonal=1)
    cross = cls_t.unsqueeze(0) != cls_t.unsqueeze(1)
    mask = upper & cross

    if not mask.any():
        return V_all.new_zeros(())

    idx = mask.nonzero(as_tuple=False)
    if max_pairs is not None and idx.shape[0] > max_pairs:
        perm = torch.randperm(idx.shape[0], device=V_all.device)[:max_pairs]
        idx = idx[perm]

    u = cos[idx[:, 0], idx[:, 1]]
    return repulsion_kernel(u, cos_gamma, kernel, beta).mean()


def cone_feature_margin_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    rays_by_class: Dict[int, torch.Tensor],
    cos_gamma: float,
    kernel: str = "hinge_sq",
    beta: float = 10.0,
) -> torch.Tensor:
    """
    L_marg: E[ (1/((C-1)K)) Σ_{c'≠c,k} κ(⟨z, v_{c'}^(k)⟩) ].
    """
    if not rays_by_class:
        return z.new_zeros(())

    z = F.normalize(z, p=2, dim=1)
    all_parts: list = []
    cls_tags: list = []
    for c, V in rays_by_class.items():
        V = F.normalize(V.to(z.device), p=2, dim=1)
        all_parts.append(V)
        cls_tags.extend([c] * V.shape[0])
    V_all = torch.cat(all_parts, dim=0)
    cls_t = torch.tensor(cls_tags, device=z.device, dtype=torch.long)

    cos = z @ V_all.T
    per_sample: list = []
    for i, c in enumerate(labels.tolist()):
        other = cls_t != int(c)
        if not other.any():
            continue
        viol = repulsion_kernel(cos[i, other], cos_gamma, kernel, beta)
        per_sample.append(viol.mean())

    if not per_sample:
        return z.new_zeros(())
    return torch.stack(per_sample).mean()


def geometric_alignment_metrics(
    z: torch.Tensor,
    labels: torch.Tensor,
    rays_by_class: Dict[int, torch.Tensor],
) -> Dict[str, float]:
    """Diagnostics: own-cone alignment and worst other-class alignment."""
    z = F.normalize(z, p=2, dim=1)
    own_vals: list = []
    worst_other: list = []
    for i, c in enumerate(labels.tolist()):
        V_own = rays_by_class.get(int(c))
        if V_own is not None:
            own_vals.append(float((z[i : i + 1] @ V_own.T).max().item()))
        max_other = -1.0
        for c2, V in rays_by_class.items():
            if c2 == int(c):
                continue
            max_other = max(max_other, float((z[i : i + 1] @ V.T).max().item()))
        if max_other >= 0:
            worst_other.append(max_other)
    out: Dict[str, float] = {}
    if own_vals:
        out["own_align"] = sum(own_vals) / len(own_vals)
    if worst_other:
        out["worst_other"] = sum(worst_other) / len(worst_other)
    return out


def geometric_cone_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    rays_by_class: Dict[int, torch.Tensor],
    cos_gamma: float,
    lambda_attr: float = 1.0,
    lambda_rep: float = 0.1,
    lambda_marg: float = 0.5,
    kernel: str = "hinge_sq",
    beta: float = 10.0,
    max_rep_pairs: Optional[int] = 50_000,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Full globally-aware cone loss. Returns (total, components, metrics).
    """
    l_attr = cone_attraction_loss(z, labels, rays_by_class, cos_gamma, kernel, beta)
    l_rep = cone_ray_repulsion_loss(
        rays_by_class, cos_gamma, kernel, beta, max_pairs=max_rep_pairs,
    )
    l_marg = cone_feature_margin_loss(z, labels, rays_by_class, cos_gamma, kernel, beta)

    total = lambda_attr * l_attr + lambda_rep * l_rep + lambda_marg * l_marg
    parts = {"attr": l_attr, "rep": l_rep, "marg": l_marg}
    metrics = geometric_alignment_metrics(z, labels, rays_by_class)
    return total, parts, metrics
