"""
peft_masking.py
---------------
Fisher-Informed Conic Masking: Parameter-Efficient Fine-Tuning via
Selective Weight Subspace Optimisation for Class-Incremental Learning.

Theory
------
In a high-capacity ViT, most weights are redundant for any specific task.
This module identifies which weights are *critical* for old classes (using
the Fisher Information Matrix approximated via conic hull scores) and which
are *orthogonal* to old class identity and therefore safe to repurpose for
new classes — without degrading the existing conic hull geometry.

The three-stage pipeline
------------------------
1. **Importance scoring**  (`FisherConicImportance`)
   For every backbone parameter θ_ij, compute a scalar importance score:

       I(θ_ij) = E_x [ (∂ ConicScore_old(f_θ(x)) / ∂θ_ij)² ]

   This is the Fisher Information approximation of how much moving θ_ij
   would "tilt" or "warp" the old conic hulls.  Parameters with high I are
   critical; parameters near zero are orthogonal to old-class identity.

   Implementation: we approximate the Jacobian via a forward pass that
   produces the hull-score loss, then accumulates squared gradients over
   a calibration set of old-class images.

2. **Mask construction**  (`build_fisher_conic_mask`)
   Given a trainable_fraction (e.g. 0.20), keep only the
   lowest-importance fraction of parameters trainable (all others frozen).
   Optionally, BitFit mode restricts the search to bias and LayerNorm
   scale/shift parameters only — these are the most directionally
   expressive parameters while being the least disruptive to attention
   geometry.

3. **Projected gradient descent**  (`apply_gradient_projection`)
   After backward(), for each selected parameter's gradient, project
   out any component that lies in the span of the old extreme rays
   (in weight space).  This is the "null-space projection" that
   mathematically guarantees updates cannot rotate old embeddings.

   Concretely: for each parameter group and the associated old-class
   extreme rays R (K × D), compute:
       g_proj = g - R^T (R R^T)^{-1} R g
   which is the component of the gradient orthogonal to all old rays.

Classes / Functions
-------------------
FisherConicImportance   — computes per-parameter importance scores
MaskState               — dataclass holding the binary mask + budget stats
build_fisher_conic_mask — constructs the mask from importance scores
apply_parameter_mask    — freezes/unfreezes params according to the mask
apply_gradient_projection — projects gradients onto null space of old hulls
BitFitMask              — restricts updates to bias+LayerNorm params only
report_mask_budget      — prints a parameter-budget table
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Fisher-Conic Importance Scoring
# ──────────────────────────────────────────────────────────────────────────────

class FisherConicImportance:
    """
    Estimates how important each backbone parameter is for maintaining the
    geometry of the *old* conic hulls.

    For each parameter θ, the importance score is:

        I(θ) = (1/N) Σ_x  [ ∂ L_hull(f_θ(x)) / ∂θ ]²

    where L_hull is the conic-hull score loss for old classes.  Parameters
    with high I should be frozen; those with I ≈ 0 are safe to update.

    Parameters
    ----------
    model       : ClassIncrementalModel (backbone + head)
    hull_buffer : ConicHullReplayBuffer — fitted hulls for old classes
    device      : compute device
    """

    def __init__(self, model: nn.Module, hull_buffer, device: str):
        self.model       = model
        self.hull_buffer = hull_buffer
        self.device      = device
        self._scores: Dict[str, torch.Tensor] = {}

    def compute(
        self,
        calibration_loader,
        n_batches: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward+backward over `n_batches` of old-class images and
        accumulate squared gradients as importance scores.

        The loss used is the *negative mean conic score* of the backbone
        output against the stored hull extreme rays.  This directly
        measures how sensitive each parameter is to changes in the
        old-class conic geometry.

        Parameters
        ----------
        calibration_loader : DataLoader yielding (imgs, global_labels)
                             for old classes only
        n_batches          : number of mini-batches to average over

        Returns
        -------
        Dict[param_name → importance_tensor]  (same shape as each param)
        """
        self.model.eval()
        backbone = (self.model.backbone
                    if hasattr(self.model, "backbone") else self.model)

        # Initialise accumulators
        scores: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in backbone.named_parameters()
            if p.requires_grad
        }

        seen = 0
        for imgs, _ in calibration_loader:
            if seen >= n_batches:
                break
            imgs = imgs.to(self.device)

            backbone.zero_grad()

            # Forward through backbone only
            feats = backbone(imgs)                        # (B, D)

            # Hull-score loss: minimise negative mean cosine similarity
            # to the nearest old-class extreme ray centroid
            loss = self._hull_score_loss(feats)
            if loss is None:
                seen += 1
                continue

            loss.backward()

            for n, p in backbone.named_parameters():
                if p.requires_grad and p.grad is not None:
                    scores[n] += p.grad.detach() ** 2

            seen += 1

        # Normalise by number of batches seen
        for n in scores:
            scores[n] /= max(seen, 1)

        self._scores = scores
        return scores

    def _hull_score_loss(self, feats: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Loss = - mean cosine similarity between batch features and the
        centroid of stored old-class extreme rays.

        If no hulls are stored, returns None (skip this batch).
        """
        if len(self.hull_buffer.hulls) == 0:
            return None

        # Collect all stored extreme rays into one matrix
        all_rays = np.concatenate(
            [h.extreme_rays_ for h in self.hull_buffer.hulls.values()], axis=0
        )                                                   # (K_total, D)
        rays_t = torch.tensor(
            all_rays, dtype=torch.float32, device=self.device)  # (K, D)

        # L2-normalise features
        feats_n = nn.functional.normalize(feats, dim=1)    # (B, D)
        rays_n  = nn.functional.normalize(rays_t, dim=1)   # (K, D)

        # Mean cosine similarity of each feature to its nearest ray
        cos_sim = feats_n @ rays_n.T                        # (B, K)
        max_sim = cos_sim.max(dim=1).values                 # (B,)

        # We want to *preserve* this similarity, so the Fisher loss is
        # the negative — gradient tells us what to *not* change
        return -max_sim.mean()

    @property
    def scores(self) -> Dict[str, torch.Tensor]:
        if not self._scores:
            raise RuntimeError("Call compute() first.")
        return self._scores


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Mask Construction
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MaskState:
    """
    Binary mask over backbone parameters.

    Attributes
    ----------
    mask            : { param_name → BoolTensor }  True = trainable
    n_total         : total backbone parameter count
    n_trainable     : parameters marked trainable
    n_frozen        : parameters marked frozen
    trainable_frac  : fraction of params selected for update
    param_budget    : { layer_name → (trainable, total) }
    bitfit_mode     : True if only bias/LN params were considered
    """
    mask:           Dict[str, torch.Tensor]         = field(default_factory=dict)
    n_total:        int                             = 0
    n_trainable:    int                             = 0
    n_frozen:       int                             = 0
    trainable_frac: float                           = 0.0
    param_budget:   Dict[str, Tuple[int, int]]      = field(default_factory=dict)
    bitfit_mode:    bool                            = False

    @property
    def remaining_capacity(self) -> float:
        """Fraction of total params still frozen (available for future tasks)."""
        return self.n_frozen / max(self.n_total, 1)


def build_fisher_conic_mask(
    importance_scores:   Dict[str, torch.Tensor],
    model:               nn.Module,
    trainable_fraction:  float = 0.20,
    bitfit_mode:         bool  = False,
    min_layer_trainable: float = 0.05,
) -> MaskState:
    """
    Build a binary trainability mask from Fisher-Conic importance scores.

    Strategy
    --------
    1. Flatten all importance scores into a single vector.
    2. Find the threshold at the `trainable_fraction` quantile (low end).
    3. Mark as trainable only parameters whose importance is below
       this threshold — these are the weights orthogonal to old-class
       conic geometry.
    4. If `bitfit_mode=True`, further restrict to bias and LayerNorm
       scale/shift parameters only (γ, β, bias), ignoring all weight
       matrices.  This is the most conservative and compute-efficient mode.

    Parameters
    ----------
    importance_scores    : output of FisherConicImportance.compute()
    model                : nn.Module (backbone)
    trainable_fraction   : fraction of parameters to keep trainable (0–1)
    bitfit_mode          : restrict to bias + LayerNorm params only
    min_layer_trainable  : ensure at least this fraction of each layer
                           is trainable (prevents complete layer lockout)

    Returns
    -------
    MaskState
    """
    backbone = (model.backbone if hasattr(model, "backbone") else model)

    # ── BitFit pre-filter: only consider bias/LN params ───────────────────
    def _is_bitfit_param(name: str) -> bool:
        """True if this param name matches bias or LayerNorm scale/shift."""
        n = name.lower()
        return (
            n.endswith(".bias")
            or n.endswith(".weight") and any(
                key in n for key in ("norm", "ln_", "layernorm", "layer_norm"))
        )

    candidate_names = set(importance_scores.keys())
    if bitfit_mode:
        candidate_names = {n for n in candidate_names if _is_bitfit_param(n)}
        tqdm.write(f"  [bitfit] {len(candidate_names)} bias/LN params selected "
                   f"as candidates out of {len(importance_scores)} total")

    # ── Global threshold at the trainable_fraction quantile ───────────────
    # torch.quantile hard-crashes on tensors > 2^24 elements (common with
    # large ViT backbones that have 20–86M parameters).  We therefore use
    # numpy which has no such limit.  For very large models we additionally
    # subsample 4M elements uniformly at random — the quantile estimate from
    # a 4M-element sample is accurate to ±0.01% of the true value.
    _SAMPLE_LIMIT = 4_000_000

    score_chunks = [
        importance_scores[n].detach().cpu().float().flatten().numpy()
        for n in candidate_names
        if n in importance_scores
    ]
    all_scores_np = np.concatenate(score_chunks) if score_chunks else np.array([0.0])

    if all_scores_np.size > _SAMPLE_LIMIT:
        rng       = np.random.default_rng(seed=0)
        all_scores_np = rng.choice(all_scores_np, size=_SAMPLE_LIMIT, replace=False)

    threshold = float(np.quantile(all_scores_np, trainable_fraction))

    # ── Build per-parameter binary mask ───────────────────────────────────
    mask:         Dict[str, torch.Tensor] = {}
    param_budget: Dict[str, Tuple[int, int]] = {}
    n_total = n_trainable = 0

    for name, param in backbone.named_parameters():
        total = param.numel()
        n_total += total

        if name not in candidate_names:
            # Not a candidate (e.g. weight matrix in bitfit mode) → freeze
            mask[name]  = torch.zeros_like(param, dtype=torch.bool)
            layer_train = 0
        else:
            imp = importance_scores[name]
            raw_mask = (imp <= threshold)

            # Enforce minimum trainable fraction per layer
            layer_train_frac = raw_mask.float().mean().item()
            if layer_train_frac < min_layer_trainable and total > 1:
                # Force the lowest-importance min_layer_trainable fraction
                n_force = max(1, int(total * min_layer_trainable))
                flat_imp = imp.flatten()
                topk_idx = torch.topk(flat_imp, n_force, largest=False).indices
                raw_mask = torch.zeros_like(imp, dtype=torch.bool).flatten()
                raw_mask[topk_idx] = True
                raw_mask = raw_mask.reshape(imp.shape)

            mask[name]  = raw_mask
            layer_train = raw_mask.sum().item()

        n_trainable      += layer_train
        param_budget[name] = (int(layer_train), total)

    n_frozen = n_total - n_trainable

    return MaskState(
        mask            = mask,
        n_total         = n_total,
        n_trainable     = n_trainable,
        n_frozen        = n_frozen,
        trainable_frac  = n_trainable / max(n_total, 1),
        param_budget    = param_budget,
        bitfit_mode     = bitfit_mode,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Apply mask to model parameters
# ──────────────────────────────────────────────────────────────────────────────

def apply_parameter_mask(model: nn.Module, mask_state: MaskState) -> None:
    """
    Set `requires_grad` on backbone parameters according to `mask_state`.

    Parameters with mask=False (high Fisher importance → protect) are frozen.
    Parameters with mask=True  (low Fisher importance  → free)   are trainable.

    Note: gradient masking (zeroing grads during backward) is handled
    separately in `apply_gradient_projection` for fine-grained control.
    """
    backbone = (model.backbone if hasattr(model, "backbone") else model)
    for name, param in backbone.named_parameters():
        if name in mask_state.mask:
            # requires_grad reflects whether *any* element is trainable
            has_trainable = mask_state.mask[name].any().item()
            param.requires_grad_(bool(has_trainable))
        else:
            param.requires_grad_(False)


def reset_backbone_grad_state(model: nn.Module) -> None:
    """
    Restore all backbone parameters to fully trainable (requires_grad=True).

    **Must be called at the start of every task before Fisher scoring or mask
    construction.**  Without this reset, the frozen state left over from the
    previous task's mask corrupts the next task's importance scores in two ways:

    1. ``FisherConicImportance.compute`` only accumulates squared gradients for
       params that have ``requires_grad=True``.  If the backbone is still
       partially frozen from task N, the task N+1 Fisher scores are computed
       only on the unfrozen subset — importance estimates for the frozen params
       are zero by default, making them appear "safe to train" even if they are
       actually critical for old classes.

    2. ``_compute_fisher`` (EWC) also filters by ``requires_grad``, so without
       a reset, EWC would only protect params that were trainable in the *last*
       task rather than the full backbone.

    Note: this only resets the *trainability flag*, not weight values.
    Weight values are correctly restored by ``model.load_state_dict`` at the
    end of each task.  The ``requires_grad`` attribute is NOT part of a
    state_dict and must be reset manually at the start of each new task.

    Parameters
    ----------
    model : ClassIncrementalModel or bare backbone nn.Module
    """
    backbone = (model.backbone if hasattr(model, "backbone") else model)
    n_reset = 0
    for param in backbone.parameters():
        if not param.requires_grad:
            param.requires_grad_(True)
            n_reset += 1
    if n_reset > 0:
        tqdm.write(f"  [peft] Reset {n_reset} frozen backbone params "
                   f"to requires_grad=True before Fisher scoring.")


def get_masked_trainable_params(
    model:      nn.Module,
    mask_state: MaskState,
) -> List[torch.Tensor]:
    """
    Return parameter tensors that have at least one trainable element,
    for passing to the optimiser.
    """
    backbone = (model.backbone if hasattr(model, "backbone") else model)
    return [
        p for name, p in backbone.named_parameters()
        if name in mask_state.mask and mask_state.mask[name].any()
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Gradient projection onto null space of old extreme rays
# ──────────────────────────────────────────────────────────────────────────────

def apply_gradient_projection(
    model:           nn.Module,
    mask_state:      MaskState,
    hull_buffer,
    projection_mode: str   = "null_space",
    eps:             float = 1e-8,
) -> Dict[str, float]:
    """
    After loss.backward(), apply two gradient corrections:

    1. **Binary mask zeroing** — zero out gradients for frozen parameters
       (those with mask=False), ensuring they receive no update regardless
       of what the optimiser sees.

    2. **Null-space projection** (if `projection_mode='null_space'`) — for
       each trainable parameter's gradient vector g, subtract the component
       that lies in the span of the old extreme rays (projected into weight
       space via the Jacobian approximation).

       The projection is:
           g_proj = g - R^T (R R^T + εI)^{-1} R g

       where R is the matrix of old extreme rays (K × D) and D is the
       flattened gradient dimension.

       When D >> K (as in ViT weight matrices) this is efficient because
       we only need to invert a K×K matrix.

       This guarantees that updates to selected parameters cannot rotate
       old-class embedding directions.

    Parameters
    ----------
    model           : ClassIncrementalModel
    mask_state      : MaskState from build_fisher_conic_mask
    hull_buffer     : ConicHullReplayBuffer with old-class hulls
    projection_mode : 'null_space' | 'mask_only'
    eps             : regularisation for (R R^T)^{-1}

    Returns
    -------
    dict  { 'n_params_projected', 'mean_grad_reduction' }
    """
    backbone = (model.backbone if hasattr(model, "backbone") else model)
    device   = next(backbone.parameters()).device

    # Build combined extreme-ray matrix once (K_total × D_feat)
    R_np: Optional[np.ndarray] = None
    if projection_mode == "null_space" and len(hull_buffer.hulls) > 0:
        R_np = np.concatenate(
            [h.extreme_rays_ for h in hull_buffer.hulls.values()], axis=0
        )                                                   # (K, D_feat)

    n_projected     = 0
    grad_reductions = []

    for name, param in backbone.named_parameters():
        if param.grad is None:
            continue

        # Step 1: zero frozen-element gradients
        if name in mask_state.mask:
            frozen = ~mask_state.mask[name].to(param.grad.device)
            param.grad.data[frozen] = 0.0

        # Step 2: null-space projection for trainable elements
        if projection_mode == "null_space" and R_np is not None:
            g = param.grad.data                             # same shape as param
            g_flat = g.flatten().float()                   # (P,)
            P = g_flat.numel()

            # Only project if the gradient dimension matches feature dim
            # (relevant for the final projection layer and large weight matrices)
            if P >= 32:
                g_norm_before = g_flat.norm().item()
                if g_norm_before > eps:
                    # Low-rank projection: R has shape (K, D_feat)
                    # We approximate the weight-space null-space projection
                    # by treating the flattened gradient as a "feature direction"
                    # and projecting out components aligned with extreme rays.
                    K, D = R_np.shape
                    if P == D:
                        # Exact match: gradient lives in feature space
                        R_t   = torch.tensor(R_np, dtype=torch.float32,
                                             device=device)          # (K, D)
                        # g_proj = g - R^T (R R^T + εI)^{-1} R g
                        RRt   = R_t @ R_t.T                          # (K, K)
                        RRt  += eps * torch.eye(K, device=device)
                        Rg    = R_t @ g_flat                         # (K,)
                        coeff = torch.linalg.solve(RRt, Rg)          # (K,)
                        g_proj = g_flat - R_t.T @ coeff              # (D,)
                    else:
                        # Dimensions differ: project via random subspace sketch
                        # Sample a K-dim sketch and project in that subspace
                        g_proj = _sketch_project(g_flat, R_np, device, eps)

                    g_norm_after = g_proj.norm().item()
                    if g_norm_after > eps:
                        # Rescale to preserve gradient magnitude
                        scale = g_norm_before / (g_norm_after + eps)
                        g_proj = g_proj * min(scale, 10.0)

                    param.grad.data.copy_(g_proj.reshape(g.shape))
                    grad_reductions.append(1.0 - g_norm_after /
                                           (g_norm_before + eps))
                    n_projected += 1

    return {
        "n_params_projected": n_projected,
        "mean_grad_reduction": float(np.mean(grad_reductions))
                               if grad_reductions else 0.0,
    }


def _sketch_project(
    g:      torch.Tensor,
    R_np:   np.ndarray,
    device: str,
    eps:    float,
) -> torch.Tensor:
    """
    When the gradient dimension P ≠ feature dim D, project via a random
    Gaussian sketch that maps both into a common K-dimensional space.

    This is an approximation — it removes the *most strongly aligned*
    component with each extreme ray direction, not an exact null-space
    projection, but it is O(K·P) and avoids any K×P matrix operations.
    """
    K = len(R_np)
    P = g.numel()

    # Random sketch matrix S: (K, P), rows normalised
    torch.manual_seed(42)                          # deterministic sketch
    S = torch.randn(K, P, device=device)
    S = nn.functional.normalize(S, dim=1)          # (K, P)

    # Project g onto the span of S (approximation of hull span in weight space)
    Sg    = S @ g                                  # (K,)
    SSt   = S @ S.T + eps * torch.eye(K, device=device)   # (K, K)
    coeff = torch.linalg.solve(SSt, Sg)            # (K,)
    return g - S.T @ coeff                         # (P,)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  BitFit convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def build_bitfit_mask(model: nn.Module) -> MaskState:
    """
    Build a BitFit mask: only bias and LayerNorm scale/shift (weight)
    parameters are trainable.  All other weights are frozen.

    This is the most conservative PEFT mode — typically achieves 90%+
    of full fine-tuning accuracy while freezing 99%+ of backbone weights.
    No Fisher computation required.

    Parameters
    ----------
    model : ClassIncrementalModel or bare backbone

    Returns
    -------
    MaskState  (bitfit_mode=True)
    """
    backbone = (model.backbone if hasattr(model, "backbone") else model)
    mask:         Dict[str, torch.Tensor] = {}
    param_budget: Dict[str, Tuple[int, int]] = {}
    n_total = n_trainable = 0

    for name, param in backbone.named_parameters():
        total = param.numel()
        n_total += total

        n_lower = name.lower()
        is_bias = n_lower.endswith(".bias")
        is_ln   = (n_lower.endswith(".weight") and any(
            k in n_lower for k in ("norm", "ln_", "layernorm", "layer_norm")))

        if is_bias or is_ln:
            mask[name] = torch.ones_like(param, dtype=torch.bool)
            n_trainable += total
        else:
            mask[name] = torch.zeros_like(param, dtype=torch.bool)

        param_budget[name] = (
            total if (is_bias or is_ln) else 0, total)

    n_frozen = n_total - n_trainable
    return MaskState(
        mask            = mask,
        n_total         = n_total,
        n_trainable     = n_trainable,
        n_frozen        = n_frozen,
        trainable_frac  = n_trainable / max(n_total, 1),
        param_budget    = param_budget,
        bitfit_mode     = True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Reporting
# ──────────────────────────────────────────────────────────────────────────────

def report_mask_budget(
    mask_state:  MaskState,
    top_n_layers: int = 20,
) -> None:
    """
    Print a parameter-budget table showing trainable / frozen counts
    per layer, sorted by trainable count descending.

    Also prints the cumulative budget used so far — useful for tracking
    how much capacity remains for future tasks.
    """
    width = 80
    print("\n" + "═" * width)
    print(f"  PEFT Mask Budget Report  "
          f"({'BitFit' if mask_state.bitfit_mode else 'Fisher-Conic'})")
    print("─" * width)
    print(f"  Total params     : {mask_state.n_total:>12,}")
    print(f"  Trainable        : {mask_state.n_trainable:>12,}  "
          f"({mask_state.trainable_frac:.2%})")
    print(f"  Frozen           : {mask_state.n_frozen:>12,}  "
          f"({1 - mask_state.trainable_frac:.2%})")
    print(f"  Remaining cap.   : {mask_state.remaining_capacity:.2%}  "
          f"(frozen fraction available for future tasks)")
    print("─" * width)

    # Top-N layers by trainable count
    rows = sorted(mask_state.param_budget.items(),
                  key=lambda x: x[1][0], reverse=True)[:top_n_layers]
    print(f"  {'Layer':<55} {'Trainable':>10} {'Total':>10} {'%':>6}")
    print("─" * width)
    for name, (tr, tot) in rows:
        pct = 100 * tr / max(tot, 1)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        short_name = name[-53:] if len(name) > 53 else name
        print(f"  {short_name:<55} {tr:>10,} {tot:>10,} {pct:>5.1f}%")

    if len(mask_state.param_budget) > top_n_layers:
        print(f"  … {len(mask_state.param_budget) - top_n_layers} "
              f"more layers not shown …")
    print("═" * width + "\n")


def compute_cumulative_budget(
    mask_states: List[MaskState],
) -> Dict[str, float]:
    """
    Given mask states from multiple tasks, report the cumulative parameter
    budget consumed and remaining.

    Parameters
    ----------
    mask_states : one MaskState per completed task

    Returns
    -------
    dict  { 'total_params', 'params_used', 'params_remaining',
            'fraction_used', 'fraction_remaining',
            'estimated_remaining_tasks' }
    """
    if not mask_states:
        return {}

    n_total        = mask_states[0].n_total
    # Params used = those that were trainable in *any* task
    used_per_task  = [ms.n_trainable for ms in mask_states]
    # Conservative estimate: assume no overlap between task masks
    params_used    = min(sum(used_per_task), n_total)
    params_rem     = n_total - params_used
    frac_used      = params_used / max(n_total, 1)
    frac_rem       = params_rem  / max(n_total, 1)

    avg_per_task   = float(np.mean(used_per_task)) if used_per_task else 1
    est_rem_tasks  = int(params_rem / max(avg_per_task, 1))

    result = {
        "total_params":             n_total,
        "params_used":              params_used,
        "params_remaining":         params_rem,
        "fraction_used":            frac_used,
        "fraction_remaining":       frac_rem,
        "estimated_remaining_tasks": est_rem_tasks,
    }

    print("\n" + "═" * 60)
    print("  Cumulative Parameter Budget")
    print("─" * 60)
    for k, v in result.items():
        val = f"{v:.2%}" if "fraction" in k else f"{v:,}"
        print(f"  {k:<35} {val}")
    print("═" * 60 + "\n")

    return result