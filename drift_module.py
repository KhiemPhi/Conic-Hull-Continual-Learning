"""
drift_module.py
---------------
Learned inter-stage drift correction.

Instead of estimating a single affine map A (Procrustes/covariance) to rotate old
hulls forward, we *learn* a nonlinear map g that takes a CURRENT-backbone
embedding and projects it BACK into a previous stage's embedding space, where the
old-class hulls/prototypes were built:

    g_t :  phi_t(x)  ->  phi_{t-1}(x)

Training is fully supervised and exemplar-cheap: for replay images x we have both
phi_t(x) (current backbone) and phi_{t-1}(x) (the frozen old backbone the pipeline
already keeps), giving paired targets. For an old class born at stage s, a query
in stage-T space is back-projected by composing g_{s+1} o ... o g_T, then scored
against that class's frozen hull — so hulls never need to be rotated/renormalized.

A residual MLP is the natural choice for vector->vector drift (it starts at
identity). A transformer would need a token sequence (e.g. patch tokens); the
class-token-only setting here is a pure vector map, so MLP it is.
"""
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DriftMLP(nn.Module):
    """Residual MLP: g(z) = z + f(z). Starts near identity (last layer zero-init)."""

    def __init__(self, dim: int, hidden: int = 2048, depth: int = 2,
                 dropout: float = 0.0, residual: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        self.proj = nn.Linear(d, dim)
        nn.init.zeros_(self.proj.weight)            # start as identity
        nn.init.zeros_(self.proj.bias)
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(self.net(x))
        return x + out if self.residual else out


def fit_drift(
    g: DriftMLP,
    X_new: np.ndarray,
    X_old: np.ndarray,
    device,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 512,
    cos_weight: float = 1.0,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> float:
    """Train g so that g(phi_t(x)) ~= phi_{t-1}(x).

    Loss = MSE + cos_weight * (1 - cosine). The cosine term matters because the
    downstream hull score is angular.
    """
    torch.manual_seed(seed)
    Xn = torch.as_tensor(np.asarray(X_new, dtype=np.float32), device=device)
    Xo = torch.as_tensor(np.asarray(X_old, dtype=np.float32), device=device)
    g.to(device).train()
    opt = torch.optim.AdamW(g.parameters(), lr=lr, weight_decay=weight_decay)
    n = Xn.shape[0]
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = g(Xn[idx])
            mse = F.mse_loss(pred, Xo[idx])
            cos = (1.0 - F.cosine_similarity(pred, Xo[idx], dim=1)).mean()
            loss = mse + cos_weight * cos
            loss.backward()
            opt.step()
            last = float(loss.item())
    g.float().eval()   # ensure the stored map is fp32 (avoids dtype mismatches at eval)
    return last


@torch.no_grad()
def backproj_static_hull_accuracy(backbone, test_loader, static_hulls, birth_stage,
                                  drift_maps, current_stage, device) -> float:
    """Old-class accuracy when queries are back-projected into each class's birth
    space before scoring its frozen static hull.

    For a hull of class c (born stage s), the current-backbone features are mapped
    phi_T -> phi_s by composing g_T, g_{T-1}, ..., g_{s+1}, then scored against the
    (un-rotated) birth-space hull. Hulls must be in birth space (i.e. run with
    rotate_static_hulls=False).
    """
    from features import _infer_autocast
    backbone.eval()
    feats, ys = [], []
    with _infer_autocast(device):
        for imgs, y in test_loader:
            feats.append(backbone(imgs.to(device)).float().cpu().numpy())
            ys += y.tolist()
    if not feats:
        return float("nan")
    F = np.concatenate(feats)
    ys = np.array(ys)
    classes = sorted(static_hulls.keys(), key=lambda k: int(k))

    _dtype = device.type if hasattr(device, "type") else (
        "cuda" if str(device).startswith("cuda") else "cpu")
    cache = {}   # birth stage -> back-projected F (computed once per stage)
    def _bp(s):
        if s in cache:
            return cache[s]
        gs = [drift_maps[t] for t in range(current_stage, s, -1) if t in drift_maps]
        if not gs:
            cache[s] = F
            return F
        # Force fp32 and disable any ambient autocast: the drift MLP and the
        # query must share dtype (a stray bf16 cast silently skipped this eval).
        with torch.autocast(device_type=_dtype, enabled=False):
            z = torch.as_tensor(F, device=device, dtype=torch.float32)
            for g in gs:
                z = g.to(device).float().eval()(z)
        cache[s] = z.detach().float().cpu().numpy()
        return cache[s]

    cols = [static_hulls[k].score(_bp(birth_stage.get(int(k), current_stage)))
            for k in classes]
    S = np.column_stack(cols)
    pred = np.array([int(classes[i]) for i in np.argmax(S, 1)])
    return float((pred == ys).mean())


@torch.no_grad()
def apply_drift_chain(gs: Sequence[DriftMLP], X: np.ndarray, device) -> np.ndarray:
    """Back-project X (current-stage embeddings) through a chain of maps applied
    in order gs[0], gs[1], ... (caller supplies them newest-first: g_T, g_{T-1},
    ..., g_{s+1}) to reach the target birth space."""
    z = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    for g in gs:
        z = g.to(device).eval()(z)
    return z.cpu().numpy()
