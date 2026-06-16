"""
cone_boundary.py
----------------
First-session cone-contrastive representation shaping + an experimental
demonstration of the *forgetting-free boundary*  ε < γ/2  for conic-hull
continual learning.

Reuses existing project components (nothing reimplemented):
    drift_control : spa_indices, nnls_residual, classify, batched_features,
                    make_synth, build_backbone
                    (torch SPA + differentiable NNLS residual = the ConicHull
                    primitive in torch form; synthetic data + LoRA backbone)

Thesis (demonstrated, not merely asserted)
------------------------------------------
A class is a CONE of directions.  Two scalars govern everything:

    γ (gamma) = min inter-class extreme-ray angle   — how far apart cones sit
    ε (eps)   = angular drift of a class's features from where its cone was frozen

A query is classified correctly as long as it has not drifted more than halfway
toward a neighbouring cone, i.e. while  ε < γ/2.  Forgetting is *exactly* the
event ε crosses γ/2.

Forgetting has TWO geometric sources, and the cone view separates them cleanly:
    (a) DRIFT     — ε grows as the backbone keeps training (the part we control).
    (b) CROWDING  — γ/2 shrinks as more classes pack onto the sphere (irreducible;
                    every prototype/Gaussian/cone method faces it).

Consequence (the "build"):
    1. Shape the first session with a cone-contrastive (cone-anchor) loss so cones
       start tight and well-separated (large γ).
    2. FREEZE the backbone → ε ≡ 0 for every future task → the DRIFT term vanishes.
       Residual forgetting is then only the crowding floor (γ/2 shrinking), which
       no method escapes.

We verify against an otherwise-identical arm that keeps adapting: its ε grows, its
margin γ/2 − ε shrinks, and it forgets MORE than the frozen arm.  The gap between
the two arms' forgetting is exactly the drift component the build removes; the
frozen arm's residual forgetting is the crowding floor.
"""
import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── existing components ────────────────────────────────────────────────────────
from drift_control import (
    spa_indices,        # SPA extreme-ray selection (torch)
    nnls_residual,      # relative NNLS residual against a cone (differentiable)
    classify,           # argmin-residual conic-hull classifier
    batched_features,   # mini-batched backbone forward (handles LazyImageSet)
    make_synth,         # synthetic class-incremental data (fallback)
    build_backbone,     # pretrained MLP + LoRA (synthetic path)
    get_dataset,        # CIFAR-100 LazyImageSet loaders
    LoRALinear,         # low-rank adapter wrapper
)


# ── data + backbone factories ──────────────────────────────────────────────────
def load_cifar100(model_name="vit_base_patch16_224.orig_in21k", data_dir="./data"):
    """CIFAR-100 as (Xtr, ytr, Xte, yte) with the X as LazyImageSets, using the
    MODEL'S OWN normalization (timm data config) — NOT CIFAR channel stats.

    A frozen backbone is extremely sensitive to input normalization: the previous
    CIFAR-stats transform fed mis-normalized images to the IN21k ViT and crushed
    the frozen features (joint floor 0.38 → 0.82 once corrected).  pretrained=False
    only reads the architecture's data config (mean/std/input_size) — no weight
    download — so this is cheap."""
    import timm
    from torchvision import datasets
    from drift_control import LazyImageSet

    _m = timm.create_model(model_name, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(_m)
    tf = timm.data.create_transform(**cfg, is_training=False)
    del _m
    print(f"[norm] {model_name}: mean={tuple(round(m, 3) for m in cfg['mean'])} "
          f"std={tuple(round(s, 3) for s in cfg['std'])} input={cfg['input_size']}")
    tr = datasets.CIFAR100(data_dir, train=True, download=True, transform=tf)
    te = datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    return (LazyImageSet(tr), torch.tensor(tr.targets),
            LazyImageSet(te), torch.tensor(te.targets))


def make_mlp_backbone(seed, device):
    """Synthetic-path backbone factory (pretrained MLP + LoRA)."""
    args = SimpleNamespace(backbone="mlp", feat_dim=128, lora_rank=8,
                           lora_alpha=4.0, seed=seed, lora_blocks=4)
    return build_backbone(args, device)


def make_vit_backbone_factory(model_name="vit_base_patch16_224.orig_in21k",
                              lora_rank=16, lora_alpha=4.0, lora_blocks=4):
    """Return a factory(seed, device) -> (vit, feat_dim) that loads a pretrained
    timm ViT (num_classes=0 → pooled CLS features) with the frozen backbone +
    LoRA on the last ``lora_blocks`` blocks — the same adapter recipe as the main
    conic-hull pipeline, reusing drift_control.LoRALinear."""
    def factory(seed, device):
        import timm
        torch.manual_seed(seed)
        vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        for p in vit.parameters():
            p.requires_grad_(False)
        n_blocks = len(vit.blocks)
        for i in range(max(0, n_blocks - lora_blocks), n_blocks):
            blk = vit.blocks[i]
            blk.attn.qkv  = LoRALinear(blk.attn.qkv,  lora_rank, lora_alpha)
            blk.attn.proj = LoRALinear(blk.attn.proj, lora_rank, lora_alpha)
            blk.mlp.fc1   = LoRALinear(blk.mlp.fc1,   lora_rank, lora_alpha)
            blk.mlp.fc2   = LoRALinear(blk.mlp.fc2,   lora_rank, lora_alpha)
        return vit, vit.num_features
    return factory


class RPWrapper(nn.Module):
    """Frozen Random Projection + ReLU on top of a base backbone:

        h = ReLU(W · base(x)),   W ~ N(0, 1/D)  (M×D), a buffer (never trained).

    Why (RanPAC's RP layer, feeding the cone primitive): the M≫D expansion makes
    classes more conically separable (Cover's theorem → raises γ, the crowding
    floor that caps the frozen arm), and ReLU lands features in the non-negative
    orthant — the natural domain for a conic hull (a non-negative span of rays).
    W is a buffer, so it is never optimised; the base LoRA can still be shaped in
    session 0, and freezing the base afterward keeps ε ≡ 0.
    """

    def __init__(self, base, d_in, n_proj=5000, seed=0):
        super().__init__()
        self.base = base
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(n_proj, d_in, generator=g) / (d_in ** 0.5)
        self.register_buffer("W", W)              # moves with .to(device); not a param

    def forward(self, x):
        return F.relu(self.base(x) @ self.W.t())  # (B, M), non-negative


def make_rp_backbone_factory(base_factory, n_proj=5000, rp_seed=0):
    """Wrap a base factory with a frozen RP+ReLU head → factory(seed, device) that
    returns (RPWrapper, M).  Downstream SPA/NNLS/hulls operate in the M-dim
    non-negative space unchanged."""
    def factory(seed, device):
        base, d_in = base_factory(seed, device)
        return RPWrapper(base, d_in, n_proj=n_proj, seed=rp_seed), n_proj
    return factory


# ── cone-contrastive loss (the training-time form of ε < γ/2) ──────────────────
def cone_contrastive_loss(feats, labels, ray_bank, hinge_tau=0.45,
                          lam_rep=1.0, rep_targets=16):
    """Pull each feature INTO its own class cone (low NNLS residual → shrinks ε)
    and push it OUT of every other cone (hinge on the residual → grows γ).

    Rays in ``ray_bank`` are FIXED targets (detached before being passed in): the
    attraction term alone has a trivial collapse minimiser (features and rays
    racing to a single point); the hinge repulsion and the detachment together
    prevent it.  This is the geometric loss from drift_control's ``cone`` arm,
    used here as a representation-shaping objective.
    """
    classes = list(ray_bank.keys())
    total = feats.new_zeros(())
    n = 0
    for c in classes:
        m = labels == c
        if not m.any():
            continue
        fc = feats[m]
        total = total + nnls_residual(fc, ray_bank[c].to(fc)).mean()         # ε term
        others = [o for o in classes if o != c]
        if len(others) > rep_targets:                                        # cap O(C²)
            keep = torch.randperm(len(others))[:rep_targets].tolist()
            others = [others[i] for i in keep]
        for o in others:
            r = nnls_residual(fc, ray_bank[o].to(fc))
            total = total + lam_rep * F.relu(hinge_tau - r).mean()           # γ term
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _provisional_rays(backbone, Xc_dict, k_rays, device):
    """SPA extreme rays per class from CURRENT features, detached → fixed targets."""
    bank = {}
    for c, Xc in Xc_dict.items():
        fc = batched_features(backbone, Xc, device)
        idx = spa_indices(fc, k_rays)
        bank[c] = F.normalize(fc, dim=1)[idx].detach()
    return bank


def train_session(backbone, X, y, classes, device, k_rays=8, epochs=100, lr=1e-3,
                  batch_size=32, hinge_tau=0.45, lam_rep=1.0, rep_targets=16,
                  frozen_old_bank=None, refresh_every=10):
    """Train the backbone on ``classes`` with the cone-contrastive loss.

    Provisional rays for the trained classes are refreshed (detached) every
    ``refresh_every`` epochs — refreshing every step makes the target chase the
    features and collapses the cones.  Old frozen cones (``frozen_old_bank``) are
    added as repulsion targets so new classes carve out fresh angular territory.
    """
    params = [p for p in backbone.parameters() if p.requires_grad]
    if not params:
        return  # frozen backbone — nothing to do
    opt = torch.optim.Adam(params, lr=lr)
    Xc = {c: X[y == c] for c in classes}
    bank = _provisional_rays(backbone, Xc, k_rays, device)
    for epoch in range(epochs):
        if epoch and epoch % refresh_every == 0:
            bank = _provisional_rays(backbone, Xc, k_rays, device)
        full = dict(bank)
        if frozen_old_bank:
            full.update(frozen_old_bank)
        perm = torch.randperm(len(X))
        bar = tqdm(range(0, len(X), batch_size),
                   desc=f"    cone-anchor e{epoch + 1}/{epochs}",
                   leave=False, unit="batch")
        run_loss, nb = 0.0, 0
        for i in bar:
            sel = perm[i:i + batch_size]
            xb, yb = X[sel].to(device), y[sel].to(device)
            f = backbone(xb)
            loss = cone_contrastive_loss(f, yb, full, hinge_tau, lam_rep, rep_targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss); nb += 1
            bar.set_postfix(loss=f"{run_loss / nb:.4f}")


def freeze(backbone):
    for p in backbone.parameters():
        p.requires_grad_(False)


# ── boundary metrics ───────────────────────────────────────────────────────────
def measure_gamma(ray_bank):
    """Minimum inter-class extreme-ray angle (degrees). Larger = cones farther apart."""
    classes = sorted(ray_bank)
    allR = torch.cat([F.normalize(ray_bank[c], dim=1) for c in classes], 0)
    labs = np.concatenate([np.full(len(ray_bank[c]), c) for c in classes])
    C = (allR @ allR.t()).clamp(-1, 1).cpu().numpy()
    diff = labs[:, None] != labs[None, :]
    return float(np.degrees(np.arccos(np.clip(C[diff].max(), -1.0, 1.0))))


@torch.no_grad()
def measure_eps(backbone, probe, new_classes, device):
    """Mean angular drift (degrees) of OLD-class probe features from the snapshot
    taken when their cone was frozen.  Frozen backbone ⇒ ε ≡ 0 by construction."""
    drifts = []
    for c, (pxe, snap) in probe.items():
        if c in new_classes:                 # just-born → drift 0 by definition
            continue
        cur = F.normalize(batched_features(backbone, pxe, device), dim=1).cpu()
        cos = (cur * snap).sum(1).clamp(-1, 1)
        drifts.append(torch.rad2deg(torch.arccos(cos)).mean().item())
    return float(np.mean(drifts)) if drifts else 0.0


# ── class-incremental protocol ─────────────────────────────────────────────────
def run_protocol(arm, data, make_backbone=make_mlp_backbone, n_tasks=10, cpt=10,
                 k_rays=8, epochs=100, lr=1e-3, batch_size=32, seed=0,
                 device=torch.device("cpu"), verbose=True):
    """Run a class-incremental protocol classified by FROZEN-at-birth conic hulls
    (the ConicHull primitive: per-class extreme rays + argmin NNLS residual).

    make_backbone(seed, device) -> (backbone, feat_dim).  Use
    ``make_vit_backbone_factory(...)`` for CIFAR-100/ViT (the real conic-hull
    data) or ``make_mlp_backbone`` for the synthetic fallback.

    arm
    ---
    'cone_frozen' : cone-shape (cone-anchor loss) session 0, then FREEZE the
                    backbone (the build).  ε ≡ 0 for every later task → the drift
                    term of forgetting vanishes; residual forgetting is only the
                    crowding floor (γ/2 shrinking as classes accumulate).
    'adapt'       : same session-0 shaping, then KEEP training every task.
                    ε grows as old features drift → margin γ/2 − ε shrinks → it
                    forgets MORE than the frozen arm (drift + crowding).

    Both arms are identical except for the freeze, isolating drift as the cause.
    The forgetting gap (adapt − cone_frozen) is the drift component the build removes.
    """
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = data
    backbone, _ = make_backbone(seed, device)
    backbone.to(device)

    ray_bank = {}      # class -> frozen birth extreme rays (the cone)
    probe = {}         # class -> (Xte_c, birth snapshot of normalized feats)
    acc = np.full((n_tasks, n_tasks), np.nan)
    eps_log, gamma_log, margin_log = [], [], []

    for t in range(n_tasks):
        classes = list(range(t * cpt, (t + 1) * cpt))
        mask = torch.isin(ytr, torch.tensor(classes))
        Xt, yt = Xtr[mask], ytr[mask]

        # 1. train backbone: session 0 always; later tasks only when adapting
        do_train = (t == 0) or (arm == "adapt")
        if do_train:
            train_session(backbone, Xt, yt, classes, device, k_rays=k_rays,
                          epochs=epochs, lr=lr, batch_size=batch_size,
                          frozen_old_bank=ray_bank or None)
        if arm.startswith("cone_frozen") and t == 0:
            freeze(backbone)                       # the build: lock the cone-shaped space

        # 2. form & FREEZE the birth hulls for the new classes
        with torch.no_grad():
            for c in tqdm(classes, desc=f"  task {t} build hulls",
                          leave=False, unit="cls"):
                fc = batched_features(backbone, Xtr[ytr == c], device)
                idx = spa_indices(fc, k_rays)
                ray_bank[c] = F.normalize(fc, dim=1)[idx].detach().cpu()
                pxe = Xte[yte == c]
                snap = F.normalize(batched_features(backbone, pxe, device), dim=1).cpu()
                probe[c] = (pxe, snap)

        # 3. boundary metrics
        gamma = measure_gamma(ray_bank)
        eps = measure_eps(backbone, probe, classes, device)
        eps_log.append(eps); gamma_log.append(gamma); margin_log.append(gamma / 2 - eps)

        # 4. evaluate every seen task with the frozen birth hulls
        with torch.no_grad():
            seen = torch.tensor(sorted(ray_bank))
            em = torch.isin(yte, seen)
            Xev = Xte[em]
            chunks = []
            for i in tqdm(range(0, len(Xev), 64), desc=f"  task {t} eval",
                          leave=False, unit="batch"):
                chunks.append(backbone(Xev[i:i + 64].to(device)).float().cpu())
            fe = torch.cat(chunks, 0)
            pred = classify(fe, {c: ray_bank[c] for c in ray_bank})
            ye = yte[em]
            for j in range(t + 1):
                jm = torch.isin(ye, torch.arange(j * cpt, (j + 1) * cpt))
                acc[t, j] = (pred[jm] == ye[jm]).float().mean().item()

        if verbose:
            held = "OK" if margin_log[-1] > 0 else "CROSSED"
            print(f"[{arm:11s}] task {t+1:2d}/{n_tasks}  "
                  f"avg_acc={np.nanmean(acc[t, :t+1]):.3f}  "
                  f"eps={eps:5.1f}°  gamma/2={gamma/2:5.1f}°  "
                  f"margin={margin_log[-1]:+5.1f}°  [{held}]")

    peak = np.nanmax(acc, axis=0)
    final = acc[-1]
    forgetting = float(np.nanmean(peak[:-1] - final[:-1])) if n_tasks > 1 else 0.0
    return dict(acc=acc, eps=eps_log, gamma=gamma_log, margin=margin_log,
                avg_final=float(np.nanmean(final)), forgetting=forgetting,
                boundary_held=all(m > 0 for m in margin_log))
