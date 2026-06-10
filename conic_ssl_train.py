"""
End-to-end Conic SSL pipeline.

  Phase 0 — instantiate encoders, cone bank, anchor store
  Phase 1 — self-supervised pretraining (cones learned, no labels)
  Phase 2 — continual task arrival (distillation + anchors + optional labels)

Run:
  python conic_ssl_train.py
"""

from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from backbone import load_backbone
from conic_ssl import (
    AnchorStore,
    CapacityLog,
    CapacityRecord,
    ConeBank,
    ConeStatus,
    EMAEncoder,
    OnlineEncoder,
    SSLLossWeights,
    SSLStepMetrics,
    classify_nearest_vertex,
    compute_ssl_losses,
    default_gamma_star,
    effective_dimensionality,
    evaluate_cone_accuracy,
    inter_cone_margin_gamma,
    packing_number,
    soft_cone_assignments,
    spa_initialize_cone,
    cone_nearest_scores,
)

# ─── Augmentations ───────────────────────────────────────────────────────────

def ssl_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def ssl_eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def _augment_single_view(img, t_pil: transforms.Compose):
    """PIL → full SSL aug; Tensor → lightweight two-view-safe aug."""
    if isinstance(img, torch.Tensor):
        x = img.clone()
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])
        if torch.rand(1).item() > 0.5:
            x = x + torch.randn_like(x) * 0.02
        return x.clamp(-1.0, 1.0)
    return t_pil(img)


def two_view_collate(batch):
    """Return (view_a, view_b, labels) with independent augmentations."""
    t = ssl_train_transform()
    imgs_a, imgs_b, labels = [], [], []
    for img, lbl in batch:
        imgs_a.append(_augment_single_view(img, t))
        imgs_b.append(_augment_single_view(img, t))
        labels.append(lbl)
    return (
        torch.stack(imgs_a),
        torch.stack(imgs_b),
        torch.tensor(labels),
    )


def _build_ssl_stages(cfg: ConicSSLConfig):
    """Incremental stage splits with raw PIL train images + eval transform on test."""
    if cfg.dataset_name == "CIFAR100":
        train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
        test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=ssl_eval_transform())
        total_classes = 100
    else:
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=ssl_eval_transform())
        total_classes = 10

    num_stages = total_classes // cfg.classes_per_stage
    train_targets = torch.tensor(train_ds.targets)
    test_targets = torch.tensor(test_ds.targets)
    stages = []
    for i in range(num_stages):
        start = i * cfg.classes_per_stage
        end = start + cfg.classes_per_stage
        idx_train = torch.where((train_targets >= start) & (train_targets < end))[0]
        idx_test = torch.where((test_targets >= start) & (test_targets < end))[0]
        train_loader = DataLoader(
            Subset(train_ds, idx_train.tolist()),
            batch_size=cfg.batch_size, shuffle=True, num_workers=2,
            pin_memory=True, collate_fn=two_view_collate,
        )
        test_loader = DataLoader(
            Subset(test_ds, idx_test.tolist()),
            batch_size=cfg.batch_size, shuffle=False, num_workers=2,
        )
        stages.append({
            "stage_id": i,
            "classes": list(range(start, end)),
            "train_loader": train_loader,
            "test_loader": test_loader,
        })
    return stages


@dataclass
class ConicSSLConfig:
    dataset_name: str = "CIFAR100"
    model_name: str = "vit_tiny_patch16_224"
    proj_dim: int = 256
    classes_per_stage: int = 10
    c_max: int = 120
    k_rays: int = 5
    batch_size: int = 128
    # Phase 1
    pretrain_epochs: int = 20
    pretrain_lr: float = 1e-4
    cone_lr_factor: float = 0.1
    # Phase 2
    task_epochs: int = 10
    task_lr: float = 1e-4
    # SSL hyperparams
    ema_decay: float = 0.996
    tau: float = 0.1
    gamma_scale: float = 0.7
    repulsion_margin_deg: float = 40.0   # training L_rep (softer than ETF-at-C_max)
    grad_clip: float = 1.0
    cone_warmup_steps: int = 400         # no cone activation / dead-cone churn before this
    weights: SSLLossWeights = None  # type: ignore
    # Cone lifecycle
    activate_check_every: int = 500
    activation_threshold: float = 0.25
    dead_cone_fraction: float = 0.005
    novelty_threshold: float = 0.25
    anchor_batch: int = 64
    # Capacity verification tolerances
    delta_gamma: float = 0.05
    delta_d: float = 0.10
    delta_probe: float = 0.05
    # LoRA
    lora_rank: int = 8
    lora_alpha: float = 4.0
    blocks_freeze: int = 8
    visualize: bool = True

    def __post_init__(self):
        if self.weights is None:
            self.weights = SSLLossWeights()


def _freeze_backbone_blocks(backbone: nn.Module, n_blocks: int) -> None:
    if not hasattr(backbone, "patch_embed"):
        return
    for name, p in backbone.patch_embed.named_parameters():
        if "lora_" not in name:
            p.requires_grad_(False)
    if hasattr(backbone, "blocks"):
        for i in range(min(n_blocks, len(backbone.blocks))):
            for name, p in backbone.blocks[i].named_parameters():
                if "lora_" not in name:
                    p.requires_grad_(False)


def setup_components(cfg: ConicSSLConfig, device: torch.device):
    """Phase 0: instantiate all persistent state."""
    backbone = load_backbone(
        cfg.model_name, pretrained=True, num_classes=0, device=str(device),
        lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
    )
    _freeze_backbone_blocks(backbone, cfg.blocks_freeze)

    online = OnlineEncoder(backbone, cfg.proj_dim).to(device)
    ema = EMAEncoder(online, decay=cfg.ema_decay)
    gamma_star = default_gamma_star(cfg.c_max, cfg.gamma_scale)
    cone_bank = ConeBank(cfg.c_max, cfg.k_rays, cfg.proj_dim, gamma_star, device).to(device)
    anchors = AnchorStore()
    capacity_log = CapacityLog()

    print(f"[Setup] proj_dim={cfg.proj_dim}  γ*={math.degrees(gamma_star):.1f}°  "
          f"C_max={cfg.c_max}  K={cfg.k_rays}")
    return online, ema, cone_bank, anchors, capacity_log


def _make_pretrain_loader(cfg: ConicSSLConfig) -> DataLoader:
    """Unlabeled-style loader: all training images, two views per step."""
    ds = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=None,
    ) if cfg.dataset_name == "CIFAR100" else datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None,
    )
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=two_view_collate,
    )


@torch.no_grad()
def _collect_features(
    online: OnlineEncoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Tuple[np.ndarray, List[torch.Tensor], List[int]]:
    feats, imgs_store, lbls = [], [], []
    online.eval()
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        if len(batch) == 3:
            imgs = batch[0]
            labels = batch[2]
        else:
            imgs, labels = batch
        imgs = imgs.to(device)
        z = online(imgs).cpu().numpy()
        feats.append(z)
        for im, lb in zip(imgs.cpu(), labels.tolist()):
            imgs_store.append(im)
            lbls.append(int(lb))
    return np.concatenate(feats, axis=0), imgs_store, lbls


def _activate_cone_from_features(
    cone_bank: ConeBank,
    feats: np.ndarray,
    imgs: List[torch.Tensor],
    indices: np.ndarray,
    cone_id: int,
) -> None:
    rays_np, _ = spa_initialize_cone(feats, cone_bank.k)
    rays = torch.tensor(rays_np, dtype=torch.float32, device=cone_bank.rays.device)
    cone_bank.activate_cone(cone_id, rays)


def _maybe_activate_cones_pretrain(
    online: OnlineEncoder,
    cone_bank: ConeBank,
    loader: DataLoader,
    device: torch.device,
    cfg: ConicSSLConfig,
) -> int:
    """Activate reserved cones when batch assignment scores are low."""
    if cone_bank.n_reserved() == 0:
        return 0
    feats, imgs, _ = _collect_features(online, loader, device, max_batches=5)
    if len(feats) == 0:
        return 0

    usable = cone_bank.usable_indices()
    if not usable:
        c_new = cone_bank.reserved_indices()[0]
        _activate_cone_from_features(cone_bank, feats, imgs, np.arange(len(feats)), c_new)
        return 1

    z = torch.tensor(feats, device=device)
    rays = cone_bank.rays_for_indices(usable, detach_frozen=True)
    scores = cone_nearest_scores(z, rays)
    mean_best = float(scores.max(dim=1).values.mean().item())

    activated = 0
    if mean_best < cfg.activation_threshold:
        reserved = cone_bank.reserved_indices()
        if reserved:
            c_new = reserved[0]
            _activate_cone_from_features(cone_bank, feats, imgs, np.arange(len(feats)), c_new)
            activated = 1
    return activated


def _dead_cone_cleanup(
    cone_bank: ConeBank,
    z: torch.Tensor,
    q: torch.Tensor,
    cone_ids: List[int],
    cfg: ConicSSLConfig,
) -> None:
    """Mark under-used active cones as reserved and re-init."""
    if not cone_ids:
        return
    mass = q.mean(dim=0)
    for j, c in enumerate(cone_ids):
        if int(cone_bank.status[c]) != ConeStatus.ACTIVE:
            continue
        if float(mass[j].item()) < cfg.dead_cone_fraction:
            cone_bank.status[c] = ConeStatus.RESERVED
            cone_bank.reinit_reserved_farthest(c)


def train_one_epoch(
    online: OnlineEncoder,
    ema: EMAEncoder,
    cone_bank: ConeBank,
    anchors: AnchorStore,
    loader: DataLoader,
    optim_enc: torch.optim.Optimizer,
    optim_cone: torch.optim.Optimizer,
    device: torch.device,
    cfg: ConicSSLConfig,
    snapshot_encoder: Optional[OnlineEncoder] = None,
    labels_available: bool = False,
    label_to_cone: Optional[Dict[int, int]] = None,
    global_step: int = 0,
) -> Tuple[SSLStepMetrics, int]:
    online.train()
    agg = SSLStepMetrics()
    n_batches = 0

    for batch in loader:
        if len(batch) == 3:
            x_a, x_b, labels = batch
            labels = labels.to(device)
        else:
            x_a, labels = batch
            x_b = x_a
            labels = labels.to(device)

        x_a, x_b = x_a.to(device), x_b.to(device)
        z_a = online(x_a)
        z_b = ema(x_b)

        z_prev = None
        if snapshot_encoder is not None:
            with torch.no_grad():
                z_prev = snapshot_encoder(x_a)

        anchor_imgs, anchor_z_star = anchors.sample(cfg.anchor_batch, device)
        z_anchor = None
        if anchor_imgs is not None:
            z_anchor = online(anchor_imgs)

        loss, metrics, parts = compute_ssl_losses(
            z_a, z_b, cone_bank, cfg.weights, tau=cfg.tau,
            z_prev=z_prev,
            anchor_z=z_anchor, anchor_z_star=anchor_z_star,
            labels=labels if labels_available else None,
            label_to_cone=label_to_cone,
            repulsion_margin_deg=cfg.repulsion_margin_deg,
        )

        optim_enc.zero_grad()
        optim_cone.zero_grad()
        loss.backward()
        cone_bank.mask_inactive_grads()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in online.parameters() if p.requires_grad], cfg.grad_clip,
            )
            if cone_bank.rays.grad is not None:
                torch.nn.utils.clip_grad_norm_([cone_bank.rays], cfg.grad_clip)
        optim_enc.step()
        optim_cone.step()
        ema.update(online)

        batch_total = float(loss.item())
        for k in ("consist", "rep", "vol", "tight", "temp", "stab", "attr"):
            setattr(agg, f"loss_{k}", getattr(agg, f"loss_{k}") + getattr(metrics, f"loss_{k}"))
        agg.loss_total += batch_total
        n_batches += 1
        global_step += 1

        if (
            global_step >= cfg.cone_warmup_steps
            and global_step % cfg.activate_check_every == 0
            and snapshot_encoder is None
        ):
            _maybe_activate_cones_pretrain(online, cone_bank, loader, device, cfg)
            usable = cone_bank.active_indices()
            if usable:
                rays = cone_bank.rays_for_indices(usable)
                q, cids = soft_cone_assignments(z_a, rays, cfg.tau)
                _dead_cone_cleanup(cone_bank, z_a, q, cids, cfg)

    if n_batches > 0:
        for k in ("consist", "rep", "vol", "tight", "temp", "stab", "attr"):
            setattr(agg, f"loss_{k}", getattr(agg, f"loss_{k}") / n_batches)
        agg.loss_total /= n_batches
    return agg, global_step


def log_capacity(
    task_id: int,
    online: OnlineEncoder,
    cone_bank: ConeBank,
    loader: DataLoader,
    device: torch.device,
    capacity_log: CapacityLog,
) -> CapacityRecord:
    feats, _, _ = _collect_features(online, loader, device, max_batches=30)
    rays = cone_bank.rays_for_indices(cone_bank.usable_indices(), detach_frozen=True)
    rec = CapacityRecord(
        task_id=task_id,
        d_eff=effective_dimensionality(feats),
        packing_number=packing_number(rays, cone_bank.cos_gamma),
        gamma_t=inter_cone_margin_gamma(rays),
        n_active=len(cone_bank.active_indices()),
        n_frozen=len(cone_bank.frozen_indices()),
    )
    capacity_log.append(rec)
    print(f"  [Capacity t={task_id}] d_eff={rec.d_eff:.1f}  pack={rec.packing_number:.2%}  "
          f"Γ_t={math.degrees(rec.gamma_t):.1f}°  active={rec.n_active} frozen={rec.n_frozen}")
    return rec


@torch.no_grad()
def allocate_cones_for_task(
    online: OnlineEncoder,
    cone_bank: ConeBank,
    loader: DataLoader,
    device: torch.device,
    cfg: ConicSSLConfig,
    class_ids: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Step 2.2 — novelty detection + cone allocation.
    Returns label→cone_id map (identity when labels present).
    """
    label_to_cone: Dict[int, int] = {}
    feats_by_class: Dict[int, list] = {}
    imgs_by_class: Dict[int, list] = {}

    online.eval()
    for batch in loader:
        if len(batch) == 3:
            imgs, labels = batch[0], batch[2]
        else:
            imgs, labels = batch
        imgs = imgs.to(device)
        z = online(imgs).cpu().numpy()
        for feat, im, lb in zip(z, imgs.cpu(), labels.tolist()):
            feats_by_class.setdefault(lb, []).append(feat)
            imgs_by_class.setdefault(lb, []).append(im)

    if class_ids is None:
        class_ids = sorted(feats_by_class.keys())

    reserved = cone_bank.reserved_indices()
    ri = 0
    for cls in class_ids:
        if ri >= len(reserved):
            print(f"  [WARNING] No reserved cones left for class {cls}")
            break
        feats = np.stack(feats_by_class[cls])
        c_id = reserved[ri]
        rays_np, _ = spa_initialize_cone(feats, cfg.k_rays)
        cone_bank.activate_cone(c_id, torch.tensor(rays_np, dtype=torch.float32))
        label_to_cone[cls] = c_id
        ri += 1

    if not class_ids and feats_by_class:
        # label-free: allocate until under-assigned mass absorbed (one cone per detected cluster)
        all_feats = np.concatenate([np.stack(v) for v in feats_by_class.values()])
        z = torch.tensor(all_feats, device=device)
        usable = cone_bank.usable_indices()
        if usable:
            rays = cone_bank.rays_for_indices(usable)
            scores = cone_nearest_scores(z, rays).max(dim=1).values
            under = scores < cfg.novelty_threshold
            if under.float().mean() > 0.1 and reserved:
                c_id = reserved[0]
                _activate_cone_from_features(cone_bank, all_feats[under.cpu().numpy()], [], np.arange(under.sum()), c_id)

    return label_to_cone


def consolidate_task_cones(
    online: OnlineEncoder,
    cone_bank: ConeBank,
    anchors: AnchorStore,
    loader: DataLoader,
    device: torch.device,
    cfg: ConicSSLConfig,
    label_to_cone: Dict[int, int],
) -> None:
    """Step 2.4 — SPA consolidation, anchor storage, freeze."""
    feats_by_class: Dict[int, list] = {}
    imgs_by_class: Dict[int, list] = {}

    online.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, labels = batch[0], batch[2]
            else:
                imgs, labels = batch
            imgs = imgs.to(device)
            z = online(imgs)
            for feat, im, lb in zip(z.cpu().numpy(), imgs.cpu(), labels.tolist()):
                feats_by_class.setdefault(lb, []).append(feat)
                imgs_by_class.setdefault(lb, []).append(im)

    for cls, c_id in label_to_cone.items():
        if cls not in feats_by_class:
            continue
        feats = np.stack(feats_by_class[cls])
        rays_np, idxs = spa_initialize_cone(feats, cfg.k_rays)
        with torch.no_grad():
            cone_bank.rays[c_id].copy_(
                torch.tensor(rays_np, dtype=torch.float32, device=cone_bank.rays.device)
            )
        z_final = torch.tensor(feats[idxs], device=device)
        for j, vi in enumerate(idxs):
            anchors.add(imgs_by_class[cls][int(vi)], z_final[j], c_id)
        cone_bank.freeze_cone(c_id)
        print(f"  [Consolidate] class {cls} → cone {c_id}  ({len(idxs)} anchors)")


def verify_capacity(
    record: CapacityRecord,
    baseline: CapacityRecord,
    prev_probe: float,
    probe_acc: float,
    cfg: ConicSSLConfig,
) -> Dict[str, bool]:
    """Step 2.5 — three pass/fail checks."""
    checks = {
        "gamma": record.gamma_t >= (default_gamma_star(cfg.c_max, cfg.gamma_scale) - cfg.delta_gamma),
        "d_eff": record.d_eff >= (1.0 - cfg.delta_d) * baseline.d_eff,
        "probe": probe_acc >= prev_probe - cfg.delta_probe,
    }
    names = {True: "PASS", False: "FAIL"}
    print(f"  [Verify] Γ: {names[checks['gamma']]}  d_eff: {names[checks['d_eff']]}  "
          f"probe: {names[checks['probe']]}  (acc={probe_acc:.2%})")
    return checks


def phase1_pretrain(
    online: OnlineEncoder,
    ema: EMAEncoder,
    cone_bank: ConeBank,
    anchors: AnchorStore,
    capacity_log: CapacityLog,
    cfg: ConicSSLConfig,
    device: torch.device,
) -> CapacityRecord:
    print("\n" + "=" * 60)
    print("Phase 1: Self-supervised pretraining")
    print("=" * 60)

    loader = _make_pretrain_loader(cfg)
    cone_bank.ensure_seed_cone()
    print(f"  [Pretrain] seed cone {cone_bank.active_indices()[0]} activated")

    optim_enc = torch.optim.AdamW(
        [p for p in online.parameters() if p.requires_grad], lr=cfg.pretrain_lr,
    )
    optim_cone = torch.optim.AdamW(
        cone_bank.trainable_parameters(), lr=cfg.pretrain_lr * cfg.cone_lr_factor,
    )

    global_step = 0
    for epoch in range(cfg.pretrain_epochs):
        metrics, global_step = train_one_epoch(
            online, ema, cone_bank, anchors, loader,
            optim_enc, optim_cone, device, cfg, global_step=global_step,
        )
        total = metrics.loss_total
        print(f"  ep {epoch+1}/{cfg.pretrain_epochs}  total={total:.4f}  "
              f"consist={metrics.loss_consist:.4f}  rep={metrics.loss_rep:.4f}  "
              f"vol={metrics.loss_vol:.4f}  tight={metrics.loss_tight:.4f}  "
              f"active={len(cone_bank.active_indices())}")

    baseline = log_capacity(0, online, cone_bank, loader, device, capacity_log)

    # Freeze all active pretrain cones
    for c in cone_bank.active_indices():
        cone_bank.freeze_cone(c)

    if cfg.visualize:
        try:
            from incremental import visualize_extreme_rays_3d
            hulls = {
                str(c): types.SimpleNamespace(
                    extreme_rays_=cone_bank.rays[c].detach().cpu().numpy()
                )
                for c in cone_bank.frozen_indices()
            }
            visualize_extreme_rays_3d(hulls, stage_idx=0, stage_class_map={})
        except Exception as exc:
            print(f"  [Viz] skipped: {exc}")

    return baseline


def phase2_continual(
    online: OnlineEncoder,
    ema: EMAEncoder,
    cone_bank: ConeBank,
    anchors: AnchorStore,
    capacity_log: CapacityLog,
    cfg: ConicSSLConfig,
    device: torch.device,
    baseline: CapacityRecord,
) -> None:
    print("\n" + "=" * 60)
    print("Phase 2: Continual task arrival")
    print("=" * 60)

    stages = _build_ssl_stages(cfg)

    global_label_to_cone: Dict[int, int] = {}
    prev_probe = 0.0
    global_step = 0

    for task_idx, stage in enumerate(stages):
        print(f"\n--- Task {task_idx}  classes {stage['classes']} ---")

        # Step 2.1 — snapshot encoder for cross-temporal distillation
        snapshot = copy.deepcopy(online)
        for p in snapshot.parameters():
            p.requires_grad_(False)
        snapshot.eval()

        # Step 2.2 — allocate cones for new classes
        label_to_cone = allocate_cones_for_task(
            online, cone_bank, stage["train_loader"], device, cfg,
            class_ids=stage["classes"],
        )
        global_label_to_cone.update(label_to_cone)

        optim_enc = torch.optim.AdamW(
            [p for p in online.parameters() if p.requires_grad], lr=cfg.task_lr,
        )
        optim_cone = torch.optim.AdamW(
            cone_bank.trainable_parameters(), lr=cfg.task_lr * cfg.cone_lr_factor,
        )

        for epoch in range(cfg.task_epochs):
            metrics, global_step = train_one_epoch(
                online, ema, cone_bank, anchors, stage["train_loader"],
                optim_enc, optim_cone, device, cfg,
                snapshot_encoder=snapshot,
                labels_available=True,
                label_to_cone=global_label_to_cone,
                global_step=global_step,
            )
            total = metrics.loss_total
            print(f"  ep {epoch+1}/{cfg.task_epochs}  total={total:.4f}  "
                  f"consist={metrics.loss_consist:.4f}  temp={metrics.loss_temp:.4f}  "
                  f"stab={metrics.loss_stab:.4f}  attr={metrics.loss_attr:.4f}  "
                  f"vol={metrics.loss_vol:.4f}")

        # Step 2.4 — consolidate new cones
        consolidate_task_cones(
            online, cone_bank, anchors, stage["train_loader"],
            device, cfg, label_to_cone,
        )

        # Evaluate + capacity
        combined_loaders = []
        for si in range(task_idx + 1):
            combined_loaders.append(stages[si]["test_loader"])
        probe_acc = 0.0
        n_eval = 0
        for ev_loader in combined_loaders:
            probe_acc += evaluate_cone_accuracy(
                online, cone_bank, ev_loader, device, global_label_to_cone,
            )
            n_eval += 1
        probe_acc /= max(n_eval, 1)

        record = log_capacity(task_idx + 1, online, cone_bank, stage["train_loader"], device, capacity_log)
        verify_capacity(record, baseline, prev_probe, probe_acc, cfg)
        prev_probe = probe_acc

        if cfg.visualize:
            try:
                from incremental import visualize_extreme_rays_3d
                hulls = {
                    str(c): __import__("types").SimpleNamespace(
                        extreme_rays_=cone_bank.rays[c].detach().cpu().numpy()
                    )
                    for c in cone_bank.frozen_indices()
                }
                stage_map = {cls: task_idx for cls in stage["classes"]}
                visualize_extreme_rays_3d(hulls, stage_idx=task_idx, stage_class_map=stage_map)
            except Exception as exc:
                print(f"  [Viz] skipped: {exc}")


def run_conic_ssl_pipeline(cfg: Optional[ConicSSLConfig] = None) -> None:
    cfg = cfg or ConicSSLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online, ema, cone_bank, anchors, capacity_log = setup_components(cfg, device)
    baseline = phase1_pretrain(online, ema, cone_bank, anchors, capacity_log, cfg, device)
    phase2_continual(online, ema, cone_bank, anchors, capacity_log, cfg, device, baseline)

    print("\n=== Pipeline complete ===")
    for rec in capacity_log.records:
        print(f"  t={rec.task_id}: d_eff={rec.d_eff:.1f}  Γ={math.degrees(rec.gamma_t):.1f}°  "
              f"pack={rec.packing_number:.1%}")


if __name__ == "__main__":
    run_conic_ssl_pipeline(ConicSSLConfig(
        dataset_name="CIFAR100",
        classes_per_stage=10,
        pretrain_epochs=5,
        task_epochs=5,
        batch_size=128,
        lora_rank=8,
        visualize=True,
    ))
