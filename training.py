"""
training.py
-----------
Fine-tune a timm backbone to convergence with:
  - linear-probe warm-up then full fine-tuning
  - cosine-annealing LR with linear warm-up
  - early stopping + best-checkpoint saving
  - mixed-precision (AMP) support
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import timm

from features import get_default_transform
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



# ── dataset helpers ────────────────────────────────────────────────────────────

def _get_dataset(dataset_name: str, dataset_root: str, train: bool, transform):
    """Return a torchvision dataset split."""
    name = dataset_name.upper()
    if name == "CIFAR10":
        return datasets.CIFAR10(dataset_root, train=train,
                                download=True, transform=transform)
    if name == "CIFAR100":
        return datasets.CIFAR100(dataset_root, train=train,
                                 download=True, transform=transform)
    if name == "STL10":
        split = "train" if train else "test"
        return datasets.STL10(dataset_root, split=split,
                              download=True, transform=transform)
    raise ValueError(f"Unsupported dataset '{dataset_name}'. "
                     f"Choose from CIFAR10, CIFAR100, STL10.")


def _get_augmentation_transform(img_size: int = 224) -> transforms.Compose:
    """Standard ImageNet-style augmentation for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _build_classifier_head(feature_dim: int, num_classes: int) -> nn.Linear:
    """Return an initialised linear classification head."""
    head = nn.Linear(feature_dim, num_classes)
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)
    return head


# ── config ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    All training hyper-parameters in one place.

    Fields
    ------
    max_epochs       : hard ceiling on training epochs
    lr               : initial learning rate (AdamW)
    weight_decay     : AdamW weight-decay
    warmup_epochs    : linear LR warm-up before cosine decay
    min_lr           : cosine-decay floor
    patience         : early-stopping patience (epochs without val-loss improvement)
    min_delta        : minimum improvement to reset patience counter
    label_smoothing  : cross-entropy label smoothing
    grad_clip        : max gradient norm (None → disabled)
    checkpoint_path  : where to save the best-val-loss weights
    freeze_backbone  : if True, only the newly added head is trained initially
    unfreeze_after   : unfreeze backbone after this many epochs (None → never)
    mixed_precision  : use torch.cuda.amp for faster GPU training
    """
    max_epochs:      int            = 50
    lr:              float          = 3e-4
    weight_decay:    float          = 1e-4
    warmup_epochs:   int            = 3
    min_lr:          float          = 1e-6
    patience:        int            = 7
    min_delta:       float          = 1e-4
    label_smoothing: float          = 0.1
    grad_clip:       Optional[float] = 1.0
    checkpoint_path: str            = "best_model.pt"
    freeze_backbone: bool           = False
    unfreeze_after:  Optional[int]  = None
    mixed_precision: bool           = True


# ── main training function ─────────────────────────────────────────────────────

def train_to_convergence(
    model_name:   str,
    dataset_name: str                      = "CIFAR10",
    dataset_root: str                      = "./data",
    batch_size:   int                      = 128,
    num_workers:  int                      = 2,
    cfg:          Optional[TrainingConfig] = None,
    device:       Optional[str]            = None,
) -> Tuple[nn.Module, Dict]:
    """
    Fine-tune a timm backbone until convergence, then return the backbone
    with its classification head removed (ready for feature extraction).

    Returns
    -------
    model   : nn.Module — trained backbone in eval mode, head stripped
    history : dict — { 'train_loss', 'val_loss', 'train_acc', 'val_acc' }
    """
    if cfg is None:
        cfg = TrainingConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = _get_dataset(dataset_name, dataset_root, train=True,
                            transform=_get_augmentation_transform())
    val_ds   = _get_dataset(dataset_name, dataset_root, train=False,
                            transform=get_default_transform())
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\n[train_to_convergence]")
    print(f"  model      : {model_name}")
    print(f"  dataset    : {dataset_name}  ({num_classes} classes, "
          f"{len(train_ds)} train / {len(val_ds)} val)")
    print(f"  device     : {device}")
    print(f"  max_epochs : {cfg.max_epochs}  |  patience : {cfg.patience}")
    print(f"  freeze_bb  : {cfg.freeze_backbone}  |  unfreeze_after : {cfg.unfreeze_after}")

    # ── model: backbone + fresh head ──────────────────────────────────────
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    feat_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
    head     = _build_classifier_head(feat_dim, num_classes)

    class _ClassifierModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head     = head
        def forward(self, x):
            return self.head(self.backbone(x))

    model = _ClassifierModel(backbone, head).to(device)

    if cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        print("  [freeze] backbone frozen — training head only")

    # ── optimiser + loss ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    criterion  = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.max_epochs - cfg.warmup_epochs, 1),
        eta_min=cfg.min_lr,
    )
    use_amp    = cfg.mixed_precision and device == "cuda"
    scaler     = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss  = float("inf")
    patience_count = 0
    best_epoch     = 0

    # ── epoch loop ────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, cfg.max_epochs + 1), desc="Epochs",
                     unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:

        # optional unfreeze
        if (cfg.freeze_backbone and cfg.unfreeze_after is not None
                and epoch == cfg.unfreeze_after + 1):
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr / 10,
                weight_decay=cfg.weight_decay,
            )
            tqdm.write(f"  [unfreeze] backbone unfrozen at epoch {epoch} "
                       f"(lr={cfg.lr/10:.2e})")

        # linear warm-up
        if epoch <= cfg.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * epoch / cfg.warmup_epochs

        # ── train ──────────────────────────────────────────────────────
        model.train()
        train_loss_sum = train_correct = train_total = 0

        train_bar = tqdm(train_loader, desc=f"  Train ep{epoch:03d}",
                         leave=False, unit="batch", dynamic_ncols=True)
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs              = imgs.size(0)
            train_loss_sum += loss.item() * bs
            train_correct  += (logits.argmax(1) == labels).sum().item()
            train_total    += bs
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_total
        train_acc  = train_correct  / train_total

        # ── validate ───────────────────────────────────────────────────
        model.eval()
        val_loss_sum = val_correct = val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"  Val   ep{epoch:03d}",
                           leave=False, unit="batch", dynamic_ncols=True)
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                bs           = imgs.size(0)
                val_loss_sum += loss.item() * bs
                val_correct  += (logits.argmax(1) == labels).sum().item()
                val_total    += bs

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total

        if epoch > cfg.warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}", vl_loss=f"{val_loss:.4f}",
            vl_acc=f"{val_acc:.2%}", lr=f"{current_lr:.2e}",
        )
        tqdm.write(
            f"  ep {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  train_acc={train_acc:.2%}  "
            f"val_acc={val_acc:.2%}  lr={current_lr:.2e}"
        )

        # ── checkpoint + early stopping ───────────────────────────────
        if val_loss < best_val_loss - cfg.min_delta:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), cfg.checkpoint_path)
            tqdm.write(f"  ✓ checkpoint saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                tqdm.write(
                    f"\n  [early stop] no improvement for {cfg.patience} epochs. "
                    f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}."
                )
                break

    # ── reload best weights, strip head ───────────────────────────────────
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    print(f"\n[train_to_convergence] Best epoch {best_epoch}  "
          f"val_loss={best_val_loss:.4f}  "
          f"val_acc={history['val_acc'][best_epoch-1]:.2%}")

    feature_model = model.backbone
    feature_model.eval()
    print("[train_to_convergence] Head removed — ready for feature extraction.")
    return feature_model, history






