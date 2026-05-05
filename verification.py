"""
verification.py
---------------
Post-training weight verification suite.
Checks checkpoint integrity, weight sanity, gradient flow, feature quality,
and accuracy vs a pretrained baseline.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import timm
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# ── report dataclass ──────────────────────────────────────────────────────────

@dataclass
class WeightVerificationReport:
    """
    Full diagnostic report produced by verify_trained_weights().

    Sections
    --------
    checkpoint_checks   : file integrity and key matching
    architecture_checks : NaN/Inf/zero weights, changed-from-pretrained
    gradient_checks     : gradients flow through the network
    feature_checks      : feature shape, finite, non-constant, spread
    accuracy_checks     : top-1/top-5 vs minimum thresholds and baseline
    """
    checkpoint_checks:   Dict[str, bool]   = field(default_factory=dict)
    architecture_checks: Dict[str, bool]   = field(default_factory=dict)
    gradient_checks:     Dict[str, bool]   = field(default_factory=dict)
    feature_checks:      Dict[str, bool]   = field(default_factory=dict)
    accuracy_checks:     Dict[str, bool]   = field(default_factory=dict)
    details:             Dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(
            list(self.checkpoint_checks.values())
            + list(self.architecture_checks.values())
            + list(self.gradient_checks.values())
            + list(self.feature_checks.values())
            + list(self.accuracy_checks.values())
        )

    def print_report(self) -> None:
        width    = 68
        sections = [
            ("Checkpoint",   self.checkpoint_checks),
            ("Architecture", self.architecture_checks),
            ("Gradients",    self.gradient_checks),
            ("Features",     self.feature_checks),
            ("Accuracy",     self.accuracy_checks),
        ]
        print("\n" + "═" * width)
        print("  Weight Verification Report")
        print("═" * width)

        for section_name, checks in sections:
            if not checks:
                continue
            print(f"\n  ── {section_name} ──")
            for name, ok in checks.items():
                icon   = "✓" if ok else "✗"
                detail = self.details.get(name, "")
                suffix = f"  ({detail})" if detail else ""
                print(f"    [{icon}]  {name}{suffix}")

        print("\n" + "─" * width)
        print(f"  {'ALL CHECKS PASSED ✓' if self.passed else 'SOME CHECKS FAILED ✗'}")
        print("═" * width + "\n")


# ── main verification function ────────────────────────────────────────────────

def verify_trained_weights(
    model_name:          str,
    checkpoint_path:     str,
    dataset_name:        str           = "CIFAR10",
    dataset_root:        str           = "./data",
    batch_size:          int           = 128,
    num_workers:         int           = 2,
    num_classes:         Optional[int] = None,
    min_top1_acc:        float         = 0.70,
    min_top5_acc:        float         = 0.90,
    pretrained_baseline: bool          = True,
    device:              Optional[str] = None,
) -> WeightVerificationReport:
    """
    Run a comprehensive suite of checks to confirm a checkpoint contains
    useful, well-trained weights.

    Checks
    ------
    Checkpoint   — file exists, loads, keys match model architecture
    Architecture — no NaN/Inf/zero weights, weights changed from pretrained init
    Gradients    — single forward+backward produces finite non-zero gradients
    Features     — correct shape, finite, non-constant, sufficient spread
    Accuracy     — top-1/top-5 ≥ thresholds, beats pretrained baseline

    Returns
    -------
    WeightVerificationReport  (call .print_report() to display)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    report = WeightVerificationReport()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    _NC = {"CIFAR10": 10, "CIFAR100": 100, "STL10": 10}
    name = dataset_name.upper()
    if num_classes is None:
        if name not in _NC:
            raise ValueError(f"Unknown dataset '{dataset_name}'.")
        num_classes = _NC[name]

    def _make_classifier(pretrained: bool) -> nn.Module:
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model(
                    model_name, pretrained=pretrained, num_classes=0)
                feat_dim   = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
                self.head  = nn.Linear(feat_dim, num_classes)
            def forward(self, x):
                return self.head(self.backbone(x))
        return _M().to(device)

    # ── 1. Checkpoint integrity ────────────────────────────────────────────
    print("[verify] 1/5 — Checkpoint integrity …")

    exists = os.path.isfile(checkpoint_path)
    report.checkpoint_checks["file_exists"] = exists
    report.details["file_exists"] = checkpoint_path

    if not exists:
        report.checkpoint_checks.update({
            "file_loads": False, "keys_match_model": False,
            "no_missing_keys": False, "no_unexpected_keys": False,
        })
        report.details["file_loads"] = "skipped — file not found"
        report.print_report()
        return report

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        report.checkpoint_checks["file_loads"] = True
    except Exception as exc:
        report.checkpoint_checks["file_loads"] = False
        report.details["file_loads"] = str(exc)
        report.print_report()
        return report

    model  = _make_classifier(pretrained=False)
    result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys

    report.checkpoint_checks["keys_match_model"]   = not missing and not unexpected
    report.checkpoint_checks["no_missing_keys"]    = len(missing) == 0
    report.checkpoint_checks["no_unexpected_keys"] = len(unexpected) == 0
    if missing:
        report.details["no_missing_keys"]    = f"{len(missing)} missing: {missing[:3]}"
    if unexpected:
        report.details["no_unexpected_keys"] = f"{len(unexpected)} unexpected: {unexpected[:3]}"

    model.eval()

    # ── 2. Architecture / weight sanity ────────────────────────────────────
    print("[verify] 2/5 — Architecture & weight sanity …")

    all_params = torch.cat([p.detach().float().flatten() for p in model.parameters()])
    report.architecture_checks["param_count_nonzero"] = all_params.numel() > 0
    report.details["param_count_nonzero"]              = f"{all_params.numel():,} params"
    report.architecture_checks["no_nan_in_weights"]   = not torch.isnan(all_params).any().item()
    report.architecture_checks["no_inf_in_weights"]   = not torch.isinf(all_params).any().item()
    report.architecture_checks["weights_not_all_zero"] = not (all_params == 0).all().item()

    pretrained_model  = _make_classifier(pretrained=True)
    pretrained_params = torch.cat([p.detach().float().flatten()
                                   for p in pretrained_model.parameters()])
    delta = (all_params - pretrained_params).abs().max().item()
    report.architecture_checks["weights_changed_from_pretrained"] = delta > 1e-6
    report.details["weights_changed_from_pretrained"] = f"max Δ={delta:.6f}"
    del pretrained_model, pretrained_params

    # ── 3. Gradient flow ───────────────────────────────────────────────────
    print("[verify] 3/5 — Gradient flow …")

    model.train()
    dummy        = torch.randn(4, 3, 224, 224, device=device)
    dummy_labels = torch.randint(0, num_classes, (4,), device=device)
    nn.CrossEntropyLoss()(model(dummy), dummy_labels).backward()

    grad = model.head.weight.grad
    report.gradient_checks["gradients_flow"] = (
        grad is not None
        and torch.isfinite(grad).all().item()
        and grad.abs().max().item() > 1e-12
    )
    if grad is not None:
        report.details["gradients_flow"] = f"head grad max={grad.abs().max().item():.2e}"
    model.eval()

    # ── 4. Feature-vector quality ──────────────────────────────────────────
    print("[verify] 4/5 — Feature quality …")

    if name == "CIFAR10":
        probe_ds = datasets.CIFAR10(dataset_root, train=False,
                                    download=True, transform=val_transform)
    elif name == "CIFAR100":
        probe_ds = datasets.CIFAR100(dataset_root, train=False,
                                     download=True, transform=val_transform)
    else:
        probe_ds = datasets.STL10(dataset_root, split="test",
                                  download=True, transform=val_transform)

    probe_imgs, _ = next(iter(DataLoader(probe_ds, batch_size=64,
                                         shuffle=True, num_workers=num_workers)))
    probe_imgs = probe_imgs.to(device)

    with torch.no_grad():
        feats = model.backbone(probe_imgs)    # (B, D)

    B, D = feats.shape
    std_per_dim = feats.std(dim=0)

    report.feature_checks["output_shape_correct"] = feats.ndim == 2
    report.details["output_shape_correct"]         = f"shape=({B}, {D})"
    report.feature_checks["features_are_finite"]  = torch.isfinite(feats).all().item()
    report.feature_checks["features_not_constant"] = (std_per_dim > 1e-6).any().item()
    mean_std = std_per_dim.mean().item()
    report.feature_checks["feature_spread"]       = mean_std > 0.01
    report.details["feature_spread"]              = f"mean per-dim std={mean_std:.4f}"

    # ── 5. Accuracy ────────────────────────────────────────────────────────
    print("[verify] 5/5 — Test-set accuracy …")

    def _eval_accuracy(eval_model: nn.Module) -> Tuple[float, float]:
        eval_model.eval()
        c1 = c5 = total = 0
        loader = DataLoader(probe_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="  Evaluating",
                                     leave=False, dynamic_ncols=True):
                imgs, labels = imgs.to(device), labels.to(device)
                top5  = eval_model(imgs).topk(min(5, num_classes), dim=1).indices
                c1   += (top5[:, 0] == labels).sum().item()
                c5   += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)
        return c1 / total, c5 / total

    top1, top5 = _eval_accuracy(model)
    report.accuracy_checks["top1_above_threshold"] = top1 >= min_top1_acc
    report.accuracy_checks["top5_above_threshold"] = top5 >= min_top5_acc
    report.details["top1_above_threshold"] = f"{top1:.2%}  (min={min_top1_acc:.0%})"
    report.details["top5_above_threshold"] = f"{top5:.2%}  (min={min_top5_acc:.0%})"

    if pretrained_baseline:
        baseline_top1, _ = _eval_accuracy(_make_classifier(pretrained=True))
        report.accuracy_checks["beats_pretrained_baseline"] = top1 > baseline_top1
        report.details["beats_pretrained_baseline"] = (
            f"fine-tuned={top1:.2%}  baseline={baseline_top1:.2%}"
        )

    report.print_report()
    return report
