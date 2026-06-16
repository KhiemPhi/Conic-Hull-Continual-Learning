#!/usr/bin/env python3
"""
driver.py — agent smoke-driver for the Conic-Hull-Continual-Learning pipeline.

This is NOT the project's test suite. It exercises the *real* static pipeline
(the same functions demo.py calls) end-to-end on a small subset of CIFAR-100 so
an agent can confirm a change works in minutes instead of training a ViT for an
hour:

    load_backbone  ->  build feature dict (subset)  ->  build conic hulls
                   ->  analyze hull separation (saves a heatmap PNG)
                   ->  find most-confused hulls  ->  evaluate conic classifier

It runs fully OFFLINE against the timm weights already in the HF cache
(vit_base_patch16_224.orig_in21k) and the CIFAR-100 tarball already in ./data,
so it never touches the network.

The separation heatmap that analyze_hull_separation would normally `plt.show()`
is captured to <output-dir>/separation_heatmap.png — that PNG is the artifact
to eyeball ("the screenshot of the app").

Usage:
    python .claude/skills/run-conic-hull-continual-learning/driver.py
    python .../driver.py --classes 8 --n-rays 30 --output-dir /tmp/chcl_run
"""
import argparse
import os
import sys

# --- Force offline + headless BEFORE importing torch/timm/matplotlib ----------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Make the repo root importable no matter where the driver is launched from
# (driver lives at <repo>/.claude/skills/run-conic-hull-continual-learning/).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# project modules (run from the repo root so these import cleanly)
from backbone import load_backbone
from features import get_default_transform
from conic_hull import build_class_conic_hulls
from analysis import (
    analyze_hull_separation,
    find_most_confused_hulls,
    evaluate_conic_classifier,
)


def build_subset_feature_dict(model, root, n_classes, batch_size, device, train):
    """Extract per-class feature matrices for the first `n_classes` CIFAR-100
    classes only. Mirrors features.build_feature_dict but subsets the dataset
    so the smoke run stays fast."""
    tfm = get_default_transform()
    ds = datasets.CIFAR100(root, train=train, download=False, transform=tfm)
    keep_idx = [i for i, t in enumerate(ds.targets) if t < n_classes]
    sub = Subset(ds, keep_idx)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=2)

    feats, labels = [], []
    model.eval()
    split = "train" if train else "test"
    print(f"[driver] extracting {split} features: {len(sub)} imgs, "
          f"{n_classes} classes")
    with torch.no_grad():
        for imgs, lbls in loader:
            feats.append(model(imgs.to(device)).cpu().numpy())
            labels.extend(lbls.tolist())
    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    out = {}
    for idx in range(n_classes):
        name = ds.classes[idx]
        out[name] = feats[labels == idx]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="vit_base_patch16_224.orig_in21k",
                    help="timm backbone (must be in HF cache for offline run)")
    ap.add_argument("--classes", type=int, default=6,
                    help="number of CIFAR-100 classes to use (keep small)")
    ap.add_argument("--n-rays", type=int, default=20,
                    help="extreme rays per class hull")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="containment threshold for the separation matrix "
                         "(0.5 shows clean diagonal structure on a subset; "
                         "demo.py uses 0.97 which is near-empty on few classes)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output-dir", default="./driver_out")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    heatmap_path = os.path.join(args.output_dir, "separation_heatmap.png")

    # Capture the heatmap that analyze_hull_separation emits via plt.show().
    def _save_instead_of_show(*a, **k):
        plt.savefig(heatmap_path, dpi=120, bbox_inches="tight")
        print(f"[driver] saved separation heatmap -> {heatmap_path}")
    plt.show = _save_instead_of_show

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"  backbone : {args.backbone}")
    print(f"  classes  : {args.classes}   n_rays: {args.n_rays}")
    print(f"  device   : {device}")
    print("=" * 60)

    model = load_backbone(args.backbone, pretrained=True,
                          num_classes=0, device=device)

    train_fd = build_subset_feature_dict(
        model, args.data_root, args.classes, args.batch_size, device, train=True)
    test_fd = build_subset_feature_dict(
        model, args.data_root, args.classes, args.batch_size, device, train=False)

    hulls = build_class_conic_hulls(train_fd, n_rays=args.n_rays, use_pca=False)

    sep_df = analyze_hull_separation(hulls, test_fd, threshold=args.threshold)
    find_most_confused_hulls(sep_df, top_n=5)
    metrics = evaluate_conic_classifier(hulls, test_fd)

    # --- sanity assertions: the pipeline actually produced sane output --------
    assert sep_df.shape == (args.classes, args.classes), "bad separation matrix"
    assert os.path.exists(heatmap_path), "heatmap was not written"
    diag = np.diag(sep_df.values)
    print("\n" + "=" * 60)
    print(f"[driver] SMOKE PASS")
    print(f"  separation matrix : {sep_df.shape}")
    print(f"  mean self-containment (diag): {diag.mean():.3f}")
    print(f"  classifier accuracy: {metrics['accuracy']:.4f}  "
          f"f1: {metrics['f1']:.4f}")
    print(f"  heatmap artifact  : {heatmap_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
