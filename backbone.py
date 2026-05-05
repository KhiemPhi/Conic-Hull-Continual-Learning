"""
backbone.py
-----------
Load any timm CNN or Transformer backbone.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

import timm


def load_backbone(
    model_name: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 0,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Load any timm-supported CNN or Transformer backbone.

    Parameters
    ----------
    model_name  : timm model identifier, e.g. 'resnet50', 'vit_small_patch16_224'
    pretrained  : download ImageNet-1k weights when True
    num_classes : 0 → strip head and return raw feature vectors;
                  >0 → keep / replace the classification head
    device      : 'cuda' | 'cpu' | None (auto-detect)

    Returns
    -------
    nn.Module  (eval mode, on `device`)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(model_name, pretrained=pretrained,
                              num_classes=num_classes)
    model.eval()
    model.to(device)

    print(f"[load_backbone] '{model_name}'  pretrained={pretrained}  "
          f"device={device}  feature_dim={_feature_dim(model, device)}")
    return model


def _feature_dim(model: nn.Module, device: str) -> int:
    """Quick forward pass to determine the output feature dimension."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    return out.shape[-1]


def list_popular_backbones() -> Dict[str, List[str]]:
    """Return a curated dict of popular timm model names by family."""
    return {
        "CNN": [
            "resnet18", "resnet50", "resnet101",
            "efficientnet_b0", "efficientnet_b4",
            "convnext_tiny", "convnext_base",
            "mobilenetv3_large_100",
        ],
        "Transformer": [
            "vit_tiny_patch16_224", "vit_small_patch16_224",
            "vit_base_patch16_224", "vit_large_patch16_224",
            "swin_tiny_patch4_window7_224", "swin_base_patch4_window7_224",
            "deit_small_patch16_224", "deit_base_patch16_224",
            "beit_base_patch16_224",
        ],
    }
