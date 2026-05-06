"""
backbone.py
-----------
Load any timm CNN or Transformer backbone, with optional LoRA injection.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


# ─────────────────────────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adaptation.

    The original weight matrix W is frozen; only the rank-decomposition
    matrices A (rank × in) and B (out × rank) are trained.

    Forward:   y = x W^T + x A^T B^T * (alpha / rank)

    Initialisation follows the original LoRA paper: A is Kaiming-uniform,
    B is zeros, so the adapter contributes zero at the start of training.
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features

        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        self.lora_A   = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B   = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling  = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def extra_repr(self) -> str:
        rank = self.lora_A.shape[0]
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={rank}, scaling={self.scaling:.4f}")


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, int]:
    """
    Replace targeted nn.Linear layers in a timm ViT with LoRALinear wrappers.

    Parameters
    ----------
    model          : timm ViT backbone (in-place mutation)
    rank           : LoRA rank r (controls adapter capacity)
    alpha          : LoRA scaling factor α (effective scale = α/r)
    target_modules : list of name suffixes to match, e.g.
                     ["attn.qkv", "attn.proj"]
                     Defaults to QKV + output projection in every block.

    Returns
    -------
    (model, n_replaced)  — mutated model and count of replaced layers
    """
    if target_modules is None:
        target_modules = ["attn.qkv", "attn.proj"]

    # Collect replacements first to avoid mutating while iterating
    replacements: List[Tuple[nn.Module, str, nn.Linear]] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for target in target_modules:
            if full_name == target or full_name.endswith("." + target):
                parts  = full_name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                replacements.append((parent, parts[-1], module))
                break

    for parent, attr, linear in replacements:
        setattr(parent, attr, LoRALinear(linear, rank=rank, alpha=alpha))

    n = len(replacements)
    print(f"[inject_lora] {n} layer(s) replaced with LoRA "
          f"(rank={rank}, alpha={alpha}, targets={target_modules})")
    return model, n


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return only the LoRA A/B parameters — pass these to the optimiser."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params += [module.lora_A, module.lora_B]
    return params


def freeze_non_lora(model: nn.Module) -> int:
    """
    Freeze every parameter that is NOT a LoRA adapter (lora_A / lora_B).

    Useful when you want the backbone to serve purely as a frozen feature
    extractor that adapts only through its LoRA parameters.

    Returns the number of parameters frozen.
    """
    lora_ids = {id(p) for m in model.modules()
                if isinstance(m, LoRALinear)
                for p in [m.lora_A, m.lora_B]}
    n_frozen = 0
    for p in model.parameters():
        if id(p) not in lora_ids:
            p.requires_grad_(False)
            n_frozen += 1
    return n_frozen


# ─────────────────────────────────────────────────────────────────────────────
# Backbone loading
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(
    model_name: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 0,
    device: Optional[str] = None,
    lora_rank: int = 0,
    lora_alpha: float = 1.0,
    lora_target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Load any timm-supported CNN or Transformer backbone.

    Parameters
    ----------
    model_name          : timm model identifier, e.g. 'resnet50', 'vit_small_patch16_224'
    pretrained          : download ImageNet-1k weights when True
    num_classes         : 0 → strip head and return raw feature vectors;
                          >0 → keep / replace the classification head
    device              : 'cuda' | 'cpu' | None (auto-detect)
    lora_rank           : if > 0, inject LoRA adapters with this rank into the
                          specified target layers (ViT only; ignored for CNNs)
    lora_alpha          : LoRA scaling factor α (effective scale = α/rank)
    lora_target_modules : linear layer name suffixes to adapt, e.g.
                          ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
                          Defaults to ["attn.qkv", "attn.proj"].

    Returns
    -------
    nn.Module  (eval mode, on `device`)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(model_name, pretrained=pretrained,
                              num_classes=num_classes)

    if lora_rank > 0:
        model, n_lora = inject_lora(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_target_modules,
        )
        lora_param_count = sum(p.numel() for p in get_lora_params(model))
        print(f"[load_backbone] LoRA active — {n_lora} layers, "
              f"{lora_param_count:,} trainable adapter params")

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
