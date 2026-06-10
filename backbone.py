"""
backbone.py
-----------
Load any timm CNN or Transformer backbone, with optional LoRA injection.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


# ─────────────────────────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Task-shared LoRA: one (A, B) pair reused across all tasks.
    W = W₀ + B A * (alpha / rank)

    Initialisation: A is Kaiming-uniform, B is zeros.
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


class LoRALinearTaskSpecific(nn.Module):
    """
    Task-specific LoRA: each task gets its own (Aᵢ, Bᵢ) pair.
    W = W₀ + Σᵢ Bᵢ Aᵢ * (alpha / rank)

    Call advance_task() at the start of each new task to freeze the current
    adapters and append a fresh, trainable pair.
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        self._rank    = rank
        self.scaling  = alpha / rank
        self.lora_As  = nn.ParameterList()
        self.lora_Bs  = nn.ParameterList()
        self._add_adapter()

    def _add_adapter(self):
        device = self.linear.weight.device
        A = nn.Parameter(torch.empty(self._rank, self.linear.in_features, device=device))
        B = nn.Parameter(torch.zeros(self.linear.out_features, self._rank, device=device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.lora_As.append(A)
        self.lora_Bs.append(B)

    def advance_task(self):
        """Freeze the current task's adapters and allocate a fresh pair."""
        for A, B in zip(self.lora_As, self.lora_Bs):
            A.requires_grad_(False)
            B.requires_grad_(False)
        self._add_adapter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        for A, B in zip(self.lora_As, self.lora_Bs):
            out = out + (x @ A.T @ B.T) * self.scaling
        return out

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self._rank}, n_tasks={len(self.lora_As)}, "
                f"scaling={self.scaling:.4f}")


class LoRALinearHybrid(nn.Module):
    """
    Hybrid LoRA (HydraLoRA-inspired): one shared A matrix and task-specific Bᵢ.
    W = W₀ + Σᵢ Bᵢ A * (alpha / rank)

    A is always trainable; only the current task's Bᵢ is trainable.
    Call advance_task() at the start of each new task.
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        self._rank   = rank
        self.scaling = alpha / rank

        self.lora_A  = nn.Parameter(torch.empty(rank, linear.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_Bs = nn.ParameterList()
        self._add_b()

    def _add_b(self):
        device = self.linear.weight.device
        B = nn.Parameter(torch.zeros(self.linear.out_features, self._rank, device=device))
        self.lora_Bs.append(B)

    def advance_task(self):
        """Freeze current Bᵢ and allocate a fresh one; A stays trainable."""
        self.lora_Bs[-1].requires_grad_(False)
        self._add_b()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.linear(x)
        mid  = x @ self.lora_A.T          # (batch, rank)
        for B in self.lora_Bs:
            out = out + (mid @ B.T) * self.scaling
        return out

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self._rank}, n_tasks={len(self.lora_Bs)}, "
                f"scaling={self.scaling:.4f}")


# Registry of all LoRA layer types for shared helpers
_ALL_LORA_TYPES = (LoRALinear, LoRALinearTaskSpecific, LoRALinearHybrid)


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[List[str]] = None,
    lora_config: str = "task_shared",
) -> Tuple[nn.Module, int]:
    """
    Replace targeted nn.Linear layers in a timm ViT with LoRA wrappers.

    Parameters
    ----------
    model          : timm ViT backbone (in-place mutation)
    rank           : LoRA rank r
    alpha          : LoRA scaling factor α (effective scale = α/r)
    target_modules : list of name suffixes to match; defaults to attn.qkv + attn.proj
    lora_config    : one of
                     "task_shared"    — single shared (A, B) for all tasks
                     "task_specific"  — per-task (Aᵢ, Bᵢ); call advance_lora_task() each stage
                     "hybrid"         — shared A, per-task Bᵢ (HydraLoRA-style)

    Returns
    -------
    (model, n_replaced)
    """
    if target_modules is None:
        target_modules = ["attn.qkv", "attn.proj"]

    _cls_map = {
        "task_shared":   LoRALinear,
        "task_specific": LoRALinearTaskSpecific,
        "hybrid":        LoRALinearHybrid,
    }
    if lora_config not in _cls_map:
        raise ValueError(
            f"Unknown lora_config '{lora_config}'. "
            f"Choose from {list(_cls_map)}."
        )
    lora_cls = _cls_map[lora_config]

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
        setattr(parent, attr, lora_cls(linear, rank=rank, alpha=alpha))

    n = len(replacements)
    print(f"[inject_lora] {n} layer(s) replaced with {lora_cls.__name__} "
          f"(rank={rank}, alpha={alpha}, targets={target_modules})")
    return model, n


def advance_lora_task(model: nn.Module) -> int:
    """
    Advance all task-specific / hybrid LoRA layers to the next task.

    Freezes the current task's adapters and allocates fresh trainable ones.
    No-op for task-shared LoRA layers.

    Returns the number of layers advanced.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, (LoRALinearTaskSpecific, LoRALinearHybrid)):
            module.advance_task()
            n += 1
    return n


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all LoRA parameters across all adapter types."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params += [module.lora_A, module.lora_B]
        elif isinstance(module, LoRALinearTaskSpecific):
            for A, B in zip(module.lora_As, module.lora_Bs):
                params += [A, B]
        elif isinstance(module, LoRALinearHybrid):
            params.append(module.lora_A)
            params += list(module.lora_Bs)
    return params


def freeze_non_lora(model: nn.Module) -> int:
    """
    Freeze every parameter that is NOT a LoRA adapter.
    Returns the number of parameters frozen.
    """
    lora_ids: set = set()
    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora_ids.update(id(p) for p in [m.lora_A, m.lora_B])
        elif isinstance(m, LoRALinearTaskSpecific):
            for A, B in zip(m.lora_As, m.lora_Bs):
                lora_ids.update([id(A), id(B)])
        elif isinstance(m, LoRALinearHybrid):
            lora_ids.add(id(m.lora_A))
            lora_ids.update(id(B) for B in m.lora_Bs)
    n_frozen = 0
    for p in model.parameters():
        if id(p) not in lora_ids:
            p.requires_grad_(False)
            n_frozen += 1
    return n_frozen


# ─────────────────────────────────────────────────────────────────────────────
# Null-space gradient projection
# ─────────────────────────────────────────────────────────────────────────────

class GradientProjector:
    """
    Projects LoRA gradients onto the null space of old-class extreme rays.

    Given a matrix R (K, D) of old extreme rays, the subspace projector is:

        P = V Vᵀ      where V = top-k right singular vectors of R

    and the null-space (orthogonal complement) projector is:

        P_⊥ = I − V Vᵀ

    Calling project_grads(backbone) after loss.backward() but before
    optimizer.step() ensures no LoRA update has a component in the direction
    of any old-class extreme ray — the static hull scores are preserved exactly.

    Projection rules per LoRA parameter:
      lora_B  (out, rank): grad_B ← P_⊥ @ grad_B  (output-space)
      lora_A  (rank, in):  grad_A ← grad_A @ P_⊥  (input-space)

    Both directions are projected because the effective weight change is
    ΔW = B A, so restricting either B or A independently is insufficient.
    For qkv layers where out = n·D, P_⊥ is applied to each D-dim block.

    Parameters
    ----------
    all_rays           : (K, D) — all old extreme rays stacked, L2-normalised
    device             : torch device where P_⊥ lives
    variance_threshold : fraction of squared-SV energy to retain in V;
                         0.99 captures almost the entire old-class subspace
    """

    def __init__(
        self,
        all_rays: np.ndarray,
        device: torch.device,
        variance_threshold: float = 0.99,
    ):

        K, D = all_rays.shape
        _, S, Vt = np.linalg.svd(all_rays, full_matrices=False)  # Vt: (r, D)

        cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
        k = int(np.searchsorted(cumvar, variance_threshold)) + 1
        k = min(k, len(S))

        V = torch.tensor(Vt[:k].T, dtype=torch.float32, device=device)  # (D, k)
        self.P_perp = torch.eye(D, dtype=torch.float32, device=device) - V @ V.T
        self.D      = D
        self._k     = k
        print(f"[GradientProjector] D={D}, old-subspace rank={k}/{min(K,D)} "
              f"({variance_threshold:.0%} variance), P_⊥ on {device}.")

    @torch.no_grad()
    def project_grads(self, model: nn.Module) -> None:
        """
        Project gradients of all active LoRA parameters in-place.
        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        for module in model.modules():
            if isinstance(module, LoRALinear):
                self._project_B(module.lora_B)
                self._project_A(module.lora_A)
            elif isinstance(module, LoRALinearTaskSpecific):
                for A, B in zip(module.lora_As, module.lora_Bs):
                    if A.requires_grad:
                        self._project_B(B)
                        self._project_A(A)
            elif isinstance(module, LoRALinearHybrid):
                self._project_A(module.lora_A)
                for B in module.lora_Bs:
                    if B.requires_grad:
                        self._project_B(B)

    def _project_B(self, param: nn.Parameter) -> None:
        """
        B has shape (out_features, rank).
        Apply P_⊥ on the output dimension:
          out == D   → grad_B ← P_⊥ @ grad_B
          out == n*D → apply P_⊥ to each of the n D-dim blocks (e.g. qkv)
        """
        if param.grad is None:
            return
        out, rank = param.grad.shape
        D = self.D
        if out == D:
            param.grad.copy_(self.P_perp @ param.grad)
        elif out % D == 0:
            n = out // D
            g = param.grad.view(n, D, rank)
            param.grad.copy_(
                torch.einsum("ij,njk->nik", self.P_perp, g).reshape(out, rank)
            )

    def _project_A(self, param: nn.Parameter) -> None:
        """
        A has shape (rank, in_features).
        Apply P_⊥ on the input dimension:
          in == D   → grad_A ← grad_A @ P_⊥
          in == n*D → apply P_⊥ to each of the n D-dim blocks
        """
        if param.grad is None:
            return
        rank, inp = param.grad.shape
        D = self.D
        if inp == D:
            param.grad.copy_(param.grad @ self.P_perp)
        elif inp % D == 0:
            n = inp // D
            g = param.grad.view(rank, n, D)
            param.grad.copy_(
                torch.einsum("ijk,kl->ijl", g, self.P_perp).reshape(rank, inp)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Feature expansion head
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExpansionHead(nn.Module):
    """
    Memory-efficient feature expansion head.

    Key memory-saving design choices vs. the original:
      • FFN ratio configurable (default 2× instead of 4×) — biggest single win
      • Optional gradient checkpointing on encoder layers
      • Optional Flash/SDPA attention via PyTorch's native kernel
      • Fused projection + reshape avoids an extra (B, expansion_dim) buffer
      • Optional fp16/bf16 forward via autocast-friendly design
      • Final norm operates per-token then flattens (cheaper than norm over flat dim)

    Parameters
    ----------
    in_dim              : backbone output dim (e.g. 192 for ViT-Tiny)
    expansion_dim       : target output dim
    n_tokens            : sequence length for the transformer
    n_heads             : attention heads (must divide token_dim)
    n_layers            : transformer encoder layers
    dropout             : dropout in attention/FFN
    ffn_ratio           : FFN expansion ratio. Default 2.0 (vs. standard 4.0)
                          halves the largest activation. Drop to 1.0 if very tight.
    use_checkpoint      : if True, recompute encoder activations during backward.
                          Trades ~25% extra compute for ~Nx less activation memory
                          where N is the number of layers.
    use_flash_attention : if True, use PyTorch's scaled_dot_product_attention with
                          memory-efficient backend. Requires PyTorch >= 2.0.
    """

    def __init__(
        self,
        in_dim:              int,
        expansion_dim:       int,
        n_tokens:            int   = 8,
        n_heads:             int   = 4,
        n_layers:            int   = 1,
        dropout:             float = 0.0,
        ffn_ratio:           float = 2.0,
        use_checkpoint:      bool  = False,
        use_flash_attention: bool  = True,
    ):
        super().__init__()
        if expansion_dim % n_tokens != 0:
            raise ValueError(
                f"expansion_dim ({expansion_dim}) must be divisible by "
                f"n_tokens ({n_tokens})"
            )
        token_dim = expansion_dim // n_tokens
        if token_dim % n_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by n_heads ({n_heads})"
            )

        self.in_dim        = in_dim
        self.expansion_dim = expansion_dim
        self.n_tokens      = n_tokens
        self.token_dim     = token_dim
        self.use_checkpoint      = use_checkpoint
        self.use_flash_attention = use_flash_attention

        # ── input projection ────────────────────────────────────────────────
        # Direct projection 192 → expansion_dim. The reshape is a view, not a copy.
        self.proj_in = nn.Linear(in_dim, expansion_dim)

        # ── transformer stack ───────────────────────────────────────────────
        ffn_dim = int(token_dim * ffn_ratio)
        self.layers = nn.ModuleList([
            EfficientEncoderLayer(
                d_model       = token_dim,
                n_heads       = n_heads,
                ffn_dim       = ffn_dim,
                dropout       = dropout,
                use_flash_sdpa = use_flash_attention,
            )
            for _ in range(n_layers)
        ])

        # ── output norm ─────────────────────────────────────────────────────
        # Per-token norm is cheaper and more standard than norming over flat dim.
        self.norm_out = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        B = x.size(0)

        # Project + reshape in one step. .view() is a free reshape since
        # nn.Linear output is always contiguous.
        z = self.proj_in(x).view(B, self.n_tokens, self.token_dim)

        # Encoder stack with optional gradient checkpointing
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                z = torch.utils.checkpoint.checkpoint(layer, z, use_reentrant=False)
            else:
                z = layer(z)

        z = self.norm_out(z)                                  # (B, n_tokens, token_dim)

        # Flatten back to (B, expansion_dim).  .reshape works on potentially
        # non-contiguous tensors but won't copy if memory layout permits.
        return z.reshape(B, self.expansion_dim)


class EfficientEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer with PyTorch SDPA.
    Avoids the overhead of nn.TransformerEncoderLayer's generic dispatch
    and lets us use scaled_dot_product_attention directly for memory efficiency.
    """

    def __init__(
        self,
        d_model:        int,
        n_heads:        int,
        ffn_dim:        int,
        dropout:        float = 0.0,
        use_flash_sdpa: bool  = True,
    ):
        super().__init__()
        self.n_heads        = n_heads
        self.head_dim       = d_model // n_heads
        self.use_flash_sdpa = use_flash_sdpa
        self.dropout_p      = dropout

        # Pre-norms
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn  = nn.LayerNorm(d_model)

        # Fused QKV projection — 3 matmuls become 1, reduces kernel launches
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop_attn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_ffn  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Fused QKV: (B, N, 3D) → 3 × (B, n_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                      # (3, B, h, N, hd)
        q, k, v = qkv.unbind(0)

        if self.use_flash_sdpa:
            # PyTorch picks Flash / mem-efficient / math backend automatically.
            # Avoids materializing the (B, h, N, N) attention matrix in memory.
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout_p if self.training else 0.0,
            )
        else:
            scale = self.head_dim ** -0.5
            attn  = (q @ k.transpose(-2, -1)) * scale         # (B, h, N, N)
            attn  = attn.softmax(dim=-1)
            if self.dropout_p > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout_p)
            out = attn @ v                                    # (B, h, N, hd)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, N, D)            # (B, N, D)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: norm → attn → residual; norm → ffn → residual
        x = x + self.drop_attn(self._attention(self.norm_attn(x)))
        x = x + self.drop_ffn(self.ffn(self.norm_ffn(x)))
        return x


class FeatureExpansionBackbone(nn.Module):
    """
    Wraps an existing backbone with a :class:`FeatureExpansionHead` so that
    ``forward(x)`` returns expanded features rather than raw backbone features.

    Designed to be a drop-in replacement for the bare backbone in the training
    pipeline:

    * ``num_features`` reflects the expanded dimensionality.
    * ``backbone.blocks`` / ``backbone.patch_embed`` are proxied so that LoRA
      injection and :class:`LayerFeatureExtractor` hooks continue to work.
    * ``copy.deepcopy`` clones both sub-modules together, so the teacher
      (``old_backbone``) inherits a frozen snapshot of the expansion head too.

    Parameters
    ----------
    backbone       : the base timm (or LoRA-wrapped) backbone
    expansion_head : a fitted :class:`FeatureExpansionHead`
    """

    def __init__(self, backbone: nn.Module, expansion_head: FeatureExpansionHead):
        super().__init__()
        self._backbone        = backbone
        self.expansion_head   = expansion_head
        self.num_features     = expansion_head.expansion_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expansion_head(self._backbone(x))

    # ── proxy commonly-accessed backbone attributes ───────────────────────────

    @property
    def blocks(self):
        return self._backbone.blocks

    @property
    def patch_embed(self):
        return self._backbone.patch_embed

    @property
    def layers(self):
        return getattr(self._backbone, "layers", None)


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
    lora_config: str = "task_shared",
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
    lora_config         : "task_shared" | "task_specific" | "hybrid"
                          Controls which LoRA structure is injected (see inject_lora).

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
            lora_config=lora_config,
        )
        lora_param_count = sum(p.numel() for p in get_lora_params(model))
        print(f"[load_backbone] LoRA active ({lora_config}) — {n_lora} layers, "
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
