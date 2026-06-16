"""
CHCL drift-control experiment: does backprop through the conic prediction help,
and does null-space projection stop backbone drift?

Three training arms over a 10-task class-incremental protocol:
  ce         : cross-entropy with a growing linear head (baseline; head drags backbone)
  cone       : cone-margin loss replacing CE. Danskin gradient through NNLS residual:
               attraction to own (provisional) cone + hinge repulsion from all other cones.
               No classification head at all.
  cone_proj  : cone arm + GPM-style null-space gradient projection. Protected subspace
               per layer = span of input activations of the RAY-ANCHOR samples
               (SPA rays are actual training samples, so the cone-defining inputs
               double as the projection basis -- no extra exemplar machinery).

Deployment-realistic evaluation for all arms: argmin NNLS residual against rays
STORED AT FORMATION TIME, while the backbone keeps training. epsilon = mean angular
drift of probe features vs their snapshot at storage; gamma = min inter-class ray angle.

Swap to real data / ViT on your 5090:
    python drift_control.py --data cifar100 --backbone vit --device cuda
The synthetic+MLP path is verified end-to-end; the ViT path follows your config
(LoRA r=8, alpha=4.0 on the last N blocks) but is written blind here -- expect to
touch the timm module-name matching once.
"""

import argparse
import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "cifar100"], default="synthetic")
    p.add_argument("--backbone", choices=["mlp", "vit"], default="mlp")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--classes_per_task", type=int, default=10)
    p.add_argument("--k_rays", type=int, default=8)
    p.add_argument("--feat_dim", type=int, default=128)  # mlp path only
    p.add_argument("--steps_per_task", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--eval_bs", type=int, default=64
    )  # mini-batch for no_grad full-set forwards (lower if ViT OOMs)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ray_refresh", type=int, default=50)  # DEPRECATED (rays now fixed per task)
    p.add_argument(
        "--hinge_tau", type=float, default=0.45
    )  # repulsion margin (rel. residual)
    p.add_argument("--lam_rep", type=float, default=1.0)  # repulsion weight
    p.add_argument("--rep_targets", type=int, default=16)  # max cones repelled per step
    p.add_argument("--gpm_energy", type=float, default=0.97)  # basis energy threshold
    p.add_argument(
        "--gpm_max_frac", type=float, default=0.5
    )  # cap protected rank at frac*in_dim (keeps free capacity for plasticity)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["ce", "cone", "cone_proj"])
    # ViT-specific
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=4.0)
    p.add_argument("--lora_blocks", type=int, default=4)  # adapt last N blocks
    return p.parse_args()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_synth(rng, D_in=64, n_classes=100, n_super=20, n_tr=100, n_te=20):
    sc = rng.normal(size=(n_super, D_in))
    sc /= np.linalg.norm(sc, axis=1, keepdims=True)
    Xtr, ytr, Xte, yte = [], [], [], []
    for c in range(n_classes):
        cd = rng.normal(size=D_in)
        cd /= np.linalg.norm(cd)
        mean = 3.0 * sc[c // (n_classes // n_super)] + 2.2 * cd
        for n, X, y in [(n_tr, Xtr, ytr), (n_te, Xte, yte)]:
            X.append(mean[None] + 0.85 * rng.normal(size=(n, D_in)))
            y.append(np.full(n, c))
    return (
        torch.tensor(np.concatenate(Xtr), dtype=torch.float32),
        torch.tensor(np.concatenate(ytr)),
        torch.tensor(np.concatenate(Xte), dtype=torch.float32),
        torch.tensor(np.concatenate(yte)),
    )


class LazyImageSet:
    """Index-bookkeeping view over a torchvision dataset. Boolean-mask indexing
    returns another lazy view (cheap); slice / int-tensor indexing materializes
    only those images into a real tensor. Lets the rest of the pipeline treat it
    like Xtr/Xte without ever holding all 50k decoded 224px tensors in RAM."""

    def __init__(self, ds, idx=None):
        self.ds = ds
        self.idx = torch.arange(len(ds)) if idx is None else idx

    def __len__(self):
        return len(self.idx)

    def _gather(self, global_idx):
        return torch.stack([self.ds[int(i)][0] for i in global_idx])

    def __getitem__(self, key):
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            return LazyImageSet(self.ds, self.idx[key])  # lazy view
        if isinstance(key, slice):
            return self._gather(self.idx[key])
        return self._gather(self.idx[torch.as_tensor(key)])  # int tensor/list


def get_dataset(args, rng):
    """Returns (Xtr, ytr, Xte, yte) as tensors. Swap point for real CIFAR-100."""
    if args.data == "synthetic":
        return make_synth(rng)
    if False:
        D_in, n_classes, n_super = 64, 100, 20
        sc = rng.normal(size=(n_super, D_in))
        sc /= np.linalg.norm(sc, axis=1, keepdims=True)
        Xtr, ytr, Xte, yte = [], [], [], []
        for c in range(n_classes):
            cd = rng.normal(size=D_in)
            cd /= np.linalg.norm(cd)
            mean = 3.0 * sc[c // 5] + 2.2 * cd
            for n, X, y in [(100, Xtr, ytr), (20, Xte, yte)]:
                X.append(mean[None] + 0.85 * rng.normal(size=(n, D_in)))
                y.append(np.full(n, c))
        return (
            torch.tensor(np.concatenate(Xtr), dtype=torch.float32),
            torch.tensor(np.concatenate(ytr)),
            torch.tensor(np.concatenate(Xte), dtype=torch.float32),
            torch.tensor(np.concatenate(yte)),
        )
    else:
        # ---- CIFAR-100 path (run on your machine; downloads ~170MB) ----
        from torchvision import datasets, transforms

        tf = transforms.Compose(
            [
                transforms.Resize(224),  # ViT input size
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
        tr = datasets.CIFAR100("./data", train=True, download=True, transform=tf)
        te = datasets.CIFAR100("./data", train=False, download=True, transform=tf)
        # Index lazily: images are decoded+transformed on access, never all at
        # once. Labels are tiny, so keep them as plain tensors.
        return (
            LazyImageSet(tr),
            torch.tensor(tr.targets),
            LazyImageSet(te),
            torch.tensor(te.targets),
        )


# ----------------------------------------------------------------------------
# Backbones
# ----------------------------------------------------------------------------
class MLPBackbone(nn.Module):
    def __init__(self, d_in=64, d_feat=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_feat),
        )

    def forward(self, x):
        return self.net(x)


class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a trainable low-rank update (your r=8, alpha=4)."""

    def __init__(self, base: nn.Linear, r=8, alpha=4.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


PRETRAINED_STATE = None  # cached across arms for fairness


def pretrain_mlp(args, device):
    """ImageNet analog: CE-pretrain on a DISJOINT synthetic class set, then freeze."""
    global PRETRAINED_STATE
    if PRETRAINED_STATE is not None:
        return
    rng = np.random.default_rng(args.seed + 777)
    Xp, yp, _, _ = make_synth(rng)
    torch.manual_seed(args.seed)
    net = MLPBackbone(64, args.feat_dim).to(device)
    head = nn.Linear(args.feat_dim, 100).to(device)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=2e-3)
    Xp, yp = Xp.to(device), yp.to(device)
    for step in range(600):
        sel = torch.randint(0, len(Xp), (128,), device=device)
        loss = F.cross_entropy(head(net(Xp[sel])), yp[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
    PRETRAINED_STATE = copy.deepcopy(net.state_dict())


def build_backbone(args, device=torch.device("cpu")):
    """Swap point for the ViT path."""
    if args.backbone == "mlp":
        pretrain_mlp(args, device)
        net = MLPBackbone(64, args.feat_dim)
        net.load_state_dict(PRETRAINED_STATE)
        for p in net.parameters():
            p.requires_grad_(False)
        # inject LoRA on every Linear (mirrors your ViT r=8, alpha=4 setup)
        for i, m in enumerate(net.net):
            if isinstance(m, nn.Linear):
                net.net[i] = LoRALinear(m, args.lora_rank, args.lora_alpha)
        return net, args.feat_dim
    # ---- ViT + LoRA path (untested in this sandbox; matches your config) ----
    import timm

    vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    for p in vit.parameters():
        p.requires_grad_(False)
    n_blocks = len(vit.blocks)
    for i in range(n_blocks - args.lora_blocks, n_blocks):
        blk = vit.blocks[i]
        blk.attn.qkv = LoRALinear(blk.attn.qkv, args.lora_rank, args.lora_alpha)
        blk.attn.proj = LoRALinear(blk.attn.proj, args.lora_rank, args.lora_alpha)
        blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, args.lora_rank, args.lora_alpha)
        blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, args.lora_rank, args.lora_alpha)
    return vit, vit.num_features  # forward() returns pooled CLS features


# ----------------------------------------------------------------------------
# Conic machinery (torch, batched)
# ----------------------------------------------------------------------------
def spa_indices(F_cls: torch.Tensor, k: int):
    """SPA on normalized features. Returns indices of selected extreme-ray SAMPLES."""
    R = F.normalize(F_cls, dim=1).double()
    residual = R.clone()
    idx = []
    for _ in range(k):
        j = int(torch.linalg.norm(residual, dim=1).argmax())
        u = F.normalize(R[j], dim=0)
        idx.append(j)
        residual = residual - torch.outer(residual @ u, u)
    return idx


def nnls_residual(X: torch.Tensor, R: torch.Tensor, iters=40):
    """Relative NNLS residual of each row of X against cone spanned by rows of R.
    w* solved with projected gradient under no_grad (Danskin: dL/dx = residual,
    holding w* fixed) -> returned residual is differentiable in X only."""
    A = R.t()  # (d, k), detached rays
    G = A.t() @ A
    L = torch.linalg.eigvalsh(G)[-1] + 1e-9
    with torch.no_grad():
        AtX = A.t() @ X.t().detach()
        W = (AtX / L).clamp_min(0)
        for _ in range(iters):
            W = (W - (G @ W - AtX) / L).clamp_min(0)
    recon = (A @ W).t()  # constant wrt X
    return (X - recon).norm(dim=1) / (X.norm(dim=1) + 1e-12)


@torch.no_grad()
def classify(feats, ray_bank):
    classes = sorted(ray_bank.keys())
    res = torch.stack([nnls_residual(feats, ray_bank[c]) for c in classes], dim=1)
    return torch.tensor(classes)[res.argmin(dim=1)]


@torch.no_grad()
def batched_features(backbone, X, device, bs=64):
    """Forward X (kept on CPU) through the backbone in mini-batches.
    Returns features on `device`. Use for every full-set / full-class pass so we
    never materialize all activations for 224x224 images at once."""
    outs = []
    for i in range(0, len(X), bs):
        outs.append(backbone(X[i : i + bs].to(device)))
    return torch.cat(outs, 0)


# ----------------------------------------------------------------------------
# GPM-style null-space projection
# ----------------------------------------------------------------------------
class GradProjector:
    """Per-Linear protected input subspace from ray-anchor activations.
    After backward(): G <- G - G U U^T  (zero first-order output change on span(U))."""

    def __init__(self, energy=0.97, max_frac=0.5):
        self.energy = energy
        # Fix 3: cap protected rank at max_frac * in_dim so every layer keeps free
        # capacity for new tasks. Without it the 64-dim first layer and r=8 LoRA-B
        # subspaces saturate to FULL rank by ~task 3, projecting all gradients to
        # zero -> no plasticity (acc decays). When over budget we keep the
        # highest-ENERGY directions, so the most-used old activations stay protected.
        self.max_frac = max_frac
        self.basis = {}  # layer name -> (U (in_dim, r) orthonormal, energy (r,))
        self._acts = {}
        self._hooks = []

    def _trainable_linears(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and any(
                p.requires_grad for p in m.parameters()
            ):
                yield name, m

    def collect(self, model, anchor_inputs, forward_fn):
        self._acts = {}

        def mk_hook(name):
            def hook(mod, inp, out):
                self._acts.setdefault(name, []).append(
                    inp[0].detach().reshape(-1, inp[0].shape[-1])
                )

            return hook

        hs = [
            m.register_forward_hook(mk_hook(n))
            for n, m in self._trainable_linears(model)
        ]
        with torch.no_grad():
            forward_fn(anchor_inputs)
        for h in hs:
            h.remove()
        for name, chunks in self._acts.items():
            Acts = torch.cat(chunks, 0).t().double()  # (in_dim, n)
            in_dim = Acts.shape[0]
            cap = max(1, int(self.max_frac * in_dim))
            U, energy = self.basis.get(name, (None, None))
            if U is not None:  # credit existing dirs for energy in the new task
                proj = U.t() @ Acts
                energy = energy + proj.pow(2).sum(dim=1)
                Acts = Acts - U @ proj  # residual orthogonal to U
            try:
                Unew, S, _ = torch.linalg.svd(Acts, full_matrices=False)
            except RuntimeError:
                if U is not None:
                    self.basis[name] = (U, energy)
                continue
            if S.numel() and S[0] > 1e-8:
                cum = torch.cumsum(S**2, 0) / (S**2).sum()
                r = int((cum < self.energy).sum()) + 1
                Unew, Snew = Unew[:, :r], S[:r] ** 2
                if U is not None:
                    U = torch.cat([U, Unew], 1)
                    energy = torch.cat([energy, Snew])
                else:
                    U, energy = Unew, Snew
            if U is None:
                continue
            if U.shape[1] > cap:  # keep the highest-energy directions
                keep = torch.argsort(energy, descending=True)[:cap]
                U, energy = U[:, keep], energy[keep]
            self.basis[name] = (U, energy)

    def project_grads(self, model):
        for name, m in self._trainable_linears(model):
            if name in self.basis and m.weight.grad is not None:
                U = self.basis[name][0].to(m.weight.grad.dtype)
                G = m.weight.grad
                m.weight.grad = G - (G @ U) @ U.t()

    def snapshot(self, model):
        """Pre-step weights of the protected layers, for project_update()."""
        return {
            n: m.weight.detach().clone()
            for n, m in self._trainable_linears(model)
            if n in self.basis
        }

    def project_update(self, model, prev):
        """Project the REALIZED weight delta onto the null space of each protected
        subspace: DW <- DW - DW U U^T. GPM's DW.U=0 guarantee only holds for raw
        SGD; Adam's per-element rescaling + momentum re-inject components along U
        even after the gradient is projected. Projecting the post-step delta makes
        the guarantee hold for ANY optimizer."""
        for name, m in self._trainable_linears(model):
            if name in self.basis and name in prev:
                U = self.basis[name][0].to(m.weight.dtype)
                with torch.no_grad():
                    delta = m.weight - prev[name]
                    m.weight.copy_(prev[name] + (delta - (delta @ U) @ U.t()))


# ----------------------------------------------------------------------------
# Training arms
# ----------------------------------------------------------------------------
def run_arm(arm, args, data, device):
    torch.manual_seed(args.seed)
    # Keep data on CPU; stream batches to `device` per step. Moving the whole
    # CIFAR-100/224px tensor to the GPU is ~30GB and OOMs before training starts.
    Xtr, ytr, Xte, yte = data
    backbone, d_feat = build_backbone(args, device)
    backbone.to(device)
    head = None
    if arm == "ce":
        head = nn.Linear(d_feat, 0, bias=True).to(device)  # grows per task

    projector = (
        GradProjector(args.gpm_energy, args.gpm_max_frac)
        if arm == "cone_proj"
        else None
    )

    ray_bank = {}  # class -> stored rays (formation-time, frozen)
    anchor_x = {}  # class -> input samples that defined the rays
    probe = {}  # class -> (inputs, snapshot feats) for epsilon measurement
    n_tasks, cpt = args.n_tasks, args.classes_per_task
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    eps_log, gamma_log = [], []

    for t in range(n_tasks):
        new_classes = list(range(t * cpt, (t + 1) * cpt))
        task_mask = torch.isin(ytr, torch.tensor(new_classes))
        Xt, yt = Xtr[task_mask], ytr[task_mask]  # stay on CPU

        params = [p for p in backbone.parameters() if p.requires_grad]
        if arm == "ce":
            old_w, old_b = (head.weight.data.clone(), head.bias.data.clone())
            head = nn.Linear(d_feat, (t + 1) * cpt).to(device)
            head.weight.data[: t * cpt] = old_w
            head.bias.data[: t * cpt] = old_b
            params = params + list(head.parameters())
        opt = torch.optim.Adam(params, lr=args.lr)

        # Fix 1: anchor each class cone to FIXED rays computed once at task start
        # (detached, well-separated ~pretrained geometry). Refreshing the rays from
        # the live features every N steps made the attraction target chase the
        # features, whose trivial minimizer is collapse (gamma -> ~5deg, acc ->
        # chance). A frozen target removes that degeneracy.
        provisional = {}
        if arm != "ce":
            with torch.no_grad():
                for c in new_classes:
                    fc = batched_features(backbone, Xt[yt == c], device, args.eval_bs)
                    provisional[c] = F.normalize(fc, dim=1)[
                        spa_indices(fc, args.k_rays)
                    ].float()
        for step in range(args.steps_per_task):
            sel = torch.randint(0, len(Xt), (args.batch_size,))
            xb, yb = Xt[sel].to(device), yt[sel].to(device)
            feats = backbone(xb)

            if arm == "ce":
                loss = F.cross_entropy(head(feats), yb)
            else:
                loss = 0.0
                for c in new_classes:
                    m = yb == c
                    if not m.any():
                        continue
                    fc = feats[m]
                    loss = loss + nnls_residual(fc, provisional[c].to(feats)).mean()
                    others = [provisional[o] for o in new_classes if o != c] + [
                        ray_bank[o] for o in ray_bank
                    ]
                    if len(others) > args.rep_targets:  # subsample repulsion targets
                        sel_o = torch.randperm(len(others))[: args.rep_targets]
                        others = [others[i] for i in sel_o]
                    for Ro in others:
                        r = nnls_residual(fc, Ro.to(feats))
                        loss = loss + args.lam_rep * F.relu(args.hinge_tau - r).mean()
                loss = loss / len(new_classes)

            opt.zero_grad()
            loss.backward()
            prev = None
            if projector is not None:
                projector.project_grads(backbone)  # keeps Adam's moments cleaner
                prev = projector.snapshot(backbone)  # for the delta projection
            opt.step()
            if projector is not None:
                projector.project_update(backbone, prev)  # guarantee DW.U=0

        # ---- form & freeze hulls for the new classes ----
        with torch.no_grad():
            for c in new_classes:
                xc = Xtr[ytr == c]  # CPU
                fc = batched_features(backbone, xc, device, args.eval_bs)
                idx = spa_indices(fc, args.k_rays)
                ray_bank[c] = F.normalize(fc, dim=1)[idx].float().cpu()
                anchor_x[c] = xc[idx].clone()  # CPU input snapshot
                pm = torch.isin(yte, torch.tensor([c]))
                px = Xte[pm][:20]
                probe[c] = (
                    px,
                    F.normalize(
                        batched_features(backbone, px, device, args.eval_bs), dim=1
                    ).cpu(),
                )

        if projector is not None:
            anchors = torch.cat([anchor_x[c] for c in new_classes], 0).to(device)
            projector.collect(backbone, anchors, lambda x: backbone(x))

        # ---- measure epsilon (drift of old-class features vs snapshot) ----
        with torch.no_grad():
            drifts = []
            for c, (px, snap) in probe.items():
                if c in new_classes:
                    continue
                cur = F.normalize(
                    batched_features(backbone, px, device, args.eval_bs), dim=1
                ).cpu()
                cos = (cur * snap).sum(1).clamp(-1, 1)
                drifts.append(torch.rad2deg(torch.acos(cos)).mean().item())
            eps_log.append(float(np.mean(drifts)) if drifts else 0.0)

            # gamma: min inter-class ray angle in the stored bank
            allR = torch.cat([ray_bank[c] for c in sorted(ray_bank)], 0)
            labs = np.repeat(sorted(ray_bank), args.k_rays)
            C = (allR @ allR.t()).clamp(-1, 1).numpy()
            diff = labs[:, None] != labs[None, :]
            gamma_log.append(float(np.degrees(np.arccos(C[diff].max()))))

            # ---- deployment eval: frozen stored rays + current backbone ----
            seen = torch.tensor(sorted(ray_bank.keys()))
            em = torch.isin(yte, seen)
            fe = batched_features(backbone, Xte[em], device, args.eval_bs).cpu()
            pred = classify(fe, ray_bank)
            ye = yte[em].cpu()
            for j in range(t + 1):
                jm = torch.isin(ye, torch.arange(j * cpt, (j + 1) * cpt))
                acc_matrix[t, j] = (pred[jm] == ye[jm]).float().mean().item()
        print(
            f"[{arm}] task {t+1}/{n_tasks}  avg_acc={np.nanmean(acc_matrix[t,:t+1]):.3f}  "
            f"eps={eps_log[-1]:.1f}deg  gamma={gamma_log[-1]:.1f}deg"
        )

    final = acc_matrix[-1]
    peak = np.nanmax(acc_matrix, axis=0)
    return {
        "acc_matrix": acc_matrix.tolist(),
        "eps": eps_log,
        "gamma": gamma_log,
        "avg_final_acc": float(np.nanmean(final)),
        "forgetting": float(np.nanmean(peak[:-1] - final[:-1])),
    }


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    rng = np.random.default_rng(args.seed)
    data = get_dataset(args, rng)
    out = {}
    for arm in args.arms:
        out[arm] = run_arm(arm, args, data, torch.device(args.device))
        print(
            f"== {arm}: final={out[arm]['avg_final_acc']:.3f} "
            f"forgetting={out[arm]['forgetting']:.3f} "
            f"final_eps={out[arm]['eps'][-1]:.1f} "
            f"gamma/2={out[arm]['gamma'][-1]/2:.1f}\n"
        )
    name = "drift_results_" + "_".join(args.arms) + ".json"
    with open(name, "w") as f:
        json.dump(out, f)
