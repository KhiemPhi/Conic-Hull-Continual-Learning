"""
exp5_gradient_cone.py — THE CONIC HULL OF GRADIENTS (not features).

Every cone failure so far was in FEATURE space. But continual learning already has a hidden
conic problem: GEM / A-GEM constrain the update so that <g_new, g_old> >= 0 for stored old
gradients — an inequality system, i.e. a CONE membership condition. They just never ask what
the gradient cone LOOKS like.

Two questions this asks:
  Q1 GEOMETRY. Is the per-task gradient set actually cone-shaped and low-dimensional? What is
     the effective number of extreme rays? If a task's gradients are spanned by a few
     directions, GEM's per-exemplar storage is enormously wasteful.
  Q2 MEMORY. GEM stores EXEMPLARS to recompute old gradients. Could you instead store K
     gradient DIRECTIONS? Compare K-sized memories at matched budget for predicting the TRUE
     conflict (measured against the full old-gradient set):
        mean gradient | K random gradients | K k-means centroids | K conic extreme rays
     This is basis/spanning-use, the one regime where conic structure has ever helped — and
     feature-space ray INSTABILITY (which killed every other use) may not transfer here,
     because gradients are averages over a batch rather than single data points.

Gradients of the LoRA params (~1.77M dims) are sketched to a fixed random subspace
(Johnson-Lindenstrauss preserves inner products, which is all the conflict test needs).

Run:  python -u exp5_gradient_cone.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform
from sklearn.cluster import KMeans

from backbone import load_backbone, freeze_non_lora, get_lora_params
from conic_hull import ConicHull

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT = 10, 20
ADAPTERS = "crux_routing_adapters.pt"
SKETCH = int(os.environ.get("SKETCH", 512))     # JL sketch dim
N_GRAD = int(os.environ.get("N_GRAD", 24))      # gradient samples per task
GBS = int(os.environ.get("GBS", 32))            # batch size per gradient
BUDGETS = [1, 3, 5, 10]

TF = create_transform(**resolve_model_data_config(
    timm.create_model(MODEL, pretrained=False, num_classes=0)), is_training=False)


class HFWrap(Dataset):
    def __init__(self, ds, idx, labels):
        self.ds, self.idx, self.labels = ds, np.asarray(idx), np.asarray(labels)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return TF(img), int(self.labels[i])


from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"]); _cl = np.array(sorted(set(_w.tolist())))
_lab = np.searchsorted(_cl, _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
TRAIN = HFWrap(_ds, _p[:_n], _lab[_p[:_n]]);  TR_Y = _lab[_p[:_n]]
N_CLS = len(_cl)
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]

model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
LP = list(get_lora_params(model))
NAMES = [n for n, p in model.named_parameters() if p.requires_grad]
D = sum(p.numel() for p in LP)
print(f"[lora] {D:,} params -> JL sketch to {SKETCH}")
gen = torch.Generator(device=DEV).manual_seed(0)
S = torch.randn(D, SKETCH, generator=gen, device=DEV) / np.sqrt(SKETCH)

adapters = torch.load(ADAPTERS, map_location="cpu")


def load_adapter(s):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in adapters[s]:
                p.copy_(adapters[s][n].to(DEV))


def task_gradients(t, at_stage, n=N_GRAD):
    """n sketched gradient vectors for task t, evaluated at the stage-`at_stage` weights."""
    load_adapter(at_stage)
    cls = TASKS[t]
    remap = {int(c): i for i, c in enumerate(cls)}
    idx = np.where(np.isin(TR_Y, cls))[0]
    rng = np.random.default_rng(t)
    head = nn.Linear(768, CPT).to(DEV)
    ce = nn.CrossEntropyLoss()
    out = []
    for i in range(n):
        b = rng.choice(idx, GBS, replace=False)
        loader = DataLoader(Subset(TRAIN, b.tolist()), batch_size=GBS, num_workers=4)
        x, lab = next(iter(loader))
        x = x.to(DEV)
        y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
        model.zero_grad(set_to_none=True)
        if head.weight.grad is not None:
            head.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ce(head(model(x).float()), y)
        loss.backward()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                       for p in LP]).float()
        out.append((g @ S).detach().cpu().numpy())
    return np.stack(out)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ============================ Q1: geometry of the gradient set ============================
print("\n=== Q1: is the per-task gradient set cone-shaped / low-dimensional? ===")
G = {}
for t in range(N_TASKS):
    G[t] = task_gradients(t, at_stage=t)
    Gn = un(G[t])
    sv = np.linalg.svd(Gn, compute_uv=False)
    er = float((sv.sum() ** 2) / (sv ** 2).sum())          # participation ratio
    off = (Gn @ Gn.T)[~np.eye(len(Gn), dtype=bool)]
    print(f"  task {t}: n={len(Gn)} eff-rank {er:5.2f} | pairwise cos "
          f"mean {off.mean():+.3f} min {off.min():+.3f} | frac<0 {float((off<0).mean()):.3f}",
          flush=True)

# ============================ Q2: K-sized gradient memory ============================
print("\n=== Q2: which K-sized memory best predicts conflict with an OLD task? ===")
# "truth" = mean inner product of a NEW-task gradient against ALL of the old task's gradients
rows = []
for t_old in range(N_TASKS - 1):
    t_new = t_old + 1
    Gold = un(G[t_old])
    Gnew = un(task_gradients(t_new, at_stage=t_old))       # new-task grads at old weights
    truth = (Gnew @ Gold.T).mean(1)                        # (n_new,)
    for K in BUDGETS:
        sets = {}
        sets["mean"] = un(Gold.mean(0, keepdims=True)).repeat(max(K, 1), 0)[:1]
        r = np.random.default_rng(t_old)
        sets["random"] = Gold[r.choice(len(Gold), min(K, len(Gold)), replace=False)]
        k = int(min(K, len(Gold)))
        sets["kmeans"] = un(KMeans(n_clusters=k, n_init=4, random_state=0)
                            .fit(Gold).cluster_centers_)
        h = ConicHull(n_rays=k, use_pca=True,
                      pca_dim=int(min(16, max(len(Gold) - 1, 2)))).fit(Gold)
        sets["cone_rays"] = h.extreme_rays_
        for nm, M in sets.items():
            est = (Gnew @ M.T).mean(1)
            rho = float(np.corrcoef(est, truth)[0, 1]) if np.std(est) > 1e-12 else 0.0
            err = float(np.abs(est - truth).mean())
            rows.append(dict(t=t_old, K=K, mem=nm, rho=rho, err=err))
    print(f"  pair {t_old}->{t_new} done", flush=True)

np.save("exp5_results.npy", rows, allow_pickle=True)
print("\n" + "=" * 92)
print("EXP5 — conic hull of GRADIENTS (ImageNet-R, LoRA grads, JL-sketched)")
print("=" * 92)
print(f"{'K':>3} {'memory':>11} {'corr w/ true conflict':>22} {'abs err':>10}")
for K in BUDGETS:
    for nm in ["mean", "random", "kmeans", "cone_rays"]:
        sel = [r for r in rows if r["K"] == K and r["mem"] == nm]
        if not sel:
            continue
        print(f"{K:>3} {nm:>11} {np.mean([s['rho'] for s in sel]):>22.4f} "
              f"{np.mean([s['err'] for s in sel]):>10.4f}")
    print()
print("-" * 92)
best = {}
for nm in ["mean", "random", "kmeans", "cone_rays"]:
    best[nm] = np.mean([r["rho"] for r in rows if r["mem"] == nm])
print("mean corr by memory:", {k: round(v, 4) for k, v in best.items()})
print(f"cone_rays - kmeans = {best['cone_rays']-best['kmeans']:+.4f}   "
      "<- rays vs centroids at matched budget")
print(f"cone_rays - mean   = {best['cone_rays']-best['mean']:+.4f}   "
      "<- is ANY structure better than the mean gradient?")
print("\nWIN CONDITION: cone_rays beats mean AND kmeans. That would say the conflict set has")
print("genuine conic structure a few extreme directions capture — i.e. GEM could store K")
print("directions instead of exemplars. If 'mean' wins again, gradient space behaves like")
print("feature space and the cone line is closed everywhere.")
print("Q1 READ: eff-rank << n and frac(cos<0) > 0 => a real, low-dimensional conflict cone.")
print("=" * 92)
