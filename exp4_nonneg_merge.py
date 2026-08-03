"""
exp4_nonneg_merge.py — NON-NEGATIVE MODEL MERGING (cones in WEIGHT space, not feature space).

Task arithmetic and model soups combine adapters with SIGNED coefficients. But a signed
combination can subtract a task's contribution — destructive interference. Restricting to a
CONIC combination (alpha >= 0) of adapters is the weight-space analogue of a conic hull, and
it is BASIS-use (spanning), which is the only regime where conic structure has ever helped.

    theta_merged = theta_base + sum_s alpha_s * A_s ,   alpha_s >= 0   (conic)
                                                  vs   alpha_s free    (signed / task arith.)

Uses the 10 LoRA adapters already snapshotted in crux_routing_adapters.pt (ImageNet-R,
20 classes each). Merged model is evaluated over ALL 200 classes with NCM + RanPAC, so a
single merged backbone must serve every task — exactly where interference shows up.

Strategies compared:
  last_only        alpha = e_9                      (what sequential training leaves you)
  uniform          alpha = 1/T                      (model soup)
  signed_search    coordinate ascent, alpha free    (task arithmetic, can subtract)
  conic_search     coordinate ascent, alpha >= 0    (the proposal)
  conic_extreme    keep only the K adapters that are extreme rays of the adapter cone
                   (does a SUBSET of adapters span the family?)
Reference: per-stage oracle (each class scored in its own birth frame) and the joint ceiling.

Run:  python -u exp4_nonneg_merge.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora
from conic_hull import ConicHull

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda"
# MUST match the backbone that produced ADAPTERS (crux_routing_adapters.pt, written
# 2026-07-31 12:43 from the augreg2_in21k_ft_in1k run). Loading adapters trained on a
# different backbone would silently produce garbage, so this is not a free knob: change it
# only together with regenerating the adapters.
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg2_in21k_ft_in1k")
N_TASKS, CPT = 10, 20
ADAPTERS = "crux_routing_adapters.pt"
N_VAL = int(os.environ.get("N_VAL", 1500))     # val subset for coefficient search
ROUNDS = int(os.environ.get("ROUNDS", 2))      # coordinate-ascent sweeps
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
SIGNED_GRID = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

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
TEST  = HFWrap(_ds, _p[_n:], _lab[_p[_n:]]);  TE_Y = _lab[_p[_n:]]
N_CLS = len(_cl)
rng = np.random.default_rng(SEED)
PROTO_IDX = np.concatenate([np.where(TR_Y == c)[0][:32] for c in range(N_CLS)])
VAL_IDX = rng.choice(len(TE_Y), min(N_VAL, len(TE_Y)), replace=False)

adapters = torch.load(ADAPTERS, map_location="cpu")
KEYS = list(adapters[0].keys())
print(f"[adapters] {len(adapters)} x {sum(v.numel() for v in adapters[0].values()):,} params")

model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
BASE = {n: p.detach().clone() for n, p in model.named_parameters() if n in set(KEYS)}
# LoRA B is zero-init at base, so A_s is the whole adapter; deltas are relative to base
DELTA = [{k: (adapters[s][k].to(DEV) - BASE[k]) for k in KEYS} for s in range(N_TASKS)]


def set_alpha(alpha):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in BASE:
                v = BASE[n].clone()
                for s, a in enumerate(alpha):
                    if a != 0.0:
                        v += float(a) * DELTA[s][n]
                p.copy_(v)


@torch.no_grad()
def feats(idx, ds):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def score(alpha, idx_eval, y_eval):
    """NCM over all 200 classes with prototypes built in the merged frame."""
    set_alpha(alpha)
    P = feats(PROTO_IDX, TRAIN)
    yp = TR_Y[PROTO_IDX]
    MU = un(np.stack([un(P[yp == c]).mean(0) for c in range(N_CLS)]))
    Q = un(feats(idx_eval, TEST))
    return float((np.argmax(Q @ MU.T, 1) == y_eval).mean())


def coord_ascent(grid, nonneg, rounds=ROUNDS):
    alpha = np.zeros(N_TASKS); alpha[-1] = 1.0
    best = score(alpha, VAL_IDX, TE_Y[VAL_IDX])
    print(f"    init {best:.4f}", flush=True)
    for r in range(rounds):
        for s in range(N_TASKS):
            cur = alpha[s]
            for v in grid:
                if nonneg and v < 0:
                    continue
                if v == cur:
                    continue
                trial = alpha.copy(); trial[s] = v
                a = score(trial, VAL_IDX, TE_Y[VAL_IDX])
                if a > best:
                    best, alpha = a, trial
            print(f"    round {r} coord {s}: val {best:.4f} alpha={np.round(alpha,2)}",
                  flush=True)
    return alpha, best


# ---- extreme rays of the adapter cone (which adapters span the family?) ----
V = np.stack([torch.cat([DELTA[s][k].flatten() for k in KEYS]).cpu().numpy()
              for s in range(N_TASKS)])
hull = ConicHull(n_rays=5, use_pca=True, pca_dim=min(9, N_TASKS - 1)).fit(V)
extreme = np.unique(np.asarray(hull.extreme_rays_index))
print(f"[cone] extreme-ray adapters (of {N_TASKS}): {extreme.tolist()}")

STRATS = {}
STRATS["last_only"] = np.eye(N_TASKS)[-1]
STRATS["uniform"] = np.ones(N_TASKS) / N_TASKS
a = np.zeros(N_TASKS); a[extreme] = 1.0 / len(extreme)
STRATS["conic_extreme"] = a

print("\n=== coordinate ascent: SIGNED (task arithmetic) ===")
STRATS["signed_search"], _ = coord_ascent(SIGNED_GRID, nonneg=False)
print("\n=== coordinate ascent: CONIC (alpha >= 0) ===")
STRATS["conic_search"], _ = coord_ascent(GRID, nonneg=True)

print("\n=== final evaluation on the FULL test set (all 200 classes) ===")
rows = []
full = np.arange(len(TE_Y))
for name, alpha in STRATS.items():
    acc = score(alpha, full, TE_Y)
    neg = float(np.sum(np.minimum(alpha, 0)))
    rows.append(dict(name=name, acc=acc, alpha=alpha.tolist(), neg_mass=neg))
    print(f"  {name:>15} acc {acc:.4f}  neg-mass {neg:+.2f}  alpha={np.round(alpha,2)}",
          flush=True)

np.save("exp4_results.npy", rows, allow_pickle=True)
print("\n" + "=" * 92)
print("EXP4 — non-negative (conic) vs signed model merging, Split-ImageNet-R, 200-way NCM")
print("=" * 92)
for r in sorted(rows, key=lambda r: -r["acc"]):
    print(f"{r['name']:>15} {r['acc']:>8.4f}   neg-mass {r['neg_mass']:+.2f}")
print("-" * 92)
d = {r["name"]: r["acc"] for r in rows}
print(f"conic_search - signed_search = {d['conic_search']-d['signed_search']:+.4f}   "
      "<- does forbidding subtraction HELP?")
print(f"conic_search - uniform       = {d['conic_search']-d['uniform']:+.4f}")
print(f"conic_extreme ({len(extreme)} adapters) - uniform (10) = "
      f"{d['conic_extreme']-d['uniform']:+.4f}   <- do extreme rays SPAN the family?")
print("\nWIN CONDITION: conic >= signed (no destructive interference) AND conic_extreme")
print("matches uniform with fewer adapters (a storage/compute win from spanning).")
print("NOTE: if signed_search picks ~no negative mass on its own (neg-mass ~ 0), the")
print("constraint is vacuous here and the comparison is uninformative — report that.")
print("=" * 92)
