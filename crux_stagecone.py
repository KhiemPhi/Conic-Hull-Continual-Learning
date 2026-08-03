"""
crux_stagecone.py — per-stage EXACT frames + static conic hulls, with ORACLE stage routing.

Design (no learned transport anywhere):
  * each stage s trains a LoRA adapter; we SNAPSHOT the adapter (~7 MB) at the end of the
    stage.  phi_s is therefore EXACTLY reproducible forever -- "send image 1 back to stage 1"
    is just `run it through adapter_1`, not a regression problem.
  * at stage s we freeze, in that birth frame phi_s:
        - a static ConicHull per stage-s class   (the primitive under test)
        - an NCM prototype per class             (1-point control)
        - K k-means centroids per class          (multi-prototype control, K = n_rays)
    and a memory buffer of M exemplars/class, LABELLED BY STAGE.
  * evaluation, ORACLE routing: a test image of class c is sent back to the stage that owns
    c, featurised with THAT stage's exact backbone, and scored only against that stage's
    classifiers.  This is the upper bound of the per-stage design -- it assumes the routing
    problem away, isolating "how good is per-stage + static cones if routing were perfect?"

  Old features can never be damaged (we kept the old model), so forgetting is structurally
  impossible here; whatever we lose is representation quality or cone/prototype capacity.

Also reported (honest context, no oracle):
    NO-ORACLE raw : score against every stage's classifiers in every frame, argmax over all
    NO-ORACLE cal : same, with per-stage z-scoring calibrated on that stage's memory buffer
  -> the gap oracle routing hides = exactly what a real router would have to recover.

Run:  python -u crux_stagecone.py            (N_RAYS=10 default)
      N_RAYS=20 python -u crux_stagecone.py
"""
import os
import copy
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
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT = 10, 20
EPOCHS, LR, BS = 10, 1e-4, 128
M_REPLAY = 20
N_RAYS = int(os.environ.get("N_RAYS", 10))
RAY_DIVERSITY = os.environ.get("RAY_DIVERSITY", "hybrid")
SCORE_KEYS = ["cosine", "geo_residual", "margin", "angular_margin",
              "full_fidelity", "sparse_support", "max_ray_sim", "blended"]

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
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]
STAGE_OF = np.empty(N_CLS, dtype=int)
for s, cs in enumerate(TASKS):
    STAGE_OF[cs] = s
print(f"[ImageNet-R] train {len(TR_Y)} test {len(TE_Y)} classes {N_CLS} | "
      f"{N_TASKS} stages x {CPT} | n_rays={N_RAYS}")


@torch.no_grad()
def extract(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def un(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ============================ phase 1: train + snapshot ============================
model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
lora_params = list(get_lora_params(model))
lora_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"[snapshot] {sum(p.numel() for p in lora_params):,} adapter params "
      f"({sum(p.numel() for p in lora_params)*4/1e6:.1f} MB/stage)")

stages = []      # per stage: classes, hulls, protos, multiprotos, buffer idx, adapter
for t in range(N_TASKS):
    cls = np.asarray(TASKS[t])
    remap = {int(c): i for i, c in enumerate(cls)}
    tr_idx = np.where(np.isin(TR_Y, cls))[0]
    loader = DataLoader(Subset(TRAIN, tr_idx.tolist()), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True)
    head = nn.Linear(768, CPT).to(DEV)
    opt = torch.optim.AdamW(lora_params + list(head.parameters()), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train(); ok = tot = 0
        for x, lab in loader:
            x = x.to(DEV, non_blocking=True)
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(model(x).float()); loss = ce(lg, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ok += int((lg.argmax(1) == y).sum()); tot += len(y)
        sch.step()

    # ---- freeze this stage's classifiers in its BIRTH frame ----
    adapter = {n: p.detach().clone() for n, p in model.named_parameters()
               if n in set(lora_names)}
    hulls, protos, multis, buf = {}, {}, {}, []
    for c in cls:
        idx_c = np.where(TR_Y == c)[0]
        Fc = un(extract(model, TRAIN, idx_c))
        n = len(Fc)
        protos[int(c)] = un(Fc.mean(0, keepdims=True))[0]
        k = int(min(N_RAYS, max(n // 4, 1)))
        multis[int(c)] = un(KMeans(n_clusters=k, n_init=4, random_state=0)
                            .fit(Fc).cluster_centers_)
        h = ConicHull(n_rays=int(min(N_RAYS, n)), use_pca=True,
                      pca_dim=int(min(64, max(n - 1, 2))),
                      ray_diversity=RAY_DIVERSITY)
        hulls[int(c)] = h.fit(Fc)
        buf.append(idx_c[:M_REPLAY])
    stages.append(dict(cls=cls, adapter=adapter, hulls=hulls, protos=protos,
                       multis=multis, buf=np.concatenate(buf)))
    print(f"  [stage {t}] trainacc {ok/tot:.3f} | {len(cls)} classes | "
          f"buffer {len(stages[-1]['buf'])} imgs")

np.save("crux_stagecone_buffers.npy",
        {s: stages[s]["buf"] for s in range(N_TASKS)}, allow_pickle=True)


# ============================ phase 2: per-frame test features ============================
def load_adapter(a):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in a:
                p.copy_(a[n])


print("\n=== featurising test set in every stage frame (exact backbones) ===")
te_all = np.arange(len(TE_Y))
F_te = {}     # stage -> (n_test, 768) test features in that stage's exact frame
F_buf = {}    # stage -> features of that stage's OWN buffer, in its own frame
for s in range(N_TASKS):
    load_adapter(stages[s]["adapter"])
    F_te[s] = un(extract(model, TEST, te_all))
    F_buf[s] = un(extract(model, TRAIN, stages[s]["buf"]))
    print(f"  stage {s} done")


# ============================ scoring helpers ============================
def stage_scores(s, Z):
    """Score rows of Z (already in stage-s frame) against stage-s classifiers.
    Returns dict name -> (n, |cls_s|) score matrix, columns ordered by stages[s]['cls']."""
    st = stages[s]
    cls = st["cls"]
    out = {}
    out["ncm"] = Z @ np.stack([st["protos"][int(c)] for c in cls]).T
    out["multiproto"] = np.stack(
        [np.max(Z @ st["multis"][int(c)].T, axis=1) for c in cls], axis=1)
    per_key = {k: [] for k in SCORE_KEYS}
    for c in cls:
        sa = st["hulls"][int(c)].score_all(Z)
        for k in SCORE_KEYS:
            per_key[k].append(sa[k])
    for k in SCORE_KEYS:
        out["cone_" + k] = np.stack(per_key[k], axis=1)
    return out


METHODS = ["ncm", "multiproto"] + ["cone_" + k for k in SCORE_KEYS]

# ---- ORACLE routing: every test image goes back to the stage that owns its class ----
print("\n=== scoring (oracle routing) ===")
oracle_hit = {m: 0 for m in METHODS}
oracle_tot = 0
# also cache full score blocks for the no-oracle pass
blocks = {s: None for s in range(N_TASKS)}
for s in range(N_TASKS):
    blocks[s] = stage_scores(s, F_te[s])
    own = np.where(STAGE_OF[TE_Y] == s)[0]          # test images owned by stage s
    cls = stages[s]["cls"]
    for m in METHODS:
        pred = cls[blocks[s][m][own].argmax(1)]
        oracle_hit[m] += int((pred == TE_Y[own]).sum())
    oracle_tot += len(own)
    print(f"  stage {s}: {len(own)} test imgs routed")

# ---- NO-ORACLE: argmax over all stages (raw, and per-stage z-calibrated on the buffer) ----
print("\n=== scoring (no oracle) ===")
allcls = np.concatenate([stages[s]["cls"] for s in range(N_TASKS)])
noracle, noracle_cal = {}, {}
for m in METHODS:
    raw = np.concatenate([blocks[s][m] for s in range(N_TASKS)], axis=1)
    noracle[m] = float((allcls[raw.argmax(1)] == TE_Y).mean())
    calb = []
    for s in range(N_TASKS):
        ref = stage_scores(s, F_buf[s])[m]          # in-stage reference distribution
        mu, sd = float(ref.mean()), float(ref.std() + 1e-8)
        calb.append((blocks[s][m] - mu) / sd)
    cal = np.concatenate(calb, axis=1)
    noracle_cal[m] = float((allcls[cal.argmax(1)] == TE_Y).mean())

# ============================ report ============================
res = {m: dict(oracle=oracle_hit[m] / oracle_tot,
               no_oracle=noracle[m], no_oracle_cal=noracle_cal[m]) for m in METHODS}
np.save("crux_stagecone_results.npy", res, allow_pickle=True)

print("\n" + "=" * 88)
print(f"PER-STAGE EXACT FRAMES + STATIC CONES — Split-ImageNet-R (n_rays={N_RAYS})")
print("=" * 88)
print(f"{'method':>22} {'ORACLE-route':>13} {'no-oracle':>11} {'no-oracle-cal':>14}")
for m in METHODS:
    r = res[m]
    star = {"ncm": "  <- 1-point control",
            "multiproto": "  <- equal-budget multi-prototype control"}.get(m, "")
    print(f"{m:>22} {r['oracle']:>13.4f} {r['no_oracle']:>11.4f} "
          f"{r['no_oracle_cal']:>14.4f}{star}")
print("-" * 88)
best_cone = max((res['cone_' + k]['oracle'], 'cone_' + k) for k in SCORE_KEYS)
print(f"best cone (oracle) = {best_cone[1]} {best_cone[0]:.4f}  |  "
      f"ncm {res['ncm']['oracle']:.4f}  multiproto {res['multiproto']['oracle']:.4f}")
print(f"cone - ncm        = {best_cone[0]-res['ncm']['oracle']:+.4f}")
print(f"cone - multiproto = {best_cone[0]-res['multiproto']['oracle']:+.4f}   "
      "<- the honest test (cones vs equal-budget multi-prototype)")
print("\nreference bars (crux_method.py): A+ first-session RanPAC 0.7858 | "
      "joint ceiling 0.8355")
print("NOTE: best-of-8 cone score keys is selected on TEST -> optimistic for the cone.")
print("=" * 88)
