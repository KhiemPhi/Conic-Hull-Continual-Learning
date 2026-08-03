"""
exp7_covcone_diagnostic.py — WHY does the covariance-cone penalty lose A-last?

cone_only (LAM1=1.0) gets A-avg 0.8232 (near A+ 0.8313) but A-last 0.7065 (far below A+ 0.7858)
-- it holds early and degrades late. Plasticity is NOT the problem: train-acc per task matched
the unpenalised baseline (0.80-0.89), so the backbone is still learning new tasks. The failure is
UNDER-PROTECTION of old features. Three candidate causes, which different fixes address:

  H1 RANK TRUNCATION   top-64 of 768 discards too much variance
                       -> fix: low-rank + diagonal residual, or adaptive m
  H2 SUBSPACE EVICTION the bank is a RUNNING SUM re-truncated to top-64 each task; the top-64 of
                       a 10-task sum keeps SHARED directions and evicts task-specific ones
                       -> fix: incremental subspace merging (orthogonalise [U_1,U_2], compress)
  H3 STALENESS         Sigma was measured in an old frame and is wrong in the current frame
                       -> fix: covariance transport  Sigma^(t+1) = A Sigma A^T

Measurements (no retraining; reuses crux_routing_adapters.pt):
  H1  eigenspectrum of each layer's activation covariance -> cumulative energy at m in
      {16,32,64,128,256}, and the m needed for 95/98/99%.
  H2  simulate the exp6 running-sum bank across all 10 tasks, then ask how much of TASK 0's
      activation energy the final bank still protects; compare against the subspace-MERGING
      alternative on the same data.
  H3  recompute task 0's covariance in the FINAL frame (adapter 9) and compare its top-64
      subspace to the one measured in frame 0.

KEY METRIC: energy_protected(Sigma, U) = Tr(U^T Sigma U) / Tr(Sigma) in [0,1]
  -- the fraction of a task's activation energy that a given subspace actually shields.
  This is the quantity the penalty controls, so it is the right thing to compare.

CAVEAT: the cached adapters were trained WITHOUT the penalty, so H3 measures staleness under
UNCONSTRAINED drift = an UPPER BOUND on the staleness the real method suffers. If it is small
here, H3 is ruled out a fortiori. H1/H2 are essentially penalty-independent (H1 is a property of
ViT activation spectra; H2 is deterministic arithmetic on the covariances).

Run:  python -u exp7_covcone_diagnostic.py
"""
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, _ALL_LORA_TYPES

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
N_TASKS, CPT = 10, 20
M_EIG = int(os.environ.get("M_EIG", 64))
COV_BATCHES = int(os.environ.get("COV_BATCHES", 12))
ADAPTERS = "crux_routing_adapters.pt"
MS = [16, 32, 64, 128, 256]

T0 = time.time()


def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


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
TRAIN = HFWrap(_ds, _p[:_n], _lab[_p[:_n]]); TR_Y = _lab[_p[:_n]]
N_CLS = len(_cl)
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]
TASK_IDX = [np.where(np.isin(TR_Y, TASKS[s]))[0] for s in range(N_TASKS)]
log(f"[ImageNet-R] {len(TR_Y)} train | {N_TASKS}x{CPT} | m={M_EIG}")

model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                      lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
freeze_non_lora(model)
MODS = [(n, m) for n, m in model.named_modules() if isinstance(m, _ALL_LORA_TYPES)]
adapters = torch.load(ADAPTERS, map_location="cpu")
log(f"{len(MODS)} LoRA layers, {len(adapters)} cached adapters")


def load_adapter(s):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in adapters[s]:
                p.copy_(adapters[s][n].to(DEV))


@torch.no_grad()
def act_cov(idx, n_batches=COV_BATCHES):
    """Sigma_l = E[x_l x_l^T] per LoRA-layer input, at the currently loaded weights."""
    covs = {n: torch.zeros(m.in_features, m.in_features, device=DEV, dtype=torch.float64)
            for n, m in MODS}
    cnt = {n: 0 for n, _ in MODS}
    hs = []

    def mk(name):
        def hook(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            covs[name] += x.T @ x
            cnt[name] += x.shape[0]
        return hook

    for n, m in MODS:
        hs.append(m.register_forward_pre_hook(mk(n)))
    model.eval()
    loader = DataLoader(Subset(TRAIN, idx.tolist()), batch_size=64, shuffle=True, num_workers=8)
    for bi, (x, _) in enumerate(loader):
        if bi >= n_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x.to(DEV, non_blocking=True))
    for h in hs:
        h.remove()
    return {n: covs[n] / max(cnt[n], 1) for n, _ in MODS}


def top_basis(cov, m):
    """Orthonormal basis of the top-m eigenspace (no Lam^{1/2} weighting)."""
    ev, U = torch.linalg.eigh(cov)
    return U[:, -m:]


def energy_protected(cov, U):
    """Tr(U^T Sigma U)/Tr(Sigma): fraction of this covariance's energy inside subspace U."""
    return float(torch.trace(U.T.double() @ cov @ U.double()) / (torch.trace(cov) + 1e-30))


def overlap(Ua, Ub):
    """Mean squared cosine of principal angles, in [0,1]. 1 = identical subspaces."""
    return float((Ua.T.double() @ Ub.double()).pow(2).sum() / Ua.shape[1])


# ======================= collect per-task covariances (own frame) =======================
log("=== measuring per-task activation covariances (each in its own frame) ===")
COV = {}
for t in range(N_TASKS):
    load_adapter(t)
    COV[t] = act_cov(TASK_IDX[t])
    log(f"  task {t} covariance done")

LAYERS = [n for n, _ in MODS]

# ============================== H1: rank truncation ==============================
log("=== H1: how much energy does top-m capture? ===")
h1 = {m: [] for m in MS}
need = {q: [] for q in (0.95, 0.98, 0.99)}
for t in range(N_TASKS):
    for n in LAYERS:
        ev = torch.linalg.eigvalsh(COV[t][n]).flip(0).clamp(min=0)
        c = torch.cumsum(ev, 0) / (ev.sum() + 1e-30)
        for m in MS:
            h1[m].append(float(c[min(m, len(c)) - 1]))
        for q in need:
            need[q].append(int(torch.searchsorted(c, q).item()) + 1)

print("\n" + "=" * 88)
print("H1 — RANK TRUNCATION: cumulative activation energy captured by the top-m eigenspace")
print("=" * 88)
for m in MS:
    v = np.array(h1[m])
    print(f"  m={m:>4}: energy {v.mean():.4f}  (min over layers/tasks {v.min():.4f})"
          + ("   <- exp6 setting" if m == M_EIG else ""))
for q in sorted(need):
    v = np.array(need[q])
    print(f"  dims needed for {q:.0%} energy: mean {v.mean():6.1f}  median {np.median(v):6.1f}  "
          f"max {v.max():4d}")
e64 = np.array(h1[M_EIG]).mean()
print(f"\n  VERDICT: top-{M_EIG} leaves {1-e64:.1%} of activation energy UNPROTECTED.")
print(f"  -> if this is large, the diagonal-residual fix (low-rank + D) has real room;")
print(f"     if it is <2%, H1 is not the bottleneck and #1/#2 will not move A-last.")

# ============================== H2: subspace eviction ==============================
log("=== H2: does the running-sum bank evict task-0 directions? ===")
# reproduce exp6's bank exactly: running sum of covariances, re-truncated to top-m each task
bank_sum, run_prot, merge_prot = {}, [], []
S_run = {}
for t in range(N_TASKS):
    for n in LAYERS:
        prev = S_run.get(n)
        acc = COV[t][n] if prev is None else COV[t][n] + (prev.double() @ prev.double().T)
        ev, U = torch.linalg.eigh(acc)
        ev = ev[-M_EIG:].clamp(min=0); U = U[:, -M_EIG:]
        S_run[n] = (U * ev.sqrt().unsqueeze(0)).float()
    if t in (0, 4, N_TASKS - 1):
        p = np.mean([energy_protected(COV[0][n], torch.linalg.qr(S_run[n].double())[0])
                     for n in LAYERS])
        run_prot.append((t, p))
        log(f"  running-sum bank after task {t}: protects {p:.4f} of TASK-0 energy")

# alternative: incremental subspace MERGING (orthogonalise the union, then compress)
S_mrg = {}
for t in range(N_TASKS):
    for n in LAYERS:
        Ut = top_basis(COV[t][n], M_EIG)
        if n not in S_mrg:
            S_mrg[n] = Ut
        else:
            cat = torch.cat([S_mrg[n], Ut], 1)
            Q, _ = torch.linalg.qr(cat)                     # union, orthonormalised
            # compress back to the M_EIG directions carrying most ACCUMULATED energy
            wsum = sum(COV[k][n] for k in range(t + 1))
            proj = Q.T.double() @ wsum @ Q.double()
            ev, V = torch.linalg.eigh(proj)
            S_mrg[n] = (Q.double() @ V[:, -M_EIG:]).float()
    if t in (0, 4, N_TASKS - 1):
        p = np.mean([energy_protected(COV[0][n], S_mrg[n].double()) for n in LAYERS])
        merge_prot.append((t, p))
        log(f"  merged-subspace bank after task {t}: protects {p:.4f} of TASK-0 energy")

print("\n" + "=" * 88)
print("H2 — SUBSPACE EVICTION: fraction of TASK-0 activation energy still protected")
print("=" * 88)
print(f"{'after task':>11} {'running sum (exp6)':>20} {'subspace merging':>18} {'delta':>8}")
for (t, a), (_, b) in zip(run_prot, merge_prot):
    print(f"{t:>11} {a:>20.4f} {b:>18.4f} {b-a:>+8.4f}")
d0 = run_prot[0][1] - run_prot[-1][1]
print(f"\n  running-sum bank lost {d0:+.4f} of task-0 protection over 10 tasks")
print("  -> if this decay is large AND merging holds up, H2 is the bottleneck and")
print("     incremental subspace merging (#4) is the fix.")

# ============================== H3: staleness ==============================
log("=== H3: is task-0's covariance stale in the final frame? ===")
load_adapter(N_TASKS - 1)
COV0_final = act_cov(TASK_IDX[0])
ov, en_stale, en_fresh = [], [], []
for n in LAYERS:
    U0 = top_basis(COV[0][n], M_EIG)                 # measured in frame 0 (what we store)
    U9 = top_basis(COV0_final[n], M_EIG)             # task-0 stats in the CURRENT frame
    ov.append(overlap(U0, U9))
    en_stale.append(energy_protected(COV0_final[n], U0.double()))   # stored basis, new frame
    en_fresh.append(energy_protected(COV0_final[n], U9.double()))   # oracle basis, new frame
ov, en_stale, en_fresh = np.array(ov), np.array(en_stale), np.array(en_fresh)

print("\n" + "=" * 88)
print("H3 — STALENESS: task-0 covariance measured in frame 0 vs in the final frame")
print("=" * 88)
print(f"  subspace overlap (top-{M_EIG}) frame0 vs frame9 : {ov.mean():.4f} "
      f"(min layer {ov.min():.4f})")
print(f"  task-0 energy protected by the STORED  basis    : {en_stale.mean():.4f}")
print(f"  task-0 energy protected by a FRESH     basis    : {en_fresh.mean():.4f}")
print(f"  staleness cost                                  : {en_fresh.mean()-en_stale.mean():+.4f}")
print("  NOTE: adapters here were trained WITHOUT the penalty, so this is an UPPER BOUND")
print("        on the staleness the real method suffers.")
print("  -> if the cost is small, H3 is ruled out and covariance transport is unnecessary.")

np.save("exp7_diagnostic.npy",
        dict(h1={m: float(np.mean(h1[m])) for m in MS},
             need={q: float(np.mean(need[q])) for q in need},
             run_prot=run_prot, merge_prot=merge_prot,
             h3_overlap=float(ov.mean()), h3_stale=float(en_stale.mean()),
             h3_fresh=float(en_fresh.mean())), allow_pickle=True)

print("\n" + "=" * 88)
print("SUMMARY — which fix to build")
print("=" * 88)
unprot = 1 - e64
evict = d0
stale = en_fresh.mean() - en_stale.mean()
print(f"  H1 unprotected energy (top-{M_EIG})     : {unprot:.4f}   -> low-rank + diagonal (#1)")
print(f"  H2 task-0 protection lost over tasks  : {evict:.4f}   -> subspace merging (#4)")
print(f"  H3 staleness cost (upper bound)       : {stale:.4f}   -> covariance transport")
rank = sorted([(unprot, "H1 low-rank+diagonal / adaptive m"),
               (evict, "H2 incremental subspace merging"),
               (stale, "H3 covariance transport")], reverse=True)
print("\n  ranked by measured magnitude:")
for v, name in rank:
    print(f"    {v:.4f}  {name}")
print("\n  Build the largest one first. If all three are <0.02, none of these is the")
print("  bottleneck and the A-last gap is something else (e.g. class-count growth in the")
print("  accumulated head, which no covariance fix addresses).")
print("=" * 88)
