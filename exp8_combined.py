"""
exp8_combined.py — ZERO-IMAGE combined objective: covariance cone + Mahalanobis calibration
                   + patch-token self-distillation, with virtual-feature statistics.

    L = CE(task t)
      + lam1 * sum_l || dW_l U_l Lam_l^{1/2} ||_F^2          COVARIANCE CONE   (ours; control passed)
      + lam2 * L_maha                                         MAHALANOBIS CALIB (replay-free)
      + lam3 * L_distill                                      PATCH-TOKEN DISTILL

Why this combination:
  * our Gram-relational loss gave +8.3 pts but needed 20 stored IMAGES/class. The Mahalanobis
    calibration is the same family of constraint (preserve relational geometry under drift) but
    evaluated on CURRENT-task data whitened by the OLD network's per-class covariance -> ZERO
    stored images. If it matches Gram's gain, it is a strict upgrade.
  * the covariance cone is anchored to PAST activation directions (well-targeted, needs a 4.7 MB
    eigenbasis); the Mahalanobis term is anchored to CURRENT data (cheap, indirect). Complementary.
  * patch-token distillation is untested by us and orthogonal to both.

Everything runs in RECOMPUTE mode with Gaussian virtual features, because `accum` mode is capped
at the A+ bar by construction (perfect protection -> frozen -> A+). Recompute is the only
configuration whose ceiling is above the bar.

Storage: NO images. Only mu_c (768 floats/class), one shared covariance, and the per-layer
eigenbasis (4.7 MB). Sigma_c^{-1/2} for the Mahalanobis term is built per task and discarded.

Bars (Split-ImageNet-R, identical protocol):
  A_frozen 0.7272 | A+ first-session RanPAC 0.7858 (BAR) | joint ceiling 0.8355
  seqCE_recompute+synth 0.6557 (no relational loss) | seqGram_recompute+synth 0.7390
  (prev best, WITH 20 images/class)

Run:  python -u exp8_combined.py
      VARIANTS=maha_only,cov_maha python -u exp8_combined.py
      LAM2=1.0 LAM3=0.1 python -u exp8_combined.py
"""
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params, _ALL_LORA_TYPES

SEED = int(os.environ.get("SEED", 0))   # class order + init; sweep to test margin significance
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
# Standard PTM-CIL backbone is ViT-B/16 pretrained on ImageNet-21k ONLY (what RanPAC / L2P /
# DualPrompt / EASE report on). `augreg2_in21k_ft_in1k` was a stand-in because the IN21k CDN is
# blocked for the agent; it is IN1k-FINETUNED, and ImageNet-R's 200 classes ARE IN1k classes, so
# it inflates the frozen baseline and is NOT comparable to published numbers.
#   download once:  HF_HUB_DISABLE_XET=1 http_proxy=http://fwdproxy:8080 \
#     https_proxy=http://fwdproxy:8080 python -c \
#     "import timm; timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)"
#   alternatives: vit_base_patch16_224.orig_in21k
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
N_TASKS, CPT = 10, 20
EPOCHS = int(os.environ.get("EPOCHS", 10))   # 40 matches the best joint recipe
LR = float(os.environ.get("LR", 1e-4))
BS = 128
M_EIG = int(os.environ.get("M_EIG", 64))
LAM1 = float(os.environ.get("LAM1", 1.0))       # covariance cone  (calibrated in exp6)
LAM2 = float(os.environ.get("LAM2", 1.0))       # Mahalanobis calibration
LAM3 = float(os.environ.get("LAM3", 0.1))       # patch-token distillation
MAHA_SHRINK = float(os.environ.get("MAHA_SHRINK", 0.1))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))
M_RP = 10000
LAMBDAS = [1e2, 1e3, 1e4]
SYNTH_PER_CLASS = 120
COV_BATCHES = 12

T0 = time.time()


def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# --- RECIPE (matched to the crux_headroom sweep) -------------------------------------
# The sweep raised the joint ceiling 0.8332 -> 0.8510 with AUG=1 + 40 epochs. Augmentation
# alone at 10 epochs was worth +0.0008, i.e. nothing: the longer schedule is what pays.
#   AUG=0 EPOCHS=10  -> the original recipe (all results so far)
#   AUG=1 EPOCHS=40  -> matches the best joint recipe
AUG = int(os.environ.get("AUG", 0))
# ORACLE_STATS=1 adds a per-task diagnostic that rebuilds the head from ALL real seen data in
# the current frame. Cheating by design -- it separates "our statistics are approximate" from
# "our features are worse", which need opposite fixes. ~5 min extra per run.
ORACLE_STATS = int(os.environ.get("ORACLE_STATS", 0))
results_oracle = {}

# ============================ NEW PENALTY AXES =======================================
# Three orthogonal extensions of the covariance-cone penalty, ablatable independently.
#
# PEN_MODE  what is protected
#   sigma   Tr(dW Sigma dW^T) = E||dW x||^2         -- current: protects LAYER outputs
#   kfac    ||T^T dW S||_F^2 = E||J dW x||^2        -- protects FUNCTION outputs   [idea 1/4]
#           Adds the K-FAC "G" factor G=E[g g^T] (g = dL/d(layer output)) to the "A" factor
#           Sigma we already use; A (x) G IS the Fisher. Our penalty is only half of it.
#           Two-sided and still linear in rank r:  T^T B (A S), all factors small.
#           Layer-output preservation is SUFFICIENT but not NECESSARY -- a change later
#           layers absorb costs us nothing, and sigma-mode forbids it anyway.
#
# BANK_MODE how per-task covariances are combined
#   sum       Sigma_1 + ... + Sigma_t  -- current, but those were measured in DIFFERENT
#             FRAMES, so the sum is incoherent if the backbone rotated between tasks
#   transport Procrustes-align the bank into the current frame before adding [idea 3/7]
#
# COV_MODE  which covariance
#   total    E[xx^T]                    -- current; dominated by whichever term is bigger
#   between  class-mean scatter         -- where DISCRIMINATIVE information lives
#   within   E[xx^T] - between          -- intra-class shape                     [idea 9]
# --- PROTOTYPE-FREE attacks on the ACCUMULATION gap ----------------------------------
# The gap to joint is 92% "sequential features are worse", and the mechanism is that
# task-wise CE never asks the backbone to separate class 5 from class 150. These two add
# cross-task pressure WITHOUT storing any per-class information.
#   LAM4  anti-collapse: keep the representation full-volume so later classes still have
#         dimensions to live in. Pure current-batch statistic, zero storage.
#   MAHA_TEACHER=base : distil relational geometry from the FROZEN PRETRAINED phi_0 instead
#         of phi^{t-1}. phi_0 is free (always reproducible) and, unlike phi^{t-1}, its
#         structure reflects ALL 200 classes via pretraining -- global information no single
#         task provides. Relational form permits rotation, so it does not block adaptation.
LAM4 = float(os.environ.get("LAM4", 0.0))
AC_MODE = os.environ.get("AC_MODE", "logdet")      # logdet | vicreg (see anticollapse_loss)
MAHA_TEACHER = os.environ.get("MAHA_TEACHER", "old")   # old | base

PEN_MODE = os.environ.get("PEN_MODE", "sigma")     # sigma | kfac
BANK_MODE = os.environ.get("BANK_MODE", "sum")     # sum | transport
COV_MODE = os.environ.get("COV_MODE", "total")     # total | between | within
M_OUT = int(os.environ.get("M_OUT", 64))           # eigenvectors kept for the G factor
_cfg = resolve_model_data_config(timm.create_model(MODEL, pretrained=False, num_classes=0))
# Feature extraction is ALWAYS deterministic — prototypes / Gram / RanPAC statistics computed
# on random crops would be noise.
TF_EVAL = create_transform(**_cfg, is_training=False)
TF_TRAIN = (create_transform(**_cfg, is_training=True, auto_augment="rand-m9-mstd0.5",
                             re_prob=0.25, scale=(0.7, 1.0), hflip=0.5) if AUG else TF_EVAL)
TF = TF_EVAL


class HFWrap(Dataset):
    """Returns (image, label, row-index). The index lets the Mahalanobis term look up cached
    old-network features -- valid ONLY when the transform is deterministic (see AUG below)."""
    def __init__(self, ds, idx, labels, tf=None):
        self.ds, self.idx, self.labels = ds, np.asarray(idx), np.asarray(labels)
        self.tf = tf or TF_EVAL

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds[int(self.idx[i])]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(self.labels[i]), i


from datasets import load_dataset
_ds = load_dataset("axiong/imagenet-r", cache_dir="./data/hf")["test"]
_w = np.array(_ds["wnid"]); _cl = np.array(sorted(set(_w.tolist())))
_lab = np.searchsorted(_cl, _w)
_p = np.random.default_rng(1993).permutation(len(_lab))
_n = int(0.8 * len(_lab))
TRAIN_AUG = HFWrap(_ds, _p[:_n], _lab[_p[:_n]], TF_TRAIN)   # training loop only
TRAIN = HFWrap(_ds, _p[:_n], _lab[_p[:_n]], TF_EVAL);  TR_Y = _lab[_p[:_n]]
TEST  = HFWrap(_ds, _p[_n:], _lab[_p[_n:]], TF_EVAL);  TE_Y = _lab[_p[_n:]]
N_CLS = len(_cl)
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]
log(f"[ImageNet-R] train {len(TR_Y)} test {len(TE_Y)} | {N_TASKS}x{CPT} | "
    f"lam1={LAM1} lam2={LAM2} lam3={LAM3}")


def un(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


# ============================== RanPAC head ==============================
P_RP = torch.randn(768, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)


def _H(Z, bs=4096):
    for i in range(0, len(Z), bs):
        yield i, torch.relu(torch.tensor(un(Z[i:i + bs]), device=DEV,
                                         dtype=torch.float32) @ P_RP)


def build_GC(Z, y):
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    for i, h in _H(Z):
        h = h.double()
        Y = torch.zeros(h.shape[0], N_CLS, device=DEV, dtype=torch.float64)
        Y[torch.arange(h.shape[0]), torch.tensor(y[i:i + h.shape[0]], device=DEV)] = 1.0
        G += h.T @ h; C += h.T @ Y
    return G, C


def solve_eval(G, C, Zval, yval, Zte, yte, seen):
    seen_t = torch.tensor(np.asarray(seen), device=DEV)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def acc(W, Z, y):
        pr = []
        for _, h in _H(Z):
            pr.append(seen_t[(h.double() @ W)[:, seen_t].argmax(1)].cpu().numpy())
        return float((np.concatenate(pr) == y).mean())

    best, bestW = -1.0, None
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, C)
        a = acc(W, Zval, yval)
        if a > best:
            best, bestW = a, W
    return acc(bestW, Zte, yte)


# ==================== TERM 1: covariance cone (from exp6) ====================
def lora_mods(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _ALL_LORA_TYPES)]


def act_cov(model, idx, mods, n_batches=COV_BATCHES, head=None, y_task=None, remap=None):
    """Per-LoRA-layer statistics on this task's data.

    Returns (A, G, B) where
      A[n] = E[x x^T]                      input covariance  (the "A" factor of K-FAC)
      G[n] = E[g g^T], g = dL/d(out)       gradient covariance ("G" factor)  -- None unless
                                           PEN_MODE=kfac (needs a backward pass)
      B[n] = class-mean scatter of x       for COV_MODE=between/within
    """
    need_G = (PEN_MODE == "kfac") and head is not None
    A = {n: torch.zeros(m.in_features, m.in_features, device=DEV, dtype=torch.float64)
         for n, m in mods}
    G = {n: torch.zeros(m.out_features, m.out_features, device=DEV, dtype=torch.float64)
         for n, m in mods} if need_G else None
    cnt = {n: 0 for n, _ in mods}
    # per-class running sums of x, for the between/within decomposition
    csum = {n: {} for n, _ in mods} if COV_MODE in ("between", "within") else None
    cur_lab = {}          # labels of the batch currently in flight (set in the loop)
    hs = []

    def mk_fwd(name):
        def hook(mod, inp):
            x = inp[0].detach()
            xf = x.reshape(-1, x.shape[-1]).double()
            A[name] += xf.T @ xf; cnt[name] += xf.shape[0]
            if csum is not None and "y" in cur_lab:
                # average over tokens -> one vector per image, then bucket by class
                xi = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1).double()
                for c in cur_lab["y"].unique():
                    m_ = cur_lab["y"] == c
                    s, n_ = csum[name].setdefault(int(c), [torch.zeros_like(xi[0]), 0])
                    csum[name][int(c)] = [s + xi[m_].sum(0), n_ + int(m_.sum())]
        return hook

    def mk_bwd(name):
        def hook(mod, gin, gout):
            g = gout[0].detach().reshape(-1, gout[0].shape[-1]).double()
            G[name] += g.T @ g
        return hook

    for n, m in mods:
        hs.append(m.register_forward_pre_hook(mk_fwd(n)))
        if need_G:
            hs.append(m.register_full_backward_hook(mk_bwd(n)))

    loader = DataLoader(Subset(TRAIN, idx.tolist()), batch_size=64, shuffle=True, num_workers=8)
    ce = nn.CrossEntropyLoss()
    model.eval() if not need_G else model.train()
    for bi, (x, lab, _) in enumerate(loader):
        if bi >= n_batches:
            break
        x = x.to(DEV, non_blocking=True)
        cur_lab["y"] = lab.to(DEV)
        if need_G:
            y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
            model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ce(head(model(x).float()), y).backward()
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
    for h in hs:
        h.remove()
    model.zero_grad(set_to_none=True)

    Aout = {n: A[n] / max(cnt[n], 1) for n, _ in mods}
    Gout = {n: G[n] / max(cnt[n], 1) for n, _ in mods} if need_G else None
    Bout = None
    if csum is not None:
        Bout = {}
        for n, _ in mods:
            mus = [(s / max(k, 1), k) for s, k in csum[n].values()]
            tot = sum(k for _, k in mus)
            gmu = sum(mu * k for mu, k in mus) / max(tot, 1)
            Bout[n] = sum((k / max(tot, 1)) * torch.outer(mu - gmu, mu - gmu) for mu, k in mus)
    return Aout, Gout, Bout


def pick_cov(A, B):
    """Which covariance the penalty should use, per COV_MODE."""
    if COV_MODE == "between":
        return B
    if COV_MODE == "within":
        return {n: (A[n] - B[n]).clamp(min=0) if False else A[n] - B[n] for n in A}
    return A


@torch.no_grad()
def frame_transport(old_model, new_model, idx, mods, n_batches=6):
    """Procrustes map per layer from the PREVIOUS frame to the CURRENT one, fitted on this
    task's images pushed through BOTH backbones. Used to align an accumulated covariance
    bank before adding to it: Sigma_new_frame = R^T Sigma_old_frame R.
    Fixes a real incoherence -- summing covariances measured in different frames."""
    M = {n: torch.zeros(m.in_features, m.in_features, device=DEV, dtype=torch.float64)
         for n, m in mods}
    stash, hs = {}, []

    def mk(name, which):
        def hook(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            if which == "old":
                stash[name] = x
            else:
                M[name] += stash[name].T @ x        # cross-covariance old^T new
        return hook

    old_mods = [(n, m) for n, m in old_model.named_modules() if isinstance(m, _ALL_LORA_TYPES)]
    for (n, mo), (_, mn) in zip(old_mods, mods):
        hs.append(mo.register_forward_pre_hook(mk(n, "old")))
        hs.append(mn.register_forward_pre_hook(mk(n, "new")))
    old_model.eval(); new_model.eval()
    loader = DataLoader(Subset(TRAIN, idx.tolist()), batch_size=64, shuffle=False, num_workers=8)
    for bi, (x, _, _) in enumerate(loader):
        if bi >= n_batches:
            break
        x = x.to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            old_model(x); new_model(x)          # order matters: old stashes, new consumes
    for h in hs:
        h.remove()
    R = {}
    for n, _ in mods:
        U, _, Vt = torch.linalg.svd(M[n])
        R[n] = (U @ Vt)                          # orthogonal old->new
    return R


def eig_basis(cov, m=M_EIG):
    if not torch.isfinite(cov).all():
        raise FloatingPointError("non-finite activation covariance — LAM1 too large")
    ev, U = torch.linalg.eigh(cov)
    ev = torch.clamp(ev[-m:], min=0.0); U = U[:, -m:]
    return (U * ev.sqrt().unsqueeze(0)).float()


def cone_penalty(mods, S_bank, AB0, T_bank=None):
    """PEN_MODE=sigma : ||dW S||_F^2          = E||dW x||^2       (protect layer outputs)
       PEN_MODE=kfac  : ||T^T dW S||_F^2      = E||J dW x||^2     (protect the FUNCTION)
    Both stay linear in the LoRA rank: everything factors through (A S) and (T^T B)."""
    tot = 0.0
    for n, mod in mods:
        S = S_bank.get(n)
        if S is None or n not in AB0:
            continue
        A0, B0 = AB0[n]
        AS, A0S = mod.lora_A @ S, A0 @ S                 # (r, m_in)
        if PEN_MODE == "kfac" and T_bank is not None and n in T_bank:
            T = T_bank[n]                                # (out, m_out)
            cur = (T.T @ mod.lora_B) @ AS                # (m_out, m_in)
            ref = (T.T @ B0) @ A0S
        else:
            cur = mod.lora_B @ AS                        # (out, m_in)
            ref = B0 @ A0S
        tot = tot + (mod.scaling ** 2) * ((cur - ref) ** 2).sum()
    return tot


# ==================== TERM 2: Mahalanobis covariance calibration ====================
@torch.no_grad()
def build_maha(old_model, idx, y_task, classes):
    """Per current-task class c: mu_c^{t-1}, and P_c = Sigma_c^{-1/2} from the OLD network.
    Shrinkage is REQUIRED: with ~120 samples in 768-d, Sigma_c is rank-deficient and its
    inverse is meaningless without it."""
    old_model.eval()
    loader = DataLoader(Subset(TRAIN, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    feats = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _, _ in loader:
            feats.append(old_model(x.to(DEV, non_blocking=True)).float())
    F_old = torch.cat(feats)                                   # (N, 768) in the OLD frame
    F_old = Fn.normalize(F_old, dim=1)
    P = {}
    for c in classes:
        Xc = F_old[torch.tensor(y_task == c, device=DEV)]
        if len(Xc) < 4:
            continue
        mu = Xc.mean(0, keepdim=True)
        d = Xc - mu
        S = (d.T @ d) / max(len(Xc) - 1, 1)
        S = S.double()
        S = (1 - MAHA_SHRINK) * S + MAHA_SHRINK * (torch.trace(S) / S.shape[0]) * \
            torch.eye(S.shape[0], device=DEV, dtype=torch.float64)
        ev, U = torch.linalg.eigh(S)
        ev = ev.clamp(min=1e-8)
        P[int(c)] = (U @ torch.diag(ev.rsqrt()) @ U.T).float()   # Sigma^{-1/2}
    return F_old, P


def maha_loss(feat_new, feat_old, labels, P):
    """|| d_M(new_i,new_j) - d_M(old_i,old_j) || summed over within-class pairs of the batch.
    d_M(x,y,Sigma) = || Sigma^{-1/2}(x-y) ||_2, so each pair is one matvec."""
    tot, npair = feat_new.new_zeros(()), 0
    for c in labels.unique():
        ci = int(c)
        if ci not in P:
            continue
        m = labels == c
        if int(m.sum()) < 2:
            continue
        A, B = feat_new[m], feat_old[m]
        dA = (A.unsqueeze(1) - A.unsqueeze(0)).reshape(-1, A.shape[1])   # (n^2, d)
        dB = (B.unsqueeze(1) - B.unsqueeze(0)).reshape(-1, B.shape[1])
        Pc = P[ci]
        dm_new = (dA @ Pc.T).norm(dim=1)
        dm_old = (dB @ Pc.T).norm(dim=1)
        tot = tot + (dm_new - dm_old).abs().sum()
        npair += dA.shape[0]
    return tot / max(npair, 1)


# ============ TERM 4: ANTI-COLLAPSE (prototype-free cross-task capacity) ==============
# Task-wise CE only ever separates 20 classes, so it is free to COLLAPSE the representation
# onto the ~20 directions that task needs -- destroying dimensions other tasks' classes will
# require. This term keeps the feature covariance full-volume, with NO stored class
# information: it is computed on the current batch alone.
#   logdet : maximise log det(Sigma_batch + eps I)   -- volume / anti-collapse
#   vicreg : hinge on per-dimension std (cheaper, more stable than logdet in low precision)
def anticollapse_loss(feat, mode="logdet", eps=1e-4, gamma=1.0):
    """VERIFIED: the VICReg *variance* term alone is BLIND to rank collapse -- a rank-20
    projection into 768 dims still gives every dimension nonzero variance (measured:
    0.6399 full-rank vs 0.6440 collapsed). logdet sees it clearly (7.74 -> 8.93), and the
    VICReg *covariance* (decorrelation) term is what supplies the missing signal. Hence
    logdet is the default and `vicreg` includes both terms."""
    Z = feat - feat.mean(0, keepdim=True)
    d = Z.shape[1]
    if mode == "logdet":
        S = (Z.T @ Z) / max(len(Z) - 1, 1) + eps * torch.eye(d, device=Z.device)
        return -torch.linalg.slogdet(S.double())[1].float() / d      # minimise -logdet
    var = Fn.relu(gamma - torch.sqrt(Z.var(0) + eps)).mean()          # variance term
    C = (Z.T @ Z) / max(len(Z) - 1, 1)
    cov = (C - torch.diag(torch.diagonal(C))).pow(2).sum() / d        # decorrelation term
    return var + cov


# ==================== TERM 3: patch-token self-distillation ====================
def distill_loss(tok_new, tok_old, n_prefix=1):
    """(1/L) sum_j (1 - sim(p_j, cls)) * ||p_j^t - p_j^{t-1}||^2.
    Patch tokens that contribute LITTLE to the current class token (low angular similarity)
    are pulled toward their previous-network values -> preserves unused capacity.
    The similarity weight is detached, per the formulation."""
    cls = tok_new[:, 0]                                   # (B, D)
    p_new = tok_new[:, n_prefix:]                         # (B, L, D)
    p_old = tok_old[:, n_prefix:]
    with torch.no_grad():
        w = 1.0 - Fn.cosine_similarity(p_new, cls.unsqueeze(1), dim=-1)   # (B, L)
    return (w * (p_new - p_old).pow(2).sum(-1)).mean()


# ==================== virtual features (statistics, not a loss) ====================
def virtual_feats(mu, cov_shared, classes, n_per=SYNTH_PER_CLASS, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov_shared + 1e-4 * np.eye(cov_shared.shape[0]))
    Z = [mu[c] + rng.standard_normal((n_per, len(mu[c]))) @ L.T for c in classes]
    Y = [np.full(n_per, c) for c in classes]
    return np.concatenate(Z), np.concatenate(Y)


@torch.no_grad()
def extract(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


# ============================== main loop ==============================
def run(tag, lam1, lam2, lam3, stats="recompute", freeze_after=None):
    """freeze_after: None = adapt every task | 0 = adapt task 0 then FREEZE (this is A+, the
    bar) | -1 = never adapt (A_frozen, the floor). Measuring the bars here means they use the
    SAME backbone / protocol / head as the methods -- essential when MODEL changes."""
    log(f"=== {tag}  (cone={lam1}, maha={lam2}, distill={lam3}, stats={stats}, "
        f"freeze_after={freeze_after}) ===")
    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))
    mods = lora_mods(model)
    S_bank, mu_bank, cov_bank, accs = {}, {}, None, []
    T_bank = {}            # K-FAC G-factor bases (PEN_MODE=kfac)
    RAW_bank = {}          # accumulated raw covariance, kept so it can be transported
    oracle_accs = []
    # persistent statistics for stats="accum": RanPAC-style exact additivity. Old classes keep
    # the statistics recorded in THEIR birth frame, which stay valid only if the features do not
    # drift -- exactly what the loss stack is supposed to guarantee. No virtual features here
    # (they would double-count classes already in G/C).
    Gacc = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    Cacc = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    vZ, vy = [], []

    for t in range(N_TASKS):
        cls = np.asarray(TASKS[t])
        remap = {int(c): i for i, c in enumerate(cls)}
        idx = np.where(np.isin(TR_Y, cls))[0]
        y_task = TR_Y[idx]

        # frozen snapshot of the network at the END of task t-1 (serves both maha + distill)
        old_model, P_maha, F_old_cache, POS_LUT = None, {}, None, None
        maha_teacher = None          # phi^{t-1} by default; phi_0 when MAHA_TEACHER=base
        # BANK_MODE=transport also needs the previous-frame backbone, even with lam2=lam3=0
        if t > 0 and (lam2 > 0 or lam3 > 0 or BANK_MODE == "transport"):
            old_model = copy.deepcopy(model).eval()
            for p in old_model.parameters():
                p.requires_grad_(False)
            maha_teacher = old_model
            if lam2 > 0:
                # MAHA_TEACHER=base distils from the FROZEN PRETRAINED phi_0 rather than
                # phi^{t-1}. phi_0 costs nothing to reconstruct (drop the adapter) and its
                # geometry reflects all 200 classes via pretraining -- global structure that
                # no single 20-class task supplies.
                if MAHA_TEACHER == "base":
                    maha_teacher = load_backbone(MODEL, pretrained=True, num_classes=0,
                                                 device=DEV).eval()
                    for p in maha_teacher.parameters():
                        p.requires_grad_(False)
                F_old_cache, P_maha = build_maha(maha_teacher, idx, y_task, cls)
                # keep it alive: under AUG=1 the loss re-forwards the teacher per batch, and
                # using old_model there would silently pair phi_0 statistics with phi^{t-1}
                # features. Free it only when the cache is what gets used.
                if MAHA_TEACHER == "base" and not AUG:
                    del maha_teacher; maha_teacher = old_model; torch.cuda.empty_cache()
                # HFWrap yields the GLOBAL row index (0..len(TRAIN)-1) but F_old_cache holds
                # only this task's rows, in `idx` order. Map global -> local or we index a
                # 2400-row cache with values up to 23999 (device-side assert).
                POS_LUT = torch.full((len(TRAIN),), -1, dtype=torch.long, device=DEV)
                POS_LUT[torch.as_tensor(idx, device=DEV)] = torch.arange(len(idx), device=DEV)
                log(f"  [{tag} t={t}] Mahalanobis: {len(P_maha)} class Sigma^-1/2 "
                    f"(shrink={MAHA_SHRINK}), cache {tuple(F_old_cache.shape)}")

        loader = DataLoader(Subset(TRAIN_AUG, idx.tolist()), batch_size=BS, shuffle=True,
                            num_workers=8, pin_memory=True)
        head = nn.Linear(768, CPT).to(DEV)
        opt = torch.optim.AdamW(lp + list(head.parameters()), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        ce = nn.CrossEntropyLoss()
        with torch.no_grad():
            AB0 = {n: (m.lora_A.detach().clone(), m.lora_B.detach().clone())
                   for n, m in mods if n in S_bank}

        L1 = L2 = L3 = L4 = 0.0
        do_train = (freeze_after is None) or (t <= freeze_after)
        if not do_train:
            log(f"  [{tag} t={t}] backbone FROZEN (freeze_after={freeze_after}) — no training")
        for ep in range(EPOCHS if do_train else 0):
            model.train(); ok = tot = 0
            for x, lab, ridx in loader:
                x = x.to(DEV, non_blocking=True)
                y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
                labs = lab.to(DEV)
                need_tok = (lam3 > 0 and old_model is not None)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if need_tok:
                        tok = model.forward_features(x)
                        feat = tok[:, 0].float()
                    else:
                        feat = model(x).float()
                    loss = ce(head(feat), y)
                if lam1 > 0 and AB0:
                    p1 = cone_penalty(mods, S_bank, AB0, T_bank); L1 = float(p1)
                    loss = loss + lam1 * p1
                if lam2 > 0 and P_maha:
                    if AUG:
                        # The cache holds the teacher's DETERMINISTIC view; under random
                        # augmentation the current features describe a different view, so the
                        # pairing silently breaks. Re-forward the same augmented batch through
                        # the SAME teacher that produced P_maha (phi^{t-1} or phi_0).
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            fo = Fn.normalize(maha_teacher(x).float(), dim=1)
                    else:
                        pos = POS_LUT[ridx.to(DEV)]
                        assert int(pos.min()) >= 0, "batch row not in this task's cache"
                        fo = F_old_cache[pos]
                    p2 = maha_loss(Fn.normalize(feat, dim=1), fo, labs, P_maha)
                    L2 = float(p2); loss = loss + lam2 * p2
                if need_tok:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        tok_old = old_model.forward_features(x)
                    p3 = distill_loss(tok.float(), tok_old.float())
                    L3 = float(p3); loss = loss + lam3 * p3
                if LAM4 > 0:
                    p4 = anticollapse_loss(feat, AC_MODE)
                    L4 = float(p4); loss = loss + LAM4 * p4
                if not np.isfinite(float(loss)):
                    raise FloatingPointError(f"loss diverged at t={t} — lower lam1/lam2/lam3")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(lp + list(head.parameters()), GRAD_CLIP)
                opt.step()
                ok += int((head(feat).argmax(1) == y).sum()); tot += len(y)
            sch.step()
        if do_train:
            log(f"  [{tag} t={t}] train {ok/tot:.3f} | cone {L1:.3e} maha {L2:.3e} "
                f"distill {L3:.3e} anticollapse {L4:.3e}")
        # NOTE: old_model is freed AFTER the bank update below -- BANK_MODE=transport needs it.

        if lam1 > 0 and do_train:
            Acov, Gcov, Bcov = act_cov(model, idx, mods, head=head,
                                       y_task=y_task, remap=remap)
            cov = pick_cov(Acov, Bcov) if Bcov is not None else Acov

            # BANK_MODE=transport: the accumulated bank was measured in EARLIER frames; align
            # it into the current one before adding, instead of summing across frames.
            if BANK_MODE == "transport" and RAW_bank and old_model is not None:
                R = frame_transport(old_model, model, idx, mods)
                for n, _ in mods:
                    RAW_bank[n] = R[n].T @ RAW_bank[n] @ R[n]
                log(f"  [{tag} t={t}] bank transported into the current frame "
                    f"({len(R)} layers)")

            eA, eG = [], []
            for n, _ in mods:
                RAW_bank[n] = cov[n] if n not in RAW_bank else RAW_bank[n] + cov[n]
                S_bank[n] = eig_basis(RAW_bank[n])
                S = S_bank[n].double()
                eA.append(float((S * S).sum() / (torch.diagonal(RAW_bank[n]).sum() + 1e-30)))
                if Gcov is not None:
                    T_bank[n] = eig_basis(Gcov[n], m=M_OUT)   # K-FAC G factor
                    T = T_bank[n].double()
                    eG.append(float((T * T).sum() /
                                    (torch.diagonal(Gcov[n]).sum() + 1e-30)))
            if Gcov is not None:
                # Two-sided truncation COMPOUNDS: the penalty retains ~ eA * eG of its
                # full-rank value. Verified numerically (m=32/96 kept only 19%). If this
                # product is small, kfac is silently under-penalising -- raise M_EIG/M_OUT
                # before concluding the idea failed.
                log(f"  [{tag} t={t}] K-FAC: energy kept A {np.mean(eA):.3f} (m={M_EIG}) "
                    f"x G {np.mean(eG):.3f} (m_out={M_OUT}) = {np.mean(eA)*np.mean(eG):.3f}")
            else:
                log(f"  [{tag} t={t}] bank updated ({COV_MODE}, {BANK_MODE}); "
                    f"energy kept {np.mean(eA):.3f}")
        del old_model, maha_teacher; torch.cuda.empty_cache()

        # ---- prototypes + shared covariance in the CURRENT frame ----
        Ztask = extract(model, TRAIN, idx)
        for c in cls:
            mu_bank[int(c)] = un(Ztask[y_task == c]).mean(0)
        R = np.concatenate([un(Ztask[y_task == c]) - mu_bank[int(c)] for c in cls])
        cov_bank = (R.T @ R) / max(len(R) - len(cls), 1)

        # ---- head statistics: RECOMPUTE (current real + virtual old) or ACCUM (additive) ----
        seen = np.concatenate(TASKS[:t + 1])
        old_cls = np.setdiff1d(seen, cls)
        te = np.where(np.isin(TE_Y, seen))[0]
        Zte = extract(model, TEST, te)
        pm = np.random.default_rng(t).permutation(len(idx))
        nv = max(int(0.1 * len(idx)), 1)
        Zr = un(Ztask)
        vZ.append(Zr[pm[:nv]]); vy.append(y_task[pm[:nv]])
        if stats == "accum":
            g, c_ = build_GC(Zr[pm[nv:]], y_task[pm[nv:]])
            Gacc += g; Cacc += c_
            G, C = Gacc, Cacc
        else:
            Ztr, ytr = Zr[pm[nv:]], y_task[pm[nv:]]
            if len(old_cls):
                Zv, yv = virtual_feats(mu_bank, cov_bank, [int(c) for c in old_cls], seed=t)
                Ztr = np.concatenate([Ztr, Zv]); ytr = np.concatenate([ytr, yv])
            G, C = build_GC(Ztr, ytr)
        a = solve_eval(G, C, np.concatenate(vZ), np.concatenate(vy), Zte, TE_Y[te], seen)
        accs.append(a)

        # ---- ORACLE-STATS diagnostic (cheating: uses ALL old data) --------------------
        # Splits the gap to the ceiling into two parts that need OPPOSITE fixes:
        #   method -> oracle_stats : cost of APPROXIMATE STATISTICS (virtual feats / stale accum)
        #   oracle_stats -> ceiling: cost of WORSE FEATURES (sequential vs joint training)
        # Same backbone, same head, only the statistics change. No retraining.
        if ORACLE_STATS:
            all_idx = np.where(np.isin(TR_Y, seen))[0]
            Zall = un(extract(model, TRAIN, all_idx)); yall = TR_Y[all_idx]
            pmo = np.random.default_rng(999).permutation(len(all_idx))
            nvo = max(int(0.1 * len(all_idx)), 1)
            Go, Co = build_GC(Zall[pmo[nvo:]], yall[pmo[nvo:]])
            ao = solve_eval(Go, Co, Zall[pmo[:nvo]], yall[pmo[:nvo]], Zte, TE_Y[te], seen)
            oracle_accs.append(ao)
            log(f"  [{tag} t={t}] seen={len(seen):3d}  acc {a:.4f} | "
                f"oracle-stats {ao:.4f}  (statistics gap {ao-a:+.4f})")
        else:
            log(f"  [{tag} t={t}] seen={len(seen):3d}  acc {a:.4f}")

    del model; torch.cuda.empty_cache()
    if oracle_accs:
        CEIL_R = float(os.environ.get("CEILING", 0.8510))   # joint/offline bound
        log(f"  [{tag}] HEADROOM SPLIT at t=9:  method {accs[-1]:.4f} "
            f"-> oracle-stats {oracle_accs[-1]:.4f} (statistics gap "
            f"{oracle_accs[-1]-accs[-1]:+.4f}) -> ceiling {CEIL_R:.4f} (feature gap "
            f"{CEIL_R-oracle_accs[-1]:+.4f})")
        results_oracle[tag] = oracle_accs
    return accs


VAR = {           # (lam1 cone, lam2 maha, lam3 distill, statistics mode, freeze_after)
    # --- BARS: measure these on WHATEVER backbone is selected. The 0.7272 / 0.7858 / 0.8355
    #     numbers quoted elsewhere were measured on augreg2 and DO NOT TRANSFER to IN21k. ---
    "A_frozen":         (0.0,  0.0,  0.0,  "accum",     -1),   # never adapt (floor)
    "A_plus":           (0.0,  0.0,  0.0,  "accum",      0),   # adapt task 0, freeze (THE BAR)
    # --- methods ---
    "null_baseline":    (0.0,  0.0,  0.0,  "recompute", None), # lam=0 denominator
    "maha_only":        (0.0,  LAM2, 0.0,  "recompute", None),
    "cone_only":        (LAM1, 0.0,  0.0,  "recompute", None),
    "cov_maha":         (LAM1, LAM2, 0.0,  "recompute", None),
    "cov_maha_distill": (LAM1, LAM2, LAM3, "recompute", None),
    # --- ACCUM arms: accumulated (exactly additive) statistics instead of rebuilding them.
    #     accum gave the best A-avg anywhere on augreg2 (cone 0.8232, only -0.008 from the bar)
    #     while recompute gave the best A-last. The full stack has NEVER been run in accum. ---
    "cone_accum":       (LAM1, 0.0,  0.0,  "accum",     None),
    "cov_maha_accum":   (LAM1, LAM2, 0.0,  "accum",     None),
    "full_accum":       (LAM1, LAM2, LAM3, "accum",     None),
    # THE MISSING ABLATION. cone_accum alone is BELOW A_frozen at lam1 in {1,10,100}
    # (0.6557/0.5788/0.6510 vs 0.6867) yet full_accum = 0.7527. Is the cone contributing to
    # the stack, or is maha+distill carrying it and the cone is dead weight / harmful?
    "maha_distill_accum": (0.0, LAM2, LAM3, "accum",    None),
}
WANT = [v for v in os.environ.get("VARIANTS", ",".join(VAR)).split(",") if v in VAR]

# TAG distinguishes hyperparameter sweeps that reuse the same variant name. Without it,
# `VARIANTS=cone_accum LAM1=10` and `LAM1=100` both print as "cone_accum" and collapse into a
# single row in the summary, making the sweep unreadable.
TAG = os.environ.get("TAG", "")
if not TAG:
    bits = []
    if LAM1 != 1.0:  bits.append(f"l1={LAM1:g}")
    if LAM2 != 1.0:  bits.append(f"l2={LAM2:g}")
    if LAM3 != 0.1:  bits.append(f"l3={LAM3:g}")
    if M_EIG != 64:  bits.append(f"m={M_EIG}")
    if SEED != 0:    bits.append(f"s{SEED}")
    TAG = ("_" + "_".join(bits)) if bits else ""
elif not TAG.startswith("_"):
    TAG = "_" + TAG

results = {}
for v in WANT:
    results[v + TAG] = run(v + TAG, *VAR[v])

# per-backbone results file: bars and methods from different backbones must NEVER be mixed
OUT = f"exp8_results_{MODEL.split('.')[-1]}.npy"
merged = {}
if os.path.exists(OUT):
    try:
        merged = np.load(OUT, allow_pickle=True).item()
    except Exception as e:
        print(f"[warn] {e}")
for v, a in results.items():
    merged[v] = a                                                   # most recent
    merged[f"{v}@l1={LAM1:g},l2={LAM2:g},l3={LAM3:g},s{SEED}"] = a  # never clobbered
np.save(OUT, merged, allow_pickle=True)

# Bars measured on augreg2_in21k_ft_in1k. They DO NOT transfer to another backbone, so prefer
# an A_plus / A_frozen measured in THIS run (same protocol, same head, same backbone).
AUGREG2 = dict(BAR=0.7858, BAR_AVG=0.8313, FROZEN=0.7272, PREV=0.7390,
               CE_ONLY=0.6557, CEIL=0.8355)
IS_STANDIN = MODEL == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
# BAR must be RECIPE-MATCHED: an aug40 method against a 10ep/no-aug bar is not a comparison.
# BAR_KEY picks which measured A_plus to compare against (e.g. BAR_KEY=A_plus_aug40).
BAR_KEY = os.environ.get("BAR_KEY", "A_plus_aug40" if AUG else "A_plus")
if BAR_KEY in merged:
    BAR, BAR_AVG = merged[BAR_KEY][-1], float(np.mean(merged[BAR_KEY]))
    bar_src = f"{BAR_KEY}, measured THIS backbone"
elif "A_plus" in merged:
    BAR, BAR_AVG = merged["A_plus"][-1], float(np.mean(merged["A_plus"]))
    bar_src = "A_plus (WARNING: recipe may not match this run)"
else:
    BAR, BAR_AVG, bar_src = AUGREG2["BAR"], AUGREG2["BAR_AVG"], "augreg2 — STALE if MODEL changed"
FROZEN = merged["A_frozen"][-1] if "A_frozen" in merged else AUGREG2["FROZEN"]
PREV, CE_ONLY, CEIL = AUGREG2["PREV"], AUGREG2["CE_ONLY"], AUGREG2["CEIL"]

print("\n" + "=" * 100)
print(f"EXP8 — combined ZERO-IMAGE objective (Split-ImageNet-R)")
print(f"backbone: {MODEL}")
if not IS_STANDIN and "A_plus" not in merged:
    print("!! WARNING: backbone is NOT augreg2 but no A_plus was measured in this run.")
    print("!! The bars below are from augreg2 and are MEANINGLESS here.")
    print("!! Run:  VARIANTS=A_frozen,A_plus python -u exp8_combined.py")
print(f"bar source: {bar_src}")
print("=" * 100)
print(f"{'variant':>20} {'A-last':>8} {'A-avg':>8} | {'dA-last':>9} {'dA-avg':>8} | "
      f"{'vs prev best':>13}")
for v, a in results.items():        # keys carry TAG; iterating WANT would KeyError
    av = float(np.mean(a))
    win = "  *** BEATS BAR ***" if (a[-1] > BAR or av > BAR_AVG) else ""
    print(f"{v:>20} {a[-1]:>8.4f} {av:>8.4f} | {a[-1]-BAR:>+9.4f} {av-BAR_AVG:>+8.4f} | "
          f"{a[-1]-PREV:>+13.4f}{win}")
for v, oa in results_oracle.items():
    print(f"{v+' [oracle-stats]':>20} {oa[-1]:>8.4f} {float(np.mean(oa)):>8.4f} | "
          f"{oa[-1]-BAR:>+9.4f} {float(np.mean(oa))-BAR_AVG:>+8.4f} |"
          f"{'':>14}  <- diagnostic (uses all old data)")
print("-" * 96)
print(f"{'seqCE+synth':>20} {CE_ONLY:>8.4f}   no relational loss")
print(f"{'seqGram+synth':>20} {PREV:>8.4f}   prev best — used 20 IMAGES/class")
print(f"{'A_frozen':>20} {FROZEN:>8.4f}   floor")
print(f"{'A+ first-session':>20} {BAR:>8.4f} {BAR_AVG:>8.4f}   <- THE BAR (both metrics)")
print(f"{'joint ceiling':>20} {CEIL:>8.4f}")
print(f"{'cone_only accum':>20} {0.7065:>8.4f} {0.8232:>8.4f}   prior accum run "
      f"(A-avg only -0.008 from the bar)")
print("-" * 96)
if "maha_only" in results:
    print(f"replay-free Mahalanobis vs seqCE  : {results['maha_only'][-1]-CE_ONLY:+.4f}   "
          f"(Gram got +{PREV-CE_ONLY:.4f} WITH stored images)")
if "cov_maha" in results and "cone_only" in results:
    print(f"maha on top of the cone           : "
          f"{results['cov_maha'][-1]-results['cone_only'][-1]:+.4f}")
if "cov_maha_distill" in results and "cov_maha" in results:
    print(f"patch-token distillation adds     : "
          f"{results['cov_maha_distill'][-1]-results['cov_maha'][-1]:+.4f}")
print(f"\nWIN: A-last > {BAR:.4f} or A-avg > {BAR_AVG:.4f} with ZERO stored images "
      f"({bar_src}).")
print("=" * 96)
