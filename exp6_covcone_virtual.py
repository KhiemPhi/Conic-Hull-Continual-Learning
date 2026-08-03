"""
exp6_covcone_virtual.py — COVARIANCE-CONE LoRA PENALTY + VIRTUAL CONIC FEATURE REPLAY.

Implements the proposed zero-storage objective, with two math corrections (see below):

    L_total = L_t(f_theta(X_t))                                    current task
            + lam1 * sum_l || dW_l U_l Lam_l^{1/2} ||_F^2          COVARIANCE CONE  (term 1)
            + RanPAC statistics regenerated from virtual features  VIRTUAL CONE     (term 2)

CORRECTION 1 (term 1 was dimensionally invalid). The proposal penalises
Tr(dtheta^T U_k Lam_k U_k^T dtheta) with Sigma_k = E[z z^T] the covariance of the FINAL ViT
output (768-d). But dtheta is a PARAMETER-space object spread over 24 layers of heterogeneous
shape, so U_k^T dtheta is undefined. Fix (= Gradient Projection Memory / Adam-NSCL): track the
covariance of each layer's INPUT ACTIVATIONS. Per layer, with dW_l in R^{out x in} and
U_l in R^{in x m}, ||dW_l U_l Lam_l^{1/2}||_F^2 is well-defined and expresses exactly the same
idea. For LoRA, dW_l = (alpha/r)(B_l A_l - B_l^0 A_l^0), and factoring as B_l (A_l U_l) makes
the cost LINEAR IN RANK r. Storage is U_l Lam_l^{1/2} only: 24 x 768 x m floats (~4.7 MB @ m=64).
No images, no features, no replay.

CORRECTION 2 (term 2 was not computable). L_MSE(phi(W_rp f_theta(X_virtual)), phi(W_rp z_virtual))
needs an IMAGE X_virtual to push through f_theta -- but nothing is stored, and if z_virtual comes
from fixed prototypes then phi(W_rp z_virtual) is a CONSTANT, so the term is vacuous as a loss.
Its useful content is regenerating RanPAC's G/C statistics for old classes, which is not a loss.
Implemented there, and we compare the LITERAL conic version against the sane one:
    dirichlet : z = sum_c alpha_c mu_c + eps,  alpha ~ Dir(gamma)   (mass BETWEEN prototypes)
    gauss     : z = mu_c + eps,                eps ~ N(0, Sigma_c)  (mass AROUND each prototype)
    none      : no virtual features
Prediction: dirichlet places mass between prototypes (a blend of "oak" and "car" matches no real
image) while real class features cluster around each mu_c -- so gauss should win. This is a
direct, cone-specific test of the proposal's generative claim.

Bars (crux_method.py, Split-ImageNet-R, same protocol):
    A_frozen 0.7272 | A+ first-session RanPAC 0.7858 (THE BAR) | prev best seqGram+synth 0.7390
    joint ceiling (crux_headroom) 0.8355

Run:  python -u exp6_covcone_virtual.py
      LAM1=1.0 VIRT=gauss python -u exp6_covcone_virtual.py
      VARIANTS=cone_only,full python -u exp6_covcone_virtual.py
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Dataset, Subset
import timm
from timm.data import resolve_model_data_config, create_transform

from backbone import load_backbone, freeze_non_lora, get_lora_params

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda"
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
N_TASKS, CPT = 10, 20
EPOCHS, LR, BS = 10, 1e-4, 128
M_EIG = int(os.environ.get("M_EIG", 64))        # eigenvectors kept per layer
# Covariance-cone weight. Set EMPIRICALLY, not from an a-priori estimate:
#   LAM1=1.0   -> lam1*pen/CE ~ 10..100  -> A-last 0.7065, A-avg 0.8232   (WORKS)
#   LAM1=1e-4  -> lam1*pen/CE ~ 1e-2     -> tracks the no-penalty baseline (INERT)
# An earlier a-priori calibration predicted pen ~ 3.5e2-6e3 from a random eigenbasis and
# concluded LAM1~1e-4; the real penalty from actual activation covariance is 14-90, so that
# guidance was wrong by ~4 orders of magnitude. The useful regime has the penalty DOMINATING
# the CE. Sweep upward from 1.0 ({1.0, 10, 100}) to probe the frozen-backbone limit.
LAM1 = float(os.environ.get("LAM1", 1.0))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))   # required for LAM1 >= 10 (see eig_basis)
M_RP = 10000
LAMBDAS = [1e2, 1e3, 1e4]
SYNTH_PER_CLASS = int(os.environ.get("SYNTH", 120))
DIR_GAMMA = float(os.environ.get("DIR_GAMMA", 1.0))
COV_BATCHES = 12                                 # batches used to estimate activation cov

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
TRAIN = HFWrap(_ds, _p[:_n], _lab[_p[:_n]]);  TR_Y = _lab[_p[:_n]]
TEST  = HFWrap(_ds, _p[_n:], _lab[_p[_n:]]);  TE_Y = _lab[_p[_n:]]
N_CLS = len(_cl)
ORDER = np.random.default_rng(SEED).permutation(N_CLS)
TASKS = [ORDER[i * CPT:(i + 1) * CPT] for i in range(N_TASKS)]
log(f"[ImageNet-R] train {len(TR_Y)} test {len(TE_Y)} classes {N_CLS} | "
    f"{N_TASKS}x{CPT} | m_eig={M_EIG} lam1={LAM1}")


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


# ==================== TERM 1: covariance-cone machinery ====================
from backbone import _ALL_LORA_TYPES


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _ALL_LORA_TYPES)]


def delta_w(mod):
    """(alpha/r) * B A  -- the LoRA weight delta, shape (out, in). Handles the
    task_shared LoRALinear layout (single A, B pair)."""
    return (mod.lora_B @ mod.lora_A) * mod.scaling


@torch.no_grad()
def accumulate_act_cov(model, idx, mods, n_batches=COV_BATCHES):
    """Sigma_l = E[x_l x_l^T] over this task's data, per LoRA layer input.
    Zero-storage w.r.t. images: only the (in x in) covariance is kept, then reduced to
    the top-M_EIG eigenpairs."""
    covs = {n: torch.zeros(m.in_features, m.in_features, device=DEV, dtype=torch.float64)
            for n, m in mods}
    cnt = {n: 0 for n, _ in mods}
    handles = []

    def mk(name):
        def hook(module, inp):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).double()      # (tokens, in)
            covs[name] += x.T @ x
            cnt[name] += x.shape[0]
        return hook

    for n, m in mods:
        handles.append(m.register_forward_pre_hook(mk(n)))
    model.eval()
    loader = DataLoader(Subset(TRAIN, idx.tolist()), batch_size=64, shuffle=True, num_workers=8)
    for bi, (x, _) in enumerate(loader):
        if bi >= n_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x.to(DEV, non_blocking=True))
    for h in handles:
        h.remove()
    return {n: (covs[n] / max(cnt[n], 1)) for n, _ in mods}


def eig_basis(cov, m=M_EIG):
    """Top-m eigenpairs -> the matrix S = U Lam^{1/2} used by the penalty.
    NOTE: eigh on a non-finite matrix aborts inside cuSOLVER (core dump, not an exception),
    which is what a too-large LAM1 produces: big penalty grad -> weights blow up ->
    activations inf -> covariance nan -> crash. Guard explicitly."""
    if not torch.isfinite(cov).all():
        raise FloatingPointError(
            "non-finite activation covariance — the LoRA weights have diverged. "
            "LAM1 is too large (try a smaller value, or rely on grad clipping).")
    ev, U = torch.linalg.eigh(cov)                      # ascending
    ev = torch.clamp(ev[-m:], min=0.0)
    U = U[:, -m:]
    return (U * ev.sqrt().unsqueeze(0)).float()          # (in, m)


def cone_penalty(mods, S_bank, D_bank, AB0, mode):
    """Penalty on the UPDATE dW_l = (alpha/r)(B A - B0 A0), by mode:

      lowrank       || dW U Lam^{1/2} ||_F^2                       (top-m eigenspace only)
      lowrank_diag  the above  +  sum_i D_ii || dW e_i ||^2        (+ diagonal residual)
      l2            || dW ||_F^2                                    (CONTROL: plain weight decay
                    on the update -- what lowrank_diag COLLAPSES TO if D is isotropic)

    Both extra terms are computed WITHOUT ever forming the (out x in) matrix dW:
        || dW diag(d) ||_F^2 = s^2 [ Tr(B^T B  P) - 2 Tr(B^T B0  Q) + Tr(B0^T B0  R) ]
        P = Ad Ad^T,  Q = A0d Ad^T,  R = A0d A0d^T,  Ad = A * d,  all (r x r).
    Cost is O(r^2 (in+out)) per layer -- same spirit as the low-rank term."""
    tot = 0.0
    for n, mod in mods:
        if n not in AB0:
            continue
        A0, B0 = AB0[n]
        A, B, s2 = mod.lora_A, mod.lora_B, mod.scaling ** 2

        if mode in ("lowrank", "lowrank_diag"):
            S = S_bank.get(n)
            if S is not None:
                cur = B @ (A @ S)                        # (out, m)  <- linear in r
                ref = B0 @ (A0 @ S)
                tot = tot + s2 * ((cur - ref) ** 2).sum()

        if mode in ("lowrank_diag", "l2"):
            if mode == "l2":
                Ad, A0d = A, A0                          # d = 1  ->  plain ||dW||_F^2
            else:
                d = D_bank[n].sqrt().unsqueeze(0)        # (1, in)
                Ad, A0d = A * d, A0 * d
            GB, GBB0, GB0 = B.T @ B, B.T @ B0, B0.T @ B0         # (r, r)
            P, Q, R = Ad @ Ad.T, A0d @ Ad.T, A0d @ A0d.T         # (r, r)
            tot = tot + s2 * ((GB * P.T).sum() - 2 * (GBB0 * Q.T).sum()
                              + (GB0 * R.T).sum())
    return tot


# ==================== TERM 2: virtual conic features ====================
def virtual_feats(mu, cov_shared, classes, mode, n_per=SYNTH_PER_CLASS, seed=0):
    """mode='gauss'     : z = mu_c + eps            (mass AROUND each prototype)
       mode='dirichlet' : z = sum_c a_c mu_c + eps  (mass BETWEEN prototypes -- the literal
                          conic-hull proposal; a ~ Dir(gamma), so a >= 0 and sums to 1)"""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov_shared + 1e-4 * np.eye(cov_shared.shape[0]))
    Z, Y = [], []
    if mode == "gauss":
        for c in classes:
            Z.append(mu[c] + rng.standard_normal((n_per, len(mu[c]))) @ L.T)
            Y.append(np.full(n_per, c))
    else:
        M = np.stack([mu[c] for c in classes])
        for i, c in enumerate(classes):
            a = rng.dirichlet(np.full(len(classes), DIR_GAMMA), size=n_per)
            a[:, i] += 2.0                                # keep the label meaningful
            a /= a.sum(1, keepdims=True)
            Z.append(a @ M + rng.standard_normal((n_per, M.shape[1])) @ L.T)
            Y.append(np.full(n_per, c))
    return np.concatenate(Z), np.concatenate(Y)


# ============================== training ==============================
@torch.no_grad()
def extract(model, ds, idx):
    model.eval()
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=256, shuffle=False, num_workers=8)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out, 0)


def run(tag, lam1, virt, stats="recompute", pen_mode="lowrank"):
    log(f"=== VARIANT {tag}  (lam1={lam1}, virtual={virt}, stats={stats}, pen={pen_mode}) ===")
    model = load_backbone(MODEL, pretrained=True, num_classes=0, device=DEV,
                          lora_rank=32, lora_alpha=4.0, lora_config="task_shared")
    freeze_non_lora(model)
    lp = list(get_lora_params(model))
    mods = lora_modules(model)
    S_bank, D_bank, mu_bank, cov_bank = {}, {}, {}, None
    accs = []
    # persistent statistics for stats="accum" (RanPAC-style exact additivity, which is only
    # valid if the features do NOT drift -- precisely what term 1 is supposed to guarantee)
    Gacc = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    Cacc = torch.zeros(M_RP, N_CLS, device=DEV, dtype=torch.float64)
    vZ, vy = [], []

    for t in range(N_TASKS):
        cls = np.asarray(TASKS[t])
        remap = {int(c): i for i, c in enumerate(cls)}
        idx = np.where(np.isin(TR_Y, cls))[0]
        loader = DataLoader(Subset(TRAIN, idx.tolist()), batch_size=BS, shuffle=True,
                            num_workers=8, pin_memory=True)
        head = nn.Linear(768, CPT).to(DEV)
        opt = torch.optim.AdamW(lp + list(head.parameters()), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        ce = nn.CrossEntropyLoss()

        # snapshot (A0, B0) at task start -> the penalty measures the UPDATE, not the total
        # adapter. Storing the factors (not the product) keeps every term O(r^2).
        with torch.no_grad():
            AB0 = {n: (m.lora_A.detach().clone(), m.lora_B.detach().clone())
                   for n, m in mods if (n in S_bank or pen_mode == "l2")}

        pen_last, ce_last, first_pen = 0.0, 0.0, None
        for ep in range(EPOCHS):
            model.train(); ok = tot = 0
            for x, lab in loader:
                x = x.to(DEV, non_blocking=True)
                y = torch.tensor([remap[int(l)] for l in lab], device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = head(model(x).float())
                    loss = ce(lg, y)
                ce_last = float(loss)
                if lam1 > 0 and AB0:
                    pen = cone_penalty(mods, S_bank, D_bank, AB0, pen_mode)
                    loss = loss + lam1 * pen
                    pen_last = float(pen)
                    if not np.isfinite(pen_last):
                        raise FloatingPointError(
                            f"penalty diverged at t={t} (LAM1={lam1} too large)")
                    if first_pen is None and pen_last > 0:
                        first_pen = pen_last
                        # one-time sanity: is lam1 in a range where the penalty can bite?
                        log(f"    [{tag} t={t}] FIRST nonzero penalty {pen_last:.3e} | "
                            f"lam1*pen {lam1*pen_last:.3e} vs CE {ce_last:.3e} | "
                            f"ratio {lam1*pen_last/max(ce_last,1e-12):.3e}  "
                            f"(EMPIRICAL: ratio ~10-100 works [LAM1=1.0]; "
                            f"ratio ~1e-2 is INERT. Note this is the FIRST step, where the "
                            f"penalty starts near 0 -- judge from the end-of-task value below.)")
                opt.zero_grad(); loss.backward()
                # essential once the penalty is large: without it, LAM1>=10 diverges and the
                # subsequent eigh core-dumps rather than raising.
                torch.nn.utils.clip_grad_norm_(lp + list(head.parameters()), GRAD_CLIP)
                opt.step()
                ok += int((lg.argmax(1) == y).sum()); tot += len(y)
            sch.step()
        if lam1 > 0 and S_bank and pen_last == 0.0:
            log(f"  [{tag} t={t}] WARNING: penalty computed but exactly 0 — "
                f"check S_bank/W0 wiring")
        log(f"  [{tag} t={t}] train-acc {ok/tot:.3f}  cone-pen {pen_last:.4e}  "
            f"lam1*pen {lam1*pen_last:.4e}  CE {ce_last:.4f}"
            + ("  [penalty inactive: lam1=0]" if lam1 == 0 else
               ("  [no prior tasks yet]" if not S_bank else "")))

        # ---- update the covariance cone with THIS task (running sum over tasks) ----
        if lam1 > 0 and pen_mode != "l2":
            cov = accumulate_act_cov(model, idx, mods)
            aniso = []
            for n, _ in mods:
                prev = S_bank.get(n)
                acc_cov = cov[n] if prev is None else cov[n] + (prev.double() @ prev.double().T)
                S_bank[n] = eig_basis(acc_cov)
                # D = diag(Sigma - U Lam U^T): the residual variance the top-m eigenspace misses.
                # If D is ~flat, the diag term is just weight decay -> the `l2` arm should match.
                S = S_bank[n].double()
                D = (torch.diagonal(acc_cov) - (S * S).sum(1)).clamp(min=0)
                D_bank[n] = (D / (D.mean() + 1e-30)).float()      # scale-free; lam1 sets strength
                aniso.append(float(D.std() / (D.mean() + 1e-30)))
            log(f"  [{tag} t={t}] cone updated ({len(S_bank)} layers, m={M_EIG}) | "
                f"D anisotropy CV {np.mean(aniso):.3f}  "
                f"({'ANISOTROPIC -> diag != l2' if np.mean(aniso) > 0.5 else 'flat -> diag ~ l2'})")

        # ---- prototypes for this task's classes, in the CURRENT frame ----
        Ztask = extract(model, TRAIN, idx)
        ytask = TR_Y[idx]
        for c in cls:
            mu_bank[int(c)] = un(Ztask[ytask == c]).mean(0)
        R = np.concatenate([un(Ztask[ytask == c]) - mu_bank[int(c)] for c in cls])
        cov_bank = (R.T @ R) / max(len(R) - len(cls), 1)

        # ---- RanPAC: current task real features + virtual features for OLD classes ----
        seen = np.concatenate(TASKS[:t + 1])
        old = np.setdiff1d(seen, cls)
        te = np.where(np.isin(TE_Y, seen))[0]
        Zte = extract(model, TEST, te)
        pm = np.random.default_rng(t).permutation(len(idx))
        nv = max(int(0.1 * len(idx)), 1)
        Zr, yr = un(Ztask), ytask
        Ztr_all, ytr_all = Zr[pm[nv:]], yr[pm[nv:]]
        if virt != "none" and len(old):
            Zv, yv = virtual_feats(mu_bank, cov_bank, [int(c) for c in old], virt, seed=t)
            Ztr_all = np.concatenate([Ztr_all, Zv]); ytr_all = np.concatenate([ytr_all, yv])
        vZ.append(Zr[pm[:nv]]); vy.append(yr[pm[:nv]])
        if stats == "accum":
            # accumulate this task's contribution into the persistent G/C and never recompute.
            # Old classes keep the statistics recorded in THEIR birth frame -- valid only if the
            # backbone has not drifted, which is exactly what the covariance cone should enforce.
            g, c_ = build_GC(Zr[pm[nv:]], yr[pm[nv:]])
            Gacc += g; Cacc += c_
            G, C = Gacc, Cacc
        else:
            G, C = build_GC(Ztr_all, ytr_all)
        a = solve_eval(G, C, np.concatenate(vZ), np.concatenate(vy), Zte, TE_Y[te], seen)
        accs.append(a)
        log(f"  [{tag} t={t}] seen={len(seen):3d}  acc {a:.4f}")

    del model; torch.cuda.empty_cache()
    return accs


# (lam1, virtual-feature mode, statistics mode, penalty mode)
# accum  = RanPAC-style additive G/C, never recomputed -> isolates TERM 1: does the covariance
#          cone keep old-frame statistics valid?  (no old-class data of any kind is used)
# recompute = rebuild G/C each task from current-task real features + virtual old-class features
#          -> isolates TERM 2.
VAR = {
    "base":            (0.0,  "none",      "accum",     "lowrank"),
    "cone_only":       (LAM1, "none",      "accum",     "lowrank"),       # top-64 eigenspace
    # --- exp7 said H1 (rank truncation) is the biggest leak: top-64 leaves 22.7% of the
    #     activation energy unprotected. These two arms test the fix AND its control. ---
    "cone_diag":       (LAM1, "none",      "accum",     "lowrank_diag"),  # + diagonal residual
    "l2_only":         (LAM1, "none",      "accum",     "l2"),            # CONTROL: plain
    #     ||dW||_F^2. If cone_diag ~= l2_only, the covariance machinery is not what is doing
    #     the work -- the same control that killed the RanPAC-projection and cone-classifier wins.
    "virt_gauss":      (0.0,  "gauss",     "recompute", "lowrank"),
    "virt_dirichlet":  (0.0,  "dirichlet", "recompute", "lowrank"),
    "full":            (LAM1, "gauss",     "recompute", "lowrank"),
    "full_dirichlet":  (LAM1, "dirichlet", "recompute", "lowrank"),
}
WANT = [v for v in os.environ.get("VARIANTS", ",".join(VAR)).split(",") if v in VAR]

results = {}
for v in WANT:
    lam1, virt, stats, pen = VAR[v]
    results[v] = run(v, lam1, virt, stats, pen)

# MERGE with anything already on disk, and key by (variant, lam1) so different lambdas do not
# clobber each other. Previously each run overwrote the file with only its own variants.
# per-backbone: results from different backbones must never be mixed
OUT = f"exp6_results_{MODEL.split('.')[-1]}.npy"
merged = {}
if os.path.exists(OUT):
    try:
        merged = np.load(OUT, allow_pickle=True).item()
    except Exception as e:
        print(f"[warn] could not read {OUT}: {e}")
for v, a in results.items():
    merged[v] = a                                   # bare name = most recent run
    merged[f"{v}@lam{LAM1:g}"] = a                  # lambda-tagged, never clobbered
np.save(OUT, merged, allow_pickle=True)
print(f"\n[saved] {OUT} now holds: {sorted(merged)}")
BAR, FROZEN, PREV, CEIL = 0.7858, 0.7272, 0.7390, 0.8355
print("\n" + "=" * 96)
print(f"EXP6 — covariance-cone penalty + virtual conic replay (Split-ImageNet-R, lam1={LAM1})")
print("=" * 96)
print(f"{'variant':>16} {'A-last':>8} {'A-avg':>8} {'vs BAR':>9} {'vs prev best':>13}")
for v in WANT:
    a = results[v]
    print(f"{v:>16} {a[-1]:>8.4f} {float(np.mean(a)):>8.4f} "
          f"{a[-1]-BAR:>+9.4f} {a[-1]-PREV:>+13.4f}")
print("-" * 96)
print(f"{'A_frozen':>16} {FROZEN:>8.4f}   (floor)")
print(f"{'A+ first-session':>16} {BAR:>8.4f}   <- THE BAR (real RanPAC)")
print(f"{'prev best (Gram)':>16} {PREV:>8.4f}   seqGram_recompute+synth")
print(f"{'joint ceiling':>16} {CEIL:>8.4f}   (not a CIL method)")
print("-" * 96)
if "cone_only" in results and "base" in results:
    print(f"TERM 1 via additivity (cone_only - base, both accum): "
          f"{results['cone_only'][-1]-results['base'][-1]:+.4f}")
if "cone_diag" in results and "cone_only" in results:
    print(f"H1 FIX  (cone_diag - cone_only) : "
          f"{results['cone_diag'][-1]-results['cone_only'][-1]:+.4f}   "
          "<- does protecting the 22.7% residual help?")
if "cone_diag" in results and "l2_only" in results:
    d = results["cone_diag"][-1] - results["l2_only"][-1]
    print(f"THE CONTROL (cone_diag - l2_only): {d:+.4f}   "
          + ("<- covariance structure IS doing the work" if d > 0.02 else
             "<- TIE: plain weight decay on dW explains it; the covariance/cone "
             "machinery is not load-bearing"))
if "full" in results and "virt_gauss" in results:
    print(f"TERM 1 on top of TERM 2 (full - virt_gauss)        : "
          f"{results['full'][-1]-results['virt_gauss'][-1]:+.4f}")
if "virt_dirichlet" in results and "virt_gauss" in results:
    d = results["virt_dirichlet"][-1] - results["virt_gauss"][-1]
    print(f"dirichlet - gauss       : {d:+.4f}   <- is the LITERAL conic mixture better? "
          f"({'yes' if d > 0 else 'no -- mass between prototypes matches no real image'})")
print("\nWIN CONDITION: any variant A-last > 0.7858. Note A+ is the lam1->inf limit of")
print("cone_only (an infinite penalty freezes the backbone after task 0), so watch whether")
print("the curve is MONOTONE toward the degenerate endpoint again.")
print("=" * 96)
