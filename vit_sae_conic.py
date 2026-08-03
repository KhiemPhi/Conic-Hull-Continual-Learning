"""
vit_sae_conic.py
----------------
Does the CONIC (non-negative) code buy monosemanticity + editability on frozen
ViT patch tokens? The regime the cone rule predicts a win (superposition +
trained-pure atoms + code-as-output), evaluated on the CODE, not a label.

Frozen CLIP ViT-B/16 patch tokens (COCO val). Learn a dictionary 4 ways at matched
size & matched sparsity (Top-K), then evaluate:
  conic-SAE  : Top-K ReLU autoencoder -> NON-NEGATIVE code (the cone)
  signed-SAE : Top-K autoencoder, no ReLU -> signed code (isolates non-negativity)
  PCA        : dense signed components, top-k coeffs
  kmeans     : prototypes / hard assignment (L0=1)

Probes: (1) recon nMSE vs L0 frontier; (2) monosemanticity = label-free coherence
(mean cos among each atom's top patches) + image-label purity; (3) editability =
max-pool codes -> probe, ablate each concept's top atom, targeted vs collateral AP.

    HF_HUB_OFFLINE=1 python -u vit_sae_conic.py --n-img 2500 --dict 1024
"""
import argparse, io, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT, DATA_DIR = "./sae_out", "./data"


class _Imgs(Dataset):
    def __init__(self, ds, pre, ik): self.ds, self.pre, self.ik = ds, pre, ik
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img = self.ds[i][self.ik]
        if not hasattr(img, "mode"):
            from PIL import Image; img = Image.open(io.BytesIO(img["bytes"]))
        return self.pre(img.convert("RGB"))


# dataset -> (repo, split, kind).  split None => pool all splits.
DATASETS = {
    "COCO":   ("detection-datasets/coco", "val",        "detection"),
    "CPPE5":  ("rishitdagli/cppe-5",      None,         "detection"),
    "CUB200": ("Donghyun99/CUB-200-2011", None,         "classification"),
    "ADE20K": ("1aurent/ADE20K",          "validation",  "segmentation"),
}


def _load_hf(dataset, n_img):
    from datasets import load_dataset, concatenate_datasets
    repo, split, kind = DATASETS[dataset]
    cache = os.path.join(DATA_DIR, "hf")
    if split is None:
        dd = load_dataset(repo, cache_dir=cache)
        ds = concatenate_datasets([dd[s] for s in dd])
    else:
        ds = load_dataset(repo, split=split, cache_dir=cache)
    if n_img and n_img < len(ds):
        ds = ds.select(range(n_img))
    cols = ds.column_names
    ik = next(c for c in ("image", "img", "Image", "picture") if c in cols)
    # labels
    if kind == "segmentation":
        Y = np.zeros((len(ds), 1), np.float32)   # dictionary training is unsupervised;
        return ds, Y, ik                          # dense-mask GT built by the seg script
    if kind == "detection":
        objs = ds["objects"]
        n_cls = max((max(o["category"]) for o in objs if o["category"]), default=0) + 1
        Y = np.zeros((len(ds), n_cls), np.float32)
        for i, o in enumerate(objs):
            for c in o["category"]:
                Y[i, int(c)] = 1.0
    else:
        lk = next(c for c in ("label", "labels", "fine_label", "class", "target") if c in cols)
        lab = np.array([int(v) for v in ds[lk]])
        Y = np.zeros((len(ds), int(lab.max()) + 1), np.float32)
        Y[np.arange(len(ds)), lab] = 1.0
    return ds, Y, ik


def _get_backbone(backbone):
    """Return (model, transform, token_fn). token_fn(model,x)->(B,P,D) patch tokens."""
    from torchvision import transforms as T
    if backbone in ("clip", "clip_openai"):
        import open_clip
        pretrained = "openai" if backbone == "clip_openai" else "laion2b_s34b_b88k"
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=pretrained)
        model = model.to(DEVICE).eval(); model.visual.output_tokens = True
        def fn(m, x):
            _, t = m.visual(x)
            return t[:, 1:] if t.shape[1] % 2 == 1 else t
        return model, pre, fn
    if backbone == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                               trust_repo=True).to(DEVICE).eval()
        try: bic = T.InterpolationMode.BICUBIC
        except AttributeError: bic = 3
        pre = T.Compose([T.Resize(256, interpolation=bic), T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        def fn(m, x):
            return m.forward_features(x)["x_norm_patchtokens"]
        return model, pre, fn
    raise ValueError(backbone)


@torch.no_grad()
def extract(n_img, backbone="clip", dataset="COCO"):
    path = os.path.join(OUT, f"patch_{dataset}_{backbone}_{n_img}.npz")
    if dataset == "COCO" and backbone == "clip" and not os.path.exists(path):
        legacy = os.path.join(OUT, f"coco_patch_{n_img}.npz")   # reuse old cache
        if os.path.exists(legacy):
            path = legacy
    if os.path.exists(path):
        d = np.load(path); return d["tok"].astype(np.float32), d["Y"], int(d["P"])
    ds, Y, ik = _load_hf(dataset, n_img)
    model, pre, fn = _get_backbone(backbone)
    toks = []
    for x in tqdm(DataLoader(_Imgs(ds, pre, ik), batch_size=64, num_workers=8),
                  desc=f"{backbone}/{dataset} tokens"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = fn(model, x.to(DEVICE))
        toks.append(t.float().cpu().numpy().astype(np.float16))
    tok = np.concatenate(toks); P = tok.shape[1]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"patch_{dataset}_{backbone}_{n_img}.npz"),
                        tok=tok, Y=Y, P=P)
    print(f"[extract] {backbone}/{dataset} tokens {tok.shape} (P={P}), labels {Y.shape}",
          flush=True)
    return tok.astype(np.float32), Y, P


# ── Top-K SAE ────────────────────────────────────────────────────────────────
class SAE(nn.Module):
    def __init__(self, d_in, d_dict, nonneg):
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
        self.nonneg = nonneg
        with torch.no_grad():
            self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)

    def encode(self, x, k):
        c = self.enc(x - self.pre_bias)
        if self.nonneg:
            c = F.relu(c)
        score = c if self.nonneg else c.abs()
        idx = score.topk(k, dim=1).indices
        mask = torch.zeros_like(c).scatter_(1, idx, 1.0)
        return c * mask

    def forward(self, x, k):
        c = self.encode(x, k)
        return self.dec(c) + self.pre_bias, c


def train_sae(Xg, d_dict, nonneg, k, steps=1500, bs=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = SAE(Xg.shape[1], d_dict, nonneg).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator(device=DEVICE).manual_seed(seed); n = len(Xg)
    for _ in range(steps):
        xb = Xg[torch.randint(0, n, (bs,), generator=g, device=DEVICE)]
        xhat, _ = m(xb, k)
        loss = F.mse_loss(xhat, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m.dec.weight.data = F.normalize(m.dec.weight.data, dim=0)
    return m.eval()


@torch.no_grad()
def sae_codes(m, Xg, k, chunk=16384):
    return np.concatenate([m.encode(Xg[i:i+chunk], k).cpu().numpy()
                           for i in range(0, len(Xg), chunk)])


@torch.no_grad()
def sae_nmse(m, Xg, k, var, chunk=16384):
    se = sum(((m(Xg[i:i+chunk], k)[0] - Xg[i:i+chunk])**2).sum().item()
             for i in range(0, len(Xg), chunk))
    return se / Xg.numel() / var


# ── metrics ──────────────────────────────────────────────────────────────────
def mean_L0(C, tol=1e-6):
    return float((np.abs(C) > tol).sum(1).mean())


def coherence(C, tokens_unit, T=40, max_atoms=400):
    d = C.shape[1]
    atoms = (np.random.default_rng(0).choice(d, max_atoms, replace=False)
             if d > max_atoms else np.arange(d))
    Tu = torch.tensor(tokens_unit, device=DEVICE); cohs = []
    for j in atoms:
        top = np.argpartition(-C[:, j], T)[:T]
        if C[top, j].max() <= 0:
            continue
        V = Tu[top]; S = V @ V.T
        cohs.append(((S.sum() - T) / (T*(T-1))).item())
    return float(np.mean(cohs)) if cohs else float("nan")


def purity(C, patch_img, Y, T=40, max_atoms=400):
    d = C.shape[1]
    atoms = (np.random.default_rng(1).choice(d, max_atoms, replace=False)
             if d > max_atoms else np.arange(d))
    ps = []
    for j in atoms:
        top = np.argpartition(-C[:, j], T)[:T]
        if C[top, j].max() <= 0:
            continue
        ps.append(Y[patch_img[top]].sum(0).max() / T)
    return float(np.mean(ps)) if ps else float("nan")


def editability(Cimg, Y, seed=0):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(Cimg))
    n_te = len(Cimg)//2; te, tr = perm[:n_te], perm[n_te:]
    Xt = torch.tensor(Cimg[tr], device=DEVICE); Yt = torch.tensor(Y[tr], device=DEVICE)
    Xe = torch.tensor(Cimg[te], device=DEVICE)
    W = torch.zeros(Cimg.shape[1], Y.shape[1], device=DEVICE, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(Xt@W+b, Yt).backward(); opt.step()
    W_, b_ = W.detach(), b.detach()
    base = (Xe@W_+b_).cpu().numpy()
    cats = [c for c in range(Y.shape[1]) if 5 < Y[te, c].sum() < n_te]
    ap0 = {c: average_precision_score(Y[te, c], base[:, c]) for c in cats}
    tgt, coll = [], []
    for c in cats:
        a = int(W_[:, c].argmax())
        Xa = Xe.clone(); Xa[:, a] = 0
        s = (Xa@W_+b_).cpu().numpy()
        tgt.append(ap0[c] - average_precision_score(Y[te, c], s[:, c]))
        coll.append(np.mean([ap0[c2] - average_precision_score(Y[te, c2], s[:, c2])
                             for c2 in cats if c2 != c]))
    tgt, coll = np.array(tgt), np.array(coll)
    return dict(targeted=float(tgt.mean()), collateral=float(coll.mean()),
                selectivity=float(tgt.mean() / (abs(coll.mean()) + 1e-6)),
                probe_mAP=float(np.mean(list(ap0.values()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "clip_openai", "dinov2"])
    ap.add_argument("--dataset", default="COCO", choices=list(DATASETS))
    ap.add_argument("--n-img", type=int, default=2500)
    ap.add_argument("--dict", type=int, default=1024)
    ap.add_argument("--n-train-tok", type=int, default=150000)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--op-k", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print(f"[cfg] backbone={args.backbone} dataset={args.dataset}", flush=True)

    tok, Y, P = extract(args.n_img, args.backbone, args.dataset); n_img = len(tok)
    X = tok.reshape(-1, tok.shape[-1])
    patch_img = np.repeat(np.arange(n_img), P)
    mean = X.mean(0, keepdims=True); Xc = X - mean
    Xg = torch.tensor(Xc, device=DEVICE)
    tokens_unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(0)
    Xtr = torch.tensor(Xc[rng.choice(len(Xc), min(args.n_train_tok, len(Xc)),
                                     replace=False)], device=DEVICE)
    var = float((Xc**2).mean())
    print(f"[data] {n_img} imgs x {P} = {len(X)} tokens d={X.shape[1]} dict={args.dict} "
          f"op_k={args.op_k}", flush=True)

    def imgpool(C): return C.reshape(n_img, P, -1).max(1)

    res = {}
    for nonneg, name in [(True, "conic-SAE"), (False, "signed-SAE")]:
        pts = []
        for k in args.ks:
            m = train_sae(Xtr, args.dict, nonneg, k)
            pts.append({"k": k, "nmse": sae_nmse(m, Xg, k, var),
                        "L0": mean_L0(sae_codes(m, Xg, k))})
            print(f"  [{name} k={k}] nMSE {pts[-1]['nmse']:.3f} L0 {pts[-1]['L0']:.1f}",
                  flush=True)
        m = train_sae(Xtr, args.dict, nonneg, args.op_k); C = sae_codes(m, Xg, args.op_k)
        res[name] = {"frontier": pts, "nmse": sae_nmse(m, Xg, args.op_k, var),
                     "op_L0": mean_L0(C), "coherence": coherence(C, tokens_unit),
                     "purity": purity(C, patch_img, Y), **editability(imgpool(C), Y)}
        print(f"  [{name}] coher {res[name]['coherence']:.3f} purity {res[name]['purity']:.3f}"
              f" | edit sel {res[name]['selectivity']:.2f} (tgt {res[name]['targeted']:.3f}"
              f" coll {res[name]['collateral']:.4f}) probe mAP {res[name]['probe_mAP']:.3f}",
              flush=True)

    # PCA top-k
    k = args.op_k
    pca = PCA(n_components=min(args.dict, X.shape[1]-1)).fit(Xc[rng.choice(len(Xc), min(args.n_train_tok, len(Xc)), replace=False)])
    Zp = pca.transform(Xc)
    thr = np.sort(np.abs(Zp), axis=1)[:, -k][:, None]
    Csp = np.where(np.abs(Zp) >= thr, Zp, 0.0)
    res["PCA"] = {"nmse": float(((Xc - Csp @ pca.components_)**2).mean()/var), "op_L0": float(k),
                  "coherence": coherence(Csp, tokens_unit), "purity": purity(Csp, patch_img, Y),
                  **editability(imgpool(Csp), Y)}
    print(f"  [PCA k={k}] nMSE {res['PCA']['nmse']:.3f} coher {res['PCA']['coherence']:.3f} "
          f"purity {res['PCA']['purity']:.3f} sel {res['PCA']['selectivity']:.2f} "
          f"mAP {res['PCA']['probe_mAP']:.3f}", flush=True)

    # kmeans (L0=1)
    km = MiniBatchKMeans(n_clusters=args.dict, batch_size=4096, n_init=3,
                         random_state=0).fit(Xc[rng.choice(len(Xc), min(args.n_train_tok, len(Xc)), replace=False)])
    lab = km.predict(Xc); Ck = np.zeros((len(Xc), args.dict), np.float32)
    Ck[np.arange(len(Xc)), lab] = 1.0
    res["kmeans"] = {"nmse": float(((Xc - km.cluster_centers_[lab])**2).mean()/var), "op_L0": 1.0,
                     "coherence": coherence(Ck, tokens_unit), "purity": purity(Ck, patch_img, Y),
                     **editability(imgpool(Ck), Y)}
    print(f"  [kmeans] nMSE {res['kmeans']['nmse']:.3f} coher {res['kmeans']['coherence']:.3f} "
          f"purity {res['kmeans']['purity']:.3f} sel {res['kmeans']['selectivity']:.2f} "
          f"mAP {res['kmeans']['probe_mAP']:.3f}", flush=True)

    tag = f"{args.backbone}_{args.dataset}"
    with open(os.path.join(OUT, f"results_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    _report(res, tag)


def _report(res, tag="clip_COCO"):
    lines = ["# Conic vs signed dictionary on ViT patch tokens (code quality)\n",
             "Matched dict size & sparsity (Top-K). coherence/purity = monosemanticity "
             "(higher better). selectivity = targeted/collateral AP drop on atom ablation "
             "(higher = cleaner concept edits).\n",
             "| method | L0 | nMSE↓ | coherence↑ | purity↑ | edit-sel↑ | targetedΔAP | probe mAP |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for m in ("conic-SAE", "signed-SAE", "PCA", "kmeans"):
        r = res[m]
        lines.append(f"| {m} | {r['op_L0']:.1f} | {r['nmse']:.3f} | {r['coherence']:.3f} | "
                     f"{r['purity']:.3f} | {r['selectivity']:.2f} | {r['targeted']:.3f} | "
                     f"{r['probe_mAP']:.3f} |")
    rep = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, f"report_{tag}.md"), "w") as f:
        f.write(rep)
    print("\n" + rep, flush=True)


if __name__ == "__main__":
    main()
