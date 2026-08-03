"""
coco_multilabel_law.py
----------------------
Measure the "cone niche law" directly: on real multi-label COCO, does the
NNLS conic-hull decoder's advantage over a linear probe (and multi-centroid NCM)
GROW with the number of co-present objects per image?

Prediction (from the additive-closure argument): delta ≈ 0 at 1 label, rising
with co-label density, and steeper for a less-entangled patch representation than
for the CLS token. If the curve slopes up, cones have a real niche (compositional
multi-label) and we've quantified when. If flat even on patch tokens, the
parts-based story dies too.

Decoders (per-class score on an L2-normalised image feature; all multi-label):
  * linear   : linear probe (BCE-trained 80-way), score = logit
  * nnls-cone: training-free k-NN NNLS — x ≈ Σ a_i·atom_i (a≥0 over k nearest
               labelled train atoms); score_c = Σ_i a_i · Y_train[i,c]
               (labels propagate additively through the nonneg conic code)
  * ncm-multi: multi-centroid NCM — per class, spherical-k-means on its positive
               images; score_c = max cos(x, that class's centroids)

Features (two entanglement levels of the SAME CLIP ViT-B/16):
  * CLS   : projected image embedding (512-d) — global, entangled
  * Patch : mean of patch tokens (768-d) — spatially uniform, less entangled

Metric: mAP (mean average precision over 80 classes), computed overall and on
test subsets stratified by #labels/image.

Usage
-----
    python -u coco_multilabel_law.py                # full run
    python -u coco_multilabel_law.py --knn 100 --ncm-protos 4
"""
import argparse
import io
import json
import os

import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DATA_DIR = "./data"
OUT_DIR = "./coco_law_out"
CACHE = os.path.join(OUT_DIR, "coco_clip_feats.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_coco():
    from datasets import load_dataset
    ds = load_dataset("detection-datasets/coco", split="val",
                      cache_dir=os.path.join(DATA_DIR, "hf"))
    return ds


def build_label_matrix(ds):
    """(N, C) binary multi-label matrix from objects.category."""
    cats = []
    n_classes = 0
    for r in ds:
        c = r["objects"]["category"]
        cats.append(set(int(x) for x in c))
        if c:
            n_classes = max(n_classes, max(int(x) for x in c) + 1)
    Y = np.zeros((len(cats), n_classes), np.float32)
    for i, s in enumerate(cats):
        for c in s:
            Y[i, c] = 1.0
    return Y


class _CocoImgs(Dataset):
    def __init__(self, ds, preprocess):
        self.ds, self.pre = ds, preprocess

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i]["image"]
        if not hasattr(img, "mode"):
            from PIL import Image
            img = Image.open(io.BytesIO(img["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.pre(img)


@torch.no_grad()
def extract_features(ds):
    """Return (cls [N,512], patch [N,768]) CLIP features, cached."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["cls"].astype(np.float32), d["patch"].astype(np.float32)
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.to(DEVICE).eval()
    model.visual.output_tokens = True     # forward returns (pooled, tokens)

    loader = DataLoader(_CocoImgs(ds, preprocess), batch_size=128,
                        shuffle=False, num_workers=8, pin_memory=True)
    CLS, PATCH = [], []
    for x in tqdm(loader, desc="extract CLIP", unit="batch"):
        x = x.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled, tokens = model.visual(x)
        # tokens: (B, seq, width). Drop CLS token if present, mean-pool patches.
        if tokens.shape[1] % 2 == 1:      # odd => includes a leading CLS token
            tokens = tokens[:, 1:]
        CLS.append(pooled.float().cpu().numpy())
        PATCH.append(tokens.float().mean(1).cpu().numpy())
    cls = np.concatenate(CLS).astype(np.float32)
    patch = np.concatenate(PATCH).astype(np.float32)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(CACHE, cls=cls, patch=patch)
    print(f"[feat] cls {cls.shape}  patch {patch.shape}  cached {CACHE}")
    return cls, patch


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Decoders — each returns test scores (N_test, C), higher = class present
# ─────────────────────────────────────────────────────────────────────────────
def linear_probe(Xtr, Ytr, Xte, steps=800, lr=1e-2):
    Xt = torch.tensor(Xtr, device=DEVICE)
    Yt = torch.tensor(Ytr, device=DEVICE)
    W = torch.zeros(Xtr.shape[1], Ytr.shape[1], device=DEVICE, requires_grad=True)
    b = torch.zeros(Ytr.shape[1], device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(Xt @ W + b, Yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        return (torch.tensor(Xte, device=DEVICE) @ W + b).cpu().numpy()


def knn_nnls(Xtr, Ytr, Xte, k=100, chunk=512):
    """Training-free conic decoder: nonneg reconstruction over k nearest atoms,
    labels propagated additively through the coefficients."""
    A = torch.tensor(Xtr, device=DEVICE)          # (Ntr, D) unit
    scores = np.zeros((len(Xte), Ytr.shape[1]), np.float32)
    for i in range(0, len(Xte), chunk):
        Q = torch.tensor(Xte[i:i + chunk], device=DEVICE)
        sims = Q @ A.T                            # (b, Ntr)
        idx = sims.topk(k, dim=1).indices.cpu().numpy()
        Qn = Xte[i:i + chunk]
        for j in range(len(Qn)):
            nn_idx = idx[j]
            B = Xtr[nn_idx].T                     # (D, k)
            a, _ = nnls(B, Qn[j])                 # a >= 0
            scores[i + j] = a @ Ytr[nn_idx]       # (C,) additive label transfer
    return scores


def ncm_multi(Xtr, Ytr, Xte, n_protos=4, iters=30, seed=0):
    """Multi-centroid NCM: per class, spherical-k-means on its positive images;
    score_c = max cos to that class's centroids."""
    rng = np.random.default_rng(seed)
    Xte_t = torch.tensor(Xte, device=DEVICE)
    C = Ytr.shape[1]
    scores = np.zeros((len(Xte), C), np.float32)
    for c in range(C):
        pos = Xtr[Ytr[:, c] > 0]
        if len(pos) == 0:
            scores[:, c] = -1.0
            continue
        m = min(n_protos, len(pos))
        cen = pos[rng.choice(len(pos), m, replace=False)].copy()
        for _ in range(iters):
            a = np.argmax(pos @ cen.T, axis=1)
            newc = cen.copy()
            for j in range(m):
                p = pos[a == j]
                if len(p):
                    s = p.sum(0); newc[j] = s / (np.linalg.norm(s) + 1e-8)
            if np.allclose(newc, cen):
                break
            cen = newc
        cen_t = torch.tensor(cen, device=DEVICE)
        scores[:, c] = (Xte_t @ cen_t.T).max(1).values.cpu().numpy()
    return scores


def mAP(scores, Y):
    """Mean AP over classes with >=1 positive in this Y."""
    aps = []
    for c in range(Y.shape[1]):
        if Y[:, c].sum() > 0 and Y[:, c].sum() < len(Y):
            aps.append(average_precision_score(Y[:, c], scores[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", type=int, default=100)
    ap.add_argument("--ncm-protos", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_coco()
    Y = build_label_matrix(ds)
    cls, patch = extract_features(ds)
    n_lab = Y.sum(1).astype(int)
    print(f"[coco] {len(Y)} imgs, {Y.shape[1]} classes, "
          f"labels/img: mean {n_lab.mean():.2f} median {np.median(n_lab):.0f} "
          f"max {n_lab.max()}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Y))
    n_te = int(len(Y) * args.test_frac)
    te, tr = perm[:n_te], perm[n_te:]
    Ytr, Yte = Y[tr], Y[te]
    nlab_te = n_lab[te]

    feats = {"CLS": _unit(cls), "Patch": _unit(patch)}
    decoders = {
        "linear": lambda Xtr, Xte: linear_probe(Xtr, Ytr, Xte),
        "nnls-cone": lambda Xtr, Xte: knn_nnls(Xtr, Ytr, Xte, k=args.knn),
        "ncm-multi": lambda Xtr, Xte: ncm_multi(Xtr, Ytr, Xte, n_protos=args.ncm_protos),
    }

    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
    results = {}          # feat -> decoder -> {"overall":mAP, "by_n":{label:mAP}}
    for fname, Fall in feats.items():
        Xtr, Xte = Fall[tr], Fall[te]
        results[fname] = {}
        for dname, fn in decoders.items():
            S = fn(Xtr, Xte)
            overall = mAP(S, Yte)
            by_n = {}
            for lo, hi in bins:
                mask = (nlab_te >= lo) & (nlab_te <= hi)
                key = f"{lo}" if lo == hi else f"{lo}+"
                by_n[key] = mAP(S[mask], Yte[mask]) if mask.sum() > 5 else float("nan")
            results[fname][dname] = {"overall": overall, "by_n": by_n,
                                     "n": {f"{lo}" if lo == hi else f"{lo}+":
                                           int(((nlab_te >= lo) & (nlab_te <= hi)).sum())
                                           for lo, hi in bins}}
            print(f"[{fname}/{dname}] overall mAP {overall:.4f} | by #labels "
                  + "  ".join(f"{k}:{v:.3f}" for k, v in by_n.items()))

    _report(results, args, bins)


def _report(results, args, bins):
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    keys = [f"{lo}" if lo == hi else f"{lo}+" for lo, hi in bins]

    lines = ["# COCO multi-label: does the cone (NNLS) delta grow with #labels/image?\n",
             "mAP by #labels/image (test subset). Δ rows = decoder − linear probe. "
             "Prediction: NNLS−linear rises with #labels, steeper for Patch than CLS.\n"]
    for fname in results:
        lines.append(f"\n## {fname}\n")
        lines.append("| decoder | overall | " + " | ".join(f"n={k}" for k in keys) + " |")
        lines.append("|---|--:|" + "--:|" * len(keys))
        for d in ("linear", "nnls-cone", "ncm-multi"):
            r = results[fname][d]
            row = [f"{r['by_n'][k]:.3f}" for k in keys]
            lines.append(f"| {d} | {r['overall']:.3f} | " + " | ".join(row) + " |")
        # delta rows
        lin = results[fname]["linear"]["by_n"]
        for d in ("nnls-cone", "ncm-multi"):
            bn = results[fname][d]["by_n"]
            row = [f"{bn[k]-lin[k]:+.3f}" for k in keys]
            ov = results[fname][d]["overall"] - results[fname]["linear"]["overall"]
            lines.append(f"| **Δ {d}−linear** | {ov:+.3f} | " + " | ".join(row) + " |")
    lines.append("\n(n counts per bin: " + ", ".join(
        f"{k}={results[list(results)[0]]['linear']['n'][k]}" for k in keys) + ")\n")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print("\n" + report)

    _plot(results, keys)
    print(f"[done] wrote {OUT_DIR}/report.md and delta_vs_labels.png")


def _plot(results, keys):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})")
        return
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"CLS": "--", "Patch": "-"}
    colors = {"nnls-cone": "#2c7fb8", "ncm-multi": "#d95f0e"}
    for fname in results:
        lin = results[fname]["linear"]["by_n"]
        for d in ("nnls-cone", "ncm-multi"):
            bn = results[fname][d]["by_n"]
            y = [bn[k] - lin[k] for k in keys]
            ax.plot(x, y, styles[fname], marker="o", color=colors[d],
                    label=f"{d}−linear ({fname})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"n={k}" for k in keys])
    ax.set_xlabel("# labels / image"); ax.set_ylabel("Δ mAP vs linear probe")
    ax.set_title("Cone/NCM advantage vs co-label density (COCO, CLIP)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_vs_labels.png"), dpi=130)


if __name__ == "__main__":
    main()
