"""
sae_segmentation_iou.py
-----------------------
Ground-truth concept-localization metric for the conic-vs-signed dictionary claim.

Turns the coherence/purity PROXIES into a reviewer-proof number: how well does an
atom's activation map align with a ground-truth semantic region? (à la Network
Dissection, Bau et al. 2017, but for SAE atoms on ViT patch tokens.)

GT: COCO boxes rasterized through the backbone's resize+crop onto the G×G patch
grid -> per-category binary patch masks. For each category, pick the best atom and
score:
  * localization AP  (threshold-free: rank patches by atom activation)
  * prevalence-IoU   (binarize atom at top-|gt| activations; the emergent-seg number)
Mean over categories, per method (conic-SAE / signed-SAE / PCA / kmeans).

    HF_HUB_OFFLINE=1 python -u sae_segmentation_iou.py --backbone clip --n-img 2500
"""
import argparse, math, os, json
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score
import vit_sae_conic as V

# backbone -> (resize_shortest, crop)
GEOM = {"clip": (224, 224), "clip_openai": (224, 224), "dinov2": (256, 224)}
MIN_POS = 50   # a category needs >= this many GT patches to be scored


def _append_bg(GTflat):
    """Append a background row (patches with no scored class) → (n+1, NP), bg row index."""
    bg = ~GTflat.any(0)
    return np.vstack([GTflat, bg[None]]), GTflat.shape[0]


def build_gt_coco(n_img, P, backbone):
    """(rows, n_img*P) binary patch masks from COCO boxes + a background row."""
    from datasets import load_dataset
    ds = load_dataset("detection-datasets/coco", split="val",
                      cache_dir=os.path.join(V.DATA_DIR, "hf")).select(range(n_img))
    W = np.array(ds["width"]); H = np.array(ds["height"]); objs = ds["objects"]
    n_cls = max((max(o["category"]) for o in objs if o["category"]), default=0) + 1
    G = int(round(math.sqrt(P))); R, C = GEOM[backbone]
    GT = np.zeros((n_cls, n_img, G, G), dtype=bool)
    for i in range(n_img):
        s = R / min(W[i], H[i]); cell = C / G
        ox = (W[i]*s - C) / 2; oy = (H[i]*s - C) / 2
        for (x0, y0, x1, y1), c in zip(objs[i]["bbox"], objs[i]["category"]):
            gx0 = max(0, int((x0*s - ox) / cell)); gx1 = min(G-1, int((x1*s - ox) / cell))
            gy0 = max(0, int((y0*s - oy) / cell)); gy1 = min(G-1, int((y1*s - oy) / cell))
            if gx1 >= gx0 and gy1 >= gy0:
                GT[c, i, gy0:gy1+1, gx0:gx1+1] = True
    GTflat = GT.reshape(n_cls, n_img * G * G)          # all 80 categories (0..79 real)
    GTb, bg = _append_bg(GTflat)
    return GTb, GTb.shape[0], list(range(GTb.shape[0])), bg   # bg = background row idx


def build_gt_ade(n_img, P, backbone):
    """ADE20K (1aurent/ADE20K) dense GT. segmentations[0] is the RGB _seg.png with
    class = (R//10)*256 + G (ADE encoding; verified vs objects.name_ndx). Majority
    class per patch cell, through backbone resize(NEAREST)+crop. Only classes with
    >= MIN_POS patches are scored (open vocab -> most classes are rare). class 0 =
    unlabeled (excluded). Reuses V._load_hf so GT images match extracted tokens."""
    from torchvision import transforms as T
    ds, _, _ = V._load_hf("ADE20K", n_img)           # SAME images/order as extract()
    segs = ds["segmentations"]
    G = int(round(math.sqrt(P))); R, C = GEOM[backbone]; cell = C // G
    try: nn = T.InterpolationMode.NEAREST
    except AttributeError: nn = 0
    mtf = T.Compose([T.Resize(R, interpolation=nn), T.CenterCrop(C)])
    W = 3300                                          # ADE full-vocab class ceiling
    maj = np.zeros((len(segs), G * G), np.int64)
    for i in range(len(segs)):
        s = segs[i]; seg = s[0] if isinstance(s, (list, tuple)) else s
        m = np.array(mtf(seg))                        # (C,C,3) RGB seg codes
        cls = (m[..., 0].astype(np.int64) // 10) * 256 + m[..., 1].astype(np.int64)
        cells = cls.reshape(G, cell, G, cell).transpose(0, 2, 1, 3).reshape(G*G, cell*cell)
        hist = np.zeros((G * G, W), np.int32)
        np.add.at(hist, (np.arange(G * G)[:, None], np.clip(cells, 0, W - 1)), 1)
        maj[i] = hist.argmax(1)
    flat = maj.reshape(-1)
    counts = np.bincount(flat, minlength=W)
    fg = [c for c in range(1, W) if counts[c] >= MIN_POS]          # foreground classes
    GTfg = np.zeros((len(fg), len(flat)), bool)
    for j, c in enumerate(fg):
        GTfg[j] = (flat == c)
    GT, bg = _append_bg(GTfg)              # background row = patches whose majority is class 0
    print(f"[gt-ade] {len(fg)} fg classes >= {MIN_POS} px (+1 background row)", flush=True)
    return GT, GT.shape[0], list(range(GT.shape[0])), bg


def localize(GTb, C, score_cats):
    """Per category: best atom by correlation, then AP + prevalence-IoU.
    Returns parallel lists (aps, ious, cats) so callers can split with/without bg."""
    Gt = torch.tensor(GTb.astype(np.float32), device=V.DEVICE)     # (rows, NP)
    Ct = torch.tensor(C, device=V.DEVICE)                          # (NP, dict)
    corr = Gt @ Ct
    best = corr.argmax(1).cpu().numpy()
    aps, ious, cats = [], [], []
    for c in score_cats:
        pos = int(GTb[c].sum())
        if pos < MIN_POS:
            continue
        a = int(best[c]); sc = C[:, a]
        aps.append(average_precision_score(GTb[c], sc))
        top = np.argpartition(-sc, pos)[:pos]
        mask = np.zeros(len(sc), bool); mask[top] = True
        inter = (mask & GTb[c]).sum(); union = (mask | GTb[c]).sum()
        ious.append(inter / max(union, 1)); cats.append(c)
    return np.array(aps), np.array(ious), cats


def split_means(aps, ious, cats, bg):
    """means with background (all cats) and without (excluding the bg row)."""
    keep = np.array([c != bg for c in cats])
    return {"with_bg_AP": float(aps.mean()), "with_bg_IoU": float(ious.mean()),
            "no_bg_AP": float(aps[keep].mean()), "no_bg_IoU": float(ious[keep].mean()),
            "n_cats": len(cats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "clip_openai", "dinov2"])
    ap.add_argument("--dataset", default="COCO", choices=["COCO", "ADE20K"])
    ap.add_argument("--n-img", type=int, default=2500)
    ap.add_argument("--dict", type=int, default=1024)
    ap.add_argument("--op-k", type=int, default=16)
    ap.add_argument("--n-train-tok", type=int, default=150000)
    args = ap.parse_args()

    tok, _, P = V.extract(args.n_img, args.backbone, args.dataset); n_img = len(tok)
    X = tok.reshape(-1, tok.shape[-1]); Xc = X - X.mean(0, keepdims=True)
    Xg = torch.tensor(Xc, device=V.DEVICE)
    rng = np.random.default_rng(0)
    tr = rng.choice(len(Xc), min(args.n_train_tok, len(Xc)), replace=False)
    Xtr = torch.tensor(Xc[tr], device=V.DEVICE)
    gt_fn = build_gt_ade if args.dataset == "ADE20K" else build_gt_coco
    GTb, n_rows, score_cats, bg = gt_fn(n_img, P, args.backbone)
    print(f"[gt] {args.dataset}/{args.backbone} P={P} G={int(round(math.sqrt(P)))} "
          f"rows={n_rows} (bg row={bg}); scoring cats with >= {MIN_POS}px", flush=True)

    def run(name, C):
        aps, ious, cats = localize(GTb, C, score_cats)
        r = split_means(aps, ious, cats, bg)
        print(f"  [{name}] with-bg AP {r['with_bg_AP']:.3f}/IoU {r['with_bg_IoU']:.3f} | "
              f"no-bg AP {r['no_bg_AP']:.3f}/IoU {r['no_bg_IoU']:.3f} ({r['n_cats']} cats)",
              flush=True)
        return r

    res = {}
    for nonneg, name in [(True, "conic-SAE"), (False, "signed-SAE")]:
        m = V.train_sae(Xtr, args.dict, nonneg, args.op_k)
        res[name] = run(name, V.sae_codes(m, Xg, args.op_k))

    pca = PCA(n_components=min(args.dict, X.shape[1]-1)).fit(Xc[tr])
    res["PCA"] = run("PCA", pca.transform(Xc).astype(np.float32))

    km = MiniBatchKMeans(n_clusters=args.dict, batch_size=4096, n_init=3,
                         random_state=0).fit(Xc[tr])
    lab = km.predict(Xc); Ck = np.zeros((len(Xc), args.dict), np.float32)
    Ck[np.arange(len(Xc)), lab] = 1.0
    res["kmeans"] = run("kmeans", Ck)

    os.makedirs(V.OUT, exist_ok=True)
    with open(os.path.join(V.OUT, f"segiou_{args.dataset}_{args.backbone}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n| method | with-bg AP/IoU | without-bg AP/IoU |\n|---|--:|--:|")
    for m in ("conic-SAE", "signed-SAE", "PCA", "kmeans"):
        r = res[m]
        print(f"| {m} | {r['with_bg_AP']:.3f} / {r['with_bg_IoU']:.3f} | "
              f"{r['no_bg_AP']:.3f} / {r['no_bg_IoU']:.3f} |")


if __name__ == "__main__":
    main()
