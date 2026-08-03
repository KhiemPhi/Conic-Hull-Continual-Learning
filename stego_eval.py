"""
stego_eval.py
-------------
STEGO-protocol unsupervised semantic segmentation on COCO-Stuff-27.

Cluster dense patch representations into K=27, Hungarian-match clusters to the 27
coarse GT classes, report mIoU + pixel-accuracy (STEGO/PiCIE/CAUSE protocol). Our
contribution enters as the *representation being clustered*:
    raw        : L2-normalized frozen patch tokens          (STEGO-style baseline)
    conic-SAE  : NON-NEGATIVE sparse code (Top-K ReLU)       (ours)
    signed-SAE : signed sparse code (same arch, no ReLU)     (ablation)
Thesis: the non-negative code clusters into more semantically-pure regions -> higher
Hungarian mIoU, at matched dict/sparsity.

Data: shunk031/cocostuff (HF). Fine labels (COCO-Stuff 182, 255=unlabeled) are
merged to the 27 coarse STEGO classes by `fine_to_coarse` (see FINALIZE note).

    HF_HUB_OFFLINE=1 python -u stego_eval.py --backbone dinov2 --n-img 2000

NOTE: `fine_to_coarse` is finalized after a schema probe of shunk031/cocostuff
(label indexing / 255 handling). Until then it prefers an external STEGO json.
"""
import argparse, io, json, math, os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import vit_sae_conic as V

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEVICE = V.DEVICE
OUT = "./stego_out"
GEOM = {"clip": (224, 224), "clip_openai": (224, 224), "dinov2": (256, 224)}
N_COARSE = 27
IGNORE = 255


# ── fine (182) -> coarse (27) mapping ────────────────────────────────────────
def load_fine_to_coarse():
    """Return int array [n_fine] -> coarse in [0,26] or -1 (ignore).
    Prefers STEGO's official json (drop `cocostuff_fine_to_coarse.json` in cwd);
    else builds from the COCO-Stuff supercategory grouping (FINALIZE after probe)."""
    p = "cocostuff_fine_to_coarse.json"
    if os.path.exists(p):
        d = json.load(open(p))
        m = d.get("fine_index_to_coarse_index", d)
        arr = np.full(max(int(k) for k in m) + 1, -1, int)
        for k, v in m.items():
            arr[int(k)] = int(v)
        print(f"[map] loaded {p}", flush=True)
        return arr
    raise SystemExit(
        "Need the 182->27 mapping. After you download shunk031/cocostuff I'll probe "
        "its label schema and generate cocostuff_fine_to_coarse.json (STEGO grouping)."
    )


# ── data ─────────────────────────────────────────────────────────────────────
class _Imgs(Dataset):
    def __init__(self, ds, pre, ik): self.ds, self.pre, self.ik = ds, pre, ik
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img = self.ds[i][self.ik]
        if not hasattr(img, "mode"):
            from PIL import Image; img = Image.open(io.BytesIO(img["bytes"]))
        return self.pre(img.convert("RGB"))


def load_cocostuff(n_img):
    from datasets import load_dataset
    dd = load_dataset("shunk031/cocostuff", cache_dir=os.path.join(V.DATA_DIR, "hf"))
    sp = "validation" if "validation" in dd else ("val" if "val" in dd else list(dd)[0])
    ds = dd[sp]
    if n_img and n_img < len(ds):
        ds = ds.select(range(n_img))
    cols = ds.column_names
    ik = next(c for c in ("image", "img", "Image") if c in cols)
    mk = next(c for c in ("label", "annotation", "mask", "segmentation",
                          "panoptic_seg_map", "stuff_map", "sem_seg") if c in cols)
    return ds, ik, mk


@torch.no_grad()
def extract(ds, ik, backbone, n_img):
    path = os.path.join(OUT, f"feats_cocostuff_{backbone}_{n_img}.npz")
    if os.path.exists(path):
        d = np.load(path); return d["tok"].astype(np.float32), int(d["P"])
    model, pre, fn = V._get_backbone(backbone)
    toks = []
    for x in tqdm(DataLoader(_Imgs(ds, pre, ik), batch_size=64, num_workers=8),
                  desc=f"{backbone} tokens"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = fn(model, x.to(DEVICE))
        toks.append(t.float().cpu().numpy().astype(np.float16))
    tok = np.concatenate(toks); P = tok.shape[1]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, tok=tok, P=P)
    return tok.astype(np.float32), P


def build_coarse_gt(ds, mk, n_img, P, backbone, f2c):
    """Majority coarse-class per patch cell (255/ignore excluded from majority)."""
    from torchvision import transforms as T
    masks = ds[mk]
    G = int(round(math.sqrt(P))); R, C = GEOM[backbone]; cell = C // G
    try: nn = T.InterpolationMode.NEAREST
    except AttributeError: nn = 0
    mtf = T.Compose([T.Resize(R, interpolation=nn), T.CenterCrop(C)])
    gt = np.full((len(masks), G * G), -1, np.int64)
    for i in range(len(masks)):
        m = np.array(mtf(masks[i]))
        if m.ndim == 3: m = m[..., 0]
        coarse = np.where(m < len(f2c), f2c[np.clip(m, 0, len(f2c) - 1)], -1)
        coarse[m == IGNORE] = -1
        cells = coarse.reshape(G, cell, G, cell).transpose(0, 2, 1, 3).reshape(G*G, cell*cell)
        for j in range(G * G):
            v = cells[j][cells[j] >= 0]
            gt[i, j] = np.bincount(v).argmax() if len(v) else -1
    return gt.reshape(-1)


# ── STEGO cluster metric ─────────────────────────────────────────────────────
def hungarian_miou(pred, gt, k=N_COARSE):
    """pred: cluster ids [N]; gt: coarse ids [N] (-1 ignore). 1-1 Hungarian on the
    confusion matrix, then mIoU + pixel-acc over the k classes."""
    valid = gt >= 0
    pred, gt = pred[valid], gt[valid]
    conf = np.zeros((k, k), np.int64)              # rows=cluster, cols=gt
    np.add.at(conf, (pred, gt), 1)
    r, c = linear_sum_assignment(-conf)            # maximize matched mass
    remap = {ri: ci for ri, ci in zip(r, c)}
    mapped = np.array([remap.get(p, -1) for p in pred])
    ious, accs = [], (mapped == gt).mean()
    for cls in range(k):
        inter = ((mapped == cls) & (gt == cls)).sum()
        union = ((mapped == cls) | (gt == cls)).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)), float(accs)


def cluster_rep(rep, k, n_train=200000, seed=0):
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(rep), min(n_train, len(rep)), replace=False)
    km = MiniBatchKMeans(k, batch_size=4096, n_init=5, random_state=seed).fit(rep[tr])
    return km.predict(rep)


def build_gt_ade(n_img, P, backbone, topk):
    """ADE20K (1aurent) STEGO GT: decode seg RGB -> full-vocab class = (R//10)*256+G,
    majority per patch, keep the top-`topk` most frequent classes as the label set
    (others/unlabeled -> ignore). Returns (gt[-1 ignore], k)."""
    from torchvision import transforms as T
    ds, _, _ = V._load_hf("ADE20K", n_img)
    segs = ds["segmentations"]
    G = int(round(math.sqrt(P))); R, C = GEOM[backbone]; cell = C // G
    try: nn = T.InterpolationMode.NEAREST
    except AttributeError: nn = 0
    mtf = T.Compose([T.Resize(R, interpolation=nn), T.CenterCrop(C)])
    W = 3300
    maj = np.zeros((len(segs), G * G), np.int64)
    for i in range(len(segs)):
        s = segs[i]; seg = s[0] if isinstance(s, (list, tuple)) else s
        m = np.array(mtf(seg))
        cls = (m[..., 0].astype(np.int64) // 10) * 256 + m[..., 1].astype(np.int64)
        cells = cls.reshape(G, cell, G, cell).transpose(0, 2, 1, 3).reshape(G*G, cell*cell)
        hist = np.zeros((G * G, W), np.int32)
        np.add.at(hist, (np.arange(G * G)[:, None], np.clip(cells, 0, W - 1)), 1)
        maj[i] = hist.argmax(1)
    flat = maj.reshape(-1)
    counts = np.bincount(flat, minlength=W); counts[0] = 0        # drop unlabeled
    top = np.argsort(-counts)[:topk]
    remap = np.full(W, -1, np.int64)
    for j, c in enumerate(top):
        remap[c] = j
    print(f"[gt-ade] top-{topk} classes cover "
          f"{counts[top].sum()/max(1,(flat>0).sum())*100:.0f}% of labeled patches", flush=True)
    return remap[flat], topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ADE20K", choices=["ADE20K", "COCOStuff"])
    ap.add_argument("--backbone", default="dinov2", choices=["clip", "clip_openai", "dinov2"])
    ap.add_argument("--n-img", type=int, default=2000)
    ap.add_argument("--dict", type=int, default=1024)
    ap.add_argument("--op-k", type=int, default=16)
    ap.add_argument("--topk-classes", type=int, default=27,
                    help="ADE: keep top-K frequent classes (27 mirrors COCO-Stuff-27)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.dataset == "ADE20K":
        tok, _, P = V.extract(args.n_img, args.backbone, "ADE20K")
        n_img = len(tok)
        gt, K = build_gt_ade(n_img, P, args.backbone, args.topk_classes)
    else:                                          # COCOStuff (needs non-script source)
        f2c = load_fine_to_coarse()
        ds, ik, mk = load_cocostuff(args.n_img)
        tok, P = extract(ds, ik, args.backbone, len(ds)); n_img = len(tok)
        gt, K = build_coarse_gt(ds, mk, n_img, P, args.backbone, f2c), N_COARSE

    X = tok.reshape(-1, tok.shape[-1]); Xc = X - X.mean(0, keepdims=True)
    Xg = torch.tensor(Xc, device=DEVICE)
    print(f"[data] {args.dataset}/{args.backbone} {n_img} imgs P={P} K={K} | "
          f"GT valid patches {int((gt>=0).sum())}/{len(gt)}", flush=True)

    reps = {"raw": X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)}
    rng = np.random.default_rng(0)
    tr = torch.tensor(Xc[rng.choice(len(Xc), min(150000, len(Xc)), replace=False)], device=DEVICE)
    for nonneg, name in [(True, "conic-SAE"), (False, "signed-SAE")]:
        m = V.train_sae(tr, args.dict, nonneg, args.op_k)
        reps[name] = V.sae_codes(m, Xg, args.op_k)

    res = {}
    for name, rep in reps.items():
        miou, acc = hungarian_miou(cluster_rep(rep, K), gt, K)
        res[name] = {"mIoU": miou, "pixel_acc": acc}
        print(f"  [{name}] mIoU {miou*100:.1f}  pixelAcc {acc*100:.1f}", flush=True)

    with open(os.path.join(OUT, f"stego_{args.dataset}_{args.backbone}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n| representation | mIoU | pixelAcc |\n|---|--:|--:|")
    for n in ("raw", "conic-SAE", "signed-SAE"):
        print(f"| {n} | {res[n]['mIoU']*100:.1f} | {res[n]['pixel_acc']*100:.1f} |")


if __name__ == "__main__":
    main()
