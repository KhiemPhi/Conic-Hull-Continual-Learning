"""
czsl_conic.py
-------------
Compositional Zero-Shot Learning on UT-Zappos (XuVV/ut-zappos-rl) with a CONIC
compositional bottleneck.

Each example: images[0] + answer="Material,ShoeType" (e.g. "Leather,Shoes.Oxfords").
Parse -> (attr, obj). Standard CZSL split: test has 18/36 UNSEEN (attr,obj) pairs
not in train. Represent an image as NON-NEGATIVE primitive activations (attr atoms +
obj atoms); a composition is scored ADDITIVELY. Primitives trained per-primitive
across many pairs -> compose to unseen pairs a monolithic classifier can't score.

Ablation isolating our thesis: conic = ReLU (non-negative) primitive activations;
signed = same head, no ReLU. Prediction: conic >= signed on UNSEEN acc / AUC
(signed lets attributes 'cancel', breaking additive composition).

Metrics (closed-world, CGE protocol): best-seen, best-unseen, harmonic mean, AUC
as a calibration bias on unseen pairs is swept.

    HF_HUB_OFFLINE=1 python -u czsl_conic.py --backbone clip
"""
import argparse, io, json, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import vit_sae_conic as V

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEVICE = V.DEVICE
OUT = "./czsl_out"
REPO = "XuVV/ut-zappos-rl"


class _Imgs(Dataset):
    def __init__(self, ds, pre, ik): self.ds, self.pre, self.ik = ds, pre, ik
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img = self.ds[i][self.ik]
        if isinstance(img, (list, tuple)): img = img[0]
        if not hasattr(img, "mode"):
            from PIL import Image; img = Image.open(io.BytesIO(img["bytes"]))
        return self.pre(img.convert("RGB"))


def _clip_or_dino(backbone):
    if backbone in ("clip", "clip_openai"):
        import open_clip
        pw = "openai" if backbone == "clip_openai" else "laion2b_s34b_b88k"
        model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained=pw)
        model = model.to(DEVICE).eval()
        return model, pre, (lambda x: model.encode_image(x))
    from torchvision import transforms as T
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(DEVICE).eval()
    pre = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return model, pre, (lambda x: model(x))


@torch.no_grad()
def extract(backbone):
    """Return (data {split->(F,attr_ids,obj_ids)}, splits, n_attr, n_obj)."""
    from datasets import load_dataset
    dd = load_dataset(REPO, cache_dir=os.path.join(V.DATA_DIR, "hf"))
    splits = list(dd)
    ik = "images" if "images" in dd[splits[0]].column_names else "image"
    parsed = {s: [tuple(a.split(",", 1)) for a in dd[s]["answer"]] for s in splits}
    attrs = sorted({a for s in splits for (a, _) in parsed[s]})
    objs = sorted({o for s in splits for (_, o) in parsed[s]})
    a2i = {a: i for i, a in enumerate(attrs)}; o2i = {o: i for i, o in enumerate(objs)}
    print(f"[vocab] {len(attrs)} attrs x {len(objs)} objs", flush=True)

    path = os.path.join(OUT, f"utzappos_{backbone}.npz")
    if os.path.exists(path):
        d = np.load(path); feats = {s: d[f"{s}_f"] for s in splits}
    else:
        model, pre, enc = _clip_or_dino(backbone); feats, save = {}, {}
        for s in splits:
            Fs = []
            for x in tqdm(DataLoader(_Imgs(dd[s], pre, ik), batch_size=128, num_workers=8), desc=s):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Fs.append(enc(x.to(DEVICE)).float().cpu().numpy())
            feats[s] = np.concatenate(Fs).astype(np.float32); save[f"{s}_f"] = feats[s]
        os.makedirs(OUT, exist_ok=True); np.savez_compressed(path, **save)
    data = {s: (feats[s],
                np.array([a2i[a] for (a, _) in parsed[s]]),
                np.array([o2i[o] for (_, o) in parsed[s]])) for s in splits}
    return data, splits, len(attrs), len(objs)


class Factored(nn.Module):
    def __init__(self, d, n_attr, n_obj, nonneg, hid=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.attr = nn.Linear(hid, n_attr); self.obj = nn.Linear(hid, n_obj)
        self.nonneg = nonneg

    def marginals(self, x):
        h = self.proj(x); ma, mo = self.attr(h), self.obj(h)
        return (F.relu(ma), F.relu(mo)) if self.nonneg else (ma, mo)

    def pair_scores(self, x, pairs):
        ma, mo = self.marginals(x)
        return ma[:, pairs[:, 0]] + mo[:, pairs[:, 1]]


def czsl_metrics(scores, gt_pairs, seen_mask):
    is_unseen = ~seen_mask
    gt_seen = seen_mask[gt_pairs]
    biases = np.linspace(-scores.std() * 3, scores.std() * 3, 60)
    sa_l, ua_l = [], []
    for b in biases:
        pred = (scores + b * is_unseen[None, :]).argmax(1)
        correct = pred == gt_pairs
        sa_l.append(correct[gt_seen].mean() if gt_seen.any() else 0.0)
        ua_l.append(correct[~gt_seen].mean() if (~gt_seen).any() else 0.0)
    sa, ua = np.array(sa_l), np.array(ua_l)
    order = np.argsort(sa)
    hm = 2 * sa * ua / (sa + ua + 1e-8)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return dict(best_seen=float(sa.max()), best_unseen=float(ua.max()),
                best_HM=float(hm.max()), AUC=float(trapz(ua[order], sa[order])))


def train_eval(data, splits, n_attr, n_obj, nonneg, name, epochs=30, lr=1e-3):
    tr = "train" if "train" in splits else splits[0]
    te = "test" if "test" in splits else splits[-1]
    Ftr, atr, otr = data[tr]; Fte, ate, ote = data[te]
    seen = sorted(set(zip(atr.tolist(), otr.tolist())))
    pair_list = sorted({p for s in splits for p in
                        zip(data[s][1].tolist(), data[s][2].tolist())})   # closed world
    pidx = {p: i for i, p in enumerate(pair_list)}
    seen_set = set(seen)
    seen_mask = np.array([p in seen_set for p in pair_list])
    pairs_t = torch.tensor(pair_list, device=DEVICE)
    seen_local = {p: i for i, p in enumerate(seen)}
    seen_pair_ids = torch.tensor([pidx[p] for p in seen], device=DEVICE)
    tr_local = np.array([seen_local[(a, o)] for a, o in zip(atr, otr)])

    Xtr = torch.tensor(Ftr, device=DEVICE); yat = torch.tensor(atr, device=DEVICE)
    yot = torch.tensor(otr, device=DEVICE); ytl = torch.tensor(tr_local, device=DEVICE)
    Xte = torch.tensor(Fte, device=DEVICE)

    m = Factored(Ftr.shape[1], n_attr, n_obj, nonneg).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    n, bs = len(Xtr), 512
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            ma, mo = m.marginals(Xtr[idx])
            ps = ma[:, pairs_t[seen_pair_ids, 0]] + mo[:, pairs_t[seen_pair_ids, 1]]
            loss = (F.cross_entropy(ma, yat[idx]) + F.cross_entropy(mo, yot[idx])
                    + F.cross_entropy(ps, ytl[idx]))
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        scores = np.concatenate([m.pair_scores(Xte[i:i+4096], pairs_t).cpu().numpy()
                                 for i in range(0, len(Xte), 4096)])
    gt = np.array([pidx[(a, o)] for a, o in zip(ate, ote)])
    r = czsl_metrics(scores, gt, seen_mask)
    r.update(n_seen=len(seen), n_unseen_test=int((~seen_mask[gt]).sum()),
             n_pairs=len(pair_list))
    print(f"  [{name}] seen {r['best_seen']*100:.1f}  unseen {r['best_unseen']*100:.1f}  "
          f"HM {r['best_HM']*100:.1f}  AUC {r['AUC']*100:.1f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "clip_openai", "dinov2"])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data, splits, n_attr, n_obj = extract(args.backbone)
    print(f"[data] {[(s, len(data[s][0])) for s in splits]}", flush=True)
    res = {}
    for nonneg, name in [(False, "signed"), (True, "conic")]:
        res[name] = train_eval(data, splits, n_attr, n_obj, nonneg, name, epochs=args.epochs)
    with open(os.path.join(OUT, f"czsl_{args.backbone}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n| model | seen | unseen | HM | AUC |\n|---|--:|--:|--:|--:|")
    for n in ("signed", "conic"):
        r = res[n]
        print(f"| {n} | {r['best_seen']*100:.1f} | {r['best_unseen']*100:.1f} | "
              f"{r['best_HM']*100:.1f} | {r['AUC']*100:.1f} |")


if __name__ == "__main__":
    main()
