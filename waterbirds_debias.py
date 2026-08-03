"""
waterbirds_debias.py
--------------------
Debiasing by CONCEPT-ATOM ABLATION on Waterbirds (worst-group accuracy).

Story: a linear probe on frozen features leans on the spurious *background* (place).
We learn a Top-K SAE concept dictionary on the features, find the atom(s) that encode
'place', zero them, and retrain the probe on the cleaned features. The non-negative
(conic) code should give a SURGICAL edit — big worst-group gain, small clean-accuracy
cost — that a signed code / dense linear removal can't match.

Metric: worst-group accuracy (min over the 4 (label,place) groups) + average acc.
Model selection (how many atoms/directions to remove) on VALIDATION worst-group; report TEST.

Methods:
  ERM              linear probe on raw features (biased baseline)
  INLP-lite        remove top-r linear 'place'-predictive directions, then probe
  conic / signed   remove top-k 'place' atoms from the SAE code, decode, probe

    HF_HUB_OFFLINE=1 python -u waterbirds_debias.py --backbone clip
"""
import argparse, io, json, os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import vit_sae_conic as V

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEVICE = V.DEVICE
OUT = "./waterbirds_out"


class _Imgs(Dataset):
    def __init__(self, ds, pre): self.ds, self.pre = ds, pre
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img = self.ds[i]["image"]
        if not hasattr(img, "mode"):
            img = __import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(img["bytes"]))
        return self.pre(img.convert("RGB"))


@torch.no_grad()
def extract_cls(backbone):
    """Return dict split -> (feats[N,D], y[N], a[N]).  Global (CLS) embeddings."""
    path = os.path.join(OUT, f"waterbirds_{backbone}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {s: (d[f"{s}_f"], d[f"{s}_y"], d[f"{s}_a"]) for s in ("train", "validation", "test")}
    from datasets import load_dataset
    dd = load_dataset("grodino/waterbirds", cache_dir=os.path.join(V.DATA_DIR, "hf"))
    if backbone in ("clip", "clip_openai"):
        import open_clip
        pre_w = "openai" if backbone == "clip_openai" else "laion2b_s34b_b88k"
        model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained=pre_w)
        model = model.to(DEVICE).eval()
        enc = lambda x: model.encode_image(x)
    else:
        from torchvision import transforms as T
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                               trust_repo=True).to(DEVICE).eval()
        bic = T.InterpolationMode.BICUBIC
        pre = T.Compose([T.Resize(256, interpolation=bic), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        enc = lambda x: model(x)
    out, save = {}, {}
    for s in ("train", "validation", "test"):
        ds = dd[s]
        F = []
        for x in tqdm(DataLoader(_Imgs(ds, pre), batch_size=128, num_workers=8), desc=f"{backbone}/{s}"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                F.append(enc(x.to(DEVICE)).float().cpu().numpy())
        F = np.concatenate(F).astype(np.float32)
        y = np.array(ds["label"]); a = np.array(ds["place"])
        out[s] = (F, y, a); save[f"{s}_f"], save[f"{s}_y"], save[f"{s}_a"] = F, y, a
    os.makedirs(OUT, exist_ok=True); np.savez_compressed(path, **save)
    return out


def group_acc(pred, y, a):
    """avg acc + worst-group acc over the 4 (y,a) groups."""
    g = 2 * y + a
    accs = [(pred[g == k] == y[g == k]).mean() for k in range(4) if (g == k).any()]
    return float((pred == y).mean()), float(min(accs))


def probe(Ftr, ytr, Fte):
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Ftr, ytr)
    return lr.predict(Fte)


def inlp_remove(Ftr, atr, Fva, Fte, r):
    """Remove top-r linear 'place'-predictive directions (INLP-lite)."""
    Ftr, Fva, Fte = Ftr.copy(), Fva.copy(), Fte.copy()
    for _ in range(r):
        w = LogisticRegression(max_iter=1000).fit(Ftr, atr).coef_[0]
        w = w / (np.linalg.norm(w) + 1e-8)
        for F in (Ftr, Fva, Fte):
            F -= np.outer(F @ w, w)
    return Ftr, Fva, Fte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "clip_openai", "dinov2"])
    ap.add_argument("--dict", type=int, default=512)
    ap.add_argument("--op-k", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    data = extract_cls(args.backbone)
    (Ftr, ytr, atr) = data["train"]; (Fva, yva, ava) = data["validation"]; (Fte, yte, ate) = data["test"]
    mean = Ftr.mean(0, keepdims=True)
    print(f"[data] train {Ftr.shape} val {Fva.shape} test {Fte.shape}", flush=True)
    res = {}

    # ERM
    a_, w_ = group_acc(probe(Ftr, ytr, Fte), yte, ate)
    res["ERM"] = {"avg": a_, "worst": w_}
    print(f"  [ERM] avg {a_*100:.1f}  worst {w_*100:.1f}", flush=True)

    # INLP-lite (select r on val worst-group)
    best = (-1, None)
    for r in (1, 2, 4, 8):
        Ftr2, Fva2, _ = inlp_remove(Ftr, atr, Fva, Fte, r)
        _, wv = group_acc(probe(Ftr2, ytr, Fva2), yva, ava)
        if wv > best[0]: best = (wv, r)
    Ftr2, _, Fte2 = inlp_remove(Ftr, atr, Fva, Fte, best[1])
    a_, w_ = group_acc(probe(Ftr2, ytr, Fte2), yte, ate)
    res["INLP-lite"] = {"avg": a_, "worst": w_, "r": best[1]}
    print(f"  [INLP-lite r={best[1]}] avg {a_*100:.1f}  worst {w_*100:.1f}", flush=True)

    # SAE concept-atom ablation
    Xtr = torch.tensor(Ftr - mean, device=DEVICE)
    Xte = torch.tensor(Fte - mean, device=DEVICE)
    Xva = torch.tensor(Fva - mean, device=DEVICE)
    for nonneg, name in [(True, "conic-SAE"), (False, "signed-SAE")]:
        m = V.train_sae(Xtr, args.dict, nonneg, args.op_k)
        Ctr = V.sae_codes(m, Xtr, args.op_k); Cva = V.sae_codes(m, Xva, args.op_k)
        Cte = V.sae_codes(m, Xte, args.op_k)
        # rank atoms by |logistic weight| predicting place
        aw = np.abs(LogisticRegression(max_iter=1000).fit(Ctr, atr).coef_[0])
        order = np.argsort(-aw)
        W = m.dec.weight.detach().cpu().numpy()      # (D, dict)

        def decode(C, kill):
            Cc = C.copy(); Cc[:, order[:kill]] = 0.0
            return (Cc @ W.T) + mean               # back to feature space

        best = (-1, None)
        for kill in (0, 1, 2, 3, 5, 8, 15):
            a_, w_ = group_acc(probe(decode(Ctr, kill), ytr, decode(Cva, kill)), yva, ava)
            if w_ > best[0]: best = (w_, kill)
        kill = best[1]
        a_, w_ = group_acc(probe(decode(Ctr, kill), ytr, decode(Cte, kill)), yte, ate)
        # full recon (no ablation) control
        af, wf = group_acc(probe(decode(Ctr, 0), ytr, decode(Cte, 0)), yte, ate)
        res[name] = {"avg": a_, "worst": w_, "kill": kill,
                     "recon_avg": af, "recon_worst": wf}
        print(f"  [{name}] kill={kill}  avg {a_*100:.1f}  worst {w_*100:.1f}  "
              f"(recon-only worst {wf*100:.1f})", flush=True)

    with open(os.path.join(OUT, f"debias_{args.backbone}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n| method | avg acc | worst-group acc |\n|---|--:|--:|")
    for m in ("ERM", "INLP-lite", "signed-SAE", "conic-SAE"):
        print(f"| {m} | {res[m]['avg']*100:.1f} | {res[m]['worst']*100:.1f} |")


if __name__ == "__main__":
    main()
