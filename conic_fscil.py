"""
conic_fscil.py  (Build B)
-------------------------
Few-Shot Class-Incremental Learning, EXEMPLAR-FREE. The conic hull is used only as a
WITHIN-SESSION feature AUGMENTER of the given few-shot data (not stored memory), to
repair RanPAC's data-poor novel-class estimate. Nothing from past sessions is kept
except aggregate statistics (RanPAC's Gram + class sums) -- same category as RanPAC.

Protocol: base session (many samples) -> RanPAC (random proj + Gram-decorrelated
prototypes). Each novel session: k-way s-shot; for each novel class synthesize S
in-support features and fold them into the Gram + prototype.

Methods (base session identical; only novel-class augmentation differs):
  vanilla : use the s real shots only              (RanPAC FSCIL baseline)
  gaussian: resample from shot mean + diag var     (control: any augmentation?)
  conic   : non-negative combinations of the shots (conic hull of the few-shot data)

Metrics: per-session overall acc, final base/novel acc, harmonic mean.

    HF_HUB_OFFLINE=1 python -u conic_fscil.py --dataset CUB200
    HF_HUB_OFFLINE=1 python -u conic_fscil.py --dataset CIFAR100
"""
import argparse, io, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import timm
from conic_hull import ConicHull

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./fscil_out"
MODEL = "vit_base_patch16_224"
LAM = 1e3
M = 4096                                          # RanPAC projection dim
S_SYNTH = 100                                     # synthetic features per novel class

FSCIL = {"CIFAR100": dict(base=60, ways=5, shots=5, sessions=8),
         "CUB200":   dict(base=100, ways=10, shots=5, sessions=10)}


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


class _HF(Dataset):
    def __init__(self, ds, tf, ik, lk): self.ds, self.tf, self.ik, self.lk = ds, tf, ik, lk
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[i]; img = r[self.ik]
        if not hasattr(img, "mode"):
            from PIL import Image; img = Image.open(io.BytesIO(img["bytes"]))
        return self.tf(img.convert("RGB")), int(r[self.lk])


@torch.no_grad()
def load_feats(dataset):
    """Return (Ftr, ytr, Fte, yte). Frozen ViT-B/16 features."""
    path = os.path.join(OUT, f"{dataset}_feats.npz")
    if dataset == "CIFAR100" and not os.path.exists(path):
        d = np.load("./ranpac_out/cifar100_feats.npz")
        return d["ftr"], d["ytr"], d["fte"], d["yte"]
    if os.path.exists(path):
        d = np.load(path); return d["ftr"], d["ytr"], d["fte"], d["yte"]
    # extract CUB from HF
    from datasets import load_dataset
    dd = load_dataset("Donghyun99/CUB-200-2011", cache_dir="./data/hf")
    cfg = timm.data.resolve_data_config({}, model=timm.create_model(MODEL))
    tf = timm.data.create_transform(**cfg, is_training=False)
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(DEVICE).eval()
    out = {}
    for split, key in (("train", "tr"), ("test", "te")):
        sp = split if split in dd else ("validation" if "validation" in dd else list(dd)[0])
        ds = dd[sp]; cols = ds.column_names
        ik = next(c for c in ("image", "img", "Image") if c in cols)
        lk = next(c for c in ("label", "labels", "fine_label", "class", "target") if c in cols)
        F, Y = [], []
        for x, y in tqdm(DataLoader(_HF(ds, tf, ik, lk), batch_size=256, num_workers=8), desc=split):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                F.append(model(x.to(DEVICE)).float().cpu().numpy())
            Y.append(np.asarray(y))
        out[f"f{key}"] = np.concatenate(F).astype(np.float32); out[f"y{key}"] = np.concatenate(Y)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, ftr=out["ftr"], ytr=out["ytr"], fte=out["fte"], yte=out["yte"])
    return out["ftr"], out["ytr"], out["fte"], out["yte"]


def synth_conic(shots, S, seed):
    """Non-negative combinations of the (unit) few-shot features + small noise."""
    rng = np.random.default_rng(seed)
    sh = unit(shots); k = len(sh)
    a = rng.dirichlet(np.ones(k), size=S).astype(np.float32)      # (S,k) nonneg, sum 1
    x = a @ sh + 0.02 * rng.standard_normal((S, sh.shape[1])).astype(np.float32)
    return unit(x)


def synth_gauss(shots, S, seed):
    rng = np.random.default_rng(seed)
    mu = shots.mean(0); sd = shots.std(0) + 1e-3
    return unit(mu[None] + sd[None] * rng.standard_normal((S, shots.shape[1])).astype(np.float32))


def run(method, Ftr, ytr, Fte, yte, order, cfg, seed=0):
    D = Ftr.shape[1]; rng = np.random.default_rng(0)
    W = (rng.standard_normal((D, M)) / np.sqrt(D)).astype(np.float32)
    proj = lambda X: np.maximum(unit(X) @ W, 0)
    base, ways, shots, sess = cfg["base"], cfg["ways"], cfg["shots"], cfg["sessions"]
    G = np.zeros((M, M), np.float32); csum, ccount = {}, {}; seen = []
    per_sess = []
    srng = np.random.default_rng(seed)
    sessions = [order[:base]] + [order[base+i*ways:base+(i+1)*ways] for i in range(sess)]
    for si, cls in enumerate(sessions):
        seen += list(cls)
        for c in cls:
            idx = np.where(ytr == c)[0]
            if si == 0:                                   # base: all samples
                feat = Ftr[idx]
            else:                                         # novel: s-shot + augmentation
                srng.shuffle(idx); sh = Ftr[idx[:shots]]
                if method == "vanilla":
                    feat = sh
                elif method == "gaussian":
                    feat = np.concatenate([sh, synth_gauss(sh, S_SYNTH, seed*97+c)])
                else:  # conic
                    feat = np.concatenate([sh, synth_conic(sh, S_SYNTH, seed*97+c)])
            phi = proj(feat)
            G += phi.T @ phi
            csum[c] = phi.sum(0); ccount[c] = len(phi)
        Ginv = np.linalg.inv(G + LAM * np.eye(M, dtype=np.float32))
        labels = sorted(seen)
        Wc = np.stack([Ginv @ (csum[c] / ccount[c]) for c in labels])
        mt = np.isin(yte, seen); phite = proj(Fte[mt]); yt = yte[mt]
        pred = np.array(labels)[(phite @ Wc.T).argmax(1)]
        per_sess.append(float((pred == yt).mean()))
    # final base vs novel
    base_cls = set(order[:base].tolist())
    mb = np.isin(yte, list(base_cls)); mn = np.isin(yte, [c for c in seen if c not in base_cls])
    phf = proj(Fte); allpred = np.array(sorted(seen))[(phf @ Wc.T).argmax(1)]
    ba = float((allpred[mb] == yte[mb]).mean())
    na = float((allpred[mn] == yte[mn]).mean()) if mn.any() else 0.0
    hm = 2*ba*na/(ba+na+1e-8)
    return dict(per_session=[round(a*100, 1) for a in per_sess],
                avg=float(np.mean(per_sess)*100), last=float(per_sess[-1]*100),
                base_acc=ba*100, novel_acc=na*100, HM=hm*100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CUB200", choices=["CUB200", "CIFAR100"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = load_feats(args.dataset)
    ncls = int(max(ytr.max(), yte.max()) + 1); cfg = FSCIL[args.dataset]
    print(f"[fscil] {args.dataset} classes {ncls} | base {cfg['base']} + "
          f"{cfg['sessions']}x({cfg['ways']}-way {cfg['shots']}-shot)", flush=True)
    res = {}
    for method in ("vanilla", "gaussian", "conic"):
        runs = [run(method, Ftr, ytr, Fte, yte,
                    np.random.default_rng(1000+s).permutation(ncls), cfg, seed=s)
                for s in range(args.seeds)]
        res[method] = {k: float(np.mean([r[k] for r in runs]))
                       for k in ("avg", "last", "base_acc", "novel_acc", "HM")}
        print(f"  [{method:8s}] avg {res[method]['avg']:.1f}  last {res[method]['last']:.1f}  "
              f"base {res[method]['base_acc']:.1f}  novel {res[method]['novel_acc']:.1f}  "
              f"HM {res[method]['HM']:.1f}", flush=True)
    with open(os.path.join(OUT, f"fscil_{args.dataset}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n| method | avg | last | base | novel | HM |\n|---|--:|--:|--:|--:|--:|")
    for m in ("vanilla", "gaussian", "conic"):
        r = res[m]
        print(f"| {m} | {r['avg']:.1f} | {r['last']:.1f} | {r['base_acc']:.1f} | "
              f"{r['novel_acc']:.1f} | {r['HM']:.1f} |")


if __name__ == "__main__":
    main()
