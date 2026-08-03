"""
splice_cifar.py
---------------
SpLiCE-style faithfulness curve on CIFAR-100: accuracy retained when a CLIP image
embedding is replaced by a SPARSE decomposition, as a function of sparsity (mean L0).
Zero-shot eval (reconstruct embedding -> cosine to class-name text).

Three sparse representations, matched sparsity (exact L0=k), plus the full-CLIP ceiling:
  SpLiCE      : per-image top-k NNLS over a fixed CLIP-TEXT concept dictionary (training-free)
  conic-SAE   : Top-K ReLU (non-negative) LEARNED dictionary (ours)
  signed-SAE  : Top-K, no ReLU (ablation)

Positions our learned non-negative dictionary directly against SpLiCE on its own
CIFAR-100 faithfulness metric, and tests the conic-vs-signed thesis on accuracy@sparsity.

    HF_HUB_OFFLINE=1 python -u splice_cifar.py
"""
import os, json
import numpy as np
import torch
from scipy.optimize import nnls
import vit_sae_conic as V

for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEVICE = V.DEVICE
OUT = "./splice_out"
KS = [1, 2, 4, 8, 16, 32]

# curated concept words (materials/colors/textures/scenes/parts/objects/animals) to
# augment dataset class names -> the SpLiCE text dictionary.
CURATED = """red green blue yellow orange purple pink brown black white gray metal wood
plastic glass leather fur feather scale skin stone brick concrete fabric cloth paper
striped spotted furry smooth rough shiny matte round square curved sharp soft hard wet dry
sky cloud grass tree forest mountain water sea ocean river beach sand snow ice road street
building house room kitchen sky field farm garden indoor outdoor night day sunset shadow
head eye ear nose mouth leg arm wing tail paw wheel window door roof handle engine
animal bird fish insect flower plant food fruit vegetable vehicle furniture tool
person face hand body hair clothing shoe hat glasses machine screen keyboard bottle cup
chair table sofa bed lamp book box bag ball toy clock mirror sign light fire smoke""".split()


def get_clip():
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.to(DEVICE).eval()
    return model, pre, open_clip.get_tokenizer("ViT-B-16")


@torch.no_grad()
def text_emb(model, tok, prompts, chunk=256):
    E = []
    for i in range(0, len(prompts), chunk):
        E.append(model.encode_text(tok(prompts[i:i+chunk]).to(DEVICE)).float().cpu().numpy())
    e = np.concatenate(E)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


@torch.no_grad()
def extract_cifar(model, pre):
    path = os.path.join(OUT, "cifar100_clip.npz")
    if os.path.exists(path):
        d = np.load(path); return d["ftr"], d["ytr"], d["fte"], d["yte"], list(d["names"])
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader
    out = {}
    names = CIFAR100("./data", train=False, download=False).classes
    for split, tr in (("tr", True), ("te", False)):
        ds = CIFAR100("./data", train=tr, download=False, transform=pre)
        F, Y = [], []
        for x, y in DataLoader(ds, batch_size=256, num_workers=8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                F.append(model.encode_image(x.to(DEVICE)).float().cpu().numpy())
            Y.append(np.asarray(y))
        out[f"f{split}"] = np.concatenate(F).astype(np.float32)
        out[f"y{split}"] = np.concatenate(Y)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, ftr=out["ftr"], ytr=out["ytr"], fte=out["fte"],
                        yte=out["yte"], names=np.array(names))
    return out["ftr"], out["ytr"], out["fte"], out["yte"], names


def concept_words(class_names):
    from torchvision import datasets as D
    words = set(CURATED) | {n.replace("_", " ") for n in class_names}
    for fn in (lambda: D.CIFAR10("./data", train=False, download=False).classes,
               lambda: D.Food101("./data", split="test", download=False).classes,
               lambda: D.OxfordIIITPet("./data", split="test", download=False).classes,
               lambda: D.STL10("./data", split="test", download=False).classes):
        try: words |= {w.replace("_", " ").replace("-", " ") for w in fn()}
        except Exception: pass
    return sorted({w.lower().strip() for w in words if w.strip()})


def zshot(feats, class_txt):
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return (f @ class_txt.T).argmax(1)


def acc(pred, y): return float((pred == y).mean())


def splice_topk(Fte, concepts, class_txt, k):
    """top-k NNLS over the text concept dictionary; reconstruct; zero-shot."""
    U = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-8)
    Ct = torch.tensor(concepts, device=DEVICE)
    idx_all = (torch.tensor(U, device=DEVICE) @ Ct.T).topk(k, 1).indices.cpu().numpy()
    rec = np.zeros_like(U)
    for i in range(len(U)):
        ci = idx_all[i]; w, _ = nnls(concepts[ci].T, U[i]); rec[i] = w @ concepts[ci]
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    model, pre, tok = get_clip()
    Ftr, ytr, Fte, yte, names = extract_cifar(model, pre)
    class_txt = text_emb(model, tok, [f"a photo of a {n.replace('_', ' ')}" for n in names])
    vocab = concept_words(names)
    concepts = text_emb(model, tok, vocab)
    print(f"[data] train {Ftr.shape} test {Fte.shape} | {len(vocab)} concepts", flush=True)

    ceil = acc(zshot(Fte, class_txt), yte)
    print(f"[full-CLIP zero-shot] acc {ceil*100:.1f}", flush=True)

    res = {"full_clip": ceil, "SpLiCE": {}, "conic-SAE": {}, "signed-SAE": {}}
    mean = Ftr.mean(0, keepdims=True)
    Xtr = torch.tensor(Ftr - mean, device=DEVICE); Xte = torch.tensor(Fte - mean, device=DEVICE)
    W = None
    for k in KS:
        # SpLiCE
        rec = splice_topk(Fte, concepts, class_txt, k)
        res["SpLiCE"][k] = acc(zshot(rec, class_txt), yte)
        # SAEs (train per k, dict 2048 over-complete)
        for nonneg, name in [(True, "conic-SAE"), (False, "signed-SAE")]:
            m = V.train_sae(Xtr, 2048, nonneg, k)
            C = V.sae_codes(m, Xte, k)
            Wd = m.dec.weight.detach().cpu().numpy()
            rec_s = C @ Wd.T + mean
            res[name][k] = acc(zshot(rec_s, class_txt), yte)
        print(f"  k={k:2d}  SpLiCE {res['SpLiCE'][k]*100:.1f}  "
              f"conic {res['conic-SAE'][k]*100:.1f}  signed {res['signed-SAE'][k]*100:.1f}",
              flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nfull-CLIP zero-shot ceiling: {ceil*100:.1f}")
    print("| L0 (k) | SpLiCE | conic-SAE | signed-SAE |\n|--:|--:|--:|--:|")
    for k in KS:
        print(f"| {k} | {res['SpLiCE'][k]*100:.1f} | {res['conic-SAE'][k]*100:.1f} | "
              f"{res['signed-SAE'][k]*100:.1f} |")
    _plot(res, ceil)


def _plot(res, ceil):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skip ({e})"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in [("SpLiCE", "#d95f0e"), ("conic-SAE", "#2c7fb8"), ("signed-SAE", "#7fbf7b")]:
        ax.plot(KS, [res[name][k]*100 for k in KS], "-o", color=c, label=name)
    ax.axhline(ceil*100, ls="--", color="gray", label="full CLIP")
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.set_xlabel("active concepts (L0)"); ax.set_ylabel("CIFAR-100 zero-shot acc (%)")
    ax.set_title("Faithfulness–sparsity frontier (CIFAR-100)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "faithfulness.png"), dpi=130)
    print(f"[done] {OUT}/faithfulness.png")


if __name__ == "__main__":
    main()
