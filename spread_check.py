#!/usr/bin/env python3
"""spread_check.py — does per-class prototype uncertainty have STRUCTURE a scalar ridge misses?

WHY
    Every top PTM-CIL method (SLCA, SSIAT, MACIL, LoDA+CA, GR-LoRA) stores per-class
    (mu_c, Sigma_c) and samples Gaussian pseudo-features to retrain the classifier. Those
    statistics are estimated from N_c samples in d=768 dimensions -- often N_c < d.

    A robust (SOCP) head would replace sampling with a worst-case program over uncertainty
    ellipsoids of radius kappa_c ~ sqrt(d/N_c). But ridge regression is ALREADY the exact
    robust counterpart of least squares under *spherical* uncertainty (El Ghaoui & Lebret).
    So the SOCP can only buy something if the uncertainty is STRUCTURED -- varying per class
    in a way one scalar lambda cannot express.

    This script measures whether that structure exists, before anyone writes a solver.

THE TWO GATES
    The uncertainty radius of a class prototype is its standard error:

        s_c = sqrt( tr(Sigma_c) / N_c )

    (kappa_c * RMS action of Sigma_c^1/2 on a random unit w reduces to exactly this.)

    GATE A -- MATERIALITY.  s_c / d_NN, where d_NN is the median nearest-neighbour distance
    between prototypes. If the uncertainty ball is tiny next to the gap between classes, it
    cannot flip a decision and robustness buys nothing however structured it is.
        median(s_c / d_NN) >= 0.20  -> material

    GATE B -- STRUCTURE.  CV(s_c) = std(s_c)/mean(s_c) across classes. If every class carries
    the same uncertainty, a single scalar ridge already expresses it.
        CV(s_c) >= 0.25  -> structured

    BOTH must pass for a robust per-class head to have a target on that dataset. Thresholds
    are pre-registered here so the read is not chosen after seeing the numbers.

PROTOCOL
    Frozen backbone, L2-normalised features -- the space the head actually operates in
    (matches `un()` in exp8_combined.py). Train split only. No training, no adaptation.

USAGE
    source ~/venvs/ml_env/bin/activate
    python spread_check.py                 # all datasets
    python spread_check.py CUB200 IMAGENETA
    MODEL=... override the backbone.
"""

import os
import sys
import json
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "vit_base_patch16_224.augreg_in21k")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 1993          # same 80/20 convention as crux_headroom.py:112
MIN_N = 5                  # below this, Sigma_c is not estimable at all
MATERIAL_T = 0.20
STRUCTURE_T = 0.25


# ----------------------------------------------------------------- data
class HFWrap(Dataset):
    """HuggingFace split -> (tensor, label), tolerating grayscale/CMYK sources."""

    def __init__(self, ds, tf, labels):
        self.ds, self.tf, self.labels = ds, tf, labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), self.labels[i]


def load_split(name, tf):
    """Return (dataset, labels) for the TRAIN portion of `name`."""
    from datasets import load_dataset
    from torchvision import datasets as tvd

    if name == "CIFAR100":
        ds = tvd.CIFAR100(os.path.join(REPO, "data"), train=True, download=False, transform=tf)
        return ds, np.array(ds.targets)

    if name == "AIRCRAFT":
        ds = tvd.FGVCAircraft(os.path.join(REPO, "data"), split="train",
                              download=False, transform=tf)
        return ds, np.array([s[1] for s in ds._image_files and ds.samples]) \
            if hasattr(ds, "samples") else (ds, np.array(ds._labels))

    if name == "IMAGENETR":
        # single 'test' split, string wnid labels -> integer ids, then the 80/20 convention
        d = load_dataset("axiong/imagenet-r", cache_dir=os.path.join(REPO, "data/hf"))["test"]
        wn = d["wnid"]
        uniq = {w: i for i, w in enumerate(sorted(set(wn)))}
        lab = np.array([uniq[w] for w in wn])
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(lab))
        tr = perm[: int(0.8 * len(lab))]
        return HFWrap(d.select(tr.tolist()), tf, lab[tr]), lab[tr]

    spec = {
        "IMAGENETA": ("barkermrl/imagenet-a", "train", True),
        "CUB200": ("Donghyun99/cub-200-2011", "train", False),
        "CARS": ("Donghyun99/stanford-cars", "train", False),
    }[name]
    repo, split, needs_split = spec
    d = load_dataset(repo, cache_dir=os.path.join(REPO, "data/hf"))[split]
    lab = np.array(d["label"])
    if needs_split:                       # only one split exists -> carve a train portion
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(lab))
        tr = perm[: int(0.8 * len(lab))]
        return HFWrap(d.select(tr.tolist()), tf, lab[tr]), lab[tr]
    return HFWrap(d, tf, lab), lab


# ----------------------------------------------------------------- features
@torch.no_grad()
def extract(model, ds, bs=256):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            out.append(model(x.to(DEV, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def l2(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------- the check
def analyse(name, F, y):
    """Per-class prototype uncertainty vs class separation."""
    d = F.shape[1]
    classes = np.unique(y)
    mu, s, eff, Nc = [], [], [], []
    for c in classes:
        Z = F[y == c]
        n = len(Z)
        m = Z.mean(0)
        R = Z - m
        # tr(Sigma) and tr(Sigma^2) without forming Sigma when n < d
        tr1 = float((R * R).sum() / max(n - 1, 1))
        Gm = R @ R.T / max(n - 1, 1)            # n x n, same nonzero spectrum as Sigma
        tr2 = float((Gm * Gm).sum())
        mu.append(m)
        Nc.append(n)
        s.append(np.sqrt(tr1 / n))              # standard error of the prototype
        eff.append(tr1 * tr1 / (tr2 + 1e-30))   # participation ratio = effective rank
    mu = np.stack(mu)
    s = np.array(s)
    Nc = np.array(Nc)
    eff = np.array(eff)

    # nearest-neighbour prototype distance -- the scale a decision boundary lives on
    D = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    d_nn = np.median(D.min(1))

    # A class with N_c=1 has an identically-zero residual, so s_c=0 -- an artifact, not low
    # uncertainty (its true uncertainty is unbounded). Scoring the gates on those classes
    # would manufacture spread. Report them, then measure structure on estimable classes only.
    ok = Nc >= MIN_N
    n_degen = int((Nc < 2).sum())
    s_ok = s[ok] if ok.sum() > 1 else s
    ratio = s[ok] / d_nn if ok.sum() > 1 else s / d_nn
    cv = float(s_ok.std() / (s_ok.mean() + 1e-30))
    cv_all = float(s.std() / (s.mean() + 1e-30))
    s = s_ok
    return dict(
        dataset=name, n_classes=int(len(classes)), d=int(d), n_total=int(len(y)),
        Nc_min=int(Nc.min()), Nc_med=float(np.median(Nc)), Nc_max=int(Nc.max()),
        Nc_cv=float(Nc.std() / Nc.mean()),
        Nc_over_d=float(np.median(Nc) / d),
        eff_rank_med=float(np.median(eff)),
        n_degen=n_degen, n_estimable=int((Nc >= MIN_N).sum()),
        s_med=float(np.median(s)), s_cv=cv, s_cv_all=cv_all,
        s_ratio_max_min=float(s.max() / (s.min() + 1e-30)),
        d_nn=float(d_nn),
        material=float(np.median(ratio)), material_p90=float(np.percentile(ratio, 90)),
        pass_A=bool(np.median(ratio) >= MATERIAL_T),
        pass_B=bool(cv >= STRUCTURE_T),
    )


ALL = ["CIFAR100", "IMAGENETR", "IMAGENETA", "CUB200", "CARS"]


def main():
    want = [a.upper() for a in sys.argv[1:]] or ALL
    import timm
    from timm.data import resolve_data_config, create_transform

    os.chdir(REPO)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().to(DEV)
    tf = create_transform(**resolve_data_config({}, model=model))
    print(f"backbone : {MODEL}  ({model.num_features}-d)")
    print(f"gates    : materiality median(s_c/d_NN) >= {MATERIAL_T}   "
          f"structure CV(s_c) >= {STRUCTURE_T}\n")

    rows = []
    for name in want:
        try:
            ds, y = load_split(name, tf)
            F = l2(extract(model, ds))
            r = analyse(name, F, y)
            rows.append(r)
            print(f"  {name:10s} n={r['n_total']:6d} C={r['n_classes']:3d} "
                  f"N_c[{r['Nc_min']}/{r['Nc_med']:.0f}/{r['Nc_max']}] "
                  f"s_med={r['s_med']:.4f} CV={r['s_cv']:.3f} "
                  f"mat={r['material']:.3f}  A={'Y' if r['pass_A'] else 'n'} "
                  f"B={'Y' if r['pass_B'] else 'n'}", flush=True)
        except Exception as e:
            print(f"  {name:10s} ERR {type(e).__name__}: {str(e)[:110]}", flush=True)

    if not rows:
        return
    print("\n" + "=" * 104)
    print("PROTOTYPE-UNCERTAINTY SPREAD  (frozen backbone, L2-normalised, train split)")
    print("=" * 104)
    print(f"{'dataset':<11}{'C':>4}{'N_c med':>9}{'N_c/d':>8}{'N_c CV':>8}"
          f"{'effrank':>9}{'s_med':>9}{'CV(s)':>8}{'CV all':>8}{'degen':>7}"
          f"{'s/d_NN':>9}{'p90':>8}  gates")
    for r in rows:
        g = ("A" if r["pass_A"] else "-") + ("B" if r["pass_B"] else "-")
        verdict = "TARGET" if (r["pass_A"] and r["pass_B"]) else ""
        print(f"{r['dataset']:<11}{r['n_classes']:>4}{r['Nc_med']:>9.0f}"
              f"{r['Nc_over_d']:>8.2f}{r['Nc_cv']:>8.3f}{r['eff_rank_med']:>9.1f}"
              f"{r['s_med']:>9.4f}{r['s_cv']:>8.3f}{r['s_cv_all']:>8.3f}{r['n_degen']:>7d}"
              f"{r['material']:>9.3f}{r['material_p90']:>8.3f}  {g} {verdict}")
    print("-" * 104)
    print("N_c/d  < 1  -> Sigma_c is rank-deficient; the stored covariance is a guess")
    print("A (materiality) : uncertainty ball vs gap to the nearest class. fails -> "
          "robustness cannot flip a decision")
    print("B (structure)   : does s_c vary across classes? fails -> scalar ridge already "
          "expresses it, SOCP adds nothing")
    print("=" * 104)

    out = os.path.join(REPO, "spread_check.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
