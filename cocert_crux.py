"""
cocert_crux.py
--------------
Crux + follow-up controls for the "conic certificate" (CoCert) idea: does a
NON-NEGATIVE dictionary (your ConicHull extreme rays) split a novel embedding
into (a) a faithful "made of known parts" reconstruction and (b) a residual
that isolates the novel-specific signal — and is that split cone-specific?

Frozen backbone -> image embeddings. Classes split into base (known) + novel
(OOV). Fit ONE ConicHull on base-train features -> K extreme rays = the
non-negative dictionary R. Decompose each test point q (unit-norm) three ways,
sharing the SAME atoms where possible:
    nnls   : w>=0  (the cone)   -> explained = w@R,  r = q - explained
    signed : same R, no >=0     -> explained_s,      r_s = q - explained_s
    pca    : rank-K subspace     -> explained_p,      r_p = q - explained_p

CRUX (on novel points): cluster raw / explained / residual (KMeans k=#novel),
score vs true novel labels (ARI primary; NMI saturates). Claim = QUARANTINE
DIFFERENTIAL: cone gap (residual-explained) >> signed gap.

FOLLOW-UP CONTROLS
  step 1  reconstruction fraction  ‖explained‖/‖q‖ for novel vs base — guards
          against the trivial "cone can't fit novel -> residual≈input".
  step 2  --k-sweep (dictionary size) and --backbone {vit,clip,dinov2}.
  step 3  nameability:
            (a) attribution consistency — do same-novel-class images get the
                same known-part attribution? cluster base-class-attribution
                histograms, ARI vs novel labels (nnls vs signed).
            (b) semantic coherence — does the top attributed known part share
                the novel class's COARSE group? (CIFAR-100 20 superclasses;
                CUB families parsed from names) vs a random-attribution chance.

    HF_HUB_OFFLINE=1 python -u cocert_crux.py --dataset CIFAR100 --seeds 3
    HF_HUB_OFFLINE=1 python -u cocert_crux.py --dataset CUB200   --seeds 3
    HF_HUB_OFFLINE=1 python -u cocert_crux.py --dataset CIFAR100 --k-sweep 64 128 256 512
    HF_HUB_OFFLINE=1 python -u cocert_crux.py --dataset CUB200   --backbone clip
"""
import argparse, json, os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                             roc_auc_score)
from sklearn.preprocessing import normalize
from tqdm import tqdm

from conic_hull import ConicHull

os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _v in ("http_proxy", "https_proxy"):
    os.environ.setdefault(_v, "http://fwdproxy:8080")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT, DATA_DIR = "./cocert_out", "./data"

# backbone -> cache tag. "vit" tag matches the original cache so it is reused.
BACKBONES = {"vit": "vit_base_patch16_224", "clip": "clip_vitb16_laion",
             "dinov2": "dinov2_vitb14"}


# ── backbones (image-level embeddings) ───────────────────────────────────────
def _backbone_setup(backbone):
    """Return (model, transform, embed_fn). embed_fn(model, xb) -> (B, D)."""
    if backbone == "vit":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True,
                                  num_classes=0).to(DEVICE).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        tf = timm.data.create_transform(**cfg, is_training=False)
        return model, tf, (lambda m, x: m(x))
    if backbone == "clip":
        import open_clip
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k")
        model = model.to(DEVICE).eval()
        return model, pre, (lambda m, x: m.encode_image(x))
    if backbone == "dinov2":
        from torchvision import transforms as T
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                               trust_repo=True).to(DEVICE).eval()
        try: bic = T.InterpolationMode.BICUBIC
        except AttributeError: bic = 3
        tf = T.Compose([T.Resize(256, interpolation=bic), T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return model, tf, (lambda m, x: m(x))
    raise ValueError(backbone)


@torch.no_grad()
def _embed(model, embed_fn, loader, desc):
    F, Y = [], []
    for xb, yb in tqdm(loader, desc=desc):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F.append(embed_fn(model, xb.to(DEVICE)).float().cpu().numpy())
        Y.append(np.asarray(yb))
    return np.concatenate(F).astype(np.float32), np.concatenate(Y)


def _cifar_loaders(tf):
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR100
    tr = CIFAR100(DATA_DIR, train=True, download=False, transform=tf)
    te = CIFAR100(DATA_DIR, train=False, download=False, transform=tf)
    L = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=8,
                              pin_memory=True)
    return L(tr), L(te)


def _cub_loaders(tf):
    import io
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    cache = os.path.join(DATA_DIR, "hf")
    dd = load_dataset("Donghyun99/CUB-200-2011", cache_dir=cache)
    lk = "label" if "label" in dd["train"].column_names else "labels"

    class _DS(Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            r = self.ds[i]; img = r["image"]
            if not hasattr(img, "mode"):
                img = Image.open(io.BytesIO(img["bytes"]))
            return tf(img.convert("RGB")), int(r[lk])
    L = lambda ds: DataLoader(_DS(ds), batch_size=256, shuffle=False,
                              num_workers=8, pin_memory=True)
    return L(dd["train"]), L(dd["test"])


def features(dataset, backbone):
    tag = BACKBONES[backbone]
    path = os.path.join(OUT, f"feat_{dataset}_{tag}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    os.makedirs(OUT, exist_ok=True)
    model, tf, embed_fn = _backbone_setup(backbone)
    trL, teL = (_cifar_loaders if dataset == "CIFAR100" else _cub_loaders)(tf)
    Xtr, ytr = _embed(model, embed_fn, trL, f"{backbone}/{dataset} train")
    Xte, yte = _embed(model, embed_fn, teL, f"{backbone}/{dataset} test")
    np.savez_compressed(path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"[feat] {backbone}/{dataset}: train {Xtr.shape} test {Xte.shape} "
          f"({len(np.unique(ytr))} classes)", flush=True)
    return Xtr, ytr, Xte, yte


# ── coarse groups for semantic-coherence (step 3b) ───────────────────────────
# CIFAR-100 standard fine(100)->coarse(20) superclass mapping.
_CIFAR_COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13])


def coarse_groups(dataset):
    """Fine-class -> coarse-group id, or None if unavailable."""
    if dataset == "CIFAR100":
        return _CIFAR_COARSE
    if dataset == "CUB200":
        try:
            from datasets import load_dataset
            cache = os.path.join(DATA_DIR, "hf")
            d = load_dataset("Donghyun99/CUB-200-2011", split="train", cache_dir=cache)
            lk = "label" if "label" in d.column_names else "labels"
            names = d.features[lk].names            # e.g. "001.Black_footed_Albatross"
            grp = [n.split(".")[-1].split("_")[-1] for n in names]   # family-ish token
            uniq = {g: i for i, g in enumerate(sorted(set(grp)))}
            return np.array([uniq[g] for g in grp])
        except Exception as e:
            print(f"[warn] CUB coarse groups unavailable ({e}); skip coherence",
                  flush=True)
            return None
    return None


# ── decompositions x = explained + residual ──────────────────────────────────
def signed_weights(R, q_n):
    A = R @ R.T
    return q_n @ R.T @ np.linalg.pinv(A)        # (N, K) least-squares weights


# ── metrics ──────────────────────────────────────────────────────────────────
def cluster_scores(V, labels, k, seed):
    Vn = normalize(V, axis=1)
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Vn).labels_
    pur = sum(np.bincount(labels[pred == c]).max() for c in np.unique(pred)) / len(labels)
    return dict(nmi=float(normalized_mutual_info_score(labels, pred)),
                ari=float(adjusted_rand_score(labels, pred)),
                purity=float(pur))


def class_attribution(w, ray_class, base_sorted):
    """Group per-ray weights by the base class each ray belongs to -> (N, n_base)."""
    H = np.zeros((w.shape[0], len(base_sorted)), np.float32)
    for i, c in enumerate(base_sorted):
        cols = ray_class == c
        if cols.any():
            H[:, i] = w[:, cols].sum(1)
    return H


def coherence(H, base_sorted, y_novel, coarse):
    """Fraction of novel imgs whose top attributed base class shares the novel
    class's coarse group. Returns (coherence, chance) or (None, None)."""
    if coarse is None:
        return None, None
    top_fine = base_sorted[H.argmax(1)]            # (N,) attributed base fine class
    hit = (coarse[top_fine] == coarse[y_novel]).mean()
    base_coarse = coarse[base_sorted]              # (n_base,)
    # chance = per-img prob a random base class matches the novel's coarse group
    chance = np.mean([(base_coarse == coarse[y]).mean() for y in y_novel])
    return float(hit), float(chance)


def mahalanobis_auroc(Xtr_base, base_res, novel_res, Xte_base, Xte_novel):
    mu = Xtr_base.mean(0); Xc = Xtr_base - mu
    cov = (Xc.T @ Xc) / len(Xc)
    cov += 1e-3 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    P = np.linalg.inv(cov)
    md = lambda X: np.einsum("nd,de,ne->n", X - mu, P, X - mu)
    y = np.r_[np.zeros(len(Xte_base)), np.ones(len(Xte_novel))]
    return (float(roc_auc_score(y, np.r_[md(Xte_base), md(Xte_novel)])),
            float(roc_auc_score(y, np.r_[base_res, novel_res])))


# ── one seed ─────────────────────────────────────────────────────────────────
def run_seed(Xtr, ytr, Xte, yte, seed, n_rays, base_frac, coarse):
    D = Xtr.shape[1]
    K = min(n_rays, D - 1)                          # signed needs a residual: K < D
    rng = np.random.default_rng(seed)
    classes = np.unique(ytr); rng.shuffle(classes)
    n_base = int(round(len(classes) * base_frac))
    base, novel = set(classes[:n_base].tolist()), set(classes[n_base:].tolist())
    base_sorted = np.array(sorted(base))

    tr_b = np.array([y in base for y in ytr])
    te_b = np.array([y in base for y in yte]); te_n = ~te_b
    Xtr_base, ytr_base = Xtr[tr_b], ytr[tr_b]
    Xte_base, Xte_novel = Xte[te_b], Xte[te_n]
    y_novel = yte[te_n]
    remap = {c: i for i, c in enumerate(sorted(novel))}
    y_novel_c = np.array([remap[c] for c in y_novel])

    hull = ConicHull(n_rays=K, use_pca=True, pca_dim=64, ray_diversity="hybrid")
    hull.fit(Xtr_base)
    R = hull.extreme_rays_                          # (K, D) unit rows
    ray_class = ytr_base[hull.extreme_rays_index]   # base class of each ray
    pca = PCA(n_components=min(K, D - 1)).fit(normalize(Xtr_base, axis=1))

    qn_novel = normalize(Xte_novel, axis=1)
    qn_base = normalize(Xte_base, axis=1)

    out = {"n_base": len(base), "n_novel": len(novel), "K": R.shape[0]}
    frac = lambda M: float(np.linalg.norm(M, axis=1).mean())

    # nnls (cone) — keep weights for attribution
    w_nov = hull.reconstruct(qn_novel); expl_nov = w_nov @ R
    w_bas = hull.reconstruct(qn_base);  expl_bas = w_bas @ R
    # signed (same atoms)
    ws_nov = signed_weights(R, qn_novel); expls_nov = ws_nov @ R
    ws_bas = signed_weights(R, qn_base);  expls_bas = ws_bas @ R
    # pca
    exp_p_nov = pca.inverse_transform(pca.transform(qn_novel))
    exp_p_bas = pca.inverse_transform(pca.transform(qn_base))

    decomps = {
        "nnls":   (expl_nov, qn_novel - expl_nov, expl_bas, qn_base - expl_bas),
        "signed": (expls_nov, qn_novel - expls_nov, expls_bas, qn_base - expls_bas),
        "pca":    (exp_p_nov, qn_novel - exp_p_nov, exp_p_bas, qn_base - exp_p_bas),
    }
    for name, (en, rn, eb, rb) in decomps.items():
        out[f"{name}_residual"] = cluster_scores(rn, y_novel_c, len(novel), seed)
        out[f"{name}_explained"] = cluster_scores(en, y_novel_c, len(novel), seed)
        # step 1: reconstruction fraction (q is unit-norm so ‖expl‖/‖q‖ = ‖expl‖)
        out[f"recon_novel_{name}"] = frac(en)
        out[f"recon_base_{name}"] = frac(eb)
    out["raw"] = cluster_scores(qn_novel, y_novel_c, len(novel), seed)

    # crux secondary: novelty AUROC (Mahalanobis vs nnls residual magnitude)
    r_b = np.linalg.norm(decomps["nnls"][3], axis=1)
    r_n = np.linalg.norm(decomps["nnls"][1], axis=1)
    out["auroc_mahalanobis"], out["auroc_nnls_residual_mag"] = mahalanobis_auroc(
        Xtr_base, r_b, r_n, Xte_base, Xte_novel)

    # step 3a: attribution consistency (cluster base-class attribution histograms)
    H_nn = class_attribution(w_nov, ray_class, base_sorted)
    H_sg = class_attribution(ws_nov, ray_class, base_sorted)
    out["attr_nnls"] = cluster_scores(H_nn, y_novel_c, len(novel), seed)
    out["attr_signed"] = cluster_scores(H_sg, y_novel_c, len(novel), seed)
    # step 3b: semantic coherence of top attributed known part
    coh_nn, chance = coherence(H_nn, base_sorted, y_novel, coarse)
    coh_sg, _ = coherence(H_sg, base_sorted, y_novel, coarse)
    out["coherence_nnls"] = coh_nn
    out["coherence_signed"] = coh_sg
    out["coherence_chance"] = chance
    return out


# ── aggregate + report ───────────────────────────────────────────────────────
def _agg(seeds_res):
    agg = {}
    for k, v0 in seeds_res[0].items():
        if isinstance(v0, dict) and "ari" in v0:
            for m in ("nmi", "ari", "purity"):
                vals = [s[k][m] for s in seeds_res]
                agg[f"{k}.{m}"] = [float(np.mean(vals)), float(np.std(vals))]
        elif isinstance(v0, (int, float)) and not isinstance(v0, bool):
            vals = [s[k] for s in seeds_res if s[k] is not None]
            agg[k] = [float(np.mean(vals)), float(np.std(vals))] if vals else [None, None]
    agg["_per_seed"] = seeds_res
    return agg


def _verdict(agg):
    """ARI-primary gates. Claim = quarantine differential: cone gap
    (residual-explained) >> signed gap on the same atoms."""
    A = lambda k: agg[k + ".ari"][0]
    raw, nn_r, nn_e = A("raw"), A("nnls_residual"), A("nnls_explained")
    sg_r, sg_e = A("signed_residual"), A("signed_explained")
    gap_nnls, gap_signed = nn_r - nn_e, sg_r - sg_e
    quarantine = nn_r >= 0.90 * raw and nn_e <= 0.60 * raw
    cone_specific = gap_nnls > gap_signed + 0.10

    lines = [
        "# CoCert crux — does the conic residual quarantine novel-class identity?\n",
        "KMeans(k=#novel) clustering vs true novel labels (mean±std over seeds).",
        "ARI primary (NMI saturates). Claim = QUARANTINE DIFFERENTIAL.\n",
        "| vector | ARI | NMI | purity |", "|---|--:|--:|--:|",
    ]
    for k in ("raw", "nnls_residual", "nnls_explained", "signed_residual",
              "signed_explained", "pca_residual", "pca_explained"):
        lines.append(f"| {k} | {agg[k+'.ari'][0]:.3f}±{agg[k+'.ari'][1]:.3f} | "
                     f"{agg[k+'.nmi'][0]:.3f} | {agg[k+'.purity'][0]:.3f} |")
    lines += [
        "",
        f"quarantine gap (ARI, residual−explained):  cone {gap_nnls:+.3f}   "
        f"signed {gap_signed:+.3f}   Δ {gap_nnls - gap_signed:+.3f}",
        f"[GATE 1 quarantine]  residual>=0.9*raw ({nn_r:.3f}>={0.9*raw:.3f}) AND "
        f"explained<=0.6*raw ({nn_e:.3f}<={0.6*raw:.3f}) : {'PASS' if quarantine else 'FAIL'}",
        f"[GATE 2 cone-specific] cone-gap > signed-gap+0.10 "
        f"({gap_nnls:+.3f} > {gap_signed + 0.10:+.3f}) : {'PASS' if cone_specific else 'FAIL'}",
        ("VERDICT: GO — non-negativity quarantines novel-class identity into an "
         "auditable KKT residual (build the interpretability/certificate paper, "
         "NOT acquisition: residual only ties raw)." if quarantine and cone_specific
         else "VERDICT: GATED — quarantine holds but not cone-specific." if quarantine
         else "VERDICT: KILL — residual does not retain novel-class identity."),
    ]
    return "\n".join(lines) + "\n"


def _report_controls(agg):
    g = lambda k: agg[k][0] if agg.get(k) and agg[k][0] is not None else None
    lines = ["", "## step 1 — reconstruction fraction  ‖explained‖/‖q‖  (q unit-norm)",
             "| method | base | novel |", "|---|--:|--:|"]
    for m in ("nnls", "signed", "pca"):
        lines.append(f"| {m} | {g('recon_base_'+m):.3f} | {g('recon_novel_'+m):.3f} |")
    lines += [
        "(nontrivial only if nnls reconstructs a substantial fraction of NOVEL "
        "points yet still strips their identity — cf. crux explained ARI.)",
        "",
        "## step 3 — nameability of the 'known parts' attribution",
        f"3a attribution consistency (ARI, cluster base-class attribution of novel imgs):"
        f"  nnls {agg['attr_nnls.ari'][0]:.3f}   signed {agg['attr_signed.ari'][0]:.3f}",
    ]
    coh_nn, coh_sg, ch = g("coherence_nnls"), g("coherence_signed"), g("coherence_chance")
    if coh_nn is None:
        lines.append("3b semantic coherence: n/a (coarse groups unavailable)")
    else:
        lines.append(f"3b semantic coherence (top attributed part shares novel's coarse "
                     f"group):  nnls {coh_nn:.3f}   signed {coh_sg:.3f}   chance {ch:.3f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR100", choices=["CIFAR100", "CUB200"])
    ap.add_argument("--backbone", default="vit", choices=list(BACKBONES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-rays", type=int, default=256)
    ap.add_argument("--k-sweep", type=int, nargs="+", default=None,
                    help="sweep dictionary sizes K (overrides --n-rays)")
    ap.add_argument("--base-frac", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print(f"[cfg] dataset={args.dataset} backbone={args.backbone} seeds={args.seeds} "
          f"base_frac={args.base_frac}", flush=True)

    Xtr, ytr, Xte, yte = features(args.dataset, args.backbone)
    coarse = coarse_groups(args.dataset)
    Ks = args.k_sweep or [args.n_rays]

    sweep = []
    for K in Ks:
        seeds_res = []
        for s in range(args.seeds):
            r = run_seed(Xtr, ytr, Xte, yte, s, K, args.base_frac, coarse)
            print(f"[K={r['K']} seed {s}] raw {r['raw']['ari']:.3f} | "
                  f"nnls-res {r['nnls_residual']['ari']:.3f} "
                  f"nnls-expl {r['nnls_explained']['ari']:.3f} | "
                  f"recon novel {r['recon_novel_nnls']:.2f} | "
                  f"attr {r['attr_nnls']['ari']:.3f} | "
                  f"coh {r['coherence_nnls'] if r['coherence_nnls'] is None else round(r['coherence_nnls'],3)}",
                  flush=True)
            seeds_res.append(r)
        agg = _agg(seeds_res)
        tag = f"{args.dataset}_{args.backbone}_K{agg['K'][0]:.0f}"
        with open(os.path.join(OUT, f"results_{tag}.json"), "w") as f:
            json.dump(agg, f, indent=2)
        rep = _verdict(agg) + _report_controls(agg)
        with open(os.path.join(OUT, f"report_{tag}.md"), "w") as f:
            f.write(rep)
        gap = agg["nnls_residual.ari"][0] - agg["nnls_explained.ari"][0]
        sgap = agg["signed_residual.ari"][0] - agg["signed_explained.ari"][0]
        sweep.append(dict(K=agg["K"][0], cone_gap=gap, signed_gap=sgap,
                          recon_novel=agg["recon_novel_nnls"][0],
                          attr=agg["attr_nnls.ari"][0],
                          coh=agg.get("coherence_nnls", [None])[0]))
        if len(Ks) == 1:
            print("\n" + rep, flush=True)

    if len(Ks) > 1:
        print("\n# K-sweep (step 2)  —  cone-gap = quarantine strength (ARI)")
        print("| K | cone-gap | signed-gap | recon-novel | attr-ARI | coherence |")
        print("|--:|--:|--:|--:|--:|--:|")
        for r in sweep:
            coh = "n/a" if r["coh"] is None else f"{r['coh']:.3f}"
            print(f"| {r['K']:.0f} | {r['cone_gap']:+.3f} | {r['signed_gap']:+.3f} | "
                  f"{r['recon_novel']:.2f} | {r['attr']:.3f} | {coh} |")
        with open(os.path.join(OUT,
                  f"sweep_{args.dataset}_{args.backbone}.json"), "w") as f:
            json.dump(sweep, f, indent=2)


if __name__ == "__main__":
    main()
