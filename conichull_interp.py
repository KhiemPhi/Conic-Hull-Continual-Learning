"""
conichull_interp.py
-------------------
Validate the interpretability finding with THE repo's ConicHull (SPA extreme rays +
NNLS), not just the learned Top-K SAE. Does the actual conic-hull dictionary give
monosemantic + localizable atoms too?

Dictionaries (matched size K=512) on COCO CLIP patch tokens:
  ConicHull   : your conic_hull.ConicHull -- SPA/FPS extreme rays (DATA exemplars),
                code = NNLS weights (reconstruct()).  [your implementation]
  conic-SAE   : learned Top-K ReLU dictionary
  signed-SAE  : learned Top-K, no ReLU
Metrics: coherence (mean cos among each atom's top patches) + best-atom localization
AP/IoU vs COCO boxes (Network-Dissection style).

    HF_HUB_OFFLINE=1 python -u conichull_interp.py
"""
import os, json, math
import numpy as np
import torch
import vit_sae_conic as V
import sae_segmentation_iou as S
from conic_hull import ConicHull

DEVICE = V.DEVICE
OUT = "./conichull_interp_out"
K = 512
N_IMG = 1500


def main():
    os.makedirs(OUT, exist_ok=True)
    tok, _, P = V.extract(N_IMG, "clip", "COCO")
    n_img = len(tok)
    Xraw = tok.reshape(-1, tok.shape[-1])                       # (N, D) raw patch tokens
    tokens_unit = Xraw / (np.linalg.norm(Xraw, axis=1, keepdims=True) + 1e-8)
    GTb, n_rows, score_cats, bg = S.build_gt_coco(n_img, P, "clip")
    print(f"[data] {n_img} imgs P={P} N={len(Xraw)} | GT rows {n_rows}", flush=True)

    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xraw), 40000, replace=False)           # SPA fit subsample

    res = {}

    # 1) YOUR ConicHull: SPA extreme rays + NNLS code
    ch = ConicHull(n_rays=K, use_pca=True, pca_dim=64, ray_diversity="hybrid")
    ch.fit(Xraw[sub])
    print(f"[conichull] fit {ch.extreme_rays_.shape[0]} extreme rays", flush=True)
    code = np.empty((len(Xraw), ch.extreme_rays_.shape[0]), np.float32)
    for i in range(0, len(Xraw), 16384):
        code[i:i+16384] = ch.reconstruct(Xraw[i:i+16384])       # NNLS weights (GPU FISTA)
    ap, iou, nc = S.localize(GTb, code, score_cats)
    res["ConicHull"] = {"coherence": V.coherence(code, tokens_unit),
                        "loc_AP": float(np.mean(ap)), "IoU": float(np.mean(iou)), "n": nc}
    print(f"  [ConicHull] coher {res['ConicHull']['coherence']:.3f}  "
          f"loc-AP {res['ConicHull']['loc_AP']:.3f}  IoU {res['ConicHull']['IoU']:.3f}", flush=True)

    # 2/3) learned SAEs (matched K)
    Xc = Xraw - Xraw.mean(0, keepdims=True)
    Xg = torch.tensor(Xc, device=DEVICE)
    Xtr = torch.tensor(Xc[sub], device=DEVICE)
    for nonneg, name in [(True, "conic-SAE"), (False, "signed-SAE")]:
        m = V.train_sae(Xtr, K, nonneg, 16)
        C = V.sae_codes(m, Xg, 16)
        ap, iou, nc = S.localize(GTb, C, score_cats)
        res[name] = {"coherence": V.coherence(C, tokens_unit),
                     "loc_AP": float(np.mean(ap)), "IoU": float(np.mean(iou)), "n": nc}
        print(f"  [{name}] coher {res[name]['coherence']:.3f}  "
              f"loc-AP {res[name]['loc_AP']:.3f}  IoU {res[name]['IoU']:.3f}", flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n| dictionary | coherence | loc-AP | IoU |\n|---|--:|--:|--:|")
    for n in ("ConicHull", "conic-SAE", "signed-SAE"):
        print(f"| {n} | {res[n]['coherence']:.3f} | {res[n]['loc_AP']:.3f} | {res[n]['IoU']:.3f} |")


if __name__ == "__main__":
    main()
