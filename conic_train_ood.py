"""
conic_train_ood.py
------------------
Can a representation TRAINED to be conic make the real (NNLS) cone finally beat
multi-prototype NCM at OOD?  This is the one lever left after every post-hoc test
tied or lost: reshape the space instead of scoring a frozen one.

We train a projection g: R^512 -> R^d on frozen CLIP features with a differentiable
cone-margin loss (own-class cone alignment up, other classes down), then run the
exact cone-vs-multiproto OOD test in the learned space.

Differentiable cone score (one-step nonneg reconstruction, a soft conic membership):
    z = normalize(g(x));  a_c = ReLU(W_c^T z);  ẑ_c = W_c a_c;  s_c = cos(z, ẑ_c)
Classes are learnable generators W_c (m per class); logits = scale * s_c, trained
with cross-entropy — so the space is shaped to be separable BY CONE MEMBERSHIP.

Three conditions, same downstream test (real ConicHull vs spherical-k-means):
    raw     : frozen CLIP (baseline — cone loses here, per cone_vs_multiproto_ood)
    generic : projection trained with plain linear-head CE (control)
    conic   : projection trained with the cone-margin loss

If 'conic' flips cone >= multiproto while 'raw'/'generic' don't, that is the
existence proof that a conic-trained representation is what cones need.

Usage
-----
    python -u conic_train_ood.py --id CIFAR100 --budgets 2 4 --steps 600
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cone_vs_multiproto_ood import (load_clip, available, split_sq, cap,
                                    fit_cones, idness_cone, fit_multiproto,
                                    idness_multiproto, auroc_fpr)

OUT_DIR = "./conic_train_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class ConicProj(nn.Module):
    """Linear projection + per-class learnable generators; logits = cone alignment."""

    def __init__(self, in_dim, d, n_classes, m, scale=10.0):
        super().__init__()
        self.g = nn.Linear(in_dim, d)
        self.W = nn.Parameter(torch.randn(n_classes, m, d) * 0.1)
        self.scale = scale

    def project(self, x):
        return F.normalize(self.g(x), dim=1)

    def cone_scores(self, z):
        Wn = F.normalize(self.W, dim=2)                 # (C,m,d) unit generators
        proj = torch.einsum("cmd,bd->bcm", Wn, z)       # (B,C,m)
        a = F.relu(proj)
        recon = torch.einsum("bcm,cmd->bcd", a, Wn)     # (B,C,d)
        return F.cosine_similarity(z.unsqueeze(1), recon, dim=2)  # (B,C)

    def forward(self, x):
        return self.scale * self.cone_scores(self.project(x))


class GenericProj(nn.Module):
    """Linear projection + linear classification head (plain CE control)."""

    def __init__(self, in_dim, d, n_classes):
        super().__init__()
        self.g = nn.Linear(in_dim, d)
        self.head = nn.Linear(d, n_classes)

    def project(self, x):
        return F.normalize(self.g(x), dim=1)

    def forward(self, x):
        return self.head(self.project(x))


def train_proj(model, Ftr, ytr, steps, lr=1e-3, bs=512, seed=0):
    model = model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(Ftr, device=DEVICE)
    y = torch.tensor(ytr.astype(np.int64), device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    n = len(X)
    for step in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g, device=DEVICE)
        logits = model(X[idx])
        loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (model(X).argmax(1) == y).float().mean().item()
            print(f"    step {step:4d}  loss {loss.item():.3f}  train_acc {acc:.3f}")
    return model.eval()


@torch.no_grad()
def project_np(model, F_np, chunk=8192):
    out = []
    for i in range(0, len(F_np), chunk):
        x = torch.tensor(F_np[i:i + chunk], device=DEVICE)
        out.append(model.project(x).cpu().numpy())
    return _norm(np.concatenate(out))


# ─────────────────────────────────────────────────────────────────────────────
# The downstream cone-vs-multiproto OOD test in a given space
# ─────────────────────────────────────────────────────────────────────────────
def eval_space(name, Ftr, ytr, Fq, ood_feats, budgets):
    rows = {}
    for m in budgets:
        cones = fit_cones(Ftr, ytr, m)
        protos = fit_multiproto(Ftr, ytr, m)
        idc_id = idness_cone(cones, Fq)
        idp_id = idness_multiproto(protos, Fq)
        ca, pa = [], []
        for _, Fo in ood_feats:
            a, _ = auroc_fpr(idc_id, idness_cone(cones, Fo)); ca.append(a)
            a, _ = auroc_fpr(idp_id, idness_multiproto(protos, Fo)); pa.append(a)
        rows[m] = dict(cone=float(np.mean(ca)), mp=float(np.mean(pa)),
                       delta=float(np.mean(ca) - np.mean(pa)))
        print(f"  [{name} m={m}] cone {rows[m]['cone']:.4f}  "
              f"multiproto {rows[m]['mp']:.4f}  Δ {rows[m]['delta']:+.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="CIFAR100")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--gen-m", type=int, default=4,
                    help="generators/class used INSIDE the conic training loss")
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--n-query-cap", type=int, default=2000)
    ap.add_argument("--n-ood-cap", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    F_id = load_clip(args.id)
    if F_id is None:
        raise SystemExit(f"no CLIP cache for {args.id}")
    Fall, y = F_id
    n_classes = int(y.max()) + 1
    s_i, q_i = split_sq(Fall, y)
    Ftr, ytr = Fall[s_i], y[s_i]
    Fq = cap(Fall[q_i], args.n_query_cap)
    ood = [(o, cap(load_clip(o)[0], args.n_ood_cap, seed=1))
           for o in available() if o != args.id]
    print(f"[id] {args.id} classes={n_classes} train={len(Ftr)} query={len(Fq)} "
          f"| OOD {[o for o,_ in ood]}")

    all_results = {}

    # 1) RAW CLIP space
    print("\n=== RAW CLIP ===")
    all_results["raw"] = eval_space("raw", Ftr, ytr, Fq, ood, args.budgets)

    # 2) GENERIC-CE projection (control)
    print("\n=== GENERIC-CE projection ===")
    gm = train_proj(GenericProj(Ftr.shape[1], args.d, n_classes),
                    Ftr, ytr, args.steps)
    Ptr, Pq = project_np(gm, Ftr), project_np(gm, Fq)
    ood_g = [(o, project_np(gm, Fo)) for o, Fo in ood]
    all_results["generic"] = eval_space("generic", Ptr, ytr, Pq, ood_g, args.budgets)

    # 3) CONIC-margin projection
    print("\n=== CONIC-margin projection ===")
    cm = train_proj(ConicProj(Ftr.shape[1], args.d, n_classes, args.gen_m),
                    Ftr, ytr, args.steps)
    Ctr, Cq = project_np(cm, Ftr), project_np(cm, Fq)
    ood_c = [(o, project_np(cm, Fo)) for o, Fo in ood]
    all_results["conic"] = eval_space("conic", Ctr, ytr, Cq, ood_c, args.budgets)

    _report(all_results, args)


def _report(res, args):
    with open(os.path.join(OUT_DIR, f"results_{args.id}.json"), "w") as f:
        json.dump(res, f, indent=2)
    lines = [f"# Conic-trained representation: does the cone beat multi-prototype? ({args.id})\n",
             "Cone vs multi-prototype OOD AUROC (mean over OOD datasets) in three "
             "spaces. If 'conic' flips Δ positive while raw/generic stay negative, a "
             "conic-trained space is what the cone needs.\n",
             "| space | budget m | cone | multiproto | Δ(cone−mp) |",
             "|---|--:|--:|--:|--:|"]
    for space in ("raw", "generic", "conic"):
        for m, r in res[space].items():
            lines.append(f"| {space} | {m} | {r['cone']:.4f} | {r['mp']:.4f} | "
                         f"{r['delta']:+.4f} |")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, f"report_{args.id}.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"[done] wrote {OUT_DIR}/report_{args.id}.md")


if __name__ == "__main__":
    main()
