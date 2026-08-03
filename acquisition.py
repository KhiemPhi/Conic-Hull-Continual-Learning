"""
acquisition.py
--------------
Is the conic hull a LABEL-EFFICIENT acquisition primitive? On the unit hypersphere of
frozen ViT/CLIP features, does DIRECTIONAL coverage (conic extreme rays) select more
informative points to label than METRIC coverage (k-center) or herding/random?

This is the ONE regime the cone should win: the objective IS coverage efficiency
(accuracy per label), not a decision. Extreme rays are BOUNDARY points -> useless for a
centroid classifier (NCM), but potentially the margin-defining points for a linear probe
(support-vector-like). So the PRIMARY classifier is a linear probe; NCM is a diagnostic
that should FAVOR herding (central points) -- exposing the classifier-dependence.

Protocol (class-balanced pool AL): for each class, SELECT b points from its train pool by
each strategy, train one classifier on the b*C labeled points, eval on the full test set.
Sweep b in {3,5,10,20,40}. Curve = accuracy vs labels.

Strategies (all per-class selection):
  random  : b random points
  kcenter : greedy farthest-first coverage (metric coverage -- the baseline to beat)
  herding : iCaRL herding (tracks the class mean -- central points)
  conic   : ConicHull extreme rays (directional coverage -- YOUR conic hull)

WIN (novel): conic's acc-vs-label curve dominates kcenter at small b (label-scarce),
especially on fine-grained/multimodal classes -> "directional coverage is label-efficient
acquisition on the feature sphere." KILL: conic ~= kcenter -> angular = metric coverage.

    HF_HUB_OFFLINE=1 python -u acquisition.py --dataset CIFAR100
    HF_HUB_OFFLINE=1 python -u acquisition.py --dataset CUB200
"""
import argparse, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from conic_hull import ConicHull

OUT = "./acquisition_out"
BUDGETS = [3, 5, 10, 20, 40]
STRATS = ["random", "kcenter", "herding", "conic"]


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def load_feats(dataset):
    if dataset == "CIFAR100":
        d = np.load("./ranpac_out/cifar100_feats.npz")
        return d["ftr"], d["ytr"], d["fte"], d["yte"]
    if dataset == "CUB200":
        d = np.load("./fscil_out/CUB200_feats.npz")
        return d["ftr"], d["ytr"], d["fte"], d["yte"]
    raise ValueError(dataset)


def pad(idx, b, n, rng):
    """Ensure exactly min(b,n) unique indices."""
    idx = list(dict.fromkeys(int(i) for i in idx))
    if len(idx) < min(b, n):
        rest = [i for i in range(n) if i not in set(idx)]
        rng.shuffle(rest)
        idx += rest[:min(b, n) - len(idx)]
    return idx[:min(b, n)]


def sel_random(Xc, b, rng):
    return pad(rng.choice(len(Xc), min(b, len(Xc)), replace=False).tolist(), b, len(Xc), rng)


def sel_kcenter(Xc, b, rng):
    Xn = unit(Xc); n = len(Xn)
    c = unit(Xn.mean(0, keepdims=True))
    first = int((1 - (Xn @ c.T).ravel()).argmax())          # farthest from centroid
    chosen = [first]; mind = 1 - Xn @ Xn[first]
    while len(chosen) < min(b, n):
        j = int(mind.argmax()); chosen.append(j)
        mind = np.minimum(mind, 1 - Xn @ Xn[j])
    return pad(chosen, b, n, rng)


def sel_herding(Xc, b, rng):
    Xn = unit(Xc); mu = Xn.mean(0); chosen = []; s = np.zeros_like(mu)
    for t in range(min(b, len(Xn))):
        scores = Xn @ (mu - s / (t + 1e-8)) if t else Xn @ mu
        scores[chosen] = -np.inf
        j = int(np.argmax(scores)); chosen.append(j); s += Xn[j]
    return pad(chosen, b, len(Xn), rng)


def sel_conic(Xc, b, rng):
    Xn = unit(Xc); n = len(Xn)
    if n < 12:
        return pad(list(range(n)), b, n, rng)
    ch = ConicHull(n_rays=min(b, n), use_pca=False, ray_diversity="hybrid")
    ch.fit(Xc)
    rays = unit(np.asarray(ch.extreme_rays_))
    idx = (rays @ Xn.T).argmax(1).tolist()                   # nearest real point per ray
    return pad(idx, b, n, rng)


SELECT = {"random": sel_random, "kcenter": sel_kcenter,
          "herding": sel_herding, "conic": sel_conic}


def build_labeled(Ftr, ytr, classes, strat, b, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for c in classes:
        Xc = Ftr[ytr == c]
        idx = SELECT[strat](Xc, b, rng)
        X.append(Xc[idx]); Y.append(np.full(len(idx), c))
    return unit(np.concatenate(X)), np.concatenate(Y)


def eval_probe(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


def eval_ncm(Xtr, ytr, Xte, yte, classes):
    cents = unit(np.stack([Xtr[ytr == c].mean(0) for c in classes]))
    pred = classes[(unit(Xte) @ cents.T).argmax(1)]
    return float((pred == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR100", choices=["CIFAR100", "CUB200"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = load_feats(args.dataset)
    classes = np.unique(ytr)
    Xte_u = unit(Fte)
    print(f"[acq] {args.dataset}  C={len(classes)}  train={len(Ftr)} test={len(Fte)}  "
          f"pool/class~{len(Ftr)//len(classes)}", flush=True)

    res = {c: {s: {} for s in STRATS} for c in ("probe", "ncm")}
    for b in BUDGETS:
        for strat in STRATS:
            pa, na = [], []
            for seed in range(args.seeds):
                Xtr, Ytr = build_labeled(Ftr, ytr, classes, strat, b, seed)
                pa.append(eval_probe(Xtr, Ytr, Xte_u, yte))
                na.append(eval_ncm(Xtr, Ytr, Fte, yte, classes))
            res["probe"][strat][b] = (float(np.mean(pa)), float(np.std(pa)))
            res["ncm"][strat][b] = (float(np.mean(na)), float(np.std(na)))
        row = "  ".join(f"{s}:{res['probe'][s][b][0]*100:5.1f}" for s in STRATS)
        print(f"[probe b={b:3d}] {row}", flush=True)

    with open(os.path.join(OUT, f"{args.dataset}.json"), "w") as f:
        json.dump(res, f, indent=2)

    for clf in ("probe", "ncm"):
        print(f"\n=== {args.dataset} accuracy vs labels/class  ({clf}) ===")
        print("  b   | " + "  ".join(f"{s:>8s}" for s in STRATS) + "   | conic-kcenter")
        for b in BUDGETS:
            cells = "  ".join(f"{res[clf][s][b][0]*100:6.1f}  " for s in STRATS)
            dc = (res[clf]["conic"][b][0] - res[clf]["kcenter"][b][0]) * 100
            print(f"  {b:<3d} | {cells} |  {dc:+.1f}")


if __name__ == "__main__":
    main()
