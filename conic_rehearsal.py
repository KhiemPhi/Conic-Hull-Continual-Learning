"""
conic_rehearsal.py  (Build A)
-----------------------------
Rehearsal CIL at a MATCHED memory budget. Owns the fact that storing extreme rays IS
a (feature-space) memory, and benchmarks it fairly against other feature-memories at
the same budget k items/class. Frozen ViT-B/16 features; a trained linear head is
updated incrementally with replay (so memory genuinely matters).

Memory strategies (budget = k items/class):
  raw       : store k random features/class, replay those k
  herding   : store k iCaRL-herded features/class
  conic-gen : store k extreme rays/class, SYNTHESIZE S in-support features/class (ours)
  gaussian-gen : store mean+diag-std/class, sample S/class (control)

The generative memories (conic/gaussian) turn a k-item budget into unlimited balanced
replay. conic-gen vs gaussian-gen isolates whether the CONIC support (not any
augmentation) is what helps.

NOTE on bytes: a 768-d feature (~3 KB fp32) is ~50x lighter than a 224x224 image
(~150 KB), so at a matched BYTE budget these feature-memories store ~50x more than
image-rehearsal (iCaRL/GEM) -- that axis is analytical here (no backbone training).

Metrics: avg-incremental accuracy, last, average forgetting.

    HF_HUB_OFFLINE=1 python -u conic_rehearsal.py --k 20
"""
import argparse, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from conic_hull import ConicHull

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./rehearsal_out"
S_SYNTH = 100


def unit(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def feats():
    d = np.load("./ranpac_out/cifar100_feats.npz")
    return d["ftr"], d["ytr"], d["fte"], d["yte"]


def herding(X, k):
    """iCaRL herding: greedily pick k features whose running mean tracks the class mean."""
    Xu = unit(X); mu = Xu.mean(0); chosen = []; s = np.zeros_like(mu)
    for t in range(min(k, len(Xu))):
        scores = Xu @ (mu - s / (t + 1e-8)) if t else Xu @ mu
        j = int(np.argmax(scores)); chosen.append(j); s += Xu[j]
    return X[chosen]


def build_memory(method, Xc, k, seed):
    """Return a dict describing the class memory for later replay."""
    rng = np.random.default_rng(seed)
    if method == "raw":
        idx = rng.choice(len(Xc), min(k, len(Xc)), replace=False)
        return {"feats": Xc[idx]}
    if method == "herding":
        return {"feats": herding(Xc, k)}
    if method == "gaussian-gen":
        return {"mu": Xc.mean(0), "sd": Xc.std(0) + 1e-3}
    # conic-gen
    Xu = unit(Xc); kk = int(min(k, len(Xu)))
    ch = ConicHull(n_rays=kk, use_pca=False, ray_diversity="hybrid")
    if len(Xu) >= 12:
        ch.fit(Xc); rays = ch.extreme_rays_
    else:
        rays = Xu[:kk]
    return {"rays": rays}


def replay_feats(method, mem, S, seed):
    rng = np.random.default_rng(seed)
    if method in ("raw", "herding"):
        return mem["feats"]
    if method == "gaussian-gen":
        return unit(mem["mu"][None] + mem["sd"][None] * rng.standard_normal((S, len(mem["mu"]))).astype(np.float32))
    rays = mem["rays"]; a = rng.dirichlet(np.ones(len(rays)), size=S).astype(np.float32)
    return unit(a @ rays + 0.02 * rng.standard_normal((S, rays.shape[1])).astype(np.float32))


def train_head(Xtr, ytr, n_cls, epochs=15, lr=1e-2, bs=512, seed=0):
    torch.manual_seed(seed)
    W = torch.zeros(Xtr.shape[1], n_cls, device=DEVICE, requires_grad=True)
    b = torch.zeros(n_cls, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-4)
    X = torch.tensor(Xtr, device=DEVICE); Y = torch.tensor(ytr, device=DEVICE)
    n = len(X); g = torch.Generator(device=DEVICE).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(X[idx] @ W + b, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()


def run(method, Ftr, ytr, Fte, yte, order, tasks, k, seed):
    tsz = len(order) // tasks
    mem = {}; seen = []; accs = []; acc_matrix = []
    for t in range(tasks):
        cls = list(order[t*tsz:(t+1)*tsz]); seen += cls
        # assemble training set: current real + replay of old
        Xtr, Ytr = [], []
        for c in cls:
            Xc = Ftr[ytr == c]; Xtr.append(Xc); Ytr.append(np.full(len(Xc), c))
        for c, m in mem.items():
            rf = replay_feats(method, m, S_SYNTH, seed*7+c)
            Xtr.append(rf); Ytr.append(np.full(len(rf), c))
        Xtr = unit(np.concatenate(Xtr)); Ytr = np.concatenate(Ytr).astype(np.int64)
        W, b = train_head(Xtr, Ytr, int(max(seen)+1), seed=seed)
        # store memory for new classes
        for c in cls:
            mem[c] = build_memory(method, Ftr[ytr == c], k, seed*7+c)
        # eval on seen test
        mt = np.isin(yte, seen)
        pred = (torch.tensor(unit(Fte[mt]), device=DEVICE) @ W + b).argmax(1).cpu().numpy()
        accs.append(float((pred == yte[mt]).mean()))
        # per-task acc (for forgetting)
        row = []
        for tt in range(t+1):
            tc = order[tt*tsz:(tt+1)*tsz]; mm = np.isin(yte, tc)
            p = (torch.tensor(unit(Fte[mm]), device=DEVICE) @ W + b).argmax(1).cpu().numpy()
            row.append(float((p == yte[mm]).mean()))
        acc_matrix.append(row)
    # forgetting: for each task i (except last), max acc it ever had minus its final acc
    finals = acc_matrix[-1]
    fg = []
    for i in range(tasks - 1):
        prev = [acc_matrix[t][i] for t in range(i, tasks - 1)]   # steps after task i was learned
        if prev:
            fg.append(max(prev) - finals[i])
    forget = float(np.mean(fg)) if fg else 0.0
    return dict(avg=float(np.mean(accs)*100), last=float(accs[-1]*100), forget=float(forget*100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--k", type=int, default=20, help="memory budget items/class")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = feats()
    ncls = 100
    print(f"[rehearsal] CIFAR100 {args.tasks} tasks, budget k={args.k}/class, "
          f"synth S={S_SYNTH}", flush=True)
    res = {}
    for method in ("raw", "herding", "gaussian-gen", "conic-gen"):
        runs = [run(method, Ftr, ytr, Fte, yte,
                    np.random.default_rng(1000+s).permutation(ncls), args.tasks, args.k, s)
                for s in range(args.seeds)]
        res[method] = {kk: float(np.mean([r[kk] for r in runs])) for kk in ("avg", "last", "forget")}
        print(f"  [{method:12s}] avg {res[method]['avg']:.1f}  last {res[method]['last']:.1f}  "
              f"forget {res[method]['forget']:.1f}", flush=True)
    with open(os.path.join(OUT, f"rehearsal_k{args.k}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== matched budget k={args.k}/class (CIFAR-100, {args.tasks} tasks) ===")
    print("| method | avg-inc | last | forgetting |\n|---|--:|--:|--:|")
    for m in ("raw", "herding", "gaussian-gen", "conic-gen"):
        r = res[m]
        print(f"| {m} | {r['avg']:.1f} | {r['last']:.1f} | {r['forget']:.1f} |")


if __name__ == "__main__":
    main()
