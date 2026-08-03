"""
sae_uncertainty.py
------------------
Can NON-NEGATIVITY (the one property that survived controls -> monosemantic codes) buy
better UNCERTAINTY / ERROR-CORRECTION in classification -- not as a scalar confidence
(that ties softmax, the redundancy wall), but through STRUCTURE?

Setup: frozen ViT features. Softmax linear probe = the classifier + baselines (MSP,
entropy). Two Top-K SAEs trained on the same features -- nonneg (ReLU) and signed
(control) -- give codes z(x). Non-negativity is the ONLY difference between them.

Three tests:
  A. Selective classification (risk-coverage / AURC): does a non-neg signal
     (recon residual, concept-margin) improve error detection OVER MSP, and beat the
     SIGNED SAE control? Fusion = z(MSP)+z(nonneg).
  B. Conformal prediction sets (APS): does non-neg evidence give SMALLER sets at equal
     coverage -- overall (expect tie) vs on the AMBIGUOUS subset (top-2 softmax gap<0.1),
     where softmax's winner-take-all miscalibrates (the hypothesized niche)?
  C. Attribution + correction: on errors, is a single discriminative atom identifiable,
     and does re-ranking top-2 by non-neg class-evidence recover accuracy?

WIN: non-neg (not signed) improves selective risk / shrinks ambiguous-subset sets /
enables actionable correction. KILL: scalar signals tie MSP everywhere -> redundancy wall.

    HF_HUB_OFFLINE=1 python -u sae_uncertainty.py --dataset CIFAR100
    HF_HUB_OFFLINE=1 python -u sae_uncertainty.py --dataset CUB200
"""
import argparse, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "./sae_unc_out"


def load_feats(dataset):
    if dataset == "CIFAR100":
        d = np.load("./ranpac_out/cifar100_feats.npz")
    elif dataset == "CUB200":
        d = np.load("./fscil_out/CUB200_feats.npz")
    else:
        raise ValueError(dataset)
    return d["ftr"], d["ytr"], d["fte"], d["yte"]


# ── Top-K SAE (nonneg = ReLU before top-K ; signed = top-K by |value|) ──────────
class TopKSAE(nn.Module):
    def __init__(self, D, m, K, nonneg):
        super().__init__()
        self.K, self.nonneg = K, nonneg
        self.b_dec = nn.Parameter(torch.zeros(D))
        self.W_e = nn.Parameter(torch.randn(D, m) * (1 / np.sqrt(D)))
        self.b_e = nn.Parameter(torch.zeros(m))
        self.W_d = nn.Parameter(torch.randn(m, D) * (1 / np.sqrt(m)))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_e + self.b_e
        act = F.relu(pre) if self.nonneg else pre
        rank = act if self.nonneg else act.abs()
        thr = torch.topk(rank, self.K, dim=1).values[:, -1:]
        return act * (rank >= thr)

    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_d + self.b_dec, z


def train_sae(X, m, K, nonneg, epochs=40, bs=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    D = X.shape[1]; model = TopKSAE(D, m, K, nonneg).to(DEVICE)
    with torch.no_grad():
        model.b_dec.copy_(torch.tensor(X.mean(0), device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, device=DEVICE); n = len(Xt)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g, device=DEVICE)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i+bs]]
            xh, _ = model(xb)
            loss = F.mse_loss(xh, xb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def codes(model, X, bs=8192):
    Z, R = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], device=DEVICE)
        xh, z = model(xb)
        Z.append(z.cpu().numpy()); R.append((xb - xh).norm(dim=1).cpu().numpy())
    return np.concatenate(Z), np.concatenate(R)


# ── metrics ────────────────────────────────────────────────────────────────────
def zscore(v):
    v = np.asarray(v, np.float64); return (v - v.mean()) / (v.std() + 1e-8)


def aurc(unc, correct):
    """Area under risk-coverage (lower=better) + accuracy@80% coverage."""
    order = np.argsort(unc)                       # most confident first
    c = correct[order].astype(np.float64)
    n = len(c); cov = np.arange(1, n + 1)
    risk = 1 - np.cumsum(c) / cov
    return float(risk.mean()), float(c[:int(0.8 * n)].mean())


def aps(proba_cal, y_cal, proba_te, y_te, alpha=0.1):
    def true_score(P, y):
        order = np.argsort(-P, 1)
        cum = np.cumsum(np.take_along_axis(P, order, 1), 1)
        ranks = np.argmax(order == y[:, None], 1)
        return cum[np.arange(len(y)), ranks]
    sc = true_score(proba_cal, y_cal); n = len(sc)
    q = np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    order = np.argsort(-proba_te, 1)
    cum = np.cumsum(np.take_along_axis(proba_te, order, 1), 1)
    sizes = np.minimum((cum < q).sum(1) + 1, proba_te.shape[1])
    ranks = np.argmax(order == y_te[:, None], 1)
    return (ranks < sizes), sizes


def softmax(a, t=1.0):
    a = a / t; a = a - a.max(1, keepdims=True)
    e = np.exp(a); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR100", choices=["CIFAR100", "CUB200"])
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--K", type=int, default=32)
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    Ftr, ytr, Fte, yte = load_feats(args.dataset)
    C = int(max(ytr.max(), yte.max()) + 1)
    print(f"[data] {args.dataset} tr {Ftr.shape} te {Fte.shape} C={C}", flush=True)

    # classifier + softmax baselines
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Ftr, ytr)
    P = clf.predict_proba(Fte); pred = clf.classes_[P.argmax(1)]
    correct = (pred == yte)
    print(f"[probe] test acc {correct.mean()*100:.1f}", flush=True)
    msp = 1 - P.max(1); ent = -(P * np.log(P + 1e-12)).sum(1)

    # two SAEs (nonneg vs signed control)
    sae_nn = train_sae(Ftr, args.m, args.K, nonneg=True)
    sae_sg = train_sae(Ftr, args.m, args.K, nonneg=False)
    ztr, _ = codes(sae_nn, Ftr)
    zte, resid_nn = codes(sae_nn, Fte)
    _, resid_sg = codes(sae_sg, Fte)

    # class atom-profiles (nonneg) -> concept evidence & margin
    zbar = np.stack([ztr[ytr == c].mean(0) for c in range(C)])          # (C,m) >=0
    zbar /= (np.linalg.norm(zbar, axis=1, keepdims=True) + 1e-8)
    zte_u = zte / (np.linalg.norm(zte, axis=1, keepdims=True) + 1e-8)
    E = zte_u @ zbar.T                                                   # (N,C) nonneg evidence
    Es = np.sort(E, 1)
    concept_margin = Es[:, -1] - Es[:, -2]                              # high = confident

    # ── A. selective classification ──
    scores = {"MSP": msp, "entropy": ent, "resid_nonneg": resid_nn,
              "resid_signed": resid_sg, "neg_margin": -concept_margin,
              "MSP+resid": zscore(msp) + zscore(resid_nn),
              "MSP+margin": zscore(msp) + zscore(-concept_margin)}
    print("\n=== A. selective classification (AURC lower=better | acc@80% coverage) ===")
    A = {}
    for name, s in scores.items():
        a, acc80 = aurc(s, correct); A[name] = {"aurc": a, "acc80": acc80}
        tag = " (control)" if name == "resid_signed" else ""
        print(f"  {name:14s} AURC {a*100:5.2f}   acc@80 {acc80*100:5.1f}{tag}")

    # ── B. conformal (APS): softmax vs nonneg-evidence vs fused ──
    rng = np.random.default_rng(0); idx = rng.permutation(len(Fte)); h = len(idx) // 2
    cal, tst = idx[:h], idx[h:]
    Qnn = softmax(E, t=max(E.std(), 1e-3))
    Pf = 0.5 * (P + Qnn); Pf /= Pf.sum(1, keepdims=True)
    gap = np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2]
    amb_tst = tst[gap[tst] < 0.1]
    print("\n=== B. conformal APS @ alpha=0.1 (coverage | mean set size) ===")
    B = {}
    for name, Prob in (("softmax", P), ("nonneg", Qnn), ("fused", Pf)):
        cov, sz = aps(Prob[cal], yte[cal], Prob[tst], yte[tst])
        # ambiguous subset sizes (recompute sets on amb using same calib)
        cov_a, sz_a = aps(Prob[cal], yte[cal], Prob[amb_tst], yte[amb_tst])
        B[name] = {"cov": float(cov.mean()), "size": float(sz.mean()),
                   "cov_amb": float(cov_a.mean()), "size_amb": float(sz_a.mean())}
        print(f"  {name:8s} overall cov {cov.mean():.3f} size {sz.mean():5.2f} | "
              f"ambiguous(n={len(amb_tst)}) cov {cov_a.mean():.3f} size {sz_a.mean():5.2f}")

    # ── C. attribution + correction ──
    err = np.where(~correct)[0]
    top2 = np.argsort(-P, 1)[:, :2]
    disc_atom = np.argmax(zbar[yte[err]] - zbar[pred[err]], 1)          # atom true>pred
    under = zte_u[err, disc_atom] < zbar[yte[err], disc_atom]           # under-activated
    attributable = float(under.mean()) if len(err) else 0.0
    # correction: among top-2 softmax, pick higher non-neg evidence; eval where true in top2
    in_top2 = np.array([yte[i] in top2[i] for i in range(len(yte))])
    ev_top2 = np.take_along_axis(E, top2, 1)
    rerank = top2[np.arange(len(yte)), ev_top2.argmax(1)]
    base_acc = float((pred[in_top2] == yte[in_top2]).mean())
    corr_acc = float((rerank[in_top2] == yte[in_top2]).mean())
    print("\n=== C. attribution + correction ===")
    print(f"  errors: {len(err)}  attributable (discriminative atom under-activated): "
          f"{attributable*100:.1f}%")
    print(f"  re-rank top-2 by non-neg evidence  (subset true-in-top2, n={in_top2.sum()}): "
          f"softmax {base_acc*100:.1f} -> nonneg {corr_acc*100:.1f} ({(corr_acc-base_acc)*100:+.1f})")

    with open(os.path.join(OUT, f"{args.dataset}.json"), "w") as f:
        json.dump({"acc": float(correct.mean()), "A": A, "B": B,
                   "attributable": attributable,
                   "correction": {"base": base_acc, "nonneg": corr_acc}}, f, indent=2)


if __name__ == "__main__":
    main()
