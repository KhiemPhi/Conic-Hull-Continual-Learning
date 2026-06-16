"""
classifier_alignment.py
-----------------------
Gaussian-replay classifier alignment (a la SLCA / SSIAT / MACIL).

The incrementally-trained head is biased toward the current task's classes — the
single biggest source of forgetting in PTM-CIL (we measured the head drop far
faster than the hull: 95->79 vs 92->83). The fix the SOTA methods all use:
after each task, *retrain the classifier over ALL seen classes* on pseudo-
features sampled from each class's stored Gaussian N(mu_c, Sigma_c). Because the
pseudo-samples are class-balanced, the retrained head is debiased.

This leaves the backbone and the conic hull untouched — the hull stays as the
feature-preserving boundary; the Gaussian-replay head does the closed-set
decision.
"""
import numpy as np
import torch
import torch.nn as nn


def sample_gaussian(mean: np.ndarray, cov: np.ndarray, n: int, rng) -> np.ndarray:
    """Draw n samples from N(mean, cov), robust to singular / low-rank cov.

    Uses an eigendecomposition square-root and clamps negative eigenvalues, so a
    rank-deficient covariance (e.g. estimated from few samples) still yields valid
    samples in its support instead of failing a Cholesky.
    """
    d = mean.shape[0]
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    A_sqrt = V * np.sqrt(w)                       # (D, D) columns scaled
    z = rng.standard_normal((n, d))
    return (mean[None, :] + z @ A_sqrt.T).astype(np.float32)


def align_linear_head(
    head: nn.Module,
    ca_stats: dict,
    class_to_idx: dict,
    device,
    samples_per_class: int = 256,
    epochs: int = 20,
    lr: float = 1e-2,
    batch_size: int = 1024,
    seed: int = 0,
) -> float:
    """Retrain `head` over all seen classes on Gaussian pseudo-features.

    Parameters
    ----------
    head         : IncrementalLinearHead (or any module mapping (N,D) features ->
                   (N, n_classes) logits; it may L2-normalise internally).
    ca_stats     : {class_id: {"mean": (D,), "cov": (D,D)}}  in feature space.
    class_to_idx : {class_id: head output index}.
    Returns the final training loss.
    """
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for cid, st in ca_stats.items():
        if cid not in class_to_idx:
            continue
        X = sample_gaussian(st["mean"].astype(np.float64),
                            st["cov"].astype(np.float64),
                            samples_per_class, rng)
        feats.append(X)
        labels.append(np.full(samples_per_class, class_to_idx[cid], dtype=np.int64))
    if not feats:
        return float("nan")

    X = torch.from_numpy(np.concatenate(feats)).to(device)
    y = torch.from_numpy(np.concatenate(labels)).to(device)

    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = X.shape[0]
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(head(X[idx]), y[idx])
            loss.backward()
            opt.step()
            last = float(loss.item())
    head.eval()
    return last
