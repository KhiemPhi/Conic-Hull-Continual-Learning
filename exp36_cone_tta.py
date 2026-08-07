#!/usr/bin/env python3
"""exp36_cone_tta.py — test-time adaptation of the cone by residual pricing.

THE QUESTION
    Every conic result so far has been rule-neutral: exp35's decisive ablation was
    fuse_pm_eig 80.73 vs fuse_eig 80.80, i.e. the conic rule contributes +0.07 fused,
    inside the beta-selection noise band.  Anything that improves the ATOMS improves
    max-cosine just as much, so the cone is interchangeable.

    Residual pricing is the one mechanism that CANNOT be run without the conic rule.
    That makes it the last real test of whether the cone is load-bearing.

THE MECHANISM (why it is conic and not generic)
    A cone C(A) = {A^T w : w >= 0} depends only on the DIRECTIONS of its rays --
    rescaling a ray leaves the cone bit-identical.  So the entire edit space is
    {add a ray, delete a ray, rotate a ray}; there is no reweighting.

    The NNLS you already solve for the score returns the certificate for the "add":
        w = argmin_{w>=0} ||q - A^T w||,     r = q - A^T w,     and KKT gives  A r <= 0.
    Every existing ray makes a non-acute angle with r, so r is exactly the direction
    the cone cannot reach.  For a unit candidate ray g the first-order score gain is
    the column-generation reduced cost
        delta(s^2) >= (g^T r)_+^2                (equality when g _|_ the active face)
    so the pricing problem is already solved -- r is free.

    ONE-SIDEDNESS is the native part.  Appending g to a SUBSPACE basis adds span{g},
    i.e. +g and -g.  Appending g to a cone adds only the ray R_+ g.  A cone can absorb
    a directional test-time shift WITHOUT inflating the class region in the opposite
    direction; a subspace, a Gaussian and a covariance ellipsoid all inflate
    symmetrically.  max-cosine has no residual at all -- its leftover is orthogonal to
    nothing, so there is no principled direction to append.

    DELETION is the only shrink operation a cone has (you cannot scale a ray down).
    Rays of class c whose NNLS mass comes mostly from queries predicted as some other
    class are absorbing the wrong region; deleting them shrinks c more for competitors
    than for c.  This targets the overlap diagnosis of A-Last directly.
    Add + delete together rotate the cone toward the discriminative region without any
    global map -- which matters, because global transport is already closed in both
    directions.

THE ATTRIBUTION DESIGN — this is the whole point of the file
    2 rules x 4 edits, fully crossed, so "conic" is a measured claim not an assertion:

                    none          mean            resid           prune
        pm      base max-cos   +test-mean atom  +residual atom  drop leaky atom
        cone    base conic     +test-mean atom  +residual atom  drop leaky ray

    NATIVE iff   (cone,resid)-(cone,mean)  >>  (pm,resid)-(pm,mean).
    If the two differences match, the certificate is doing nothing the generic
    test-time class mean does not already do, and the mechanism is dead like the rest.

    For pm the weight vector W is defined as the one-hot argmax scaled by (q^T a*)_+,
    so W @ A is the rank-1 projection onto the best ray and r = q - W @ A is a genuine
    residual.  Every downstream formula (pricing, leak mass, pruning) is then literally
    the same code for both rules -- no rule gets a hand-tuned variant.

PROTOCOL — read this before quoting a number
    MODE=trans   two passes over the whole test set: score, edit, rescore.  This is
                 TRANSDUCTIVE and is NOT protocol-legal -- it uses the entire test set
                 before answering any of it.  It is here because it is the cheap
                 upper bound: if the mechanism does not work transductively it cannot
                 work online, so falsify here first.  Default.
    MODE=online  single pass in chunks; each chunk is scored with state built only from
                 strictly earlier chunks.  This is what PLASTIC does and it is the only
                 number that may be reported as the headline.  Chunk 1 gets no benefit,
                 so the online gain is always <= the transductive gain.

    TTA state is rebuilt from scratch at every stage and never uses labels.  The val
    stream gets its own independent TTA pass (val is unlabeled as far as TTA is
    concerned), so beta is selected under the same distribution it is applied to.

RUNTIME (measured, ImageNet-R T=10, R=5, on this box)
    One full cone pass at stage 9 costs 52 s (test, 6000 rows x 200 classes) + 35 s (val).
    FISTA is launch-overhead bound at these sizes -- a 512-row call costs ~60% of a
    6000-row one -- so cost tracks the NUMBER of chunks, not the rows in them.
        MODE=trans                 ~45-60 min   (2 passes per TTA arm)
        MODE=online CHUNK=2048     ~1 h         (3 chunks at stage 9; 1/3 of the stream
                                                 is scored before any adaptation exists)
        MODE=online CHUNK=1024     ~2 h         (halves the untreated fraction)
    The untreated head of the stream is the honesty tax and is why online <= trans.

REGRESSION CHECKS — if either fails, nothing else on the page means anything
    ranpac    must reproduce the exp16 bar   (ImageNet-R T=10 s0 A-Last: 80.28)
    cone_none must reproduce exp35's cone_eig at the same R/NK/NE/SHRINK  (79.80)

USAGE
    source ~/venvs/ml_env/bin/activate
    DS=IMAGENETR T=10 SEED=0 python -u exp36_cone_tta.py                 # trans, R=4
    DS=IMAGENETR T=10 SEED=0 GUARD=0 python -u exp36_cone_tta.py         # unguarded
    DS=IMAGENETR T=10 SEED=0 MODE=online python -u exp36_cone_tta.py     # the legal one
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

import exp19_dataset_hull as E
from conic_hull import ConicHull

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DSETS = os.environ.get("DS", "IMAGENETR").split(",")
TS = [int(x) for x in os.environ.get("T", "10").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEED", "0").split(",")]

# ---- atoms: exp35's budget arithmetic, so cone_none is a paired regression check
R = int(os.environ.get("R", 4))
NK = int(os.environ.get("NK", 2))
NE = int(os.environ.get("NE", 1))
ALPHA = float(os.environ.get("ALPHA", 0.5))
assert NK + 2 * NE == R, f"budget mismatch: nk={NK} + 2*ne={NE} != R={R}"

# ---- TTA knobs
MODE = os.environ.get("MODE", "trans")             # trans (upper bound) | online (legal)
CHUNK = int(os.environ.get("CHUNK", 2048))         # online chunk size; unused in trans
TAU = float(os.environ.get("TAU", 0.5))            # accept margins above this QUANTILE
NMIN = int(os.environ.get("NMIN", 3))              # min accepted rows before any edit
GUARD = int(os.environ.get("GUARD", 1))            # gain-minus-leak veto on the added ray
KAPPA = float(os.environ.get("KAPPA", 0.0))        # add iff gain - leak > KAPPA
PLEAK = float(os.environ.get("PLEAK", 1.5))        # prune rays whose leak SHARE RATIO exceeds

# ---- read-out / metric (identical to exp35)
M_RP = int(os.environ.get("MRP", 10000))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
SHRINK = float(os.environ.get("SHRINK", 3e-2))
ITERS = int(os.environ.get("ITERS", 500))
OUT = os.path.join(REPO, f"exp36_cone_tta_{TAG}.json")

RULES = ["pm", "cone"]
EDITS = ["none", "mean", "resid", "rand", "prune"]
assert MODE in ("trans", "online"), MODE


def zs(A, seen):
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def km(X, k, seed):
    k = int(min(k, len(X)))
    return E.un(X.mean(0, keepdims=True) if k <= 1 else
                KMeans(k, n_init=4, random_state=seed).fit(X).cluster_centers_)


def build_gens(Xw, nk, ne, alpha, c):
    """exp35's generator set verbatim: nk centroids + ne eigen-displaced pairs."""
    parts = [km(Xw, nk, c)] if nk > 0 else []
    if ne > 0:
        mu = Xw.mean(0)
        Y = Xw - mu
        sv, Vt = np.linalg.svd(Y, full_matrices=False)[1:]
        lam = (sv ** 2) / max(len(Y) - 1, 1)
        for j in range(min(ne, len(lam))):
            step = alpha * np.sqrt(max(lam[j], 0.0)) * Vt[j]
            parts.append(np.stack([mu + step, mu - step]))
    return E.un(np.concatenate(parts, 0))


def solve(A, Q, rule):
    """Score unit queries Q against unit rays A, returning (s, W).

    W >= 0 with W @ A the reconstruction, so r = Q - W @ A is the residual for BOTH
    rules.  cone: FISTA NNLS over the full ray set (bit-identical to exp35's
    cone_score).  pm: one-hot argmax scaled by the cosine, i.e. the rank-1 projection
    onto the single best ray.  Sharing the (s, W) interface is what lets the pricing /
    leak / prune code be literally the same for both rules -- the attribution would be
    worthless if either rule got a hand-tuned variant.
    """
    An = E.un(A)
    if rule == "cone":
        h = ConicHull(n_rays=len(An), nnls_iters=ITERS)
        h.extreme_rays_ = An
        W = h.reconstruct(Q).astype(np.float32)
        s = (np.asarray(Q, np.float32) * E.un(W @ An)).sum(1)
        return s.astype(np.float32), W
    Cs = np.asarray(Q, np.float32) @ An.T
    j = Cs.argmax(1)
    s = Cs[np.arange(len(Q)), j]
    W = np.zeros_like(Cs)
    W[np.arange(len(Q)), j] = np.maximum(s, 0.0)     # so W @ An = (q^T a*)_+ a*
    return s, W


class TTAState:
    """Per-class accumulators for one stream.  Every edit is a pure function of these,
    so `trans` and `online` differ ONLY in how many chunks feed them before a score."""

    def __init__(self, base, edit):
        self.base, self.edit = base, edit
        self.acc = {}                 # (d,) running sum of residuals or of queries
        self.n = {}                   # accepted rows so far
        self.own, self.oth = {}, {}   # (K,) base-aligned mass, own- vs other-predicted
        self.veto = {}                # guard verdict from the most recent chunk
        self.coh_num = self.coh_den = 0.0
        self.used_n = self.used_d = 0.0   # how often the appended ray is actually used

    def atoms(self, c):
        """(rays, idx) with idx[j] the BASE row of ray j, or -1 for an appended ray.

        idx is what keeps the leak accounting honest: under `prune` the solved weight
        vector is SHORTER than the base ray set and under `mean`/`resid` it is LONGER,
        so callers must scatter W back to base width before touching self.own/self.oth.
        """
        A = self.base[c]
        ident = np.arange(len(A))
        if self.edit == "none" or self.n.get(c, 0) < NMIN:
            return A, ident
        if self.edit == "prune":
            # Leak must be measured as a SHARE RATIO, not a raw fraction.  With 200 seen
            # classes only ~1/200 of queries are predicted c, so raw oth/(own+oth) exceeds
            # any fixed threshold for every ray and the edit either deletes everything or
            # (with a keep>=1 fallback) silently becomes a no-op.  The prior-free quantity
            # is how over-represented a ray is among impostors relative to its own class:
            #     ratio_j = (oth_j / sum oth) / (own_j / sum own)
            # ratio > 1 means the ray pulls in rivals harder than it serves its own class.
            ow, ot = self.own[c], self.oth[c]
            so = ot / max(ot.sum(), 1e-8)
            sw = ow / max(ow.sum(), 1e-8)
            ratio = so / np.maximum(sw, 1e-8)
            keep = ratio <= PLEAK
            if keep.sum() < 1:                       # keep the single least-leaky ray --
                keep = np.zeros(len(A), bool)        # reverting to the full set would make
                keep[int(ratio.argmin())] = True     # the arm a silent no-op
            return A[keep], ident[keep]
        if self.veto.get(c, True):
            return A, ident
        g = self.acc[c]
        nrm = float(np.linalg.norm(g))
        if nrm < 1e-8:
            return A, ident
        return (np.concatenate([A, (g / nrm)[None].astype(np.float32)], 0),
                np.concatenate([ident, [-1]]))

    def update(self, Qc, S, Wf, Af, Ix, cls):
        """Absorb one scored chunk.  Uses NO labels -- only the model's own argmax."""
        arr = np.asarray(cls)
        sub = S[:, arr]
        pred = arr[sub.argmax(1)]
        if sub.shape[1] >= 2:
            t2 = np.partition(sub, -2, axis=1)[:, -2:]     # [:,1] is the max, [:,0] the 2nd
            marg = t2[:, 1] - t2[:, 0]
        else:
            marg = np.zeros(len(Qc), np.float32)
        thr = np.quantile(marg, TAU, method="higher")
        ok = marg >= thr
        for c in cls:
            m = ok & (pred == c)
            if self.edit == "prune":
                K = len(self.base[c])
                sel = Ix[c] >= 0
                wb = np.zeros((len(Qc), K), np.float32)
                wb[:, Ix[c][sel]] = Wf[c][:, sel]          # scatter back to base width
                self.own[c] = self.own.get(c, np.zeros(K, np.float32)) + wb[pred == c].sum(0)
                self.oth[c] = self.oth.get(c, np.zeros(K, np.float32)) + wb[pred != c].sum(0)
                self.n[c] = self.n.get(c, 0) + int(m.sum())
                continue
            if m.sum() == 0:
                continue
            Qm = Qc[m]
            if self.edit == "rand":
                # Budget-matched null direction: one appended unit ray, same gating and
                # same guard as `resid`, but carrying no information. cone_resid - cone_rand
                # is the ONLY clean test of whether the KKT certificate's DIRECTION matters,
                # because it is measured inside the conic rule at a fixed ray count.
                self.acc[c] = np.random.default_rng(1000 + int(c)).normal(
                    size=self.base[c].shape[1]).astype(np.float32)   # fixed, not accumulated
            else:
                if self.edit == "resid":
                    rr = Qm - Wf[c][m] @ Af[c]             # exact: full current ray set
                    self.coh_num += float(np.linalg.norm(rr.sum(0)))
                    self.coh_den += float(np.linalg.norm(rr, axis=1).sum())
                    g = rr.sum(0)
                else:
                    g = Qm.sum(0)
                self.acc[c] = self.acc.get(c, 0.0) + g     # accumulate first, veto later,
            self.n[c] = self.n.get(c, 0) + int(m.sum())    # so a vetoed chunk is not lost
            nrm = float(np.linalg.norm(self.acc[c]))
            if not GUARD:
                self.veto[c] = False
            elif nrm < 1e-8:
                self.veto[c] = True
            else:
                u = self.acc[c] / nrm
                gain = float(np.maximum(Qm @ u, 0).mean())
                oq = ok & (pred != c)
                leak = float(np.maximum(Qc[oq] @ u, 0).mean()) if oq.any() else 0.0
                self.veto[c] = (gain - leak) <= KAPPA


def run_stream(base, Qw, seen, n_cls, rule, edit):
    """Score one stream under one (rule, edit).  Returns (S, n_edited, coherence)."""
    st = TTAState(base, edit)
    S = np.full((len(Qw), n_cls), -np.inf, np.float32)
    cls = [c for c in seen if c in base]

    def score_block(sl, tally=False):
        Wf, Af, Ix = {}, {}, {}
        for c in cls:
            A, idx = st.atoms(c)
            s, w = solve(A, Qw[sl], rule)
            S[sl, c] = s
            Wf[c], Af[c], Ix[c] = w, E.un(A), idx
            if tally and len(idx) and idx[-1] == -1:
                # USED = fraction of queries whose reconstruction actually puts weight on
                # the appended ray.  This is the check that catches a DEAD CONTROL: under
                # max-cosine an appended ray only registers if it becomes the single
                # closest atom, and a residual direction is near-orthogonal to the queries
                # by construction, so pm_resid collapses onto pm_none exactly.  A cross-rule
                # comparison against a rule that cannot consume the atom is not a control.
                st.used_n += float((w[:, -1] > 1e-6).sum())
                st.used_d += float(w.shape[0])
        return Wf, Af, Ix

    full = slice(0, len(Qw))
    if edit == "none":
        score_block(full)
        return S, 0, 0.0, 0.0
    if MODE == "trans":
        Wf, Af, Ix = score_block(full)                     # pass 1: base cones
        st.update(Qw, S, Wf, Af, Ix, cls)                  # one edit from the whole stream
        score_block(full, tally=True)                      # pass 2: rescore with the edits
    else:
        for i in range(0, len(Qw), CHUNK):
            sl = slice(i, min(i + CHUNK, len(Qw)))         # strictly-earlier state only
            Wf, Af, Ix = score_block(sl, tally=True)
            st.update(Qw[sl], S[sl], Wf, Af, Ix, cls)
    n_ed = sum(1 for c in cls if len(st.atoms(c)[0]) != len(base[c]))
    return (S, n_ed, (st.coh_num / st.coh_den if st.coh_den > 0 else 0.0),
            (st.used_n / st.used_d if st.used_d > 0 else 0.0))


def run_cell(ds, T, seed):
    E.T, E.SEED = T, seed
    assert (E.T, E.SEED) == (T, seed)
    F = E.adapted_features(ds)
    assert F is not None, f"no exp16 cache for {ds} T={T} s={seed}"
    Ztr, Zte = F
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]
    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm[:nv]]); FIT.append(ix[pm[nv:]])
    VAL_ALL = np.concatenate(VAL)
    Qv, Qt = Ztr[VAL_ALL], Zte

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)

    def _H(X, bs=4096):
        for i in range(0, len(X), bs):
            yield i, torch.relu(torch.as_tensor(X[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def logits(X, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(X)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    Aorig = {}
    arms = ["ranpac"] + [f"{p}{r}_{e}" for p in ("", "f_") for r in RULES for e in EDITS]
    res = {a: [] for a in arms}
    diag = []

    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        Sc = scatter / max(n_scat, 1)
        Sc = Sc + SHRINK * np.trace(Sc) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(Sc)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)
        for c in tasks[t]:                            # atoms are born in the birth metric
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Aorig[c] = build_gens(E.un(Ztr[r] @ Wh), NK, NE, ALPHA, c) @ Wh_inv

        for i, h in _H(E.un(Ztr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y
        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        yv = ytr[VAL_ALL[:nval]]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(Z, y):
            return float((np.asarray(seen)[Z[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            a = acc(logits(E.un(Qv[:nval]), Wm), yv)
            if a > best:
                best, bw = a, Wm
        Lv, Lt = logits(E.un(Qv[:nval]), bw), logits(E.un(Qt), bw)[tei]
        zLv, zLt = zs(Lv, seen), zs(Lt, seen)
        res["ranpac"].append(acc(zLt, yt))

        Qvw, Qtw = E.un(Qv[:nval] @ Wh), E.un(Qt[tei] @ Wh)
        base = {c: E.un(Aorig[c] @ Wh) for c in seen if c in Aorig}

        row = {}
        for rule in RULES:
            for edit in EDITS:
                Sv, _, _, _ = run_stream(base, Qvw, seen, n_cls, rule, edit)
                St, ned, coh, used = run_stream(base, Qtw, seen, n_cls, rule, edit)
                res[f"{rule}_{edit}"].append(acc(zs(St, seen), yt))
                zSv, zSt = zs(Sv, seen), zs(St, seen)
                b = max(BETAS, key=lambda bb: acc(zLv + bb * zSv, yv))
                res[f"f_{rule}_{edit}"].append(acc(zLt + b * zSt, yt))
                row[f"{rule}_{edit}"] = {"edited": ned, "coh": round(coh, 4),
                                         "used": round(used, 4), "beta": b,
                                         "n_cls": len(base)}
        diag.append(row)
        cert = (res["cone_resid"][-1] - res["cone_rand"][-1]) * 100
        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}  |  "
            + "  ".join(f"{r}:" + " ".join(f"{e[:2]} {res[f'{r}_{e}'][-1]*100:.2f}"
                                           for e in EDITS) for r in RULES)
            + f"  |  CERT {cert:+.2f}"
            + f"  [ed {row['cone_resid']['edited']}/{len(base)}"
              f"  used {row['cone_resid']['used']:.2f}"
              f"  pm-used {row['pm_resid']['used']:.2f}"
              f"  coh {row['cone_resid']['coh']}]")

    del G, C, P, eye
    torch.cuda.empty_cache()
    for a, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{a} out of range"
    return {"arms": {a: {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
                     for a, v in res.items()},
            "diag": diag}


allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in DSETS:
    for T in TS:
        for seed in SEEDS:
            key = (f"{ds}|{T}|{seed}|R{R}_k{NK}e{NE}_a{ALPHA:g}"
                   f"|{MODE}_c{CHUNK}_t{TAU:g}_n{NMIN}_g{GUARD}_k{KAPPA:g}_p{PLEAK:g}"
                   f"|m{M_RP}_s{SHRINK:g}_i{ITERS}|v1")
            if key in allres:
                log(f"skip {key}"); continue
            log(f"=== {key}")
            allres[key] = run_cell(ds, T, seed)
            json.dump(allres, open(OUT, "w"), indent=2)

W_ = 88
print("\n" + "=" * W_)
print("EXP36 — cone TTA by residual pricing")
print("=" * W_)
for key, blob in allres.items():
    r = blob["arms"]
    g = lambda k: r[k]["A_last"] * 100
    print(f"\n--- {key}")
    print(f"{'A-Last':<9}" + "".join(f"{e:>11}" for e in EDITS) + f"{'ranpac':>11}")
    for lab, pre in (("raw", ""), ("fus", "f_")):
        for rule in RULES:
            print(f"{lab+' '+rule:<9}" + "".join(f"{g(pre+rule+'_'+e):>11.2f}" for e in EDITS)
                  + (f"{g('ranpac'):>11.2f}" if (lab, rule) == ("raw", "pm") else ""))
    print("\n  TTA gain over 'none' (raw):")
    for rule in RULES:
        print(f"    {rule:<5}" + "".join(
            f"   {e} {g(rule+'_'+e)-g(rule+'_none'):+.2f}" for e in EDITS[1:]))
    print(f"\n  CERTIFICATE = cone_resid - cone_rand = {g('cone_resid')-g('cone_rand'):+.2f}"
          "     <-- HEADLINE (within-rule, ray-count matched)")
    print(f"  vs generic  = cone_resid - cone_mean = {g('cone_resid')-g('cone_mean'):+.2f}")
    print("  RULE at fixed edit (cone - pm):  " + "  ".join(
        f"{e} {g('cone_'+e)-g('pm_'+e):+.2f}" for e in EDITS))
    for e in EDITS[1:]:
        d = blob["diag"][-1]
        print(f"    {e:<6} used: cone {d['cone_'+e]['used']:.3f}  pm {d['pm_'+e]['used']:.3f}"
              f"   edited cone {d['cone_'+e]['edited']}/{d['cone_'+e]['n_cls']}")
    d0 = blob["diag"][-1]["cone_resid"]
    print(f"  last stage: residual coherence {d0['coh']},  beta {d0['beta']}")
print("\n" + "-" * W_)
print("REGRESSION: ranpac must be the exp16 bar (IN-R T=10 s0: 80.28) and cone_none must")
print("   equal exp35's cone_eig at the same R/NK/NE/SHRINK (79.80). Both are paired")
print("   re-computations of known cells; if either drifts the replay is broken.")
print("CERTIFICATE = cone_resid - cone_rand is the headline. Both append exactly ONE unit")
print("   ray under identical gating, so the ray count is matched and the only difference")
print("   is the DIRECTION. <= 0 means the KKT certificate is worth no more than a random")
print("   ray and the mechanism is dead. cone_resid - cone_mean is secondary: does the")
print("   certificate beat the generic test-time class mean.")
print("The cross-rule difference (cone_e - pm_e) is NOT a control for resid/rand. Under")
print("   max-cosine an appended ray changes nothing unless it becomes the single closest")
print("   atom, and a near-orthogonal residual direction never does -- pm_resid collapses")
print("   onto pm_none exactly. Read the `used` line: pm used ~0 means that cell is")
print("   structurally dead, not informative, and any cone-minus-pm margin built on it is")
print("   an artifact of the rule's inability to consume the atom, NOT evidence that the")
print("   cone is native. pm_mean and pm_prune remain valid controls.")
print("COHERENCE = ||sum r|| / sum||r|| over accepted residuals.  Near 0 means the")
print("   residuals cancel -- there is no single missing direction, so pricing cannot work")
print("   regardless of accuracy.  That is a cleaner negative than the accuracy is.")
print("EDITED near 0 means the GUARD vetoed, not that the mechanism failed -- re-run with")
print("   GUARD=0 before concluding anything from a flat resid column.")
print("MODE=trans is TRANSDUCTIVE and NOT protocol-legal; it is the cheap upper bound.")
print("   Re-run MODE=online before quoting anything; online <= trans by construction.")
print("=" * W_)
print(f"wrote {OUT}")
