#!/usr/bin/env python3
"""exp53_synthetic_material.py -- GMS-style synthetic features for the conic reader.

WHY THIS FILE EXISTS
    Two separate problems, one fix.

    PROBLEM 1: THE RAY BUDGET IS A FICTION ON HALF THE BENCHMARKS.
        b_kmeans clamps at `k = min(R_, len(Xw))` -- rays cannot exceed the number of
        points. Measured clamp_frac from the exp52 grid:

            ds          fit rows/cls   f32 delivered  clamp   f64 delivered  clamp
            CIFAR100        ~448           32.0       0.00        64.0       0.00
            IMAGENETR       ~101           32.0       0.00        60.8       0.23
            IMAGENETA        ~26           18.1       0.69        23.4       0.90
            CUB200           ~27           27.0       1.00        27.0       1.00

        So `f32` on IMAGENETA is an 18-RAY method, and on CUB200 `f32` and `f64` deliver
        the SAME 27 rays and differ only in the eigen-subspace dimension. Every
        cross-dataset ray-budget statement in exp52 is comparing different things, and
        "does R help" has never actually been asked on these two datasets.

    PROBLEM 2: THE FOREIGN SCATTER IS FED THE WORST AVAILABLE ESTIMATE.
        The oPCA rays are the top-R eigenvectors of S_c - g*S_F, and S_F is built as

            past = [A[a][o] for o in A[a] if o not in tasks[t]]     # STORED RAYS
            Fr   = np.concatenate([Ztr[oth]] + past, 0)

        Every past class contributes only its 8-64 EXTREME RAYS. Extreme rays are the
        atypical boundary of a cone, not typical samples, so S_F is badly biased. exp52
        measured that 66% of the fused gain comes from the linear combination over the
        rays, whose quality is set entirely by that discriminative direction selection --
        which is set entirely by S_F. This is the highest-leverage term in the method and
        it is being estimated from the least representative material available.

    Both are fixed by the same idea, borrowed from the Global Mismatch Suppression step of
    the framework in the reference figure: synthesise features from statistical class
    prototypes instead of reusing stored exemplars.

THE FOUR ARMS (a clean 2x2; ID and OOD synthesis are the same code path)

    real   real rows, stored rays as foreign material.  THE CONTROL -- reproduces exp52.
    ood    real rows, PAST-CLASS SAMPLES as foreign material.       fixes problem 2.
    id     real rows + OWN-CLASS SAMPLES, stored rays foreign.      fixes problem 1.
    both   the full GMS analogue.

STORAGE, WHICH IS THE WHOLE POINT OF THIS PROJECT
    ID synthesis costs NOTHING. It happens at class birth, when the real rows are still in
    hand; nothing is retained. It is not replay.
    OOD synthesis costs a prototype and one scalar per class -- 769 floats -- and REPLACES
    the R x 768 stored rays in the foreign set. At R=32 that is a 32x REDUCTION in what the
    foreign path stores. `bytes/class` is reported per arm so this is auditable rather than
    asserted.

    We do NOT store a per-class covariance. 768^2 floats/class is ~470MB over 200 classes
    and would break the claim outright. The pooled within-class scatter is already computed
    for the whitener and IS THE IDENTITY in whitened coordinates by construction, so the
    natural zero-cost class model there is N(mu_c, I) -- a prototype plus a scale.

THE TWO SAMPLERS, and why they differ
    OOD (`synth_ood`): isotropic around the stored prototype in whitened space. Isotropy is
        correct here -- and necessary. A per-class empirical covariance from ~26 rows in 768
        dims has rank <= 25, so samples drawn from it cannot leave that class's row space
        and would add no new directions to S_F. Isotropic whitened samples span all 768.
    ID (`synth_id`): the class's OWN empirical covariance, shrunk toward isotropic. Isotropy
        would be wrong here -- it would wash out exactly the anisotropy oPCA exists to find.
        Sampled as mu + (G @ Y)/sqrt(n), a random combination of the centred rows, which
        reproduces the empirical covariance exactly without ever forming a d x d matrix.

WHAT ID SYNTHESIS DOES AND DOES NOT CHANGE -- read this before believing a gain
    Synthetic own-class samples are drawn from the class's own (shrunk) covariance, so they
    do not move S_c and therefore do not move the oPCA DIRECTIONS. All they do is give
    k-means enough points to return R centroids instead of len(Xw). That is the intended
    and only mechanism: it unclamps the ray count without touching direction selection.
    If accuracy moves, it moved because of ray COUNT. If `mean_rays` does not rise, the
    arm did nothing and any accuracy delta is noise.

    HONEST PRIOR: this may well be a no-op. Gaussian samples carry no information beyond
    (mu, Sigma), so a finer k-means tiling of the same ellipsoid can just produce redundant
    rays -- and exp52 already found f32 -> f64 flat-to-negative on IMAGENETR where nothing
    was clamped. But it is the only route to R > n_rows on IMAGENETA and CUB200, where the
    question has never been asked, so it is worth the cells.

DECOUPLED R AND D, which is the other thing R was hiding
    b_opca uses R_ for TWO things: the eigen-subspace dimension (`k = min(R_, d)`, capped at
    768) and the k-means centroid count (capped at n_points). Only the second is limited by
    class size. `b_opca_rd` separates them, so `realR32D128` is 32 rays chosen inside a
    128-dimensional oriented subspace. D is nearly free -- the eigh is computed in full
    either way and cone_score cost scales with R, not D. Default D=R reproduces exp52.

THE CONTROL IS ASSERTABLE, NOT ARGUED
    `realR32` must reproduce exp52's `f32` cell bit-for-bit: same whitener, same splits,
    same k-means seeds, same foreign subsample. The foreign RNG is keyed on the ARM NAME via
    zlib.crc32, so a renamed arm silently draws a different subsample and nothing would
    match -- `_negkey` maps realR32 -> "f32" for exactly this reason. VERIFY=1 loads the
    exp52 JSON and asserts agreement to 1e-9. If that assert fires, this file is not
    measuring a delta against exp52 and none of its arms mean anything.

PIN YOUR THREADS. exp49 measured the unpinned noise floor at 0.27, larger than any effect
    here. The file refuses to start unless OMP/MKL are pinned (ALLOW_UNPINNED=1 to override).

READ IN THIS ORDER
    1. `mean_rays` and `clamp_frac`. If `id` did not raise mean_rays toward R, the mechanism
       did not fire and the accuracy columns are meaningless. This is a precondition, not a
       result.
    2. `ood - real` on cone A-Avg. The S_F hypothesis. A-Avg not A-Last: exp52 showed A-Last
       cannot resolve sub-0.5 effects at 3 seeds, and A-Avg detected a +0.17 that A-Last read
       as a coin.
    3. `id - real` ACROSS THE R SWEEP. A single R is uninterpretable; the claim is that
       accuracy now responds to R where before it was pinned at n_rows.
    4. `both - ood - id + real`, the interaction. If the two fixes are additive they are
       independent mechanisms; if `both` < the better single arm they are fighting.
    5. bytes/class, to confirm the storage claim.

USAGE
    source ~/venvs/ml_env/bin/activate

    # smoke (~5 min, own JSON)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CUB200 T=2 SEED=0 ARMS=realR32,idR32 SUFFIX=_smoke \
      python -u exp53_synthetic_material.py

    # THE CONTROL. Run this before anything else -- asserts repro against exp52.
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CUB200 T=10 SEED=0 ARMS=realR32 VERIFY=1 python -u exp53_synthetic_material.py

    # the 2x2 on the two clamped datasets
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CUB200,IMAGENETA T=10 SEED=0,1,2 ARMS=realR32,oodR32,idR32,bothR32 \
      python -u exp53_synthetic_material.py

    # the R sweep that ID synthesis makes askable for the first time
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CUB200,IMAGENETA T=10 SEED=0,1,2 ARMS=realR32,idR32,idR64,idR128 \
      python -u exp53_synthetic_material.py

    # the free axis: rays fixed, oriented subspace widened
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      DS=CUB200 T=10 SEED=0,1,2 ARMS=realR32,realR32D128,realR32D768 \
      python -u exp53_synthetic_material.py

    RULES=cone,sub drops the `pm` arm and about a third of the runtime.
    Cells are written as they finish and existing keys are skipped -- kill and rerun to
    resume. Run SEQUENTIALLY; concurrency on this box is measured strictly worse.
"""
import json
import os
import re
import time
import zlib

import numpy as np
import torch

_DS = os.environ.get("DS", "CUB200,IMAGENETA").split(",")
_TS = [int(x) for x in os.environ.get("T", "10").split(",")]
_SEEDS = [int(x) for x in os.environ.get("SEED", "0,1,2").split(",")]
os.environ["T"], os.environ["SEED"] = str(_TS[0]), str(_SEEDS[0])

import exp19_dataset_hull as E              # noqa: E402
import exp39_cone_construction as X         # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


REPO = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "augreg_in21k"
DSETS, TS, SEEDS = _DS, _TS, _SEEDS
ARMS = os.environ.get("ARMS", "realR32,oodR32,idR32,bothR32").split(",")
RULES = os.environ.get("RULES", "cone,sub,pm").split(",")
GAMMA = float(os.environ.get("GAMMA", 0.5))
F_MAX = int(os.environ.get("F_MAX", 2000))
SHRINK = float(os.environ.get("SHRINK", 3e-2))
M_RP = int(os.environ.get("MRP", 10000))
ITERS = int(os.environ.get("ITERS", 500))
N_OOD = int(os.environ.get("N_OOD", 32))        # synthetic samples per past class
ID_MULT = float(os.environ.get("ID_MULT", 4.0))  # target own-set size = ID_MULT * R
ID_ALPHA = float(os.environ.get("ID_ALPHA", 0.1))  # isotropic shrinkage in synth_id
VERIFY = int(os.environ.get("VERIFY", 0))
LAMBDAS = [1e2, 1e3, 1e4]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
OUT = os.path.join(REPO,
                   f"exp53_synthetic_material{os.environ.get('SUFFIX', '')}_{TAG}.json")
EXP52 = os.path.join(REPO, f"exp52_fusion_rule_control_{TAG}.json")
BAR = json.load(open(os.path.join(REPO, f"exp16_full_table_{TAG}.json")))

assert set(RULES) <= {"cone", "sub", "pm"}, f"unknown rule in {RULES}"
if not int(os.environ.get("ALLOW_UNPINNED", 0)):
    _th = [os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    assert _th == ["1", "1"], (
        f"threads not pinned (OMP={_th[0]} MKL={_th[1]}). eigh/KMeans reduce in a "
        f"thread-count-dependent order; exp49 measured the unpinned noise floor at 0.27, "
        f"larger than every effect here, and it would also break the exp52 repro assert. "
        f"Prefix with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, or set ALLOW_UNPINNED=1.")

un = X.un
_ARM_RE = re.compile(r"^(real|ood|id|both)R(\d+)(?:D(\d+))?$")


def parse_arm(a):
    """`{mode}R{R}[D{D}]`. D defaults to R, which is exactly what b_opca does."""
    m = _ARM_RE.match(a)
    assert m, (f"bad arm {a!r}; expected e.g. realR32, oodR32, idR64, bothR32, "
               f"realR32D128")
    mode, R_, D_ = m.group(1), int(m.group(2)), m.group(3)
    return mode, R_, int(D_) if D_ else R_


def _negkey(a):
    """RNG key for the foreign subsample.

    exp52 seeded this with zlib.crc32 of the arm NAME, so renaming an arm silently draws a
    different foreign subsample and the cell stops being comparable. `realR32` is exp52's
    `f32` and must hash to the same value or the VERIFY assert cannot pass. Every other arm
    keys on its own name, which is what we want -- different material, different draw."""
    mode, R_, D_ = parse_arm(a)
    return f"f{R_}" if (mode == "real" and D_ == R_) else a


def bar_for(ds, T, seed):
    v = BAR.get(f"{ds}|{T}|{seed}|ep40_lr0.0003_aug1")
    assert v is not None, f"no exp16 bar for {ds} T={T} s={seed}"
    return v


def zs(A, seen):
    """Row-wise z-score over SEEN columns, -inf mapped to the row min. Verbatim from
    exp35/exp52 so fused numbers stay comparable across all three files."""
    B = np.full(A.shape, -1e9, np.float64)
    sub = np.asarray(A[:, seen], np.float64)
    fin = np.isfinite(sub)
    sub = np.where(fin, sub, sub[fin].min() if fin.any() else 0.0)
    B[:, seen] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-8)
    return B


def score(rule, Ac, Q):
    if rule == "cone":
        return X.cone_score(Ac, Q, iters=ITERS)
    if rule == "pm":
        return (Q @ Ac.T).max(1)
    U, s, _ = np.linalg.svd(Ac.T, full_matrices=False)
    B = U[:, s > max(s[0], 1e-12) * 1e-6]
    return np.linalg.norm(Q @ B, axis=1)


# ------------------------------------------------------------------ synthesis
def synth_id(Xw, n_syn, rng, alpha=ID_ALPHA):
    """n_syn samples sharing Xw's empirical covariance, shrunk toward isotropic.

    mu + (G @ Y)/sqrt(n) with G standard normal reproduces Y^T Y / n exactly and never
    forms a d x d matrix -- which matters, because with ~26 rows in 768 dims the empirical
    covariance is rank <= 25 and any explicit factorisation is mostly noise. The alpha term
    adds isotropic jitter at the scale of the class's own dispersion so that k-means is not
    handed exact duplicates of the row space (which is what produces the sklearn
    'distinct clusters < n_clusters' warning and silently clamps the ray count again)."""
    if n_syn <= 0:
        return np.zeros((0, Xw.shape[1]), np.float32)
    n, d = Xw.shape
    mu = Xw.mean(0)
    Y = Xw - mu
    s = float(np.sqrt((Y ** 2).sum(1).mean())) if n else 0.0
    G = rng.standard_normal((n_syn, n)).astype(np.float32)
    Z = mu + (G @ Y) / np.sqrt(max(n, 1))
    if alpha > 0 and s > 0:
        Z = Z + (alpha * s / np.sqrt(d)) * rng.standard_normal((n_syn, d)).astype(np.float32)
    return un(Z.astype(np.float32))


def synth_ood(mu_w, s, n_syn, rng):
    """n_syn isotropic samples around a whitened unit prototype.

    Isotropic ON PURPOSE -- see the module docstring. The point of OOD material is to cover
    directions other classes actually occupy, and a rank-25 per-class covariance cannot
    leave its own row space. `s` is the class's mean displacement from its prototype,
    measured at birth and stored as ONE float; dividing by sqrt(d) makes the expected
    sample displacement equal s."""
    d = mu_w.shape[0]
    Z = mu_w[None, :] + (s / np.sqrt(d)) * rng.standard_normal((n_syn, d)).astype(np.float32)
    return un(Z.astype(np.float32))


def b_opca_rd(Xw, Fw, R_, D_, seed, g):
    """oPCA with the ray count R_ and the oriented-subspace dimension D_ DECOUPLED.

    Identical to X.b_opca when D_ == R_, which is the only way the exp52 repro can pass.
    D_ is capped at d (768) and NOT at len(Xw): the eigenvectors of S_c - g*S_F are defined
    for the full ambient dimension regardless of how few rows the class has. Only the
    k-means centroid count is limited by the number of points, and that is the cap ID
    synthesis exists to lift."""
    d = Xw.shape[1]
    M = (Xw.T.astype(np.float64) @ Xw) / len(Xw) - g * X._sf(Fw, d)
    k = int(min(D_, d))
    V = np.linalg.eigh(M)[1][:, ::-1][:, :k].astype(np.float32)
    return un(X.b_kmeans(Xw @ V, None, R_, seed, 0) @ V.T)


def run_cell(ds, T, seed, verify):
    E.T, E.SEED = T, seed
    F_ = E.adapted_features(ds)
    assert F_ is not None, f"no exp16 feature cache for {ds} T={T} s={seed}"
    Ztr, Zte = F_
    ytr, yte, n_cls = E.get_labels(ds)
    d = Ztr.shape[1]
    cpt = n_cls // T
    order = np.random.default_rng(seed).permutation(n_cls)
    tasks = [order[i * cpt:(i + 1) * cpt] for i in range(T)]

    FIT, VAL = [], []
    for t in range(T):
        ix = np.where(np.isin(ytr, tasks[t]))[0]
        pm_ = np.random.default_rng(t).permutation(len(ix))
        nv = max(int(0.1 * len(ix)), 1)
        VAL.append(ix[pm_[:nv]]); FIT.append(ix[pm_[nv:]])
    VAL_ALL = np.concatenate(VAL)

    P = torch.randn(d, M_RP, generator=torch.Generator().manual_seed(0)).to(DEV)
    G = torch.zeros(M_RP, M_RP, device=DEV, dtype=torch.float64)
    C = torch.zeros(M_RP, n_cls, device=DEV, dtype=torch.float64)
    eye = torch.eye(M_RP, device=DEV, dtype=torch.float64)

    def _H(Z, bs=4096):
        for i in range(0, len(Z), bs):
            yield i, torch.relu(torch.as_tensor(Z[i:i + bs], device=DEV,
                                                dtype=torch.float32) @ P)

    def logits(Z, Wm):
        return torch.cat([(h.double() @ Wm) for _, h in _H(Z)]).cpu().numpy()

    scatter = np.zeros((d, d), np.float64); n_scat = 0
    A = {a: {} for a in ARMS}          # stored rays, RAW space
    MU = {}                            # stored prototypes, RAW space (shared across arms)
    SP = {}                            # stored scalar dispersion, whitened units
    nray = {a: [] for a in ARMS}
    clamp = {a: [0, 0] for a in ARMS}
    res = {"ranpac": []}
    for a in ARMS:
        for r_ in RULES:
            res[f"{a}|{r_}"] = []
            res[f"{a}|fuse_{r_}"] = []
    beta_log = {f"{a}|{r_}": [] for a in ARMS for r_ in RULES}

    for t in range(T):
        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xc = Ztr[r] - Ztr[r].mean(0)
            scatter += Xc.T @ Xc; n_scat += len(Xc)
        S_ = scatter / max(n_scat, 1)
        S_ = S_ + SHRINK * np.trace(S_) / d * np.eye(d)
        Wh = np.linalg.cholesky(np.linalg.inv(S_)).astype(np.float32)
        Wh_inv = np.linalg.inv(Wh).astype(np.float32)

        for c in tasks[t]:
            r = FIT[t][ytr[FIT[t]] == c]
            if len(r) < 2:
                continue
            Xw_real = un(Ztr[r] @ Wh)
            # Prototype + dispersion, measured at birth in the CURRENT whitened frame but
            # stored RAW, exactly as the rays are. Storing whitened would freeze them into
            # a stale metric -- the whitener changes every stage.
            MU[int(c)] = Ztr[r].mean(0).astype(np.float32)
            mu_w0 = un(MU[int(c)][None, :] @ Wh)[0]
            SP[int(c)] = float(np.sqrt(((Xw_real - mu_w0) ** 2).sum(1).mean()))

            for a in ARMS:
                mode, R_, D_ = parse_arm(a)
                rng = np.random.default_rng(1234 + 97 * t
                                            + zlib.crc32(_negkey(a).encode()) % 1000)

                # ---- own material
                Xw = Xw_real
                if mode in ("id", "both"):
                    need = int(np.ceil(ID_MULT * R_)) - len(Xw_real)
                    Xw = np.concatenate([Xw_real, synth_id(Xw_real, need, rng)], 0) \
                        if need > 0 else Xw_real

                # ---- foreign material
                oth = FIT[t][~np.isin(ytr[FIT[t]], [c])]
                if mode in ("ood", "both"):
                    # Past classes contribute SAMPLES around their stored prototype,
                    # re-whitened with the CURRENT whitener, then pushed back to raw so the
                    # concatenation below stays in one space.
                    past = []
                    for o in MU:
                        if o in tasks[t]:
                            continue
                        mo = un(MU[o][None, :] @ Wh)[0]
                        past.append(synth_ood(mo, SP[o], N_OOD, rng) @ Wh_inv)
                else:
                    past = [A[a][o] for o in A[a] if o not in tasks[t]]
                Fr = np.concatenate([Ztr[oth]] + past, 0) if past else Ztr[oth]
                if len(Fr) > F_MAX:
                    Fr = Fr[rng.choice(len(Fr), F_MAX, replace=False)]
                Fw = un(Fr @ Wh) if GAMMA > 0 else np.zeros((0, d), np.float32)

                clamp[a][1] += 1
                clamp[a][0] += int(R_ > len(Xw))
                A[a][int(c)] = b_opca_rd(Xw, Fw, R_, D_, int(c), GAMMA) @ Wh_inv

        for i, h in _H(un(Ztr[FIT[t]])):
            h = h.double()
            Y = torch.zeros(h.shape[0], n_cls, device=DEV, dtype=torch.float64)
            Y[torch.arange(h.shape[0]),
              torch.tensor(ytr[FIT[t]][i:i + h.shape[0]], device=DEV)] = 1.0
            G += h.T @ h; C += h.T @ Y

        seen = np.concatenate(tasks[:t + 1])
        nval = sum(len(v) for v in VAL[:t + 1])
        vix = VAL_ALL[:nval]
        yv = ytr[vix]
        tei = np.where(np.isin(yte, seen))[0]
        yt = yte[tei]

        def acc(S, y):
            return float((np.asarray(seen)[S[:, seen].argmax(1)] == y).mean())

        best, bw = -1.0, None
        for lam in LAMBDAS:
            Wm = torch.linalg.solve(G + lam * eye, C)
            av = acc(logits(un(Ztr[vix]), Wm), yv)
            if av > best:
                best, bw = av, Wm
        Lv = logits(un(Ztr[vix]), bw)
        Lt = logits(un(Zte), bw)[tei]
        res["ranpac"].append(acc(zs(Lt, seen), yt))
        zLv, zLt = zs(Lv, seen), zs(Lt, seen)

        Qvw, Qtw = un(Ztr[vix] @ Wh), un(Zte[tei] @ Wh)
        for a in ARMS:
            miss = [c for c in seen if c not in A[a]]
            if miss:
                log(f"      s{t} {a}: {len(miss)} seen classes have NO rays -> "
                    f"neutralised in fusion, -inf raw")
            tot = 0
            for r_ in RULES:
                Sv = np.full((nval, n_cls), -np.inf, np.float32)
                St = np.full((len(tei), n_cls), -np.inf, np.float32)
                for c in seen:
                    if c not in A[a]:
                        continue
                    Ac = un(A[a][c] @ Wh)
                    Sv[:, c] = score(r_, Ac, Qvw)
                    St[:, c] = score(r_, Ac, Qtw)
                    if r_ == RULES[0]:
                        tot += len(Ac)
                res[f"{a}|{r_}"].append(acc(zs(St, seen), yt))
                zSv, zSt = zs(Sv, seen), zs(St, seen)
                if miss:
                    zSv[:, miss] = 0.0
                    zSt[:, miss] = 0.0
                b = max(BETAS, key=lambda bb: acc(zLv + bb * zSv, yv))
                beta_log[f"{a}|{r_}"].append(b)
                res[f"{a}|fuse_{r_}"].append(acc(zLt + b * zSt, yt))
            nray[a].append(tot / max(len(seen), 1))

        log(f"    s{t}: ranpac {res['ranpac'][-1]*100:.2f}   " + "   ".join(
            f"{a}[" + " ".join(f"{r_} {res[f'{a}|{r_}'][-1]*100:.2f}"
                               f"/{res[f'{a}|fuse_{r_}'][-1]*100:.2f}"
                               for r_ in RULES)
            + f" | rays {nray[a][-1]:.1f}]" for a in ARMS))

    del G, C, P, eye
    if DEV == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for k, v in res.items():
        assert all(0.0 <= x <= 1.0 for x in v), f"{k} out of range"
        out[k] = {"A_last": v[-1], "A_avg": float(np.mean(v)), "accs": v}
    for a in ARMS:
        mode, R_, D_ = parse_arm(a)
        # Storage the FOREIGN path needs per past class: rays are R x d floats; synthetic
        # OOD needs a prototype (d) plus one scalar. ID synthesis retains nothing.
        per_cls = (769 if mode in ("ood", "both") else float(np.mean(nray[a])) * d + 1)
        out[f"{a}|_rays"] = {"mean_rays": float(np.mean(nray[a])),
                             "clamp_frac": clamp[a][0] / max(clamp[a][1], 1),
                             "R": R_, "D": D_, "mode": mode,
                             "foreign_floats_per_class": float(per_cls)}
    for k, v in beta_log.items():
        out[f"{k}|_beta"] = {"betas": v, "beta_last": v[-1]}

    if verify:
        b = bar_for(ds, T, seed)
        assert abs(res["ranpac"][-1] - b["A_last"]) < 1e-6, (
            f"recomputed RanPAC {res['ranpac'][-1]:.6f} != exp16 bar {b['A_last']:.6f}; "
            f"the replay protocol does not match exp16 and nothing here is comparable")
        log("    VERIFY ok: RanPAC matches the exp16 bar")
        if os.path.exists(EXP52) and "realR32" in ARMS:
            d52 = json.load(open(EXP52))
            hit = [v for k, v in d52.items()
                   if k.startswith(f"{ds}|{T}|{seed}|") and "f32|cone" in v]
            if hit and "cone" in RULES:
                got = out["realR32|cone"]["A_last"]
                want = hit[0]["f32|cone"]["A_last"]
                assert abs(got - want) < 1e-9, (
                    f"realR32 cone A_last {got:.10f} != exp52 f32 {want:.10f}. The control "
                    f"arm does not reproduce exp52, so every delta in this file is measured "
                    f"against a moving baseline. Check _negkey and thread pinning before "
                    f"reading anything else.")
                log(f"    VERIFY ok: realR32 reproduces exp52 f32 exactly ({got:.6f})")
            else:
                log("    VERIFY skipped exp52 repro: no matching f32 cell cached yet")
    return out


if __name__ == "__main__":
    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    first = True
    for ds in DSETS:
        for T in TS:
            for seed in SEEDS:
                key = (f"{ds}|{T}|{seed}|{'+'.join(ARMS)}|{'+'.join(RULES)}"
                       f"|g{GAMMA:g}_f{F_MAX}_s{SHRINK:g}_i{ITERS}"
                       f"|nood{N_OOD}_idm{ID_MULT:g}_ida{ID_ALPHA:g}|m{M_RP}|v1")
                if key in allres:
                    log(f"skip {key}"); continue
                log(f"=== {key}")
                allres[key] = run_cell(ds, T, seed, VERIFY and first)
                first = False
                json.dump(allres, open(OUT, "w"), indent=2)

    W = 100
    cells = {}
    for k, v in allres.items():
        p = k.split("|")
        if len(p) < 5 or p[3] != "+".join(ARMS) or p[4] != "+".join(RULES):
            continue
        cells[(p[0], int(p[2]))] = v

    def g(v, n, f):
        return v[n][f] * 100 if n in v else float("nan")

    print("\n" + "=" * W)
    print("EXP53 — synthetic ID / OOD material for the conic reader")
    print("=" * W)
    print(f"\narms {ARMS}   rules {RULES}   N_OOD {N_OOD}   ID_MULT {ID_MULT}   "
          f"cells {len(cells)}")

    print(f"\n{'-'*W}\nPRECONDITION — did the mechanism fire? (read this first)\n{'-'*W}")
    print(f"  {'ds':<10}{'seed':<5}{'arm':<14}{'R':>5}{'D':>5}{'rays':>8}{'clamp':>8}"
          f"{'foreign floats/cls':>21}")
    for (ds, seed), v in sorted(cells.items()):
        for a in ARMS:
            r = v.get(f"{a}|_rays")
            if r:
                print(f"  {ds:<10}{seed:<5}{a:<14}{r['R']:>5}{r['D']:>5}"
                      f"{r['mean_rays']:>8.1f}{r['clamp_frac']:>8.2f}"
                      f"{r['foreign_floats_per_class']:>21,.0f}")
    print("  An `id` arm whose mean_rays did not rise toward R did NOT fire; its accuracy")
    print("  delta is noise and must not be reported as a result.")

    for fld, lbl in (("A_last", "A-Last"), ("A_avg", "A-Avg")):
        print(f"\n{'-'*W}\n{lbl}\n{'-'*W}")
        hdr = f"  {'ds':<10}{'seed':<5}{'ranpac':>8}"
        for a in ARMS:
            for r_ in RULES:
                hdr += f"{a+':'+r_:>16}{a+':f_'+r_:>16}"
        print(hdr)
        for (ds, seed), v in sorted(cells.items()):
            row = f"  {ds:<10}{seed:<5}{g(v,'ranpac',fld):>8.2f}"
            for a in ARMS:
                for r_ in RULES:
                    row += (f"{g(v,f'{a}|{r_}',fld):>16.2f}"
                            f"{g(v,f'{a}|fuse_{r_}',fld):>16.2f}")
            print(row)

        base = "realR32"
        if base in ARMS:
            print(f"\n  PAIRED vs {base}, mean +/- sd over {len(cells)} cells")
            print(f"  {'contrast':<34}{'mean':>9}{'sd':>9}{'wins':>8}   per-dataset")
            for a in ARMS:
                if a == base:
                    continue
                for r_ in RULES:
                    for pre in ("", "fuse_"):
                        dl = {}
                        for (ds, seed), v in cells.items():
                            hi, lo = f"{a}|{pre}{r_}", f"{base}|{pre}{r_}"
                            if hi in v and lo in v:
                                dl.setdefault(ds, []).append(g(v, hi, fld) - g(v, lo, fld))
                        flat = [x for xs in dl.values() for x in xs]
                        if not flat:
                            continue
                        sd = float(np.std(flat, ddof=1)) if len(flat) > 1 else float("nan")
                        per = "  ".join(f"{k}{np.mean(x):+.2f}" for k, x in sorted(dl.items()))
                        print(f"  {a+' '+pre+r_+' - '+base:<34}{np.mean(flat):>+9.2f}"
                              f"{sd:>9.2f}{sum(x>0 for x in flat):>5}/{len(flat):<3}   {per}")

    print("\n" + "-" * W)
    print("""HOW TO READ THIS
  1. THE PRECONDITION TABLE IS NOT OPTIONAL. `id` arms exist to raise mean_rays. If rays
     did not move, nothing was tested and the accuracy delta is a seed draw.
  2. `ood - real` on cone A-Avg is the S_F hypothesis -- the one with a mechanism argument
     behind it (66% of exp52's fused gain runs through the discriminative direction
     selection, which runs entirely through S_F). Use A-Avg: exp52 showed A-Last cannot
     resolve sub-0.5 effects at three seeds and read a real +0.17 as a coin.
  3. `id - real` is only interpretable ACROSS AN R SWEEP. One R tells you nothing; the
     claim is that accuracy now RESPONDS to R where it was previously pinned at n_rows.
  4. Synthetic own-class samples share the class covariance by construction, so they do not
     move the oPCA directions -- only the k-means centroid count. If `id` changes accuracy
     without changing mean_rays, something is wrong with the sampler, not with the theory.
  5. foreign floats/class: `ood` should read 769 against `real`'s R*768+1. If the storage
     column does not favour ood, the zero-storage argument for this file is void.""")
    print("=" * W)
    log(f"wrote {OUT}")
