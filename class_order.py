#!/usr/bin/env python3
"""class_order.py -- THE single source of truth for the CIL class order.

WHY THIS FILE EXISTS
    exp16_full_table.py's docstring claimed:

        "Class order = rng(SEED).permutation(n_classes), the PyCIL/LAMDA-PILOT convention."

    That is wrong, and the claim propagated into exp54/exp55/exp56 because they all copied
    the line rather than the convention. PyCIL and LAMDA-PILOT do this, in
    DataManager._setup_data:

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)                          # LEGACY MT19937 global RandomState
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order                     # iImageNetR: np.arange(200)
        self._train_targets = _map_new_class_index(self._train_targets, order)

    `np.random.seed` + `np.random.permutation` draws from the legacy MT19937 RandomState.
    `np.random.default_rng(seed)` is PCG64. Same seed, unrelated streams. Measured overlap of
    our task 0 with PILOT's on IMAGENETR T=10: 3/20, 0/20, 2/20 at seeds 0/1/2 -- essentially
    disjoint first sessions. For a method that does ALL of its adaptation in the first
    session, no other protocol detail matters as much.

    Targets are remapped to POSITIONS in `order`, so task t is order[t*cpt:(t+1)*cpt]. That
    is the only thing the order actually controls, and it is what this module returns.

MODES
    legacy   np.random.default_rng(seed).permutation(n_cls)
             What exp16/54/55/56 have always used. THE DEFAULT, so importing this module
             changes no existing cache path, no existing results key, and no existing number.
    pilot    np.random.RandomState(1993 + seed).permutation(n_cls)
             PILOT's generator and PILOT's base seed. seed=0 reproduces their exact order, so
             it is directly comparable to a published number; seeds 1,2 give orders 1994/1995
             so error bars carry class-order variance the way theirs do (their table is a
             mean of 3 seeds, and PILOT's seed drives the order).
    identity np.arange(n_cls)
             PILOT with shuffle=False, i.e. iImageNetR.class_order. Kept because we have not
             verified which of shuffle=true/false GR-LoRA's config actually used; if their
             number turns out to be unshuffled, this is the mode that matches it.

ORDER TAG -- WHY THE MODE HAS TO REACH THE FILENAMES
    Task 0's class set changes with the order, so every cached LoRA feature file is specific
    to one order. exp16 already carries the scar of exactly this class of bug (see
    recipe_tag's docstring: a results key that did not carry the recipe caused 36 cells to be
    silently skipped and the OLD numbers reprinted under a NEW header). So `order_tag()` goes
    into BOTH the feature-cache filename and the results key, and `legacy` deliberately maps
    to the EMPTY string so pre-existing artefacts keep their exact names.

USAGE
    import class_order as CO
    order = CO.class_order(n_cls, seed)          # honours $ORDER, default "legacy"
    tag   = CO.order_tag()                       # "" | "_ordpilot" | "_ordidentity"
"""
import os

import numpy as np

MODES = ("legacy", "pilot", "identity")
PILOT_BASE_SEED = 1993


def mode(m=None):
    m = m if m is not None else os.environ.get("ORDER", "legacy")
    assert m in MODES, f"ORDER must be one of {MODES}, got {m!r}"
    return m


def class_order(n_cls, seed, m=None):
    """The class permutation. Task t is class_order(...)[t*cpt:(t+1)*cpt]."""
    m = mode(m)
    if m == "legacy":
        return np.random.default_rng(seed).permutation(n_cls)
    if m == "pilot":
        # RandomState(s).permutation(n) is bit-identical to
        # `np.random.seed(s); np.random.permutation(n)` -- same MT19937 stream, same
        # algorithm -- without mutating numpy's global RNG, which the training loops rely on.
        return np.random.RandomState(PILOT_BASE_SEED + seed).permutation(n_cls)
    return np.arange(n_cls)


def order_tag(m=None):
    """Suffix for cache filenames and results keys. EMPTY for legacy, by design."""
    m = mode(m)
    return "" if m == "legacy" else f"_ord{m}"


def describe(n_cls, seed, m=None):
    m = mode(m)
    o = class_order(n_cls, seed, m)
    src = {"legacy": f"default_rng({seed})",
           "pilot": f"RandomState({PILOT_BASE_SEED}+{seed})",
           "identity": "arange"}[m]
    return f"ORDER={m} [{src}] first 8 = {o[:8].tolist()}"


if __name__ == "__main__":
    n = int(os.environ.get("N_CLS", 200))
    for m in MODES:
        for s in (0, 1, 2):
            print(describe(n, s, m))
    # The check that motivated this module.
    p0 = set(class_order(n, 0, "pilot")[:20].tolist())
    for s in (0, 1, 2):
        l0 = set(class_order(n, s, "legacy")[:20].tolist())
        print(f"legacy s{s} task0 overlap with pilot s0 task0: {len(l0 & p0)}/20")
