#!/usr/bin/env python3
"""splits.py -- the canonical train/test split constructions, in ONE place.

WHY THIS FILE EXISTS
    The 80/20 split logic was duplicated verbatim in exp16_full_table.get_data,
    fsa_train.get_data and exp19_dataset_hull.get_labels. Those three must agree EXACTLY:
    get_data decides which IMAGES are featurised and in what order, get_labels decides
    which LABELS the read-out lines up against them. If they drift, every label is silently
    attached to the wrong feature row and nothing crashes -- accuracy just quietly collapses
    toward chance and looks like a method result. Three copies of a split rule is that bug
    waiting to happen, so any NEW split lives here and is imported.

WHAT WENT WRONG WITH THE GLOBAL SPLIT, measured 2026-08-08
    `global_indices` permutes all N images and cuts at 0.8N. On balanced datasets that is
    fine. ImageNet-A is not balanced -- 7500 images over 200 classes, per-class total
    min 3 / median 26 / max 100, with 20 classes under 10 images. A global cut on that gives:

        our global random 80/20:  train/cls min 1,  4 CLASSES WITH ZERO TEST IMAGES
        stratified per-class:     train/cls min 2,  0 classes with zero test

    Four classes that can never be scored but still hold a classifier column: they can only
    ever produce false positives, and they are pure loss. They are also the source of the
    "N seen classes have NO rays" warnings and the -inf score columns that exp35/52/53 all
    carry special-case handling for.

    `stratified_indices` splits WITHIN each class and guarantees >=1 train and >=1 test for
    every class that has at least 2 images. On ImageNet-A it lands at 6000/1500 against the
    published 5981/1519 -- the residual difference is rounding, not protocol.

DO NOT RETROFIT THE GLOBAL PATH
    `global_indices` is kept, exactly as it was, because every cached exp16 feature file and
    every results key already in the repo's JSONs was produced under it. Changing it in place
    would silently redefine numbers that are already recorded. New protocol == new dataset
    name (IMAGENETAP, CUB200P), never a redefinition of an old one.

USAGE
    from splits import SPLIT_SEED, stratified_indices, audit_split
    tri, tei = stratified_indices(lab)
    audit_split(lab, tri, tei, "IMAGENETAP")     # raises if an invariant is violated
"""
import numpy as np

SPLIT_SEED = 1993


def global_indices(n_items, frac=0.8, seed=SPLIT_SEED):
    """The ORIGINAL convention: permute everything, cut at frac. Reproduces the inline code
    in exp16/fsa_train/exp19 bit-for-bit -- do not change it, cached features depend on it."""
    p = np.random.default_rng(seed).permutation(n_items)
    k = int(frac * n_items)
    return p[:k], p[k:]


def stratified_indices(lab, frac=0.8, seed=SPLIT_SEED):
    """Per-class frac split. Guarantees >=1 train and >=1 test for any class with >=2 items.

    Each class draws from its OWN rng seeded (seed, class), not from one sequential stream,
    so adding or removing a class cannot shift any other class's split. That matters because
    a split that silently reshuffles when the class set changes is unreproducible across
    dataset versions.

    A class with exactly 1 item goes entirely to train: it cannot be both fitted and scored,
    and putting it in test would create a class the classifier has never seen. Returns
    globally shuffled index arrays so downstream code cannot depend on class-major order."""
    lab = np.asarray(lab)
    tri, tei = [], []
    for c in np.unique(lab):
        ix = np.where(lab == c)[0]
        ix = ix[np.random.default_rng([seed, int(c)]).permutation(len(ix))]
        if len(ix) == 1:
            tri.append(ix)
            continue
        k = int(round(frac * len(ix)))
        k = min(max(k, 1), len(ix) - 1)          # >=1 train AND >=1 test
        tri.append(ix[:k]); tei.append(ix[k:])
    tri = np.concatenate(tri) if tri else np.zeros(0, int)
    tei = np.concatenate(tei) if tei else np.zeros(0, int)
    rng = np.random.default_rng(seed)
    return tri[rng.permutation(len(tri))], tei[rng.permutation(len(tei))]


# Datasets whose class balance makes a GLOBAL cut unsafe. Membership here, and nowhere
# else, decides which rule a dataset gets -- a per-loader `if name.endswith(...)` would
# drift the moment one of the three loaders is edited and the others are not.
#   IMAGENETAP  per-class totals 3..100; a global cut orphans 4 classes with no test images.
#   CUB200P     deliberately NOT stratified: it exists to reproduce the published 9430/2358,
#               which is a global 80/20 of all 11,788. CUB is balanced (~59/class) so a
#               global cut cannot orphan anything, and stratifying would move the counts off
#               the published protocol, defeating the point of the dataset.
STRATIFIED = {"IMAGENETAP"}


def split_indices(name, lab, frac=0.8, seed=SPLIT_SEED):
    """THE dispatch. Every loader calls exactly this, so get_data and get_labels cannot
    disagree about which images are train -- the silent failure this module exists to stop."""
    if name in STRATIFIED:
        tri, tei = stratified_indices(lab, frac, seed)
    else:
        tri, tei = global_indices(len(lab), frac, seed)
    audit_split(lab, tri, tei, name, strict=name in STRATIFIED)
    return tri, tei


def audit_split(lab, tri, tei, name="", n_cls=None, strict=True):
    """Assert the invariants a CIL split must satisfy, and return the stats either way.

    These are exactly the failures the global split produced on ImageNet-A. Checking them at
    construction time is the whole point of this module -- a class with no test images is
    invisible in an accuracy number and shows up only as an unexplained deficit."""
    lab = np.asarray(lab)
    n = int(n_cls or lab.max() + 1)
    ctr = np.bincount(lab[tri], minlength=n)
    cte = np.bincount(lab[tei], minlength=n)
    st = {"name": name, "n_train": len(tri), "n_test": len(tei), "n_cls": n,
          "train_min": int(ctr.min()), "train_med": float(np.median(ctr)),
          "train_max": int(ctr.max()), "test_min": int(cte.min()),
          "no_train": int((ctr == 0).sum()), "no_test": int((cte == 0).sum()),
          "train_lt2": int((ctr < 2).sum()), "overlap": len(np.intersect1d(tri, tei))}
    if strict:
        assert st["overlap"] == 0, f"{name}: {st['overlap']} indices in BOTH train and test"
        assert st["no_train"] == 0, f"{name}: {st['no_train']} classes have no train images"
        assert st["no_test"] == 0, (
            f"{name}: {st['no_test']} classes have NO TEST IMAGES. They hold a classifier "
            f"column that can only produce false positives and can never be scored. Use "
            f"stratified_indices, not global_indices, on an imbalanced dataset.")
    return st


if __name__ == "__main__":
    import os
    from datasets import load_dataset
    REPO = os.path.dirname(os.path.abspath(__file__))
    d = load_dataset("barkermrl/imagenet-a", cache_dir=os.path.join(REPO, "data/hf"))["train"]
    lab = np.array(d["label"])
    for nm, fn in (("global", global_indices(len(lab))),
                   ("stratified", stratified_indices(lab))):
        s = audit_split(lab, *fn, name=nm, strict=False)
        print(f"{nm:<12} train {s['n_train']:5d}  test {s['n_test']:5d}  "
              f"train/cls min {s['train_min']:3d} med {s['train_med']:5.0f}  "
              f"no_test {s['no_test']:3d}  train<2 {s['train_lt2']:3d}")
