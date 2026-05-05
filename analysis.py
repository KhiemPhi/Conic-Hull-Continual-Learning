"""
analysis.py
-----------
Post-hoc analysis tools for fitted conic hulls:
  - analyze_hull_separation  : pairwise containment / separation matrix
  - find_most_confused_hulls : rank the worst cross-class overlaps
  - evaluate_conic_classifier: classify test vectors by hull score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn.functional as F

from conic_hull import ConicHull


# ── CIFAR-100 superclass taxonomy ─────────────────────────────────────────────
# 20 superclasses × 5 classes each, indexed 0-99 in the standard label order.
CIFAR100_SUPERCLASSES: List[Tuple[str, List[int]]] = [
    ("aquatic mammals",       [4, 30, 55, 72, 95]),    # beaver dolphin otter seal whale
    ("fish",                  [1, 32, 67, 73, 91]),    # aquarium_fish flatfish ray shark trout
    ("flowers",               [54, 62, 70, 82, 92]),   # orchids poppies roses sunflowers tulips
    ("food containers",       [9, 10, 16, 28, 61]),    # bottles bowls cans cups plates
    ("fruit and vegetables",  [0, 51, 53, 57, 83]),    # apples mushrooms oranges pears sweet_peppers
    ("household electrical",  [22, 39, 40, 86, 87]),   # clock keyboard lamp telephone television
    ("household furniture",   [5, 20, 25, 84, 94]),    # bed chair couch table wardrobe
    ("insects",               [6, 7, 14, 18, 24]),     # bee beetle butterfly caterpillar cockroach
    ("large carnivores",      [3, 42, 43, 88, 97]),    # bear leopard lion tiger wolf
    ("large man-made outdoor",[12, 17, 37, 68, 76]),   # bridge castle house road skyscraper
    ("large natural outdoor", [23, 33, 49, 60, 71]),   # cloud forest mountain plain sea
    ("large omnivores/herbivores", [15, 19, 21, 31, 38]), # camel cattle chimpanzee elephant kangaroo
    ("medium-sized mammals",  [34, 63, 64, 66, 75]),   # fox porcupine possum raccoon skunk
    ("non-insect invertebrates", [26, 45, 77, 79, 99]),# crab lobster snail spider worm
    ("people",                [2, 11, 35, 46, 98]),    # baby boy girl man woman
    ("reptiles",              [27, 29, 44, 78, 93]),   # crocodile dinosaur lizard snake turtle
    ("small mammals",         [36, 50, 65, 74, 80]),   # hamster mouse rabbit shrew squirrel
    ("trees",                 [47, 52, 56, 59, 96]),   # maple oak palm pine willow
    ("vehicles 1",            [8, 13, 48, 58, 90]),    # bicycle bus motorcycle pickup_truck train
    ("vehicles 2",            [41, 69, 81, 85, 89]),   # lawn_mower rocket streetcar tank tractor
]

# Flat lookup: class_id → superclass_index
CLASS_TO_SUPERCLASS: Dict[int, int] = {
    cls_id: sc_idx
    for sc_idx, (_, cls_ids) in enumerate(CIFAR100_SUPERCLASSES)
    for cls_id in cls_ids
}

CIFAR100_CLASS_NAMES: List[str] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "chair", "chimpanzee", "clock", "cloud",
    "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin",
    "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
    "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
    "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
    "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree",
    "wolf", "woman", "worm",
]


def analyze_hull_separation(
    class_hulls:  Dict[str, ConicHull],
    feature_dict: Dict[str, np.ndarray],
    threshold:    float = 0.99,
) -> pd.DataFrame:
    """
    Build a pairwise separation matrix.

    matrix[i][j] = fraction of class-i vectors whose conic-angular-similarity
    score against hull-j is ≥ `threshold`  (i.e. "fits inside hull j").

    A well-separated model has high values on the diagonal and low values
    everywhere else.

    Parameters
    ----------
    class_hulls  : fitted hulls from build_class_conic_hulls
    feature_dict : raw feature vectors per class
    threshold    : cosine-similarity threshold to count as "inside"

    Returns
    -------
    pd.DataFrame  shape (n_classes, n_classes)
    """
    classes = list(class_hulls.keys())
    n = len(classes)
    matrix = np.zeros((n, n))

    print(f"Analyzing hull separation  (threshold={threshold}) …")

    for i, target_class in enumerate(tqdm(classes, desc="Analyzing separation")):
        vectors = feature_dict[target_class]
        for j, hull_class in enumerate(classes):
            scores          = class_hulls[hull_class].score(vectors)
            matrix[i, j]   = float(np.mean(scores >= threshold))

    df = pd.DataFrame(matrix, index=classes, columns=classes)
    diagonal_values = np.diag(df.values)
    self_containment = pd.Series(diagonal_values, index=df.index)
    self_containment.to_csv("self_containment.csv")  # save for later analysis

    print("Self-Containment Scores (Diagonal):")
    print(self_containment)

    plt.figure(figsize=(max(8, n), max(6, n - 2)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, square=True)
    plt.title(f"Conic Hull Separation Matrix  (threshold={threshold})")
    plt.xlabel("Hull Class")
    plt.ylabel("Vector Class")
    plt.tight_layout()
    plt.show()

    return df


def find_most_confused_hulls(
    df:    pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank class pairs by cross-containment score (off-diagonal entries of the
    separation matrix), highest first.

    Parameters
    ----------
    df    : output of analyze_hull_separation
    top_n : how many pairs to return

    Returns
    -------
    pd.DataFrame  columns=['Vector_Class', 'Hull_Class', 'Containment_Score']
    """
    confusions = df.stack().reset_index()
    confusions.columns = ["Vector_Class", "Hull_Class", "Containment_Score"]
    confusions = (confusions[confusions["Vector_Class"] != confusions["Hull_Class"]]
                  .sort_values("Containment_Score", ascending=False))

    print(f"\nTop {top_n} Geometric Confusions (Containment):")
    print("-" * 60)
    for _, row in confusions.head(top_n).iterrows():
        print(f"  {row['Vector_Class']:<15} fits in  "
              f"{row['Hull_Class']:<15} hull: "
              f"{row['Containment_Score']:.2%}")

    return confusions.head(top_n)


def evaluate_conic_classifier(
    class_hulls:       Dict[str, ConicHull],
    test_feature_dict: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Nearest-hull classifier: assigns each vector to the class whose hull
    gives the highest conic angular similarity score.

    Returns
    -------
    dict  { 'accuracy': float, 'f1': float }
    """
    classes   = list(class_hulls.keys())
    all_preds = []
    all_true  = []

    print(f"Evaluating Conic Classifier on {len(classes)} classes …")

    for true_label, vectors in tqdm(test_feature_dict.items(), desc="Predicting"):
        if true_label not in class_hulls:
            continue

        # (N_vectors, N_classes) score matrix
        scores = np.column_stack([
            class_hulls[cls].score(vectors) for cls in classes
        ])
        preds = [classes[int(i)] for i in np.argmax(scores, axis=1)]

        all_preds.extend(preds)
        all_true.extend([true_label] * len(vectors))

    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="weighted")

    print("\n" + "=" * 40)
    print("CONIC HULL CLASSIFICATION RESULTS")
    print("-" * 40)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}  (weighted)")
    print("=" * 40)

    return {"accuracy": acc, "f1": f1}


# ── Superclass confusion analysis ─────────────────────────────────────────────

def _collect_head_predictions(
    backbone, head, test_loader, device
) -> Tuple[List[int], List[int]]:
    """Return (all_true, all_pred) using the classifier head."""
    from incremental import FixedConicHead, IncrementalConicHead

    backbone.eval()
    head.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  Head predictions", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            features = backbone(imgs)
            if isinstance(head, (FixedConicHead, IncrementalConicHead)):
                logits = head(features, labels=None)
            else:
                logits = head(features)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return all_true, all_pred


def _collect_hull_predictions(
    backbone, hulls: Dict, test_loader, device, seen_class_ids: List[int]
) -> Tuple[List[int], List[int]]:
    """Return (all_true, all_pred) using argmax-of-hull-scores classification."""
    backbone.eval()
    # Only score against hulls that actually exist for seen classes
    hull_keys  = [str(c) for c in seen_class_ids if str(c) in hulls]
    class_ids  = [int(k) for k in hull_keys]
    all_true, all_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  Hull predictions", leave=False):
            feats  = backbone(imgs.to(device)).cpu().numpy()            # (N, D)
            scores = np.stack(
                [hulls[k].score(feats) for k in hull_keys], axis=1
            )                                                            # (N, C)
            best   = np.argmax(scores, axis=1)
            all_true.extend(labels.tolist())
            all_pred.extend([class_ids[i] for i in best])

    return all_true, all_pred


def _run_confusion_analysis(
    all_true:       List[int],
    all_pred:       List[int],
    seen_class_ids: List[int],
    class_names:    List[str],
    top_n_pairs:    int,
    label:          str  = "",
    plot:           bool = True,
) -> Dict:
    """Core analysis given raw prediction arrays. Prints report and returns dict."""
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    seen_set = set(seen_class_ids)
    mask     = np.isin(all_true, list(seen_set))
    all_true = all_true[mask]
    all_pred = all_pred[mask]

    n_total   = len(all_true)
    n_correct = int((all_true == all_pred).sum())
    accuracy  = n_correct / n_total if n_total > 0 else 0.0

    wrong_mask = all_true != all_pred
    true_wrong = all_true[wrong_mask]
    pred_wrong = all_pred[wrong_mask]
    n_wrong    = int(wrong_mask.sum())

    within_mask = np.array([
        CLASS_TO_SUPERCLASS.get(t, -1) == CLASS_TO_SUPERCLASS.get(p, -2)
        for t, p in zip(true_wrong, pred_wrong)
    ])
    n_within   = int(within_mask.sum())
    n_cross    = n_wrong - n_within
    pct_within = 100.0 * n_within / n_wrong if n_wrong > 0 else 0.0
    pct_cross  = 100.0 * n_cross  / n_wrong if n_wrong > 0 else 0.0

    pair_counts: Dict[Tuple[int, int], int] = {}
    for t, p in zip(true_wrong.tolist(), pred_wrong.tolist()):
        pair_counts[(t, p)] = pair_counts.get((t, p), 0) + 1
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])

    sc_correct: Dict[int, int] = {}
    sc_total:   Dict[int, int] = {}
    for t, p in zip(all_true.tolist(), all_pred.tolist()):
        sc = CLASS_TO_SUPERCLASS.get(t, -1)
        sc_total[sc]   = sc_total.get(sc, 0) + 1
        sc_correct[sc] = sc_correct.get(sc, 0) + (1 if t == p else 0)
    sc_acc = {sc: sc_correct.get(sc, 0) / sc_total[sc] for sc in sc_total}

    W     = 70
    title = f"SUPERCLASS CONFUSION — {label}" if label else "SUPERCLASS CONFUSION ANALYSIS"
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)
    print(f"  Samples evaluated : {n_total:,}")
    print(f"  Overall accuracy  : {accuracy:.2%}  ({n_correct:,} / {n_total:,} correct)")
    print(f"  Total errors      : {n_wrong:,}")
    print(f"  Within-superclass : {n_within:,}  ({pct_within:.1f}% of errors)")
    print(f"  Cross-superclass  : {n_cross:,}  ({pct_cross:.1f}% of errors)")

    print(f"\n  Top {top_n_pairs} confused pairs  (true → predicted):")
    print(f"  {'True class':<22} {'Predicted class':<22} {'Count':>6}  {'Type'}")
    print("  " + "-" * (W - 2))
    for (t, p), cnt in sorted_pairs[:top_n_pairs]:
        t_name = class_names[t] if t < len(class_names) else str(t)
        p_name = class_names[p] if p < len(class_names) else str(p)
        kind   = "within" if CLASS_TO_SUPERCLASS.get(t, -1) == CLASS_TO_SUPERCLASS.get(p, -2) else "cross"
        print(f"  {t_name:<22} {p_name:<22} {cnt:>6}  {kind}  ({100.*cnt/n_wrong:.1f}%)")

    print(f"\n  Per-superclass accuracy:")
    print(f"  {'Superclass':<30} {'Acc':>7}  {'Correct':>8}  {'Total':>7}")
    print("  " + "-" * (W - 2))
    for sc_idx, (sc_name, _) in enumerate(CIFAR100_SUPERCLASSES):
        if sc_idx not in sc_total:
            continue
        print(f"  {sc_name:<30} {sc_acc[sc_idx]:>7.2%}  "
              f"{sc_correct.get(sc_idx, 0):>8,}  {sc_total[sc_idx]:>7,}")
    print("=" * W)

    if plot and n_wrong > 0:
        _plot_superclass_confusion(all_true, all_pred, seen_class_ids, class_names, sc_acc)

    return {
        "accuracy":           accuracy,
        "n_total":            n_total,
        "n_correct":          n_correct,
        "n_wrong":            n_wrong,
        "n_within":           n_within,
        "n_cross":            n_cross,
        "pct_within":         pct_within,
        "pct_cross":          pct_cross,
        "top_confused_pairs": sorted_pairs[:top_n_pairs],
        "sc_accuracy":        sc_acc,
        "all_true":           all_true,
        "all_pred":           all_pred,
    }


def analyze_superclass_confusion(
    backbone:       "torch.nn.Module",
    head:           "torch.nn.Module",
    test_loader,
    device,
    seen_class_ids: Optional[List[int]] = None,
    class_names:    Optional[List[str]] = None,
    top_n_pairs:    int  = 20,
    plot:           bool = True,
    hulls:          Optional[Dict] = None,
) -> Dict:
    """
    Evaluate the classifier and break confusion into within- vs cross-superclass.

    Runs the analysis for the linear/conic head classifier.  When ``hulls`` is
    provided (a dict of fitted ConicHull objects keyed by class-id strings), the
    same analysis is also run for the argmax-of-hull-scores classifier so you get
    a direct side-by-side comparison in the printout.

    Parameters
    ----------
    backbone        : feature extractor (set to eval mode inside)
    head            : classifier head (FixedConicHead / IncrementalConicHead /
                      IncrementalLinearHead / IncrementalMLPHead)
    test_loader     : DataLoader yielding (images, labels)
    device          : torch device
    seen_class_ids  : class IDs the model has been trained on.  If None,
                      inferred from the test-loader ground-truth labels.
    class_names     : 100-entry name list indexed by class id.
                      Defaults to CIFAR100_CLASS_NAMES.
    top_n_pairs     : how many confused pairs to print per classifier
    plot            : render confusion heatmaps
    hulls           : optional Dict[str, ConicHull] — when supplied, a second
                      analysis pass is run using hull-score argmax as the
                      classifier.  Use ``hull_mgr.static_hulls`` or
                      ``hull_mgr.get_dynamic_hulls(feature_dict)``.

    Returns
    -------
    Dict with top-level keys from the head analysis (backwards compatible) plus
    a ``"hull"`` sub-dict when hulls are provided.
    """
    if class_names is None:
        class_names = CIFAR100_CLASS_NAMES

    # ── Head-based predictions ────────────────────────────────────────────────
    head_true, head_pred = _collect_head_predictions(backbone, head, test_loader, device)

    if seen_class_ids is None:
        seen_class_ids = sorted(set(head_true))

    head_result = _run_confusion_analysis(
        head_true, head_pred, seen_class_ids, class_names,
        top_n_pairs, label="Head Classifier", plot=plot,
    )

    result = dict(head_result)   # top-level = head results (backwards compat)

    # ── Hull-based predictions (optional) ─────────────────────────────────────
    if hulls is not None:
        hull_true, hull_pred = _collect_hull_predictions(
            backbone, hulls, test_loader, device, seen_class_ids
        )
        hull_result = _run_confusion_analysis(
            hull_true, hull_pred, seen_class_ids, class_names,
            top_n_pairs, label="Conic Hull Classifier", plot=plot,
        )
        result["hull"] = hull_result

        # Convenience delta printout
        delta = hull_result["accuracy"] - head_result["accuracy"]
        sign  = "+" if delta >= 0 else ""
        print(f"\n  Hull vs Head accuracy delta: {sign}{delta:.2%}")
        w_delta = hull_result["pct_within"] - head_result["pct_within"]
        print(f"  Within-superclass error shift: {'+' if w_delta>=0 else ''}{w_delta:.1f}pp")

    return result


def _plot_superclass_confusion(
    all_true: np.ndarray,
    all_pred: np.ndarray,
    seen_class_ids: List[int],
    class_names: List[str],
    sc_acc: Dict[int, float],
) -> None:
    """Two plots: superclass-level confusion matrix and per-superclass accuracy bar."""
    seen_sc = sorted({CLASS_TO_SUPERCLASS[c] for c in seen_class_ids if c in CLASS_TO_SUPERCLASS})
    sc_labels = [CIFAR100_SUPERCLASSES[i][0] for i in seen_sc]
    n_sc = len(seen_sc)

    # Map raw predictions to superclass indices
    true_sc  = np.array([CLASS_TO_SUPERCLASS.get(c, -1) for c in all_true])
    pred_sc  = np.array([CLASS_TO_SUPERCLASS.get(c, -1) for c in all_pred])

    # Filter to seen superclasses
    mask = np.isin(true_sc, seen_sc) & np.isin(pred_sc, seen_sc)
    true_sc_f = true_sc[mask]
    pred_sc_f = pred_sc[mask]

    sc_idx_map = {sc: i for i, sc in enumerate(seen_sc)}
    true_sc_m  = np.array([sc_idx_map[s] for s in true_sc_f])
    pred_sc_m  = np.array([sc_idx_map[s] for s in pred_sc_f])

    cm = confusion_matrix(true_sc_m, pred_sc_m, labels=list(range(n_sc)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: normalised superclass confusion matrix
    ax = axes[0]
    im = ax.imshow(cm_norm, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(n_sc))
    ax.set_yticks(range(n_sc))
    ax.set_xticklabels(sc_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sc_labels, fontsize=8)
    ax.set_xlabel("Predicted superclass")
    ax.set_ylabel("True superclass")
    ax.set_title("Superclass Confusion Matrix (row-normalised)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate diagonal in green, off-diagonal when > 5%
    for i in range(n_sc):
        for j in range(n_sc):
            val = cm_norm[i, j]
            if val > 0.05 or i == j:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    # Plot 2: per-superclass accuracy bar chart
    ax2 = axes[1]
    accs  = [sc_acc.get(sc, 0.0) for sc in seen_sc]
    colors = ["#2ecc71" if a >= 0.5 else "#e74c3c" for a in accs]
    bars = ax2.barh(range(n_sc), [a * 100 for a in accs], color=colors)
    ax2.set_yticks(range(n_sc))
    ax2.set_yticklabels(sc_labels, fontsize=8)
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_xlim(0, 100)
    ax2.set_title("Per-Superclass Accuracy")
    ax2.axvline(x=50, color="gray", linestyle="--", linewidth=0.8)
    for bar, a in zip(bars, accs):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{a:.1%}", va="center", fontsize=7)

    plt.tight_layout()
    plt.show()
