
from collections import defaultdict

from addict import Dict
import numpy as np

from conic_hull import ConicHull, build_class_conic_hulls

def generate_disjoint_socs(class_hulls: Dict[str, "ConicHull"], margin: float = 0.01) -> Dict[str, dict]:
    """
    Converts a dictionary of fitted ConicHull objects into disjoint 
    Second-Order Cone (SOC) parameters.
    
    Args:
        class_hulls: Dictionary mapping class names to fitted ConicHull objects.
        margin: Angular gap (in radians) to enforce between cones.
        
    Returns:
        Dictionary containing the center and aperture for each class's SOC.
    """
    class_names = list(class_hulls.keys())
    centers = []
    hull_apertures = []

    # 1. Extract geometry from the extreme rays of each ConicHull
    for name in class_names:
        hull = class_hulls[name]
        
        if hull.extreme_rays_ is None:
            raise ValueError(f"ConicHull for class '{name}' has not been fitted.")
            
        rays = hull.extreme_rays_  # Shape: (K, D)
        
        # Center: The normalized mean of the SPA extreme rays
        mean_ray = np.mean(rays, axis=0)
        u_i = mean_ray / np.linalg.norm(mean_ray)
        centers.append(u_i)
        
        # Aperture: The maximum angle to the farthest extreme ray
        cos_thetas = np.clip(np.dot(rays, u_i), -1.0, 1.0)
        thetas = np.arccos(cos_thetas)
        hull_apertures.append(np.max(thetas))

    u = np.array(centers)
    h_alphas = np.array(hull_apertures)
    
    # 2. Compute pairwise angular distances between all class centers
    dist_matrix = np.arccos(np.clip(np.dot(u, u.T), -1.0, 1.0))
    np.fill_diagonal(dist_matrix, np.inf)
    
    # 3. Resolve Overlaps to enforce disjointness
    # Condition: alpha_i + alpha_j < distance_ij
    final_apertures = np.copy(h_alphas)
    
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            d_ij = dist_matrix[i, j]
            combined = final_apertures[i] + final_apertures[j]
            
            if combined >= d_ij:
                # Shrink both cones proportionally so they meet at the boundary minus the margin
                shrink_factor = max(0.0, (d_ij - margin) / combined)
                final_apertures[i] *= shrink_factor
                final_apertures[j] *= shrink_factor

    # 4. Package results
    soc_params = {}
    for i, name in enumerate(class_names):
        soc_params[name] = {
            "center": u[i],
            "aperture_rad": max(0.0, final_apertures[i]),
            "aperture_deg": np.degrees(max(0.0, final_apertures[i])),
            "original_hull_aperture_rad": h_alphas[i]
        }
        
    return soc_params

def generate_soft_socs(class_hulls: Dict[str, "ConicHull"]) -> Dict[str, dict]:
    """
    Directly converts ConicHulls to SOCs without forcing disjointness.
    """
    soc_params = {}
    
    for name, hull in class_hulls.items():
        rays = hull.extreme_rays_
        
        # Center: Normalized mean of extreme rays
        mean_ray = np.mean(rays, axis=0)
        u_i = mean_ray / np.linalg.norm(mean_ray)
        
        # Aperture: Max angle to the farthest extreme ray
        cos_thetas = np.clip(np.dot(rays, u_i), -1.0, 1.0)
        thetas = np.arccos(cos_thetas)
        alpha = np.max(thetas)
        
        soc_params[name] = {
            "center": u_i,
            "aperture_rad": alpha
        }
        
    return soc_params

def classify_with_soft_margin(new_embedding: np.ndarray, soc_params: Dict[str, dict], tolerance: float = 0.05) :
    """
    Classifies an embedding by finding the cone it is deepest inside of.
    """
    norm_x = np.linalg.norm(new_embedding)
    if norm_x == 0:
        return "Not Seen", 0.0
        
    u_x = new_embedding / norm_x
    
    best_class = "Not Seen"
    best_margin = -float('inf')
    best_theta = 0.0
    best_alpha = 0.0
    
    for class_name, params in soc_params.items():
        center = params['center']
        alpha = params['aperture_rad']
        
        # Calculate angle to this class's center
        theta = np.arccos(np.clip(np.dot(u_x, center), -1.0, 1.0))
        
        # Margin: How far inside the cone boundary is the point?
        margin = alpha - theta
        
        # We only consider it if it's inside the cone + tolerance
        if margin + tolerance >= 0:
            if margin > best_margin:
                best_margin = margin
                best_class = class_name
                best_theta = theta
                best_alpha = alpha
                
    if best_class == "Not Seen":
        return "Not Seen", 0.0
        
    # Convert the winning margin into a readable confidence score [0.0 to 1.0]
    # 1.0 = dead center, 0.0 = exactly on the tolerance boundary
    max_allowed_angle = best_alpha + tolerance
    confidence = max(0.0, 1.0 - (best_theta / max_allowed_angle))
    
    return best_class, round(confidence, 4)

def classify_with_confidence(new_embedding, soc_params, tolerance=0.1):
    """
    Args:
        new_embedding (np.array): Vector to classify.
        soc_params (dict): Output from the previous function.
        tolerance (float): How far outside the cone (in radians) to still
                           consider a class before returning "Not Seen".
    """
    # Normalize input
    norm_x = np.linalg.norm(new_embedding)
    if norm_x == 0:
        return "Not Seen", 0.0
    u_x = new_embedding / norm_x
    
    results = []
    
    for class_name, params in soc_params.items():
        center = params['center']
        alpha = params['aperture_rad']
        
        # Calculate angular distance theta
        cos_theta = np.clip(np.dot(u_x, center), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Calculate confidence score
        # (1.0 at center, 0.0 at edge, negative outside)
        score = 1 - (theta / alpha)
        
        # We only care about the class if it's within (alpha + tolerance)
        if theta <= (alpha + tolerance):
            results.append((class_name, score))
            
    if not results:
        return "Not Seen", 0.0
    
    # Sort by score descending and return the best match
    results.sort(key=lambda x: x[1], reverse=True)
    best_class, best_score = results[0]
    
    return best_class, round(best_score, 4)

def _print_confusion_matrix(cm, row_labels, col_labels):
    print("\n=== Confusion Matrix ===")
    col_w = max(max(len(c) for c in col_labels), 6) + 2
    row_w = max(len(r) for r in row_labels) + 2
 
    header = " " * row_w + "".join(f"{c:>{col_w}}" for c in col_labels)
    print(header)
    for i, r in enumerate(row_labels):
        row = f"{r:<{row_w}}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(col_labels)))
        print(row)
 
 
def _print_metrics(per_class, macro, accuracy):
    print("\n=== Classification Report ===")
    name_w = max(max(len(c) for c in per_class), 10) + 2
    print(f"{'class':<{name_w}}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12}")
    for cls, m in per_class.items():
        print(
            f"{cls:<{name_w}}"
            f"{m['precision']:>12.4f}"
            f"{m['recall']:>12.4f}"
            f"{m['f1']:>12.4f}"
            f"{m['support']:>12d}"
        )
    print(
        f"{'macro avg':<{name_w}}"
        f"{macro['precision']:>12.4f}"
        f"{macro['recall']:>12.4f}"
        f"{macro['f1']:>12.4f}"
    )
    print(f"\nAccuracy: {accuracy:.4f}")
 
 
def _print_scores(scores_log):
    print("\n=== Per-sample Confidence Scores ===")
    print(f"{'true':<20}{'predicted':<20}{'score':>10}")
    for true_cls, pred_cls, score in scores_log:
        print(f"{true_cls:<20}{pred_cls:<20}{score:>10.4f}")
 
    # Aggregate mean score per (true, pred) pair.
    agg = defaultdict(list)
    for t, p, s in scores_log:
        agg[(t, p)].append(s)
    print("\n=== Mean Score by (true -> predicted) ===")
    for (t, p), vals in sorted(agg.items()):
        print(f"  {t:<20} -> {p:<20}  mean={np.mean(vals):.4f}  n={len(vals)}")
 

def evaluate_socs(labeled_embeddings, soc_params, tolerance=0.1):
    """
    Evaluate the SOC classifier on a labeled set of embeddings.
 
    Args:
        labeled_embeddings (dict): {true_class_name: [np.array(embedding), ...]}
            Each key is the ground-truth class; value is a list of embeddings
            belonging to that class. (A single np.array is also accepted and
            treated as one sample.)
        soc_params (dict): Output from generate_disjoint_socs.
        tolerance (float): Passed through to classify_with_confidence.
 
    Returns:
        dict with keys:
            'confusion_matrix'  : 2D np.array (rows=true, cols=pred+NotSeen)
            'labels'            : row labels (true classes)
            'pred_labels'       : column labels (classes + 'Not Seen')
            'per_class'         : {class: {'precision','recall','f1','support'}}
            'macro'             : {'precision','recall','f1'}
            'accuracy'          : float
            'scores'            : list of (true, pred, score) tuples
    """
    # --- Normalize input shape ------------------------------------------------
    norm_data = {}
    for cls, emb in labeled_embeddings.items():
        arr = np.asarray(emb)
        if arr.ndim == 1:
            norm_data[cls] = [arr]
        else:
            norm_data[cls] = [row for row in arr]
 
    # Column labels include every SOC class plus a 'Not Seen' bucket.
    class_labels = list(soc_params.keys())
    pred_labels = class_labels + ["Not Seen"]
    # Row labels are the true classes actually present in the eval set.
    true_labels = list(norm_data.keys())
 
    label_to_row = {c: i for i, c in enumerate(true_labels)}
    label_to_col = {c: i for i, c in enumerate(pred_labels)}
 
    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    scores_log = []  # (true, pred, score)
 
    # --- Classify every sample ------------------------------------------------
    for true_cls, samples in norm_data.items():
        for x in samples:
            pred_cls, score = classify_with_soft_margin(x, soc_params, tolerance)
            cm[label_to_row[true_cls], label_to_col[pred_cls]] += 1
            scores_log.append((true_cls, pred_cls, score))
 
    # --- Per-class precision / recall / F1 -----------------------------------
    # Precision/recall are computed only over SOC classes (not 'Not Seen').
    per_class = {}
    for cls in class_labels:
        col = label_to_col[cls]
        tp = cm[label_to_row[cls], col] if cls in label_to_row else 0
 
        # Predicted as `cls` across all true rows.
        pred_total = cm[:, col].sum()
        # Actually `cls` across all predictions (including Not Seen).
        true_total = cm[label_to_row[cls]].sum() if cls in label_to_row else 0
 
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / true_total if true_total > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
 
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(true_total),
        }
 
    # Macro average over classes that have support in the eval set.
    supported = [c for c in class_labels if per_class[c]["support"] > 0]
    if supported:
        macro = {
            "precision": np.mean([per_class[c]["precision"] for c in supported]),
            "recall": np.mean([per_class[c]["recall"] for c in supported]),
            "f1": np.mean([per_class[c]["f1"] for c in supported]),
        }
    else:
        macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
 
    # Accuracy: diagonal hits over total samples (Not Seen counts as a miss
    # unless the true class literally is 'Not Seen', which it isn't here).
    total = cm.sum()
    correct = sum(
        cm[label_to_row[c], label_to_col[c]]
        for c in true_labels
        if c in label_to_col
    )
    accuracy = correct / total if total > 0 else 0.0
 
    # --- Pretty print ---------------------------------------------------------
    
    _print_metrics(per_class, macro, accuracy)
    #_print_scores(scores_log)
 
    return {
        "confusion_matrix": cm,
        "labels": true_labels,
        "pred_labels": pred_labels,
        "per_class": per_class,
        "macro": macro,
        "accuracy": accuracy,
        "scores": scores_log,
    }
 