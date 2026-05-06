"""
demo.py
-------
End-to-end pipeline entry point:
  train → verify → extract features → build hulls → analyse separation
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional

from backbone      import load_backbone
from training      import TrainingConfig, train_to_convergence
from features      import extract_features, build_feature_dict
from conic_hull    import ConicHull, build_class_conic_hulls
from analysis      import (
    analyze_hull_separation, find_most_confused_hulls, evaluate_conic_classifier,
    analyze_superclass_confusion,
)
from verification  import verify_trained_weights



from second_order import generate_disjoint_socs, evaluate_socs, generate_soft_socs
from incremental import  train_incremental_pipeline_replay, train_incremental_pipeline_benchmark

import matplotlib.pyplot as plt
import numpy as np

def plot_incremental_comparison_grid(
    acc_pairs,
    titles=None,
    labels=("Replay (Baseline Head)", "SPA Conic Hull (Geometric)"),
    classes_per_stage=1,
):
    """
    acc_pairs: list of (acc_matrix_baseline, acc_matrix_hull) tuples, one per subplot.
    titles:    optional list of subplot titles, same length as acc_pairs.
    labels:    (baseline_label, hull_label) for the legend.
    """
    n = len(acc_pairs)
    if titles is None:
        titles = [f"Comparison {i+1}" for i in range(n)]
    assert len(titles) == n, "titles must match number of acc_pairs"

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), sharey=True, squeeze=False)
    axes = axes[0]  # flatten the single row

    for ax, (acc_baseline, acc_hull), title in zip(axes, acc_pairs, titles):
        num_stages = acc_baseline.shape[0]
        x_axis = np.arange(1, num_stages + 1) * classes_per_stage

        avg_acc_baseline = [np.mean(acc_baseline[i, :i+1]) * 100 for i in range(num_stages)]
        avg_acc_hull     = [np.mean(acc_hull[i, :i+1])     * 100 for i in range(num_stages)]

        ax.plot(x_axis, avg_acc_baseline, marker='o', linestyle='-', color='#800000',
                label=labels[0], linewidth=2)
        ax.plot(x_axis, avg_acc_hull, marker='s', linestyle='-', color='#FF0000',
                label=labels[1], linewidth=2)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Number of Classes", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        gap = avg_acc_hull[-1] - avg_acc_baseline[-1]
        ax.text(x_axis[-1], avg_acc_hull[-1] + 2, f"+{gap:+.2f}%",
                ha='center', fontweight='bold', color='red')

    axes[0].set_ylabel("Average Accuracy (%)", fontsize=12)
    plt.tight_layout()
    plt.show()

def demo(
    model_name:    str  = "vit_small_patch16_224",
    dataset_name:  str  = "CIFAR10",
    dataset_root:  str  = "./data",
    batch_size:    int  = 128,
    skip_training: bool = True,
    n_rays:        int  = 300,
    use_pca:       bool = True,
    pca_dim:       int  = 64,
    threshold:     float = 0.97,
    top_n:         int  = 5,
) -> Dict:
    """
    Full pipeline:
      1. Train (or reload) backbone
      2. Verify checkpoint weights
      3. Extract test-set features per class
      4. Build per-class conic hulls
      5. Analyse hull separation & print top confusions

    Parameters
    ----------
    skip_training : if True, load pretrained weights without fine-tuning
    n_rays        : extreme rays per class hull
    use_pca       : apply PCA inside ConicHull
    pca_dim       : PCA target dim
    threshold     : containment threshold for separation matrix
    top_n         : number of confused pairs to print

    Returns
    -------
    dict with keys: model, history, feature_dict, class_hulls, sep_df, confusions
    """
   
    (backbone, head, acc_matrix, acc_matrix_conic_hull_static_procrustes, acc_matrix_conic_hull_dynamic,
     drift_matrix, final_forgetting, final_forgetting_static, final_forgetting_dynamic,
     _sc_history) = train_incremental_pipeline_replay(
        dataset_name="CIFAR10",
        model_name="vit_tiny_patch16_224",
        classes_per_stage=1,
        epochs_per_stage=20,
        alpha=0.1,
        distill_weight=10.0, # distillation loss must be used
        min_delta=0.01,
        patience=3,
        batch_size=256,
        reserved_space=False,
        conic_hull_margin=0.7,
        conic_hull_n_rays=200,
        learning_rate=1e-4,
        pointwise_distill_weight=0.,
        memory_budget=0.004,
        use_analytical_head_update=True, # most key update 
        head_type="mlp", # need to test heads 
        drift_method="covariance",
        track_superclass_confusion=False,
        superclass_confusion_top_n=15,
        superclass_confusion_plot=False,
        head_update_method="covariance", # method does not matter
        head_update_magnitude_preserving=True, 
        use_conic_hull_rotation=True, 
        rotate_dynamic_hulls=False,
        rotate_static_hulls=False, 
        lora_alpha=8.0, 
        lora_rank=16
    )


 

if __name__ == "__main__":
    demo()
