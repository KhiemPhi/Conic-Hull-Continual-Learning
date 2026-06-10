"""Minimal incremental demo: replay buffer + geometric loss + LoRA + hull viz."""

from incremental import train_incremental_pipeline_replay


def demo():
    train_incremental_pipeline_replay(
        dataset_name="CIFAR100",
        model_name="vit_base_patch16_224.orig_in21k",
        classes_per_stage=10,
        epochs_per_stage=10,
        batch_size=48,
        learning_rate=1e-3,
        head_type="mlp",
        # --- ray budget knob ---
        # Fixed mode (adaptive_hull_rays=False): 400 rays per class always.
        # Adaptive mode: total_ray_budget / total_classes rays per class,
        #   capped at conic_hull_n_rays.  With D=768 and 100 classes that
        #   is 768//100 = 7 rays/class, keeping the combined ray matrix rank
        #   within one subspace (preventing saturation and hull overlap).
        #
        # Tuning guide:
        #   hull_ray_budget=None  → budget = D (feature dim, recommended)
        #   hull_ray_budget=384   → half-D budget (~3 rays/class for 100 cls)
        #   hull_ray_budget=2000  → looser budget (~20 rays/class for 100 cls)
        #   adaptive_hull_rays=False → revert to fixed conic_hull_n_rays
        conic_hull_n_rays=400,          # hard cap; also used when adaptive=False
        adaptive_hull_rays=False,        # scale rays/class with 1/total_classes
        hull_ray_budget=None,           # None → D (auto); set int to override
        # --- ray diversity (addresses packed-ray problem) ---
        # "hybrid" (default): SPA oversample → FPS selection.
        #   SPA finds genuine extreme directions; FPS spreads them maximally.
        #   spa_oversample=3 → collect 3× rays, keep the most angular-diverse n.
        # "spa"  : original robust SPA (may cluster on dominant axis)
        # "fps"  : pure farthest-point on inliers (max spread, no extremity guarantee)
        ray_diversity="spa",
        spa_oversample=3,
        # fps_replay_fill: use FPS to select the ~20 stored replay samples per class.
        #   True  → maximally-spread exemplars → better dynamic-hull coverage
        #   False → evenly-spaced indices (original behaviour)
        fps_replay_fill=True,
        # replay buffer
        memory_budget=0.04,
        distill_weight=20,
        use_rehearsal_cls_loss=False,
        # geometric loss (replaces CE)
        training_loss="ce",
        lambda_geo_attr=1.0,
        lambda_geo_rep=0.1,
        lambda_geo_marg=0.5,
        geo_margin_deg=35.0,
        geo_kernel="hinge_sq",
        # LoRA
        lora_rank=32,
        lora_alpha=4.0,
        lora_config="task_specific",
        blocks_freeze=12,
        # visualization
        visualize_extreme_rays=True,
        shuffle_class_order =True, 
        memory_loss_enabled=True, 
        use_analytical_head_update=True, 
        use_stage_confinement_loss=False,
        # OOD detection
        evaluate_ood_hull=True,
        ood_calibrate_percentile=50.0,
        ood_score_key="cosine",
        # Collaborative scoring (joint NNLS over all class dictionaries at once)
        # When True, adds collab_energy / collab_residual / collab_margin rows to
        # the per-stage scoring table.  Pure inference-time change — no retraining.
        # collab_energy and collab_margin outperform independent residuals when
        # class cones share extreme-ray directions (correlated geometry).
        # collaborative_lasso_lambda: L1 penalty for the joint solve.
        #   0.0  → pure NNLS (recommended starting point)
        #   1e-2 → mild sparsity; cleaner class attribution when many cones overlap
        evaluate_collaborative_scoring=True,
        collaborative_lasso_lambda=1e-2,
    )


if __name__ == "__main__":
    demo()
