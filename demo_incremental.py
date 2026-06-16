"""Minimal incremental demo: replay buffer + geometric loss + LoRA + hull viz."""

from incremental import train_incremental_pipeline_replay


def demo():
    train_incremental_pipeline_replay(
        dataset_name="CIFAR100",
        model_name="vit_base_patch16_224.orig_in21k",
        classes_per_stage=10,
        epochs_per_stage=10,
        batch_size=256,
        learning_rate=1e-4,
        # Gaussian-replay head: a normalized-linear classifier retrained over ALL
        # seen classes each task on samples ~ N(mu_c, Sigma_c) (SLCA/MACIL-style).
        # This debiases the head (the part that forgets fastest); the conic hull is
        # left as the feature-preserving boundary.
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
        conic_hull_n_rays=400,  # hard cap; also used when adaptive=False
        adaptive_hull_rays=False,  # scale rays/class with 1/total_classes
        hull_ray_budget=None,  # None → D (auto); set int to override
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
        blocks_freeze=10,
        # visualization
        visualize_extreme_rays=True,
        shuffle_class_order=True,
        memory_loss_enabled=True,
        use_analytical_head_update=True,
        # Learned inter-stage drift projection: each stage fit g_t: phi_t->phi_(t-1)
        # (a residual MLP) on replay pairs, and at eval back-project each old class's
        # query into its hull's BIRTH space before scoring. Hulls stay frozen in
        # birth space, so rotate_static_hulls MUST be False (no double-correction).
        rotate_static_hulls=False,
        use_learned_drift=True,
        drift_epochs=200,
        drift_hidden=2048,
        drift_lr=1e-3,
        # Transported skeleton hulls (RanPAC-inspired): score old classes against
        # their RICH frozen birth hull (full-data, up to conic_hull_n_rays rays)
        # FORWARD-transported into current space via a drift map fit on the replay
        # buffer — instead of re-fitting from the ~20 buffer exemplars (the
        # dynamic-hull ceiling). New classes use full-data hulls. Adds a "Transp."
        # column next to Dynamic.
        evaluate_transported_hulls=True,
        # "procrustes": rigid rotation+translation (shape-preserving; best default)
        # "ridge_affine": allows shear — try if drift is non-rigid
        transport_pair_method="procrustes",
        transport_ridge=1e-3,  # ridge for ridge_affine map fitting
        transport_pca_subspace=True,  # fit the map in a low-rank subspace (robust for ~20 pts)
        # PCA dims for the map ≈ drift/LoRA rank. Keep WELL BELOW n_pairs (≈200):
        # None→n_pairs-1 interpolates the noisy pairs and overfits (in-sample
        # residual looks ~0 but generalises badly). 32 = lora_rank. Watch the
        # printed oos-resid (out-of-sample) — the honest map-quality number.
        transport_pca_components=32,
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
        # --- speed ---
        # Mixed-precision: run the ViT backbone fwd/bwd in bf16 (~1.5-2x faster
        # per epoch on A100) while losses/hulls stay fp32. bf16 needs no
        # GradScaler and matches fp32 accuracy. Set use_amp=False to disable.
        use_amp=True,
        amp_dtype="bf16",
        project_hulls_to_stage_cap=True,
    )


if __name__ == "__main__":
    demo()
