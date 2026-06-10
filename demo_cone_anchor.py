"""
Replay-free incremental learning via frozen cones + anchor stability.

Compare against demo_incremental.py (replay + hybrid distillation) by running
both entry points with the same stage schedule.
"""

from incremental import train_incremental_pipeline_replay


def demo_cone_anchor():
    train_incremental_pipeline_replay(
        dataset_name="CIFAR100",
        model_name="vit_tiny_patch16_224",
        classes_per_stage=10,
        epochs_per_stage=10,
        batch_size=256,
        learning_rate=1e-4,
        lora_rank=24,
        lora_alpha=12.0,
        lora_config="task_shared",
        blocks_freeze=8,
        head_type="mlp",           # MLP head (cone init via fc layer)
        conic_hull_n_rays=20,
        evaluate_staged_hulls=False,
        staged_drift_align=True,
        staged_drift_align_mode="dense_rays",
        staged_drift_routing_only=True,
        staged_score_hulls="dynamic",
        staged_plane_routing="cascade",
        # ── Cone + anchor (replay-free) ──────────────────────────────────────
        use_cone_anchor=True,
        lambda_cone_stab=1.0,
        lambda_cone_marg=0.0,              # subsumed by geometric L_marg
        cone_margin_deg=35.0,
        cone_anchor_batch=128,
        cone_init_candidates=4096,
        # ── Globally geometric loss (replaces CE) ────────────────────────────
        training_loss="geometric",
        lambda_geo_attr=1.0,
        lambda_geo_rep=0.1,
        lambda_geo_marg=0.5,
        geo_margin_deg=35.0,
        geo_kernel="hinge_sq",
        geo_softplus_beta=10.0,
        geo_rep_max_pairs=50_000,
        # Disable replay-based terms (ignored automatically, but explicit here)
        distill_weight=20.0,
        use_rehearsal_cls_loss=False,
        memory_budget=0.04,
        use_analytical_head_update=True,
        use_stage_confinement_loss=True,
    )


if __name__ == "__main__":
    demo_cone_anchor()
