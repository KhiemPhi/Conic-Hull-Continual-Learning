"""
conic_hull_pipeline
===================
Public API — import everything you need from here.

    from conic_hull_pipeline import (
        load_backbone, train_to_convergence, TrainingConfig,
        extract_features, build_test_feature_dict,
        ConicHull, build_class_conic_hulls,
        analyze_hull_separation, find_most_confused_hulls,
        evaluate_conic_classifier,
        verify_trained_weights, WeightVerificationReport,
        demo,
    )
"""

from .backbone   import load_backbone, list_popular_backbones
from .training   import (
    TrainingConfig, train_to_convergence,
    train_class_incremental,
)
from .features   import get_default_transform, extract_features, build_feature_dict
from .conic_hull import ConicHull, build_class_conic_hulls
from .analysis   import (
    analyze_hull_separation,
    find_most_confused_hulls,
    evaluate_conic_classifier,
    analyze_superclass_confusion,
    CIFAR100_SUPERCLASSES,
    CLASS_TO_SUPERCLASS,
    CIFAR100_CLASS_NAMES,
)
from .verification import WeightVerificationReport, verify_trained_weights
from .incremental  import (
    ClassStatisticsStore,
    estimate_affine_drift,
    update_head_weights_analytically,
    log_barrier_margin_loss,
    logdet_volume_penalty,
    IncrementalMLPHead,
)
from .demo         import demo


__all__ = [
    # backbone
    "load_backbone", "list_popular_backbones",
    # training
    "TrainingConfig", "train_to_convergence",
    "IncrementalConfig", "ClassIncrementalModel", "train_class_incremental",
    # features
    "get_default_transform", "extract_features", "build_feature_dict",
    # hull
    "ConicHull", "build_class_conic_hulls",
    # analysis
    "analyze_hull_separation", "find_most_confused_hulls", "evaluate_conic_classifier",
    "analyze_superclass_confusion", "CIFAR100_SUPERCLASSES", "CLASS_TO_SUPERCLASS", "CIFAR100_CLASS_NAMES",
    # verification
    "WeightVerificationReport", "verify_trained_weights",
    # analytical drift correction
    "ClassStatisticsStore", "estimate_affine_drift", "update_head_weights_analytically",
    "log_barrier_margin_loss", "logdet_volume_penalty", "IncrementalMLPHead",
    # entry point
    "demo",
    
]