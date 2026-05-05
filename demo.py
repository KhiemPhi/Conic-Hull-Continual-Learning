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
from analysis      import analyze_hull_separation, find_most_confused_hulls, evaluate_conic_classifier
from verification  import verify_trained_weights



from second_order import generate_disjoint_socs, evaluate_socs, generate_soft_socs
from incremental import FixedConicHead, get_incremental_dataloaders
def demo(
    model_name:    str  = "vit_small_patch16_224",
    dataset_name:  str  = "CIFAR100",
    dataset_root:  str  = "./data",
    batch_size:    int  = 128,
    skip_training: bool = False,
    n_rays:        int  = 20,
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
    device            = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path   = f"best_{model_name}_{dataset_name.lower()}.pt"

    print("=" * 60)
    print(f"  backbone  : {model_name}")
    print(f"  dataset   : {dataset_name}")
    print(f"  device    : {device}")
    print("=" * 60)

    # ── 1. Train or reload ────────────────────────────────────────────────
    if skip_training:
        print("\n[demo] Skipping training — loading pretrained weights only.")
        model   = load_backbone(model_name, pretrained=True,
                                num_classes=0, device=device)
        history = {}

    elif os.path.exists(checkpoint_path):
        print(f"\n[demo] Found checkpoint '{checkpoint_path}' — loading.")
        model = load_backbone(model_name, pretrained=False,
                              num_classes=0, device=device)
        raw = torch.load(checkpoint_path, map_location=device)
        clean = {
            (k.replace("backbone.", "") if k.startswith("backbone.") else k): v
            for k, v in raw.items()
        }
        msg = model.load_state_dict(clean, strict=False)
        print(f"  missing keys     : {msg.missing_keys}")
        print(f"  unexpected keys  : {msg.unexpected_keys}")
        history = {"loaded_checkpoint": checkpoint_path}

    else:
        cfg = TrainingConfig(
            max_epochs      = 30,
            lr              = 1e-3,
            weight_decay    = 0.05,
            warmup_epochs   = 5,
            patience        = 8,
            freeze_backbone = True,
            unfreeze_after  = 3,
            checkpoint_path = checkpoint_path,
            mixed_precision = True,
        )
        model, history = train_to_convergence(
            model_name   = model_name,
            dataset_name = dataset_name,
            dataset_root = dataset_root,
            batch_size   = batch_size,
            cfg          = cfg,
            device       = device,
        )

    # ── 2. Verify checkpoint ──────────────────────────────────────────────
    if os.path.exists(checkpoint_path):
        verify_trained_weights(
            model_name      = model_name,
            checkpoint_path = checkpoint_path,
            dataset_name    = dataset_name,
            dataset_root    = dataset_root,
        )

    # ── 3. Single-image sanity check ──────────────────────────────────────
    dummy = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    feat = extract_features(model, dummy, device=device)
    print(f"\n[demo] Single-image feature shape: {feat.shape}")

    # ── 4. Extract full test-set features ─────────────────────────────────
    feature_dict_train = build_feature_dict(
        model, dataset_root=dataset_root, dataset_name=dataset_name,
        batch_size=batch_size, device=device, train=True,  # Use train set for demo purposes
    )
    feature_dict_test = build_feature_dict(
        model, dataset_root=dataset_root, dataset_name=dataset_name,
        batch_size=batch_size, device=device, train=False,
    )

    # # ── 5. Build per-class conic hulls ────────────────────────────────────
    class_hulls_train = build_class_conic_hulls(
        feature_dict_train, n_rays=n_rays, use_pca=False
    )

    # # ── 6. Separation analysis ────────────────────────────────────────────
    sep_df     = analyze_hull_separation(class_hulls_train, feature_dict_test,
                                         threshold=threshold)
    confusions = find_most_confused_hulls(sep_df, top_n=top_n)
    
    evaluate_conic_classifier(class_hulls_train, feature_dict_test)


    soc_params = generate_soft_socs(class_hulls_train)
    eval_results = evaluate_socs(feature_dict_test, soc_params, tolerance=0.02)

   

    # return {
    #     "model":        model,
    #     "history":      history,
    #     "feature_dict": feature_dict_train,
    #     "class_hulls":  class_hulls_train,
    #     "sep_df":       sep_df,
    #     "confusions":   confusions,
    # }


if __name__ == "__main__":
    demo()
