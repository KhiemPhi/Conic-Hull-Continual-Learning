# Conic vs signed dictionary on ViT patch tokens (code quality)

Matched dict size & sparsity (Top-K). coherence/purity = monosemanticity (higher better). selectivity = targeted/collateral AP drop on atom ablation (higher = cleaner concept edits).

| method | L0 | nMSE↓ | coherence↑ | purity↑ | edit-sel↑ | targetedΔAP | probe mAP |
|---|--:|--:|--:|--:|--:|--:|--:|
| conic-SAE | 16.0 | 0.409 | 0.571 | 0.764 | 952.98 | 0.030 | 0.676 |
| signed-SAE | 16.0 | 0.418 | 0.414 | 0.649 | 320.40 | 0.053 | 0.615 |
| PCA | 16.0 | 0.636 | 0.240 | 0.590 | -11.55 | -0.000 | 0.457 |
| kmeans | 1.0 | 0.647 | 0.525 | 0.807 | 294.25 | 0.003 | 0.575 |
