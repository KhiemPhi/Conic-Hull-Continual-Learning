# Conic vs signed dictionary on ViT patch tokens (code quality)

Matched dict size & sparsity (Top-K). coherence/purity = monosemanticity (higher better). selectivity = targeted/collateral AP drop on atom ablation (higher = cleaner concept edits).

| method | L0 | nMSE↓ | coherence↑ | purity↑ | edit-sel↑ | targetedΔAP | probe mAP |
|---|--:|--:|--:|--:|--:|--:|--:|
| conic-SAE | 16.0 | 0.227 | 0.659 | 0.498 | 464.66 | 0.008 | 0.594 |
| signed-SAE | 16.0 | 0.237 | 0.521 | 0.409 | 53.47 | 0.012 | 0.613 |
| PCA | 16.0 | 0.514 | 0.334 | 0.533 | -22.31 | -0.001 | 0.396 |
| kmeans | 1.0 | 0.478 | 0.620 | 0.607 | 41.13 | 0.004 | 0.649 |
