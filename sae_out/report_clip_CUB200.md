# Conic vs signed dictionary on ViT patch tokens (code quality)

Matched dict size & sparsity (Top-K). coherence/purity = monosemanticity (higher better). selectivity = targeted/collateral AP drop on atom ablation (higher = cleaner concept edits).

| method | L0 | nMSE↓ | coherence↑ | purity↑ | edit-sel↑ | targetedΔAP | probe mAP |
|---|--:|--:|--:|--:|--:|--:|--:|
| conic-SAE | 16.0 | 0.286 | 0.600 | 0.486 | 100.00 | 0.004 | 0.625 |
| signed-SAE | 16.0 | 0.298 | 0.498 | 0.325 | 18.09 | 0.007 | 0.552 |
| PCA | 16.0 | 0.449 | 0.390 | 0.453 | -26.76 | -0.003 | 0.372 |
| kmeans | 1.0 | 0.484 | 0.653 | 0.585 | 49.85 | 0.010 | 0.566 |
