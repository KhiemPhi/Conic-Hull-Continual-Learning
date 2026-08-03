# Conic vs signed dictionary on ViT patch tokens (code quality)

Matched dict size & sparsity (Top-K). coherence/purity = monosemanticity (higher better). selectivity = targeted/collateral AP drop on atom ablation (higher = cleaner concept edits).

| method | L0 | nMSE↓ | coherence↑ | purity↑ | edit-sel↑ | targetedΔAP | probe mAP |
|---|--:|--:|--:|--:|--:|--:|--:|
| conic-SAE | 16.0 | 0.325 | 0.630 | 0.793 | 1079.06 | 0.051 | 0.686 |
| signed-SAE | 16.0 | 0.335 | 0.453 | 0.695 | 681.05 | 0.067 | 0.636 |
| PCA | 16.0 | 0.655 | 0.219 | 0.611 | -24.65 | -0.002 | 0.502 |
| kmeans | 1.0 | 0.612 | 0.534 | 0.844 | 302.10 | 0.012 | 0.607 |
