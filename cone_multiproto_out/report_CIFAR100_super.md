# Cone vs multi-prototype NCM on multimodal ID (CIFAR100_super)

ID = CIFAR100 merged into 20 super-classes (~5 modes each). OOD = other datasets. Matched budget: cone n_rays = m vs m prototypes/class. Mean over OOD sets.

| budget m | AUROC cone | AUROC multiproto | Δ(cone−mp) | FPR95 cone | FPR95 mp |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.9157 | 0.9583 | -0.0426 | 0.327 | 0.131 |
| 2 | 0.9342 | 0.9646 | -0.0304 | 0.220 | 0.120 |
| 3 | 0.9428 | 0.9668 | -0.0241 | 0.208 | 0.106 |
| 5 | 0.9566 | 0.9725 | -0.0159 | 0.126 | 0.093 |
| 8 | 0.9580 | 0.9723 | -0.0143 | 0.115 | 0.098 |
