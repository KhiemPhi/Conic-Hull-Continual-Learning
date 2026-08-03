# Cone vs multi-prototype NCM on multimodal ID (CUB200_merge10)

ID = CUB200 merged into 20 super-classes (~10 modes each). OOD = other datasets. Matched budget: cone n_rays = m vs m prototypes/class. Mean over OOD sets.

| budget m | AUROC cone | AUROC multiproto | Δ(cone−mp) | FPR95 cone | FPR95 mp |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.9963 | 0.9976 | -0.0013 | 0.013 | 0.010 |
| 2 | 0.9980 | 0.9985 | -0.0005 | 0.008 | 0.008 |
| 3 | 0.9983 | 0.9987 | -0.0004 | 0.008 | 0.006 |
| 5 | 0.9983 | 0.9987 | -0.0003 | 0.008 | 0.006 |
| 8 | 0.9982 | 0.9988 | -0.0006 | 0.009 | 0.005 |
