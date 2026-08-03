# CUB fine-tuned OOD: cone vs benchmark detectors

AUROC (ID = CUB test, OOD = other datasets' test). Higher better. Fine-tuned timm ViT-B/16 + LoRA. FPR@95 in the second table.

## AUROC

| method | StanfordCars | FGVCAircraft | Flowers102 | OxfordIIITPet | Food101 | CIFAR100 | ImageNet-A | MEAN |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MSP | 0.997 | 0.999 | 0.989 | 0.998 | 0.996 | 0.980 | 0.975 | 0.991 |
| Energy | 0.939 | 0.976 | 0.818 | 0.959 | 0.888 | 0.752 | 0.773 | 0.872 |
| MaxLogit | 0.994 | 0.999 | 0.968 | 0.996 | 0.988 | 0.940 | 0.937 | 0.975 |
| NCM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| Mahalanobis | 1.000 | 0.999 | 1.000 | 0.998 | 1.000 | 1.000 | 0.999 | 0.999 |
| KNN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |
| **Cone(min_c r_c)** | 0.976 | 0.978 | 0.977 | 0.986 | 0.970 | 0.968 | 0.970 | 0.975 |

## FPR@95 (lower better)

| method | StanfordCars | FGVCAircraft | Flowers102 | OxfordIIITPet | Food101 | CIFAR100 | ImageNet-A | MEAN |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MSP | 0.005 | 0.001 | 0.054 | 0.002 | 0.011 | 0.107 | 0.124 | 0.043 |
| Energy | 0.517 | 0.055 | 0.965 | 0.280 | 0.787 | 0.933 | 0.914 | 0.636 |
| MaxLogit | 0.011 | 0.001 | 0.170 | 0.009 | 0.045 | 0.324 | 0.314 | 0.125 |
| NCM | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.001 |
| Mahalanobis | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.001 |
| KNN | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.001 |
| **Cone(min_c r_c)** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 |
