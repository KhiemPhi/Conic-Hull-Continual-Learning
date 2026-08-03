# Cone-separability diagnostic

Intra **diameter** = max pairwise angle (512-subsample). **half** = mean angle to normalised centroid (cone half-angle). **inter** = centroid-centroid angle; **nn** = nearest-centroid angle (median). Verdict per protocol decision rule.

| backbone | dataset | C | N | diam med / p95 | half med | inter med / nn / min | tight<75° | acute<90° | verdict |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| vit_base_patch16_224 (timm/IN1k) | CIFAR10 | 10 | 10000 | 89.2/94.0 | 49.6 | 64.2/52.6/44.1 | 0% | 80% | **OVERLAP** |
| vit_base_patch16_224 (timm/IN1k) | CIFAR100 | 100 | 10000 | 86.3/90.0 | 45.6 | 69.4/41.1/16.1 | 0% | 94% | **OVERLAP** |
| vit_base_patch16_224 (timm/IN1k) | STL10 | 10 | 8000 | 90.7/97.4 | 48.3 | 70.7/63.4/59.1 | 0% | 40% | **RED** |
| vit_base_patch16_224 (timm/IN1k) | FGVCAircraft | 100 | 3333 | 54.9/77.0 | 22.5 | 17.1/7.7/4.8 | 89% | 100% | **OVERLAP** |
| vit_base_patch16_224 (timm/IN1k) | Flowers102 | 102 | 6149 | 61.9/77.1 | 26.9 | 66.7/44.1/28.4 | 91% | 100% | **GREEN** |
| vit_base_patch16_224 (timm/IN1k) | OxfordIIITPet | 37 | 3669 | 83.1/92.2 | 38.7 | 86.3/52.4/23.4 | 14% | 89% | **RED** |
| vit_base_patch16_224 (timm/IN1k) | Food101 | 101 | 25250 | 86.6/92.2 | 42.1 | 62.7/31.8/13.9 | 0% | 86% | **OVERLAP** |
| vit_base_patch16_224 (timm/IN1k) | CUB200 | 200 | 5794 | 68.5/80.8 | 32.5 | 71.3/25.0/9.7 | 80% | 100% | **OVERLAP** |
| vit_base_patch16_224 (timm/IN1k) | StanfordCars | 196 | 8040 | 72.5/81.7 | 34.9 | 50.5/15.5/8.9 | 65% | 100% | **OVERLAP** |
| vit_base_patch16_224 (timm/IN1k) | ImageNet-A | 200 | 7500 | 93.2/96.4 | 62.4 | 81.8/58.8/37.2 | 0% | 12% | **RED** |
| vit_base_patch16_224 (timm/IN1k) | ImageNet-R | | | | | | | | ERROR: axiong/imagenet-r: can't find image/labe |
| dinov2_vitb14 (SSL) | CIFAR10 | 10 | 10000 | 94.4/99.3 | 57.7 | 76.1/69.4/56.5 | 0% | 0% | **RED** |
| dinov2_vitb14 (SSL) | CIFAR100 | 100 | 10000 | 91.1/94.5 | 54.1 | 80.4/55.7/20.7 | 0% | 36% | **RED** |
| dinov2_vitb14 (SSL) | STL10 | 10 | 8000 | 98.8/102.3 | 61.6 | 83.8/81.3/77.6 | 0% | 0% | **RED** |
| dinov2_vitb14 (SSL) | FGVCAircraft | 100 | 3333 | 83.6/96.3 | 37.3 | 79.8/14.9/8.4 | 29% | 59% | **RED** |
| dinov2_vitb14 (SSL) | Flowers102 | 102 | 6149 | 63.0/81.4 | 27.8 | 87.6/65.0/35.4 | 85% | 98% | **OVERLAP** |
| dinov2_vitb14 (SSL) | OxfordIIITPet | 37 | 3669 | 87.0/95.6 | 39.9 | 87.4/56.9/16.1 | 11% | 68% | **RED** |
| dinov2_vitb14 (SSL) | Food101 | 101 | 25250 | 91.1/95.7 | 43.1 | 82.4/44.5/20.1 | 0% | 35% | **RED** |
| dinov2_vitb14 (SSL) | CUB200 | 200 | 5794 | 72.3/87.3 | 33.5 | 86.4/31.1/12.1 | 60% | 98% | **OVERLAP** |
| dinov2_vitb14 (SSL) | StanfordCars | 196 | 8040 | 88.1/94.1 | 43.7 | 77.3/25.8/11.1 | 6% | 66% | **RED** |
| dinov2_vitb14 (SSL) | ImageNet-A | 200 | 7500 | 92.6/95.2 | 61.6 | 84.5/60.4/37.9 | 0% | 14% | **RED** |
| dinov2_vitb14 (SSL) | ImageNet-R | | | | | | | | ERROR: axiong/imagenet-r: can't find image/labe |
| CLIP ViT-B-16 (laion2b) | CIFAR10 | 10 | 10000 | 72.9/79.2 | 35.7 | 42.5/30.8/23.2 | 80% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | CIFAR100 | 100 | 10000 | 66.9/74.4 | 33.2 | 44.4/24.1/10.6 | 97% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | STL10 | 10 | 8000 | 75.9/82.0 | 39.1 | 51.6/39.4/33.6 | 40% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | FGVCAircraft | 100 | 3333 | 59.4/67.0 | 32.0 | 36.7/13.1/7.4 | 98% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | Flowers102 | 102 | 6149 | 48.2/58.2 | 21.9 | 41.7/24.6/16.3 | 100% | 100% | **GREEN** |
| CLIP ViT-B-16 (laion2b) | OxfordIIITPet | 37 | 3669 | 63.8/69.5 | 29.1 | 43.2/21.9/11.4 | 97% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | Food101 | 101 | 25250 | 66.4/72.9 | 31.4 | 41.5/24.5/11.3 | 100% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | CUB200 | 200 | 5794 | 53.5/62.5 | 26.4 | 43.4/18.5/9.2 | 100% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | StanfordCars | 196 | 8040 | 68.3/73.9 | 34.1 | 56.2/26.6/12.1 | 96% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | ImageNet-A | 200 | 7500 | 79.0/86.2 | 45.7 | 49.1/28.7/11.8 | 22% | 100% | **OVERLAP** |
| CLIP ViT-B-16 (laion2b) | ImageNet-R | | | | | | | | ERROR: axiong/imagenet-r: can't find image/labe |
