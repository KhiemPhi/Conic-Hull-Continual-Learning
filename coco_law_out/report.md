# COCO multi-label: does the cone (NNLS) delta grow with #labels/image?

mAP by #labels/image (test subset). Δ rows = decoder − linear probe. Prediction: NNLS−linear rises with #labels, steeper for Patch than CLS.


## CLS

| decoder | overall | n=1 | n=2 | n=3 | n=4 | n=5+ |
|---|--:|--:|--:|--:|--:|--:|
| linear | 0.487 | 0.560 | 0.463 | 0.441 | 0.506 | 0.489 |
| nnls-cone | 0.677 | 0.837 | 0.706 | 0.689 | 0.641 | 0.617 |
| ncm-multi | 0.480 | 0.611 | 0.501 | 0.482 | 0.471 | 0.446 |
| **Δ nnls-cone−linear** | +0.191 | +0.277 | +0.243 | +0.248 | +0.135 | +0.128 |
| **Δ ncm-multi−linear** | -0.006 | +0.051 | +0.037 | +0.041 | -0.035 | -0.043 |

## Patch

| decoder | overall | n=1 | n=2 | n=3 | n=4 | n=5+ |
|---|--:|--:|--:|--:|--:|--:|
| linear | 0.351 | 0.386 | 0.363 | 0.389 | 0.391 | 0.423 |
| nnls-cone | 0.645 | 0.744 | 0.653 | 0.649 | 0.599 | 0.590 |
| ncm-multi | 0.438 | 0.461 | 0.394 | 0.420 | 0.419 | 0.436 |
| **Δ nnls-cone−linear** | +0.295 | +0.358 | +0.291 | +0.260 | +0.208 | +0.167 |
| **Δ ncm-multi−linear** | +0.087 | +0.075 | +0.031 | +0.031 | +0.027 | +0.013 |

(n counts per bin: 1=406, 2=600, 3=385, 4=241, 5+=348)

