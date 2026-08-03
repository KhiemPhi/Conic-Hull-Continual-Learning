# Conic-trained representation: does the cone beat multi-prototype? (CIFAR100)

Cone vs multi-prototype OOD AUROC (mean over OOD datasets) in three spaces. If 'conic' flips Δ positive while raw/generic stay negative, a conic-trained space is what the cone needs.

| space | budget m | cone | multiproto | Δ(cone−mp) |
|---|--:|--:|--:|--:|
| raw | 2 | 0.9643 | 0.9754 | -0.0111 |
| raw | 4 | 0.9685 | 0.9763 | -0.0078 |
| generic | 2 | 0.9285 | 0.9485 | -0.0199 |
| generic | 4 | 0.9399 | 0.9492 | -0.0093 |
| conic | 2 | 0.8921 | 0.9491 | -0.0570 |
| conic | 4 | 0.9221 | 0.9497 | -0.0276 |
