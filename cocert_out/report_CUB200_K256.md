# CoCert crux — does the conic residual quarantine novel-class identity?

KMeans(k=#novel) clustering vs true novel labels (mean±std over seeds).
ARI is primary (NMI saturates on strong ViT features). Claim = QUARANTINE
DIFFERENTIAL: cone gap (residual−explained) >> signed gap on the same atoms.

| vector | ARI | NMI | purity |
|---|--:|--:|--:|
| raw | 0.680±0.015 | 0.882 | 0.773 |
| nnls_residual | 0.699±0.003 | 0.890 | 0.788 |
| nnls_explained | 0.366±0.015 | 0.729 | 0.533 |
| signed_residual | 0.666±0.008 | 0.873 | 0.764 |
| signed_explained | 0.669±0.031 | 0.877 | 0.762 |
| pca_residual | 0.552±0.021 | 0.788 | 0.694 |
| pca_explained | 0.682±0.009 | 0.882 | 0.776 |

quarantine gap (ARI, residual−explained):  cone +0.333   signed -0.003   Δ +0.336
direction vs magnitude:  residual-direction ARI 0.699  vs  residual-magnitude AUROC 0.527 (mahalanobis 0.649)

raw ARI = 0.680   nnls-residual = 0.699   nnls-explained = 0.366   signed-residual = 0.666   signed-explained = 0.669

[GATE 1 quarantine]  residual>=0.9*raw (0.699>=0.612) AND explained<=0.6*raw (0.366<=0.408) : PASS
[GATE 2 cone-specific] cone-gap > signed-gap+0.10 (+0.333 > +0.097) : PASS

VERDICT: GO — non-negativity quarantines novel-class identity into an auditable KKT residual that signed decomposition does not. Build the interpretability/certificate paper (NOT the acquisition-efficiency one: residual only ties raw for clustering).

NOTE: still confounded until the reconstruction-fraction control (‖explained‖/‖q‖ for novel vs base) and the CUB-200 (non-Gaussian) run agree — CIFAR features are ~Gaussian and cone wins flip on dataset 2.
