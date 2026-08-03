# CoCert crux — does the conic residual quarantine novel-class identity?

KMeans(k=#novel) clustering vs true novel labels (mean±std over seeds).
ARI is primary (NMI saturates on strong ViT features). Claim = QUARANTINE
DIFFERENTIAL: cone gap (residual−explained) >> signed gap on the same atoms.

| vector | ARI | NMI | purity |
|---|--:|--:|--:|
| raw | 0.618±0.006 | 0.800 | 0.763 |
| nnls_residual | 0.660±0.003 | 0.815 | 0.793 |
| nnls_explained | 0.215±0.011 | 0.502 | 0.380 |
| signed_residual | 0.540±0.017 | 0.803 | 0.786 |
| signed_explained | 0.581±0.020 | 0.767 | 0.726 |
| pca_residual | 0.357±0.011 | 0.737 | 0.711 |
| pca_explained | 0.612±0.022 | 0.789 | 0.750 |

quarantine gap (ARI, residual−explained):  cone +0.445   signed -0.041   Δ +0.486
direction vs magnitude:  residual-direction ARI 0.660  vs  residual-magnitude AUROC 0.587 (mahalanobis 0.762)

raw ARI = 0.618   nnls-residual = 0.660   nnls-explained = 0.215   signed-residual = 0.540   signed-explained = 0.581

[GATE 1 quarantine]  residual>=0.9*raw (0.660>=0.557) AND explained<=0.6*raw (0.215<=0.371) : PASS
[GATE 2 cone-specific] cone-gap > signed-gap+0.10 (+0.445 > +0.059) : PASS

VERDICT: GO — non-negativity quarantines novel-class identity into an auditable KKT residual that signed decomposition does not. Build the interpretability/certificate paper (NOT the acquisition-efficiency one: residual only ties raw for clustering).

NOTE: still confounded until the reconstruction-fraction control (‖explained‖/‖q‖ for novel vs base) and the CUB-200 (non-Gaussian) run agree — CIFAR features are ~Gaussian and cone wins flip on dataset 2.
