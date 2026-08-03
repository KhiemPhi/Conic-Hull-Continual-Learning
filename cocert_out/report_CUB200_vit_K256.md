# CoCert crux — does the conic residual quarantine novel-class identity?

KMeans(k=#novel) clustering vs true novel labels (mean±std over seeds).
ARI primary (NMI saturates). Claim = QUARANTINE DIFFERENTIAL.

| vector | ARI | NMI | purity |
|---|--:|--:|--:|
| raw | 0.680±0.015 | 0.882 | 0.773 |
| nnls_residual | 0.686±0.013 | 0.892 | 0.786 |
| nnls_explained | 0.379±0.014 | 0.739 | 0.549 |
| signed_residual | 0.688±0.028 | 0.878 | 0.783 |
| signed_explained | 0.664±0.031 | 0.877 | 0.767 |
| pca_residual | 0.533±0.018 | 0.792 | 0.697 |
| pca_explained | 0.675±0.026 | 0.881 | 0.769 |

quarantine gap (ARI, residual−explained):  cone +0.307   signed +0.023   Δ +0.283
[GATE 1 quarantine]  residual>=0.9*raw (0.686>=0.612) AND explained<=0.6*raw (0.379<=0.408) : PASS
[GATE 2 cone-specific] cone-gap > signed-gap+0.10 (+0.307 > +0.123) : PASS
VERDICT: GO — non-negativity quarantines novel-class identity into an auditable KKT residual (build the interpretability/certificate paper, NOT acquisition: residual only ties raw).

## step 1 — reconstruction fraction  ‖explained‖/‖q‖  (q unit-norm)
| method | base | novel |
|---|--:|--:|
| nnls | 0.493 | 0.480 |
| signed | 0.932 | 0.916 |
| pca | 0.966 | 0.953 |
(nontrivial only if nnls reconstructs a substantial fraction of NOVEL points yet still strips their identity — cf. crux explained ARI.)

## step 3 — nameability of the 'known parts' attribution
3a attribution consistency (ARI, cluster base-class attribution of novel imgs):  nnls 0.484   signed 0.680
3b semantic coherence (top attributed part shares novel's coarse group):  nnls 0.134   signed 0.451   chance 0.037
