# CoCert crux — does the conic residual quarantine novel-class identity?

KMeans(k=#novel) clustering vs true novel labels (mean±std over seeds).
ARI primary (NMI saturates). Claim = QUARANTINE DIFFERENTIAL.

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
[GATE 1 quarantine]  residual>=0.9*raw (0.660>=0.557) AND explained<=0.6*raw (0.215<=0.371) : PASS
[GATE 2 cone-specific] cone-gap > signed-gap+0.10 (+0.445 > +0.059) : PASS
VERDICT: GO — non-negativity quarantines novel-class identity into an auditable KKT residual (build the interpretability/certificate paper, NOT acquisition: residual only ties raw).

## step 1 — reconstruction fraction  ‖explained‖/‖q‖  (q unit-norm)
| method | base | novel |
|---|--:|--:|
| nnls | 0.421 | 0.408 |
| signed | 0.883 | 0.833 |
| pca | 0.940 | 0.900 |
(nontrivial only if nnls reconstructs a substantial fraction of NOVEL points yet still strips their identity — cf. crux explained ARI.)

## step 3 — nameability of the 'known parts' attribution
3a attribution consistency (ARI, cluster base-class attribution of novel imgs):  nnls 0.286   signed 0.521
3b semantic coherence (top attributed part shares novel's coarse group):  nnls 0.193   signed 0.350   chance 0.039
