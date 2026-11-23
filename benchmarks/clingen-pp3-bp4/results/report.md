# ClinGen PP3/BP4 Score Calibration on Temporally Held-Out ClinVar Variants

Calibrated following Pejaver et al. 2022 (`PMC9748256`, `alpha = 0.0441`, `c = 1124`) and
Bergquist et al. 2025 (`PMC12208618`, adding +/-3pt intermediate levels and one-sided 95% conservative bootstrap bounds over `n_boot = 1000`).

## Primary Stratum: ClinVar >=2-star missense SNVs held out from Dec 2023 (`post_2023_12`)

| Predictor | Role | n P/LP | n B/LB | AUROC (95% CI) | BP4 Mod (-2) | BP4 Supp (-1) | PP3 Supp (+1) | PP3 Mod (+2) | PP3 Strong (+4) | P/LP >=+1 | B/LB <=-1 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| GPN-MSA (-LLR) | foundation_model | 5508 | 3942 | 0.863 (0.855, 0.870) | 3.81 | 4.72 | 9.27 | 10.75 | not reached | 47.2% | 48.5% |
| GPN-Star vertebrate v100 (-LLR) | foundation_model | 5508 | 3942 | 0.889 (0.882, 0.895) | 3.57 | 4.94 | 8.77 | 10.14 | not reached | 55.9% | 57.6% |
| GPN-Star mammal m447 (-LLR) | foundation_model | 5508 | 3942 | 0.851 (0.843, 0.859) | 3.69 | 4.90 | 8.33 | 10.18 | not reached | 45.1% | 54.2% |
| AlphaGenome AVI (raw) | foundation_model | 5508 | 3942 | 0.944 (0.940, 0.949) | 0.551 | 0.682 | 1.148 | 1.323 | 2.876 | 78.0% | 69.0% |
| AlphaGenome AVI (PHRED) | foundation_model | 5508 | 3942 | 0.944 (0.940, 0.949) | 19.8 | 21.8 | 25.2 | 26.3 | 52.5 | 77.7% | 69.3% |
| AlphaMissense | clinical_baseline | 5438 | 3623 | 0.949 (0.945, 0.953) | 0.120 | 0.162 | 0.628 | 0.826 | 0.975 | 77.4% | 69.8% |
| REVEL v1.3 | clinical_baseline | 5493 | 3839 | 0.968 (0.965, 0.972) | 0.282 | 0.372 | 0.665 | 0.758 | 0.909 | 85.1% | 80.4% |
| CADD v1.7 (PHRED) | clinical_baseline | 5508 | 3942 | 0.900 (0.894, 0.906) | 19.7 | 22.3 | 26.0 | 29.3 | not reached | 68.4% | 65.0% |
| EVEE (ClinVar-supervised, descriptive) | descriptive_leaked | 5344 | 3802 | 0.966 (0.963, 0.970) | 0.046 | 0.094 | 0.471 | 0.701 | 0.891 | 81.8% | 79.8% |

## Allele-Frequency Controlled Stratum (`post_2023_12_af001`, gnomAD v3.1.2 grpmax < 0.1%)

| Predictor | Role | n P/LP | n B/LB | AUROC (95% CI) | BP4 Mod (-2) | BP4 Supp (-1) | PP3 Supp (+1) | PP3 Mod (+2) | PP3 Strong (+4) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| GPN-MSA (-LLR) | foundation_model | 5454 | 2687 | 0.851 (0.842, 0.861) | 3.80 | 4.67 | 9.37 | 11.03 | not reached |
| GPN-Star vertebrate v100 (-LLR) | foundation_model | 5454 | 2687 | 0.876 (0.867, 0.883) | 3.43 | 4.88 | 9.01 | 10.46 | not reached |
| GPN-Star mammal m447 (-LLR) | foundation_model | 5454 | 2687 | 0.830 (0.820, 0.840) | 3.72 | 4.88 | 8.96 | 11.94 | not reached |
| AlphaGenome AVI (raw) | foundation_model | 5454 | 2687 | 0.941 (0.935, 0.946) | 0.560 | 0.683 | 1.170 | 1.334 | not reached |
| AlphaGenome AVI (PHRED) | foundation_model | 5454 | 2687 | 0.941 (0.935, 0.946) | 20.0 | 21.9 | 25.3 | 26.6 | not reached |
| AlphaMissense | clinical_baseline | 5384 | 2471 | 0.947 (0.942, 0.952) | not reached | not reached | 0.723 | 0.873 | 0.987 |
| REVEL v1.3 | clinical_baseline | 5439 | 2615 | 0.969 (0.965, 0.972) | 0.283 | 0.371 | 0.683 | 0.754 | 0.916 |
| CADD v1.7 (PHRED) | clinical_baseline | 5454 | 2687 | 0.894 (0.886, 0.902) | 19.7 | 22.4 | 26.0 | 32.0 | not reached |
| EVEE (ClinVar-supervised, descriptive) | descriptive_leaked | 5290 | 2628 | 0.968 (0.964, 0.972) | 0.046 | 0.095 | 0.468 | 0.710 | 0.883 |

## Sensitivity Stratum (`post_2025_03`, held out from 2025-03-30 ClinVar)

| Predictor | Role | n P/LP | n B/LB | AUROC | PP3 Supp (+1) | PP3 Strong (+4) |
|---|---|---:|---:|---:|---:|---:|
| GPN-MSA (-LLR) | foundation_model | 2306 | 1512 | 0.868 | 9.58 | not reached |
| GPN-Star vertebrate v100 (-LLR) | foundation_model | 2306 | 1512 | 0.890 | 8.76 | not reached |
| GPN-Star mammal m447 (-LLR) | foundation_model | 2306 | 1512 | 0.856 | 8.66 | not reached |
| AlphaGenome AVI (raw) | foundation_model | 2306 | 1512 | 0.941 | 1.148 | not reached |
| AlphaGenome AVI (PHRED) | foundation_model | 2306 | 1512 | 0.941 | 25.2 | not reached |
| AlphaMissense | clinical_baseline | 2268 | 1390 | 0.947 | 0.766 | not reached |
| REVEL v1.3 | clinical_baseline | 2302 | 1481 | 0.961 | 0.713 | 0.994 |
| CADD v1.7 (PHRED) | clinical_baseline | 2306 | 1512 | 0.895 | 26.9 | not reached |
| EVEE (ClinVar-supervised, descriptive) | descriptive_leaked | 2153 | 1376 | 0.958 | 0.585 | 0.916 |
