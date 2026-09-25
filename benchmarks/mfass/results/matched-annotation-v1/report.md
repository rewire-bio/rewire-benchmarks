# MFASS annotation-matched specialist study: results

Manifest SHA-256 `203567b9055ae1350a910e100c879cace9567411f4e57d932318feda1b0231dd`. Generated 2026-09-25T09:11:00+00:00.
Exploratory, unadjusted intervals; not a confirmatory comparison. Missing scores are coverage gaps, not negative predictions.

## Conditions

| Condition | Scored / 8,324 | P@100 | AP | AUROC | Tied at the 100th score |
|---|---:|---:|---:|---:|---:|
| S0 | 8,297 | 0.630 | 0.295 | 0.804 | 1 |
| S1 | 8,297 | 0.650 | 0.313 | 0.815 | 2 |
| P0 | 8,297 | 0.650 | 0.389 | 0.876 | 1 |
| P1 | 8,297 | 0.660 | 0.411 | 0.873 | 3 |

## Contrasts (candidate minus baseline, common scored variants)

| Contrast | Common | Positives | Groups | Metric | Observed delta | 95% interval |
|---|---:|---:|---:|---|---:|---|
| S1-S0 | 8,297 | 314 | 460 | precision_at_capacity | +0.020 | [0.000, 0.062] |
| S1-S0 | 8,297 | 314 | 460 | average_precision_sklearn | +0.017 | [0.006, 0.028] |
| S1-S0 | 8,297 | 314 | 460 | auroc | +0.011 | [-0.007, 0.027] |
| P1-P0 | 8,297 | 314 | 460 | precision_at_capacity | +0.010 | [0.000, 0.040] |
| P1-P0 | 8,297 | 314 | 460 | average_precision_sklearn | +0.022 | [0.011, 0.034] |
| P1-P0 | 8,297 | 314 | 460 | auroc | -0.004 | [-0.019, 0.013] |
| P0-S0 | 8,297 | 314 | 460 | precision_at_capacity | +0.020 | [-0.033, 0.083] |
| P0-S0 | 8,297 | 314 | 460 | average_precision_sklearn | +0.093 | [0.064, 0.124] |
| P0-S0 | 8,297 | 314 | 460 | auroc | +0.073 | [0.048, 0.096] |
