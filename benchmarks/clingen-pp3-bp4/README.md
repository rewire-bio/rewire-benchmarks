# ClinGen PP3/BP4 Score Calibration Benchmark

Calibrates genomic foundation models and clinical variant-effect predictors into ACMG/AMP computational evidence thresholds (`PP3` and `BP4`, points `+1` to `+8` and `-1` to `-8`) on temporally held-out ClinVar GRCh38 missense SNVs using CPU-only precomputed score tables.

## Method

- **Calibration framework**: Pejaver et al. 2022 (`PMC9748256`, disease prior `alpha = 0.0441`, `c = 1124`, `minpoints = 100`, gnomAD population reference fraction `gft = 0.03`) combined with Bergquist et al. 2025 (`PMC12208618`, adding `+3` / `-3` intermediate strength levels and 1,000-bootstrap one-sided 95% conservative discounted thresholds, `seed = 20261123`).
- **Allele-frequency control**: Following `rewire-bio/rewire-benchmarks#53` and `#54`, rare variants in `post_2023_12_af001` are restricted to gnomAD v3.1.2 non-1000G-overlapping genome `grpmax < 0.1%` (`0.001`) queried via MyVariant.info so benign common polymorphisms do not flatter pathogenic separation.
- **Temporal holdouts**:
  - `post_2023_12` (primary): `9,450` ClinVar `2026-11-08` `>=2-star` clean missense SNVs (`5,508` P/LP, `3,942` B/LB; `4,188` novel since `2023-12-26`, excluding all variants classified P/LP or B/LB in `clinvar_20231226.vcf.gz`).
  - `post_2023_12_af001` (AF-controlled): `8,141` missense SNVs (`5,454` P/LP, `2,687` B/LB) with gnomAD v3.1.2 `grpmax < 0.1%`.
  - `post_2025_03` (sensitivity): `3,818` ClinVar `2026-11-08` `>=2-star` missense SNVs (`2,306` P/LP, `1,512` B/LB; `1,134` novel since `2025-03-30`) excluding variants classified P/LP or B/LB in `clinvar_20250330.vcf.gz`.
  - `gnomad_ref_missense`: `6,312` population reference missense SNVs (`5,155` ClinVar VUS with population AF and `1,157` gnomAD missense background SNVs, disjoint from held-out evaluation cohorts).

## Primary Results (`post_2023_12` and `post_2023_12_af001`)

### Held out from Dec 2023 (`post_2023_12`)

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

### Allele-Frequency Controlled (`post_2023_12_af001`, gnomAD v3.1.2 `grpmax < 0.1%`)

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

Key findings:

- **Rare-variant calibration separation**: Under `grpmax < 0.1%`, AlphaGenome AVI loses `PP3 Strong (+4)` (`not reached`, down from `2.876` raw / `52.5` PHRED unfiltered), whereas REVEL (`0.916`) and AlphaMissense (`0.987`) retain `PP3 Strong (+4)`. Unsupervised alignment and genome models (GPN-MSA, GPN-Star v100/m447) and CADD v1.7 reach up to `PP3 Moderate (+2)` or `+3` (`not reached` at `+4`) across cohorts.
- **AlphaMissense BP4 shift under AF control**: Restricting benign controls to rare missense variants (`grpmax < 0.1%`) eliminates AlphaMissense `BP4` thresholds (`not reached`).
- **EVEE leakage caveat**: EVEE (`Zenodo 19701997`) incorporates ClinVar supervision through `2026-03-20` and therefore has direct target exposure on the `post_2023_12` and `post_2025_03` cohorts. It is included solely as a descriptive leaked control and withheld from unconfounded model comparisons.

## Reproduction (CPU Only)

Re-run full 1,000-bootstrap calibration from the committed `cohorts/` and `scores/` tables on CPU (~170s on Apple Silicon):

```bash
uv run --no-sync python -m clingen_pp3_bp4.calibrate --n-boot 1000
cd benchmarks/clingen-pp3-bp4/results && shasum -a 256 -c SHA256SUMS
```

Run the unit and integrity tests:

```bash
uv run --no-sync pytest benchmarks/clingen-pp3-bp4/tests
```
