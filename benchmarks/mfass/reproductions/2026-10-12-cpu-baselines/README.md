# MFASS v2 registered baselines, CPU run, 2026-10-12

A full-protocol run of the two registered MFASS v2 baselines with the released `rewirebench` 0.5.0 wheel, following the `bio-model-benchmarks` skill. It is a rerun with the project's own code and data on new hardware, not an independent reproduction by another group (`independently_reproduced: false` in each report).

## How it was run

- Wheel: `rewirebench-0.5.0-py3-none-any.whl`, checked against the release `SHA256SUMS`; SDK code SHA-256 `ac1615768c09a5a1ed785d95af2bdaf32673fc9a268fcd47b875a7491e6d5eba`.
- Protocol `mfass-v2`, prepared artifact SHA-256 `cbba55d19fea7cb204127343ecb554da169a16f047117eb1db6472a5707f38d3`, split SHA-256 `999ebcb7e63a5c5eaa8780fa468e59ac1f934260ad50102814174c396317f052`, runner revision `bee9133b83f3aedaf2bbb9013f1875515845607e`.
- `rewirebench prepare mfass-v2 ...` then `rewirebench run-baselines`, on macOS-26.6.2-arm64-arm-64bit, Python 3.11.13, CPU only.
- Fit and inference took 4.8 seconds for the k-mer baseline.

## Results (test split, 8,324 variants, 315 positives)

| Baseline | P@100 | AP | AUROC | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `mfass-kmer-position-v2` | 0.610 | 0.2864 | 0.7779 | 8,324/8,324 |
| `training-prior-v1` | 0.040 | 0.0378 | 0.5000 | 8,324/8,324 |

The k-mer baseline matches the published corrected values in the [MFASS README](../../README.md) (P@100 0.610, AP 0.286, AUROC 0.778).

## Files

`baseline-plan.json`, `baseline-manifest.json`, one status file and one folder per baseline (`report.json`, `predictions.json`, `unscored.json`). `SHA256SUMS` covers every file. Nothing here is submitted to the database; the database cites this folder by commit and hash as the run recipe's validation receipt.
