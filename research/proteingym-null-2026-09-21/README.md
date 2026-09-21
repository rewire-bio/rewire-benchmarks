# ProteinGym AMFR random-ranking control, 21 September 2026

The registered `seeded-random-v1` null control, with seed **0** fixed before execution, scored all **2,972 variants** in `AMFR_HUMAN_Tsuboyama_2023_4G3O`. This is one complete assay out of 217 in ProteinGym v1.3. The report and bundle retain **subset / partial track** labels and contain no whole-track score.

| Metric | Null-control value, upstream three-decimal reporting |
|---|---:|
| Spearman | 0.008 |
| AUC | 0.514 |
| MCC | 0.019 |
| NDCG | 0.526 |
| Top recall | 0.087 |

Each score is determined by SHA256 of `0:<prepared_id>`, using the first 64 bits divided by 2^64. No assay labels, sequences or learned model enter that calculation. This is one fixed random ranking, not an estimated distribution of chance performance. No seed search or uncertainty calculation was performed.

## Evidence and validation

The control uses the exact prepared cohort used for the earlier ESM-2 8M AMFR evaluation. All 2,972 IDs, mutated sequences and labels matched the existing local assay CSV. Its hash matched the previously observed download bytes. **That hash is not an independently published upstream checksum.** The original `local_bytes_hashed_not_independently_source_verified` status remains unchanged in the report, bundle and receipt.

The official packaged reference metadata hash matched its pin and confirmed 2,972 assay variants and 217 track assays. The pinned original upstream evaluator's NDCG and top-recall functions were executed locally. Spearman, AUC and median-threshold MCC were independently recalculated according to the same source. All five rounded values matched exactly. The complete predicted mapping was independently reconstructed from seed 0; canonical prediction, prepared-input and frozen-code hashes matched the report.

`sources.csv` records the origins, evidence locations, hashes and verification limitations. `evidence/verification.json` contains raw and rounded metrics. The exact package archive is referenced by `package-snapshot.json`. The existing distribution metadata says version 0.4.0; the frozen code digest identifies the development implementation actually executed.

No downloads, model inference, raw-data publication or upload occurred. Source CSV, prepared records and predictions remain local. The new SDK contribution bundle is ready for private review submission; the numerical result has not been published by this script.

## Reproduction

Use the frozen package referenced by `package-snapshot.json` and the existing local assay/prepared files. New output directories are required.

```bash
PYTHONPATH=./frozen-package python run_null.py \
  --prepared ./amfr-prepared \
  --assay-source ./AMFR_HUMAN_Tsuboyama_2023_4G3O.csv \
  --output ./new-amfr-null-execution \
  --evidence ./new-amfr-null-evidence
```

The verifier imports the pinned original ProteinGym evaluator and requires pandas, NumPy, SciPy and scikit-learn from the recorded execution environment. It makes no network requests.
