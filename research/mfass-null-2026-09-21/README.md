# MFASS v2 training-prior control, 21 September 2026

A constant null control was evaluated on all **8,324** canonical held-out variants. Each variant receives the training prevalence, **735 / 19,409 = 0.03786902983152146**. This control uses no sequence information, pretrained model or test labels for fitting.

| Metric | Value |
|---|---:|
| AUROC | 0.5 |
| Average precision, scikit-learn | 0.037842383469485825 |
| Splice-disrupting variants in the first 100 | 4 |
| Precision at 100 | 0.04 |
| Recall at 100 | 0.012698412698412698 |
| Scoring coverage | 8,324 / 8,324 |

All scores are tied. The first-100 result comes from the protocol's fixed, label-independent tie break using NumPy seed 0. **Four of 100 is one realised tie break, not an estimate of ranking ability or expected chance yield.** Average precision equals the test prevalence; no confidence intervals were computed.

## Source and validation

Existing local raw and annotation files matched the pinned upstream hashes. Rebuilding the corrected cohort produced byte-identical output. The old cohort in the separate local `repro` directory did not match and was not used.

The selected corrected cohort has 27,733 variants: 19,409 training records with 735 positives and 8,324 test records with 315 positives. The canonical split checksum matched. Training/test groups were independently confirmed disjoint. Independent inspection of every reference/mutant pair confirmed the position and the 7,770 legacy reverse-complement rows.

`run_baselines(..., baseline_ids=["training-prior-v1"])` executed the control through the package's existing protocol evaluator. The verifier separately parsed the original cohort labels, recalculated the training prior and both ranking metrics, and reconstructed the fixed tie break. Every published number agreed within an absolute tolerance of 1e-12.

The SDK contribution bundle is in `evidence/training-prior.bundle.json`. `sources.csv` records claim origins, checksums and review status. `verification.json` is an automated audit receipt, not a human review or reproduction of a paper result. No source data, labels, raw prediction files, embeddings or machine paths are included in the public evidence. Nothing was downloaded or uploaded by this run.

## Reproduction

Use the exact archived package referenced by `package-snapshot.json`, the local source tables and canonical split at corrected revision `bee9133b83f3aedaf2bbb9013f1875515845607e`. Source reuse terms remain unreported; the data are not redistributed here.

```bash
PYTHONPATH=./frozen-package python run_null.py \
  --source ./local-mfass-inputs \
  --split ./split-v2.tsv \
  --output ./new-null-execution \
  --evidence ./new-null-evidence
```

The source directory must contain the corrected `cohort.tsv`, `snv_data_clean.txt` and `snv_func_annot.txt`. New output directories are required. The script validates hashes and performs no network access. The prepared dataset and prediction files remain local.
