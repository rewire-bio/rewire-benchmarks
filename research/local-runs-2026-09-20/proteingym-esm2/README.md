# ESM-2 8M on the AMFR ProteinGym stability assay

This is a new local Rewire evaluation of **all 2,972 variants** in `AMFR_HUMAN_Tsuboyama_2023_4G3O`, selected before observing the scores. It is one complete assay out of ProteinGym v1.3's 217 DMS substitution assays, **not a whole-suite score or reproduction of a published model result**.

| Metric | Value |
|---|---:|
| Spearman | -0.209 |
| AUC | 0.394 |
| MCC | -0.139 |
| NDCG | 0.440 |
| Top-10% recall | 0.057 |

Values follow upstream three-decimal rounding. Higher values are better for all five metrics. The negative correlation is retained; we did not reverse predictions or choose a different assay after seeing performance. This result does not establish performance on other proteins or the full benchmark. No uncertainty interval was estimated.

## Data and method

The pinned [ProteinGym reference](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/reference_files/DMS_substitutions.csv) identifies a 47-residue human AMFR construct and cDNA-display proteolysis stability measurements from [Tsuboyama et al. (2023)](https://doi.org/10.1038/s41586-023-06328-6). There are 820 single and 2,152 multiple substitutions. The target is ProteinGym's direction-adjusted `DMS_score`, with its supplied `DMS_score_bin` labels; no new label transformation or split was introduced.

The `esm2_t6_8M_UR50D` checkpoint uses masked-marginal log odds: mask each substituted position in the wild-type sequence, calculate log P(mutant) minus log P(wild type), and sum these terms for multiple substitutions. This follows the [pinned upstream ESM procedure](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/esm/compute_fitness.py). The adapter receives no assay labels, MSA or structure. There is no fitting. UniRef50 training overlap is unreported and may exist.

The official archive was obtained using [ProteinGym's v1.3 download instructions](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/README.md#resources). Only the selected CSV was extracted. Its observed checksum and the archive checksum are in `retrieval.json`; these are download receipts, not independently published official checksums. The [official Meta checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt) matches the SHA256 pinned in rewirebench 0.4.0.

## Execution and verification

The released wheel ran in an isolated Python 3.11 environment on macOS arm64 CPU, with one inference thread and seed 0. A separate ten-variant smoke run preceded preparation of the complete assay with **no limit**. Network sockets were blocked throughout inference, evaluation and export. All 2,972 selected variants were scored; none were missing or excluded. No container or accelerator execution is claimed.

The SDK's inference timer was **0.1878 seconds**, excluding model loading, preparation and metric calculation. This short construct reuses cached masked-position probabilities across variants. The timing is an implementation-specific observation, not a general model-speed benchmark.

Saved predictions were rescored exactly. The five metrics also match the [pinned official evaluator](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/performance_DMS_benchmarks.py): its NDCG/top-recall functions were executed directly, with its exact Spearman/AUC/MCC formulas. We did not execute the complete 217-assay upstream aggregation program. `audit.json` and `audit.csv` record automated checks, raw reference values and unresolved evidence gaps.

`report.json` retains the SDK's `scope: subset`, `completion: partial` and `status: partial_track`, because this is not the complete ProteinGym track. Its individual assay has `status: complete`; the top-level suite metrics are empty. The 2,972 denominator applies only to the selected assay.

## Reproduce

Install the [v0.4.0 wheel](https://github.com/rewire-bio/rewire-benchmarks/releases/tag/v0.4.0) with its `esm` extra and pandas; exact executed versions appear in `audit.json`. For the exact executed dependencies, install the wheel and the included lock list:

```sh
python -m pip install --no-deps ./rewirebench-0.4.0-py3-none-any.whl
python -m pip install -r requirements-executed.txt
```

Obtain the official assay CSV and checkpoint above, then run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python reproduce.py \
  --data /data/DMS_ProteinGym_substitutions \
  --checkpoint /weights/esm2_t6_8M_UR50D.pt \
  --output /new/private/amfr-run
```

The independently replayed upstream masked-marginal loop and its actual `label_row` function also agree with **all 2,972 saved predictions exactly**, including multi-mutants. Cache repetition and caller-input immutability pass (`masking-verification.json`). Contact-regression weights are not loaded because contact prediction is unused.

The script verifies the observed assay pin, executes smoke and complete-assay runs separately, rescores saved predictions, checks upstream metric parity, and creates a sanitized SDK contribution bundle. Raw data, sequences, weights, predictions and machine paths remain private. The aggregate report and contribution are public review evidence; they do not automatically publish database records. Submission uses a catalogue review PR while production API submissions remain disabled.
