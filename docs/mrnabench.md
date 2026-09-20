# mRNABench Sample mean-ribosome-load evaluation

`mrnabench-sample-mrl-v1` evaluates one target in one of four Sample datasets.
It does not implement the full mRNABench catalogue. Predictions concern mean
ribosome load measured in reporter assays, rather than endogenous translation
in every biological context.

| Dataset | Target columns | Eligible rows | Test rows (seed 2541) |
|---|---|---:|---:|
| `egfp` | `target_mrl_egfp_m1pseudo` | 318,459 | 47,769 |
| `egfp` | `target_mrl_egfp_pseudo` | 318,468 | 47,771 |
| `egfp` | `target_mrl_egfp_unmod` | 318,468 | 47,771 |
| `mcherry` | `target_mrl_mcherry` | 191,797 | 28,770 |
| `designed` | `target_mrl_designed` | 100,017 | 15,003 |
| `varying` | `target_mrl_varying_length` | 106,530 | 15,980 |

Input is the complete processed sequence from the source parquet, including
reporter context. The first implementation deliberately excludes CDS, splice,
uORF and Kozak tracks. Comparisons with multi-track models must retain that
input difference. Source data use DNA letters for RNA constructs; the runner
does not silently truncate sequences or change their alphabet.

## Prepare verified data

Install `rewirebench[sequence]`. Data access is public, but the publisher's
[dataset card](https://huggingface.co/datasets/morrislab/mrl-sample/blob/ef67f7cf8a999bb1c412ad6551aa7d9f901cbb95/README.md)
reports the data licence as unknown. The runner does not redistribute datasets.
Upstream software is AGPL-3.0; this independent implementation does not bundle it.

Example immutable eGFP download, followed by explicit checksum verification:

```bash
mkdir -p inputs
curl --fail --location \
  'https://huggingface.co/datasets/morrislab/mrl-sample/resolve/ef67f7cf8a999bb1c412ad6551aa7d9f901cbb95/mrl-sample-egfp.parquet' \
  --output inputs/mrl-sample-egfp.parquet
python - <<'PY'
import hashlib
from pathlib import Path
path = Path('inputs/mrl-sample-egfp.parquet')
with path.open('rb') as handle:
    assert hashlib.file_digest(handle, 'sha256').hexdigest() == 'c007cf66cfcea0807e90baf92ac24b6078102729ad050100bde00f7f0ab4c9b3'
PY
rewirebench inspect mrnabench-sample-mrl-v1
rewirebench prepare mrnabench-sample-mrl-v1 \
  --source inputs/mrl-sample-egfp.parquet --output prepared-mrl \
  --options '{"dataset":"egfp","target":"target_mrl_egfp_unmod","require_official":true,"limit":32}'
```

`inspect` supplies pinned URLs and checksums for every dataset. Remove `limit`
only when requesting all rows for the selected target. Limits remain labelled
`smoke`, retain the original test denominator and never count as complete.
CSV files with `sequence` and the selected target are also accepted as explicit
local, unverified subsets. Use `require_official` to reject altered or unpinned
files. Unverified files cannot claim a complete official evaluation.

Missing target values are excluded before splitting, matching upstream. Nine
eGFP methylpseudouridine targets are missing. Other invalid or infinite values
cause a preparation error. Preparation records source ordinals, exclusions,
all split memberships and their checksum. A two-step sklearn split allocates
70% training and 30% held-out, then divides held-out equally into validation and
test, using seed 2541 each time. These are random splits, not a homology barrier.

## Run your model or a CPU control

```bash
rewirebench run --prepared prepared-mrl \
  --adapter rewirebench.adapters.sequence:SequenceComposition \
  --prediction-type embedding --output mrl-composition-smoke
```

This untrained composition control is supplied by Rewire; it is not the
published mRNABench `NaiveBaseline`. Any local class exposing
`embed(inputs) -> {opaque_id: [float, ...]}` can replace it. Adapter inputs
contain only `id` and `sequence`. Scalar adapters may instead implement
`fit(train_inputs, train_targets)` and `predict(inputs) -> {opaque_id: float}`.
Training labels are provided only through fitting. No code, sequences or
predictions are uploaded.

For frozen embeddings, the evaluator fits `RidgeCV` with alphas
`[0.001, 0.01, 0.1, 1, 10]` on training embeddings and labels only. It does not
normalize features or use validation/test labels to choose the ridge penalty.
Unlike upstream's emergency fallback, numerical fitting failures stop with an
error; they do not silently change the solver to stochastic Ridge.

Import predictions or embeddings generated in an existing private environment:

```bash
rewirebench evaluate --prepared prepared-mrl \
  --predictions my-predictions.json --output mrl-imported
rewirebench evaluate --prepared prepared-mrl \
  --embeddings my-embeddings.json --output mrl-imported-embeddings
```

Prediction JSON maps prepared opaque IDs to scalar scores. Embedding JSON maps
those IDs to finite equal-width vectors. All selected embeddings are required
for fitting. Evaluation reports test MSE (lower is better), Pearson and Spearman
correlations (higher is better), plus coverage and missing predictions.
Correlations with constant values or fewer than two scored rows are unavailable,
not zero. There is no average across chemically distinct targets or datasets.
Prepared evaluation is offline; no network request occurs in fitting/scoring.
Use native CPU execution or the package's common Podman/Apptainer environment;
model-specific accelerator requirements depend on your adapter.

## Validation and scope

The packaged `resources/mrnabench/validation-receipt.json` records automated
checks against the actual upstream splitter, regression probe and metrics at
[revision 74f96b8](https://github.com/morrislab/mRNABench/tree/74f96b8e6ae9f41cc3cccff089d826a62d5604b8).
All four source files were downloaded and checksum verified. Every target's
full split was prepared; CPU probes used only the first 80 original rows per
target. Splits matched exactly and metric differences were below `1e-9`.
This validates implementation parity on small inputs, not published model
performance or an entire mRNABench execution. The upstream default displays
validation metrics; our test metrics correspond to its `eval_all_splits=True`
held-out test output.

Original assay: [Sample et al. (2019)](https://doi.org/10.1038/s41587-019-0164-5).
Processed data: [pinned mRL Sample release](https://huggingface.co/datasets/morrislab/mrl-sample/tree/ef67f7cf8a999bb1c412ad6551aa7d9f901cbb95).
The packaged source manifest retains exact code and dataset hashes.
