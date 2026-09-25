# DART-Eval task 1: regulatory elements against shuffled controls

`dart-eval-task1-zero-shot-v1` scores a higher-is-better scalar sequence score
for each element and its existing paired control. No training or embedding probe
is permitted. This is one zero-shot task, not the complete DART-Eval suite.

The official task uses ENCODE v3 cCREs expanded to 350 bp, centred on the element
midpoint. Only the element within each window is dinucleotide shuffled; flanking
sequence is retained. The pinned loader also handles reverse complements. Do not
replace this procedure with a shuffle of the complete window or a different random
seed. The official evaluation selects chr5, chr10, chr14, chr18, chr20 and chr22,
with control-generation seed 0.

## Inputs and access

The reviewed upstream revision is
[`af2a86d666c35304257c2fa7e15180e1fbcabb01`](https://github.com/kundajelab/DART-Eval/tree/af2a86d666c35304257c2fa7e15180e1fbcabb01).
The source-hash and statistical reference receipts are packaged in
`rewirebench/resources/dart_eval/` with file hashes and URLs; upstream source code is not redistributed because its software licence was not established. Exact source files
used are `task_1_paired_control/export.py`, `components.py`,
`zero_shot/evaluators.py` and `zero_shot/encode_ccre/dnabert2.py`.

The upstream exported inputs are
[Synapse `syn64314109`, version 1](https://www.synapse.org/Synapse:syn64314109.1).
Download this file through an authenticated Synapse account after reviewing its
access and reuse terms. Rewire does not obtain credentials or accept terms for you.
No dataset licence is inferred from the software repository.

The file contains `test/seqs`, `test/ctrls` (N × 350 × 4 one-hot arrays), and
`test/idxs` (N original row indices). Rewire imports these existing pairs; it does
not regenerate controls. The train/validation groups are not exposed to adapters.

**Access limitation recorded on 20 September 2026:** metadata was public, but
anonymous downloads of this input file, the DNABERT-2 `scores.tsv` (`syn62153297`)
and its `metrics.json` (`syn62153294`) returned HTTP 403. Their bytes and complete
denominators have not been independently checked. Consequently, local imports
remain labelled `subset` (or `smoke` when limited), never a verified complete
official run. A provided SHA256 checks the local file's integrity, not its source
authenticity. The HDF5 shape alone cannot establish chromosome membership,
seed, genome reference or whether every official pair is present.

Install the optional HDF5 reader:

```bash
python -m pip install 'rewirebench[sequence]'
```

Prepare a local file using its SHA256 from your download receipt:

```bash
rewirebench prepare dart-eval-task1-zero-shot-v1 \
  --source /data/dart/data.h5 --output ./dart-prepared \
  --options '{"source_sha256":"REPLACE_WITH_YOUR_64_CHARACTER_SHA256"}'
```

Preparation and subsequent scoring are local and work offline. The only preparation
writes are the requested new output directory. For a small software check, use the
packaged synthetic data:

```bash
rewirebench prepare dart-eval-task1-zero-shot-v1 \
  --source demo --output ./dart-demo
```

This demo is 12 pairs of random DNA with dinucleotide-preserving shuffled controls.
It is not ENCODE data or evidence about model quality. `limit` selects that many
source pairs while retaining the original local-file denominator.

## Private model adapter

Place this example in `private_adapter.py`, then replace its score function with
your local model's sequence likelihood or explicitly declared sequence score:

```python
class PrivateScore:
    def predict(self, inputs):
        # Toy sequence score for testing only; not a published biological baseline.
        return {row["id"]: float(row["sequence"].count("AAA")) for row in inputs}
```

```bash
rewirebench run --prepared ./dart-demo \
  --adapter private_adapter:PrivateScore --output ./dart-smoke \
  --model-name 'Local toy trinucleotide count; software test only'
```

Adapters receive `id` and `sequence` only, independently ordered with random opaque
IDs persisted in `prepared.json`. Pair identities, control status, labels, source
coordinates and source order remain evaluator metadata. These boundaries reduce
accidental leakage; executing arbitrary adapter code locally is not a sandbox.
The example needs only CPU and no model downloads. Hardware and memory needs for
real inference depend on the adapter and are otherwise unreported here.

For private models in another environment, produce keyed scalar predictions and
use the common `evaluate` command instead of `run`. Do not negate a loss silently:
record how the supplied score relates to sequence likelihood. A generic ranker's
result is not reproduction of a published masked or causal language-model result.

## Motif-hit reference

`dart-h12core-fimo-hit-count-v1` counts FIMO 5.5.9 hits from all HOCOMOCO v12
H12CORE motifs in each sequence. It is a newly specified Rewire aggregation, not
a published DART Task 1 baseline, and has no measured performance. It needs a
local FIMO build and the exact motif file; see
[conventional references](conventional-references.md).

## Scoring and coverage

| Field | Meaning |
|---|---|
| `acc` | Fraction of complete scored pairs with element score strictly greater than control score; ties are incorrect |
| `signed_rank_sum`, `pval` | One-sided Wilcoxon greater-than-zero test on paired score differences, preserving SciPy 1.12 defaults |
| `mean_diff`, `q05_diff`, `q25_diff`, `median_diff`, `q75_diff`, `q95_diff` | Distribution of element-minus-control scores on complete scored pairs |
| `n` | Scored sequences, including a scored member whose partner is missing |
| `n_pairs`, `pairs_denominator` | Complete scored pairs and original local-file pair count |
| `pair_diagnostics.tied_pairs` | Complete scored pairs with equal scores (reported outside `metrics`); these count as incorrect |

Wilcoxon uses the upstream SciPy 1.12 auto rule: the rank-sum distribution for at
most 50 nonzero differences, otherwise a normal approximation with zero differences
removed and tie correction. Its historical integer truncation in the tied exact
branch is preserved. These historical p-values should not be interpreted as a
new exact test for tied data. All-zero differences produce accuracy 0 and null
Wilcoxon statistics with an explicit reason rather than NaN or a fabricated p-value.

Missing one pair member excludes the pair from metrics. Reports retain sequence
coverage and pair coverage separately, with missing-element, missing-control,
missing-both and not-selected counts. Scores from a partial set are not comparable
to a complete set without a matched-sample analysis. Empty scored sets have null
metrics. Duplicate IDs, unknown IDs, invalid HDF5 shapes and nonfinite scores fail
validation.

Expected files are the shared runner's `prepared.json`, `predictions.json`,
`report.json` and `unscored.json`; execution details are recorded inside the report. Inspect the report before an explicit contribution
export; local data and private adapter code are never uploaded automatically.

## Validation performed

The reference receipt records execution of the pinned upstream `evaluate` method with inference
replaced by deterministic synthetic scores. Tests check against those saved outputs. Separate numerical receipts were
produced by running SciPy 1.12.0 in an isolated environment for ties, zeros, positive
and negative differences and the large-sample branch. The implementation freezes
those rules so newer SciPy auto defaults cannot silently change results.

Offline private-adapter execution, input allowlisting, partial coverage, HDF5 local
hash checks and malformed-input rejection are tested. Official biological input
execution, full-model inference, and reproduction of a published DART result have
not been performed. Container execution status is recorded separately by the
package release validation, not implied by these native tests.
