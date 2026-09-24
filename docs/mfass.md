# Run MFASS v2 locally

MFASS measures splice disruption in 170-base minigene assay sequences. The fixed
protocol has 19,409 training and 8,324 held-out variants, assigned by whole exon/gene
components. A larger score means more disruption. The principal operating point is
100 selected variants; average precision and AUROC are reported alongside it.

The corrected baseline centres sequence windows on validated assay-oriented pairs.
SpliceAI and Pangolin instead use genomic context. Their required inputs and
methods are different; a matching metric does not make the procedures equivalent.
The [specialist comparison protocol](mfass-specialist-comparison.md) documents the
remaining annotation/masking confounds, the archived Pangolin masking erratum,
and a proposed sensitivity study. It introduces no new evaluated model scores.

## Access and preparation

The upstream [MFASS source tables](https://github.com/KosuriLab/MFASS/tree/9a8e4f27106be52aeb11acad27f95f5cded663a8/processed_data/snv)
are publicly accessible but do not declare a data reuse licence. Assess the source
terms for your use; the SDK does not redistribute these tables or imply permission.
No genome download, account, GPU or model weights are needed for the baseline.

Install from a checkout of the SDK release, with Python 3.11:

```bash
uv sync --locked --package rewirebench --python 3.11
```

Prepare a new local input directory. The following downloads use immutable source
revisions; preparation also verifies both SHA-256 hashes before reading labels.
Do not repeat these commands in an existing input directory.

```bash
mkdir mfass-input
curl --fail --location \
  https://raw.githubusercontent.com/KosuriLab/MFASS/9a8e4f27106be52aeb11acad27f95f5cded663a8/processed_data/snv/snv_data_clean.txt \
  --output mfass-input/snv_data_clean.txt
curl --fail --location \
  https://raw.githubusercontent.com/KosuriLab/MFASS/9a8e4f27106be52aeb11acad27f95f5cded663a8/processed_data/snv/snv_func_annot.txt \
  --output mfass-input/snv_func_annot.txt

.venv/bin/rewirebench prepare mfass-v2 \
  --source mfass-input --output mfass-baseline-prepared \
  --options '{"limit":80}'
.venv/bin/rewirebench run --prepared mfass-baseline-prepared \
  --adapter rewirebench.adapters.mfass:KmerBaseline \
  --model-name 'Corrected k-mer and position baseline' \
  --training-overlap 'Fitted only on the selected canonical training rows' \
  --output mfass-baseline-smoke
```

| Input | SHA-256 |
|---|---|
| `snv_data_clean.txt` | `a637ca0e307e66ff48811ec7efa22b9ce453bc7883b04f0cacb867f7283132d8` |
| `snv_func_annot.txt` | `71a857fe647c4e68acbb41ca61e959c47e1176de89b1442bd6ca1772aa60d5a1` |
| Generated canonical cohort | `389702ff4c647d7ce10a90092a6fa811ae777d15997baf39ce9aae0346247bd0` |
| Packaged canonical split | `999ebcb7e63a5c5eaa8780fa468e59ac1f934260ad50102814174c396317f052` |

The builder operates in memory without modifying source files. Alternatively,
provide an existing `cohort.tsv` with the exact canonical hash. An explicitly
supplied split must match the packaged split exactly.

`limit=80` takes the first 80 training and first 80 test variants **after validation
of the complete cohort**. It is a smoke test, with coverage 80/8,324, not a benchmark
result. Smaller limits may contain only one training class and correctly refuse
fitting. Omit the `--options` argument and use fresh output directories to request
full evaluation. Historical result files are never overwritten.

The example uses CPU only. The 80-row baseline and encoder smoke tests were executed
on macOS arm64; Linux/container support has its own validation status. Memory and
runtime depend on hardware. No general runtime or minimum RAM guarantee is made.

## A frozen public encoder example

The optional DNABERT-2 adapter loads a local checkpoint and emits masked-mean token
representations for each reference/mutant pair. The evaluator concatenates the
reference and mutant-minus-reference vectors, then fits StandardScaler and a fixed
balanced logistic regression (`C=0.1`, `liblinear`, seed `20260914`) on training rows
only. It does not use the randomly initialised pooler.

The model example requires separately obtained weights and executable model code.
Review [DNABERT-2's model card](https://huggingface.co/zhihan1996/DNABERT-2-117M/tree/b5ae377faa374ee160eec1c27b8494436cc94451)
and access/licensing terms before use. Checkpoints and source data stay local.

```bash
uv sync --locked --package rewirebench --extra dnabert2 --python 3.11
mkdir dnabert2-local
for file in model.safetensors tokenizer.json tokenizer_config.json config.json; do
  curl --fail --location \
    "https://huggingface.co/zhihan1996/DNABERT-2-117M/resolve/b5ae377faa374ee160eec1c27b8494436cc94451/$file" \
    --output "dnabert2-local/$file"
done
for file in bert_layers.py bert_padding.py configuration_bert.py flash_attn_triton.py; do
  curl --fail --location \
    "https://huggingface.co/zhihan1996/DNABERT-2-117M/resolve/7bce263b15377fc15361f52cfab88f8b586abda0/$file" \
    --output "dnabert2-local/$file"
done

.venv/bin/rewirebench prepare mfass-v2-frozen-encoder \
  --source mfass-input --output mfass-encoder-prepared \
  --options '{"limit":80}'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/rewirebench run --prepared mfass-encoder-prepared \
  --adapter rewirebench.adapters.mfass:DNABERT2 \
  --adapter-options '{"checkpoint":"dnabert2-local","batch_size":2}' \
  --model-name 'DNABERT-2 frozen pair embeddings plus logistic regression' \
  --training-overlap 'Exact MFASS sequence overlap with encoder pretraining is unreported' \
  --output mfass-encoder-smoke
```

All eight local files are checked against `DNABERT2.ARTIFACTS` before executable
model code is imported. The separate code revision reproduces the exact auxiliary
code loaded by the historical run; the `bert_layers.py` hash is identical at both
revisions. The adapter always uses CPU float32 and does not download missing files.
Use the supplied pinned environment; another framework version is not an identical
execution environment. The preparation/download phase can run on a login node;
subsequent inference can run without network access.

The actual verified encoder smoke used 160 sequence pairs, a train-only head and
80 held-out predictions. It took about 6.3 seconds on the test machine, excluding
resource acquisition. This observation is not a throughput benchmark or evidence
for model quality.

## Use your model or existing predictions

An adapter implements `predict(inputs)`, returning `{id: finite_score}`. Each input
contains `id`, the reference/mutant sequences and allowlisted biological fields.
Neither assay outcomes nor test labels are supplied. An optional `fit(inputs,
targets)` receives only the training partition. The supplied baseline is a complete
example. Label exclusion is a caller contract, not an isolation boundary for
arbitrary Python code.

For the frozen-encoder protocol, implement `embed(inputs)` and return:

```python
{
    variant_id: {"reference": reference_vector, "mutant": mutant_vector}
}
```

Every selected train/test row needs an embedding. Vector dimensions must agree;
values must be finite. The head is owned by the evaluator. Private model training
and access requirements must be described separately; local execution does not
establish absence of training overlap.

For precomputed held-out scores, supply CSV/TSV columns `id,score`, or JSON keyed by
ID. Missing predictions need an explicit reason or `--allow-partial`. Duplicate,
unknown or nonfinite predictions are rejected. Empty scores are not negatives.

```bash
.venv/bin/rewirebench evaluate --prepared mfass-baseline-prepared \
  --predictions private-predictions.tsv --model-name 'My private model' \
  --output mfass-private-evaluation
```

The Python API follows the same path:

```python
from rewirebench import prepare, run
from rewirebench.adapters.mfass import KmerBaseline

prepared = prepare("mfass-v2", source="mfass-input",
                   output="python-prepared", limit=80)
report = run(prepared, KmerBaseline(), output="python-smoke")
```

Outputs include `report.json`, local predictions and unscored reasons. Reports
record hashes, scope, coverage, environment, timing scope and adapter provenance.
Metrics use scored held-out rows while coverage retains the original eligible
denominator. No files are submitted automatically. Smoke tests cannot be exported
as benchmark contributions.

## Evidence and verification

- [Specialist comparison evidence and future evaluation criterion](mfass-specialist-comparison.md).
- [Reviewed MFASS v2 code and correction](https://github.com/rewire-bio/rewire-benchmarks/tree/bee9133b83f3aedaf2bbb9013f1875515845607e/benchmarks/mfass).
- [DNABERT-2 fixed-head implementation](https://github.com/rewire-bio/rewire-benchmarks/blob/bee9133b83f3aedaf2bbb9013f1875515845607e/benchmarks/mfass/src/mfass/run_dnabert2.py).
- Packaged `resources/mfass/validation-2026-09-17.json` distinguishes executed smoke
  tests, archived metric replay, and untested environments.

Archived baseline and DNABERT full-precision predictions reproduce saved metrics
within `1e-12`. Their printed TSVs round predictions to ten decimal places; the
DNABERT AUROC shifts by approximately `1.98e-7` because rounding creates ties.
TSV comparisons use an explicit `1e-6` absolute tolerance. Recomputing these metrics
is not a new full model-inference reproduction.
