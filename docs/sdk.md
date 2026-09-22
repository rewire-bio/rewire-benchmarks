# Evaluate a public or private biological model

The `rewirebench` Python library runs locally. It does not send model code,
weights, inputs or predictions to Rewire. Optional `submit` sends an explicitly
exported contribution to the verified-email review queue; it never publishes a result.
As of 22 September 2026, production intake is enabled for verified contributors.
Notification delivery remains paused; the signed-in submission record is the
authoritative status. Intake, notification delivery and public dataset publication
are separate operations.

## Install

The latest published SDK is [v0.4.0](https://github.com/rewire-bio/rewire-benchmarks/releases/tag/v0.4.0).
Version 0.5.0 in this branch is a release candidate, not a published release.
Use an explicitly selected release wheel, a reviewed CI candidate artifact, or a
build from a pinned source revision; record which one you used. Neither version
is published to PyPI. See [0.5 release status](releases/0.5.0.md) for validation scope. From this repository, `uv sync --locked` installs the core tools and MFASS
package. `uv build --package rewirebench` builds a standalone wheel; it contains the
protocols, reference metadata and split resources and works outside a checkout.
Optional public-model environments: `uv sync --locked --extra dnabert2 --package rewirebench`
or `uv sync --locked --extra esm --package rewirebench`. These may replace the active
environment; use separate project environments for incompatible model stacks.

Install the sequence readers for FLIP2, DART-Eval and mRNABench with
`uv sync --locked --all-packages --extra sequence`, or install the released wheel
with its `[sequence]` extra in your own environment. It adds pandas, Arrow and
HDF5 readers, not neural-model weights. The core environment retains earlier
protocols and optional model dependencies remain separate.

Inspect exact datasets, targets, metrics and fitting rules before preparing:

```sh
rewirebench protocols
rewirebench inspect flip2-fitness-v1
rewirebench inspect dart-eval-task1-zero-shot-v1
rewirebench inspect mrnabench-sample-mrl-v1
```

FLIP2 evaluates one archived dataset/split at a time. mRNABench evaluates one
Sample dataset and target; different RNA chemistries are not averaged. DART's
source archive requires authenticated access; a synthetic demonstration is
packaged separately. None is a universal or whole-suite score.

## Your own model

Supply a Python object with `predict(inputs)` returning `{id: score}`. Input fields
are protocol-specific: MFASS uses validated assay-oriented sequence pairs and
permitted variant features; ProteinGym uses reference/mutated proteins and mutation
identifiers. Inputs contain stable `id` values, never held-out assay labels.
Genomic Benchmarks v2 uses opaque IDs and a random order saved in the prepared
artifact. Its IDs are stable within that artifact, not across preparations.
V1 Genomic Benchmarks artifacts must be prepared and scored again because their
IDs disclosed labels.

```python
import rewirebench
from my_private_model import load_model

class MyAdapter:
    def __init__(self):
        self.model = load_model()  # Your own environment and private local weights.

    def predict(self, inputs):
        # Implement the selected protocol's score semantics here.
        return {row["id"]: float(self.model.score_variant(row)) for row in inputs}

prepared = rewirebench.prepare(
    "mfass-v2", source="/data/mfass", output="./prepared-mfass"
)
report = rewirebench.run(
    prepared, MyAdapter(), output="./my-evaluation",
    model={"name": "My model", "training_overlap": "unreported"},
)
```

This is an adapter template: replace the private import and scoring method. MFASS
scores increase with splice disruption; ProteinGym scores increase with fitness.
Return `{"score": None, "reason": "unsupported input"}` for an unscored ID.
Omitting an ID is rejected unless `allow_partial=True`; it still remains in the denominator.
Model training history is a declaration, not independently verified by running the SDK.

A scalar MFASS adapter may implement `fit(train_inputs, train_targets)`. No test
labels are passed. ProteinGym's zero-shot track refuses fitting. A frozen MFASS
encoder implements `embed(inputs)` returning `{id: {"reference": vector, "mutant": vector}}`.
Rewire then fits the fixed scaler/head on training embeddings only. The underlying
model and an encoder-plus-head pipeline remain different evaluated configurations.

For FLIP2 and mRNABench, `embed(inputs)` returns `{id: [float, ...]}` for each
sequence. A protocol-owned regression head fits training embeddings only.
Do not return MFASS-style pairs for these single-sequence protocols. Scalar
adapters can optionally implement `fit_with_validation(train_inputs,
train_targets, validation_inputs, validation_targets)` where the protocol permits
validation; no test labels enter fitting. DART is zero-shot and refuses fitted
adapters. Its opaque independent sequence inputs hide element/control roles and
pair membership from the model adapter.

The supplied `SequenceComposition` control provides untrained features for the
protocol-owned head. `TrainMean` is a constant fitted on training labels.
These are Rewire controls, not claims of reproducing published learned baselines.

## Evaluate predictions from another environment

```sh
rewirebench evaluate --prepared ./prepared-mfass \
  --predictions ./private-scores.tsv --model-name 'My model' \
  --training-overlap 'unreported' --output ./scored-mfass
```

CSV/TSV require `id,score` (tabs for TSV); unscored rows need `reason`. JSON is a
mapping of IDs to scores or explicit unscored objects. Duplicate IDs, unknown IDs,
nonfinite numbers and missing values without reasons are rejected. Importing scores
records scoring time only; it does not invent inference runtime.

Frozen single-sequence embeddings can be evaluated separately:

```sh
rewirebench evaluate --prepared ./prepared-mrl \
  --embeddings ./private-embeddings.json --output ./scored-mrl
```

Use keyed JSON vectors or an NPZ with `ids` and `embeddings` arrays. Pickle/object
arrays, duplicate/unknown IDs, changing vector dimensions and nonfinite values
are rejected. The protocol requires all selected embeddings for fitting. Imported
embeddings record fitting/scoring time, without inventing encoder inference time.

Outputs are `report.json`, `predictions.json`, and `unscored.json`. Output directories
must be new, preventing accidental replacement of earlier results. Reports record
protocol/data hashes, coverage, model declaration, local environment and execution
scope. Containers and checksums do not establish independent scientific reproduction.

## Export, inspect, then submit

```python
bundle = rewirebench.export(report, output="./contribution.json")
payload = rewirebench.submit(
    bundle,
    title="MFASS evaluation of My model",
    summary="Local evaluation using the fixed MFASS v2 split and scoring procedure.",
    source_url="https://example.org/my-public-evaluation-report",
    metric="AUROC", value=str(report["metrics"]["auroc"]),
    source_locator="Evaluation report, results table",
    dry_run=True,
)
```

The source URL above is a placeholder. Supply an accessible evidence location for
your actual result. Review the exported JSON and dry-run payload before sending.
Model name and training-overlap text are deliberate contribution fields; do not put
private names or confidential details in those fields. The default export excludes
raw sequences, predictions, embeddings, weights, arbitrary configuration, local paths
and environment variables. Per-assay metrics may be exported for a partial ProteinGym
run; they are not a suite-wide score. Contributions exceeding 24,000 bytes of details must
be reduced to an appropriately scoped reviewed report, not silently truncated.

TDC ADMET and Genomic Benchmarks v2 exports accept only their dataset-specific
scalar metrics and reconciled counts. They explicitly declare a local evaluation,
not paper reproduction, and retain local source hashes without claiming an
independently verified upstream split. A complete run covers the local copy.

Sign in to [the contribution page](https://benchmarks.rewire.it/contribute/) with your verified email. Its SDK access control
provides a short-lived Firebase ID token only when contributions are configured.
Pass it as the `token=` argument, or set `REWIRE_SUBMISSION_TOKEN` for the CLI.
Never put tokens in notebooks, command arguments, Git or result artifacts.

```sh
rewirebench export --report ./my-evaluation --output ./contribution.json
rewirebench submit --bundle ./contribution.json \
  --title 'MFASS evaluation of My model' \
  --summary 'Local evaluation using the fixed MFASS v2 split and scoring procedure.' \
  --source-url 'https://example.org/my-public-evaluation-report' \
  --metric AUROC --value 'REPLACE_WITH_YOUR_VALUE' \
  --source-locator 'Evaluation report, results table' --dry-run
```

Remove `--dry-run` only after reviewing and supplying an access token. If intake
is closed, the service returns a clear disabled-service message; keep the bundle
and try again after intake opens. The contribution page shows current access.
A successful request returns a
submission ID, not a public result URL. Identical requests reuse a deterministic
idempotency key; retry the same payload/key after uncertain network outcomes.
Curators review the contribution, and publication occurs only through a dataset release.

For Python, prompt for the short-lived token without saving it in a notebook:

```python
from getpass import getpass
import rewirebench

# Reuse the exact inspected payload from the dry run above.
details = payload["contribution"]["details"]
receipt = rewirebench.submit(
    bundle,
    title=payload["contribution"]["title"],
    summary=payload["contribution"]["summary"],
    source_url=payload["contribution"]["source_urls"][0],
    metric=details["metric"],
    value=details["value"],
    source_locator=details["source_locator"],
    idempotency_key=payload["idempotencyKey"],
    token=getpass("Paste your library access token: "),
)
print(receipt["submission"]["id"], receipt["publication_status"])
```

Save the returned submission ID and idempotency key privately. Refresh the
contribution page while signed in with the same email to track the request or
respond to review notes. The receipt means the submission entered the private
review queue; it does not mean that a result is published or independently verified.

If authentication expires, copy a new token and retry the same payload and key.
For rate limits, wait before retrying; for a timeout, the request may already have
arrived, so retain the same key. Fix malformed bundles or missing evidence rather
than repeatedly sending them. Notification delivery is currently paused.
Use the owned submission record to confirm intake and track review; lack of email
is not evidence that submission failed.

## Protocol-specific instructions

- [MFASS](mfass.md): canonical assay pairs, baseline and DNABERT-2 frozen encoder.
- [ProteinGym](proteingym.md): v1.3 zero-shot substitutions and ESM-2.
- [TDC ADMET](tdc-admet.md): the 22-dataset ADMET benchmark group, scored with each dataset's own metric.
- [Genomic Benchmarks](genomic-benchmarks.md): nine sequence classification datasets.
- [FLIP2](flip2.md): seven datasets, 16 archived splits; fitness rank metrics.
- [DART-Eval](dart-eval.md): task 1 zero-shot paired sequence scoring.
- [mRNABench](mrnabench.md): four Sample MRL datasets with six separate targets.
- [HPC and containers](hpc.md): optional Podman and Apptainer execution.

No universal model API coerces arbitrary embeddings, structures or generated sequences
into a score. Additional biological tasks require explicit protocols and evaluators.

### Local configuration provenance

Pass optional `model={"name": "...", "training_overlap": "...", "configuration": {...},
"input_information": "assay sequence only"}` to retain your model settings and input
conditions in the local report. Configuration stays out of the contribution export.
The report includes protocol settings, a digest of installed SDK source, and
`REWIRE_CONTAINER_DIGEST` when explicitly supplied by the container launcher. An
unreported container identity is shown as unreported, never inferred from a mutable tag.
Model loading happens when the adapter is constructed; inference-and-fitting timing
starts afterwards and is labelled accordingly.

Only explicitly named provenance fields are exported. Adapter declarations may
use `checkpoint_revision`, `code_revision`, `checkpoint_sha256`,
`implementation_sha256`, `weights_sha256`, and `configuration_sha256`; exports
prefix these names with `model_`. Revisions require 40 lowercase hex characters
and SHA-256 values require 64. Arbitrary keys are excluded even when their values
look like hashes, because a key can contain a private path or project name.

The export also writes a `.evidence.json` sidecar containing allowlisted public assay
and public-checkpoint artifact hashes. Its checksum is included in the small submission
bundle. Keep it with the public evidence report cited in your submission; the SDK does
not upload files. Data verification is declared separately from review status: hashing
local ProteinGym files does not establish that their bytes match an official archive.
