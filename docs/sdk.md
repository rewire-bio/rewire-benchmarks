# Evaluate a public or private biological model

The `rewirebench` 0.2 Python library runs locally. It does not send model code,
weights, inputs or predictions to Rewire. Optional `submit` sends an explicitly
exported contribution to the verified-email review queue; it never publishes a result.
Production submissions remain disabled until the existing service and email launch checks pass.

## Install

Use a wheel from the reviewed GitHub Actions artifacts or build it from the pinned
source revision. This release has not been published to PyPI. From this repository, `uv sync --locked` installs the core tools and MFASS
package. `uv build --package rewirebench` builds a standalone wheel; it contains the
protocols, reference metadata and split resources and works outside a checkout.
Optional public-model environments: `uv sync --locked --extra dnabert2 --package rewirebench`
or `uv sync --locked --extra esm --package rewirebench`. These may replace the active
environment; use separate project environments for incompatible model stacks.

## Your own model

Supply a Python object with `predict(inputs)` returning `{id: score}`. Input fields
are protocol-specific: MFASS uses validated assay-oriented sequence pairs and
permitted variant features; ProteinGym uses reference/mutated proteins and mutation
identifiers. Inputs contain stable `id` values, never held-out assay labels.

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

Sign in to the contribution page with your verified email. Its SDK access control
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

Remove `--dry-run` only after reviewing and supplying an access token. Production
currently returns a clear disabled-service message. A successful request returns a
submission ID, not a public result URL. Identical requests reuse a deterministic
idempotency key; retry the same payload/key after uncertain network outcomes.
Curators review the contribution, and publication occurs only through a dataset release.

## Protocol-specific instructions

- [MFASS](mfass.md): canonical assay pairs, baseline and DNABERT-2 frozen encoder.
- [ProteinGym](proteingym.md): v1.3 zero-shot substitutions and ESM-2.
- [TDC ADMET](tdc-admet.md): the 22-dataset ADMET benchmark group, scored with each dataset's own metric.
- [Genomic Benchmarks](genomic-benchmarks.md): nine sequence classification datasets.
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

The export also writes a `.evidence.json` sidecar containing allowlisted public assay
and public-checkpoint artifact hashes. Its checksum is included in the small submission
bundle. Keep it with the public evidence report cited in your submission; the SDK does
not upload files. Data verification is declared separately from review status: hashing
local ProteinGym files does not establish that their bytes match an official archive.
