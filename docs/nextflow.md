# Run benchmark matrices with Nextflow

The workflow in `workflows/benchmark` runs complete protocol jobs through the
same `rewirebench` implementation as native Python. It does not combine arbitrary
shard scores, submit results, publish records or provision infrastructure.

## Validation status

On 21 September 2026, native execution was tested on macOS arm64 with Nextflow
24.10.5 and OpenJDK 21. The synthetic FLIP2 fixture exercised preparation, a local
adapter, a registered baseline, independent rescoring, smoke export blocking,
cache reuse and configuration-change invalidation. This is a workflow test, not
a benchmark result or model-quality claim.

Podman, Apptainer, DGX, Slurm and Google Batch profiles are prepared configurations.
They have not been executed on those platforms. No container parity, GPU support
or cloud cost claim follows from the native test. First deployment requires a
small fixture check on the actual target and a resource review.

## Native smoke test

Install this checkout's package, including the `sequence` dependencies, into a
Python environment. Activate it so `python` imports that exact checkout. Install
Nextflow 24.10.5 and Java 21. The workflow requires that exact Nextflow version.
Run from the repository root:

```bash
nextflow run workflows/benchmark -profile native \
  --jobs "$PWD/workflows/benchmark/examples/smoke.jobs.json" \
  --environment_id 'your-package-revision-and-dependency-lock-sha256' \
  --outdir /absolute/private/output/smoke \
  -work-dir /absolute/private/work/smoke
```

The environment identity is required and becomes part of the task cache key.
Use the actual package revision plus dependency-lock hash, not a floating name
such as `latest`. A changed native installation requires a changed identity.
Reports also record the SDK code digest and Python/platform information.

Repeat the exact command with `-resume` to reuse validated cached tasks. Keep the
Nextflow launch directory, `.nextflow` cache and work directory available. Input
files use deep/content hashing. Job configuration, prepared data, staged assets
and the orchestration script are process inputs; their changes invalidate the
relevant cache. A startup hash snapshots the checkout's SDK and MFASS source and
resource files, so scientific code edits invalidate cached jobs too. Keep all of these directories private.

## Job matrix

The manifest is a nonempty JSON array. Use a unique filesystem-safe `id` per
exact configuration. Each job selects one model adapter or one baseline:

```json
[
  {
    "id": "flip2-amylase-training-mean",
    "kind": "baseline",
    "protocol": "flip2-fitness-v1",
    "source": "/data/pinned/amylase.csv",
    "prepare_options": {"dataset": "flip2-amylase-one-to-many"},
    "baseline": "training-mean-v1"
  },
  {
    "id": "flip2-amylase-composition",
    "kind": "model",
    "protocol": "flip2-fitness-v1",
    "source": "/data/pinned/amylase.csv",
    "prepare_options": {"dataset": "flip2-amylase-one-to-many"},
    "adapter": "rewirebench.adapters.sequence:SequenceComposition",
    "adapter_options": {"alphabet": "ACDEFGHIKLMNPQRSTVWY"},
    "prediction_type": "embedding",
    "model": {
      "name": "Protein composition with protocol-owned ridge probe",
      "training_overlap": "protocol training split only",
      "configuration": {"alphabet": "ACDEFGHIKLMNPQRSTVWY"}
    },
    "resources": {"cpus": 1, "memory": "4 GB", "time": "1h"}
  }
]
```

Inspect valid baseline IDs with `rewirebench baselines PROTOCOL`. A blocked
baseline fails the workflow's result check rather than producing a scored row.
Keep one baseline per job so retries, resource requirements and results remain
individually attributable. Model families, checkpoints and different supervised
heads require distinct configurations.

`source` can be a file or an upstream-resource directory. Use absolute paths,
or GCS URIs for Cloud Batch. Preparation checks the existing protocol's source
contracts. `allow_unverified` and `limit` are for explicitly scoped development
fixtures; never add them to bypass a real source or coverage failure.

For model weights, provide an `assets` directory and use `asset:relative/path`
inside adapter options. For example, `"checkpoint": "asset:esm2.pt"`. This makes
checkpoint bytes staged and cache-visible. Never refer to unstaged absolute
checkpoint paths from adapter options. Private adapter code must already be
installed in the pinned environment. Use separate workflow invocations for
models with incompatible dependency environments.

Declare seeds in the adapter's supported options or use a fixed registered
baseline. The workflow does not silently inject a seed argument into arbitrary
adapters. Resource defaults are one CPU, 4 GB and one hour for inference; other
steps default to 2 GB and one hour. Override with a reviewed Nextflow config for
large evaluator/preparation steps. Native execution is not a resource sandbox.

## Containers and HPC

Build a shell-compatible image from a pinned Rewire package image with Podman:

```bash
podman build -f workflows/benchmark/Containerfile \
  --build-arg REWIRE_IMAGE='registry/rewire@sha256:REPLACE_WITH_REAL_DIGEST' \
  -t registry/rewire-nextflow:reviewed .
```

Push through the approved artifact-release process and use its actual resulting
**digest**, not the example placeholder or a mutable tag. The workflow refuses
container execution without a digest-pinned image. The image needs Python,
`rewirebench`, its selected optional dependencies and any private adapter.
Pre-stage permitted weights/data and set model-framework offline mode in the
image after those resources are available.

Append these flags to the native launch, replacing `-profile native`:

- Podman: `-profile podman --container_image registry/image@sha256:DIGEST`.
- Linux Apptainer: `-profile apptainer --container_image registry/image@sha256:DIGEST`.
- Standalone DGX: `-profile dgx --gpu_devices 0 --container_image registry/image@sha256:DIGEST`.
- Slurm: `-profile slurm --slurm_partition PARTITION --slurm_account ACCOUNT --container_image registry/image@sha256:DIGEST`.

DGX uses Apptainer `--nv` and requires explicitly allocated device indices; it
limits inference concurrency to one. Obtain device allocation from the machine
owner. Slurm and Cloud Batch request CPU jobs by default. To request GPUs, set
`resources.gpu_count` to a positive integer and `resources.gpu_type` to an explicit
platform-supported name. Slurm emits `--gres=gpu:TYPE:COUNT` and enables Apptainer
`--nv`; confirm that syntax/type with the cluster owner. Cloud Batch uses the
Nextflow accelerator directive and also requires
`--google_install_gpu_drivers true`. DGX checks the requested count against the
explicit device allocation. GPU requests on other profiles fail before tasks. Apptainer converts the pinned
OCI image to its cached SIF. Offline HPC launch requires populating that cache
and the Nextflow runtime cache beforehand.

Container isolation is not a guarantee that arbitrary adapters are safe. Run
only trusted adapter code. Preparation inputs are staged by Nextflow; provision
source/weight storage read-only at the host or scheduler boundary where required.
Keep writable scratch and task/output paths separate from archived artifacts.

## Google Cloud Batch

The profile is ready for an operator to configure; it is not a deployment or a
budget approval. Use an existing approved GCP project, explicit runtime identity,
region and private GCS locations:

```bash
nextflow run workflows/benchmark -profile google_batch \
  --jobs /absolute/path/reviewed-jobs.json \
  --environment_id 'package-revision-and-lock-hash' \
  --container_image 'registry/image@sha256:REPLACE_WITH_REAL_DIGEST' \
  --google_project PROJECT_ID \
  --google_region REGION \
  --google_service_account 'RUNTIME@PROJECT_ID.iam.gserviceaccount.com' \
  --google_work_dir 'gs://PRIVATE_BUCKET/rewire/work/BATCH_ID' \
  --outdir 'gs://PRIVATE_BUCKET/rewire/output/BATCH_ID' \
  --max_concurrent 2
```

The workflow requires all four cloud settings and GCS work/output paths with
nonempty directory prefixes. Its pinned Google Batch plugin uses `google.location`
for the supplied `--google_region`; `google.region` is not used. It does
not inherit the unrelated default gcloud project. Cloud Batch uses its own OCI
runtime; Podman is the image-build tool. Jobs, storage and data egress may incur
charges. Configure operator-reviewed IAM, network, machine types, quotas and
storage retention before any execution. GPU execution additionally requires the explicit job resource declaration and
driver-install flag above; availability, quota and image compatibility still
need review.

## Outputs, privacy and submissions

Each job publishes a content-addressed directory under `outdir/JOB_ID/` with:

- `evaluated/report.json`: original scientific result and execution provenance.
- Predictions and unscored reasons, which remain private.
- `evaluated/workflow-receipt.json`: job/configuration and runner hashes, plus a
  successful independent-rescoring check.
- `bundles/contribution.json`, or `NOT_SUBMITTABLE.txt` for smoke inputs.

A changed input/configuration produces a different directory. Existing published
outputs are not overwritten. The evaluation stage independently recomputes scores
from saved predictions and requires identical metrics, coverage and hashes.
The `run` SDK itself already evaluates; the additional stage checks orchestration
integrity rather than replacing protocol scoring. No metrics are averaged across
jobs. A partial result retains its original scope and coverage.

Do not upload the entire output or work directory. Review the allowlisted
contribution bundle and evidence permissions before passing it to the separate
submission coordinator. No submission credentials belong in manifests, adapter
options or Nextflow parameters. Common credential keys are rejected before task
creation and `REWIRE_SUBMISSION_TOKEN` is removed before adapter execution.
Run compute from a credential-minimised environment; arbitrary adapter code must
not receive a login session merely because submission happens afterward.

Failures terminate the workflow. Exit codes 137/143 are retried at most once;
other errors need inspection. Task logs, trace records and incomplete output
remain local/private. `-resume` continues successful tasks after correction.
This workflow does not provide public storage, email authentication or a result
publication service.

## Sources

Implementation references checked on 21 September 2026:

- [Nextflow process/cache reference](https://www.nextflow.io/docs/latest/reference/process.html)
- [Nextflow containers](https://www.nextflow.io/docs/latest/container.html)
- [Google's Nextflow and Batch guide](https://docs.cloud.google.com/batch/docs/nextflow)
- [Nextflow configuration reference](https://www.nextflow.io/docs/latest/reference/config.html)
- [Pinned 24.10.5 Google Batch configuration](https://github.com/nextflow-io/nextflow/blob/v24.10.5/docs/google.md)
- [Pinned accelerator directive](https://github.com/nextflow-io/nextflow/blob/v24.10.5/docs/reference/process.md#accelerator)

Run workflow tests with `python -m pytest workflows/benchmark/tests`. Without a
usable Java/Nextflow installation, only the real workflow test is skipped;
Python-stage safety and rescoring tests still run.
