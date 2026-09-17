# Local and HPC environments

The Python API is independent of containers. Run your private adapter in an existing
environment or use a pinned OCI environment built with Podman. Docker is not required.
Apptainer can execute the equivalent SIF artifact on clusters that support it.

## Build an environment

From a reviewed checkout:

```sh
uv build --package rewirebench
podman build --platform linux/amd64 -f containers/Containerfile \
  --build-arg ENVIRONMENT=core -t localhost/rewirebench:core .
podman save --format oci-archive -o rewirebench-core.oci.tar localhost/rewirebench:core
apptainer build rewirebench-core.sif "oci-archive://$PWD/rewirebench-core.oci.tar"
sha256sum rewirebench-core.sif rewirebench-core.oci.tar
```

`ENVIRONMENT=esm` and `ENVIRONMENT=dnabert2` add separately locked public-model
dependencies. These are Linux x86-64 environments; other architectures are not
claimed supported by these images. GPU use is not covered by CPU validation.

Release preparation CI produces wheels, OCI archives, SIF files, checksums, and
native/container scoring comparisons. It does not publish without a separate
reviewed GitHub release. Images contain software and reference/split metadata, not
private checkpoints or upstream assay archives. Large model environments can require
several GB: check disk quotas before building or pulling them.

Source OCI digest and the actual SIF checksum are distinct identifiers. Locally
converting the same OCI image can change SIF metadata; compare a downloaded SIF
against its own published checksum.

## Rootless Podman

```sh
podman run --rm --network none --read-only --userns=keep-id \
  --tmpfs /tmp:rw,size=512m \
  -v "$PWD/prepared-mfass:/input:ro" \
  -v "$PWD/output:/output:rw" \
  localhost/rewirebench:core run --prepared /input \
  --adapter rewirebench.adapters.mfass:KmerBaseline \
  --model-name 'Corrected k-mer baseline' --output /output/run-001
```

Create the host output directory first; the individual run directory must be new.
Podman rootless storage must use a filesystem supported by the cluster configuration;
use node-local scratch for container storage when required. Your administrator's
user-namespace and scheduler policies apply.

## Apptainer and Slurm

Prepare data, authorised checkpoints and SIF files before submitting a compute job.
The adapter loaders operate offline. Request the resources measured for your workload;
these parameters are user-supplied, not published performance estimates.

```sh
sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
  --cpus-per-task="$CPUS" --mem="$MEMORY" --time="$WALLTIME" examples/run.slurm
```

Set `REWIRE_IMAGE`, `REWIRE_PREPARED`, and `REWIRE_OUTPUT_ROOT` to absolute paths
before submission. See `examples/run.slurm` for a CPU baseline job. For a private
adapter, mount its code and weights read-only and use the adapter's own environment
or a derived image. Never embed credentials or private weights in distributed images.

Apptainer is integrated with the host filesystem; it is not an isolation guarantee
for untrusted model code. Use only model code you trust. The SDK separates labels
from inference inputs but cannot prevent a user from reading public test labels.

Sources: https://docs.podman.io/en/latest/markdown/podman.1.html and
https://apptainer.org/docs/user/latest/docker_and_oci.html (reviewed 2026-09-17).
Execution status is reported in validation receipts; an example alone is not a claim
of testing on your institution's HPC cluster.
