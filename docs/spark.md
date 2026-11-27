# Running on a DGX Spark

A DGX Spark is an owned arm64 GPU machine. It can give independent verification runs with no
hourly price. Nothing in this repository runs on one today. This page lists what has to be
rebuilt, what to check for each model, and the comparison against an H100 that must pass
before any number measured on the Spark is published.

Status on 2026-11-21: the `spark` backend is built and tested against a fake transport
([mcp.md](mcp.md#spark)). No run has been made on a Spark. Everything below marked "not
verified" is a claim to test on the machine, not a finding.

Update on 2026-11-23: the CPU image is published to GitHub Container Registry for
`linux/amd64` and `linux/arm64`, and both recipes in `recipes/` name its digests. See
[Published images](#published-images). No recipe names the `spark` backend yet, and no GPU
image exists.

## The machine

From NVIDIA's product and documentation pages, read on 2026-11-20:

- NVIDIA GB10 system on chip: a 20-core Arm CPU and a Blackwell GPU.
- 128 GB of unified memory shared by CPU and GPU. A 64 GB version is sold through partners;
  confirm which one arrived.
- DGX OS (based on Ubuntu 24.04), CUDA 13.0, Docker with the NVIDIA container runtime. A
  forum thread reports that the NVIDIA runtime needs configuring before `--gpus all` works.
- The container platform is `linux/arm64`.

Sources are listed at the end.

## What must be rebuilt for arm64

Every image and lock in the stack targets `linux/amd64`. None of them runs on the Spark as it
stands. An amd64 image under emulation is not an option for GPU work.

| What | Where | Pinned to | Needed for the Spark |
| --- | --- | --- | --- |
| Runner image | `containers/Containerfile` | Base `python:3.11.13-slim-bookworm`, by the digest of its multi-platform index since 2026-11-23. Before that it was the amd64 manifest, which the index still resolves to | Done for CPU: `.github/workflows/image.yml` builds `core` and `sequence` for `linux/arm64` on an arm64 runner. No GPU image |
| CPU locks | `containers/core.lock.txt`, `sequence.lock.txt` | Exported from `uv.lock` with hashes | Usable as they are. On 2026-11-23 every package in both locks had the hash of a pure Python or `manylinux` aarch64 CPython 3.11 wheel on PyPI, and the arm64 build installs them with `--require-hashes` |
| Model locks | `containers/esm.lock.txt`, `dnabert2.lock.txt`, `generate_cpu_locks.py` | The wheel `torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl`, by URL and SHA-256 | New locks. That wheel cannot install on arm64, and the generator asserts its name |
| Model extras | `packages/rewirebench/pyproject.toml` | `esm`: `torch==2.2.2`, `fair-esm==2.0.0`. `dnabert2`: `torch==2.2.2`, `transformers==4.29.2`, `einops==0.6.1` | A torch built for arm64 with CUDA 13. torch 2.2.2 predates the GB10, and `uv.lock` gives it CUDA libraries on x86-64 only, so on arm64 it is a CPU build. A newer torch is a different model stack: see the concordance check |
| CI image build | `.github/workflows/sdk.yml`, `sdk-release.yml` | `podman build --platform linux/amd64`, Apptainer `amd64` package | Done for the pushed image: `.github/workflows/image.yml` builds and runs the parity check on a GitHub arm64 runner. These two workflows, the release artifacts and the Apptainer check are still amd64 only |
| HPC instructions | `docs/hpc.md` | "These are Linux x86-64 environments" | A note that the Spark uses the `spark` backend and its own image |
| Recipes | `recipes/**/*.yaml`, `environment.images` | Both recipes name the `core` image digest for both platforms since 2026-11-23, written with `scripts/set_recipe_image.py` | `backends.spark` in every recipe that should run there, and a GPU image digest for a recipe that needs the GPU |
| Nextflow `dgx` profile | `workflows/benchmark/nextflow.config` | Apptainer with `--nv`; never run | Not for the Spark. DGX OS ships Docker; use the `spark` backend |
| Evo 2 image | rewire-bio/evo2-cloud-operations | The `linux/amd64` manifest of `nvcr.io/nvidia/pytorch:25.04-py3` | An arm64 base with a matching PyTorch. Whether `flash_attn` and `transformer_engine` (FP8) build and run on the GB10 is not verified |
| genomics-mcp image | rewire-bio/genomics-mcp | `linux/amd64` only | Nothing, unless the server itself should run on the Spark. It runs beside the agent, not beside the model |

The likely GPU base is NVIDIA's NGC PyTorch container for arm64. Confirm that the tag chosen
publishes a `linux/arm64` manifest, and pin that manifest's digest.

### Order of work

1. Build the `core` and `sequence` environments for `linux/arm64` and publish the image by
   digest. Done on 2026-11-23: see [Published images](#published-images).
2. Add a `spark` block with `gpus: false` to `mfass-v2-kmer-baseline`, which names the digest, and run
   it on the Spark. A spark recipe with `gpus: false` is held to `device: cpu`. This tests SSH,
   Docker, the image and the receipt with no model stack in the way. The metrics must equal
   the values in [demo/](demo/README.md).
3. Build the GPU image on the arm64 base, with a torch that sees the GPU.
4. Work through the per-model checklist below, one model at a time.
5. Run the concordance check for each model before publishing its numbers.

### Gaps in the `spark` backend to close before step 2

These are known from review and are not fixed:

- The image's entry point must be `rewirebench`.
- A recipe step gets an allowlisted environment (`receipts.step_environment`). It passes no
  `LD_LIBRARY_PATH`, `NVIDIA_*`, `HF_HOME` or `HF_TOKEN`. A GPU step in an NGC image is likely to
  need the first two, and gated weights need a token route. The spark path must not reuse the
  local allowlist unchanged.
- A receipt written inside the image has no code commit and no recipe commit, so it cannot pass
  `receipt verify --strict`. A published image carries its commit in the label
  `org.opencontainers.image.revision`, and nothing reads that label into a receipt yet.
- Inputs must be URLs. `--inputs` and `repo_path` inputs are refused.
- `weights_pin` is validated, and nothing fetches or verifies weights yet.
- The run folder stays on the host. If the local machine loses the SSH connection the
  container keeps running; stop it with `docker kill rewirebench-<output folder name>`.
- `hardware_floor` is enforced for host memory only.

## Published images

Decided on 2026-11-23 (issue #88): runner images go to GitHub Container Registry, public, as
`ghcr.io/rewire-bio/rewirebench`.

`.github/workflows/image.yml` builds `containers/Containerfile` for the `core` and `sequence`
environments, each for `linux/amd64` and `linux/arm64`. Each platform builds on a GitHub
runner of its own architecture (`ubuntu-24.04` and `ubuntu-24.04-arm`), with no emulation,
and each build must pass `containers/check_image.sh`: the parity check against the native
environment, and the `protocols` and `recipe` commands.

| When | What happens |
| --- | --- |
| Pull request that touches the image inputs | Both platforms are built and checked. Nothing is pushed |
| Push to main that touches the image inputs, or a manual run from main | The checked images are pushed, one manifest per environment covering both platforms. The workflow prints the digests, then pulls the `linux/amd64` image by digest and checks it again |

The image inputs are `containers/`, `packages/rewirebench/`, `uv.lock` and the workflow file.
Every later push to main that touches them publishes new digests. A recipe keeps the digest it
names until someone changes it.

Tags: `core-<commit>` and `sequence-<commit>` name the build of one commit, and `core` and
`sequence` follow main. A second run on the same commit builds again and moves the tag to new
digests, because the build is not byte-reproducible. Only a digest is stable, and a recipe
never uses a tag. The run summary and the `image-digests` artifact give three digests
for each environment: the manifest for both platforms, and one for each platform. A recipe
records the platform digests.

Pull by digest:

```sh
docker pull ghcr.io/rewire-bio/rewirebench@sha256:<digest>
docker buildx imagetools inspect ghcr.io/rewire-bio/rewirebench@sha256:<digest>
```

The second command shows the platform of a digest, or both platforms for a manifest digest.

### What is published

The first run on main, on 2026-11-23, built commit `9bf14fd` and pushed these. The package is
public: a pull needs no token.

| Environment | Platform | Reference |
| --- | --- | --- |
| `core` | both | `ghcr.io/rewire-bio/rewirebench@sha256:7add14344f889982b037e57a80e8cc58cc15cb603243a0926993f75e0a0bf7bd` |
| `core` | `linux/amd64` | `ghcr.io/rewire-bio/rewirebench@sha256:beb2537c26950553f0df1e61b161507d484511988848f7973b0190285fd4f55c` |
| `core` | `linux/arm64` | `ghcr.io/rewire-bio/rewirebench@sha256:0d9636331a18f40aadb35bf8f118b1811fd45491aafbeef6fa0db4ba9ade998c` |
| `sequence` | both | `ghcr.io/rewire-bio/rewirebench@sha256:0ddfd70dc2aa7858c96f6320ce102bfe8438fe80b9d40a0cc7a759c111da62b9` |
| `sequence` | `linux/amd64` | `ghcr.io/rewire-bio/rewirebench@sha256:670a375e966e52a2ac64ef8531e6c7b39bc2cc7336be42c449d1bc4057eda2a1` |
| `sequence` | `linux/arm64` | `ghcr.io/rewire-bio/rewirebench@sha256:51c32d89faa0bbcae532f41977b8096e22334c37242fbbb211a456b9b4a6d0d8` |

Both recipes, `mfass-v2-kmer-baseline` and `flip2-synthetic-smoke`, name the two `core`
platform digests. The workflow checked the pushed `linux/amd64` images by digest. Nobody has
pulled or run the `linux/arm64` images since the push; they passed the same check on the arm64
runner before it.

### Changing the digest a recipe names

A later run of the workflow on main publishes new digests. A recipe keeps the digest it names
until someone changes it:

1. Download `image-digests.json` from the run (the `image-digests` artifact) and write the
   digests into each recipe:

   ```sh
   gh run download <run id> --name image-digests
   uv run python scripts/set_recipe_image.py mfass-v2-kmer-baseline --digests image-digests.json --environment core
   uv run python scripts/set_recipe_image.py flip2-synthetic-smoke --digests image-digests.json --environment core
   ```

   The script changes only `environment.images`, in `recipes/` and in the plugin's copy. To
   set one platform, give `--platform` and `--reference`.
2. A recipe's SHA-256 changes with it. The recorded demo in `docs/demo/` quotes the SHA-256
   of `mfass-v2-kmer-baseline` and is tied to the plugin's rewirebench pin, so the same pull
   request must record the demo again at a commit that holds the new recipe, and raise the
   pin to that commit. Merge it with a merge commit, so that commit stays in the history of
   main.

A package made by a first push is private. The owner makes it public once: the rewire-bio
organisation, Packages, `rewirebench`, Package settings, Change visibility, Public.

### Not covered

- No GPU image. The repository has no GPU Containerfile. The published images are CPU
  only, so on the Spark they serve a recipe with `gpus: false`.
- The `esm` and `dnabert2` environments are not published. Their locks pin an x86-64 torch
  wheel; see the model locks row above.
- A run inside the image still cannot pass `receipt verify --strict`; see the gaps above.
- The published image is built with Docker. `sdk.yml` builds the same Containerfile with
  Podman for `linux/amd64` and converts it for Apptainer. No Apptainer check runs on arm64.

## Per-model readiness checklist

A model is ready on the Spark when every row below is done and recorded. Today no model has
started. Copy the list into the issue that tracks each model.

1. **Image.** The recipe names a `linux/arm64` digest, and `docker buildx imagetools inspect`
   shows an arm64 manifest for it.
2. **Dependencies.** Every package has an aarch64 wheel or is built from source in the image.
   List the ones built from source, with versions.
3. **GPU visible.** Inside the image, torch reports CUDA available and names the device. Record
   the torch version, the CUDA version and the driver version.
4. **Weights.** The revision and SHA-256 in `weights_pin` match what was loaded.
5. **Smoke on CPU.** The model's smoke recipe passes in the arm64 image with `device: cpu`.
6. **Smoke on GPU.** The same recipe passes with the GPU.
7. **No silent fallback.** The precision mode (fp32, bf16, FP8) and the attention path are the
   ones the recipe states. A missing `flash_attn` or an unsupported FP8 path must fail, not
   fall back.
8. **CPU against GPU on the Spark.** Same inputs, same weights: scores agree inside a tolerance
   stated before the run.
9. **Repeatability.** Two GPU runs on the Spark agree inside that tolerance.
10. **Memory and time.** Peak memory and wall time are measured on the Spark. Memory is shared
    between CPU and GPU there, so record how it was measured. Do not estimate from H100 figures.
11. **Receipt.** The receipt lists the GPU under `hardware.gpus`, has `image_used` equal to the
    recipe's arm64 digest, and passes `receipt verify`.
12. **Concordance.** The check in the next section has passed for this model and protocol.

Where each registered model stands (`rewirebench models`):

| Model | Extra | What stands in the way on arm64 |
| --- | --- | --- |
| `train-mean`, `seeded-random`, `sequence-composition`, `training-prior`, `string-ngram-linear`, `paired-composition`, `mfass-kmer-position`, `clingen-score-table` | none | Nothing known. NumPy and scikit-learn only. The MFASS baseline has run on macOS arm64, not on Linux arm64 |
| `evcouplings-independent` | none | Nothing known. CPU only. Not verified on Linux arm64 |
| `esm2-t6-8m-masked-marginal`, `esm2-embeddings` | `esm` | `torch==2.2.2` is CPU only on arm64. A GPU run needs a newer torch, and `fair-esm==2.0.0` is not verified against one |
| `dnabert2-frozen` | `dnabert2` | The same torch pin, with `transformers==4.29.2`. DNABERT-2 loads its own model code; not verified on a newer torch or on arm64 |
| `mistral-dna-138m-frozen`, `dnabert-s-frozen`, `plantrna-fm-embeddings`, `protbert-embeddings`, `esm1b-masked-marginal`, `protgpt2-log-likelihood` | one extra each, named after the model | The same torch pin. DNABERT-S needs `transformers==4.27.0` and loads its own model code; the others need `transformers==4.41.1`. None has run anywhere yet |
| `fimo-h12core-hit-count` | none | Starts the FIMO executable. Needs an arm64 build of the MEME Suite. It cannot be used in a recipe in any case |
| Evo 2 (not in rewirebench) | | See the Evo 2 image row above |

## Spark against H100: the concordance check

A result from an arm64 GB10 and one from an x86-64 H100 will not be bit-identical. The torch
build, the CUDA version and the kernels differ. Before a number measured on the Spark is
published as a verification result, show that the Spark gives the same answer as a reference
machine for that model and protocol.

No part of this has run. The H100 leg is a paid run on RunPod. No launch is built, and a paid
run needs a priced proposal that a person approves. Until both legs exist for a model, its
Spark numbers are not published.

### Design

For each model and protocol pair:

1. **Same everything but the machine.** The same recipe commands, inputs by digest and
   weights by revision and SHA-256. The image differs by platform; record both digests and
   both torch versions. If the torch version differs between the two images, say so in the
   result: the comparison then covers the software as well as the hardware.
2. **Same precision.** If the H100 run used bf16 or FP8 and the Spark cannot, that is a
   different configuration. Publish it as one. It is not a concordance pass or failure.
3. **Same rows.** For FLIP2, DART-Eval, mRNABench and Genomic Benchmarks, `prepare` gives rows
   random IDs, so two separate runs cannot be joined row by row. Prepare once and give both
   machines the same prepared folder. This needs a recipe step or option that does not exist
   yet. MFASS v2, ProteinGym, TDC ADMET and ClinGen PP3/BP4 keep their rows' own IDs, so two runs of those can
   be joined as they are. One case measured on CPU, on 2026-11-21: the MFASS v2 k-mer
   baseline on macOS arm64 and on Linux x86-64, with the same locked packages, differed in
   19 of 8,324 predictions, by at most 1.4e-17, and in no metric.
4. **Repeat on each machine first.** Two runs on the H100 and two on the Spark give the
   run-to-run difference on each. A difference between machines means little unless it is
   compared with that.

### What to compare

| Level | Measure | Passes when |
| --- | --- | --- |
| Coverage | Scored and unscored inputs | Identical sets on both machines |
| Per input | Largest absolute difference between scores; Pearson and Spearman correlation | Inside the tolerance written in the recipe before the runs |
| Metric | Absolute difference of each reported metric | Inside the recipe's tolerance, and small beside the metric's own interval |
| Ranking | Order of the systems on the protocol | Unchanged by swapping the H100 value for the Spark value |

The tolerances are set per recipe, before the runs, with the reason. As a starting point, the
Evo 2 study in rewire-bio/evo2-cloud-operations used an absolute tolerance of 0.01 for a
reproduced metric and 1e-3 for its gate value. A deterministic CPU recipe run on the Spark with
`gpus: false` should match its reference much more closely; the repository uses 1e-12 for
archived replays.

### Outcome

- **Pass.** Spark numbers for that model and protocol can be published, each with its receipt
  and with the hardware stated. Keep the two receipts and the comparison together.
- **Fail.** Publish nothing from the Spark for that pair. Record the difference and its size,
  and find the cause before changing any tolerance.
- **Run it again** whenever the image digest, the torch version, the driver or the precision
  mode changes.

### Cost

A Spark run has no hourly price. Its receipt records `cost.amount: null` with reason
`owned hardware`, and the wall time. Whether to also record energy or an amortised rate is not
decided. The H100 leg has a price, and its receipt records the amount and currency.

## The day it arrives

1. Confirm the memory size, and that `uname -m` prints `aarch64`.
2. Check that Docker sees the GPU: run `nvidia-smi` in a CUDA base image with `--gpus all`.
3. Set up key-based SSH from the machine that will start runs. The backend uses `BatchMode`
   and never asks for a password.
4. Print the steps without contacting the host:
   `rewirebench recipe run <id> --backend spark --spark-host user@host --output runs/x`.
   Nothing starts without `--execute`, and the MCP server never starts a spark run.
5. Start on the order of work above.

## Sources

- https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/software-requirements.html
- https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html
- https://forums.developer.nvidia.com/t/docker-nvidia-runtime-not-enabled-by-default/349220
