# Benchmarks MCP server

`rewirebench mcp` serves rewirebench to a coding agent over the Model Context Protocol, on
stdio, on your own machine. An agent can list what there is to run, prepare data, plan a run,
run it on the local backend and read and verify the receipt, without a shell.

It runs code and can only run on your machine. There is no hosted version.

## Install and start

The server needs the `mcp` extra of rewirebench (the official MCP Python SDK). In a checkout
of this repository `uv sync` installs it with the development tools. rewirebench is not on
PyPI yet, so elsewhere install the wheel from a release as `rewirebench[mcp]`.

```sh
uv run rewirebench mcp               # speaks MCP on stdin and stdout
uv run rewirebench mcp --recipes-dir /path/to/rewire-benchmarks/recipes
```

For an MCP client, from a checkout of this repository:

```json
{
  "mcpServers": {
    "rewirebench": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rewire-benchmarks", "rewirebench", "mcp"]
    }
  }
}
```

The wheel ships no recipes. Outside a checkout, `list_protocols`, `list_models`, `prepare`,
`evaluate` and a model `run` work, and the recipe tools say that no recipes directory was
found. Pass `--recipes-dir` or set `REWIREBENCH_RECIPES_DIR` to use recipes from a checkout.

The server is not yet in the `bio-model-benchmarks` plugin's `.mcp.json`, because the plugin
pins a release and no release has this command yet.

## Tools

| Tool | What it does |
| --- | --- |
| `list_protocols` | The protocols, with the recipes and registered models for each |
| `describe_protocol` | Inputs, datasets, metrics, execution rules, baselines and models of one protocol |
| `list_recipes` | The recipes under `recipes/` with SHA-256, scope and backends |
| `describe_recipe` | One recipe in full |
| `list_models` | The registered model adapters, or those one protocol accepts |
| `prepare` | Prepare a protocol's data from files already on the machine |
| `plan_run` | What a recipe needs on a backend: hardware, expected wall time, price or "owned hardware", the exact command and a `plan_sha256`. Starts and contacts nothing |
| `run` | Start a recipe on the local backend, or a registered model on prepared data. Returns a `run_id` at once |
| `run_status` | Whether a run is going or finished, its exit code, receipt path and whether its checks passed |
| `evaluate` | Score a predictions or embeddings file against prepared data |
| `get_receipt` | `receipt.json` of a run, by `run_id` or by folder |
| `verify_receipt` | Recompute a receipt folder's digests and checks. `strict` also fails when a check could not be made |

A typical sequence: `list_recipes`, `describe_recipe`, `plan_run`, `run`, `run_status` until
`finished`, `get_receipt`, `verify_receipt`.

## What `run` will and will not do

- **local**: starts `rewirebench recipe run` (or `rewirebench run --model`) as its own process
  and returns. The command is the one `plan_run` printed. One run at a time. `output` is a new
  absolute folder. Every command the server starts has `CUDA_VISIBLE_DEVICES` empty and
  records `started_by: agent`.
- **spark**: never started from here. The reply is the dry-run plan with the SSH and Docker
  steps. A person can run `rewirebench recipe run <id> --backend spark --execute` at a terminal.
- **runpod**: refused. Without `confirmed_amount` the reply names the plan's amount. With an
  amount or currency that differs from the plan it is refused. With the right amount it is
  still refused, because no launch is built. It is also refused whenever `CI` is set. The
  server reads no cloud API key.

A model run takes a model ID from `list_models`, never a Python path, and only the options that
model declares. `device` must be `cpu`.

The server keeps the logs of the commands it starts in its own temporary folder, not in the
run folder, so a run folder holds only what its `SHA256SUMS` covers. Runs are known by
`run_id` for the life of the server; afterwards use the folder.

This is not a sandbox. The server and everything it starts run with your user's rights.

## Plans

`rewirebench recipe plan <id> --backend local|spark|runpod` prints the same plan as `plan_run`.

| Field | Content |
| --- | --- |
| `hardware` | The recipe's `hardware_floor`, or null with `hardware_note` |
| `expected_wall_seconds` | From `estimate.wall_clock_seconds`, else `hardware_floor.wall_clock_seconds`. Null when the recipe states neither; `expected_wall_seconds_basis` says which |
| `time_limit_seconds` | The sum of the commands' timeouts, or the pod's `max_hours` on runpod |
| `quote` | `paid: false` with `reason: "owned hardware"` on local and spark. On runpod: `max_amount` (hourly rate times `max_hours`, rounded up to the cent), `estimated_amount` (rate times `estimate.gpu_hours`), currency, and the rate's source and date |
| `confirmation` | Whether a person must confirm, and the amount and currency to confirm |
| `command` | The exact argv. On spark it is the `ssh` call that starts the container |
| `steps` | On spark, every command in order and where it runs |
| `pod_specs` | On runpod, one pod request body per GPU type in fallback order |
| `runnable`, `not_runnable_reason` | Whether this version can run it, and why not |
| `plan_sha256` | SHA-256 of the plan's canonical JSON. Pass it to `run` to be sure the plan has not changed |

The runpod quote uses the dated rate written in the recipe. It is not a live price, and it
leaves out storage and network charges.

## Backends

| Backend | State |
| --- | --- |
| `local` | Works. CPU only |
| `spark` | Built and tested against a fake transport. Never run on a DGX Spark. Prints its steps unless `--execute` is given |
| `runpod` | Plan and price only. Nothing can launch a pod |

### spark

`rewirebench recipe run <id> --backend spark --spark-host user@host --output runs/x` prints the
steps. With `--execute` it:

1. checks the host is arm64 (`uname -m`);
2. makes `<workdir>/<name of the output folder>` on the host, and stops if it exists;
3. copies `recipes/` there with `scp`;
4. pulls the recipe's `linux/arm64` image by digest;
5. runs `docker run --platform linux/arm64 --gpus all ... recipe run <id> --backend spark
   --in-container`, which writes the receipt on the host, so it records the host's hardware;
6. copies the run folder back and runs `receipt verify` on it here.

The host comes from `--spark-host` or `REWIREBENCH_SPARK_HOST` and is a plain SSH name or
`user@name`. The work folder comes from `--spark-workdir` or `REWIREBENCH_SPARK_WORKDIR`
(default `/var/tmp/rewirebench-runs`). SSH uses your own keys and `BatchMode`, so it never asks
for a password. The run folder stays on the host; remove it there by hand.

Inside the container, `--backend spark` is accepted only when `REWIRE_CONTAINER_DIGEST` equals
the recipe's `linux/arm64` digest and the machine is arm64. Anywhere else it is refused, so it
is not a way to run a GPU recipe on some other machine. CUDA devices stay visible to the steps
only on this path.

Known gaps: no `linux/arm64` image is published, so no recipe in `recipes/` names `spark`. A
spark recipe's inputs must be URLs (`--inputs` and `repo_path` inputs are refused). The receipt
written in the image has no code or recipe commit, so it cannot pass `receipt verify --strict`
yet. Gated weights have no way in: the step environment passes no token.

### runpod

`rewirebench.backends.runpod` ports two things from `cloud/runpod/launch.sh` in
rewire-bio/evo2-cloud-operations: the pod request body (REST v2, one body per GPU type in
fallback order, Secure Cloud, one SSH port, a `MAX_SECONDS` lifetime) and the instance record
read from a created pod, which is where that script learns the hourly rate. That script makes
no price call before it creates a pod, so the quote comes from the recipe.

To hand a planned pod to RunPod yourself, or to the RunPod MCP server, use a body from
`pod_specs`. A rewirebench image has no start script that installs the SSH key, starts sshd
and deletes the pod after `MAX_SECONDS`, so the body carries no `cmd`. Add one before any
launch, or the pod will not stop on its own.

## Models

`rewirebench models` lists the registered adapters. The interface is what the existing
protocols call: `predict(inputs)` for one score per input, `embed(inputs)` for one vector per
input, and an optional `fit(inputs, targets)` on the training split
(`rewirebench.models.ScoreModel`, `EmbeddingModel`, `FittedModel`). `rewirebench run --model
<id>` runs one on prepared data. `--adapter module:Class` still runs your own adapter from the
command line; it is not available through MCP or in a recipe.

`score-variants`, and `embed` or `predict` on your own sequences outside a protocol (#58), are
not built.
