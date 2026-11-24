# Benchmarks MCP server

`rewirebench mcp` serves rewirebench to a coding agent over the Model Context Protocol, on
stdio, on your own machine. An agent can list what there is to run, prepare data, plan a run,
run it on the local backend, stop it, and read and verify the receipt, without a shell.

It runs code and can only run on your machine. There is no hosted version.

## Install and start

The server needs the `mcp` extra of rewirebench (the official MCP Python SDK). In a checkout
of this repository `uv sync` installs it with the development tools. Outside a checkout, install
from PyPI as `uvx --from 'rewirebench[mcp]==0.6.0' rewirebench mcp` or install the release wheel.

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

The released wheel ships packaged copies of `recipes/` as a fallback outside a checkout.
Pass `--recipes-dir` or set `REWIREBENCH_RECIPES_DIR` to use recipes from another folder.

The `bio-model-benchmarks` plugin starts this server from its `.mcp.json` using the pinned PyPI
release (`rewirebench[mcp]==0.6.0`) with `uvx`, and passes `--recipes-dir ${CLAUDE_PLUGIN_ROOT}/recipes`,
a byte-identical copy of `recipes/` that ships inside the plugin. A server started from a PyPI
wheel has `code.commit` and `code.commit_source` set to null (whereas a wheel installed from a
git URL reads `code.commit` from `install_metadata`) and `code.dirty` null, so receipts verify
but report `commits` as not checked under `strict`. `recipe.commit` is null when the plugin
folder is a plain copy; if the folder is inside a git work tree whose HEAD holds the recipe
file, it is that commit. A recorded run through these tools is in [demo/](demo/README.md).

## Tools

| Tool | What it does |
| --- | --- |
| `list_protocols` | The protocols, with the recipes and registered models for each |
| `describe_protocol` | Capabilities, baselines, registered models and recipes of one protocol. Four protocols also give datasets, metrics and data access; the other five give capabilities only, and the reply says so in `note`. Inputs with sizes and licence are in `describe_recipe` |
| `list_recipes` | The recipes under `recipes/` with SHA-256, scope, backends and `kg_record_id`, the graph record of the protocol (null when the recipe names none) |
| `describe_recipe` | One recipe in full |
| `list_models` | The registered model adapters, or those registered for one protocol, with each option's type |
| `prepare` | Prepare a protocol's data from files already on the machine |
| `plan_run` | What a recipe needs on a backend: hardware, expected wall time, price or "owned hardware", the exact command and a `plan_sha256`. Starts and contacts nothing |
| `run` | Start a recipe on the local backend, or a registered model on prepared data. Returns a `run_id` at once |
| `run_status` | Whether a run is going or finished, its exit code, receipt path, whether its checks passed, and why it was stopped if it was |
| `cancel_run` | Stop a run that is still going, with everything it started |
| `evaluate` | Score a predictions or embeddings file against prepared data |
| `get_receipt` | `receipt.json` of a run, by `run_id` or by folder |
| `verify_receipt` | Check that a run folder is consistent with its receipt and recipe. `strict` also fails when a check could not be made; `not_checked` lists those |

A typical sequence: `list_recipes`, `describe_recipe`, `plan_run`, `run`, `run_status` until
`finished`, `get_receipt`, `verify_receipt`.

Every argument is described in the tool's schema. An argument a tool does not declare is an
error, so a misspelt `backend` or `confirmed_amount` is refused and never read as a default.
`run` also refuses an argument of the other kind of run: `timeout_seconds`, `batch_size`,
`allow_partial`, `model_name`, `training_overlap` and `model_options` on a recipe run, and
`inputs`, `kg_release_id`, `plan_sha256` and the confirmation arguments on a model run.

`kg_release_id` is recorded in the receipt as given. It must look like an identifier (letters,
digits, `.`, `_` and `-`, at most 64 characters) and is not checked against the graph.

The plugin installs this server from a pinned commit. A rule on this page that is newer than
the pin reaches an installed plugin when the pin is raised.

### What `verify_receipt` shows

It recomputes the digests in `SHA256SUMS` and the receipt. For a recipe run it also compares
the inputs with the recipe's pins and reads the checked values again from the outputs. It
does not show where a folder came from. Nothing is signed: hardware, times and `started_by`
are as trustworthy as the place the folder came from, and a metric the recipe does not check
can be changed together with its digests. A model run outside a recipe has no recipe checks,
so only its digests are verified, `not_checked` lists the rest, and `strict` never passes.

Three limits of a pass, `strict` included:

- The checked values are read again from the output files. The predictions are not scored
  again, because the prepared data is not in the run folder. A report edited inside the
  recipe's tolerance, with the receipt and digests rewritten, still passes.
- The `commits` check reads the commit the receipt states. It does not look the commit up, so
  it shows that a commit was recorded, not that it exists or holds this code.
- A recipe can pin the SHA-256 of an output under `expected_output.artifacts`. No shipped
  recipe pins its predictions, and for MFASS v2 one digest would not hold on every machine.
  Measured on 2026-11-21 with the same code and the same locked packages: two runs in CI on
  Linux x86-64 (one AMD and one Intel runner) gave one digest, and three runs on macOS arm64
  gave another. Row IDs, their order and every metric value were the same; 19 of the 8,324
  predictions differed in the last bit (by at most 1.4e-17). A check of predictions across
  machines needs a tolerance per row, which the recipe schema does not have yet.

Two runs of the same recipe do not give the same output digests, because the report, the
manifest, the logs and the receipt hold times. Whether `predictions.json` repeats depends on
the protocol. `prepare` gives every row a random opaque ID for FLIP2, DART-Eval, mRNABench and Genomic Benchmarks, so
that an adapter cannot key on it, and the IDs appear in the predictions. MFASS v2, ProteinGym, TDC ADMET and ClinGen PP3/BP4 keep their rows' own IDs: three MFASS v2 baseline runs on one machine, on Python
3.11 and 3.12, gave a byte-identical `predictions.json`. That holds on one machine only, as
the list above says. Compare two runs by the metric values the recipe checks, not by a digest.

## What `run` will and will not do

- **local**: starts `rewirebench recipe run` (or `rewirebench run --model`) as its own process
  and returns. The command is the one `plan_run` printed. A recipe run needs the plan's
  `plan_sha256`, so nothing is downloaded or started before a plan was made for the same
  arguments. One run at a time. `output` is a new absolute folder. Every command the server starts has `CUDA_VISIBLE_DEVICES` empty and
  records `started_by: agent`.
- **time limit and stopping**: a model run is stopped after `timeout_seconds` (default one
  hour, at most 24). A recipe run is stopped after the sum of its step timeouts plus an hour
  for fetching inputs. `cancel_run` stops a run at once. When the client closes, or the server
  gets SIGTERM, the server stops the run it started and removes its log folder, so nothing is
  left running. A server that is killed outright (SIGKILL) cannot do this; look for
  `python -m rewirebench.cli` in the process list. A stopped run has no receipt.
- **spark**: never started from here. The reply is the dry-run plan with the SSH and Docker
  steps. A person can run `rewirebench recipe run <id> --backend spark --execute` at a terminal.
- **runpod**: refused. It needs the plan's `plan_sha256`, then `confirmed_amount` and
  `confirmed_currency` as text. With an amount or currency that differs from the plan it is
  refused. With the right amount it is still refused, because no launch is built. It is also
  refused whenever `CI` is set. The server reads no cloud API key.

The confirmed amount is a value, not proof that a person gave it: an agent can read the figure
in the plan and pass it back. The tool text tells the agent to ask, and the refusals do not
repeat the figure, but that is guidance. Before a launch exists, the confirmation has to move
to something only a person can answer: a prompt at a terminal, or MCP elicitation, where the
client asks the user directly.

A model run takes a model ID from `list_models`, never a Python path, and only the options that
model declares, each of its declared type. `device` must be `cpu`. The data must be of a
protocol the model is registered for. A model that starts an executable named in its options
(`fimo-h12core-hit-count`, marked `runs_program`) is refused: the server starts no program
other than rewirebench. A person can still run it with `rewirebench run --model` at a terminal.

The receipt of a model run records `--model <id>` and `--batch-size`. Other option values are
local paths and private names and are recorded as `<local value>`.

The server keeps the logs of the commands it starts in its own temporary folder, not in the
run folder, so a run folder holds only what its `SHA256SUMS` covers. `run_status` gives the
log paths. The folder is removed when the server closes. Runs are known by `run_id` for the
life of the server; afterwards use the folder.

This is not a sandbox. The server and everything it starts run with your user's rights.

## Plans

`rewirebench recipe plan <id> --backend local|spark|runpod` prints the same plan as `plan_run`.

| Field | Content |
| --- | --- |
| `hardware` | The recipe's `hardware_floor`, or null with `hardware_note` |
| `expected_wall_seconds` | From `estimate.wall_clock_seconds`, else `hardware_floor.wall_clock_seconds`. Null when the recipe states neither; `expected_wall_seconds_basis` says which |
| `time_limit_seconds` | The sum of the commands' timeouts, or the pod's `max_hours` on runpod |
| `quote` | `paid: false` with `reason: "owned hardware"` on local and spark. On runpod: `max_amount` (hourly rate per GPU, times `accelerator_count`, times `max_hours`, rounded up to the cent), `estimated_amount` (rate times `estimate.gpu_hours`), currency, the rate's source and date, and `limit_enforced` |
| `confirmation` | Whether a person must confirm, and the amount and currency to confirm |
| `command` | The exact argv. On spark it is the `ssh` call that starts the container |
| `steps` | On spark, every command in order and where it runs |
| `pod_specs` | On runpod, the pod request body for the GPU type the quote prices. For review only |
| `runnable`, `not_runnable_reason` | Whether this version can run it, and why not |
| `plan_sha256` | SHA-256 of the plan's canonical JSON. Pass it to `run` to be sure the plan has not changed |

The runpod quote uses the dated rate written in the recipe. It is not a live price, and it
leaves out storage and network charges. The recipe schema requires that rate to be for
provider `runpod` and for the first entry of `gpu_types`, so the rate and the pod asked for
cannot drift apart. Later entries of `gpu_types` have no dated rate: they are not priced and
get no pod spec. `limit_enforced` is false, because nothing deletes the pod at `max_hours`
yet. Until an image has a start script that does, `max_amount` is not a ceiling.

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
   --in-container` in a container named `rewirebench-<name of the output folder>`, which
   writes the receipt on the host, so it records the host's hardware;
6. copies the run folder back and runs `receipt verify` on it here.

The host comes from `--spark-host` or `REWIREBENCH_SPARK_HOST` and is a plain SSH name or
`user@name`. The work folder comes from `--spark-workdir` or `REWIREBENCH_SPARK_WORKDIR`
(default `/var/tmp/rewirebench-runs`). SSH uses your own keys and `BatchMode`, so it never asks
for a password. The run folder stays on the host; remove it there by hand.

The plan's `scp` steps are the commands the backend runs, option for option. If the run step
passes its time limit, the backend runs `docker kill` on the container by name and says
whether that worked. If this machine loses the connection instead, the container keeps going:
stop it on the host with `docker kill rewirebench-<name of the output folder>`.

Inside the container, `--backend spark` is accepted only when `REWIRE_CONTAINER_DIGEST` equals
the recipe's `linux/arm64` digest and the machine is arm64. Anywhere else it is refused, so it
is not a way to run a GPU recipe on some other machine. CUDA devices stay visible to the steps
only on this path.

Known gaps: the published `linux/arm64` image is CPU only and no recipe in `recipes/` names
`spark`. A spark recipe's inputs must be URLs (`--inputs` and `repo_path` inputs are refused). The receipt
written in the image has no code or recipe commit, so it cannot pass `receipt verify --strict`
yet. Gated weights have no way in: the step environment passes no token.

### runpod

`rewirebench.backends.runpod` ports two things from `cloud/runpod/launch.sh` in
rewire-bio/evo2-cloud-operations: the pod request body (REST v2, one body per GPU type in
fallback order, Secure Cloud, one SSH port, a `MAX_SECONDS` lifetime) and the instance record
read from a created pod, which is where that script learns the hourly rate. That script makes
no price call before it creates a pod, so the quote comes from the recipe. `launch.sh` asks
for one GPU; here the count comes from the recipe's hardware floor and the quote multiplies
by it.

The pod spec is for review. Do not send it to RunPod by hand or with another tool, the RunPod
MCP server included. A rewirebench image has no start script that installs the SSH key, starts
sshd and deletes the pod after `MAX_SECONDS`, so the body carries no `cmd`, nothing would stop
the pod, and the quote would not bound its cost. That start script is a precondition of
building the launch.

## Models

`rewirebench models` lists the registered adapters. The interface is what the existing
protocols call: `predict(inputs)` for one score per input, `embed(inputs)` for one vector per
input, and an optional `fit(inputs, targets)` on the training split
(`rewirebench.models.ScoreModel`, `EmbeddingModel`, `FittedModel`). `rewirebench run --model
<id>` runs one on prepared data, and refuses data of a protocol the model is not registered
for. `--adapter module:Class` still runs your own adapter from the command line, on any data;
it is not available through MCP or in a recipe.

`score-variants`, and `embed` or `predict` on your own sequences outside a protocol (#58), are
not built.
