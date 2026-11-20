# Recipes and receipts

A recipe is a YAML file under [`recipes/`](../recipes) that says exactly what to run: pinned
inputs, the environment, the commands and how to tell the run worked. A receipt is the record a
run leaves behind. The design is in [adr-c-recipes-and-receipts.md](adr-c-recipes-and-receipts.md).

Run these from a checkout of this repository. The recipes directory is, in order:
`--recipes-dir`, `REWIREBENCH_RECIPES_DIR`, the `recipes/` of the checkout rewirebench runs
from, then a `recipes/` in the working directory or one of its parents (an installed wheel has
no checkout, so it uses this last one). `recipe list` prints the directory it used.

```sh
rewirebench recipe list
rewirebench recipe show mfass-v2-kmer-baseline
rewirebench recipe validate                 # every file under recipes/
rewirebench recipe validate path/to/new.yaml
rewirebench recipe schema > recipes/recipe.schema.json
rewirebench recipe run mfass-v2-kmer-baseline --output runs/mfass
rewirebench receipt verify runs/mfass
```

## Recipes in this repository

| Recipe | Scope | What it does |
| --- | --- | --- |
| `mfass-v2-kmer-baseline` | `full_protocol` | Fetches the two MFASS tables (80 MB) at a pinned commit, prepares `mfass-v2` and runs the corrected k-mer baseline. Expects P@100 0.610, average precision 0.286 and AUROC 0.778 on 8,324 variants. About 20 seconds on a laptop CPU. |
| `flip2-synthetic-smoke` | `synthetic_smoke` | Runs the training-mean baseline on a synthetic table in this repository. Offline. It checks the plumbing and is not a benchmark result. |

## Writing a recipe

Copy an existing file to `recipes/<subject>/<id>.yaml`. The file name must equal the `id`, and
the extension is `.yaml` (a `.yml` file is reported as invalid, not skipped).
`recipes/recipe.schema.json` gives editor completion. `recipe list` shows the valid recipes and
reports every other file under `invalid`, with exit code 1.

- Every input has a SHA-256, and a size in `bytes` if you state one. It comes from an `https`
  URL with no user name or password, or from a `repo_path` in this repository. A run stops
  before any command if an input does not match.
- `commands` are argv lists. The first element is `rewirebench` and the second is one of
  `prepare`, `run`, `evaluate`, `run-baselines` or `prepare-baseline-artifact`. Nothing else is
  accepted: no shell, no other program, and none of the commands that send data (`submit`,
  `submit-queue`, `enqueue`). The step runs the same installation that is executing the recipe.
  `{inputs}` is the folder holding the inputs, `{work}` is a scratch folder that is deleted,
  and `{output}` is the run folder.
- A command may set `timeout_seconds` (default 3600, at most 86400). A step still running then
  is stopped and the run fails.
- `outputs` lists the files the run must write, relative to the run folder. `receipt.json`,
  `SHA256SUMS` and `logs/` belong to the runner and cannot be listed.
- `expected_output` holds values read from JSON outputs with a tolerance, and SHA-256 digests for
  files that are bit-reproducible. Numbers are written as numbers: `"3"` and `true` are errors,
  and so are `.inf` and `.nan`. A tolerance cannot be wider than the expected value (or wider
  than 1 when the expected value is 0), because such a check passes whatever the run produced.
- `environment.python` is a bound such as `">=3.11"` or `"==3.12"`. `environment.lockfile` is a
  path in this repository with its SHA-256. `hardware_floor.host_memory_gb` is the memory the
  run needs. A run is refused on a machine or checkout that does not meet these.
- A recipe that names the `local` backend is CPU only: its `hardware_floor` cannot ask for a
  GPU. A recipe that names `runpod` must carry `estimate.price_basis` (a dated hourly rate)
  and a `max_hours` of at most 24, so there is always an amount to quote before a launch.
- `weights_pin.host` is `huggingface`, `github`, `zenodo` or `other`. The first two need a
  40-hex `revision`.
- `weights_pin`, `environment`, `hardware_floor` and `expected_output` are each either present or
  named in `missing` with a reason (`inapplicable`, `unavailable`, `unextracted`, `unreported`).
  Do not fill a field from memory.
- Images are referenced by digest, one per platform (`linux/amd64`, `linux/arm64`).
- There is no status field. The evidence store records whether a recipe has been executed.

## Running a recipe

`rewirebench recipe run <id> --output <new folder>` runs on this machine. `--inputs <folder>`
uses files you already have instead of fetching them; they are still checked against the
recipe, and the receipt records them as `supplied`. The command exits 0 when every command and
check passed and 1 otherwise. A receipt is written in both cases.

What keeps a recipe run from starting a GPU or paid run:

- A recipe can only name rewirebench's own data commands (see above), so it cannot call a
  shell, `runpodctl` or any other program.
- Each step gets a short list of environment variables (`PATH`, `HOME`, locale, temp folder,
  Python and certificate settings, the container digest variables). API keys and tokens in your
  environment are not passed on, so they cannot reach a step or its logs.
- `CUDA_VISIBLE_DEVICES` is set to empty for every step, and a recipe that asks for a GPU cannot
  name the `local` backend.
- `recipe run` accepts `--backend local` only.

This is not a sandbox. `rewirebench run --adapter module:Class` imports Python code that is
installed in the environment, and that code runs with your user's rights. Read a recipe you
did not write before you run it.

The run folder holds the outputs, `logs/` with each command's output, `receipt.json` and
`SHA256SUMS`. Logs can contain local paths and anything a command printed, so read them before
sharing the folder.

`started_by` is `ci` in CI and `agent` when Claude Code runs the command. Otherwise it is
`human`. An agent on another harness must pass `--started-by agent`.

## What a receipt holds

| Field | Content |
| --- | --- |
| `recipe` | ID, path and SHA-256 of the recipe file. The commit is given only when the file at that commit has the same bytes |
| `code` | Repository and commit of the checkout rewirebench ran from (null for an installed wheel), whether it had uncommitted changes, rewirebench version, `sdk_code_sha256` |
| `environment` | Image digest per platform from the recipe, the image used if any, Python and OS |
| `inputs` | Source, size and SHA-256 of each input, and how the run obtained it (`fetched`, `repository`, `supplied`, or `local` for a run without a recipe) |
| `hardware` | Architecture, CPU model and cores, memory, GPUs |
| `commands` | Each command as written in the recipe, with exit code, wall time and peak memory |
| `wall_seconds`, `peak_memory_bytes` | Totals for the run |
| `outputs` | Path, size and SHA-256 of every file written |
| `metrics`, `checks` | The values read from the outputs and whether each was within tolerance |
| `cost` | Amount and currency for a paid backend. On `local` it is null with the reason `owned hardware` |
| `receipt_kind`, `scope`, `verified_on`, `started_by`, `backend` | Who ran what, how much of the protocol, when and where |

`rewirebench run`, `evaluate` and `run-baselines` also write `receipt.json` when used directly
and the command succeeds. A failed direct command writes none. That receipt has no recipe, and
it records option names without their values.

## Verifying a receipt

`rewirebench receipt verify <folder>` (or the path of the `receipt.json` in it) reports eight
checks, each `match`, `mismatch` or `not_available`:

- `sha256sums`: every file matches `SHA256SUMS`, no file is missing, unlisted or listed twice.
  A folder without `SHA256SUMS` is a mismatch, because every run writes one.
- `outputs`: every output matches the digest in the receipt.
- `run`: every command exited 0 and every check passed. A failed run is a mismatch whether or
  not the recipe can be found.
- `recipe`: the recipe file in `recipes/` has the SHA-256 the receipt cites. If the file has
  changed since the run, the recipe is read from the commit in `recipe.commit` instead, when
  this checkout has that commit.
- `inputs`: the receipt's input digests and sizes equal the recipe's pins. With
  `--inputs <folder>` the files are hashed again as well.
- `checks`: the expected values are read again from the outputs and still pass.
- `metrics`: the receipt's `metrics` equal the values read from the outputs.
- `commits`: the receipt names a code commit from a clean tree and a commit that holds the
  recipe. This is `not_available`, never a mismatch, for a run from a wheel, a dirty tree or an
  uncommitted recipe.

It exits 1 on any `mismatch`. With `--strict` it also exits 1 when a check could not be made.
`--strict` passes only for a recipe run from a clean commit, which is what a reviewer needs
before a receipt supports a promotion. A run without a recipe never passes `--strict`.

Verification shows the folder is intact and consistent with the recipe. It does not prove the
computation happened; only a rerun does that. Nothing is signed: someone who edits
`receipt.json` can also rewrite `SHA256SUMS`. The fields that cannot be recomputed (hardware,
times, `started_by`, the commits) are as trustworthy as the place the folder came from. The
`Recipe receipt check` workflow uploads a folder produced on a GitHub runner at a named commit.
