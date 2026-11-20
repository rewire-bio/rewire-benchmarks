# Recipes and receipts

A recipe is a YAML file under [`recipes/`](../recipes) that says exactly what to run: pinned
inputs, the environment, the commands and how to tell the run worked. A receipt is the record a
run leaves behind. The design is in [adr-c-recipes-and-receipts.md](adr-c-recipes-and-receipts.md).

Run these from a checkout of this repository. Elsewhere, pass `--recipes-dir` or set
`REWIREBENCH_RECIPES_DIR`.

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

Copy an existing file to `recipes/<subject>/<id>.yaml`. The file name must equal the `id`.
`recipes/recipe.schema.json` gives editor completion.

- Every input has a SHA-256. It comes from an `https` URL or from a `repo_path` in this
  repository. A run stops before any command if an input does not match.
- `commands` are argv lists. `{inputs}` is the folder holding the inputs, `{work}` is a scratch
  folder that is deleted, and `{output}` is the run folder. A first element of `rewirebench`
  runs the same installation that is executing the recipe.
- `outputs` lists the files the run must write, relative to the run folder.
- `expected_output` holds values read from JSON outputs with a tolerance, and SHA-256 digests for
  files that are bit-reproducible.
- `weights_pin`, `environment`, `hardware_floor` and `expected_output` are each either present or
  named in `missing` with a reason (`inapplicable`, `unavailable`, `unextracted`, `unreported`).
  Do not fill a field from memory.
- Images are referenced by digest, one per platform (`linux/amd64`, `linux/arm64`).
- There is no status field. The evidence store records whether a recipe has been executed.

## Running a recipe

`rewirebench recipe run <id> --output <new folder>` runs on this machine. It never starts a GPU
or paid run. `--inputs <folder>` uses files you already have instead of fetching them; they are
still checked against the recipe. The command exits 0 when every command and check passed and 1
otherwise. A receipt is written in both cases.

The run folder holds the outputs, `logs/` with each command's output, `receipt.json` and
`SHA256SUMS`. Logs can contain local paths, so read them before sharing the folder.

## What a receipt holds

| Field | Content |
| --- | --- |
| `recipe` | ID, path and SHA-256 of the recipe file. The commit is given only when the file at that commit has the same bytes |
| `code` | Repository and commit of the checkout rewirebench ran from (null for an installed wheel), whether it had uncommitted changes, rewirebench version, `sdk_code_sha256` |
| `environment` | Image digest per platform from the recipe, the image used if any, Python and OS |
| `inputs` | Source, size and SHA-256 of each input |
| `hardware` | Architecture, CPU model and cores, memory, GPUs |
| `commands` | Each command as written in the recipe, with exit code, wall time and peak memory |
| `wall_seconds`, `peak_memory_bytes` | Totals for the run |
| `outputs` | Path, size and SHA-256 of every file written |
| `metrics`, `checks` | The values read from the outputs and whether each was within tolerance |
| `cost` | Amount and currency for a paid backend. On `local` it is null with the reason `owned hardware` |
| `receipt_kind`, `scope`, `verified_on`, `started_by`, `backend` | Who ran what, how much of the protocol, when and where |

`rewirebench run`, `evaluate` and `run-baselines` also write `receipt.json` when used directly.
That receipt has no recipe, and it records option names without their values.

## Verifying a receipt

`rewirebench receipt verify <folder>` reports five checks, each `match`, `mismatch` or
`not_available`:

- `sha256sums`: every file matches `SHA256SUMS`, and no file is missing or unlisted.
- `outputs`: every output matches the digest in the receipt.
- `recipe`: the recipe file in `recipes/` has the SHA-256 the receipt cites. If the recipe has
  changed since the run, check out the commit in `recipe.commit` and verify again.
- `inputs`: with `--inputs <folder>`, the inputs match the digests in the receipt.
- `checks`: the expected values are read again from the outputs and still pass.

It exits 1 on any `mismatch`. With `--strict` it also exits 1 when a check could not be made.
Verification shows the folder is intact and consistent with the recipe. It does not prove the
computation happened; only a rerun does that.
