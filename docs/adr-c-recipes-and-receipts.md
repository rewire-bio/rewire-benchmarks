# ADR-C acceptance note: run recipes and receipts

Status: accepted with changes, 2026-11-20.
Proposal: [issue #64](https://github.com/rewire-bio/rewire-benchmarks/issues/64) (ADR-C, proposed 2026-11-17).
Related: ADR 0005 "Runnable recipes and independent runs" in rewire-benchmark-data
(`docs/adr/0005-run-recipes.html`, proposed 2026-11-17).

How to use what this note decides is in [recipes.md](recipes.md).

## Decision

1. A run recipe is a YAML file at `recipes/<subject>/<id>.yaml` in this repository. It is
   validated by pydantic models in `rewirebench.recipes`. The JSON Schema generated from those
   models is committed as `recipes/recipe.schema.json`, and a test keeps the two in step.
2. A recipe carries no status. The evidence store holds a pointer to the file (repository,
   commit, path, SHA-256 of the file bytes) and the verified status, using its existing values
   `source_reviewed_not_executed`, `smoke_tested` and `executed`. Status changes only through a
   reviewed change in the store that cites a receipt. Editing a recipe changes its SHA-256, so
   an old receipt can never be read as evidence for new bytes.
3. Every run writes `receipt.json` and a `SHA256SUMS` into its output folder.
   `rewirebench receipt verify` recomputes the digests and re-reads the checked values from the
   output files. A receipt with a failed check is still written, and cannot support a promotion.
4. Backends are `local`, `spark`, `runpod` and `nextflow`. A recipe names the ones it may run
   on. Only `local` runs in this step.

## How this reconciles with ADR 0005

ADR-C puts the executable recipe in this repository. ADR 0005 keeps `run_recipes` in the store
as the one how-to attribute and adds a `harness_run` purpose for a rewirebench job. Neither
document mentions the other. They fit together like this:

- The YAML file is what runs. The store's `harness_run` recipe for the same protocol is the
  reviewed claim about it: it carries the pointer and the status, and its receipt cites a
  receipt folder produced from the YAML file.
- The recipe field `store_recipe_id` names that store recipe, and `kg_record_id` names the
  record that owns it. `mfass-v2-kmer-baseline` points at store recipe `mfass-v2-baseline` on
  record `rewire-mfass-v2`.
- ADR 0005 binds a receipt to an instruction with `code_sha256`, the SHA-256 of the instruction
  text. For a recipe in this repository the same job is done by `recipe.sha256` in the receipt,
  the SHA-256 of the YAML file bytes. It can be recomputed with `shasum -a 256` and no code.
  When the store adds the pointer, its `verified_against` value for a `harness_run` recipe
  should be this digest.

Where the two documents name the same thing, the recipe and the receipt use the ADR 0005 name.

| Subject | ADR-C draft | ADR 0005 | Used here |
| --- | --- | --- | --- |
| Protocol | none | `protocol_id`, a store record ID | `rewirebench_protocol` for the runner's protocol name (`mfass-v2`) and `kg_record_id` for the store record (`rewire-mfass-v2`). `protocol_id` is not used, because the store already gives it another meaning. |
| What the recipe is for | `kind` | `purpose` | Both. `kind` is `model_run`, `benchmark_run` or `data_prep`. `purpose` uses the ADR 0005 values, including `harness_run` and `obtain_and_run`. |
| Container and runtime | `container.image` | `environment` | `environment`, with `images` keyed by `linux/amd64` and `linux/arm64`. Each is a reference by digest or a stated reason for its absence. |
| Model weights | `weights` | `weights_pin` | `weights_pin`: host, repository, 40-hex revision, per-file SHA-256, licence, gated flag, `resolved_on`. |
| Hardware | `hardware` | `hardware_floor` | `hardware_floor`, with `basis` of `source_stated` or `measured_by_rewire`. |
| How to tell it worked | `checks` | `expected_output` | `expected_output`: values with a tolerance, and artifacts with a SHA-256. The receipt records the evaluated result as `checks`. |
| An absent block | `to pin`, `to measure` | `requirements.missing` | `missing`, with a reason from the store's missingness vocabulary. Each of the four blocks is present or named in `missing`, never both and never neither. |
| How much was run | receipt scope | `scope` | `scope`: `synthetic_smoke`, `selected_assay` or `full_protocol`, stated in the recipe and copied to the receipt. |
| Run date | none | `verified_on` | `verified_on` in the receipt. |
| Who produced the run | `started_by` | `receipt_kind` | Both, because they answer different questions. See below. |
| Price | `estimate.usd`, `price_basis` | hourly rate, rate source, rate date | `estimate.price_basis` with `rate_source` and `rate_date`. No dated source, no money figure. |

Fields ADR-C has and ADR 0005 does not keep their ADR-C names: `inputs`, `estimate`, `backends`,
`licence` (split into code, weights and data, free text or an SPDX identifier).

### Three places where the two had to be adjusted

**`receipt_kind` and measured values.** ADR 0005 says a `local_execution` receipt may not carry
measured time or peak memory, because nothing instrumented those runs. The runner does measure
both, and every receipt must carry both. So rewirebench never writes
`local_execution`. It writes `runner_ci_smoke` for a `synthetic_smoke` run in CI and
`runner_reproduction` otherwise. `local_execution` stays for runs made outside the runner.

ADR 0005 says `runner_ci_smoke` and `runner_reproduction` "come from rewire-benchmarks at a
commit". The runner does not enforce that when it writes a receipt: a run from an installed
wheel, a dirty tree or an agent's laptop is also written as `runner_reproduction`, and a run
without a recipe has `scope: null`. The commit condition is checked when the receipt is read.

`code.commit` is the commit of the checkout rewirebench runs from. For an installed wheel it is
the commit the installer recorded when the wheel was installed from a git URL, with
`code.commit_source` set to `install_metadata` and `code.dirty` null, and it is null for any
other install. It is also null for an install whose files no longer have the digests in the
wheel's `RECORD`, and for a source tree that is not the top of its own git work tree. A receipt
with `install_metadata` and a `dirty` value, or a `commit_source` without a commit, does not
load. A remote that names no host is not recorded as a repository. `recipe.commit` is set only when the recipe file at that commit has the bytes
that ran. A reviewer should not promote a recipe on a receipt where `recipe.commit` is null, or
where `code.commit` is null or `code.dirty` is not false. `rewirebench receipt verify --strict` fails on
such a receipt (its `commits` check), and on a receipt whose run failed.

The `Recipe receipt check` workflow checks out the pull request head, not the temporary merge
commit GitHub builds for a pull request, and fails unless both commits in the receipt equal
the commit under test. It also runs on every push to main that touches the runner or a recipe,
so there is a receipt at a commit on main.

**Cost.** The receipt records `amount` and `currency` for paid backends, and null with a
`reason` otherwise (`"owned hardware"` for `local` and `spark`). ADR 0005's rule is kept around
it: `paid_compute: true` requires `hourly_rate`, `rate_source` and `rate_date` as well, and
`paid_compute: false` forbids every money field. The models reject a `local` or `spark` receipt
that claims a cost and a `runpod` receipt that claims none. `amount` with `currency` replaces
ADR 0005's `computed_usd`.

**`started_by`.** ADR-C required `human` for GPU and paid backends. The owner's comment of
2026-11-17 on rdf-kg-mcp#6 changed that rule: nothing scheduled starts a GPU run, and an agent
working with a person may start a paid run once that person has confirmed the quoted amount.
The receipt keeps `started_by` (`human`, `agent` or `ci`) as a record. The confirmation rule
belongs to the backend that spends money: `rewirebench.backends.runpod` refuses a launch
without a confirmed amount and currency equal to the plan's quote, and launches nothing yet
(see [mcp.md](mcp.md)). An argument cannot show that a person supplied it, so the launch
step has to take the confirmation from a terminal prompt or MCP elicitation.

## Changes from the ADR-C draft

- `commands` is a list of named argv lists, not one `command`. The MFASS v2 baseline needs two
  steps. No shell is involved. The placeholders `{inputs}`, `{work}` and `{output}` are the only
  substitution.
- Schema 1.0 accepts only `rewirebench` followed by `prepare`, `evaluate`, `run-baselines`,
  `prepare-baseline-artifact` or `run`. A step gets an allowlisted environment with no CUDA
  device and has a timeout. This is what makes "a local recipe run never starts a GPU or paid
  run" a property of the code. `run` is accepted only with `--model` and a model from the
  registry in `rewirebench.models`, because `--adapter` calls any installed Python callable.
  On the local backend a `run` step's `device` must be `cpu`, checked at validation and
  before the step starts, since an empty `CUDA_VISIBLE_DEVICES` does not hide Apple MPS. The
  FIMO model and the FIMO baseline start an executable named in their options and are
  refused in a recipe.
  The spark backend did not widen the schema: the container runtime is started by the backend
  on the host, and the steps inside the image are the same rewirebench commands.
- A `runpod` backend block requires `estimate.price_basis` and caps `max_hours` at 24, so the
  amount a launch must quote and have confirmed can always be computed from the recipe.
- `outputs` lists the files a run must write. A missing one is a failed check.
- The receipt records `hardware` (CPU, GPU, memory, architecture) for every backend instead of
  the RunPod-only `instance` block. Inputs and outputs are lists with size and SHA-256.
- The receipt records each command as written in the recipe, with exit code, wall time and peak
  memory. Rendered local paths are not recorded. Standard output and error go to `logs/` in the
  run folder, with the run's own folders replaced by the recipe's placeholders, and are covered
  by `SHA256SUMS`.
- No `events.jsonl` yet. It arrives with the first backend that has events to record.
- `weights[].verified` is not in the receipt yet, because no recipe downloads weights yet.

## What the receipt records

Recipe ID, path, commit and SHA-256. Code repository, commit, whether the tree was dirty, the
rewirebench version and `sdk_code_sha256`. The image digest for `linux/amd64` and `linux/arm64`
from the recipe, and the digest of the image the run used, if it ran in one. Each input with
its source, size and SHA-256. Hardware. Each command, the total wall time and the peak memory.
Each output with size and SHA-256. The metric values and the checks made on them. The cost.

`rewirebench run`, `evaluate` and `run-baselines` also write a receipt when used without a
recipe. It has `recipe: null` with a reason, and it records option names but not their values,
because those are local paths and private model names. The Nextflow workflow calls the SDK
directly and still writes its own `workflow-receipt.json`; it does not write `receipt.json`.

## Consequences

- `rewirebench` depends on pydantic and PyYAML. `uv.lock` and the four container locks include
  them.
- The wheel does not ship `recipes/`. The recipe commands find the directory in a checkout, or
  through `--recipes-dir` or `REWIREBENCH_RECIPES_DIR`. The MCP server (#57) kept it that way:
  a recipe that pins a `repo_path` input needs the checkout anyway, and a receipt names a
  recipe commit only when the file sits in a git checkout.
- No rewirebench image is published, so both recipes record both image digests as missing with
  a reason and run in the host Python environment. The `spark` and `runpod` backends refuse a
  recipe without the image for their platform.
- The 2026-10-12 receipt folder under `benchmarks/mfass/reproductions/` is unchanged. The store
  cites it by hash.

## Still open

- The store-side pointer attribute and typed triples (ADR-C consequences, ADR 0005 PRs).
- Whether ADR 0005 adopts the `receipt_kind` narrowing and the `amount` and `currency` cost
  shape above. ADR 0005 is still proposed.
- Signing receipts and exporting them as a Process Run Crate.
- Full-protocol recipes for the other seven of the eight protocols. Each needs pinned inputs
  and an expected value that was measured, not assumed.
- A byte-level check on the MFASS predictions. `predictions.json` from the same commit has
  SHA-256 `2683ef05...` on macOS arm64 and `563ae79d...` on Linux x86_64 (2026-11-20), with
  the five checked metrics equal, so the recipe pins no artifact digest.
