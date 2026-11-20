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
both, and the lead requires both on every receipt. So rewirebench never writes
`local_execution`. It writes `runner_ci_smoke` for a `synthetic_smoke` run in CI and
`runner_reproduction` otherwise. `local_execution` stays for runs made outside the runner. A
receipt whose `code.commit` is null or whose `code.dirty` is true did not come from a clean
commit of this repository, and a reviewer should not promote a recipe on it.

**Cost.** The lead's shape is used: `amount` and `currency` for paid backends, and null with a
`reason` otherwise (`"owned hardware"` for `local` and `spark`). ADR 0005's rule is kept around
it: `paid_compute: true` requires `hourly_rate`, `rate_source` and `rate_date` as well, and
`paid_compute: false` forbids every money field. The models reject a `local` or `spark` receipt
that claims a cost and a `runpod` receipt that claims none. `amount` with `currency` replaces
ADR 0005's `computed_usd`.

**`started_by`.** ADR-C required `human` for GPU and paid backends. The owner's comment of
2026-11-17 on rdf-kg-mcp#6 changed that rule: nothing scheduled starts a GPU run, and an agent
working with a person may start a paid run once that person has confirmed the quoted amount.
The receipt keeps `started_by` (`human`, `agent` or `ci`) as a record. The confirmation rule
belongs to the backend that spends money, which is not part of this step.

## Changes from the ADR-C draft

- `commands` is a list of named argv lists, not one `command`. The MFASS v2 baseline needs two
  steps. No shell is involved. The placeholders `{inputs}`, `{work}` and `{output}` are the only
  substitution.
- `outputs` lists the files a run must write. A missing one is a failed check.
- The receipt records `hardware` (CPU, GPU, memory, architecture) for every backend instead of
  the RunPod-only `instance` block. Inputs and outputs are lists with size and SHA-256.
- The receipt records each command as written in the recipe, with exit code, wall time and peak
  memory. Rendered local paths are not recorded. Standard output and error go to `logs/` in the
  run folder and are covered by `SHA256SUMS`.
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
  through `--recipes-dir` or `REWIREBENCH_RECIPES_DIR`. Shipping recipes with the package is a
  decision for the MCP server work (#57).
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
- Recipes for the other six CPU protocols. Each needs pinned inputs and an expected value that
  was measured, not assumed.
