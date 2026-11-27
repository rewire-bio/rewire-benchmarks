# Adapters from run guides

A recipe can run a model only through `rewirebench run --model <id>`, so a model is
reproducible with the SDK and CLI only once it has a registered adapter. The data repository
(rewire-bio/rewire-benchmark-data) holds a run guide on many model records: install lines and
a minimal inference script read from the official README or model card, with their sources.
The guides were read, not executed. This page is how one becomes an adapter.

## Steps

1. Pick the record. Use `data/entities/models.jsonl` and the model page spec under
   `data/omics/model-pages-*/specs/<model>.json`. The saved sources sit beside it in
   `artifacts/<model>/`.
2. Check the licence first. Read the weights licence in the spec's `structured` rows and
   its `Weights licence` fact. Do not adapt weights that are non-commercial, gated or have no
   stated licence.
3. Check the pin. The checkpoint needs a 40-hex revision in a saved source, usually the
   model card or `config.json` fetched at `/raw/<revision>/`. Do not use a branch name.
4. Decide on remote code from `config.json`, not from the snippet. If `auto_map` is
   absent, transformers ships the class and `trust_remote_code` loads nothing. Set it to
   False. If `auto_map` names files in the checkpoint, the model cannot load without them:
   pin each Python file like the weights and say so in the adapter's docstring.
5. Pin every file the adapter loads. List each file with its SHA-256 in a
   `pinned.Checkpoint`, with names read from the repository's file listing at the pinned
   revision (`https://huggingface.co/api/models/<repo>/tree/<revision>?recursive=1`).
   Save the listing in `resources/models/hf-trees/`.
   - Weights are stored with Git LFS: the listing's `lfs.oid` is the SHA-256. Never
     download them here.
   - A small file (config, tokenizer, vocabulary, model code) has only a git SHA-1 object
     ID in the listing. Fetch it at the revision into an empty folder, check its git
     object ID against the listing, hash it, and delete it. Do not import or run it.
   - A digest that cannot be had this way stays `PENDING`, and its recipe input carries
     the all-zero digest.
6. Copy the loading and inference lines. Keep the card's tokenizer call, model class and
   pooling. Where the guide and the saved README differ, follow the README and note it.
   Write every departure (dtype, batching, pooling the source does not give) in the
   docstring, with the record ID and the saved file it came from.
7. Register it in `models.py` with a `ModelSpec`. Add an extra to
   `packages/rewirebench/pyproject.toml` with exact pins, from the README or requirements
   where they state versions. Declare a conflict in the root `pyproject.toml` when its
   transformers pin differs from another extra's. Then regenerate `uv.lock`.
8. Write a recipe that runs it on the smallest honest input. Reuse a table already in
   this repository where the protocol has one. List every checkpoint file as an input at
   `https://huggingface.co/<repo>/resolve/<revision>/<file>` and pass
   `{"checkpoint": "{inputs}"}`.

## Safety rules

- The checkpoint is a local folder. `Checkpoint.verify` refuses a `PENDING` digest, then a
  missing or changed file, before torch or transformers is imported.
- No network at run time. `offline_frameworks()` sets `HF_HUB_OFFLINE` and
  `TRANSFORMERS_OFFLINE`, and every `from_pretrained` call is `local_files_only`.
- No `trust_remote_code` unless `config.json` requires it. Then the code files are pinned.
- The adapter checks the installed transformers version against its extra.
- Inputs are checked before the model runs: unique IDs, the protocol's fields only, never a
  label, and the model's alphabet. A sequence longer than the context is refused, or for a
  scalar protocol left unscored with a reason. Nothing is cropped.
- Device is an option. A `local` recipe must pass `"device": "cpu"`.

## Pending digests

`Checkpoint.verify` refuses to load while any digest is `PENDING`, and `recipe plan` reports
a recipe with an all-zero input digest as not runnable and names the input. A recipe lists
pending inputs first, so a run that is started anyway stops after a small download. Replace
`PENDING` in the adapter module and the all-zero digest in its recipe in the same commit;
`test_pinned_adapters.py` checks that the two agree and that each pin matches the saved
listing.

## Tests

`packages/rewirebench/tests/test_pinned_adapters.py` runs with no weights, no torch and no
network:

- registration and option checks for each model ID;
- the extra pins the transformers version the adapter checks;
- `verify` refuses a pending digest, a missing file and a changed file, and each adapter
  verifies before it imports a framework (the import is replaced by a stub that raises);
- pooling and scoring arithmetic on a fake tokenizer and model with known outputs, built
  with `object.__new__` so `__init__` never runs;
- each pin's name is in the saved listing, and each LFS digest equals its `lfs.oid`;
- each recipe validates, names its model, and pins the adapter's files at the adapter's
  URLs and digests, in its inputs and in `weights_pin`.

## The first six

All six are `local` recipes on CPU, and every file they load is pinned by SHA-256.
`.github/workflows/model-recipe-receipt.yml` runs each one on a GitHub runner and verifies its
receipt with `--strict`, when these adapters, pins, recipes or the lock change. Tier is the record's `depth_tier`. No Tier 1 protein model had a run guide, so
the protein adapters come from Tier 2.

| Model ID | Data record | Tier | Interface | Protocol | Extra | Recipe |
| --- | --- | --- | --- | --- | --- | --- |
| `mistral-dna-138m-frozen` | `lit-genomics-20261011-guanine-model-mistral-dna` | 1 | `embed` | `mfass-v2-frozen-encoder` | `mistral-dna` | `mfass-v2-mistral-dna-frozen-smoke` |
| `dnabert-s-frozen` | `gfm2-20261011-buzkova2026-model-dnabert-s` | 1 | `embed` | `mfass-v2-frozen-encoder` | `dnabert-s` | `mfass-v2-dnabert-s-frozen-smoke` |
| `plantrna-fm-embeddings` | `lit-rna-20261011-rnalm-zeroshot-model-plantrna-fm` | 1 | `embed` | `mrnabench-sample-mrl-v1` | `plantrna-fm` | `mrnabench-plantrna-fm-smoke` |
| `protbert-embeddings` | `lit-protein-20261011-deepprotein-model-protbert` | 2 | `embed` | `flip2-fitness-v1` | `protbert` | `flip2-protbert-synthetic-smoke` |
| `esm1b-masked-marginal` | `uc20260930-model-esm-1b` | 2 | `predict` | `proteingym-v1.3-dms-substitutions` | `esm1b` | `proteingym-v1-3-esm1b-synthetic-smoke` |
| `protgpt2-log-likelihood` | `uc20260930-model-protgpt2` | 2 | `predict` | `proteingym-v1.3-dms-substitutions` | `protgpt2` | `proteingym-v1-3-protgpt2-synthetic-smoke` |

DNABERT-S has a twin record, `lit-genomics-20261011-geneb-model-dnabert-s`, with the same run
guide. It is the only one of the six that loads remote code.

Follow-up: the plugin under `plugins/bio-model-benchmarks` installs `rewirebench[mcp]==0.6.2`,
which has none of these models, so it does not ship their recipes yet.
`tests/test_agent_plugin.py` holds them back by model ID. After a release that includes them,
raise the plugin's pin, copy the six recipes into the plugin, empty `NOT_IN_PINNED_RELEASE` in
that test, and record `docs/demo/plugin-servers.json` again (see `docs/demo/README.md`).
