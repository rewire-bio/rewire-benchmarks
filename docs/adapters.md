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
   `pinned.Checkpoint`. Take digests from saved files only:
   - `config.json`: the SHA-256 of the saved copy, which the spec records.
   - Weights: `lfs.oid` in a saved Hugging Face tree listing (`hf-tree.json`) at the pinned
     revision.
   - Anything else is `PENDING`. A listing gives git SHA-1 object IDs for small files, not
     SHA-256. Where a listing exists, check a saved file against it: the git object ID of
     the bytes must equal the listing's `oid`.
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

## Filling in a pending digest

Read the repository's file listing at the pinned revision
(`https://huggingface.co/api/models/<repo>/tree/<revision>`). Each LFS file's `lfs.oid` is
its SHA-256. A small file has only a git SHA-1 there, so download it at the revision and
hash it. Replace `PENDING` in the adapter module and the all-zero digest in its recipe in the
same commit. `test_pinned_adapters.py` checks that the two agree. Remove the pending sentence
from the model's `note` in `models.py` once nothing is pending.

## Tests

`packages/rewirebench/tests/test_pinned_adapters.py` runs with no weights, no torch and no
network:

- registration and option checks for each model ID;
- the extra pins the transformers version the adapter checks;
- `verify` refuses a pending digest, a missing file and a changed file, and each adapter
  verifies before it imports a framework (the import is replaced by a stub that raises);
- pooling and scoring arithmetic on a fake tokenizer and model with known outputs, built
  with `object.__new__` so `__init__` never runs;
- each recipe validates, names its model, and pins the adapter's files at the adapter's
  URLs and digests.

## The first six

All six are `local` recipes on CPU. None has been run yet, and each refuses to load until its
pending digests are filled in. Tier is the record's `depth_tier`. No Tier 1 protein model had
a run guide, so the protein adapters come from Tier 2.

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
