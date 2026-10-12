---
name: bio-model-benchmarks
description: Find published benchmark results for biological models (DNA, RNA, protein, single-cell and molecular models) in the Rewire benchmark knowledge graph, and reproduce or run a benchmark locally with rewirebench against the user's own model or configuration, then compare the run with the published result. Use when the user asks how a model scored on a benchmark, wants to reproduce a published benchmark number, or wants to benchmark their own model on MFASS, ProteinGym, FLIP2, DART-Eval, mRNABench, Genomic Benchmarks or TDC ADMET.
---

# Biological model benchmarks

Two tools work together:

- The **knowledge graph** (MCP server `rewire-benchmark-graph`, read-only) holds reviewed
  published results, protocols, coverage, intervals, sources and run recipes.
- **`rewirebench`** (Python CLI and SDK) prepares a protocol, runs baselines and models,
  and scores them. It runs locally and uploads nothing.

Reference files in this skill:

- `reference/protocols.md`: protocol IDs, data access and licences, extras, baselines, KG records.
- `reference/kg-queries.md`: tested KG queries and the vocabulary.
- `reference/adapter_template.py`: a `predict()` and `embed()` adapter to copy.

Detailed per-protocol docs: https://github.com/rewire-bio/rewire-benchmarks/tree/main/docs
(`sdk.md` first). Link to them; do not restate them to the user from memory.

## 1. Find

1. `dataset_info`: note the release `version`. Every number you quote comes from that release.
2. `find_entity` with a `type` (`rb:Protocol`, `rb:Benchmark`, `rb:System`) to get record IDs.
   Several records can share a label; check `types` and `describe_resource`.
3. `sparql_query` for results with coverage, status, interval and split. Start from the
   queries in `reference/kg-queries.md`; they were run against a real release.
4. For a `rewirebench` protocol, read its run recipes (`rb:runRecipes`) and, for a
   result, the evaluation's `rb:reproduction` and `rb:comparison` JSON.

A result is only comparable with a `rewirebench` run if it is on the same protocol. Most
results in the graph come from papers with their own protocols. Say so rather than
setting a paper's number next to a local run.

If the MCP server is not available, tell the user how to add it (see the repository
README, "Use with an agent") instead of guessing values.

## 2. Install

Use the pinned release wheel and verify it. The package is not on PyPI.

```bash
set -euo pipefail
mkdir rewirebench-wheel && cd rewirebench-wheel
curl --fail --location --remote-name https://github.com/rewire-bio/rewire-benchmarks/releases/download/v0.5.0/rewirebench-0.5.0-py3-none-any.whl
curl --fail --location --remote-name https://github.com/rewire-bio/rewire-benchmarks/releases/download/v0.5.0/SHA256SUMS
awk '$2 == "rewirebench-0.5.0-py3-none-any.whl"' SHA256SUMS | shasum -a 256 --check
cd .. && uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python 'rewirebench-wheel/rewirebench-0.5.0-py3-none-any.whl'
.venv/bin/rewirebench protocols
```

Check for a newer release first (`gh release list -R rewire-bio/rewire-benchmarks`) and
record the version used. Extras, installed as `'rewirebench-0.5.0-py3-none-any.whl[sequence]'`:
`sequence` for FLIP2, DART-Eval and mRNABench; `esm` and `dnabert2` for the public model
examples. Model stacks conflict; give each its own environment.

## 3. Run

For each protocol the steps are the same. Use a new output directory every time;
`rewirebench` refuses to overwrite earlier results.

1. **Inspect.** `rewirebench inspect ID` and `rewirebench baselines ID`.
2. **Prepare.** Get the data as the protocol doc says. Before downloading, tell the
   user the source, size, licence or terms, and whether an account is needed. Never
   accept terms or sign in for them.
   `rewirebench prepare ID --source DATA --output PREPARED [--options JSON]`.
   Check that the result says `"scope": "full"` before treating it as a benchmark run.
3. **Baselines.** `rewirebench run-baselines --prepared PREPARED --output BASELINES`.
   A floor always runs. Blocked baselines are reported, not run.
4. **The user's or the published configuration.** Write an adapter from
   `reference/adapter_template.py`, then
   `rewirebench run --prepared PREPARED --adapter module:Class --adapter-options JSON --model-name NAME --training-overlap TEXT --output RUN`.
   For an embedding model add `--prediction-type embedding`. For predictions made
   elsewhere, use `rewirebench evaluate --prepared PREPARED --predictions FILE` (or
   `--embeddings FILE`). To re-run a published configuration, take its checkpoint,
   options and inputs from the KG record and the run recipe; the recipe's
   `requirements` lists data, weights, hardware and licence.
5. **Score.** `run` and `evaluate` score as they go. Read `RUN/report.json`:
   `metrics`, `coverage` (denominator, scored, unscored), `scope`, `completion`,
   `provenance` (source, cohort and split hashes), `execution` and `environment`.

Per-protocol notes are in `reference/protocols.md`. A worked example on MFASS v2, the
cheapest real run (about 80 MB of public data, CPU, seconds):

```bash
mkdir mfass-input
for f in snv_data_clean.txt snv_func_annot.txt; do
  curl --fail --location "https://raw.githubusercontent.com/KosuriLab/MFASS/9a8e4f27106be52aeb11acad27f95f5cded663a8/processed_data/snv/$f" --output "mfass-input/$f"
done
.venv/bin/rewirebench prepare mfass-v2 --source mfass-input --output mfass-prepared
.venv/bin/rewirebench run-baselines --prepared mfass-prepared --output mfass-baselines
```

`mfass-baselines/mfass-kmer-position-v2/report.json` should give P@100 0.61,
average precision 0.2864 and AUROC 0.7779 on 8,324/8,324 variants, the published
corrected baseline.

## 4. Compare

Put the run beside the published result in one table, one row per metric:

| Field | From the run | From the KG |
| --- | --- | --- |
| Metric and direction | `metrics` key | `rb:metric`, `rb:metricDirection` |
| Value | `metrics[...]` | `mls:hasValue` (and `rb:printedValue`) |
| Interval | none unless the protocol reports one | `rb:uncertainty` bounds and level, or `rb:missingValue` reason |
| Coverage | `coverage.scored` / `coverage.denominator` | `rb:coverage` scored / eligible |
| Split identity | `provenance.split_sha256`, `protocol_version`, `scope` | `rb:comparison` (`protocol_id`, `split`, `dataset_version`) |
| Status | `review_status`, `independently_reproduced` | `rb:status` and the recipe instruction `status` |

Map metric names with care: in MFASS, `metric:precision-at-100` is
`precision_at_capacity` at capacity 100.

## 5. What counts as a reproduction

Call a run a reproduction of a published result only if all of these hold:

- Same protocol ID and version, same split (matching hash or `split` name and
  `dataset_version`) and the same original denominator. A `smoke` or `subset` scope,
  or a different denominator, is not a reproduction.
- The same system and configuration: checkpoint revision, pooling, context, precision,
  fitting rule and inputs.
- Each metric agrees within a stated tolerance. Deterministic CPU baselines should
  match to about `1e-12` (the repo uses `1e-12` for archived replays and `1e-6` for values
  printed to ten places). For a stochastic or GPU run, state the tolerance before
  comparing and give the reason. A published interval is not a tolerance for your run.
- Coverage reconciles: unscored inputs stay in the denominator and are reported.

Then separate three things in the report: the recipe's execution status
(`source_reviewed_not_executed` means Rewire reviewed the commands but did not run
them), the KG status (`reproduced` means a Rewire run produced it; `source_checked`
means it was checked against a source, not re-run), and what you ran today.

House rules (README), enforced by `rewirebench` and to be kept in the report:
a trivial baseline runs with a fair budget; the grouping rule and independent-group
count are stated; contamination is stated for every pretrained model ("unknown" is
acceptable, silence is not); coverage reconciles against the original denominator;
throughput is measured end to end beside accuracy; configuration is recorded.

## 6. Report

Give the user, in plain words:

- what was run: protocol ID, rewirebench version, KG release, scope, environment;
- the comparison table, with the baseline floor beside the model;
- differences, with their size, and whether they are within tolerance;
- coverage gaps and their reasons;
- contamination or training-overlap status, as declared;
- configuration and the timing from `execution.inference_and_fit_seconds`;
- what is not established (for example, local data hashes not verified against an
  official archive, or a smoke run).

Do not round away a difference or call a partial run complete. If a step failed, say
which and show the error.

## 7. Never without consent

`rewirebench export`, `submit`, `enqueue` and `submit-queue --send` send or prepare a
contribution to Rewire's review queue. Run none of them unless the user explicitly asks
in this conversation. Even then, run `submit --dry-run` first, show the payload, and let
the user supply the token through `REWIRE_SUBMISSION_TOKEN` themselves. Never put tokens
in commands, files or notebooks. Do not send personal or private model details to any
external service.

## 8. Large runs

For many configurations or long jobs, use the Nextflow workflow (`docs/nextflow.md`)
or containers on HPC (`docs/hpc.md`).

- Validated: native Nextflow on macOS arm64 (synthetic FLIP2 fixture), and Linux x86-64
  Podman and Apptainer parity checks in CI for the sequence environment.
- Not validated on the platform: the Podman, Apptainer, DGX, Slurm and Google Batch
  Nextflow profiles, any GPU execution, and any particular institution's cluster.
  Say so when you suggest them, and start with a small fixture on the real target.
- Cloud profiles can incur charges. Do not launch them without the user's approval.
