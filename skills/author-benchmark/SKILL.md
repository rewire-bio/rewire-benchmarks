---
name: author-benchmark
description: Design, package, and run a reproducible CPU-first biological benchmark using the Rewire benchmark graph, genomics-mcp truth sets, and rewirebench recipes and receipts. Use when the user asks to design a missing benchmark, turn a gap proposal into a Rewire-run protocol and recipe, or register a benchmark on CPU before any GPU spend.
---

# Author a reproducible Rewire benchmark

Use `rewire-benchmark-graph` to pinpoint what evidence is missing, `genomics-mcp` to ground cohorts
and reference variants, and `rewirebench` to plan, run on `cpu`, and emit a verifiable receipt.

## 1. Anchor the gap and avoid duplicates

1. `dataset_info`: record the active release `version`.
2. `run_query` with `gaps` (`"level": "use_case"` and tier 1) plus `rank_for_use_case` to find
   use cases lacking direct, multi-study, or independent evidence.
3. `find_entity` and `search` over benchmarks and protocols, plus `run_query` with `recipe_for`, to
   confirm no comparable protocol or recipe already covers the question.
4. `list_protocols` and `list_recipes` on `rewirebench` to see which baselines and adapters exist.
   Use `describe_protocol` only for capabilities and registered baselines (it gives no inputs for
   most protocols; do not infer inputs or sizes from it).

## 2. Design a leak-free, CPU-first protocol

- State the biological decision, positive truth definition, matched benign or negative control set,
  independent group split rule (patient, family, or gene holdout), sample-size floor, and metric.
- Ground cohorts and truth sets through `genomics-mcp` (`get_truth_set`, `lookup_gene`,
  `lookup_variant`, `list_synthetic_protocols`, `describe_synthetic_protocol`). Note that when
  background cohorts come from 1000 Genomes, allele-frequency filters must use non-1000G gnomAD
  frequencies so background variants are not leaked as artificially rare.
- Prefer precomputed published prediction tracks, lookup tables, and regression/motif baselines on
  `cpu` (`local` backend) before requesting any GPU run. Never start a GPU or paid run; if GPU
  inference is needed later, report the requirements and stop cleanly.

## 3. Package protocol, recipe, and receipt links

When working inside a checkout of `rewire-benchmarks`:

- Implement the deterministic cohort builder and scorer under `benchmarks/<slug>/` with unit
  tests covering strict JSON output, split leakage checks, and bootstrap intervals.
- Add a registered `rewirebench` protocol module (or standalone runner) plus a YAML recipe at
  `recipes/<slug>/<name>.yaml` matching `recipes/recipe.schema.json`.
- Set `kind`: `benchmark_run`, `purpose`: `harness_run`, `scope`: `full_protocol`, exact pinned
  inputs (`sha256`, byte size, URL, licence), pinned container image digests, expected metrics
  with tolerances, `kg_record_id`, and `store_recipe_id` so the knowledge graph builder can join
  future receipts back to `rb:runReceipt` (`reproduced` rows on `benchmarks.rewirebio.io`).

## 4. Run, verify receipt, and prepare store records

- Run `prepare`, `plan_run` on `local`, `run` with `plan_sha256` and `kg_release_id`, then
  `get_receipt` and `verify_receipt`.
- In `rewire-benchmark-data`, build an isolated `data/omics/<batch>/` generator ingesting `source`
  records (with raw URL SHA-256 digests), `benchmark`, `dataset_subset`, `protocol`, `configuration`,
  `evaluation` (`rewire_run`), `result` rows (literal numeric tokens plus uncertainty intervals),
  and pinned `links:assessed_by:` claims so the new board links directly to its use case.
- Remember: a plugin `verify_receipt` check reads values from `report.json` without re-scoring
  predictions, and `rewirebench` never exports or submits results over the network automatically.
