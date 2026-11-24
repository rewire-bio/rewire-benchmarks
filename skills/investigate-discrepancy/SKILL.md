---
name: investigate-discrepancy
description: Investigate why two studies or benchmarks disagree in the Rewire benchmark knowledge graph, inspect provenance and locators, run local CPU checks when recipes or truth sets exist, and draft structured adjudication findings. Use when the user asks why scores conflict, asks to explain an outlier or ordering reversal, or asks to adjudicate an open discrepancy.
---

# Investigate a discrepancy between studies

Uses `rewire-benchmark-graph` (read-only derived discrepancies, provenance, and results),
`genomics-mcp` (reference variants and truth sets), and `rewirebench` (CPU reruns when a recipe
or registered baseline applies).

## 1. Locate the contested results and history

1. `dataset_info`: capture the release `version` so every cited finding traces to one bundle.
2. `run_query` with `open_discrepancies` (optional `"severity"`), `discrepancies_for_system`
   (`"system": "id:..."`), or `contested_results_for_use_case` (`"use_case": "id:..."`).
3. `run_query` with `discrepancy_history` on the discrepancy IRI to see prior `discrepancy_review`
   actions, upstream issue links, and current review state ("open", "explained", "confirmed-error",
   or "false-positive").
4. `run_query` with `results_for_model` on each compared model or method to see every other
   protocol, metric, use case, and receipt for those systems across the full graph.

## 2. Pull source locators and evidence rules

- Call `run_query` with `provenance` on every result IRI named in the finding (`rb:outlierResult`,
  `rb:consensusResult`, or `rb:partyResult`) to inspect cited `source` URLs, DOIs, artifact hashes,
  and table/pointer locators.
- Run targeted `sparql_query` checks for split definitions, denominator / `rb:coverage` (`scored`
  vs `eligible`), metric qualifiers, variant consequence filters, and `rb:evidenceConcerns`
  (`applies_to` targets) on the competing evaluations.
- Check whether the two studies actually measure different regimes rather than a calculation bug:
  for example, variant-type mix (SNVs vs indels), allele-frequency leakage, gene-holdout vs random
  splits, zero-shot vs supervised probes, or metric scaling mismatches.

## 3. Run CPU checks when warranted

- Call `run_query` with `recipe_for` on the disputed protocol or benchmark and `list_recipes` on
  `rewirebench`. If a registered recipe exists, plan (`plan_run`) and execute (`run`) on `local`
  (`cpu`) with `plan_sha256` and `kg_release_id`, then `get_receipt` and `verify_receipt`.
- Use `genomics-mcp` (`get_truth_set`, `lookup_variant`) to verify whether variant identifiers,
  genome assemblies, or filtering sets explain coverage drops across studies.
- Remember guardrail limits: never launch GPU runs, never pass `confirmed_amount`, and never send
  `pod_specs` anywhere. Show any GPU plan and stop.

## 4. Report the adjudication clearly

Separate observed facts from hypotheses in a compact summary:

- Discrepancy ID, severity, rule type, current state, and release `version`.
- Side-by-side table comparing both sides (study source, locator, protocol, configuration,
  metric, value, `scored` / `eligible` coverage, split, and review `status`: `source_checked` vs
  `reproduced`).
- Concrete root cause hypothesis (e.g., metric definition, cohort overlap, filtering, domain shift,
  or transcription error) and whether `discrepancy_review` claim status should move or trigger an
  upstream report drafted for the user. Never mutate graph or store files directly from this flow.
