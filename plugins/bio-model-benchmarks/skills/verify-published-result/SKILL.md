---
name: verify-published-result
description: Check a published benchmark result for a biological model by rerunning it locally and comparing the run with the value in the Rewire benchmark knowledge graph. Use when the user asks whether a published number can be reproduced, wants to rerun a result from the graph, or wants a rerun compared field by field with the published value.
---

# Verify a published result

Uses two MCP servers: `rewire-benchmark-graph` (what was published) and `rewirebench` (the
rerun). The `find-model-for-use-case` and `run-benchmark` skills cover each half in detail.

## 1. Pin down the published result

1. `dataset_info`: note the release `version`.
2. `find_entity` for the protocol and the system (`"type": "rb:Protocol"`, `"type": "rb:System"`).
3. `sparql_query` for the values, coverage and status. For the MFASS v2 baseline:

   ```sparql
   SELECT ?metric ?value ?scored ?eligible ?status WHERE {
     ?result rb:resultFor id:rewire-model-baseline-kmer-position-v2 ; rb:resultOn id:rewire-mfass-v2 ;
             rb:metric ?metric ; mls:hasValue ?value ; rb:status ?status .
     OPTIONAL { ?result rb:coverage ?c . ?c rb:scoredCount ?scored ; rb:eligibleCount ?eligible }
   } ORDER BY ?metric
   ```

   On release `2026-11-20-72316db0764b` this returns AUROC 0.7779498064677238, average
   precision 0.2864167459589237 and precision at 100 of 0.61, each on 8324 of 8324.
4. `describe_resource` on a result for its source and locator.

## 2. Check that a rerun can be compared

Before running anything, check each of these and tell the user the outcome:

- A recipe exists. Call `list_recipes` on `rewirebench` and match `kg_record_id` to the protocol
  record. Today one full-protocol recipe exists, `mfass-v2-kmer-baseline`. If there is none, say
  that this result cannot be rerun here yet and stop. Do not improvise a protocol.
- Same protocol, split and denominator as the published result.
- Same system and configuration: checkpoint revision, options and inputs.
- The recipe's `scope` is `full_protocol`. A `synthetic_smoke` run verifies nothing.

## 3. State the tolerance, then run

State the tolerance before the run, with the reason. A deterministic CPU baseline should
match the published value to many decimal places. For a stochastic or GPU run, choose and
state a tolerance first. A published confidence interval is not a tolerance for your run.

Then follow `run-benchmark`: `plan_run`, `run` with the `plan_sha256` and the release ID as
`kg_release_id`, `run_status`, `get_receipt`, `verify_receipt`.

## 4. Compare

One row per metric: published value, rerun value, difference, tolerance, coverage on both
sides. Metric names differ between the two: `metric:precision-at-100` is
`precision_at_capacity`, and `metric:average-precision` is `average_precision_sklearn`.

## 5. Say what was shown

- "The rerun agrees with the published value" when every metric is inside the stated tolerance
  and coverage matches.
- A rerun with the publisher's own code and data is not an independent reproduction. Say which
  it was. The recipe's `limitations` state this.
- If a value differs, report the size of the difference. Do not round it away, rerun until it
  passes, or widen the tolerance afterwards.
- Do not change any record. A discrepancy is a lead for the user to report, with the receipt.

A recorded run of exactly this flow is in the repository at `docs/demo/`.
