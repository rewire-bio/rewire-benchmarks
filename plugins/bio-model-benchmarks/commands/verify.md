---
description: Rerun a published result and compare it with the value in the benchmark graph
argument-hint: <model or baseline> on <benchmark or protocol>
---

Use the `verify-published-result` skill for: $ARGUMENTS

Read the published values, coverage and status from `rewire-benchmark-graph`. Check that a
recipe exists and that the rerun is comparable, and state the tolerance before running. Then
run it on `rewirebench` with the graph's release ID as `kg_release_id`, and compare one row per
metric. If no recipe exists for the result, say that it cannot be rerun here yet and stop.
