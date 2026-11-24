---
description: Investigate and explain a discrepancy between published studies in the benchmark graph
argument-hint: <discrepancy ID, model, or use case>
---

Use the `investigate-discrepancy` skill for: $ARGUMENTS

Start with `dataset_info` and `run_query` (`open_discrepancies`, `discrepancies_for_system`,
`discrepancy_history`, `results_for_model`, and `provenance`) on `rewire-benchmark-graph` to compare
locators, denominators, splits, and filters across both sides of the disagreement. Run CPU verification
on `local` only if a registered recipe exists, never pass `confirmed_amount`, and never mutate store
records directly.
