---
description: Rerun the MFASS v2 baseline on CPU and compare it with the published value
argument-hint: [offline]
---

Run the plugin's demo. Argument: $ARGUMENTS

If the argument is `offline`, run recipe `flip2-synthetic-smoke` instead: `plan_run`, `run`,
`run_status`, `get_receipt`, `verify_receipt` on the `rewirebench` MCP server. It needs no
network and checks only that the tools work. Say plainly that it is not a benchmark result.

Otherwise follow the `verify-published-result` skill for the corrected k-mer baseline on MFASS
v2:

1. On `rewire-benchmark-graph`: `dataset_info`, then the published values of
   `id:rewire-model-baseline-kmer-position-v2` on `id:rewire-mfass-v2`.
2. Tell the user the run downloads two files (80 MB in total) from the KosuriLab/MFASS
   repository on GitHub, whose reuse terms are unreported upstream, and takes about 20 seconds
   on a laptop CPU. Continue when they agree.
3. On `rewirebench`: `plan_run` and `run` for recipe `mfass-v2-kmer-baseline` on `local`, with
   the release ID as `kg_release_id`, then `run_status`, `get_receipt` and `verify_receipt`.
4. Compare: precision at 100 should be 0.61, average precision 0.2864 and AUROC 0.7779, on
   8,324 of 8,324 variants.

A recorded run of the same calls is in the repository under `docs/demo/`.
