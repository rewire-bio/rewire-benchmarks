---
name: run-benchmark
description: Plan and run a biological model benchmark on this machine with the rewirebench MCP server, and get a receipt that records the recipe, code commit, input and output digests, hardware, wall time, peak memory, metrics and cost. Use when the user wants to run a benchmark recipe, run a registered model on a protocol, see what a run would need or cost before starting it, or check a receipt.
---

# Plan and run a benchmark with a receipt

All tools here are on the MCP server `rewirebench`. It runs on the user's machine, on CPU,
with the user's rights. It is not a sandbox.

What it never does: start a GPU run, start a paid run, start a program other than
rewirebench, read a cloud API key, or send a result anywhere.

## A recipe run

A recipe pins the inputs, the commands and the values the run should produce.

1. `list_recipes`: the recipes, with scope and backends. `full_protocol` is a benchmark run.
   `synthetic_smoke` checks the plumbing and is never a benchmark result.
2. `describe_recipe`: inputs with their SHA-256 and size, licence, commands, expected values
   and tolerances. Before any download, tell the user the source, the size and the licence or
   terms. Never accept terms or sign in for them.
3. `plan_run` with `recipe`, `backend` and the absolute `output` folder you will use. It starts
   nothing. Read `runnable`, `hardware`, `expected_wall_seconds`, `quote` and `command`, and show
   them to the user. Keep `plan_sha256`.
4. `run` with the same `recipe`, `backend`, `output` and the `plan_sha256`. Always pass the
   `plan_sha256`: a run must start from a plan the user has seen. It returns a `run_id` at
   once. `output` must be a new folder. One run at a time. Pass no model-run argument
   (`timeout_seconds`, `batch_size`, `allow_partial`, `model_name`, `training_overlap`): a
   recipe run uses the recipe's own timeouts and does not read them.
5. `run_status` until `state` is `finished`. Exit code 0 and `checks_passed: true` mean every
   command ran and every expected value was inside its tolerance. Exit code 1 with a receipt
   means a check failed: report it, do not retry until it passes.
6. `get_receipt`: the record of the run.
7. `verify_receipt`: recomputes every digest and reads the checked values again from the
   output files. It does not score the predictions again. Pass `strict: true` to also fail
   when a check could not be made.

`cancel_run` stops a run. A stopped run has no receipt.

If the run came from a knowledge graph query, pass the graph's release ID as `kg_release_id`
to `plan_run` and `run`, so the receipt records it. It is recorded as given and not checked
against the graph, so copy it from `dataset_info`.

## A model run outside a recipe

1. `list_protocols`, `describe_protocol`: the protocol's capabilities, its baselines and the
   models it accepts. For five of the eight protocols (both MFASS v2 protocols, ProteinGym,
   TDC ADMET and Genomic Benchmarks) that is all it returns. Do not state a protocol's inputs,
   sizes or licence from it, or from memory: they are in `describe_recipe` when a recipe
   exists, and otherwise say that the server does not give them.
2. `prepare`: from files already on the machine, into a new folder. Nothing is downloaded.
3. `list_models`, then `run` with `prepared` and `model` (an ID from the list, never a Python
   path). `device` must be `cpu`. Or score predictions made elsewhere with `evaluate`.
4. `run_status`, `get_receipt`, `verify_receipt` as above. This receipt has no recipe checks,
   so only its digests are verified and `strict` never passes.

## Backends

| Backend | From these tools |
| --- | --- |
| `local` | Runs, on CPU. A recipe that needs a GPU is refused |
| `spark` | Plan only. The reply lists the SSH and Docker steps. A person runs it at a terminal |
| `runpod` | Plan and price only. `run` refuses in every case, because no launch is built |

For a paid plan, show the user `quote.max_amount`, the currency, the rate's source and date,
and `limit_enforced`. The quote is the dated rate written in the recipe, not a live price.
Never copy an amount from the plan into `confirmed_amount`. Never send a plan's `pod_specs` to
RunPod with another tool: nothing in it stops the pod, so the quote would not bound the cost.

## What a receipt shows, and what it does not

- It shows that the run folder is intact and consistent with the recipe. Nothing is signed:
  hardware, times and `started_by` are as trustworthy as the place the folder came from.
- A pass, `strict` included, does not show that the metrics follow from the predictions.
  The values are read from the report, not computed again. The `commits` check reads the
  commit the receipt states and does not look it up.
- `cost` is `null` with reason `owned hardware` on `local` and `spark`.
- Installed from the plugin, the server runs from a wheel and the recipes are the plugin's
  copy, so `code.commit` and `recipe.commit` are `null`. `verify_receipt` passes; `strict`
  reports `commits` as not checked. For a receipt that passes `strict`, run from a clean
  checkout of rewire-bio/rewire-benchmarks (`uv run rewirebench mcp`).
- Installed from the plugin, Python and the numeric packages are not locked. The last digits
  of a metric can then differ from a run in a locked checkout: average precision on MFASS v2
  was 0.28641674595892375 from the plugin and 0.2864167459589237 from the checkout.
- Two runs of one recipe give different output digests, because the report, the logs and
  the receipt hold times. For FLIP2, DART-Eval, mRNABench and Genomic Benchmarks, `prepare`
  also gives rows random IDs, so the predictions differ too. For MFASS v2 `predictions.json`
  was byte-identical between runs on one machine. In the runs compared so far it differed
  in the last bit of 19 of 8,324 rows between macOS arm64 and Linux x86-64, so do not compare
  digests across machines. Compare runs by the checked metric values.

## Report

Give: recipe ID and SHA-256, the plan (hardware, time, cost), the backend, each checked value
with its expected value and tolerance, wall time and peak memory, the `verify_receipt` result
with anything not checked, and the run folder. State the scope. Do not call a smoke run a
benchmark result, and do not send or submit a result unless the user asks.
