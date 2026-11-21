---
description: Run a benchmark recipe on this machine (CPU) and return its verified receipt
argument-hint: <recipe id> [absolute output folder]
---

Use the `run-benchmark` skill. Recipe and output folder: $ARGUMENTS

1. `describe_recipe` and `plan_run` with backend `local`. Tell the user what will be downloaded
   (source, size, licence) and the expected time before starting.
2. `run` with the plan's `plan_sha256` and a new absolute output folder. If none was given, use
   a new folder under the current project and say which.
3. `run_status` until finished, then `get_receipt` and `verify_receipt`.

Report each checked value against its expected value, wall time, peak memory, cost and the
verification result. This command only uses the `local` backend. For `spark` or `runpod`, show
the plan and stop: these tools start neither. Never pass `confirmed_amount`, and never start
the run another way (a shell, or another RunPod tool).
