---
description: Show what a benchmark recipe needs and costs on a backend, without starting anything
argument-hint: <recipe id> [local|spark|runpod]
---

Use the `run-benchmark` skill. Recipe and backend: $ARGUMENTS (backend `local` if none is given).

Call `describe_recipe` and then `plan_run` on the `rewirebench` MCP server. Show the inputs with
their sizes and licence, the hardware, the expected wall time, the quote (or "owned hardware"),
the exact command, whether it is runnable and the `plan_sha256`. Start nothing. If no recipe ID
was given, call `list_recipes` and ask which one.
