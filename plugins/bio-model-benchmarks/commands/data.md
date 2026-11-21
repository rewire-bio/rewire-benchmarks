---
description: Find the data or truth set a benchmark needs and say how to obtain it
argument-hint: <protocol, sample or dataset>
---

Use the `prepare-benchmark-data` skill for: $ARGUMENTS

For a rewirebench protocol, call `describe_protocol` (and `describe_recipe` if a recipe exists)
and list each input with its source, size and licence. For a truth set, call `get_truth_set` on
`genomics-mcp` and list the bundle's files with their assembly and digests. Download nothing
until the user has seen the sizes and terms and agreed.
