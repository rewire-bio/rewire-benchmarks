---
description: Find the data or truth set a benchmark needs and say how to obtain it
argument-hint: <protocol, sample or dataset>
---

Use the `prepare-benchmark-data` skill for: $ARGUMENTS

For a rewirebench protocol, call `list_protocols` to see whether it has a recipe. If it has,
call `describe_recipe` and list each input with its source, size and licence from the reply.
`describe_protocol` gives the protocol's capabilities, baselines and models; for most
protocols, `mfass-v2` among them, it gives no inputs. Where no tool gives the inputs, say so.
Do not fill them in from memory. For a truth set, call `get_truth_set` on `genomics-mcp` and
list the bundle's files with their assembly and digests. Download nothing until the user has
seen the sizes and terms and agreed.
