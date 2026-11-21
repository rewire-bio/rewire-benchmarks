---
name: prepare-benchmark-data
description: Find and fetch the data a biological benchmark needs, including GIAB and HPRC truth sets and published synthetic-data protocols, using genomics-mcp, and prepare a rewirebench protocol from local files. Use when the user needs benchmark inputs, a truth set for a sample and assembly, a bounded slice of a public genomic file, or wants to know how a protocol's data is obtained.
---

# Prepare data, including truth sets

Uses two MCP servers: `genomics-mcp` (find and read genomic data; fetch bounded files) and
`rewirebench` (what a protocol needs; prepare it from local files).

## What a protocol needs

On `rewirebench`:

1. `list_protocols`: the protocols, with the recipes for each.
2. `describe_recipe`, if the protocol has a recipe: each input with its URL, SHA-256 and size,
   and the licence. This is the only tool that gives a protocol's inputs with sizes and terms.
   A recipe run fetches its own inputs and checks each digest, so you do not fetch them
   yourself.
3. `describe_protocol`: the protocol's capabilities, its baselines and the models it accepts.
   Three protocols (`flip2-fitness-v1`, `dart-eval-task1-zero-shot-v1`,
   `mrnabench-sample-mrl-v1`) also return datasets, metrics and data access. The other five,
   `mfass-v2` among them, return capabilities only. If neither tool gives the inputs, say so
   and point to the protocol's page under `docs/` in rewire-bio/rewire-benchmarks. Do not
   list inputs, sizes or licences from memory.
4. `prepare` with `protocol`, `source` (a file or folder already on the machine) and a new
   `output` folder. Check that the reply says `"scope": "full"` before calling the result a
   benchmark input.

Before any download, tell the user the source, the size, the licence or terms, and whether an
account is needed. Never accept terms or sign in for them.

## Truth sets

On `genomics-mcp`:

1. `get_truth_set` with a `sample`, `assembly` and `variant_type`, and a `contig` to get the
   interval. For example `{"sample": "HG002", "assembly": "GRCh38", "variant_type": "snv",
   "contig": "chr20"}` returns the GIAB v4.2.1 GRCh38 bundle: benchmark VCF and index,
   benchmark regions BED and the reference, each as a FileRef with digests.
2. `get_variants` with the VCF FileRef and a bounded interval reads records over HTTP ranges
   without downloading the file.
3. `fetch_file` with a FileRef starts a bounded download into the server's work folder and
   verifies the checksum. A file above the default limit needs `budget_bytes`: state the size
   to the user first. Poll `get_transfer_status`; stop with `cancel_transfer`.

Rules the server enforces, which you should not work around: every file in a bundle has one
assembly, nothing is lifted over, and contig names are not translated (`chr20` on GRCh38, `20`
on GRCh37). The HPRC HG002 assemblies are haplotype genomes in their own coordinates.

## Synthetic data with known truth

1. `list_synthetic_protocols`, filtered by `variant_type`, `method` or `setting`.
2. `describe_synthetic_protocol`: citations, the pinned simulator, parameters with defaults and
   published values, linked truth sets and the reserved rewirebench recipe ID.

These two tools are a catalogue. They simulate nothing and create no data. Every reserved
recipe has status `planned`: no rewirebench recipe runs a simulator yet.

## Other public data

`list_sources` shows the archives and their state. `search_datasets`, `describe_dataset`,
`list_files` and `list_samples` find studies and files in ENA, ENCODE, GEO, NCBI and EGA.
`get_reads`, `get_coverage`, `get_sequence`, `get_features` and `get_signal` read one interval
at a time. Controlled-access data needs the user's own credentials, configured by the user.

## Limits to state

- No rewirebench protocol reads a VCF, BAM or FileRef yet. Today genomics-mcp supplies data
  and truth sets for inspection and for benchmarks still to be built. The eight existing
  protocols take the tabular files their own documentation names.
- The truth-set and synthetic-protocol tools come from a pinned commit on the main branch of
  genomics-mcp. They are not in its 0.1.0 release.
- Keep downloads bounded and say where files were written. Do not send values derived from the
  user's private files to an external source unless they ask.
