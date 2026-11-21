# Recorded demo: MFASS v2 baseline, rerun through MCP tools only

On 2026-11-21 the corrected k-mer and position baseline on MFASS v2 was rerun on a laptop CPU
through MCP tool calls only, and compared with the value in the benchmark knowledge graph.
This folder holds the record.

| File | What it is |
| --- | --- |
| [`transcript.json`](transcript.json) | Every tool call, in order, with its arguments and result |
| [`mfass-v2-kmer-baseline/`](mfass-v2-kmer-baseline) | The complete run folder: `receipt.json`, `SHA256SUMS`, the outputs and the step logs |
| [`plugin-servers.json`](plugin-servers.json) | The three servers started exactly as the plugin's `.mcp.json` declares them, with one read-only call to each |
| [`plugin-run.json`](plugin-run.json) | The same MFASS v2 rerun made through the plugin's own launch lines, from a copy of the plugin folder outside any checkout. Every tool call; the run folder is not committed |

## What ran

The client is [`scripts/demo/mcp_demo.py`](../../scripts/demo/mcp_demo.py). It starts MCP
servers on stdio and reaches them only through `call_tool`. It calls no rewirebench function
and starts no rewirebench command itself.

| # | Server | Tool | Result |
| --- | --- | --- | --- |
| 1 | `rewire-benchmark-graph` (rdf-kg-mcp v0.4.0) | `dataset_info` | Release `2026-11-20-72316db0764b` |
| 2 | | `find_entity` | `id:rewire-mfass-v2` |
| 3 | | `sparql_query` | Published values of `id:rewire-model-baseline-kmer-position-v2`, each on 8324 of 8324, status `reproduced` |
| 4 | `rewirebench` | `list_recipes` | Two recipes |
| 5 | | `describe_recipe` | `mfass-v2-kmer-baseline`: two inputs, 80 MB, pinned by SHA-256 |
| 6 | | `plan_run` | Local, CPU, 21.6 seconds expected, owned hardware, no amount to confirm |
| 7 | | `run` | Started, with the plan's `plan_sha256` and the graph release ID |
| 8 to 19 | | `run_status` | Polled every 2 seconds until finished, exit code 0, checks passed |
| 20 | | `get_receipt` | The receipt |
| 21 | | `verify_receipt` with `strict` | Passed: all eight checks match, nothing left unchecked |

## Result

| Metric | Published in the graph | This run | Difference |
| --- | --- | --- | --- |
| Precision at 100 | 0.61 | 0.61 | 0 |
| Average precision | 0.2864167459589237 | 0.2864167459589237 | 0 |
| AUROC | 0.7779498064677238 | 0.7779498064677238 | 0 |
| Scored variants | 8324 of 8324 | 8324 | 0 |

The difference of 0 is for this run, in the environment `uv.lock` pins (Python 3.11.13, numpy
1.26.4). The run through the plugin's launch lines, below, gave an average precision of
0.28641674595892375, which differs in the last digit (5.6e-17). The client's comparison
tolerance is 1e-9.

From the receipt:

- recipe `mfass-v2-kmer-baseline`, SHA-256 `7552c6a78089e0e6eab19eba58fc9108ee7f2ffb5974dd69fcc4815bf5d4f2ab`;
- code and recipe commit `e5c5662a19d4a7b9a776ae6dd52e797edfb9b6a2` on `main`, clean tree;
- inputs fetched from KosuriLab/MFASS at commit `9a8e4f27`, both digests matched;
- Apple M4, arm64, 10 logical cores, 16 GiB, no GPU; macOS, Python 3.11.13;
- wall time 22.37 seconds including the download, peak memory 807 MB;
- cost: no paid compute, reason "owned hardware";
- container image: none. No rewirebench image is published, so the run used the host Python
  environment and both platform digests in the receipt are null.

## What this does and does not show

It shows that the path works: a client with only MCP tools can read the published value,
plan the run, run it, get a receipt and verify it, and the rerun gives the published numbers.

It does not show these:

- **A model did not choose the calls.** A script made them, in the order the
  `verify-published-result` skill describes. `started_by` in the receipt is `agent` because
  the MCP server records that for every run it starts. A session in which a coding agent
  with the plugin installed makes these calls has not been recorded.
- **It is not an independent reproduction.** It reruns the project's own baseline with the
  project's own code and data.
- **genomics-mcp took no part.** The recipe fetches its own two inputs. The server is checked
  only for starting and answering one call, in `plugin-servers.json`.
- **No GPU and no paid backend.** Nothing here can start either.
- **The rewirebench server ran from a checkout, not from the plugin's install line.** That is
  what lets the receipt name a code commit and pass `strict`. The plugin path is recorded
  separately: see the next section.
- **Nothing is signed.** The receipt shows the folder is intact and consistent with the
  recipe. It does not prove where the folder came from.
- **The predictions were not scored again by `verify_receipt`.** It reads the checked values
  from `report.json`. The `commits` check reads the commit the receipt states and does not
  look it up. The recipe pins no digest for `predictions.json`, and one digest would not hold
  on every machine: the same run in CI on Linux x86-64 gives the same metrics with 19 of the
  8,324 predictions different in the last bit.

## The same run through the plugin as shipped

[`plugin-run.json`](plugin-run.json) is the transcript of the same 17 calls (3 to the graph, 14
to rewirebench) with both servers started by the launch lines in the plugin's `.mcp.json`.
The plugin folder was copied outside any checkout first, as an installed plugin is, so
`${CLAUDE_PLUGIN_ROOT}/recipes` was a plain folder. It was recorded on 2026-11-21 on the same
laptop.

| | Checkout run (above) | Plugin run |
| --- | --- | --- |
| rewirebench | `python -m rewirebench.cli mcp` from a clean checkout of `e5c5662` | `uvx` wheel built from `e5c5662` |
| Recipes | `recipes/` of the checkout | the plugin's copy, same SHA-256 |
| Python, packages | 3.11.13, locked by `uv.lock` | 3.12.7, not locked |
| `code.commit`, `recipe.commit` | `e5c5662` | null |
| `verify_receipt` | passed with `strict` | passed; `commits` not checked, so `strict` does not pass |
| Precision at 100 | 0.61 | 0.61 |
| Average precision | 0.2864167459589237 | 0.28641674595892375 |
| AUROC | 0.7779498064677238 | 0.7779498064677238 |
| `sdk_code_sha256` | `3297cfe0...` | `3297cfe0...` |
| SHA-256 of `predictions.json` | `2683ef05...` | `2683ef05...` |
| Wall time | 22.37 s | 13.82 s (inputs downloaded again) |

This is what a user of the installed plugin should expect. It is still a scripted client and
not a Claude Code session with the plugin loaded: `${CLAUDE_PLUGIN_ROOT}` was substituted by
the client. The receipt at that commit does not record package versions; the code on `main`
now does.

In the transcripts, the `version` recorded for the `rewirebench` server (1.30.0) is the version
of the MCP SDK it is built on. The rewirebench version, 0.5.0, is in the receipt.

One line of `logs/01-prepare.stdout.txt` names a temporary folder on the machine that ran it
(`/var/folders/...`, with no user name). The file is covered by `SHA256SUMS` and the receipt,
so it is left as recorded. The runner on `main` now writes `{work}` in a step log instead, and
a test holds this folder to that one line.

In the transcripts local folder names are replaced by labels such as `<run folder>`. In
`transcript.json` the temporary folder was replaced by hand after the run, because the client
then labelled only its own temporary folder. The client now replaces any such folder itself
and refuses to write a transcript that still names a local folder. `plugin-run.json` is its
output as written.

## Check it yourself

From a checkout of this repository:

```sh
uv run rewirebench receipt verify docs/demo/mfass-v2-kmer-baseline
```

It recomputes every digest in the folder and reads the checked values again from the outputs.
If the recipe file has changed since the run, the recipe check needs commit `e5c5662` in your
clone's history, to read the recipe as it was.

## Run it again

```sh
uv run python scripts/demo/mcp_demo.py --recipe mfass-v2-kmer-baseline \
  --output "$PWD/runs/mfass-demo" --transcript transcript.json \
  --graph plugins/bio-model-benchmarks/.mcp.json --strict
```

This downloads the graph bundle and 80 MB of MFASS data, and takes about a minute. `--strict`
passes only from a clean tree. The digests of the report, the manifest, the logs and the
receipt differ between runs, because they hold times. `predictions.json` does not, on one
machine: it had the same SHA-256 in all three runs made on this laptop. On Linux x86-64 it has
another. (Protocols whose `prepare` gives rows random IDs, such as FLIP2, differ there too.)

To record the plugin path, copy `plugins/bio-model-benchmarks` to a folder outside the
checkout and pass its `.mcp.json` as both `--graph` and `--runner`, without `--strict`.

To check the plugin's three launch lines (the first start builds pyBigWig, which needs a C
compiler and libcurl):

```sh
uv run python scripts/demo/mcp_demo.py --servers plugins/bio-model-benchmarks/.mcp.json \
  --transcript plugin-servers.json
```

The same client code runs the three-row fixture offline on every pull request, in
`tests/test_plugin_demo.py`.

## When a pin is raised

Tests tie this folder to the plugin's `.mcp.json`. After raising the rewirebench pin, record
the run again from a clean checkout of the new commit and replace `transcript.json` and the
run folder, and record `plugin-run.json` again. After raising any of the three pins, record
`plugin-servers.json` again. After raising the graph pin, record `plugin-run.json` again too.
