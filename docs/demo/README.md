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
| 4 | `rewirebench` | `list_recipes` | Two recipes, each with its `kg_record_id` |
| 5 | | `describe_recipe` | `mfass-v2-kmer-baseline`: two inputs, 80 MB, pinned by SHA-256 |
| 6 | | `plan_run` | Local, CPU, 21.6 seconds expected, owned hardware, no amount to confirm |
| 7 | | `run` | Started, with the plan's `plan_sha256` and the graph release ID |
| 8 to 17 | | `run_status` | Polled every 2 seconds until finished, exit code 0, checks passed |
| 18 | | `get_receipt` | The receipt |
| 19 | | `verify_receipt` with `strict` | Passed: all eight checks match, nothing left unchecked |

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
- code and recipe commit `da7694213d5148c4d7fa2ac4db85c6d15077cbea` on `main`, clean tree, `commit_source` `checkout`;
- inputs fetched from KosuriLab/MFASS at commit `9a8e4f27`, both digests matched;
- Apple M4, arm64, 10 logical cores, 16 GiB, no GPU; macOS, Python 3.11.13, numpy 1.26.4,
  scikit-learn 1.9.1, scipy 1.17.1;
- wall time 18.08 seconds including the download, peak memory 812 MB;
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
  what lets the receipt name a clean tree and a recipe commit, and pass `strict`. The plugin
  path is recorded separately: see the next section.
- **Nothing is signed.** The receipt shows the folder is intact and consistent with the
  recipe. It does not prove where the folder came from.
- **The predictions were not scored again by `verify_receipt`.** It reads the checked values
  from `report.json`. The `commits` check reads the commit the receipt states and does not
  look it up. The recipe pins no digest for `predictions.json`, and one digest would not hold
  on every machine: two runs of it in CI on Linux x86-64 gave the same metrics with 19 of the
  8,324 predictions different in the last bit.

## The same run through the plugin as shipped

[`plugin-run.json`](plugin-run.json) is the transcript of the same 18 calls (3 to the graph, 15
to rewirebench) with both servers started by the launch lines in the plugin's `.mcp.json`.
The plugin folder was copied outside any checkout first, as an installed plugin is, so
`${CLAUDE_PLUGIN_ROOT}/recipes` was a plain folder. It was recorded on 2026-11-21 on the same
laptop. It makes one `run_status` call fewer than the checkout run, because the run finished
one poll sooner.

| | Checkout run (above) | Plugin run |
| --- | --- | --- |
| rewirebench | `python -m rewirebench.cli mcp` from a clean checkout of `da76942` | `uvx` wheel built from `da76942` |
| Recipes | `recipes/` of the checkout | the plugin's copy, same SHA-256 |
| Python, packages | 3.11.13, locked by `uv.lock` (numpy 1.26.4, scipy 1.17.1) | 3.12.7, not locked (numpy 2.5.5, scipy 1.18.1) |
| `code.commit` | `da76942` | `da76942` |
| `code.commit_source` | `checkout` | `install_metadata` |
| `code.dirty` | false | null |
| `recipe.commit` | `da76942` | null |
| `verify_receipt` | passed with `strict` | passed; `commits` not checked, so `strict` does not pass |
| Precision at 100 | 0.61 | 0.61 |
| Average precision | 0.2864167459589237 | 0.28641674595892375 |
| AUROC | 0.7779498064677238 | 0.7779498064677238 |
| `sdk_code_sha256` | `ef938d72...` | `ef938d72...` |
| SHA-256 of `predictions.json` | `2683ef05...` | `2683ef05...` |
| Wall time | 18.08 s | 17.24 s (inputs downloaded again) |

This is what a user of the installed plugin should expect. It is still a scripted client and
not a Claude Code session with the plugin loaded: `${CLAUDE_PLUGIN_ROOT}` was substituted by
the client.

The plugin receipt names the pinned commit. A wheel has no checkout, so the runner reads the
commit that `uvx` recorded when it installed the package from git (`vcs_info.commit_id` in
the package's `direct_url.json`) and marks it `install_metadata`. It still does not pass
`strict`, for two reasons the `commits` check states: no tree was checked for changes, and
the plugin's recipes are a plain copy, so there is no recipe commit.

Both receipts record the installed package versions under `environment.packages`.

In the transcripts, the `version` recorded for the `rewirebench` server (1.30.0) is the version
of the MCP SDK it is built on. The rewirebench version, 0.5.0, is in the receipt.

The step log in the run folder names `{work}` where the run used its scratch folder, and a
test checks that nothing in the folder names a local folder.

In the transcripts local folder names are replaced by labels such as `<run folder>`. The
client replaces them itself and refuses to write a transcript that still names a local
folder. All three transcripts are its output as written, with nothing edited by hand. The
checkout run was recorded with `--label <path of the checkout> "<checkout>"`.

## Check it yourself

From a checkout of this repository:

```sh
uv run rewirebench receipt verify docs/demo/mfass-v2-kmer-baseline
```

It recomputes every digest in the folder and reads the checked values again from the outputs.
If the recipe file has changed since the run, the recipe check needs commit `da76942` in your
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
machine: it had the same SHA-256 in every run made on this laptop, at both pinned commits. On Linux x86-64 it has
another. (Protocols whose `prepare` gives rows random IDs, such as FLIP2, differ there too.)

To record the plugin path, copy `plugins/bio-model-benchmarks` to a folder outside the
checkout and pass its `.mcp.json` as both `--graph` and `--runner`, without `--strict`. If the
checkout or the output folder is under a path the client does not label by itself, add
`--label PATH NAME`.

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
the run again from a clean checkout of the new commit, with Python 3.11 and `uv sync --locked`,
and replace `transcript.json` and the run folder, and record `plugin-run.json` again. The pin
must name a commit on `main`, so a runner change and the pin that picks it up are two pull
requests. Record `plugin-servers.json` from the copied plugin folder too, so it starts the new
pin. After raising any of the three pins, record
`plugin-servers.json` again. After raising the graph pin, record `plugin-run.json` again too.
