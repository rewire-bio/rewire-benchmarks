# Recorded demo: MFASS v2 baseline, rerun through MCP tools only

On 2026-11-22 the corrected k-mer and position baseline on MFASS v2 was rerun on a laptop CPU
through MCP tool calls only, and compared with the value in the benchmark knowledge graph.
This folder holds the record.

| File | What it is |
| --- | --- |
| [`transcript.json`](transcript.json) | Every tool call, in order, with its arguments and result |
| [`mfass-v2-kmer-baseline/`](mfass-v2-kmer-baseline) | The complete run folder: `receipt.json`, `SHA256SUMS`, the outputs and the step logs |
| [`plugin-servers.json`](plugin-servers.json) | The three servers started exactly as the plugin's `.mcp.json` declares them, with one read-only call to each |
| [`plugin-run.json`](plugin-run.json) | The same MFASS v2 rerun made through the plugin's own launch lines, from a copy of the plugin folder outside any checkout. Every tool call; the run folder is not committed |
| [`agent-session.md`](agent-session.md) | One Claude Code session in which a model made the calls, recorded at an older pin (`e5c5662`). Every tool call and result, with local folders replaced by labels; the run folder is not committed |

## What ran

The client is [`scripts/demo/mcp_demo.py`](../../scripts/demo/mcp_demo.py). It starts MCP
servers on stdio and reaches them only through `call_tool`. It calls no rewirebench function
and starts no rewirebench command itself.

| # | Server | Tool | Result |
| --- | --- | --- | --- |
| 1 | `rewire-benchmark-graph` (rdf-kg-mcp v0.4.0) | `dataset_info` | Release `2026-11-21-c8bea25e5abb` |
| 2 | | `find_entity` | `id:rewire-mfass-v2` |
| 3 | | `sparql_query` | Published values of `id:rewire-model-baseline-kmer-position-v2`, each on 8324 of 8324, status `reproduced` |
| 4 | `rewirebench` | `list_recipes` | Two recipes, each with its `kg_record_id` |
| 5 | | `describe_recipe` | `mfass-v2-kmer-baseline`: two inputs, 80 MB, pinned by SHA-256 |
| 6 | | `plan_run` | Local, CPU, 21.6 seconds expected, owned hardware, no amount to confirm |
| 7 | | `run` | Started, with the plan's `plan_sha256` and the graph release ID |
| 8 to 14 | | `run_status` | Polled every 2 seconds until finished, exit code 0, checks passed |
| 15 | | `get_receipt` | The receipt |
| 16 | | `verify_receipt` with `strict` | Passed: all eight checks match, nothing left unchecked |

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
- code and recipe commit `d74a030ee735f26f7214f245de107b245a47a664` on `main`, clean tree, `commit_source` `checkout`;
- inputs fetched from KosuriLab/MFASS at commit `9a8e4f27`, both digests matched;
- Apple M4, arm64, 10 logical cores, 16 GiB, no GPU; macOS, Python 3.11.13, numpy 1.26.4,
  scikit-learn 1.9.1, scipy 1.17.1;
- wall time 12.78 seconds including the download, peak memory 807 MB;
- cost: no paid compute, reason "owned hardware";
- container image: none. No rewirebench image is published, so the run used the host Python
  environment and both platform digests in the receipt are null.

## What this does and does not show

It shows that the path works: a client with only MCP tools can read the published value,
plan the run, run it, get a receipt and verify it, and the rerun gives the published numbers.

It does not show these:

- **A model did not choose the calls.** A script made them, in the order the
  `verify-published-result` skill describes. `started_by` in the receipt is `agent` because
  the MCP server records that for every run it starts. One session in which a coding agent
  with the plugin loaded makes the calls is recorded separately, at an older pin: see
  [An agent session](#an-agent-session). None has been recorded at the current pin.
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

[`plugin-run.json`](plugin-run.json) is the transcript of the same run with both servers started by
the launch lines in the plugin's `.mcp.json`. It holds the same 18 calls (3 to the graph, 15 to
rewirebench) as the checkout run with two `run_status` polls added. The checkout run made 16.
The plugin folder was copied outside any checkout first, as an installed plugin is, so
`${CLAUDE_PLUGIN_ROOT}/recipes` was a plain folder. It was recorded on 2026-11-22 on the same
laptop. It makes 9 `run_status` calls where the checkout run made 7, because it took
longer.

| | Checkout run (above) | Plugin run |
| --- | --- | --- |
| rewirebench | `python -m rewirebench.cli mcp` from a clean checkout of `d74a030` | `uvx` wheel built from `d74a030` |
| Recipes | `recipes/` of the checkout | the plugin's copy, same SHA-256 |
| Python, packages | 3.11.13, locked by `uv.lock` (numpy 1.26.4, scipy 1.17.1) | 3.12.7, not locked (numpy 2.5.5, scipy 1.18.1) |
| `code.commit` | `d74a030` | `d74a030` |
| `code.commit_source` | `checkout` | `install_metadata` |
| `code.dirty` | false | null |
| `recipe.commit` | `d74a030` | null |
| `verify_receipt` | passed with `strict` | passed; `commits` not checked, so `strict` does not pass |
| Precision at 100 | 0.61 | 0.61 |
| Average precision | 0.2864167459589237 | 0.28641674595892375 |
| AUROC | 0.7779498064677238 | 0.7779498064677238 |
| `sdk_code_sha256` | `e81d753b...` | `e81d753b...` |
| SHA-256 of `predictions.json` | `2683ef05...` | `2683ef05...` |
| Wall time | 12.78 s | 15.85 s (inputs downloaded again) |

This is what a user of the installed plugin should expect. It is still a scripted client and
not a Claude Code session with the plugin loaded: `${CLAUDE_PLUGIN_ROOT}` was substituted by
the client.

The plugin receipt names the pinned commit. A wheel has no checkout, so the runner reads the
commit that `uvx` recorded when it installed the package from git (`vcs_info.commit_id` in
the package's `direct_url.json`) and marks it `install_metadata`. It still does not pass
`strict`, for two reasons the `commits` check states: no tree was checked for changes, and
the plugin's recipes are a plain copy, so there is no recipe commit.

The first reason holds for every installed plugin. The second depends on where the plugin
folder is. A recipe gets a commit when its folder is inside a git work tree whose HEAD holds
the file with the same bytes, whichever repository that is. Where Claude Code puts an
installed plugin has not been tested here, so `recipe.commit` in a user's receipt may be null
or a commit. `strict` fails in both cases.

The runner records the installed commit only while every installed file has the SHA-256 in
the wheel's `RECORD`. If the installed package is edited afterwards, `code.commit` is null.

Both receipts record the installed package versions under `environment.packages`.

In the transcripts, the `version` recorded for the `rewirebench` server (1.30.0) is the version
of the MCP SDK it is built on. The rewirebench version, 0.5.0, is in the receipt.

The step log in the run folder names `{work}` where the run used its scratch folder, and a
test checks that nothing in the folder names a local folder.

In the transcripts local folder names are replaced by labels such as `<run folder>`. The
client replaces them itself and refuses to write a transcript that still names a local
folder. All three JSON transcripts are its output as written, with nothing edited by hand. The
checkout run was recorded with `--label <path of the checkout> "<checkout>"`.

## An agent session

[`agent-session.md`](agent-session.md) is a copy of one Claude Code session (version 2.1.330)
in which a model with the plugin loaded made the calls. It was recorded on 2026-11-21 in two
headless runs of one session. The recordings were kept as recordings of
`/bio-model-benchmarks:demo`; they do not hold what the user typed.

It was recorded at an older pin than the two runs above. The plugin was version 0.2.0, loaded
from the plugin folder of a checkout at commit `26fd89b`, where `.mcp.json` pins rewirebench
at `e5c5662`. The recording does not name that pin itself. The receipt in it has the
`sdk_code_sha256` `3297cfe0...`, which is the digest of the rewirebench sources at `e5c5662`,
computed here from that commit. The plugin is now 0.2.2 and pins `d74a030`. The agent session
has not been recorded again: not at `d74a030`, and not at `023a7a2`, the pin before it. What
follows describes the recording made at `e5c5662`.

What the recording shows:

- The agent made 14 tool calls. Nine went to the plugin's servers: `describe_recipe`,
  `dataset_info`, `sparql_query` and `plan_run`, then, after it had asked for agreement to the
  download and the run, `run`, `run_status`, `get_receipt` and `verify_receipt` twice. The
  other five were Claude Code's own tools.
- It did not make the calls in the scripted order. It called neither `find_entity` nor
  `list_recipes`, and it waited for the receipt with a shell loop before one `run_status` call.
- The run finished with exit code 0 and the 8 recipe checks passed. Precision at 100 was 0.61,
  AUROC 0.7779498064677238 and average precision 0.28641674595892375, on 8324 variants, with
  Python 3.12.7.
- `verify_receipt` passed with `commits` not checked. With `strict` it did not pass. In that
  receipt `code.commit` is null and `recipe.commit` is `26fd89b`. The plugin folder
  was inside a checkout.

The copy was made by a script from the two stream-json recordings. It replaces local folders
by labels and leaves out IDs, times, token counts and the session's list of other tools. The
file states each change.

## The command line at the pinned commit

The same recipe was run once more on 2026-11-22 from the command line, without MCP, from a
fresh shallow clone of `main` at commit `d74a030`, the commit the plugin now pins,
with Python 3.11.13 and `uv sync --locked --all-packages`:

```sh
uv run rewirebench recipe run mfass-v2-kmer-baseline --output <folder>
uv run rewirebench receipt verify <folder> --strict
```

- The run exited 0 with `checks_passed` true: 8324 variants, 315 positives, precision at 100
  of 0.61, average precision 0.2864167459589237 and AUROC 0.7779498064677238. These equal the
  published values in the Result table above. This run did not read the graph.
- `receipt verify --strict` passed: all eight checks match and nothing was left unchecked. The
  `commits` check read code commit `d74a030ee735f26f7214f245de107b245a47a664` from the receipt.
- Wall time was 29.442 seconds including the download, and peak memory 811 MB, on an
  Apple M4 laptop (numpy 1.26.4, scikit-learn 1.9.1, scipy 1.17.1).
- `predictions.json` had the SHA-256 `2683ef05...`, as in the two recorded runs.
- `sdk_code_sha256` was `e81d753b...`, as in the two recorded runs.

That run folder is not committed. The committed run folder is the one recorded through MCP
from the same clone.

Only the command line and the two scripted MCP runs were made again at this pin. The agent
session in [`agent-session.md`](agent-session.md) is from `e5c5662` and has not been recorded
again.

## Check it yourself

From a checkout of this repository:

```sh
uv run rewirebench receipt verify docs/demo/mfass-v2-kmer-baseline
```

It recomputes every digest in the folder and reads the checked values again from the outputs.
If the recipe file has changed since the run, the recipe check needs commit `d74a030` in your
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
machine: it had the same SHA-256 in every run made on this laptop, at each of the four commits pinned so far. On Linux x86-64 it has
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
