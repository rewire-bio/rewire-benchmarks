# Scheduled agents

Agents that run from GitHub Actions in this repository. They follow the pattern of the agents in rewire-bio/rewire-benchmark-data (its `docs/agents.md`): plain Node scripts on the Anthropic SDK, three modes, a cost cap per run, and a `usage.json` artifact with each run's token use and estimated cost.

## Gap scanner

`.github/workflows/gaps.yml` runs `scripts/agents/gaps.mjs` every Wednesday at 06:00 UTC, and by hand (#60). The prompt is [agents/gaps.md](../agents/gaps.md); change it in a pull request.

It reads the newest published release of the data store: the newest `serving/<id>` release that also has a `kg/<id>` bundle, the same pair the disagreement watcher in the data store reads. From `serving/<id>` it takes `use-cases.json.gz` and `records.jsonl.gz`; from `kg/<id>`, `discrepancies.json`. The data store is the `DATA_REPO` constant in the script, or the repository variable `GAPS_DATA_REPO` when set. Proposals are filed in the repository the workflow runs in.

### What it looks for

Only tier 1 use cases (DNA and RNA genomics, `depth_tier` 1). A **ranking** is one such use case's active direct mapping on which at least two systems have results.

| Gap | Rule |
| --- | --- |
| Use case with no direct evidence | No active mapping with relevance `direct`. Filed as "Proposed benchmark". |
| Ranked on a single study | Every result for the system on the use case's direct protocols comes from one study. A study is a set of sources sharing a DOI or GitHub repository, or joined by `companion_of`, as in the data store's discrepancy rules. A system whose only results are Rewire runs is not flagged: a second Rewire run is not a second study. |
| No independent evaluation | No result on those protocols has origin `independent_paper` or `rewire_run`. `author_reported`, `paper_compilation` (copied results) and `unreported` do not count. |
| Open discrepancy a rerun would settle | A live, unsettled finding with review state `open`, scope `same-protocol` or `same-task`, and a tier 1 system. The run is on a protocol two studies share; else the most specific task or benchmark both studies' protocols are part of; else the outlier study's own protocols, since reproducing its order or not is what settles it. |
| Run recipe never executed | A tier 1 benchmark or protocol (or one mapped to a tier 1 use case) whose run recipes and run guide all have status `source_reviewed_not_executed`, and with no Rewire run on any protocol under it. A CPU smoke check in `scripts/recipe_smoke/` is mentioned but is not an executed run. |

System identity follows the discrepancy rules: a configuration counts as its model through `configuration_of`, then `family`, then `alias_of`.

### Value, estimate and order

- **Value** is the clinical weight (2 when a use case the gap serves has a clinical context, else 1) times one plus the number of rankings the run could change.
- **GPU-hours** come from a recorded runtime result for the same system on the same protocol whose hardware names a GPU and its count (hours times GPU count), or are 0 when the recipe's hardware field says no GPU or accelerator is needed. Otherwise the issue says "no estimate" and why. A runtime on an evaluation that records only training hardware is not used.
- **Dollars** come only from a dated, sourced hourly price in `GPU_PRICES` in the script. It is empty, so every GPU run says "no estimate" for cost until a price is added with its source.
- **Order:** runs that need no GPU first, by value; then runs with a GPU-hour estimate, by value per GPU-hour; then runs with no estimate, by value. Nothing is divided by an assumed figure. At most one proposal per benchmark a week, and at most five a week.

### What each issue holds

The gap, "Why it matters", the exact model and benchmark (with record IDs), the estimate and its basis, the ranking rule, and the command a person would run, copied verbatim from the store's run guide or recipe. When the store has no recipe, the issue says no command exists. Every number, name, command and link comes from the release data. Sonnet 5.5 writes only the "Why it matters" paragraph, from computed facts; a paragraph that states a number (outside a name such as BRCA1) or a link is dropped and the issue says so.

### Dedupe

- Each issue carries a hidden `gap-proposal` marker with the gap's key. A key with a marker on any issue, open or closed, is never filed again, so a closed or declined proposal stays closed.
- Hand-written open issues are matched by name: the system and its benchmark or protocol (in full, or by at least two of the benchmark's acronyms and words with digits, such as ClinGen, PP3 and BP4); both systems of a discrepancy and its anchor, or the finding ID; the recipe's benchmark with "recipe", "run guide" or "smoke"; or the use case's title or slug. Closed hand-written issues do not count. This keeps it off #40 to #56, #65 and #67 to #71. Every skip is listed in the job summary with the issue that covers it.

### Modes

| Mode | Model calls | GitHub writes | When |
| --- | --- | --- | --- |
| `dry-run` | No | No | Pull requests that touch the scanner, and by hand. Runs `tests/agents-gaps.test.mjs` and the scan on `tests/agents-gaps-fixtures/`. |
| `report-only` | Yes, if the key is set | No | The Wednesday schedule while `GAPS_MODE` is unset, and by hand. Proposals are in the job summary and the `gaps-<run>` artifact. |
| `live` | Yes, if the key is set | Issues | The schedule once `GAPS_MODE` is `live`, and by hand. |

To go live: `gh variable set GAPS_MODE --body live -R rewire-bio/rewire-benchmarks`.

Locally: `node scripts/agents/gaps.mjs --download /tmp/rel`, then `GITHUB_REPOSITORY=rewire-bio/rewire-benchmarks GH_TOKEN=$(gh auth token) node scripts/agents/gaps.mjs --report-only --release-dir /tmp/rel --out /tmp/gaps`. Tests: `node --test tests/agents-gaps.test.mjs`.

### Setup and cost

- Issues are filed as the rewire-bot app (`REWIRE_BOT_APP_ID` variable, `REWIRE_BOT_PRIVATE_KEY` secret), with a token limited to this repository and issues write. A live run with no app token fails without filing.
- `ANTHROPIC_API_KEY` is optional. Without it, every mode runs, no model call is made, and proposals have no "Why it matters" paragraph.
- One Sonnet 5.5 call per proposal, effort low, up to 2,000 output tokens, with the server-side refusal fallback. The facts are about 1,000 to 2,000 tokens, so a call costs about $0.01 to $0.03 and a week about $0.15. No call is made once the run's estimate passes $1.
- `usage.json` in the artifact has the shape the data store's weekly digest reads (`byModel`, `estimatedUsd`). The digest reads only its own repository's runs, so this spend is not in it yet.

### Known gaps

- Receipts are read from the store's recipe statuses and Rewire-run evaluations only. The CPU smoke receipts this repository's workflow uploads are not read until the store records them (run-recipe schema, #64).
- The value counts rankings, not how far a ranking might move.
- Few records state GPU runtime with a GPU count, and no GPU price is sourced, so most proposals say "no estimate".
- Name matching against hand-written issues is a heuristic. Check the job summary's skipped list after each run.
