# Investigate benchmark discrepancies locally

`rewirebench research` connects pinned biological evidence to bounded, reproducible
investigations. It identifies methodological discrepancies and exploratory
hypotheses. It does not establish biological mechanisms, claim novelty, submit
contributions or publish database records.

The worker operates on public artifact manifests plus a **private resolver** from
artifact IDs to local absolute filenames. The normalized test table links outcomes,
predictions, dependence groups and annotations by a unique ID. Missing predictions
remain null and stay in the coverage denominator. Catalogue protocol IDs and native
SDK protocol IDs are separate. Prepared SDK snapshots retain their original IDs,
train/validation/test assignments and semantic checksum.

## Run a campaign

Prepare the four initial cases from the existing local benchmark artifacts. This
requires the MFASS v2 files under `benchmarks/mfass/` and the original prepared
snapshots and saved predictions under `workbench/local-runs-2026-09-20/`. The
preparation script checks cohort hashes, original observation IDs, splits and
metric replay; it does not fetch missing inputs or silently substitute new runs.

```sh
uv sync --locked --all-packages --extra sequence --group dev

uv run --no-sync python scripts/prepare-research-seeds.py \
  --catalogue ../rewire-database/public/omics/releases/2026-09-20-370b30415b09/catalogue.json \
  --output workbench/research-seeds
```

Use a new output directory for each preparation. Only `manifests.json` is
distributable; the resolver and normalized assay tables stay in the ignored
workspace. Regenerate after changing runner Python code or the dependency lock,
because execution checks the installed implementation against the manifest pin.

From the repository environment:

```sh
uv run --no-sync rewirebench research scan \
  --manifests workbench/research-seeds/manifests.json \
  --resolver workbench/research-seeds/resolver.private.json

uv run --no-sync rewirebench research start \
  --manifests workbench/research-seeds/manifests.json \
  --resolver workbench/research-seeds/resolver.private.json \
  --workspace workbench/research-campaign

uv run --no-sync rewirebench research status --workspace workbench/research-campaign
uv run --no-sync rewirebench research stop --workspace workbench/research-campaign
uv run --no-sync rewirebench research resume --workspace workbench/research-campaign

uv run --no-sync rewirebench research export \
  --workspace workbench/research-campaign --output workbench/research-review-export
```

`start` and `resume` run in the foreground. Add `--background` to detach the single
supervisor; its output stays in `supervisor.private.log`. `stop` records a durable
stop request and the watchdog terminates the active process tree. No recurring
scheduler or service is installed. The campaign finishes when its finite eligible
work is exhausted.

Use `--replay-only` on `start` for artifact verification, metric replay and an
available training-mean baseline without any Codex calls. This is a numerical
preflight, not an autonomous research result. For a supervised pilot, reduce the
limits, for example `--limits '{"followup_rounds":0,"codex_calls":12}'` for four
cases with hypothesis generation, planning and criticism. Limits can
only be reduced from the defaults; a new campaign cannot overwrite an old one.

## Execution and resource limits

| Limit | Default |
|---|---:|
| Campaign wall-clock deadline, preserved across resume | 8 hours |
| Concurrent reasoning or numerical jobs | 1 |
| Numerical job timeout | 1 hour |
| Codex invocations | 24 |
| Codex invocation timeout | 15 minutes |
| Process-tree sampled resident memory | 8 GiB |
| Campaign workspace storage | 20 GiB |
| Follow-up rounds per case | 2 |

Numerical jobs run on local CPUs with a restricted environment, one numerical
library thread and offline model-library settings. The registered ESM adapter
requires already acquired, SHA-256-pinned local weights. No operation installs a
package, downloads model weights implicitly, starts paid cloud compute or executes
model-generated code. Explicit artifact HTTPS locations can fill an absent local
resolver entry in a private campaign cache; the worker enforces byte hashes and
storage limits and never overwrites original data. A null URI with no local file
is a blocker. Public host checks and redirect checks reject private network targets.

The watchdog samples process trees and workspace size; it is not a kernel memory
allocation ceiling or disk quota. Small transient overshoots are possible. Time,
storage and memory failures are retained. Children are terminated even if their
leader exits successfully. Temporary prompt files avoid blocking on an unread
stdin pipe. Workspaces, process output and caches are private and excluded from
the public export.

SQLite records every state transition. Plans commit before tests start. Duplicate
operations over identical pinned evidence reuse the recorded attempt. Completed,
failed and interrupted attempts are never silently rerun. Resume retains the
original deadline and consumed reasoning-call budget. Interrupted reasoning calls
also retain their consumed budget; start a separate campaign for an explicit
retry. Authentication or account-limit errors stop reasoning with state intact.
Resuming after runner source changes is refused; keep the original environment or
start a newly pinned campaign.

## Codex integration

The worker reads the configured default model and reasoning effort and reuses
existing Codex authentication. It starts fresh ephemeral `codex exec` sessions for
planning and criticism, with structured JSON outputs and private JSON event logs.
Only aggregate evidence and public manifest metadata enter prompts; raw sequences,
sample-level data and the resolver do not. Standard Codex usage may consume the
account's existing allowance. The worker does not acquire credits.

User-configured plugins, MCP servers, hooks and execution tools are not loaded.
Known tool features are disabled, the CLI uses a read-only sandbox with no approval
requests, and unexpected tool events stop the session. The installed CLI must
support the reviewed flags; unknown configurations fail preflight. Custom provider
profiles require a reviewed integration. The selected model, CLI version and token
usage are recorded in the report. See the [official OpenAI documentation for
non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).

Use `--codex /absolute/path/to/codex` on `start` or `resume` when the configured
model requires a newer installed CLI than the one on `PATH`. The worker preserves
the configured model and reasoning effort; it does not downgrade them to work
around an incompatible executable.

Each case starts with a fresh hypothesis-generation session, followed by planning
and an independent critique of numerical receipts. Hypothesis generation consumes
one call from the same campaign budget. With two follow-ups, a case can use seven
calls; the campaign-wide ceiling remains 24. Additional tool requests become
review items rather than executable commands.

## Form a precise research question

Before selecting experiments, the hypothesizer proposes one to three candidate
questions within the manifest's research scope. Each candidate records:

- The population, comparison and measured outcome.
- A falsifiable hypothesis and a competing explanation.
- Results that would support or contradict the hypothesis.
- Confounders, missing evidence and the independent validation still required.
- Why answering the question would be useful, and a decisive registered test.

The session compares explanatory value and feasibility, selects one question and
records its reasoning. Novelty remains explicitly unverified: generating a
plausible question is not a literature review or a biological finding. Candidate
tests can use only the case's available methods, metrics, annotations and recipes.
Metadata availability does not establish subgroup variation or independence.

The supervisor inserts the selected decisive test into the initial plan, after
artifact verification and metric replay. The planner can add a complementary
hypothesis with up to two tests; it cannot omit the chosen test or replace it with
a prose tool request. It may add no tests when the decisive comparison is enough.
The critic evaluates the resulting evidence against the
predeclared supporting and contradicting observations, including alternatives.

A follow-up request must contain both its precise question and a structured
`next_test` chosen from the same manifest-bound registry. The supervisor inserts
that test into the next plan and freezes it as `followup_test`. The planner may
add up to two complementary hypotheses, or none. The decisive operation cannot
be omitted, replaced by prose or silently changed to another method population.
Previously attempted operations cannot justify another follow-up, and all
existing call, time and round limits still apply. `question_design` belongs only
to the initial specification; it remains historical context for later questions.

Planners distinguish executable hypotheses (`blocker=null`, with real tests)
from blocked hypotheses (an explicit blocker and `tests=[]`). The latter become
`requested_tools` review items and never enter the executable plan. Every listed
test executes unless evidence prerequisites or resource limits stop it; prose
such as “skip this test” does not implement conditional execution. A blocked
hypothesis with a dummy registered operation is rejected before plan freezing.

The optional `followup_test` field preserves historical specification hashes.
Legacy critic checkpoints without `next_test` are read without modification and
cannot schedule a new follow-up; the report records that limitation instead of
guessing an operation from their question text. Existing source-pin checks still
prevent resuming a campaign under a changed implementation.

If the campaign stops during a follow-up, the report retains the latest completed,
validated critique and names its pending question. It distinguishes a test that
never ran, an attempt that did not complete, and completed numerical work awaiting
a final critique. Retained findings describe the evidence that critic actually saw.

If no candidate has a discriminating test with the available evidence, every
candidate retains an explicit blocker. The runner freezes that question design
with a core-only specification, marks the case blocked and executes no tests.
This preserves the missing-data explanation without pretending it was answered.

The initial frozen specification includes `question_design`, retaining all
candidates, the choice and its rationale under the existing plan hashes. Its
`question` is the selected precise question; the manifest and report-level
`question` retain the original scope. Historical specifications without this
optional field remain valid. Completed hypothesis sessions are checkpointed and
are not repeated on resume. A completed, valid design also appears in the bundle's
checksummed `question_design_artifact`, so its candidates and rationale remain
reviewable even if planning fails or the remaining call budget is exhausted before
a specification can be frozen. When both are present, their designs must match.
`--replay-only` bypasses question generation entirely.

## Scientific operations and interpretation

The first frozen plan always verifies evidence and replays registered metrics.
Failed hashes or metric replay block dependent analysis. Available local
training-mean controls are fitted through the SDK. Their installed source,
prepared inputs, environment lock and dependency versions are checked against
pins; no test labels enter training.

The registry supports coverage reconciliation, paired common-population
comparisons, registered subgroup summaries, grouped bootstrap intervals and
sensitivity controls. Numeric annotations use a fixed annotation-only quartile
rule; curated categorical bins retain their labels. Subgroup scans never optimize
cutoffs against prediction error. Sign reversal is a diagnostic and does not alter
the original score direction. Undefined correlations remain null. ProteinGym's
per-assay rounded metrics and FLIP2's minimum-shifted NDCG preserve their evaluator
definitions; selected-assay results never become whole-suite claims.

Bootstrap requires a documented independence unit and complete group assignments;
it does not substitute independent rows for unknown dependence. Intervals are
explicitly exploratory and unadjusted for multiple comparisons. No operation
produces a confirmatory significance claim. All evidence is marked as exploration,
even if it was not examined previously, and cannot be relabelled as independent
validation. The system does not claim a new finding without independent evidence
and a primary-literature novelty review; those capabilities are outside this
initial registry.

## Contracts and review

`research/contract-schema.json` contains the manifest, frozen specification and
investigation bundle JSON Schemas. `research/contract-fixtures.json` supplies valid
and invalid cross-language examples with RFC 8785 canonical JSON hashes. File-byte
SHA-256 and semantic JSON SHA-256 remain distinct.

A bundle includes every frozen hypothesis, executable operation, numerical
receipt, attempted explanation, failed test, model usage and interpretation
limitation. It binds receipts to the exact plan, manifest, table and runner source.
Raw logs, credentials, environment variables and machine paths are excluded.
Exports always remain `review.status = pending`, `claim_level = exploratory`.

The database repository validates the bundle against its source catalogue release,
manifest and artifact resolver before staging it. Human scientific review and a
separate immutable database release are required for public reports. A successful
worker exit or an AI critic is not publication approval.
