# Reference baselines and model evaluations

This implements the first execution batch of the approved programme. It does not
mean every catalogue protocol now has measured baselines. The database coverage
register separately tracks published and proposed releases, missing protocol
definitions, baseline selection and model compatibility.

## Agreed programme

Cover every concrete protocol with a null control and a useful conventional
reference where scientifically applicable. Prioritise more model families on
matched complete tasks before whole suites. Credit original methods. Keep exact
checkpoints, information inputs, fitting, dataset versions, splits, denominators,
failures and score directions explicit. Record source review separately from local
execution and reproduction of a published score. Retain negative findings.

Use local tests and existing audited artifacts now. Prepare larger runs for a
future DGX or Google Cloud Batch through Nextflow. No paid infrastructure is
provisioned. Every run exports locally; only a separately requested SDK submission
reaches private review. Publication is a later reviewed database release.

## Run registered baselines

```bash
rewirebench baselines flip2-fitness-v1
rewirebench inspect flip2-fitness-v1
rewirebench run-baselines --prepared ./prepared-flip --output ./baseline-runs
```

The output directory must be new. `baseline-plan.json` fixes selected baseline
versions and parameters before execution. Each method writes its own predictions,
coverage and report; `baseline-manifest.json` distinguishes evaluated, blocked and
failed entries. A manifest is not evidence of successful scientific reproduction.
Use `--baseline BASELINE_ID` repeatedly to choose an explicit subset.

The registry covers the eight supported SDK protocol identifiers. A proposed
conventional ProteinGym or DART reference does not produce a score: its missing
method/input review remains a blocker. The TDC string n-gram reference is explicitly
not a chemical fingerprint model or invariant to alternate SMILES representations.

Python offers the same interfaces:

```python
from rewirebench import describe_baselines, run_baselines

specification = describe_baselines("flip2-fitness-v1")
manifest = run_baselines("./prepared-flip", output="./baseline-runs")
```

## Frozen protein example

`rewirebench.adapters.esm_embeddings:ESM2Embeddings` supports pinned ESM-2
`esm2_t6_8M_UR50D` and `esm2_t12_35M_UR50D` checkpoints. Set the explicit
`model_name` adapter option for 35M; 8M is the default. It returns final-layer mean residue embeddings,
excluding special tokens and padding. It rejects sequences longer than 1022
residues rather than cropping them. A FLIP2 protocol-owned ridge head is fitted on
training embeddings only. This encoder-plus-head is a specific pipeline, not a
model-wide score or reproduction of a published FLIP2 result.

```bash
rewirebench run --prepared ./prepared-flip --output ./esm2-flip \
  --adapter rewirebench.adapters.esm_embeddings:ESM2Embeddings \
  --adapter-options '{"checkpoint":"/weights/esm2_t6_8M_UR50D.pt","device":"cpu"}' \
  --prediction-type embedding --batch-size 4 \
  --model-name 'ESM-2 8M final-layer mean + fixed FLIP2 ridge' \
  --training-overlap 'Unreported; UniRef50 pretraining may overlap benchmark proteins'
```

The checkpoint is obtained during explicit preparation, not by the adapter.
See [Nextflow execution](nextflow.md) for local, HPC and cloud profiles. Remote
profiles are not claimed tested without an execution receipt from that platform.

The [matched Rhomax evidence](../research/baseline-programme-2026-09-21/README.md)
records complete 184-row evaluations of both checkpoints and the composition
reference, including negative findings, source verification and exact code snapshots.

## Queue and submit through the package

Create the sanitized export, inspect it and supply an immutable public evidence
URL for the actual run. `enqueue_submission()` derives the headline value from a
key path in the bundle. All other metrics remain in that same contribution.

```python
from rewirebench import export, enqueue_submission, drain_submissions
from getpass import getpass

bundle = export("./esm2-flip", output="./esm2-flip-bundle.json")
enqueue_submission(
    bundle, queue="./private-submission-queue",
    title="ESM-2 8M frozen probe on FLIP2 Rhomax",
    summary="Evaluation on the full declared Rhomax test split with a train-only head.",
    source_url="https://github.com/OWNER/REPOSITORY/blob/COMMIT/report.json",
    source_locator="report.json: metrics and coverage",
    metric_path=["spearman"],
)
preview = drain_submissions("./private-submission-queue")  # No network.
# After reviewing the payload and verifying your contribution account:
receipt = drain_submissions("./private-submission-queue",
    token=getpass("Short-lived contribution token: "), dry_run=False)
```

Replace the illustrative evidence URL with the inspected committed artifact. Keep
the queue outside Git. It stores approved bundle fields and public evidence
metadata in owner-readable files; it never stores authentication tokens. The
queue requires a POSIX system, matching the local/HPC execution targets.

CLI equivalents are `rewirebench enqueue` and `rewirebench submit-queue`.
`submit-queue` previews by default; `--send` explicitly sends a bounded batch using
`REWIRE_SUBMISSION_TOKEN`. Do not include tokens in shell command arguments.

The coordinator calls `rewirebench.submit`, then confirms the returned private ID
through `submission.get`. An acknowledged POST whose tracking lookup fails is
retained and its GET is retried without another POST. Uncertain POST outcomes reuse
the original payload and idempotency key. Authentication, quota or disabled-service
errors pause the queue. A fingerprint of the Firebase issuer, project and account
binds each attempted item to its original contributor; a refreshed token for the
same account is allowed. These locally decoded claims do not authenticate a token:
the service verifies its signature, expiry, revocation and email. Validation/conflict
errors require a reviewed correction.
Changed metadata for the same evaluation is rejected rather than silently creating
a second scientific result. Existing contributions need the site's revision flow.

No queue drain occurs inside Nextflow model tasks. This keeps submission credentials
outside model containers and avoids duplicate contributions on workflow retries.
Live upload completion requires a real returned ID and successful tracking lookup;
a local test or queue file is not an upload. Intake activation and receipt-email
testing are dependencies tracked in the separate contribution-service PR.

## Delivery status, 21 September 2026

- Runner and evidence review: [PR10](https://github.com/rewire-bio/rewire-benchmarks/pull/10).
- Website audit and coverage: [database PR26](https://github.com/rewire-bio/rewire-database/pull/26).
- Live intake dependency: [database PR25](https://github.com/rewire-bio/rewire-database/pull/25).
- Five new evaluated configurations: ESM-2 8M and 35M frozen Rhomax probes,
  the 22-feature composition reference, the [MFASS training-prior control](../research/mfass-null-2026-09-21/README.md),
  and a [fixed-seed ProteinGym AMFR null](../research/proteingym-null-2026-09-21/README.md).
- Ten distinct evaluation bundles passed queue dry-run validation: the five
  prior evaluations plus these five. All ten payloads also passed the current
  TypeScript contribution schema. Repeated controls are excluded. No upload
  or publication is claimed; production intake returned HTTP 503 while disabled.

Local validation passed 473 Python tests with 11 optional skips. Both ESM pooling
checks also passed in the actual Torch execution environment. The built 0.5.0
wheel was installed and imported outside the checkout. Native Nextflow smoke,
resume, code/configuration invalidation and failure checks passed. The actual
scientific runs preserve their older development snapshots and exact hashes;
rebuilding the release wheel does not alter that provenance.

## Outstanding programme work

- Review baseline choices for catalogue protocols beyond the existing SDK adapters.
- Implement structured-output protocols and conventional reference methods in
  reviewed modality batches.
- Verify more model checkpoints, access conditions and compatible complete tasks.
- Validate actual DGX/Slurm/cloud execution when compute becomes available.
- Submit audited runs after live intake and receipt delivery are verified.
- Release accepted numerical evidence through the database's normal review process.

The downloadable audit must keep implementation, smoke tests, full evaluation,
submission and publication as separate states throughout these batches.
