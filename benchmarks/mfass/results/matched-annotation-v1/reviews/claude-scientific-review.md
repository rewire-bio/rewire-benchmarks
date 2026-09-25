# Scientific review of the completed issue #19 study

Reviewer: Claude (automated), 2026-09-25. Same reviewer as `claude-final-review.md`.
This is automated review, not human review. It is also not an independent reproduction:
I did not rerun any model. I recomputed everything below from the raw prediction, checkpoint,
cohort and split files, without relying on `report.json` or `verification.json`. I edited no
source, registration, manifest, run output or archive, and posted nothing.

**VERDICT: FINDINGS.** There are no blocking findings. Run integrity, provenance and every
reported number check out exactly (sections 1 to 5). Four non-blocking findings concern how
the P@100 results are worded and one reproducibility detail (section 7). Section 8 gives
recommended wording.

## Files reviewed

Repository at `b52a3daef6e85ee869e70732a586355f83dc30fb`, clean working tree.

| File | SHA-256 |
|---|---|
| `workbench/issue-19/manifest-v1.json` | `203567b9055ae1350a910e100c879cace9567411f4e57d932318feda1b0231dd` |
| `docs/mfass-matched-study-registration.md` | `7f03c13d3c7fa73ffe91d3170fe51f06a277d34b5745476ce45ca8a7cece9162` |
| `workbench/issue-19/execution-status.json` | `92ed82330584bb831e96883085cba01fb3904c719bca5cf973b291c862b86054` |
| `$STUDY_DATA/runs/v1/report.json` | `fdd60c840787125758ef7d0e9fe8bc8d44b5a3ef84d95722ada86261c3211339` |
| `$STUDY_DATA/runs/v1/report.md` | `a2b06fa2f942cdb91242ac1280c1ed3bb38ff74735b5f0c1c3a037165ddd234d` |
| `$STUDY_DATA/runs/v1/verification.json` | `815db8b9c5724bab79af5514bca0c446722077e2aede1d3114b415fcb63daa75` |
| `$STUDY_DATA/preflight/variant-eligibility.tsv` | `a0b1a01c8be33e95bc1e0635a30c0f439f8eb732527982b3db13a194989a66a5` (equals `output_sha256` in the committed eligibility summary) |
| `docs/mfass-matched-study-evidence/variant-eligibility-summary.json` | `ec2c57b5ac88ca4011ff02351e4a20092e80d7de75f81acbe8ffb00851ddff93` (equals the manifest's `eligibility.summary_sha256`) |
| `docs/mfass-matched-study-evidence/annotation-summary.json` | `0921c19d1f6a71c4afaf94f366dfbdf2e87fb3435e8a7389493c0d99c04de7f8` (equals the SSD annotation `summary.json` and the manifest) |
| Condition JSONs, prediction, unscored and checkpoint tables, and contrast JSONs (23 files) | all equal to `execution-status.json` `outputs_sha256` (checked, 0 differences) |
| Historical `benchmarks/mfass/results/pangolin-maskFalse.json`, `spliceai-1.3.1.json` | `bb0bb673...`, `6d5c59eb...` (unchanged in `b52a3da`) |

## Commands run

All run with `env.sh` sourced and `$STUDY_DATA/envs/dev/bin/python`. The throwaway scripts
are on the SSD and were not added to the repository.

- `$TMPDIR/sci-review/recompute.py` (`af51f2f0...`): hashes, events, row binding, metrics,
  ties, pair configs and a full bootstrap recomputation of the contrasts.
- `$TMPDIR/sci-review/dist.py` (`0d21668d...`): the distribution of bootstrap P@100 deltas
  (the same draw loop as `paired_group_bootstrap`) and how many scores changed per contrast.
- `shasum`, `git rev-parse`, `git status`, `git diff --stat` for provenance.

## 1. Hashes and event log: PASS

- The registration file equals the manifest hash. `git HEAD` equals the manifest's `git.head`,
  and the manifest's `dirty_paths` list is empty.
- All 15 `runner_files_sha256` entries equal the committed files. All 6 resources (cohort,
  split, FASTA, `.fai`, SpliceAI table, Pangolin DB) equal the manifest.
- Each result config's full `code` block equals the manifest's `code`, including every
  weight's SHA-256 and git blob. Annotation and reference hashes match.
- Recorded output-table and checkpoint hashes equal the files on disk.
- `events.jsonl` holds exactly one manifest hash (`203567b9...`). Each of S0, S1, P0, P1,
  S1-S0, P1-P0 and P0-S0 starts once and exits 0, then come `verify`, `report` and `complete`.
  There is no retry or resume.
- All four results were created after `frozen_at_utc` 2026-09-24T19:00:42Z. S0 was created
  at 20:57 and P1 at 09:10 the next day.

## 2. Rows, denominators and exclusions: PASS

- Every prediction table contains all 8,324 canonical test IDs in cohort order, with labels
  and groups equal to the cohort (`sdv`) and the split. The arm has 315 positives and 463 groups.
- Every condition scored 8,297 and left 27 unscored, with denominator 8,324; the result
  `coverage` matches. The scored rows cover 460 groups and 314 positives, so one positive is
  among the 27 exclusions.
- In all four conditions the unscored IDs are exactly the 27 label-free preflight exclusions:
  23 reference-allele mismatches and 4 variants with no gene under the canonical span.
  - SpliceAI reasons: `ref issue` 23, `no annotated gene` 4.
  - Pangolin reasons: `Mismatch between FASTA...` 23, `Variant not contained in a gene` 4.
- **Object address in SpliceAI reasons.** SpliceAI reason strings embed
  `<__main__._Record object at 0x...>`. This makes the S0 and S1 `unscored.tsv` bytes differ
  (`cd22dae5...` and `ccab724c...`) and makes them non-reproducible across runs. The IDs and
  reason classes are identical, and the text never enters any score, metric or verification
  decision beyond the expected-prefix check. It does not affect the results (finding 4).
- Every prediction equals its checkpoint score formatted to 4 decimals, and each checkpoint
  has 8,324 rows. All scores are 2-decimal values, as upstream prints them.

## 3. Condition metrics: PASS (exact)

Recomputed with `rewirebench.metrics.point_metrics(labels, scores, 100)` from the prediction
rows. Each value equals both the result JSON and `report.json` to within `1e-12`.

| Condition | n | Positives | P@100 | Recall@100 | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 8,297 | 314 | 0.63 | 0.2006 | 0.2954 | 0.8036 |
| S1 | 8,297 | 314 | 0.65 | 0.2070 | 0.3126 | 0.8148 |
| P0 | 8,297 | 314 | 0.65 | 0.2070 | 0.3887 | 0.8763 |
| P1 | 8,297 | 314 | 0.66 | 0.2102 | 0.4106 | 0.8726 |

**Ties at the 100th score:**
- S0: 99 scores above the cutoff 0.68 and 1 tied, so 1 tied score for 1 slot.
- S1: 2 scores tied at 0.61 for 2 slots.
- P0: 1 score tied at 0.55 for 1 slot.
- P1: 3 scores tied at 0.54 for 2 slots, and one of the three is positive.

So P@100 does not depend on tie order for S0, S1 and P0. For P1, any tie order gives either
0.65 or 0.66; the fixed label-independent permutation gave 0.66 (finding 1).

## 4. Pair compatibility: PASS

- The S0/S1 and P0/P1 configs differ in no key except mask, checkpoint, output tables and the
  orphan list, which was empty in all four.
- Threads were TensorFlow 5 intra-op and 1 inter-op for S0 and S1, and Torch 6 and 1 for P0
  and P1, as frozen.
- SpliceAI identity was `upstream-b3c7f17`, verified. P0 and P1 both used identity
  `pangolin-5cf94b8-mask-per-gene-1`, verified, with identical code blocks.

## 5. Contrasts: PASS (exact)

I recomputed each contrast with `rewirebench.metrics.paired_group_bootstrap`: capacity 100,
2,000 draws, seed `20260914`, whole-group resampling, 95% percentile interval and the 5%
single-class refusal. Every field of every metric equals the stored contrast JSON: the
observed delta, the interval, the resample mean and bias, the realised capacity (mean 99.864,
range 83 to 126), the skipped draws and the group count. `report.json` equals the stored
values too.

Each contrast has 8,297 common variants, 314 common positives and 460 groups. The recomputed
common-ID SHA-256 and the full denominators block match, and the input prediction hashes
match the condition tables. `report.md` shows `observed_delta`, not the bootstrap mean. No
draws were single-class and no interval was refused.

| Contrast | Metric | Observed delta | 95% interval |
|---|---|---:|---|
| S1 - S0 | P@100 | +0.020 | [0.000, 0.062] |
| S1 - S0 | AP | +0.017 | [0.006, 0.028] |
| S1 - S0 | AUROC | +0.011 | [-0.007, 0.027] |
| P1 - P0 | P@100 | +0.010 | [0.000, 0.040] |
| P1 - P0 | AP | +0.022 | [0.011, 0.034] |
| P1 - P0 | AUROC | -0.004 | [-0.019, 0.013] |
| P0 - S0 | P@100 | +0.020 | [-0.033, 0.083] |
| P0 - S0 | AP | +0.093 | [0.064, 0.124] |
| P0 - S0 | AUROC | +0.073 | [0.048, 0.096] |

Why two P@100 lower bounds are exactly 0.000: P@100 deltas are discrete, in steps of about
0.01, and many draws tie at zero.

| Contrast | Draws with delta < 0 | Delta = 0 | Delta > 0 |
|---|---:|---:|---:|
| S1 - S0 | 0.5% | 22.5% | 76.9% |
| P1 - P0 | 0.0% | 44.4% | 55.6% |
| P0 - S0 | 17.9% | 10.2% | 71.9% |

The 2.5th percentile therefore lands on 0 exactly. The interval includes zero and does not
exclude "no difference" (finding 2).

Masking changed the score of 1,562 variants for SpliceAI and 2,042 for Pangolin. P0 and S0
differ at 4,943 of 8,297.

## 6. Interpretation limits (as the protocol and registration require)

- **Exploratory and unadjusted.** The MFASS test outcomes had been inspected in earlier work,
  and there are nine intervals (three contrasts times three metrics) with no multiplicity
  adjustment.
- **P0 - S0 does not isolate architecture.** Under a matched annotation, the tools still
  differ in:
  - N-padding outside the transcript (SpliceAI only);
  - masking semantics;
  - training data and labels;
  - ensemble size;
  - score definition (max delta against max absolute change).
- **The future improvement rule does not apply.** It requires a prospectively registered
  comparison on an independent unseen cohort. None of these contrasts qualifies, so it should
  not be reported as "met" or "failed". For reference only: no P@100 delta reaches 0.05, and
  no P@100 lower bound is strictly above zero.
- **Historical context, not a paired comparison.**
  - Historical Pangolin `mask=False` (`mfass-v1`, revision unrecorded, GTF gene spans) scored
    8,301 variants: P@100 0.65, AP 0.3888, AUROC 0.8757. P0 scored 8,297: 0.65, 0.3887,
    0.8763. The 4 fewer are exactly the span-rule losses registered in A6.
  - Historical SpliceAI (bundled GENCODE v24 table) scored 8,194: P@100 0.64, AP 0.2987,
    AUROC 0.8055. S0 scored 8,297: 0.63, 0.2954, 0.8036.
  - The populations, annotations and pinning differ, so these pairs must not be read as
    effect estimates.

## 7. Findings (none blocking)

1. **P1 - P0 P@100 (+0.010) is within tie-order ambiguity (non-blocking, wording).** P1 has 3
   variants tied at 0.54 for the last 2 slots, and one of them is positive. Any
   label-independent tie order gives P1 P@100 of 0.65 or 0.66, so the delta is either 0.00 or
   +0.01. The reported +0.010 comes from the fixed permutation in `precision_at_n`. It is
   valid under the registered rule but carries no evidence of a masking effect at the top 100.
   The S1 - S0 and P0 - S0 P@100 deltas do not depend on tie order.
2. **P@100 lower bounds of exactly 0.000 (non-blocking, wording).** For S1 - S0 and P1 - P0,
   report the interval as including zero. Do not say it is "non-negative" or "borderline
   significant". Say instead that no P@100 difference is established.
3. **`report.md` omits both points (non-blocking).** The report lists the tie counts but not
   their effect on P@100, and it prints `[0.000, ...]` without explanation. Add the wording in
   section 8 to any published summary. Leave the frozen run outputs unchanged.
4. **Non-deterministic SpliceAI reason text (non-blocking, reproducibility).** Unscored-reason
   strings contain a memory address, so `unscored.tsv` bytes are not reproducible across runs,
   and S0 and S1 differ in bytes only for that reason. No result is affected. If the runner is
   reused, strip the object repr from the captured warning. For this study, document it
   rather than change the frozen outputs.

## 8. Recommended wording

> **Results (exploratory).** All four conditions scored the same 8,297 of 8,324 held-out
> variants (314 positives, 460 groups). The 27 unscored variants are exactly the label-free
> preflight exclusions: 23 reference-allele mismatches and 4 variants outside every canonical
> transcript.
>
> **Masking within each tool.** Masking raised average precision for both SpliceAI (+0.017,
> 95% interval 0.006 to 0.028) and Pangolin (+0.022, 0.011 to 0.034). It did not establish a
> change in AUROC (SpliceAI +0.011, -0.007 to 0.027; Pangolin -0.004, -0.019 to 0.013) or in
> P@100 (SpliceAI +0.02, 0.000 to 0.062; Pangolin +0.01, 0.000 to 0.040). Both P@100 intervals
> include zero. The Pangolin P@100 difference depends on how three tied scores at the 100th
> rank are ordered: any label-independent order gives +0.00 or +0.01.
>
> **Unmasked tools under a matched annotation.** Pangolin exceeded SpliceAI in AP (+0.093,
> 0.064 to 0.124) and AUROC (+0.073, 0.048 to 0.096). The P@100 difference was +0.02 (-0.033 to
> 0.083), which includes zero.
>
> **Limitations.** The test outcomes had been inspected before, so these nine intervals are
> exploratory and unadjusted. The matched annotation does not isolate network architecture:
> the tools still differ in input padding, masking rules, training data, ensemble and score
> definition. The improvement rule for future unseen cohorts does not apply to these
> retrospective contrasts. Differences from the historical archived runs reflect different
> annotations, coverage (Pangolin 8,301, SpliceAI 8,194) and pinning, and are context only.
> This analysis was checked by automated review, not human review.
