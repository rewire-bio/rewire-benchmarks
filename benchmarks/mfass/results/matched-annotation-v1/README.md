# MFASS annotation-matched specialist study (issue #19): results

This directory holds the complete outputs of the S0/S1/P0/P1 study registered in
[`docs/mfass-matched-study-registration.md`](../../../../docs/mfass-matched-study-registration.md)
under the protocol in
[`docs/mfass-specialist-comparison.md`](../../../../docs/mfass-specialist-comparison.md).
**The study is exploratory.** The MFASS test outcomes had been inspected in earlier work.
It is not a confirmatory model comparison.

## Run identity

- **Source commit:** `b52a3daef6e85ee869e70732a586355f83dc30fb`.
- **Registration:** SHA-256 `7f03c13d3c7fa73ffe91d3170fe51f06a277d34b5745476ce45ca8a7cece9162`.
- **Frozen manifest:** raw SHA-256 `203567b9055ae1350a910e100c879cace9567411f4e57d932318feda1b0231dd`,
  frozen 2026-09-24T19:00:42Z. The public copy is `manifest-v1.json`, with paths substituted;
  see Provenance below.
- **Pre-registration notice:** posted on issue #19 before any MFASS variant was scored
  ([comment](https://github.com/rewire-bio/rewire-benchmarks/issues/19#issuecomment-5820362807)).
- **Execution:** one sequential local run on an Apple M4 CPU, from 2026-09-24T19:01:51Z to
  2026-09-25T09:11:00Z (50,949 s).
  - Threads: TensorFlow 5 intra-op and 1 inter-op; Torch 6 and 1.
  - `logs/events.jsonl` shows each step starting once and exiting 0. There were no retries or
    resumes.
- **Review:** automated Claude review only, not human review. The scientific review is in
  [`reviews/claude-scientific-review.md`](reviews/claude-scientific-review.md). It recomputed
  every hash, row, metric and bootstrap interval from the raw files and found exact agreement.
  Its four findings are non-blocking and concern wording and reproducibility; they are
  addressed below.

## Configuration

| Condition | Tool and code | Annotation | Mask | Score per variant |
|---|---|---|---|---|
| S0 | SpliceAI 1.3.1 (upstream `b3c7f17` bytes; 5 models) | matched GENCODE v44 table | `0` | max DS_AG/AL/DG/DL over returned genes |
| S1 | as S0 | as S0 | `1` | as S0 |
| P0 | Pangolin `5cf94b8` with patch `pangolin-5cf94b8-mask-per-gene-1` (12 models) | matched GENCODE v44 gffutils DB | `False` | max absolute reported change over genes and sites |
| P1 | as P0 | as P0 | `True` | as P0 |

- **Annotation:** one `Ensembl_canonical` transcript for every one of the 62,754 genes in the
  GENCODE v44 primary-assembly GTF. Both formats are built from that single selection, and
  the gene span equals the transcript span.
- **Reference:** GENCODE 44 GRCh38 primary-assembly FASTA; MD5 checked against the official
  `MD5SUMS`.
- **Distance:** 50 bases.
- **Pangolin patch:** gives each gene its own copy of the score arrays before masking
  (upstream issue #29). Its identity, file and GPL-3.0 licence are in
  [`../../patches/`](../../patches/).
- **Identity checks:** every code file and model weight was checked against pinned upstream
  hashes and Git blob IDs before scoring. The hashes are in `manifest-v1.json`, under `code`.

## Coverage

All four conditions scored the same **8,297 of 8,324** held-out variants: 314 of the 315
positives, in 460 of the 463 groups. The 27 unscored variants are exactly the label-free
preflight exclusions:

| Reason | Count | SpliceAI text | Pangolin text |
|---|---:|---|---|
| Reference allele differs from the FASTA | 23 | `skipped by SpliceAI: Skipping record (ref issue): …` | `… Mismatch between FASTA … and variant file …` |
| No gene under any canonical transcript span | 4 | `no annotated gene overlapping the variant` | `… Variant not contained in a gene body …` |

The SpliceAI reason strings embed a Python object address (`<__main__._Record object at 0x…>`),
taken from upstream's log message. As a result, S0 and S1 `unscored.tsv` bytes differ, and
they would not be byte-identical in a repeat run. The IDs and reason classes are identical,
and the text affects no score or metric. The files are published unchanged.

## Results

All values below are copied from `report.json` and the contrast files. They are shown to
3 or 4 decimals; the exact values are in those files.

| Condition | P@100 | Recall@100 | AP | AUROC | Tied at the 100th score |
|---|---:|---:|---:|---:|---|
| S0 | 0.63 | 0.2006 | 0.2954 | 0.8036 | 1 tied for 1 slot |
| S1 | 0.65 | 0.2070 | 0.3126 | 0.8148 | 2 tied for 2 slots |
| P0 | 0.65 | 0.2070 | 0.3887 | 0.8763 | 1 tied for 1 slot |
| P1 | 0.66 | 0.2102 | 0.4106 | 0.8726 | 3 tied for 2 slots |

Paired contrasts compare candidate minus baseline on the same 8,297 common variants (314
positives, 460 groups):
- whole-group bootstrap, 2,000 draws, seed `20260914`;
- 95% percentile intervals;
- no single-class draws and no refusals.

The point estimate is the observed difference, not the bootstrap mean.

| Contrast | P@100 | AP | AUROC |
|---|---|---|---|
| S1 − S0 (SpliceAI masking) | +0.020 [0.000, 0.062] | +0.017 [0.006, 0.028] | +0.011 [−0.007, 0.027] |
| P1 − P0 (Pangolin masking) | +0.010 [0.000, 0.040] | +0.022 [0.011, 0.034] | −0.004 [−0.019, 0.013] |
| P0 − S0 (unmasked tools, matched annotation) | +0.020 [−0.033, 0.083] | +0.093 [0.064, 0.124] | +0.073 [0.048, 0.096] |

**Reading the results.**
- **Masking within each tool:** masking raised average precision for both SpliceAI and
  Pangolin. It did not establish a change in AUROC or in P@100.
- **The two P@100 lower bounds of 0.000 include zero.** P@100 moves in steps of about 0.01,
  and many bootstrap draws have a difference of exactly zero: 22.5% for S1 − S0 and 44.4% for
  P1 − P0. No P@100 difference is established.
- **The P1 − P0 P@100 difference depends on tie order.** P1 has three scores tied at 0.54 for
  the last two review slots, and one of them is positive. Any label-independent tie order gives
  P1 a P@100 of 0.65 or 0.66, so a difference of +0.00 or +0.01. The reported +0.010 comes from
  the registered fixed tie-break permutation. The S1 − S0 and P0 − S0 P@100 values do not
  depend on tie order.
- **Unmasked tools under a matched annotation:** Pangolin exceeded SpliceAI in AP and AUROC.
  The P@100 difference includes zero.

## Limitations

- **Exploratory and unadjusted.** There are nine intervals, with no multiplicity adjustment,
  on outcomes that had been inspected before.
- **The matched annotation does not isolate network architecture.** P0 − S0 still includes
  differences in:
  - N-padding outside the transcript (SpliceAI only);
  - masking rules;
  - training data;
  - ensemble;
  - score definition.
- **Equal mask flags are not equal operations.** SpliceAI masks against the single nearest
  boundary. Pangolin masks every annotated site in the window.
- **The improvement rule does not apply.** The five-per-100 P@100 rule in the protocol
  applies only to future, prospectively registered, unseen cohorts. It is not applied to
  these retrospective contrasts, whether as "met" or "failed". The results say nothing about
  other models or datasets.
- **MFASS measures something else.** It assays exon recognition in a minigene, while both
  tools score genomic context.
- **The archived runs are context only.** The earlier SpliceAI run (bundled v24-derived table,
  8,194 scored) and Pangolin run (GTF gene spans, 8,301 scored) used different annotations and
  coverage, and incompletely pinned environments. They are not paired estimates. The matched
  P0 scores 4 fewer variants than the archived Pangolin run, exactly the span-rule losses
  registered in A6.

## Files

| Path | Contents |
|---|---|
| `S0/`, `S1/`, `P0/`, `P1/` | Per condition: the result JSON (configuration, code and weight identities, metrics, coverage, timing); `*.predictions.tsv` (id, group, label, score for all 8,324 rows; blank score means unscored); and `*.unscored.tsv` (unscored IDs and the tool's own reason) |
| `contrasts/` | The three paired comparison reports |
| `verification.json` | Pre-contrast verification: table bytes, IDs, labels, groups, checkpoint agreement, recomputed metrics and pair compatibility. The checkpoints themselves are not published; see Provenance. |
| `report.json`, `report.md` | The study report as generated. `report.md` is unchanged; `report.json` has the cohort reference base withheld in reason text (see Provenance). `report.md` was generated before review: it lists tie counts but does not explain their effect on P@100 or the 0.000 lower bounds. See "Reading the results". |
| `manifest-v1.json` | The frozen manifest: resources, code, weights, environments and exact commands |
| `logs/events.jsonl` | Step start and exit events, bound to the manifest hash |
| `reviews/claude-scientific-review.md` | Automated scientific review |
| `provenance.json`, `SHA256SUMS` | Path substitutions, the allele-withholding pattern, raw and public hashes, and hashes of unpublished files; checksums for every file |

## Provenance of public copies

Two text transformations were applied to the public copies. No number, ID or score changed.

- **Path substitution (9 files).** The four result JSONs, three contrast reports,
  `events.jsonl` and `manifest-v1.json` contained private absolute paths. Exact path prefixes
  were replaced by `$STUDY_DATA` (study data disk), `$REPO` (repository checkout) or
  `$MFASS_DATA` (local MFASS source data).
- **Cohort reference base withheld (4 files).** The MFASS cohort's reuse terms are unreported,
  so its alleles are not republished. Pangolin's message for the 23 reference mismatches quotes
  the cohort's reference base. In the P0 and P1 `unscored.tsv`, `report.json` and
  `verification.json`, that base is replaced by `<withheld>`; `provenance.json` records the
  exact pattern. The FASTA base, which is public reference data, is kept.

**Not published:**
- the four `checkpoint.tsv` files, because SpliceAI's raw output begins with the cohort's
  alternate allele;
- the step logs.

Their raw SHA-256 values are in `provenance.json`, and the raw originals are kept on the local
study disk. `verification.json` records that every prediction agreed with its checkpoint.

`provenance.json` records the raw and public SHA-256 of every published file. Applying the
recorded transformations to a raw original reproduces its public hash. The prediction tables
are byte-identical to the raw files. Their hashes equal the `output_tables` hashes in each
result JSON and the input hashes in each contrast report. The two Pangolin `unscored.tsv`
public hashes differ from the raw hashes recorded in the result JSONs because of the
withholding.

## Reproduction

Check the published outputs (no inference):

```sh
(cd benchmarks/mfass/results/matched-annotation-v1 && shasum -a 256 -c SHA256SUMS)
uv run pytest benchmarks/mfass/tests/test_matched_annotation_v1_bundle.py
# Recompute one contrast exactly from the published predictions (example: P0 - S0):
uv run python -m mfass.compare \
  --baseline benchmarks/mfass/results/matched-annotation-v1/S0/spliceai-1.3.1-gencode44-canonical-mask0.predictions.tsv \
  --candidate benchmarks/mfass/results/matched-annotation-v1/P0/pangolin-gencode44-canonical-maskFalse.predictions.tsv \
  --capacity 100 --draws 2000 --seed 20260914 --min-common 100
```

Rerunning `mfass.compare` on these published copies reproduced every field of all three
contrast reports exactly, apart from the file-path fields.

To rerun the study from scratch, you need the local MFASS cohort, the GENCODE 44 resources,
a clone of upstream Pangolin and the locked environments. The steps are those recorded in
`manifest-v1.json` under `steps`, run through `mfass.matched_study`:
1. `pangolin_patch prepare` and a non-editable install of the patched tree.
2. `matched_annotation prepare` and `variants`.
3. `matched_study freeze`.
4. `benchmarks/mfass/scripts/run_matched_study.sh`.

Expect about 14 h on a similar CPU. SpliceAI numerics under Keras 3 were checked against
upstream's published README examples before scoring. That check reproduced all 6 records
exactly (`docs/mfass-matched-study-evidence/preflight.json`).
