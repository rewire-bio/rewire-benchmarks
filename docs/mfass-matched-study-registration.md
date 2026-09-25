# MFASS annotation-matched specialist study: registration (issue #19)

Written 24 September 2026, before any S0, S1, P0 or P1 score exists. This document
registers the execution plan and the amendments to the
[specialist comparison protocol](mfass-specialist-comparison.md). The frozen run manifest
records this file's SHA-256, and the orchestrator refuses to run a step if the file has
changed since.

**Review status.** The Pangolin patch, annotation build, runners and orchestration were
reviewed by automated Claude agents. This included source review, synthetic control-flow
tests, label-free data checks, and an upstream example reproduction for SpliceAI. **No
human has reviewed them.** The existing MFASS test outcomes have already been inspected in
earlier work, so this study is exploratory. It is not a confirmatory test of the
improvement rule.

## Question and fixed conditions

The study measures how much masking changes each specialist's ranking when the annotation
is matched between the two tools. It also compares the unmasked tools under that shared
annotation.

| Condition | Tool and code | Annotation | Mask | Distance | Per-variant score |
|---|---|---|---|---:|---|
| S0 | SpliceAI 1.3.1, upstream `b3c7f17` bytes, 5 models | matched v44 table | `0` | 50 | max AG/AL/DG/DL delta over returned genes |
| S1 | as S0 | as S0 | `1` | 50 | as S0 |
| P0 | Pangolin `5cf94b8` with patch `pangolin-5cf94b8-mask-per-gene-1`, 12 models | matched v44 gffutils DB | `False` | 50 | max absolute reported change over genes and sites |
| P1 | as P0 | as P0 | `True` | 50 | as P0 |

Contrasts are candidate minus baseline: **S1 − S0** and **P1 − P0** (masking within a
tool), and **P0 − S0** (unmasked tools under a matched annotation). None of them isolates
network architecture from all implementation differences.

## Inputs

- **Cohort and split:** the canonical validated MFASS v2 cohort (SHA-256 `389702ff…`) and
  `split-v2.tsv` (`999ebcb7…`). The held-out arm has 8,324 variants in 463 groups. There is
  no new split and no tuning.
- **Reference:** GENCODE 44 `GRCh38.primary_assembly.genome.fa.gz`, whose MD5 matches the
  official `MD5SUMS`. Its decompressed SHA-256 is `e49b92b3…`.
- **Annotation:** GENCODE 44 primary-assembly GTF, MD5 checked against `MD5SUMS`, SHA-256
  `31ae5d8f…`.
  - One `Ensembl_canonical` transcript per gene, for all 62,754 genes. None were ambiguous
    and none were excluded.
  - Both formats are built from that single selection.
  - The cross-check of every gene's IDs, contig, strand, span, transcript and exon
    boundaries passed with 0 mismatches.
- **Exact hashes:** code, weights, environments, resources and runner files are listed in
  the frozen manifest.

## Amendments to the documented protocol

- **A1. Pangolin masking patch (required by protocol gate 7).** Upstream `process_variant`
  masks per-strand arrays in place, so a gene's masked scores depend on which same-strand
  genes came before it (upstream issue #29).
  - The patch gives each gene its own copy of the arrays. It is equivalent to unmerged
    upstream PR #30.
  - P0 and P1 both use the patched code. Unmasked output is byte-identical to upstream: this
    holds in the mocked tests and in a real-inference probe on 60 synthetic non-MFASS SNVs.
  - The patch and patched tree are GPL-3.0.
- **A2. Code and weight identity.** Every run checks the installed files against pinned
  SHA-256 values and upstream Git blob IDs:
  - SpliceAI: `utils.py`, `__main__.py`, `__init__.py` and the five `.h5` models.
  - SpliceAI wheel-only files `__clean_main__.py` and `normalise_chrom.py`, by SHA-256 only.
  - Pangolin: `model.py`, `__init__.py` and the 12 ensemble weights, plus the patched
    `pangolin.py`.

  A mismatch stops the run.
- **A3. Runtime compatibility evidence.** In the study environment (TensorFlow 2.21.0,
  Keras 3.15.1), the unmodified SpliceAI 1.3.1 CLI reproduced all 6 published README example
  records on GRCh37 chromosomes 2 and 19, including `T|RYR1|0.00|0.00|0.91|0.08|-28|-46|-2|-31`.
  The runner's `one_hot_encode` replacement gave byte-identical output. Pangolin has no GRCh38
  upstream example; its evidence is the byte-identical unmasked probe in A1.
- **A4. Thread configuration.** Steps run one at a time.
  - TensorFlow: intra-op 5, inter-op 1.
  - Torch: intra-op 6, inter-op 1.
  - `OMP_NUM_THREADS` matches the intra-op count.

  Thread counts can change floating-point reduction order, so they are fixed within each
  pair and recorded in every result. The historical Pangolin run used 10 threads.
- **A5. Biotype and contig policy.** All gene biotypes with a canonical transcript are kept,
  as in the upstream Pangolin builder: 20,070 protein-coding among 62,754 genes, on 47 contigs.
  This includes lncRNA and pseudogenes, which increases same-strand overlaps.
- **A6. Span rule consequences.** Gene span is the canonical transcript span in both formats.
  This narrowed 16,710 GTF gene rows.
  - Compared with GTF gene spans, 185 test variants lose at least one gene.
  - 4 test variants lose every gene and so become unscoreable.
  - At the issue #29 locus, `ENSG00000256966.6` is no longer eligible.
  - Matched Pangolin coverage will therefore be lower than the historical run (8,301). This
    is a known coverage difference, not a regression.
- **A7. Expected exclusions, computed without labels.** The 4 variants with no gene and the
  23 reference-allele mismatches mean both tools can score at most **8,297** variants. The
  23 mismatches cluster in two exons. SpliceAI's and Pangolin's reference-case rules agree
  for this FASTA and cohort.
  - Gene sets agree between the tools for all 8,324 variants.
  - 344 variants have at least two same-strand genes.
  - Pangolin omits a variant on a gene's lowest coordinate; SpliceAI includes it. This
    affects 0 test variants. It is reported, not corrected.

  Alleles and coordinates are not corrected.
- **A8. Failure policy.** Only the tools' own expected skips become unscored rows:
  - no coordinate;
  - no annotated gene;
  - SpliceAI's logged skips;
  - Pangolin's `-1` skip;
  - nonfinite model scores.

  Any other error stops the run without logging the variant, so a resumed run retries it.
- **A9. Resume and binding.** Each run logs every variant to a fingerprinted, locked,
  append-only checkpoint, and a resumed run must match the fingerprint exactly. Result
  tables are written first and the result JSON last, and the JSON records the tables'
  SHA-256. Before any contrast, verification checks each condition:
  - table bytes;
  - canonical IDs, labels and groups, in order;
  - checkpoint agreement;
  - recomputed coverage, group count and metrics;
  - that the pair differs only in its mask.
- **A10. Timing.** Scoring time sums per-variant time over checkpoint segments. Hashing and
  load times come from the final segment. Manifest hashing is outside all recorded timing.
- **A11. Reporting additions.**
  - Each contrast reports the SHA-256 of both prediction files, the common-ID hash and the
    common positive count.
  - Each condition and contrast reports how many scores tie at the 100th-ranked value after
    upstream 2-decimal rounding.

## Differences between the tools that remain by design

- SpliceAI pads bases outside the transcript with N; Pangolin does not. The span choice
  changes S0/S1 inputs, and P0 − S0 includes this difference.
- SpliceAI masks only when an extremum falls on the single nearest annotated boundary.
  Pangolin masks every annotated position in the window before taking extrema, and zeroes
  every loss when the window has no annotated site. Equal mask flags do not mean equal
  operations.
- Both tools treat transcript start and end as annotated sites.

## Analysis (unchanged from the protocol)

- **Common scored population:** each contrast uses only the variants scored by both of its
  conditions, with identical canonical labels and groups. At least 100 common variants and
  both outcome classes are required.
- **Metrics:** P@100, sklearn average precision and AUROC.
- **Intervals:** paired whole-group bootstrap with 2,000 draws, seed `20260914` and 95%
  percentile intervals. The bootstrap capacity scales with list length. The interval is
  refused when more than 5% of draws are single-class.
- **Reporting:**
  - the observed delta, not the bootstrap mean;
  - realised capacities, skipped draws and any refusals;
  - each condition's coverage and unscored reasons against 8,324;
  - common, baseline-only and candidate-only counts for each contrast.
- **Interpretation:** missing scores are coverage gaps, not negative predictions. Intervals
  are exploratory and unadjusted.

## Deviations

A change to any frozen input, code or setting after freezing stops execution. It then
requires a documented amendment and a new manifest in a new output directory. No condition
is rerun to change its scores. A condition that cannot complete is reported as blocked.
Historical archived results are not modified.
