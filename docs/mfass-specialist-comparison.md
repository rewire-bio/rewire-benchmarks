# MFASS specialist comparison: confounds and next analysis

Reviewed 24 September 2026. Status: **protocol documented; the registered
annotation-matched sensitivity study was executed on 24 and 25 September 2026**
([results and automated review](../benchmarks/mfass/results/matched-annotation-v1/README.md)). This resolves the documentation work in
[issue #11](https://github.com/rewire-bio/rewire-benchmarks/issues/11); it does not
resolve the scientific confounds or establish a stronger model comparison.

The [claim audit](mfass-specialist-comparison-sources.csv) records source URLs,
revisions, SHA-256 hashes, locations and review outcomes. This is automated source
review, not human review or independent reproduction. The
[retrieval receipt](mfass-specialist-comparison-evidence/retrievals.json) also pins
unchanged historical artifacts. No model was run or rescored for this review.

## What the existing results establish

| Configuration | Annotation recorded | Mask | Scored / eligible |
|---|---|---|---:|
| SpliceAI 1.3.1 | Bundled `grch38`; upstream describes it as GENCODE v24 canonical | `0` (unmasked) | 8,194 / 8,324 |
| Pangolin | `gencode.v44.annotation.db`; release declared as GENCODE v44 | `False` (unmasked) | 8,301 / 8,324 |

These are fields from the archived results, not newly measured quantities
[audit A01–A02]. Neither archive records annotation, reference FASTA or weight
hashes. The historical Pangolin upstream revision is also unreported. A filename
or declared release does not authenticate the historical file or establish which
transcripts were selected [A04].

The specialists use genomic GRCh38 context and no MFASS training labels. The
corrected k-mer baseline uses assay sequences and MFASS training labels. DNABERT-2
is a frozen encoder, but its logistic head is supervised on those training labels.
“Frozen” therefore does not mean the complete evaluated pipeline is zero-shot.
These input and adaptation differences must remain visible [D01]. Exact overlap
with pretraining sequences remains unchecked; local execution cannot prove the
absence of contamination.

Use the [current v2 results and correction](../benchmarks/mfass/README.md#mfass-v2-results),
the [SDK and DNABERT-2 instructions](mfass.md), and the
[reviewed corrected revision](https://github.com/rewire-bio/rewire-benchmarks/tree/bee9133b83f3aedaf2bbb9013f1875515845607e/benchmarks/mfass).
The original three methods have no established top-100 winner. The corrected
SpliceAI-versus-baseline AUROC interval crosses zero. Do not reuse the superseded
v1 baseline conclusion or extend the three-method conclusion to DNABERT-2.

### Erratum: the Pangolin archive does not contain both masking settings

The archived [`pangolin-maskFalse.json`](../benchmarks/mfass/results/pangolin-maskFalse.json)
contains the sentence **“Both are run so the effect of the differing defaults is
measured rather than assumed.”** That execution claim is unsupported and withdrawn.
The committed archive at `fe72dddb767207f67dd5d310cbf9099a8eda1142` contains
`mask=False` predictions, not a `mask=True` counterpart [A03]. This is a statement
about available evidence, not proof that no unarchived experiment ever occurred.

The JSON remains unchanged with SHA-256
`bb0bb6732808699e54938233df1835dfc1f775f33ba7d6acd916e53d588a3c44`.
Its historical `benchmark: mfass-v1` identifier also remains intact; the unchanged
specialist predictions are reused in corrected v2 comparisons. New runner notes
must describe only the selected invocation. **The effect of Pangolin masking has
not been measured in this archive.**

## Annotation and masking are separate questions

[SpliceAI v1.3.1](https://github.com/Illumina/SpliceAI/tree/b3c7f17b4137cb32b30c56c064f8b90b9b8f38d0)
resolves exact `grch37`/`grch38` strings to packaged annotation files; other strings
are custom paths. Its CLI allows masking values `0` and `1`, defaulting to `0`
[S01–S03].

The reviewed [Pangolin implementation](https://github.com/tkzeng/Pangolin/tree/5cf94b8db938c658391b4305cd7ce33297d44ff7)
defaults to `mask=True`. Its database builder normally selects
`Ensembl_canonical` transcript/exon features [P01–P03]. That reviewed code revision
must not be presented as the unrecorded version used by the historical run.

Both tools describe masking as suppressing annotated gains and unannotated losses.
However, their implementations differ: SpliceAI selects extrema before masking
and uses a nearest exon boundary; Pangolin masks position arrays before selecting
reported extrema [S05, P01]. Equal boolean settings do not make these operations
identical. Annotation also affects gene eligibility and sequence handling, even
when masking is off. Matching a release label alone is insufficient.

The reviewed Pangolin code also reuses prediction arrays across overlapping genes
on the same strand and masks them in place. A
[synthetic control-flow check](mfass-specialist-comparison-evidence/masking-order-check.json)
of its original `process_variant` function produced extracted scores of 0.7 and
0.8 when only gene iteration order changed [P04]. Reference retrieval, gene lookup
and score generation were mocked; these are synthetic values, not benchmark
results. This demonstrates an unresolved implementation problem in the pinned
reviewed version. **Do not launch the proposed masked comparison unchanged.**
A separately reviewed fix or version change and renewed registration are required;
this documentation change does not patch the model.

## Safeguards for new runner outputs

Both specialist runners accept an optional `--annotation-release`. A supplied
label is recorded as **declared**; omitted, blank or `unreported` values remain
**unreported**. A filename never establishes an annotation release. New records
include SHA-256 hashes of the annotation and reference bytes actually supplied,
with `artifact_identity_status` explicitly separating local identity from upstream
verification. These hashes cannot repair missing historical provenance.

SpliceAI's exact `grch38` token still selects its bundled table. Use `./grch38`
(or another explicit path) for a custom file with that name. The runner resolves
the path before hashing and passes the same absolute file to the annotation
reader. It accepts only mask `0` or `1`; Pangolin retains `True`/`False`.

Hashing has its own `hash_reference_and_annotation` timing field and is included
in new total-per-variant timing. Cohort preparation remains outside that timing
scope. Old timings are unchanged. Newly generated notes describe only the chosen
masking setting, never imply that another condition ran, and retain annotation and
input-context limitations.

## Proposed annotation-matched sensitivity study

**Executed as registered on 24 and 25 September 2026.** The outputs, contrasts, limitations
and automated scientific review are in
[`benchmarks/mfass/results/matched-annotation-v1`](../benchmarks/mfass/results/matched-annotation-v1/README.md).
The text below is the protocol as written before execution. Preparation, resource acquisition and inference belong to a
[execution issue #19](https://github.com/rewire-bio/rewire-benchmarks/issues/19). Freeze its input manifest and analysis plan before
viewing any new condition's scores. The existing MFASS test outcomes have already
been inspected, so this remains exploratory. The execution plan and its amendments
to this protocol, including the Pangolin masking patch, are registered in
[the study registration](mfass-matched-study-registration.md).

### Shared resources and preparation gates

1. Use the canonical validated MFASS v2 cohort and `split-v2.tsv`: 8,324 held-out
   variants in 463 exon/gene groups [C01]. Require the cohort and split hashes in
   the [SDK guide](mfass.md#access-and-preparation); do not create a new split or
   tune against its labels.
2. Obtain the [GENCODE v44 primary-assembly GTF](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.primary_assembly.annotation.gtf.gz)
   and [GRCh38 primary-assembly FASTA](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz)
   listed on the [official release page](https://www.gencodegenes.org/human/release_44.html)
   [G01]. Pin compressed and decompressed hashes during future preparation,
   before inference; these large resources have not been downloaded here.
3. Select transcripts tagged `Ensembl_canonical` once from that GTF. Keep an
   explicit gene/transcript ID manifest; retain no gene without a selected
   transcript. If a gene has multiple selected transcripts, stop preparation
   for review rather than silently choosing or merging them. Record exclusions.
4. Generate both formats from that same selection: a SpliceAI custom table and
   Pangolin gffutils database. Use stable gene IDs for the crosswalk. Set the
   database gene span to the selected transcript span so gene eligibility agrees
   with SpliceAI's transcript span; record this departure from the default
   database builder. Preserve selected exon coordinates and strand.
5. Compare round-tripped gene/transcript IDs, chromosome names, spans, exon
   boundaries and strand. SpliceAI expects zero-based starts and retains end
   coordinates; its reader adds one to starts [S04]. Validate positive- and
   negative-strand boundary fixtures and overlaps before marking annotations
   matched. Reject missing contigs or reference-allele mismatches; report affected
   variants explicitly rather than correcting alleles from outcomes.
6. Pin both upstream implementations, every ensemble weight, the environment,
   conversion code and generated annotations. Use the SpliceAI v1.3.1 and Pangolin
   source revisions reviewed above as the planned code baseline. Verify the actual
   installed bytes and acquire weight hashes before execution. If resources cannot
   be authenticated or conversion checks fail, stop and record a blocked run.
7. Require positive- and negative-strand overlapping-gene fixtures to give the
   same per-gene scores and extracted variant score regardless of gene iteration
   order. The reviewed Pangolin pin fails a minimal synthetic case [P04]. Stop
   the four-condition study until a separately reviewed fix/version passes this
   gate, then pin that implementation and register the amended plan before
   viewing its new scores. Do not silently copy arrays, reorder genes or discard
   overlapping genes as part of an unrecorded workaround.

### Four fixed conditions

| Condition | Annotation | Mask | Scoring distance | Per-variant score |
|---|---|---|---:|---|
| S0 | Matched v44 SpliceAI table | `0` | 50 bp | Maximum AG/AL/DG/DL delta over returned genes |
| S1 | Same table and weights as S0 | `1` | 50 bp | Same extraction as S0 |
| P0 | Matched v44 Pangolin database | `False` | 50 bp | Maximum absolute reported usage change over genes/sites |
| P1 | Same database and weights as P0 | `True` | 50 bp | Same extraction as P0 |

Use the existing five-model SpliceAI and twelve-model Pangolin ensembles. Keep
all non-mask settings fixed within each pair, including context, CPU precision,
patches and output parsing. Retain upstream output rounding consistently; report
its tie implications. Do not adjust thresholds, retrain or add a calibrator.
Write each condition and its manifest to a new output directory, never an archive.

### Three contrasts and reporting

Report **S1 − S0**, **P1 − P0**, and **P0 − S0**. The first two estimate masking
sensitivity within a model; the third compares unmasked tools under matched
annotations. None isolates network architecture from all implementation differences.

For each contrast, use only its commonly scored variants and identical canonical
labels/groups. Publish their ID-manifest hash, positive count and group count.
Report each condition's coverage and failure reasons against **8,324**, plus
pairwise common, baseline-only and candidate-only counts. Do not compare deltas
from different paired populations as if they shared a denominator. Missing scores
are not negative predictions.

Report P@100, average precision and AUROC, using the existing paired whole-group
bootstrap with **2,000 draws**, **seed `20260914`** and 95% percentile intervals
[C02–C04]. Require at least 100 common scored variants and both outcome classes.
The observed P@100 uses 100 entries; bootstrap list length varies, so its capacity
scales to preserve the review fraction. Report realised capacities, skipped draws
and the observed delta, not the bootstrap mean as the effect estimate. Retain the
existing refusal when more than 5% of draws contain one class. Report a refused
interval rather than changing the method to obtain one.

Intervals are exploratory and unadjusted across these contrasts/metrics. Do not
use them to declare a confirmatory winner. Historical default-annotation runs
may provide context but are not a paired estimate of annotation alone: their
reference, weights and environment are incompletely pinned.

## Improvement rule for a future unseen evaluation

A prospectively fixed primary candidate/comparator comparison meets the research
improvement rule only when both conditions hold:

- Observed **P@100 gain ≥ 0.05**, meaning at least **five additional positives in
  100 reviewed variants** on the common population.
- The paired **95% interval has a lower bound strictly above zero**.

Before viewing outcomes or model scores, register the independent unseen cohort,
its grouping rule, the primary comparator and candidate, exact configurations,
scoring/coverage exclusions, tie handling, sample-size rationale and analysis
procedure. Use the paired group-bootstrap procedure above where its assumptions
apply. At least 100 commonly scored variants and both classes are required;
100 variants alone does not establish adequate statistical power. Report coverage
separately and do not replace the primary comparison after seeing results.

The five-per-100 margin is an agreed research decision threshold, not an
empirically established clinical-benefit threshold. It is **not prespecified for
existing MFASS results**. The planned sensitivity study cannot be relabelled a
confirmatory test of this rule. Multiple future candidate comparisons require a
separately registered multiplicity policy before outcomes are inspected.

## Completion boundary

Closing the documentation issue records corrected execution claims, an auditable
study specification and safeguards for future provenance. Keep [execution issue #19](https://github.com/rewire-bio/rewire-benchmarks/issues/19) open until matched annotation validation, four-condition
execution and reviewed analysis receipts exist. No stronger model claim follows
from this documentation change.
