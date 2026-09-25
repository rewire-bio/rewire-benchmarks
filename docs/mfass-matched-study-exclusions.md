# MFASS matched study: exclusions and publication decision

Updated 25 September 2026. This addendum explains the 27 exclusions in the
[matched-annotation-v1 study](../benchmarks/mfass/results/matched-annotation-v1/README.md).
The archived results, prediction tables, registration and checksums remain unchanged.
No inference was rerun and no excluded record was assigned a score of zero.

## Evaluation population

All four configurations (SpliceAI S0/S1 and Pangolin P0/P1) score the **same 8,297
of 8,324 held-out variants**, including 314 of 315 positives and 460 of 463 groups.
Metrics and paired comparisons apply to those 8,297 scored variants, not the full
held-out set. The exclusions were detected in label-free preflight; this follow-up
explains their causes without selecting a different evaluation set.

| Exclusion | Count | Interpretation and current treatment |
|---|---:|---|
| hg19-to-hg38 mapping reverses orientation but genomic alleles retain the hg19 orientation | 23 | Retain as unscored in v1; report the upstream conversion issue and revisit in a separately versioned evaluation. |
| Outside the selected canonical transcript span | 4 | Retain as unscored under the registered canonical-transcript protocol; this is not evidence of a faulty variant. |

The [exclusion inventory](mfass-matched-study-exclusions/inventory.tsv) identifies
every excluded record without republishing assay sequences or alleles. The
[verification receipt](mfass-matched-study-exclusions/verification.json) binds the
shared scored/excluded IDs to the unchanged prediction tables and checks their
agreement with the original tool exclusion files.

## Assembly-orientation finding

The original MFASS liftOver script submits four-column BED records. The formatting
step joins the new coordinates without converting the genomic alleles for inverted
mappings. The affected records have unique, exact coordinate mappings in the
original chain. Their original reference bases and assay reference windows match
hg19; the full assay reference windows match GRCh38 after applying the mapping
orientation. Complementing both genomic alleles preserves the assayed single-base
change. This concerns assembly conversion, not the assay measurements.

The 23 held-out records comprise 21 from ENSE00001321140 (chain 107) and two from
ENSE00001002968 (chain 98). A cohort-wide investigation found 90 such records among
27,733 eligible variants: 23 test and 67 training. None of the 8,297 scored test
records has this detected defect; all passed the reference-window and allele checks.
Additional reference-context differences in 37 training records do not affect this
zero-shot specialist comparison, which does not fit on MFASS training labels.

Sources checked at MFASS revision `9a8e4f27106be52aeb11acad27f95f5cded663a8`:

- [Original liftOver script](https://github.com/KosuriLab/MFASS/blob/9a8e4f27106be52aeb11acad27f95f5cded663a8/scripts/snv/snv_ref_liftover.sh).
- [Formatting and coordinate join](https://github.com/KosuriLab/MFASS/blob/9a8e4f27106be52aeb11acad27f95f5cded663a8/scripts/snv/snv_format_ref.R#L70-L83).
- [Assembly mapping chain](https://github.com/KosuriLab/MFASS/blob/9a8e4f27106be52aeb11acad27f95f5cded663a8/ref/hg19ToHg38.over.chain).
- [Upstream report with verified reproduction](https://github.com/KosuriLab/MFASS/issues/1).

These findings were checked computationally by Codex. They have not received an
independent human review or author confirmation. Source hashes and retrieval dates
are retained in the [source receipt](mfass-matched-study-exclusions/sources.jsonl).
Do not repair arbitrary reference mismatches by complementing a base: validated
mapping orientation and sequence context are required.

## Canonical-transcript exclusions

The four records are ENSE00002361772_001, _004, _005 and _008 in ARHGEF3. Their
coordinates and assay windows agree with the reference. They fall outside the
selected GENCODE 44 canonical/MANE transcript ENST00000296315.8, whose span is
chr3:56,727,420–56,801,949 (one-based inclusive). Ten alternative transcript spans
cover each position. The exclusions therefore follow from the frozen transcript
selection, not an established source-data error. They are not part of the upstream
allele-orientation bug report.

Including these sites would require a different transcript protocol. Extending the
gene span while retaining only canonical exon boundaries would not constitute a
consistent all-transcript comparison.

## Publication decision and limits

Retain all 27 exclusions and publish the existing filtered comparison with explicit
coverage. Do not overwrite v1, alter its denominator to hide missing predictions,
or claim the upstream issue is fixed. Author feedback or corrected genomic inputs
can motivate a new version with fresh verification and metrics.

The study remains exploratory: test outcomes were inspected previously, the nine
intervals are unadjusted, and matched annotation does not isolate architecture.
No top-100 precision difference is established; Pangolin's masked P@100 difference
also depends on the registered tie order. The study's existing automated Claude
review checked numerical artifacts; it is distinct from this later Codex exclusion
investigation. Neither is independent human review.
