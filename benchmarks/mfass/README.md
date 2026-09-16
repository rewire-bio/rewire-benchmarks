# MFASS benchmark

> **Correction, September 2026:** The published `mfass-v1` baseline used the raw
> `sequence` field with assay-coordinate `rel_position`. In 7,770 of the 27,733
> eligible variants, that field is the reverse complement of `original_seq`, so
> its 21bp 3-mer window was centred at the wrong base. The original v1 JSON and
> prediction artifacts remain as an auditable historical snapshot. **Do not use
> the v1 baseline comparisons or the v1 split-cost claims as current
> evidence.** `mfass-v2` rebuilds a validated assay-oriented reference/mutant
> pair from `natural_seq`/`original_seq` and reruns the baseline and comparisons.

Does a genomic foundation model improve splice-variant prioritisation over the tools a diagnostic
laboratory already runs, measured against independent functional labels?

## Dataset

[Chong et al., *Molecular Cell* 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6599603/). MFASS is a
multiplexed functional assay of splicing using Sort-seq: each variant sits in a minigene reporter
wired to GFP, cells are sorted by fluorescence, and sequencing quantifies which variants disrupted
exon recognition.

Chosen for four reasons:

- **Functional labels, not clinical ones.** No ClinVar circularity and no worry that a model saw the
  clinical assertion during pretraining.
- **3.79% prevalence.** A realistic prioritisation problem, so precision at a bounded review capacity
  is the natural primary metric.
- **Only ~17% of the functional rare variation found is in canonical splice sites.** Canonical sites
  are largely solved; the rest is where a sequence model would have to earn its place.
- **Self-contained 170bp windows.** The minigene construct is the context, so no genome retrieval is
  needed anywhere and the benchmark reproduces from one downloaded file.

Scope limit worth stating plainly: this measures exon recognition in an artificial construct, not
splicing in patient RNA.

> **Author name.** The first author is Rockie Chong. Cell Press's own "In Brief" blurb inside the
> paper says "Cheung et al.", contradicting its byline, and that error propagated into the reference
> lists of both the Pangolin and CADD-Splice papers. Crossref and PubMed both give Chong.

## Cohort reconciliation

The source table has 32,669 rows. The paper reports 27,733 variants and 1,050 splice-disrupting
variants. The filter reproducing those exactly:

```
category == "mutant"  AND  strong_lof != "NA"
```

| Quantity | Observed | Published |
|---|---:|---:|
| Variants | 27,733 | 27,733 |
| Splice-disrupting | 1,050 | 1,050 |
| Prevalence | 3.786% | 3.8% |
| Exons in the mutant set | 2,198 | 2,198 |

`build_dataset` asserts all three and refuses to write on drift. One wrinkle: the paper's 2,198 exons
is counted before the `strong_lof` filter, so the evaluation cohort itself spans 2,185.

The source `original_seq` is the assayed mutant and `natural_seq` its reference.
For every eligible variant, they are both 170bp and differ at exactly
`rel_position - 1`, with alleles checked against the recorded strand. The raw
`sequence` equals `original_seq` for 19,963 rows and its reverse complement for
7,770. The v2 cohort keeps the legacy field for provenance but exposes validated
`reference_sequence` and `mutant_sequence` for all sequence-based methods.

## Split

Grouping unit is the connected component of relations that must not cross the boundary.

`split-v1` groups by exon alone: many variants share an exon and the same minigene context. 8,320
held-out variants in 662 independent groups.

`split-v2` adds the gene, and is the canonical split. The 2,185 exons map to 1,615 genes, and 27 exons
map to more than one gene, so variants in two exons of one gene are not independent.

| | v1 (exon) | v2 (exon + gene) |
|---|---:|---:|
| Independent groups | 2,185 | **1,590** |
| Naive concatenated key would claim | 2,185 | 2,267 |
| Overstatement | none | **43%** |
| Largest single group | 37 | 354 |
| Held-out groups | 662 | 463 |

The naive concatenated key claims *more* units under two keys than the real count under one, which is
the failure this machinery exists to prevent.

Groups are assigned whole. Prevalence is matched between arms by the best of 200 seeded draws on
closeness to the cohort rate, fixed before any model runs and never using a prediction.

## Reproduce v2 on a local machine

The two source tables are downloaded from the authors' public repository and kept
outside Git. Their SHA-256 hashes in this run are
`a637ca0e307e66ff48811ec7efa22b9ce453bc7883b04f0cacb867f7283132d8`
(`snv_data_clean.txt`) and
`71a857fe647c4e68acbb41ca61e959c47e1176de89b1442bd6ca1772aa60d5a1`
(`snv_func_annot.txt`). The canonical checked-in split is
`benchmarks/mfass/splits/split-v2.tsv`; its SHA-256 is
`999ebcb7e63a5c5eaa8780fa468e59ac1f934260ad50102814174c396317f052`.

```bash
mkdir -p benchmarks/mfass/data
curl -L -o benchmarks/mfass/data/snv_data_clean.txt \
  https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_data_clean.txt
curl -L -o benchmarks/mfass/data/snv_func_annot.txt \
  https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_func_annot.txt
uv sync --package mfass --extra dnabert2-pilot
uv run --package mfass --extra dnabert2-pilot mfass-build --check
uv run --package mfass --extra dnabert2-pilot mfass-build
uv run --package mfass --extra dnabert2-pilot mfass-baseline
uv run --package mfass --extra dnabert2-pilot mfass-dnabert2-pilot
# Only after the pilot confirms <=12 hours projected and >=1 GiB free disk:
uv run --package mfass --extra dnabert2-pilot mfass-dnabert2
```

No paid API or cloud compute is used. The pilot deterministically samples 16
held-out reference/mutant pairs from each legacy orientation stratum, embeds
both 170bp sequences, and records token lengths, runtime, memory, disk reserve,
and failures. It does **not** compute accuracy. The verified local pilot embedded
all 32 pairs in 0.690 seconds, peaked at about 1.25 GB process RSS, and left
3.59 GB disk free; its refreshed artifact checks the loaded code and weight
SHA-256 against the pinned repository revision. DNABERT-2's released checkpoint
has untrained pooler weights; the full protocol therefore uses the attention-mask
mean of `last_hidden_state`, concatenates the reference vector with the
mutant-minus-reference vector, and fits a fixed balanced L2 logistic head
(`C=0.1`) solely on the 19,409 training variants. It scores the full 8,324-row
held-out arm once, with no tuning against it. The source/revision and runtime
metadata are recorded with the result. If any full-score gate fails, no partial
accuracy artifact is written.

The [run provenance manifest](provenance/mfass-v2-local-dnabert2.json)
binds the pinned checkpoint and source hashes, the exact runner source,
the original pilot used for the full run, complete predictions, trained
head, result JSON, and paired comparison. The checkpoint comes from the
[official DNABERT-2 repository](https://huggingface.co/zhihan1996/DNABERT-2-117M)
and is never redistributed here.

## MFASS-v2 results

The canonical split has 8,324 held-out variants (315 disrupting) in 463 groups.
Each method's point metrics use its own scored subset; paired deltas use only
variants scored by both methods. SpliceAI and Pangolin predictions came from
the unchanged genomic-context specialist runs and were not silently relabelled
as new assay-sequence runs.

| Method | Family/protocol | P@100 | AP | AUROC | Coverage |
|---|---|---:|---:|---:|---:|
| Corrected k-mer/position baseline | supervised trivial baseline | 0.610 | 0.286 | 0.778 | 8,324/8,324 |
| SpliceAI 1.3.1 | zero-shot specialist, unchanged | 0.640 | 0.299 | 0.806 | 8,194/8,324 |
| Pangolin, mask=False | zero-shot specialist, unchanged | 0.650 | 0.389 | 0.876 | 8,301/8,324 |
| DNABERT-2 117M, frozen pair + logistic head | supervised pretrained encoder | 0.030 | 0.045 | 0.550 | 8,324/8,324 |

### Paired comparisons

Candidate minus the **corrected** baseline on the common subset, with 95%
intervals from 2,000 resamples of whole connected exon/gene groups. The
precision@100 intervals account for a fixed review-list fraction under
group resampling; no method's high-level ranking is inferred from P@100 alone.

| Candidate | Common variants | P@100 delta [95% interval] | AP delta [95% interval] | AUROC delta [95% interval] |
|---|---:|---|---|---|
| SpliceAI | 8,194 | +0.030 [-0.068, +0.107] | +0.009 [-0.039, +0.061] | +0.028 [-0.004, +0.064] |
| Pangolin | 8,301 | +0.040 [-0.041, +0.120] | +0.102 [+0.063, +0.141] | +0.098 [+0.073, +0.125] |
| DNABERT-2 frozen pair + head | 8,324 | -0.580 [-0.713, -0.464] | -0.241 [-0.304, -0.182] | -0.228 [-0.300, -0.160] |

The v1 statement that SpliceAI clearly exceeded the baseline on AUROC is
**withdrawn**: after correcting the baseline, its paired interval crosses
zero. Pangolin still exceeds the baseline on AP and AUROC, though its P@100
interval crosses zero. The tested DNABERT-2 frozen-pair protocol is worse than
the trivial baseline on all three measures. This evaluates **one fixed
representation and head**, not the model's best achievable result or a
general claim about genomic foundation models.

### Boundary distance and runtime

On the 8,194 variants all three original methods could score, the corrected
baseline's AUROC by boundary-distance band is 0.850 (0–2 bases), 0.735
(3–10), 0.764 (11–30), and 0.756 (over 30). The specialist numbers and
shared-subset denominators are preserved in
`results/subgroups-canonical-v2.json`. The older boundary table with a
0.786 distal baseline value is superseded.

The corrected baseline took 1.399 seconds end to end to load, validate,
featurise, fit, and score this cohort on the M4 Mac mini. The local DNABERT-2
run took 618.239 seconds, including loading, embedding every training and
test reference/mutant pair, fitting its head, and scoring held-out rows.
Specialist runtimes in their original artifacts cover held-out inference and
annotation/model loading; these scopes differ, so per-variant numbers should
be read with their timing fields.

### Limits

MFASS assays exon recognition in an artificial minigene, not splicing in
patient RNA. The baseline and DNABERT-2 logistic head are supervised on
MFASS train labels; SpliceAI and Pangolin are zero-shot on the assay. Both
specialists use genomic context rather than the assay construct. They also
use different annotation releases (SpliceAI's bundled v24-derived table
versus Pangolin's GENCODE v44), leaving a model-versus-annotation confound.
The exact pretraining sequence overlap for DNABERT-2 has not been checked.

## Archived mfass-v1 snapshot

The archived `results/baseline-kmer-position.json` and its predictions
remain unchanged. They reported P@100 0.620 and AUROC 0.768 from the
mis-centred k-mer windows; they must not be mixed with v2 comparisons.
The earlier exon-only split-cost experiment also used that windowing code
and needs a corrected rerun before any split-effect claim is reused.

## Next experiments

- Configuration-match SpliceAI and Pangolin annotations before attributing
  their gap to models.
- Predeclare further frozen or fine-tuned foundation-model protocols before
  viewing their held-out scores; this run is not a tuning set.
- Check exact sequence/exon overlap in pretrained corpora where possible.
