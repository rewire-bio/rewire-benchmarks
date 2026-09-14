# mfass-v1

Does a genomic foundation model improve splice-variant prioritisation over the tools a diagnostic
laboratory already runs, measured against independent functional labels?

## Dataset

[Cheung et al., *Molecular Cell* 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6599603/). MFASS is a
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

## Results

On `split-v2`, restricted to the 8,194 variants both methods could score, across 454 independent
groups at 3.76% prevalence.

| Method | Family | Precision@100 | Recall@100 | AP | AUROC | Coverage | s/variant |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline-kmer-position | trivial baseline | 0.620 | 0.201 | 0.290 | 0.769 | 8324/8324 | 0.00003 |
| spliceai-1.3.1 | specialist | 0.640 | 0.208 | 0.299 | 0.806 | 8194/8324 | 0.537 |

### Paired comparison, SpliceAI minus baseline

Resampling whole groups, so the difference is paired and the interval respects the grouping.

| Metric | Delta | 95% interval | Distinguishable |
|---|---:|---|---|
| Precision@100 | +0.011 | [-0.090, +0.105] | no |
| Average precision | +0.008 | [-0.040, +0.056] | no |
| AUROC | +0.037 | [+0.002, +0.075] | yes |

**SpliceAI ranks better globally and is indistinguishable at the operating point a laboratory uses.**
AUROC separates the two; precision at a 100-variant review capacity does not, and neither does average
precision. This is the divergence that motivates leading on precision at capacity rather than AUROC.

Two things must be said alongside that, and they cut in opposite directions.

The baseline is **supervised**: it was trained on the training split of this same assay, so it has
in-domain label information SpliceAI never saw. SpliceAI is **zero-shot** here, having been trained on
GENCODE transcripts and never on MFASS outcomes. This is not evidence that SpliceAI is weak. It is
evidence that a simple model with in-domain training data reaches the same operating point as a strong
zero-shot specialist.

And SpliceAI costs about **28,000 times more compute per variant**, 0.537s against 0.00002s, while
scoring 130 fewer variants.

### What the split costs

Same features, same code, only the grouping rule changes:

| Split | Features | Precision@100 | AUROC |
|---|---|---:|---:|
| v1, exon only | position + 3-mers | 0.680 | 0.817 |
| v1, exon only | + phyloP, phastCons | 0.760 | 0.816 |
| v2, exon + gene | + phyloP, phastCons | 0.620 | 0.768 |

Conservation adds 8 points of precision@100. Grouping by gene as well as exon removes 14. **The honest
split costs more than the best feature gain**, which is the argument for publishing the grouping rule
beside every number.

## Not yet done

- Pangolin, configuration-matched against SpliceAI
- Pretrained encoders: DNABERT-2, NT-v2, Caduceus, SpliceBERT
- Canonical versus non-canonical subgroup breakdown
- A declared improvement margin, written down before any candidate is scored
