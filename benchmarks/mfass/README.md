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

`split-v2`, 8,324 held-out variants, 3.78% prevalence. Metrics are on each method's own scored subset;
see the paired comparisons below for like-for-like.

| Method | Family | Precision@100 | Recall@100 | AP | AUROC | Coverage | s/variant |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline-kmer-position | trivial baseline | 0.620 | 0.197 | 0.286 | 0.768 | 8324/8324 | 0.00002 |
| spliceai-1.3.1 | specialist | 0.640 | 0.208 | 0.299 | 0.806 | 8194/8324 | 0.53626 |
| pangolin (mask=False) | specialist | 0.650 | 0.207 | 0.389 | 0.876 | 8301/8324 | 1.64162 |

### Paired comparisons

Candidate minus reference, resampling whole groups, restricted to the variants both methods scored.
Bold means the 95% interval excludes zero.

| Comparison | Precision@100 | AP | AUROC |
|---|---|---|---|
| SpliceAI minus baseline | +0.011 [-0.090, +0.105] | +0.008 [-0.040, +0.056] | **+0.037 [+0.002, +0.075]** |
| Pangolin minus baseline | +0.027 [-0.054, +0.102] | **+0.101 [+0.061, +0.138]** | **+0.107 [+0.081, +0.134]** |
| Pangolin minus SpliceAI | +0.018 [-0.039, +0.076] | **+0.092 [+0.061, +0.122]** | **+0.070 [+0.043, +0.094]** |

**Every pair separates on AP and AUROC. No pair separates at precision@100.**

The ranking on global metrics is unambiguous: Pangolin, then SpliceAI, then the trivial baseline, with
every gap distinguishable. At a 100-variant review capacity none of the three can be told apart. At
3.78% prevalence the top 100 is a thin slice, and all three find broadly the same easy canonical
variants in it. The differences live across the rest of the ranking.

That divergence is the argument for reporting both, and for not choosing a tool on AUROC alone.

### By distance to the exon boundary

Canonical splice sites are largely solved. MFASS is mostly not canonical: 83.4% of its
splice-disrupting variants sit more than 2 bases from a boundary, independently reproducing the source
paper's ~83%. AUROC per band, on the 8,194 variants all three methods scored.

| Band | Variants | SDVs | Share of SDVs | baseline | SpliceAI | Pangolin |
|---|---:|---:|---:|---:|---:|---:|
| canonical, 0 to 2 | 443 | 41 | 13.3% | 0.825 | 0.902 | **0.925** |
| near, 3 to 10 | 1,671 | 77 | 25.0% | 0.745 | 0.798 | **0.857** |
| mid, 11 to 30 | 4,166 | 159 | 51.6% | 0.744 | 0.785 | **0.868** |
| distal, over 30 | 1,914 | 31 | 10.1% | *0.786* | 0.744 | **0.844** |

**Pangolin leads in every band, including the distal one where SpliceAI falls below the trivial
baseline** (0.744 against 0.786). SpliceAI's advantage is concentrated near the splice site and does
not survive past 30 bases; Pangolin's does.

For a laboratory triaging deep intronic and exonic candidates, which are 62% of the disrupting
variants here, that is the practical difference between the two specialists.

Per-band precision at capacity is coarse at these counts, so AUROC is the more stable per-band read.

### Cost and coverage

Pangolin costs about 3 times SpliceAI per variant (1.64s against 0.54s) and roughly 86,000 times the
trivial baseline. It also scores more variants: 8,301 against SpliceAI's 8,194, because it was pointed
at GENCODE v44 while SpliceAI used its bundled v24-derived table.

**That annotation difference is an open confound.** The SpliceAI figures here carry both a model
difference and an annotation difference from Pangolin. Re-running SpliceAI against a v44-derived
annotation is outstanding, and until it is done the SpliceAI-Pangolin gap should be read as an upper
bound on the model difference.

### Caveats that travel with these numbers

The baseline is **supervised** on this assay's training split. SpliceAI and Pangolin are **zero-shot**
here: neither saw MFASS outcomes. So the baseline comparison measures in-domain training against a
specialist prior, not the standalone quality of any tool.

MFASS measures exon recognition in a minigene construct. These are not predictions of splicing in
patient RNA.

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
- A declared improvement margin, written down before any candidate is scored
