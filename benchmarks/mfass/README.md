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

Grouping unit is the connected component of relations that must not cross the boundary. For v1 the
mandatory relation is the exon, since many variants share an exon and the same minigene context.

`split-v1`: 8,320 held-out variants in **662 independent groups**, both arms at 3.786% prevalence
against 19,413 training variants in 1,523 groups.

Groups are assigned whole. Prevalence is matched between arms by the best of 200 seeded draws on
closeness to the cohort rate, fixed before any model runs and never using a prediction.

## Results

| Method | Family | Precision@100 | Recall@100 | AP | AUROC | Coverage |
|---|---|---:|---:|---:|---:|---:|
| baseline-kmer-position | trivial baseline | 0.680 | 0.216 | 0.346 | 0.817 | 8320/8320 |

## Not yet done

- SpliceAI and Pangolin, configuration-matched
- Pretrained encoders: DNABERT-2, NT-v2, Caduceus, SpliceBERT
- Exon-to-gene mapping so the split can group on gene as well as exon
- Canonical versus non-canonical subgroup breakdown
- A declared improvement margin, written down before any candidate is scored
