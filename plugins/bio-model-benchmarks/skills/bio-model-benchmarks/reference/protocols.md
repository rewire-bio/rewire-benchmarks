# Protocols

The eight protocol IDs below are what `rewirebench protocols` prints for v0.5.0.
Run `rewirebench inspect ID` for the exact datasets, inputs, metrics and fitting
rules, and `rewirebench baselines ID` for the registered baselines. Docs links point at
`main`; in a checkout of this repository read `docs/` directly.

Docs: https://github.com/rewire-bio/rewire-benchmarks/blob/main/docs/

## Summary

| Protocol ID | Modality | Docs | Extra | Fitting | Registered baselines (v0.5.0) | KG records |
| --- | --- | --- | --- | --- | --- | --- |
| `mfass-v2` | DNA splice variants | `mfass.md` | none | `fit` on train | `training-prior-v1`, `mfass-kmer-position-v2` | `id:rewire-mfass-v2` (results, recipes) |
| `mfass-v2-frozen-encoder` | DNA, paired embeddings | `mfass.md` | `dnabert2` for the DNABERT-2 example | protocol-owned logistic head | `constant-paired-probe-v1`, `paired-composition-probe-v1` | `id:rewire-mfass-v2` (recipe `mfass-v2-dnabert2`) |
| `proteingym-v1.3-dms-substitutions` | Protein DMS | `proteingym.md` | `esm` for the ESM-2 example | zero-shot, `fit` refused | `seeded-random-v1`; conventional blocked | `id:rewire-proteingym-v1-3-dms-substitutions` (recipes), `id:rewire-protocol-proteingym-amfr-v13` (AMFR results) |
| `tdc-admet-group-v1` | Small molecules (ADMET) | `tdc-admet.md` | none (`PyTDC` to download) | `fit` on train | `training-control-v1`, `smiles-ngram-linear-v1` | `id:discovery-benchmark-tdc-molecular-tasks` (recipes) |
| `genomic-benchmarks-v2` | DNA classification | `genomic-benchmarks.md` | none (`genomic-benchmarks` package to download) | `fit` on train | `training-prior-v1`, `dna-ngram-logistic-v1` | `id:discovery-benchmark-genomic-benchmarks` (recipes) |
| `flip2-fitness-v1` | Protein fitness | `flip2.md` | `sequence` | `fit`, `fit_with_validation`, or embeddings with a fixed ridge head | `training-mean-v1`, `protein-composition-probe-v1` | `id:discovery-benchmark-flip2` (recipes), `id:rewire-protocol-flip2-rhomax-by-wild-type-v1` (results) |
| `dart-eval-task1-zero-shot-v1` | DNA regulatory elements | `dart-eval.md` | `sequence` | zero-shot, `fit` refused | `seeded-random-v1`; conventional blocked | `id:discovery-benchmark-dart-eval` (recipes) |
| `mrnabench-sample-mrl-v1` | RNA translation (MRL) | `mrnabench.md` | `sequence` | `fit`, or embeddings with a fixed RidgeCV head | `training-mean-v1`, `rna-composition-probe-v1` | `id:discovery-benchmark-mrnabench` (recipes), `id:rewire-protocol-mrnabench-designed-mrl-v1` (results) |

"Blocked" conventional baselines are listed with a gap reason and are not run.
On `main` (after v0.5.0) two artifact-backed references exist,
`proteingym-evcouplings-independent-v1` and `dart-h12core-fimo-hit-count-v1`; they need
external artifacts through `--baseline-options` and have no released measurement.
See `conventional-references.md`.

Other benchmarks in the KG, such as BEND, GUE, PerturBench or NABench, have no
`rewirebench` protocol. Their recipes (`*-official`) describe the upstream project's own
commands. Rewire has not executed them, and this skill does not cover them.

## Data access, licence and the smallest real run

Never accept a licence or create an account on the user's behalf. Ask first.

### mfass-v2 and mfass-v2-frozen-encoder

- Data: two public KosuriLab tables pinned at `9a8e4f27106be52aeb11acad27f95f5cded663a8`,
  about 80 MB together. No account. No declared reuse licence: the user must assess the terms.
- Preparation checks both SHA-256 hashes, rebuilds the 27,733-variant cohort and uses the
  packaged split (19,409 train, 8,324 test, split SHA-256 `999ebcb7...f052`).
- Full prepare plus both baselines took about 15 seconds and under 1 GB of memory on an
  arm64 Mac (CPU only).
- DNABERT-2 needs the pinned Hugging Face files listed in `mfass.md`; check the model card terms.
- Report keys: `precision_at_capacity` (capacity 100), `average_precision_sklearn`, `auroc`.

### proteingym-v1.3-dms-substitutions

- Data: the official v1.3 DMS substitution archive (multi-gigabyte), downloaded by the user
  following the ProteinGym README at `144fe22`. Preparation does not download it.
- Each underlying assay has its own terms.
- Use `assay_ids=[...]` (Python) or `--options '{"assay_ids": [...]}'` to select complete
  assays. Omitting it requires all 217. `limit` marks a smoke run.
- Local CSV hashes are observed, not verified against an official checksum; pass
  `expected_hashes` to pin them.
- Metrics per assay: Spearman, AUC, MCC, NDCG, top-10% recall. Suite aggregates appear
  only for a complete, fully scored track.

### flip2-fitness-v1

- Data: Zenodo record 18433203 v3, CC-BY-4.0. One dataset/split per run; the 16
  dataset IDs are in `flip2.md` (for example `flip2-rhomax-by-wild-type`).
- Smallest real run: download `rhomax/by_wild_type.csv.gz` (link in `flip2.md`).
- Archived split counts differ from the manuscript for three splits; the archive is used.
- Metrics: Spearman and NDCG on the test split.

### dart-eval-task1-zero-shot-v1

- Data: Synapse `syn64314109` version 1. Needs an authenticated Synapse account and
  acceptance of its terms. Anonymous download returned HTTP 403 when checked.
- Without it, use `--source demo` (12 synthetic pairs, scope `smoke`, not exportable).
- Local imports are labelled `subset`, never a verified complete official run.
- Metrics: pair accuracy (ties count as wrong), Wilcoxon with SciPy 1.12 rules, score
  difference quantiles, sequence and pair coverage.

### mrnabench-sample-mrl-v1

- Data: Hugging Face `morrislab/mrl-sample` at `ef67f7cf...`; dataset licence unknown.
  The eGFP parquet and its SHA-256 are in `mrnabench.md`; `inspect` lists all four.
- Options: `dataset`, `target`, `require_official: true`. One target per run; never average
  chemistries. Split is random (seed 2541), not a homology barrier.
- Metrics: test MSE (lower is better), Pearson, Spearman.

### genomic-benchmarks-v2

- Data: `pip install genomic-benchmarks`, then `download_dataset(NAME, version=0)`.
- Opaque IDs are regenerated on every prepare; reuse the same `prepared.json` for imported
  predictions. V1 artifacts are refused.
- Exports are labelled `local_evaluation_not_paper_reproduction`.

### tdc-admet-group-v1

- Data: `pip install PyTDC`, then `BenchmarkGroup(name="ADMET_Group", path="data/").get(NAME)`.
  TDC is MIT; each dataset has its own terms.
- 22 datasets, each with its own TDC metric; six are errors (lower is better). No overall score.
- Exports are labelled `local_evaluation_not_paper_reproduction`.

## Smoke options

Every protocol accepts a `limit` option, and it always labels a run `smoke`. A smoke
run checks software only, cannot be exported as a contribution, and is never a
reproduction. Remove `limit` and use a new output directory for a real run.
