# rewire.it benchmarks

Independent, reproducible benchmarks for AI models in biology.

Every benchmark here runs from public data, on open weights, on hardware a reader can afford, and
reports what it cost as well as what it scored. The point is not to rank models. It is to give anyone
a floor they can measure against and re-run themselves.

## House rules

These are enforced in `rewirebench`, not merely described here. A result that breaks one is refused
rather than written.

1. **A trivial baseline always runs**, and is given a fair budget. A leaderboard without a floor is
   misleading, and in this field the floor frequently wins.
2. **The grouping rule and the independent-group count are published** with every result. Splits leak
   in ways that do not show up in the output file.
3. **Contamination is stated** for every pretrained method: what was checked, and what is unknown.
   "unknown" is an acceptable answer. Silence is not, because a reader cannot tell it apart from clean.
4. **Coverage reconciles against the original denominator.** A method that cannot score an input has a
   coverage problem, not a negative prediction.
5. **Throughput sits beside accuracy**, measured end to end rather than on the forward pass alone.
6. **Configuration is recorded** for every pretrained method: checkpoint revision, pooling, context in
   both bases and tokens, precision, batch size.

## Benchmarks

| Benchmark | Task | Labels | Prevalence | Status |
|---|---|---|---:|---|
| [`mfass-v1`](benchmarks/mfass) | Splice-variant prioritisation | Functional, minigene exon recognition | 3.78% | Baseline, SpliceAI, Pangolin |

### mfass-v1 results

Primary metric is precision at a review capacity of 100 variants, on the 8,194 variants every
method could score, across 454 independent groups.

| Method | Family | Precision@100 | Recall@100 | AP | AUROC | Coverage | s/variant |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline-kmer-position | trivial baseline | 0.620 | 0.197 | 0.286 | 0.768 | 8324/8324 | 0.00002 |
| spliceai-1.3.1 | specialist | 0.640 | 0.208 | 0.299 | 0.806 | 8194/8324 | 0.53626 |
| pangolin (mask=False) | specialist | 0.650 | 0.207 | 0.389 | 0.876 | 8301/8324 | 1.64162 |

**Every pair separates on AP and AUROC. No pair separates at precision@100**, a 100-variant review
capacity. Pangolin leads on global ranking, distinguishably over both SpliceAI and the baseline, and
is the only method that stays ahead of the trivial baseline for variants more than 30 bases from a
splice site.

Full tables, paired intervals and the distance-band breakdown: [`benchmarks/mfass`](benchmarks/mfass).

## Layout

```
pyproject.toml                 uv workspace root
packages/rewirebench/          shared: grouping, splits, metrics, result schema
benchmarks/mfass/              one benchmark per directory
  src/mfass/                   build_dataset, split, run_baseline
  splits/                      split manifests, committed
  results/                     one JSON per method per run, committed
  data/                        downloaded and derived, gitignored
```

Shared machinery lives in `rewirebench` so that grouping and scoring cannot quietly diverge between
benchmarks. A benchmark directory holds only what is specific to its dataset.

## Running

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# mfass-v1, about a minute end to end on a laptop
curl -L -o benchmarks/mfass/data/snv_data_clean.txt \
  https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_data_clean.txt

uv run mfass-build
uv run mfass-split
uv run mfass-baseline
```

Tests:

```bash
uv run --group dev pytest packages/rewirebench/tests -q
```

## Adding a benchmark

1. `benchmarks/<name>/` with a `pyproject.toml` depending on `rewirebench`, added to the workspace.
2. A `build_dataset` step that **asserts the cohort against the source publication's own totals** and
   refuses to write on drift. Cohort definitions move silently otherwise.
3. A `split` step using `rewirebench.splits.connected_components`, reporting the naive concatenated-key
   count beside the real group count so key inflation stays visible.
4. A trivial baseline, before any pretrained model.
5. Results written through `rewirebench.results.write_result`, which enforces the house rules.

## Licensing and data

The code in this repository is MIT licensed. The benchmark data is not ours and is not
redistributed.

- **Code** (`packages/`, `benchmarks/*/src/`): MIT, see `LICENSE`.
- **Source data**: downloaded at run time from the original authors and gitignored. MFASS comes from
  [KosuriLab/MFASS](https://github.com/KosuriLab/MFASS), which declares no licence, so it remains the
  authors' work under their terms. Cite [Chong et al., *Molecular Cell* 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6599603/).
- **Derived results** (`benchmarks/*/results/`, `benchmarks/*/splits/`): committed so a reader can
  verify a comparison without rerunning a specialist. The prediction tables carry the upstream
  assay label beside our scores, for attribution and verification only. Anyone building on the labels
  themselves should take them from the original repository, which is the authoritative copy.
- **Model weights**: never redistributed. SpliceAI is PolyForm Strict 1.0.0 code with CC BY-NC 4.0
  weights; Pangolin is GPL v3. Both are installed from upstream at run time under their own terms.

## Reproducibility

Results carry the git revision, UTC timestamp, Python version and platform. Data directories are
gitignored and fully regenerated by the commands above; split manifests and results are committed so
a change in either is visible in a diff.

Source files are pinned by SHA-256 in the cohort summary, so an upstream edit shows up as a
reconciliation failure rather than as a quietly different number.
