# rewire.it benchmarks

> **MFASS correction, September 2026:** The archived `mfass-v1` baseline used a
> sequence field reverse-complemented for 7,770 variants while keeping the
> original assay coordinate. Its baseline comparisons and split-cost claims
> are superseded. See the validated assay-oriented `mfass-v2` results in
> [the MFASS benchmark README](benchmarks/mfass/README.md). V1 result files are
> preserved for audit.

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
| [`mfass-v1`](benchmarks/mfass/README.md) | Splice-variant prioritisation | MFASS minigene exon recognition | 3.78% | Archived; baseline superseded after sequence correction |
| [`mfass-v2`](benchmarks/mfass/README.md) | Splice-variant prioritisation | Same functional cohort and canonical exon/gene split | 3.78% | Corrected baseline and local DNABERT-2; specialists carried unchanged |

### MFASS-v2 results

The source assay's `original_seq` and `natural_seq` differ at the variant's
assay-coordinate position for all 27,733 eligible rows. The legacy `sequence`
field is reversed for 7,770 of them; `mfass-v2` validates the pair and uses the
assay-oriented mutant. The v1 result files are retained for audit, but v1
baseline comparisons and split-cost claims are withdrawn.

| Method | Protocol | P@100 | AP | AUROC | Coverage |
|---|---|---:|---:|---:|---:|
| Corrected k-mer/position baseline | supervised on MFASS train | 0.610 | 0.286 | 0.778 | 8,324/8,324 |
| SpliceAI 1.3.1 | zero-shot specialist, unchanged | 0.640 | 0.299 | 0.806 | 8,194/8,324 |
| Pangolin, mask=False | zero-shot specialist, unchanged | 0.650 | 0.389 | 0.876 | 8,301/8,324 |
| DNABERT-2 117M frozen pair + logistic head | supervised head on MFASS train | 0.030 | 0.045 | 0.550 | 8,324/8,324 |

Paired whole-exon/gene-group resampling on common variants shows Pangolin
still exceeds the corrected baseline on AP and AUROC. SpliceAI's corrected
AUROC difference is +0.028 with a 95% interval [-0.004, +0.064], so the
old claim of clear separation is invalid. The locally run DNABERT-2
**frozen-pair protocol** is worse than the trivial baseline: P@100 -0.580
[-0.713, -0.464], AP -0.241 [-0.304, -0.182], and AUROC -0.228
[-0.300, -0.160]. This is one declared representation and train-only head,
not a model-wide verdict. Full method, access, coverage, timing, and source
limits are in the [MFASS benchmark report](benchmarks/mfass/README.md),
with exact artifact hashes in the [run provenance manifest](benchmarks/mfass/provenance/mfass-v2-local-dnabert2.json).

## Running

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --package mfass --extra dnabert2-pilot

# Public author-maintained source tables; their exact hashes are in the MFASS README.
curl -L -o benchmarks/mfass/data/snv_data_clean.txt \
  https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_data_clean.txt
curl -L -o benchmarks/mfass/data/snv_func_annot.txt \
  https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_func_annot.txt

uv run --package mfass --extra dnabert2-pilot mfass-build --check
uv run --package mfass --extra dnabert2-pilot mfass-build
# Use the checked-in split-v2.tsv; do not overwrite it with a new draw.
uv run --package mfass --extra dnabert2-pilot mfass-baseline
uv run --package mfass --extra dnabert2-pilot mfass-dnabert2-pilot
# If the pilot passes its 12-hour projection and 1 GiB disk reserve:
uv run --package mfass --extra dnabert2-pilot mfass-dnabert2
```

Tests:

```bash
uv run --group dev pytest -q
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
