# FLIP2 fitness evaluation

`flip2-fitness-v1` evaluates scalar fitness predictions on all 16 archived FLIP2
splits, spanning seven protein datasets. Each run covers one dataset/split. There
is no combined FLIP2 score, and a selected split is never labelled a complete
suite. Legacy FLIP datasets such as GB1 and AAV are separate and unsupported by
this protocol.

## Pinned sources and important differences

Data is pinned to [Zenodo record 18433203, version v3](https://doi.org/10.5281/zenodo.18433203).
The package's `resources/flip2/sources.json` records SHA-256 hashes for every
compressed file, its decompressed CSV, the source website, the manuscript and
the evaluator. All 16 website CSVs were downloaded and found byte-identical to
the decompressed Zenodo CSVs; their gzip wrappers differ. Preparation verifies
the CSV bytes, allowing either compressed mirror or the unchanged plain CSV.

These are the **archived assignments**, not an inferred reconstruction of the
paper's experiments. Three source inconsistencies remain explicit:

- Amylase `one_to_many`: archive train/validation/test counts are 2,574/644/488;
  manuscript Table 1 says 412/77/3,217. The runner does not invert the split.
- NucB `two_to_many`: archive counts are 2,088/5,367/48,304; Table 1 says
  6,982/2,027/48,304. Its printed total also differs from the archive.
- IRED contains 8,586 archived rows, matching Table 1's split-count sum but not
  its stated total of 17,143.

The repository linked from the website has an outdated README, but its
[metric aggregation](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/baselines/aggregate.py#L131-L135)
and [linear baseline](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/baselines/linear_models.py#L110-L163)
contain the FLIP2 procedures. We independently implement their metric formulas;
AFL-3.0 source code is not copied into the MIT package.

The archive is CC-BY-4.0. Attribution files are packaged for each dataset;
original Amylase data is MIT and original TrpB data is CC0-1.0 according to those
files. Other original datasets are identified as CC-BY-4.0. Cite the FLIP2
paper, archive and original dataset authors when using the data.

## Dataset and split identifiers

| Dataset | `--options` dataset IDs (prefix `flip2-`) |
|---|---|
| Amylase | `amylase-one-to-many`, `amylase-close-to-far`, `amylase-far-to-close`, `amylase-by-mutation` |
| IRED | `ired-two-to-many` |
| NucB | `nucb-two-to-many` |
| TrpB | `trpb-one-to-many`, `trpb-two-to-many`, `trpb-by-position` |
| Hydro | `hydro-three-to-many`, `hydro-low-to-high`, `hydro-to-p06241`, `hydro-to-p0a9x9`, `hydro-to-p01053` |
| Rhomax | `rhomax-by-wild-type` |
| PDZ3 | `pdz3-single-to-double` |

Run `rewirebench inspect flip2-fitness-v1` for exact download URLs, counts, hashes,
input contracts, units and access information.

## Download, verify and prepare

No credentials, GPU or model weights are required for data preparation and the
composition baseline. Download this small dataset before going offline:

```bash
mkdir -p flip2-data/rhomax
curl --fail --location \
  'https://zenodo.org/api/records/18433203/files/rhomax/by_wild_type.csv.gz/content' \
  --output flip2-data/rhomax/by_wild_type.csv.gz
rewirebench prepare flip2-fitness-v1 \
  --source flip2-data --output prepared-flip2-demo \
  --options '{"dataset":"flip2-rhomax-by-wild-type","limit":8}'
```

Preparation checks the pinned CSV SHA-256 before selecting inputs. A changed
file is rejected. The Python-only `allow_unverified=True` escape hatch is for
explicit local fixtures: it produces a subset with unverified provenance,
never a complete published split. It cannot satisfy the official submission
contract's source hash.

Remove `limit` only when intentionally preparing the whole selected split.
`limit` always labels a smoke run, even if it happens to include every row.
Original test denominators remain in the report.

## Run the CPU example or your private model

```bash
rewirebench run --prepared prepared-flip2-demo \
  --adapter rewirebench.resources.flip2.example_adapter:CompositionEmbeddings \
  --prediction-type embedding --output results-flip2-demo \
  --model-name 'Composition ridge learning control'
```

This example uses amino-acid composition and a fixed ridge head. It is a local
learning control, **not** the paper's one-hot baseline. The eight-row smoke
scores are software checks, not model-quality estimates.

A private scalar adapter can provide `predict(inputs)` and optionally
`fit(train_inputs, targets)` or
`fit_with_validation(train_inputs, targets, validation_inputs, validation_targets)`.
Each input is `{id, sequence}`; targets and source row numbers are absent. Return
a mapping from input ID to finite scalar score. Higher scores should mean higher
source fitness. The test labels are held by the evaluator.

PDZ3 stores **protein:peptide** strings, sometimes with an empty partner. The
runner preserves the colon and empty component. Private models must explicitly
support that input or provide imported predictions; silently dropping the
partner changes the biological task. The composition example separately
encodes both components.

A frozen adapter supplies `embed(inputs)` returning one finite 1D vector per ID.
The protocol fits `Ridge(alpha=10, solver="auto", tol=1e-5, max_iter=1000000)`
on training embeddings only. Target scaling uses training labels only;
validation is unused because the hyperparameters are fixed. Unlike the published
one-hot baseline, this extension does not use validation labels in target
scaling and makes no paper-reproduction claim.

Imported predictions use the existing `--predictions` interface. Imported
embeddings use `--embeddings` with keyed JSON vectors or safe NPZ arrays:

```bash
rewirebench evaluate --prepared prepared-flip2-demo \
  --embeddings private-embeddings.json --output results-flip2-private
```

Embeddings must cover every selected input and have matching dimensions. Private
sequences and embeddings remain local. Preparation and output directories must
be new; historical results are never overwritten.

## Scoring, outputs and comparison

Spearman correlation uses average ranks for ties. NDCG uses the upstream
full-ranking calculation: subtract the minimum **scored test** target, then use
scikit-learn's linear-gain, tie-averaged NDCG. Constant-target or
constant-prediction Spearman is reported as unavailable. For constant targets,
shifted NDCG is zero, matching the upstream function. Fewer than two scored rows
makes both metrics unavailable.

Missing predictions remain in the original denominator. With explicitly allowed
partial predictions, ranks and target shifting are calculated on scored rows
only; those metrics are not directly comparable to complete results. Target
units and transformations are preserved; no averaging across landscapes or
splits is performed.

The output contains `report.json`, `predictions.json` and `unscored.json`.
The report records source hashes, split counts, preparation scope, method
configuration, software environment, coverage and source conflicts. Only an
explicit `rewirebench export` creates a contribution bundle; no command in this
guide uploads data.

Use the shared Podman/Apptainer instructions with this protocol ID and read-only
prepared input. Model-specific memory and accelerator requirements depend on
the private model; the CPU control needs no accelerator. Container compatibility
is reported in the release's execution receipts, not inferred from native tests.

## Validation status

`resources/flip2/validation-2026-09-20.json` records real-source, native CPU
examples for **all 16 splits**, using eight rows per train/validation/test group.
Each download matched the archive checksums. The actual pinned upstream metric
expressions were executed separately via AST extraction and agreed within
`1e-12` on identical labels/predictions. This does not execute the upstream
training pipeline or reproduce any published model score.

Offline unit tests cover negative targets, tied predictions, undefined metrics,
source mismatches, malformed outputs, duplicate IDs, changed splits, partial
coverage, PDZ3 delimiters, embedding dimensions and independence from
validation/test labels during head fitting.
