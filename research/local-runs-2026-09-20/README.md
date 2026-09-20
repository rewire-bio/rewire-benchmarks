# Local benchmark evaluations, 20 September 2026

Five new Rewire evaluations using rewirebench 0.4.0 and real experimental data. These are selected evaluations, not full benchmark-suite scores or reproductions of published model scores. No paid compute was used.

The batch was selected before running it: FLIP2 Rhomax `by_wild_type`, mRNABench Sample `designed`, and ProteinGym v1.3 AMFR. Two controls were fitted for each supervised dataset; ESM-2 8M was evaluated zero-shot on AMFR.

## Results

| Selected evaluation | Method | Test records scored | Metric | Value |
|---|---|---:|---|---:|
| FLIP2 Rhomax | Amino-acid composition + ridge | 184/184 | Spearman | 0.41818225155801514 |
| FLIP2 Rhomax | Training mean | 184/184 | Spearman | Unavailable: constant predictions |
| FLIP2 Rhomax | Amino-acid composition + ridge | 184/184 | NDCG | 0.954819941821582 |
| FLIP2 Rhomax | Training mean | 184/184 | NDCG | 0.9206667522227658 |
| mRNABench Sample designed | Sequence composition + RidgeCV | 15,003/15,003 | MSE | 1.914439715839851 |
| mRNABench Sample designed | Training mean | 15,003/15,003 | MSE | 2.3655294722554605 |
| mRNABench Sample designed | Sequence composition + RidgeCV | 15,003/15,003 | Pearson | 0.4367869170170987 |
| mRNABench Sample designed | Sequence composition + RidgeCV | 15,003/15,003 | Spearman | 0.49477537290902324 |
| ProteinGym AMFR stability | ESM-2 8M masked marginals | 2,972/2,972 | Spearman | -0.209 |
| ProteinGym AMFR stability | ESM-2 8M masked marginals | 2,972/2,972 | AUC | 0.394 |
| ProteinGym AMFR stability | ESM-2 8M masked marginals | 2,972/2,972 | MCC | -0.139 |
| ProteinGym AMFR stability | ESM-2 8M masked marginals | 2,972/2,972 | NDCG | 0.440 |
| ProteinGym AMFR stability | ESM-2 8M masked marginals | 2,972/2,972 | Top-10% recall | 0.057 |

The training-mean RNA control has undefined Pearson and Spearman correlations, preserved as JSON `null`. FLIP2's `n` field is a count, not a performance metric. Full precision and remaining metrics are in each run's `report.json`.

Rhomax predicts a measured spectral wavelength in nanometres, rather than generic fitness. The archived split has 584 training, 116 validation and 184 test records. Its composition control uses amino-acid fractions and a fixed alpha-10 ridge head, with target scaling fitted on training labels only. It is not the published FLIP2 one-hot baseline. NDCG ranks the reported target numerically; it does not mean higher wavelength is universally biologically preferable.

The RNA evaluation predicts mean ribosome load in a reporter assay, using the full processed source sequence. It retains the canonical seed-2541 train/validation/test split. RidgeCV selects alpha using training data only; validation and test labels do not select hyperparameters. No feature normalization is applied. This selected held-out test result is not the upstream default validation score or the complete mRNABench benchmark. The split does not establish homology separation.

The ESM-2 result is negative on the selected AMFR construct. It is retained without reversing score direction or switching assays. The official masked-marginal procedure was checked against all 2,972 predictions, including multi-substitution sums; the score sign is intentional. ProteinGym values use upstream three-decimal rounding. This result does not establish performance on other proteins or the complete suite. See [the assay report](proteingym-esm2/README.md) for full methods and limitations.

## Provenance and review

Each run contains a sanitized `report.json`, an SDK `bundle.json` with its evidence sidecar, and source/verification information. Raw predictions and prepared inputs stay in the ignored execution workspace. Hashes bind the public summaries to those local files. There are no raw biological sequences, redistributed weights, access tokens or machine paths in the public evidence.

Review is automated and recorded explicitly in audit tables. Local execution does not establish absence of model training overlap. These runs are not independent replications of previous paper results. Source verification, numerical verification and scientific reproduction are distinct claims.

FLIP2 source CSV bytes match Zenodo v3. The downloaded official mirror uses a different gzip wrapper; both compressed and decompressed hashes are recorded. mRNABench parquet bytes match the pinned Hugging Face revision; its data reuse licence is unreported, so data are not redistributed. ProteinGym's observed archive/member hashes remain distinguished from independently published upstream checksums.

The SDK export status remains `unreviewed_contribution`; review receipts are separate. Public intake is disabled. Submission is through a catalogue PR, with source URLs pinned to a public Git commit. The PR is not automatic publication.

## Original sources

- FLIP2 archived assignments: [Zenodo v3, record 18433203](https://doi.org/10.5281/zenodo.18433203). Original Rhomax measurements: [Inoue et al. (2021)](https://doi.org/10.1038/s42003-021-01878-9), CC-BY 4.0 as recorded in the source attribution.
- mRNABench processed inputs: [pinned Sample dataset](https://huggingface.co/datasets/morrislab/mrl-sample/tree/ef67f7cf8a999bb1c412ad6551aa7d9f901cbb95). Original assay: [Sample et al. (2019)](https://doi.org/10.1038/s41587-019-0164-5). Protocol implementation reference: [mRNABench revision 74f96b8](https://github.com/morrislab/mRNABench/tree/74f96b8e6ae9f41cc3cccff089d826a62d5604b8).
- ProteinGym reference and evaluator: [revision 144fe22](https://github.com/OATML-Markslab/ProteinGym/tree/144fe22b07dfaeec2b366f2346203a9838a55b4c). AMFR stability assay: [Tsuboyama et al. (2023)](https://doi.org/10.1038/s41586-023-06328-6). ESM model implementation: [official Meta repository](https://github.com/facebookresearch/esm/tree/2b369911bb5b4b0dda914521b9475cad1656b2ac).

## Reproduce the four sequence controls

Use Python 3.11.13 and the released wheel, whose SHA-256 is `f5956f616c20f58299eef4d654b1297b0ce21e0033014cc5cb2c1a406f4b494a`. The tested platform was macOS arm64 CPU. The commands require `uv` and `curl` and use new output directories. Execution is offline after installing software and downloading inputs.

```bash
mkdir -p local-inputs
curl --fail --location https://github.com/rewire-bio/rewire-benchmarks/releases/download/v0.4.0/rewirebench-0.4.0-py3-none-any.whl --output local-inputs/rewirebench-0.4.0-py3-none-any.whl
python3 - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('local-inputs/rewirebench-0.4.0-py3-none-any.whl').read_bytes()).hexdigest() == 'f5956f616c20f58299eef4d654b1297b0ce21e0033014cc5cb2c1a406f4b494a'
PY
uv venv --python 3.11.13 local-env
uv pip install --python local-env/bin/python -r research/local-runs-2026-09-20/sequence-requirements.txt 'local-inputs/rewirebench-0.4.0-py3-none-any.whl[sequence]'
curl --fail --location https://flip.protein.properties/assets/splits/rhomax/by_wild_type.csv.gz --output local-inputs/by_wild_type.csv.gz
curl --fail --location https://huggingface.co/datasets/morrislab/mrl-sample/resolve/ef67f7cf8a999bb1c412ad6551aa7d9f901cbb95/mrl-sample-designed.parquet --output local-inputs/mrl-sample-designed.parquet
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 local-env/bin/python research/local-runs-2026-09-20/run_sequence.py \
  --flip-source local-inputs/by_wild_type.csv.gz \
  --mrna-source local-inputs/mrl-sample-designed.parquet \
  --work local-execution --evidence local-evidence
```

Preparation verifies source hashes before selecting data. The script executes separate smoke checks, then full selected splits without limits. Opaque record IDs and their resulting file hashes change on re-preparation; expected split counts and numerical results should agree within `1e-9` absolute tolerance. Floating-point ordering may change the last digits. Runtime includes feature generation and fitting within the SDK's measured scope, not environment installation or downloads; it is not a cross-machine speed benchmark.

Keep at least 5 GiB free disk. Do not place `local-inputs`, `local-env`, `local-execution` or regenerated evidence under tracked repository paths. The actual run used ignored `workbench/` paths. See the ProteinGym subdirectory for its separate ESM environment and procedure.

### Audit a reproduced sequence run

Obtain the pinned upstream source files listed in each run's `audit.json`: FLIP2 `baselines_aggregate.py` and `baselines_linear_models.py`, and mRNABench files with path separators replaced by dots (for example `mrna_bench.linear_probe.evaluator.py`). The checker verifies their hashes, reconstructs source rows/splits, refits using independent feature code, and recomputes every metric. It does not use the SDK fitting or scoring functions.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 local-env/bin/python research/local-runs-2026-09-20/verify_sequence.py \
  --work local-execution --evidence local-evidence \
  --flip-source local-inputs/by_wild_type.csv.gz \
  --mrna-source local-inputs/mrl-sample-designed.parquet \
  --flip-upstream local-inputs/flip-source-code \
  --mrna-upstream local-inputs/mrna-source-code
```

The published checks passed for all four sequence runs: independent predictions matched exactly and metric differences were at most `5.56e-17`. The full ESM-2 prediction replay also matched exactly. All reviews were automated; unresolved training-overlap and data-licence/checksum limitations remain recorded.
