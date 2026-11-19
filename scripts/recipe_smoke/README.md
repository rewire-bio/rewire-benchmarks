# Run-guide CPU smoke checks

`.github/workflows/recipe-smoke.yml` runs the documented run guides below on a hosted CPU runner, weekly and on changes here. A check passes only if the guide reaches evaluation and writes its documented outputs. Each job uploads `receipt-<recipe>.json`, which records:
- the upstream revision;
- every deviation from the guide;
- the runner;
- the SHA-256 of each output;
- the resolved package versions.

`guides/` holds the seven run guides as published in the rewire benchmark store (`rb:runGuide`, release `2026-11-17-793d9cca4ad1`), except BEND. `discovery-benchmark-bend.json` is the corrected guide pinned to BEND commit 58ad410, copied verbatim from the store's `run_guide` in rewire-bio/rewire-benchmark-data#307; it will match the first store release after that PR. `corrections/discovery-benchmark-bend.json` lists each correction: what was wrong, whether it was upstream or ours, the upstream issue, PR or commit, the upstream reply as quoted, the dates found and resolved, and the store claim IDs that record it. The store keeps the same corrections as claims that hold each previous value; the file here is a cross-reference, kept separate so the guide stays an exact copy.

| Guide | Status | Smoke inputs and deviations |
| --- | --- | --- |
| MFASS v2 (`rewire-mfass-v2`) | cpu-smoke | Full cohort. Inputs checked against the README SHA-256 values. The baseline must reproduce the published P@100 0.610, AP 0.286 and AUROC 0.778. Python 3.11 (guide: >=3.11; 3.12 fails to build tokenizers 0.13.3). The optional DNABERT-2 resource pilot is skipped. |
| BEELINE | cpu-smoke | The supplied GSD example with PPCOR and GRNBOOST2 only, and only their two images pulled. The runner, evaluator (AUPRC/AUROC) and plotter run as documented. |
| ProteinGym | #67 | The documented environment no longer installs, and the v1.3 score archive does not match `merge.py`. |
| MassSpecGym | #68 | `pip install massspecgym` gives an environment where `import pytorch_lightning` fails (`pkg_resources`). |
| scIB | #69 | The R 4.0 environment could not be built on the hosted runner (mamba cache permissions); the recipe is untested. |
| TAPE | #70 | It needs a fine-tuned secondary-structure checkpoint that is not published. |
| BEND | #71 | Corrected and pinned to 58ad410, which fixed the checkpoint links and requirements upstream. The guide now downloads only the gene-finding inputs (about 3.5 GB, or 3.7 GB with the two checkpoints). Embedding precompute has no smoke-sized input yet, so it is not smoke-tested. |

Run one locally with `RUNNER_TEMP=$(mktemp -d) bash scripts/recipe_smoke/mfass_v2.sh`. BEELINE needs conda (`$CONDA`) and Docker.
