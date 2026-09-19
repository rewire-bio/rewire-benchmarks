# Run the TDC ADMET benchmark group locally

The ADMET benchmark group is 22 datasets of drug properties. Each has its own
metric, fixed by Therapeutics Data Commons rather than chosen here, and six of
them are errors where a lower number is better. There is no single ADMET score,
and this protocol does not compute one.

## Access and preparation

TDC distributes the data through its own package, which downloads it and writes
the scaffold split to disk. Install it and fetch the dataset you want:

```bash
pip install PyTDC
```

```python
from tdc import BenchmarkGroup

group = BenchmarkGroup(name="ADMET_Group", path="data/")
group.get("Caco2_Wang")
```

That leaves `data/admet_group/caco2_wang/train_val.csv` and `test.csv` on disk.
The protocol reads those files and records their SHA-256 in the run's
provenance. It does not claim the split matches a particular TDC release: that
is a property of the copy you downloaded, and the hash is what lets two copies
be told apart.

TDC is MIT licensed. The reuse terms of each underlying dataset are upstream and
are not restated here; assess them for your use.

## Prepare, run and score

```bash
rewirebench prepare tdc-admet-group-v1 \
  --source ./data --options '{"dataset": "caco2_wang"}' \
  --output ./prepared-caco2
```

Your adapter receives the SMILES string for each row and returns one number per
ID. For a regression dataset that is the predicted property; for a
classification dataset it is a score for the positive class, which is what
roc-auc and pr-auc expect.

```bash
rewirebench run --prepared ./prepared-caco2 \
  --adapter my_models.admet:MyAdapter \
  --model-name 'my model' --training-overlap 'unreported' \
  --output ./scored-caco2
```

The report names the metric, its direction, and the coverage: how many test rows
you scored out of how many exist. A partial run is scored and reported as
partial rather than refused, and it is never marked complete.

## What this does and does not establish

The metric per dataset and the way it is computed are transcribed from TDC's own
evaluator at revision `c310c35f27e3f506411018ac43d97b8ba23ca652`: the map in
`tdc/metadata.py` and the functions `tdc/evaluator.py` assigns, including the
spearman case that keeps the correlation and drops the p-value.

Matching a published number needs more than a matching metric. The split on your
disk, the featurisation, and the training procedure all have to match too, and
the published baselines on the rewire database page for this group are the
paper's own simple models rather than the current leaderboard.
