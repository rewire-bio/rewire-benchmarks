# Run Genomic Benchmarks locally

Genomic Benchmarks is nine sequence classification datasets. The published
baseline is a small convolutional network, reported twice, once built with
PyTorch and once with TensorFlow, which is the comparison the rewire database
page for this benchmark records.

## Access and preparation

The package downloads each dataset and writes it as one sequence per file:

```bash
pip install genomic-benchmarks
```

```python
from genomic_benchmarks.loc2seq import download_dataset

download_dataset("human_nontata_promoters", version=0)
```

That leaves `<cache>/human_nontata_promoters/{train,test}/<class>/<id>.txt`.
Point the protocol at the directory holding the dataset.

## A note on class labels

Upstream assigns the integer label from the order the filesystem returns the
class directories in. That order is not stable between machines. This protocol
sorts the class names instead and records the mapping it used in the prepared
dataset, so a run here is reproducible. Matching upstream's integer for a given
class is not something either implementation can promise, so compare class
names rather than indices.

## Prepare, run and score

```bash
rewirebench prepare genomic-benchmarks-v1 \
  --source ./genomic_benchmarks --options '{"dataset": "human_nontata_promoters"}' \
  --output ./prepared-promoters
```

Your adapter receives the sequence for each row. Return a class index, or for a
binary dataset a probability, which is thresholded at 0.5 the way upstream
thresholds scores before computing accuracy and F1.

```bash
rewirebench run --prepared ./prepared-promoters \
  --adapter my_models.dna:MyAdapter \
  --model-name 'my model' --training-overlap 'unreported' \
  --output ./scored-promoters
```

## What this does and does not establish

Accuracy is reported for every dataset. F1 is reported for the binary datasets,
where it is unambiguous. For `human_ensembl_regulatory`, which has three classes,
the paper does not say which averaging its F1 column uses, so macro and weighted
are both reported and neither is presented as the paper's number.
