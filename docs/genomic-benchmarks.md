# Run Genomic Benchmarks locally

Use `rewirebench` 0.3 or later and `genomic-benchmarks-v2`. Version 1 included
class names in adapter-visible IDs and supplied sequences in class order. Its
prepared files and reports are refused by the current SDK. Prepare again and
generate new predictions; renaming an old protocol or report is not a migration.
Keep earlier artifacts separately for audit.

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

V2 assigns random opaque IDs independently of class names, paths and sequences,
then sorts those IDs before selecting smoke subsets or batching adapter inputs.
The resulting order and IDs are saved in `prepared.json`; reuse that file for
repeat runs and imported predictions. Preparing again generates new IDs and a
new order, even for the same source files. Train/test source hashes remain stable:
they cover length-framed relative paths and original file bytes, including class
membership. These hashes describe your local copy, not a verified official cohort.

## Prepare, run and score

```bash
rewirebench prepare genomic-benchmarks-v2 \
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

Accuracy is reported for every dataset. Binary F1 treats the class mapped to
index 1 as positive; a paper comparison must use the same positive class.
For `human_ensembl_regulatory`, which has three classes,
the paper does not say which averaging its F1 column uses, so macro and weighted
are both reported and neither is presented as the paper's number.

Exports support the explicit review-queue submission workflow in [the SDK guide](sdk.md).
They carry `evaluation_claim: local_evaluation_not_paper_reproduction` and
`data_verification: local_bytes_hashed_not_independently_source_verified`.
Coverage is measured against the local test files read during preparation.
Scoring that entire copy does not establish that it is the paper's complete split.
