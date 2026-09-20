# Sequence benchmark expansion: rewirebench 0.4

This release adds three protocol integrations to the existing local runner. They evaluate one named dataset/split or target at a time. None produces a cross-benchmark leaderboard score.

| Protocol | Supported scope | Predictions and fitting | Source validation |
| --- | --- | --- | --- |
| FLIP2 | Seven datasets, 16 archived v3 splits | Scalar fitness scores; private supervised adapters; frozen sequence embeddings with a fixed train-only ridge probe | All 16 source files and split counts checked. Archived counts disagree with some manuscript counts; the archive remains authoritative for this protocol. |
| DART-Eval Task 1 | Paired regulatory elements and shuffled controls | Zero-shot scalar sequence scores; fitting forbidden | Pinned evaluator and HDF5 layout checked. Official Synapse downloads require access; local files remain explicitly unverified. |
| mRNABench Sample MRL | Four datasets, six separate targets | Scalar MRL predictions; private supervised adapters; frozen embeddings with train-only RidgeCV | All four source files and six canonical splits checked. This evaluates the held-out test split, not the upstream default validation split. |

Read [FLIP2](flip2.md), [DART-Eval](dart-eval.md) and [mRNABench](mrnabench.md) for access, commands, methods and limitations. `rewirebench inspect PROTOCOL` lists the supported identities and execution capabilities without downloading weights.

## Shared execution contract

Preparation owns labels, split assignments and coverage denominators. Adapters receive opaque IDs and allowlisted biological sequences. Optional `fit` receives training labels; `fit_with_validation` receives training and validation labels only when the protocol permits it. These controls prevent accidental label exposure; they do not sandbox arbitrary Python model code or establish absence of training contamination.

`evaluate --embeddings` accepts keyed JSON vectors or NPZ arrays named `ids` and `embeddings`, with pickle loading disabled. All selected training, validation and test embeddings must be present. The protocol owns fitting. Scalar prediction files can declare missing predictions and retain the original evaluation denominator.

Run reports distinguish imported scores, imported embeddings and local adapter execution. Explicit exports record the evaluation method and source hashes; submission remains a separate user action into private review. Scientific reproduction is never inferred from a successful local run.

## Validation and limits

Source and numerical checks are stored with the wheel under `rewirebench/resources/flip2`, `dart_eval` and `mrnabench`. Receipts name source revisions, checked files, execution scope and tolerances. Small CPU controls validate the implementation; they are not new foundation-model benchmark results.

FLIP2's embedding probe is an explicitly identified extension, not a reproduction of its published one-hot ridge baseline. DART preserves the pinned SciPy 1.12 Wilcoxon convention, including its limitations for ties; synthetic demonstrations cannot be exported. mRNABench uses six independent target evaluations and never silently combines them.

The `sequence` environment adds parquet/HDF5 readers without neural-network frameworks. Native Python remains supported. Linux x86-64 container parity is checked in CI using offline Podman and Apptainer execution; downloadable artifacts are released only from the immutable version tag.
