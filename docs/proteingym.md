# ProteinGym DMS substitutions v1.3

This adapter evaluates **zero-shot amino-acid substitution scores**, where higher predictions mean greater fitness. It preserves the official experimental `DMS_score` direction and binary labels. The same scalar interface accepts public or private models. Indels, supervised training and clinical tracks are not included.

## Evidence and access

The reference and evaluator are pinned to [ProteinGym revision 144fe22](https://github.com/OATML-Markslab/ProteinGym/tree/144fe22b07dfaeec2b366f2346203a9838a55b4c). The complete reference contains 217 assays. Packaged source receipts record retrieval dates, SHA256 hashes and upstream locations. The upstream software's MIT notice is retained; it does not establish blanket permissions for every underlying experimental dataset. Check each source study's conditions.

Obtain the **v1.3 DMS substitution archive** from the [official ProteinGym download instructions](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/README.md) and extract it outside the repository. Preparation does not download multi-gigabyte archives or accept a licence for you. Point `source` at the directory containing assay CSV files. Model weights are separate and never redistributed in the package.

The reference SHA256 is `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308`. The local assay CSVs are hashed at preparation; these observed hashes are **not independently verified official archive checksums**. Pass previously established `expected_hashes={assay_id: sha256}` to reject changed local inputs. Reports retain this verification gap, including complete-track reports.

## Score existing predictions

```python
from rewirebench import prepare, evaluate

assay = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
data = prepare(
    "proteingym-v1.3-dms-substitutions",
    source="/data/DMS_ProteinGym_substitutions",
    output="./prepared-amfr",
    assay_ids=[assay],
)
# predictions.csv has exactly two columns: id,score.
# IDs are globally unique: <DMS_id>::<mutant>, e.g. <DMS_id>::L2A.
report = evaluate(data, "predictions.csv", output="./amfr-result",
                  model={"name": "My private model", "training_overlap": "unreported"})
```

Use a new output directory each time. All mutants from the selected assay are required by default. Set `allow_partial=True` explicitly to score incomplete predictions; missing records retain their original denominator. Select complete assays with `assay_ids`, or omit that option to require all 217 assays. A `limit` deliberately marks the run as a **smoke test**, including when a caller supplies only a truncated local CSV. Smoke and subset runs have no full-suite metric.

Private models implement `predict(inputs) -> {id: score}`. See `examples/proteingym/private_model.py`. Inputs contain the assay ID, wild-type sequence, mutated sequence and substitution notation. Labels remain with the evaluator. An adapter with a `fit` method is rejected for this zero-shot track. Running locally prevents automatic transmission; it does not prove absence of pretraining overlap.

## Public ESM-2 example

Install the optional `esm` extra. Download the [official ESM-2 8M checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt) during preparation, before an offline compute job. Its verified SHA256 is:

```text
46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928
```

```sh
python examples/proteingym/run_esm.py \
  --data /data/DMS_ProteinGym_substitutions \
  --checkpoint /weights/esm2_t6_8M_UR50D.pt \
  --assay AMFR_HUMAN_Tsuboyama_2023_4G3O \
  --limit 10 --output ./esm-amfr-smoke
```

At every mutated position, the adapter masks that residue in the wild-type sequence, calculates log P(mutant) − log P(wild type), then sums those terms for multiple substitutions. This follows the upstream masked-marginal definition. The loader uses the local checkpoint directly and never calls the model hub. Missing contact-regression weights can produce a warning; contact prediction is not used.

The example requires `fair-esm==2.0.0` and PyTorch. It deliberately rejects sequences longer than 1,022 residues; no implicit truncation or unreviewed windowing occurs. This example does **not** reproduce a specific published ProteinGym row or promise full-suite inference support. CPU execution is available; full-track preparation/scoring may need substantial RAM and disk because this initial SDK materializes input rows. Start with a selected assay.

`examples/proteingym/synthetic_smoke.py` performs two small substitutions with no experimental labels. It is useful for checking an offline environment, and prints `benchmark_result: false`.

## Metrics, aggregation and limitations

Each assay reports **Spearman, AUC, MCC, NDCG and top-10% recall**, following the pinned official scorer. NDCG retains its upstream top-`floor(n×0.1)` and tie handling. MCC thresholds predictions at their assay median. Undefined metrics are JSON `null`, never zero-filled.

The official aggregation first rounds assay metrics to three decimals, averages assays within each **UniProt ID and selection type**, averages proteins within each selection type, then averages selection types equally. This differs from a simple mean over all assays. Only a complete, fully scored track exposes the suite metrics. Per-assay values computed from incomplete predictions are explicitly labelled partial and do not enter aggregates. Category summaries describe only fully scored selected assays; they are not a full-suite ranking.

The report includes complete-assay counts, original denominators, source hashes and scoring scope. It does not estimate an uncertainty interval: the upstream cross-model bootstrap difference is not an absolute performance confidence interval. No results are sent to rewire automatically.

## Validation performed

- The unmodified pinned upstream evaluator runs on an eight-assay synthetic fixture with unequal proteins/category sizes; all five assay metrics and the category-balanced summary agree. Only bootstrap repetitions are reduced for test speed. This validates scorer implementation, not published numerical results.
- Tests reject altered references, duplicate substitutions, incompatible mutated sequences, invalid predictions, unknown IDs and false full-track scope for subsets.
- Two actual ESM-2 8M substitutions executed on macOS arm64 CPU with socket connections blocked. The packaged receipt records weights, implementation hash and outputs. This is a synthetic inference test, not an assay benchmark.
- Full-suite model inference, accelerator environments and real HPC execution remain untested. The ESM assay example requires the user's official data and has not been executed on that full archive.
