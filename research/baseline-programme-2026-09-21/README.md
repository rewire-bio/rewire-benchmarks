# Matched Rhomax evaluations, 21 September 2026

Two frozen ESM-2 checkpoints and a conventional sequence reference were evaluated locally on the complete FLIP2 Rhomax `by_wild_type` split. All methods scored the same 184 held-out proteins. This is one task and split, not a FLIP2 suite score or reproduction of a published model score. ESM-2 8M and 35M are two configurations of the same model family.

| Configuration | Spearman | NDCG | Coverage |
|---|---:|---:|---:|
| ESM-2 8M, layer 6 residue mean + fixed ridge | -0.1463506755340845 | 0.8964799835636942 | 184/184 |
| ESM-2 35M, layer 12 residue mean + fixed ridge | -0.2217595033467806 | 0.9072580204736261 | 184/184 |
| 22-feature sequence composition + fixed ridge | 0.41798958279508075 | 0.9548154942827918 | 184/184 |
| Training mean, prior-result validation | Undefined | 0.9206667522227658 | 184/184 |

These frozen linear probes performed poorly on this split. The results do not establish general ESM-2 performance, a model-family ranking or a statistically significant difference. There are no confidence intervals. The larger checkpoint was selected after the 8M run but before its own scores, with the same fixed head and no tuning on held-out outcomes; see `35m-selection.json`.

## Procedure and evidence

The archived split contains 584 training, 116 validation and 184 test proteins, with 227–365 residues. We encoded all 884 sequences without labels, using their complete sequences and no MSA or templates. Final-layer residue representations were averaged without BOS, EOS or padding. A 320-dimensional 8M representation and 480-dimensional 35M representation each fed the existing fixed alpha 10 ridge head. Target scaling and fitting use training rows only; validation labels do not select settings. Pretraining overlap with UniRef50 is unreported.

The 22-feature conventional reference comprises log1p length, 20 residue fractions and an unknown-character fraction. It differs from the 40-feature reference recorded on 20 September. It is a new exact configuration, not an additional independent repeat of the old reference. Training mean predictions match the previous run exactly and must not be submitted again. Controls repeated beside 35M are validation-only. `evaluation-index.json` identifies the three eligible new evaluations and the duplicate bundles to exclude.

`sources.csv` records origins, versions, locators, hashes and verification status. Fresh source-file downloads matched the pinned Rhomax hash. Independently parsed source sequences, targets and assignments matched all 884 prepared rows (`source-verification.json`). Fresh pinned upstream evaluator bytes confirmed the Spearman and full-rank NDCG definitions. NDCG shifts scored target values by their minimum, including when the minimum is positive.

Each result has an SDK report, sanitised SDK contribution bundle and validation receipt. Raw sequences, labels, embeddings, prediction files and machine paths remain in ignored local execution storage. Separate package snapshots and file-hash manifests preserve the exact code imported during each execution. Installed package metadata reports 0.4.0 because the local environment retains that wheel's distribution metadata; the recorded code digest and snapshot identify the development implementation actually used.

The metric verifier recomputes both scores directly from local predictions with SciPy/scikit-learn, independent of the SDK score function, at 1e-12 absolute tolerance. Its initial NDCG check caught a verifier mistake: it originally subtracted the minimum only for negative labels. The check was corrected to match the pinned upstream procedure and rerun against unchanged predictions. No model was retuned or rerun to improve the score. The 8M aggregate elapsed field is therefore null; its original report records 17.60 seconds for encoding and fitting. Timing is not a hardware-normalised speed benchmark. Missing contact-regression weights generated an ESM warning, but no contacts were requested or used.

## Re-run or inspect

Use a fresh environment with the versions in the relevant `verification.json`, the pinned official checkpoint URL/hash from the resource manifest, and the pinned FLIP2 source. Extract the appropriate `package-source.tar.gz` into an isolated directory and set `PYTHONPATH` to that directory. This avoids confusing later package edits with the executed snapshot.

```bash
PYTHONPATH=./frozen-source python run_rhomax.py \
  --prepared ./prepared/prepared.json \
  --checkpoint ./esm2_t6_8M_UR50D.pt \
  --output ./new-execution --evidence ./new-evidence

# For 35M use its package snapshot and specify the checkpoint identity.
PYTHONPATH=./frozen-source-35m python run_rhomax.py \
  --model-name esm2_t12_35M_UR50D \
  --prepared ./prepared/prepared.json \
  --checkpoint ./esm2_t12_35M_UR50D.pt \
  --output ./new-execution-35m --evidence ./new-evidence-35m
```

These commands perform no uploads. Evidence is ready for review and explicit SDK submission; it has not been published to the database by this execution script. Native Linux, containers, HPC and cloud execution were not tested in these receipts.
