# MFASS protocol resources

`split-v2.tsv` is the unmodified rewire split at revision
`bee9133b83f3aedaf2bbb9013f1875515845607e`. Its SHA-256 is
`999ebcb7e63a5c5eaa8780fa468e59ac1f934260ad50102814174c396317f052`.
The upstream cohort and assay sequences are deliberately not redistributed.
The SDK can build the cohort from hash-verified local upstream tables without
writing into the source directory. Alternatively provide `cohort.tsv`
with SHA-256 `389702ff4c647d7ce10a90092a6fa811ae777d15997baf39ce9aae0346247bd0`.
The source repository does not declare reuse terms for its data; code licensing
does not establish data redistribution permission.

The packaged protocol validates the whole source before taking a smoke subset.
For a trainable example use `limit=80` or higher: the first positive training row
in source order has index 66. Smaller limits may correctly refuse fitting.
A smoke run retains the full eligible test denominator (8,324) and does not
establish a published model score.

DNABERT-2 uses the eight exact hashes in `DNABERT2.ARTIFACTS`. Supply these files
in one local directory. Model/tokenizer files are from checkpoint
`b5ae377faa374ee160eec1c27b8494436cc94451`; the historical run loaded auxiliary
Python code from `7bce263b15377fc15361f52cfab88f8b586abda0`. `bert_layers.py`
is identical across both revisions. No network access is attempted by the adapter.
The random pooler is ignored; only masked means of pretrained token states are used.

`validation-2026-09-17.json` records actual local tests, including a small
end-to-end public encoder example. It is not a container or full-run verification.
