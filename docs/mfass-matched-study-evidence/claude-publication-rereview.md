# Publication re-review: issue #19 matched-annotation bundle

Reviewer: Claude (automated), 2026-09-25. Same reviewer as `claude-publication-review.md`.
This is automated review, not human review. Read-only: no inference, no edits outside this
file, and no commits or posts. The throwaway scripts are under `$TMPDIR/pub-rereview` and
`$TMPDIR/pub-review` on the SSD.

**VERDICT: PASS.** All four findings from `claude-publication-review.md` are resolved, and
nothing blocking remains. Two optional non-blocking suggestions follow at the end.

## Files reviewed

Base `b52a3da` plus the revised uncommitted diff: 3 modified files and 26 new files marked
intent-to-add. The bundle has 24 files; no `checkpoint.tsv` is present.

| File | SHA-256 |
|---|---|
| `workbench/issue-19/export_bundle.py` (revised) | `53200a3cafad1bf6ad2742eeca70b425a07ce5bcc2ceba2664a0619026ed8161` |
| `.../matched-annotation-v1/README.md` | `d89341a7e79a449b2e714c769e3178bf7ad65f231fe736ae2e1379b4318e24f6` |
| `.../matched-annotation-v1/SHA256SUMS` | `a93e35702c7313e5ba604e009339eeebe025c5d6da65fb61dd94a18273cfdc3d` |
| `.../matched-annotation-v1/provenance.json` | `6304e9625e793f24cccdc2669d092e6cdcc7a508a7f789967e726816fffde062` |
| `benchmarks/mfass/tests/test_matched_annotation_v1_bundle.py` | `e97b66ee2f57e584045dc320f748360eaf7c42a54e923a757c0585bf3105c64d` |
| `benchmarks/mfass/README.md` | `4c5f73c35dc5d6fd9aece0a84c42d59553f87962f47a4d0adfe6d25c3c9eaf9b` |
| `docs/mfass-specialist-comparison.md` | `cb0f9bc175a6a0b149f6eeb6965f2a4cae4e186346f5052b954151f789f2b853` (unchanged since the first review) |
| `.gitattributes` | `7ce10a948247783369b2c174cf34a8616907cdafe225f945ea86d0bbac1eebe1` (unchanged) |
| `docs/mfass-matched-study-registration.md` | `7f03c13d3c7fa73ffe91d3170fe51f06a277d34b5745476ce45ca8a7cece9162` (unchanged) |

## Commands run

All with `env.sh` sourced and `$STUDY_DATA/envs/dev/bin/python`, from the repository root.

- `$TMPDIR/pub-rereview/fidelity2.py` (`59558196...`). For each of the 23 raw sources, it
  independently applies the recorded path substitutions and then the recorded redaction regex
  from `provenance.json`, and compares the result with the public bytes. It also checks the
  recorded raw and public hashes and redaction counts, the `not_copied_raw_sha256` hashes
  against the raw files, and `SHA256SUMS`. Finally it scans every bundled file for private
  strings, cohort reference bases, SpliceAI allele fields (`[ACGTN]+|ENSG…`, `base|symbol|score`),
  cohort allele or sequence column names, and checkpoint files.
- `$TMPDIR/pub-review/rerun_compare.py` reruns `mfass.compare` on the three published
  prediction pairs.
- `git diff --check HEAD`, `git diff --check`, `git check-attr -a`, and `git hash-object
  --path` against `--no-filters` on all 12 bundle TSVs.
- `git diff --stat HEAD` on the frozen paths, then re-hashing the 31 artifacts in
  `validation.json`.
- `MFASS_REQUIRE_GATES=1 pytest packages/rewirebench/tests tests benchmarks/mfass/tests -q -rs`.

## Results

**Fidelity: PASS.** All 23 published sources are in `provenance.json`, with no extras. Every
file reproduces exactly from its raw original under its recorded transformation:

| Transformation | Files |
|---|---|
| Identical | 10: the 4 prediction tables, the S0 and S1 `unscored.tsv`, `report.md` and the scientific review |
| Path substitution | 9: the 4 result JSONs, 3 contrasts, `events.jsonl` and the manifest |
| Cohort reference-base redaction | 4: the P0 and P1 `unscored.tsv` (23 redactions each), `report.json` and `verification.json` (46 each) |

- Recorded redaction counts equal the counts I recomputed.
- All 12 `not_copied_raw_sha256` entries (4 checkpoints, 8 step logs) equal the raw files.
- `SHA256SUMS` lists every file except itself, with no missing or wrong entries.

**No private data or cohort alleles: PASS.**
- No private path, username or email.
- No `variant file (ref base: <base>)`.
- No SpliceAI `ALLELE|gene|…` field or other base-pipe-score pattern.
- No `ref_allele`, `alt_allele`, `snp_position` or sequence columns.
- No `checkpoint.tsv`.
- The only allele-like content left is Pangolin's `FASTA (ref base: X)` for the 23
  mismatches. That is GRCh38 reference sequence, not cohort data, and keeping it is stated in
  the README and the exporter.

**README: PASS.**
- The condition table, the tie counts, all nine contrast estimates and intervals, 8,297 of
  8,324, 314 of 315 positives, 460 of 463 groups, the run window, 50,949 s and the M4 CPU all
  match the raw values, unchanged from my previous check.
- The new provenance text is accurate:
  - the counts are correct (9 substituted, 4 redacted, 4 checkpoints and the step logs not
    published);
  - the redacted files are named correctly;
  - `verification.json` did record checkpoint agreement;
  - the Pangolin `unscored.tsv` public hashes do differ from the raw hashes recorded in the
    result JSONs;
  - the prediction tables do equal the recorded `output_tables` and contrast input hashes.
- The reproduction block now works when pasted. Rerunning `mfass.compare` reproduces all
  three contrasts apart from `baseline_file` and `candidate_file`, as stated.

**Findings from the previous review:**

| # | Finding | Status |
|---|---|---|
| 1 | Blocking: cohort alleles in the bundle | **Resolved.** Checkpoints are not published, and the cohort reference base is withheld under a recorded, reproducible regex. The test asserts both. |
| 2 | README shell block | **Resolved.** It now uses a subshell. |
| 3 | `report.md` omissions only implicit | **Resolved.** The Files table states that `report.md` predates review and does not explain the tie or 0.000 points. |
| 4 | Ambiguous wording in `benchmarks/mfass/README.md` | **Resolved.** It now reads "Masking did not establish a change in AUROC or P@100". |

**Frozen and historical files: PASS.** Nothing changed under `benchmarks/mfass/src`,
`packages`, `benchmarks/mfass/splits`, `docs/mfass-specialist-comparison-evidence` or the
registration. All 31 artifacts in `validation.json` match.

**Git: PASS.** `git diff --check HEAD` and `git diff --check` are clean. The bundle TSVs carry
`-text` plus the relaxed whitespace rule and are stored unconverted (12 of 12). The README,
the tests and the archived TSVs carry no attributes.

**Test suite: PASS.** With `MFASS_REQUIRE_GATES=1` in `envs/dev`: **648 passed, 9 skipped, 0
failed.** All 9 skips are undistributed MFASS source data outside the gate modules.

## Optional suggestions (non-blocking)

1. The README Files row for `provenance.json, SHA256SUMS` still says "Path substitutions and
   raw versus public hashes". It could add "the allele redaction pattern and the hashes of
   unpublished files" to match the new provenance section.
2. The bundle test guards against the cohort reference base and against checkpoints. A
   one-line assertion against SpliceAI allele fields (for example
   `re.search(r"[ACGTN]+\|ENSG\d", text)`) would also catch a future regression in which raw
   upstream output is republished some other way.
