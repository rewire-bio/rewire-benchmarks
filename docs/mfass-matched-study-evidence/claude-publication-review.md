# Publication review: issue #19 matched-annotation bundle

Reviewer: Claude (automated), 2026-09-25. Same reviewer as `claude-scientific-review.md`.
This is automated review, not human review. Read-only: no model inference, no reruns of any
condition, no edits outside this file, and no commits or posts. The throwaway scripts are
under `$TMPDIR/pub-review` on the SSD.

**VERDICT: FINDINGS.** One blocking item and four non-blocking items. The blocking item is a
data-publication policy point, not a scientific or fidelity defect. Every published file is
faithful to the raw run, every number is correct, and the tests pass. None of the fixes
changes a frozen scientific result.

## Files reviewed

Base `b52a3daef6e85ee869e70732a586355f83dc30fb`, plus the uncommitted publication diff (32
files: 29 new marked intent-to-add, 3 modified).

| File | SHA-256 |
|---|---|
| `benchmarks/mfass/results/matched-annotation-v1/README.md` | `f4131ffd1fee8470608452a7635facf26b0c5c1018d460d9db3400cd2b61a1b7` |
| `.../matched-annotation-v1/SHA256SUMS` | `1157cc0de89591fb6fc7ca52f3090207df77dde6421bd8573e2379bb91e5e6b8` |
| `.../matched-annotation-v1/provenance.json` | `9e31cf281c6368437eeb9f4b0a23e4e0bad78cce28a7276be90675b50f394275` |
| 25 bundled run files | as listed in `provenance.json` and `SHA256SUMS`; all verified (section 1) |
| `benchmarks/mfass/tests/test_matched_annotation_v1_bundle.py` | `d5bbe0912349ac75136e812958f8920a7594955ef74d7cfdc1e7742c6c9ed2f9` |
| `.gitattributes` | `7ce10a948247783369b2c174cf34a8616907cdafe225f945ea86d0bbac1eebe1` |
| `benchmarks/mfass/README.md` | `ec1c776de2d49bac3ba8a161c8d9540356d4675ea003ac6bc727c9a86438b9c6` |
| `docs/mfass-specialist-comparison.md` | `cb0f9bc175a6a0b149f6eeb6965f2a4cae4e186346f5052b954151f789f2b853` |
| `workbench/issue-19/export_bundle.py` (exporter) | `6539643a1f4b60c8a8a3ac70bc4b48ab4c47dfc6e28051babffebba553acf4ca` |
| `workbench/issue-19/claude-scientific-review.md` (raw review) | `81d9abb7a365cf5c257b489c1ce6cf9d1aee62fe9f81fe34ef71d65e97b53f3e` |
| `docs/mfass-matched-study-registration.md` | `7f03c13d3c7fa73ffe91d3170fe51f06a277d34b5745476ce45ca8a7cece9162` (unchanged) |

## Commands run

All with `env.sh` sourced and `$STUDY_DATA/envs/dev/bin/python`, from the repository root.

- `$TMPDIR/pub-review/fidelity.py` (`cd109d59...`) checks every bundled file against its raw
  original, the recorded provenance hashes, reversal of the path substitutions,
  `SHA256SUMS`, private-string scanning and the prediction headers.
- `$TMPDIR/pub-review/rerun_compare.py` (`bdbbadc4...`) runs `mfass.compare` on the three
  published prediction pairs with the registered settings, writes to `$TMPDIR`, and diffs the
  output against the published contrasts.
- `git diff --check HEAD`, `git diff --check`, `git check-attr -a`, and `git hash-object
  --path` against `--no-filters` on all 12 bundle TSVs.
- `git diff --stat HEAD -- benchmarks/mfass/src packages docs/mfass-matched-study-registration.md docs/mfass-specialist-comparison-evidence benchmarks/mfass/splits`,
  then re-hashing the 31 artifacts in `validation.json`.
- `MFASS_REQUIRE_GATES=1 pytest packages/rewirebench/tests tests benchmarks/mfass/tests -q -rs`.

## 1. Bundle fidelity: PASS

- **Coverage:** all 25 raw sources the exporter selects appear in `provenance.json`, with no
  extras. Every `raw_sha256` equals the raw original under `$STUDY_DATA/runs/v1`,
  `workbench/issue-19/manifest-v1.json` or my raw review, and every `public_sha256` equals
  the bundled file.
- **16 files marked "identical" are byte-identical:** all predictions, unscored and checkpoint
  tables, `report.json`, `report.md`, `verification.json` and the review.
- **9 files marked "path substitution"** are the four result JSONs, three contrasts,
  `events.jsonl` and the manifest.
  - None of the raw files already contained a `$STUDY_DATA`, `$REPO` or `$MFASS_DATA` token,
    so reversal is unambiguous.
  - Replacing the tokens with the exporter's private prefixes reproduces the raw bytes, which
    I confirmed by hash.
  - Re-applying the substitution to the raw file reproduces the public bytes.
  - The recorded counts match.
- **`SHA256SUMS`:** lists all 27 files other than itself, including `README.md`, and every
  hash is correct.
- **No private paths:** there are no `/Volumes/`, `/Users/`, `/private/` or `/home/` paths,
  no username and no email in any bundled file. The only match on my credential words is the
  key `substitution_tokens` in `provenance.json`.
- **Prediction tables** use exactly `id  group  label  score`, the same header as the
  archived v2 prediction tables.
- **Cohort-derived fields beyond id/group/label/score.** These go past the brief's criterion
  (finding 1):
  - Each SpliceAI `checkpoint.tsv` `raw` field starts with SpliceAI's `ALLELE`, which is the
    variant's alternate base. It is present for all 8,297 scored variants in S0 and S1.
  - The P0/P1 `unscored.tsv` and checkpoint reasons print both reference bases for the 23
    mismatches: the cohort's (`variant file (ref base: C)`) and the FASTA's.
  - There are no coordinates, sequences or other cohort columns. No committed data table held
    alleles before this diff.

## 2. Bundle README: PASS on content; findings 2 and 3 on wording

Every number matches the raw `report.json`, the contrast JSONs and my independent
recomputation:
- condition metrics and tie counts;
- 8,297 of 8,324 scored, 314 of 315 positives, 460 of 463 groups;
- all nine observed deltas and intervals;
- 22.5% and 44.4% of draws with a difference of exactly zero;
- the P1 tie at 0.54 (3 scores for 2 slots, 1 positive);
- the run window and 50,949 s (50,948.7 recorded);
- an Apple M4 CPU (confirmed with `sysctl`).

Coverage of my scientific-review findings:

| Finding | Where it is covered |
|---|---|
| 1. Tie-order ambiguity in the P1 − P0 P@100 | Lines 99-103 |
| 2. Lower bounds of 0.000 include zero | Lines 96-98 |
| 3. `report.md` omissions | Covered only indirectly (finding 3 here) |
| 4. SpliceAI object address in reason text | Lines 62-65 |

Claims the README correctly avoids:
- It does not claim architecture isolation (lines 111-117).
- It does not claim confirmatory status (lines 7-8, 109-110).
- It does not claim human review (line 24).
- It does not apply the improvement rule (lines 120-123).
- It labels the historical runs as context only (lines 126-130).

Relative links resolve to the right files: `../../../../docs/`, `../../patches/` (which
contains `LICENSE-pangolin-GPL-3.0` and a README) and `reviews/`.

Reproduction commands: the `mfass.compare` command reproduces all three published contrast
reports field for field, apart from `baseline_file` and `candidate_file`, as the README says.
The `pyproject.toml` workspace includes `benchmarks/*`, so `uv run python -m mfass.compare`
resolves. The preflight evidence file does record 6 of 6 README example records reproduced.
The shell block has one defect (finding 2).

## 3. Bundle test: PASS

`test_matched_annotation_v1_bundle.py` checks:
- `SHA256SUMS` completeness and correctness;
- that no private path remains;
- for each condition: 8,324 unique rows, coverage and all six metrics recomputed to `1e-12`
  from the published predictions, and the predictions hash bound to the result JSON;
- for each contrast: the common count, common-ID hash, common positives, both input hashes,
  the seed and the draw count.

It runs without optional data (8 passed alone) and would catch any edit to a published table
or result. It does not rerun the bootstrap; the README's `mfass.compare` command covers that,
and I ran it.

## 4. `.gitattributes`: PASS

- The new rule applies only to `benchmarks/mfass/results/matched-annotation-v1/**/*.tsv`.
  `git check-attr` shows no attributes on the bundle README, the Python tests or the archived
  `results/pangolin-maskFalse.predictions.tsv`.
- The existing patch rule is unchanged.
- `-text` preserves bytes: `git hash-object --path` equals `--no-filters` for all 12 bundle
  TSVs.
- `whitespace=-blank-at-eol,cr-at-eol` relaxes only those files.
- `git diff --check HEAD` and `git diff --check` are both clean.

## 5. Doc edits: PASS, with finding 4

- `docs/mfass-specialist-comparison.md` changes two status lines to "executed as registered"
  and links the bundle. It keeps the pre-execution protocol text and states that this is the
  text as written before execution. This is accurate and minimal.
- `benchmarks/mfass/README.md` adds a short summary and a link, and keeps the archived-run
  erratum. It is accurate except for one ambiguous phrase (finding 4).

## 6. Frozen files: PASS

- No change under `benchmarks/mfass/src`, `packages`, `benchmarks/mfass/splits`,
  `docs/mfass-specialist-comparison-evidence` or the registration. The registration hash is
  still `7f03c13d...`.
- All 31 artifacts in `validation.json` match their recorded SHA-256.

## 7. Full test suite: PASS

`MFASS_REQUIRE_GATES=1 pytest packages/rewirebench/tests tests benchmarks/mfass/tests -q -rs`
in `envs/dev` gave **648 passed, 9 skipped, 0 failed** in 15.4 s. All 9 skips are
undistributed MFASS source data: 7 in `tests/test_sdk_mfass.py`, 1 in `test_assay_pair.py`
and 1 in `test_prediction_safety.py`. None is in a gate module, and no gate test skipped. The
17 warnings are sklearn single-class warnings from unrelated ProteinGym tests.

## Findings

1. **BLOCKING (publication policy): cohort-derived allele data in the bundle.** This breaks
   the stated criterion that the bundle hold no cohort source data beyond id/group/label/score.
   - SpliceAI checkpoints (S0, S1) include the alternate base for 8,297 variants, in the
     `ALLELE` field of the raw output.
   - The Pangolin unscored and checkpoint reasons (P0, P1) include the cohort's reference base
     for the 23 mismatches.

   The exposure is small: single bases keyed by MFASS ID, with no coordinates or sequence.
   But the preflight records the MFASS cohort's reuse terms as unreported and not
   redistributed. Choose one, without changing any frozen result:
   - (a) Record an explicit decision that single-base alleles may be published. Name these
     fields in the README's Files row for `checkpoint.tsv` and the unscored tables.
   - (b) Omit the four `checkpoint.tsv` files and the two Pangolin `unscored.tsv` files from
     the public bundle. Keep their raw hashes in `provenance.json` as "not copied", as is
     already done for the step logs, and state that the verified predictions and
     `verification.json` remain. The bundle test and `SHA256SUMS` would need regenerating.
2. **Non-blocking: the README shell block breaks when pasted.** Line 166 runs `cd … && shasum`,
   so lines 167-172 then run from inside the bundle directory, where their repo-relative paths
   fail. Minimal fix: use a subshell, `(cd benchmarks/mfass/results/matched-annotation-v1 &&
   shasum -a 256 -c SHA256SUMS)`.
3. **Non-blocking: scientific-review finding 3 is only implicit.** Add one sentence near line
   139 or in "Reading the results": "`report.md` was generated before review. It lists tie
   counts but does not explain their effect on P@100 or the 0.000 lower bounds; see Reading
   the results."
4. **Non-blocking: ambiguous wording in `benchmarks/mfass/README.md`.** The line "Masking
   established no change in AUROC or P@100" can be read as evidence of no change. Minimal fix:
   "Masking did not establish a change in AUROC or P@100."
