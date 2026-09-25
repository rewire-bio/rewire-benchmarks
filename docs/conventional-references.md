# Conventional references for ProteinGym and DART Task 1

Issue [#13](https://github.com/rewire-bio/rewire-benchmarks/issues/13) adds two
sourced, label-free reference methods. Both are implemented and tested on
synthetic fixtures against pinned upstream code. Neither has been run on
biological benchmark data, and neither has a released measurement.

| Baseline ID | Code | Default configuration | Synthetic and upstream checks | Biological evaluation | Released measurement |
|---|---|---|---|---|---|
| `proteingym-evcouplings-independent-v1` | Implemented | Blocked until a prepared artifact is supplied | Executed | Not executed | None |
| `dart-h12core-fimo-hit-count-v1` | Implemented | Blocked until FIMO and H12CORE are supplied | Executed | Not executed | None |

The registry records these states separately (`implementation_status`,
`configuration_status`, `validation_status`, `biological_evaluation_status`,
`measurement_release_status`). The earlier proposals stay blocked:
`proteingym-conventional-pending-v1` records `replaced_by` the EVCouplings entry.
`dart-conventional-pending-v1` stays open because our source review established
no published conventional zero-shot scalar Task 1 baseline; the motif count is
recorded as a related Rewire method, not its replacement.

Without artifact options, `run-baselines` reports both entries as `blocked` and
writes no scores. With options, a missing or mismatched artifact makes the entry
`failed` with its exception class, never a zero score. Option values can be local
paths, so the plan and manifest record only the option names.

## ProteinGym: EVCouplings independent model

**Identity.** ProteinGym's `Site_Independent` row: pinned `score_mutants.py`
loads a plmc_v2 `CouplingsModel`, calls `to_independent_model()` and records
`prediction_independent` with directionality +1. For each model site, the fields
are refitted from the model's stored single-site frequencies f_i, N_eff and
lambda_h by zero-initialised BFGS:

    h*_i = argmin_h  N_eff (log sum_a exp h(a) - sum_a f_i(a) h(a)) + lambda_h sum_a h(a)^2

The couplings are then cleared. A variant scores sum_i h*_i(mutant) - h*_i(wild type);
higher is fitter, and multiple substitutions add. This is neither a pseudocount
profile nor the epistatic model with J cleared; a test shows the latter
gives different scores.

**Pins.** EVCouplings `e1362407a0b65d63ca07df55f44cb17b0a3722b7`
(`evcouplings/couplings/model.py`, SHA256 `6422cfc7…fa050`) and ProteinGym
`144fe22b07dfaeec2b366f2346203a9838a55b4c` (`score_mutants.py` `9802fcd2…6050`,
`calculations.py` `07e20591…e6c`, `config.json` `7cd239d1…608`). Receipts are
in `resources/proteingym/sources.json`. These are implementation pins, not
evidence of the dependency versions ProteinGym used historically. The plmc_v2
reader and field objective are adapted from EVCouplings (MIT, Copyright (c) 2017
EVcouplings development team); the notice is packaged as
`resources/proteingym/EVcouplings-LICENSE.upstream`. No EVCouplings models or CNS
scripts are redistributed.

**Inputs and fitting.** At scoring time the adapter reads only `assay_id`,
`wild_type_sequence`, `mutant` and `mutated_sequence`, and it has no `fit`
method. The only external information is a label-free plmc_v2 model per assay.
Refitting the independent fields is a declared preparation step, done before
evaluation from the model's stored statistics. DMS labels and benchmark variants
are never read. This package performs no MSA search or Potts training.

**Preparation.** Write a manifest for reviewed models:

```json
{
  "schema": "rewire-evcouplings-independent-manifest-v1",
  "dms_labels_used": false,
  "assays": {
    "AMFR_HUMAN_Tsuboyama_2023_4G3O": {
      "model_path": "models/AMFR_HUMAN.model", "model_sha256": "<64 hex>",
      "format": "plmc_v2", "precision": "float32",
      "model_id": "AMFR_HUMAN", "msa_start": 1,
      "provenance": {
        "model_source": "...", "model_retrieval_receipt": "...",
        "alignment_release": "...", "alignment_sha256": "<64 hex>",
        "query_interval": "...", "filtering_policy": "...",
        "sequence_weights": "...", "model_generation_config": "..."
      }
    }
  }
}
```

```bash
rewirebench prepare-baseline-artifact proteingym-evcouplings-independent-v1 \
  --manifest manifest.json --output evc-independent.json
rewirebench run-baselines --prepared ./prepared-amfr --output ./runs \
  --baseline proteingym-evcouplings-independent-v1 \
  --baseline-options '{"proteingym-evcouplings-independent-v1":
     {"artifact": "evc-independent.json", "artifact_sha256": "<from preparation>"}}'
```

The adapter is also available directly as
`rewirebench.adapters.evcouplings_independent:EVCouplingsIndependent` for
`rewirebench run` with the same two options.

Preparation checks the following, and fails otherwise:

- The model file SHA256 matches. The file is a complete plmc_v2 model of the
  declared precision: an exact byte length, finite parameters, frequencies that
  sum to one, positive lambda_h and N_eff, a unique alphabet and a strictly
  increasing index list. Legacy plmc_v1 files and mean-field models are rejected.
- `model_id` equals the pinned scorer's rule: the UniProt ID, except
  `RASK_HUMAN_Ursu_2020` and `F7YBW7_MESOW` to `F7YBW8_MESOW`. Neither exception
  occurs in the pinned v1.3 reference, so both are inactive for this protocol.
- `msa_start` equals the pinned reference. Model index m maps to assay position
  m + MSA_start - 1, which is ProteinGym's offset `-(MSA_start - 1)`.
- Every model position lies inside the assay sequence, and its target residue
  equals the assay wild type.
- All eight provenance fields are present. Their contents are recorded as
  declared, not verified.

The artifact records the refitted fields per assay position, the model metadata
(L, alphabet, target, index list, N_eff, lambda_h, lambda_J, theta), the
optimizer status per site, source pins, the manifest hash and the NumPy/SciPy
versions.

**Optimizer acceptance.** The objective, zero start and `fmin_bfgs` call are
upstream's; upstream ignores the terminal status, but preparation checks it for
each site. The objective and gradient are recomputed at the returned fields and
must be finite. SciPy status 0 (converged) is accepted if the recomputed
max|gradient| is at most SciPy's default gtol, 1e-5. Status 2 (precision loss)
is accepted only if the stationarity residual max|gradient| / N_eff is at most
1e-6, in site-frequency units. Any other status (iteration limit, NaN) fails
preparation. Precision loss is routine at large N_eff: in synthetic surveys it
occurred at most sites with N_eff of 5e4 to 5e5, with residuals of at most 3e-9.
Each site's status, objective, max|gradient| and residual are stored in the
artifact, and the run provenance lists precision-loss positions and the largest
residual per assay. The 1e-6 value is a Rewire preparation acceptance guard on
the optimizer's stationarity, in site-frequency units. It is not a bound on
field or prediction error, and it says nothing about biological accuracy.
Agreement with upstream scores rests on the parity comparisons below.

**Loading.** A matching SHA256 is required but is not enough. The adapter
rechecks the artifact's meaning before scoring and rejects contradictions:

- the `dms_labels_used: false` declaration, the fixed direction, the method,
  and the reference and source pins;
- each assay's pinned wild type, `MSA_start`, offset and model ID;
- model metadata consistency: a unique alphabet matching the field rows, and a
  target and strictly increasing index list that map onto the wild type;
- the recorded plmc_v2 header, under the binary reader's rules: L, num_symbols,
  N_valid, N_invalid and num_iter must be integers with the same minimums
  (booleans and floats are rejected); theta, lambda_J and lambda_group must be
  finite; N_eff and lambda_h must be positive and finite;
- field keys that are exactly the canonical mapped positions (no aliases such as
  `0538`, no extra or missing positions) and finite rows;
- accepted optimizer states, with nonnegative norms and each recorded
  stationarity residual equal to its recorded max|gradient| / N_eff;
- the declared provenance fields.

Duplicate JSON keys and unknown fields are also rejected.

**Coverage and failures.** A variant is unscored, with a reason, if any
substitution falls outside the model index list (including before `MSA_start` or
in an index-list hole), uses a residue outside the model alphabet, or belongs to
an assay without a model. The whole multiple mutant is unscored in each case. The
SDK keeps the original denominator, so the report is `partial`. Upstream raises
`ValueError` on these mutants, which aborts its whole assay; the adapter reports
them per variant instead, and the parity receipt records the upstream errors. A
mutant whose wild-type residue disagrees with the sequence, a wrong wild-type
sequence, a changed or contradictory artifact, or unexpected input fields raise
errors.

**Limitations.**

- No reviewed EVCouplings model or MSA preparation chain has been obtained for
  any assay. The method cannot run on real data until one is.
- Four v1.3 assays have `MSA_start > 1` (KCNH2, SCN5A, POLG_HCVJF,
  A0A140D2T1_ZIKV). If a real model's index list uses full-sequence numbering,
  the pinned offset maps it to the wrong residues. Preparation then fails the
  wild-type check rather than scoring. This has to be checked against an actual
  model.
- AMFR (47 residues, `MSA_num_cov` 41) may not be fully covered; no coordinate
  audit has been done. Scoring one assay is not a 217-assay suite result.
- Fields depend on SciPy's BFGS. The receipt used SciPy 1.13.1 upstream and
  the adapter SciPy 1.17.1; the fixture agreed to 2.2e-16, including sites
  where both reported precision loss. Other versions may differ slightly; the
  artifact records the versions used.
- The optimizer status rule is stricter than upstream, which ignores the status.
  A model that upstream would score can therefore fail preparation here.

## DART Task 1: HOCOMOCO v12 CORE FIMO total motif hits

**Identity.** `H12CoreFimoHitCount` is the registered method. It is a Rewire
Task 1 aggregation using a sourced scanner. FIMO (MEME
Suite 5.5.9) scans each supplied sequence on both strands with all 1,443
HOCOMOCO v12 H12CORE motifs. The score is the total number of emitted hits:
each hit counts 1, and no hits scores 0. Higher means more recognised motif
occurrences. This is **not** a published DART Task 1 baseline. DART's motif
baseline belongs to Task 3, and this scalar is newly specified with no measured
performance. The same scan and count with caller-supplied motifs is
`FimoHitCount`, whose identity is `fimo-total-hit-count-caller-motifs-v1`.
It requires a `motif_provenance` description and never reports the H12CORE
baseline ID or the HOCOMOCO training statement. Only `H12CoreFimoHitCount` does,
after checking the file digest, motif count and motif-ID digest.

**Fixed configuration.** `fimo --text --verbosity 1 --thresh 1e-4
--motif-pseudo 0.1 --bfile uniform-background.txt <H12CORE> <fasta>`. The
background is the packaged uniform A=C=G=T=0.25 file (SHA256 `b496ad5c…e223`),
fixed before any DART score was seen. There are no q-values, no motif subset,
no positional prior, no per-batch background and no learned aggregation. The
full sequence is scanned, and overlapping and both-strand hits all count. From
the 5.5.9 source (`fimo.c`), a hit is reported when its p-value, rounded to 10
significant digits, is `<=` the threshold. Windows containing a non-ACGT symbol
are skipped. The adapter counts the rows FIMO emits and never re-filters the
printed three-digit p-values.

**Why text mode.** FIMO's default stored-score mode caps stored hits
(`--max-stored-scores`, default 100,000) and can discard matches. On a 350-base
poly-A fixture, both modes were run with the same `--max-stored-scores 50`:
stored mode emitted 0 rows and text mode all 343. Text mode prints each hit
directly and never applies the cap.

**Inputs.** Only `id` and `sequence`. FASTA names are batch positions, so IDs
never reach FIMO. Pairing, element/control role, labels, chromosome and order
are never seen.

**Ambiguous bases.** DART inputs use `N` for missing bases. FIMO's own rule
applies: windows overlapping `N` are not scored. A sequence without any
all-ACGT window at least as wide as the narrowest motif (7 bases in H12CORE) is
unscored rather than scored 0. The adapter's provenance records scanned
sequences, those containing non-ACGT symbols, valid windows at the narrowest and
widest motif width, total hits and zero-hit sequences.

**Artifacts and options.** `H12CORE_meme_format.meme` from the
[official download](https://hocomoco12.autosome.org/final_bundle/hocomoco12/H12CORE/formatted_motifs/H12CORE_meme_format.meme)
must have SHA256 `3d9c47dc396b3ba4e278cdfd29d526c1eb2fbad4895b5e2dc345f535f98c5f12`,
1,443 motifs and motif-ID digest `7bff4fc5…c308`. These are observed bytes
retrieved on 24 September 2026, not a comparison with DART's Synapse copy
`syn60756095`. FIMO must report `5.5.9` and match the `fimo_sha256` you record
for your build. A build is platform-specific, so there is no universal binary
hash. The source tarball SHA256 is `0406fb7b…fd12`. On macOS, run `configure`
from a path without spaces.

The `fimo` and `motifs` options are filesystem paths, resolved against the
current directory before hashing; a bare `fimo` never triggers a PATH search. The
resolved executable is the one that runs. Motif and background bytes are
verified at construction and scanned from private copies. Immediately before
every FIMO call, the executable, motif file and background are hashed again. If
any has changed since construction, the scan stops with an error.

```bash
rewirebench run-baselines --prepared ./dart-prepared --output ./runs \
  --baseline dart-h12core-fimo-hit-count-v1 \
  --baseline-options '{"dart-h12core-fimo-hit-count-v1":
     {"fimo": "/opt/meme/bin/fimo", "fimo_sha256": "<your build>",
      "motifs": "/refs/H12CORE_meme_format.meme"}}'
```

`rewirebench.adapters.fimo_motifs:H12CoreFimoHitCount` takes the same options
for `rewirebench run`.

**Failures.** A missing, non-executable, wrong-version or wrong-hash scanner, a
changed or malformed motif file, a changed background, a file replaced after
construction, a nonzero exit status,
any stderr other than FIMO's fixed text-mode q-value notice, an unparseable
row, or a hit naming an unknown sequence all raise errors. There is no GC,
composition or Markov fallback.

**Evaluator interaction.** The DART evaluator is unchanged: equal counts are
incorrect, not half-correct. Its result now reports `pair_diagnostics.tied_pairs`.
With tie probability t, symmetric differences give a null accuracy of (1 - t)/2,
not 0.5.

**Limitations.** HOCOMOCO motifs were learned from external experimental
TF-binding data (ChIP-seq, HT-SELEX and related assays). Their overlap with the
ENCODE regions in DART is not quantified. Related TF subtypes are counted
separately and can be overweighted; no subset will be chosen after evaluation.
DART's own input authenticity and complete-denominator gaps (see
[DART-Eval](dart-eval.md)) still apply.

## Validation performed

The tests are `packages/rewirebench/tests/test_evcouplings_independent.py`,
`test_fimo_motifs.py` and `test_baselines.py`.

- **EVCouplings parity.** Synthetic plmc_v2 models for three pinned-reference
  assays:
  - KCNH2: `MSA_start` 535, a gapped alphabet without W and an index-list hole;
  - AMFR: 20 amino acids and two non-contiguous positions;
  - ARGR: N_eff 4e5, where BFGS reports precision loss at two of four sites.

  `scripts/baseline_parity/evcouplings_upstream_receipt.py` runs the unmodified
  pinned `score_mutants.py` command and `to_independent_model()` in a separate
  Python 3.11 environment. It observes each site's BFGS status through a
  pass-through wrapper that returns the identical iterate. The committed
  `upstream-receipt.json` records the outputs. Adapter scores and fields agree
  within 1e-6 (observed maximum difference 2.2e-16), and the adapter's
  per-site statuses equal upstream's. The receipt also shows upstream raising on
  each invalid mutant. Injected optimizer results test rejection of unusable
  terminal states: iteration limit, NaN status, a false success at unmoved zeros,
  precision loss far from stationarity, and nonfinite objective or gradient.
  Artifacts altered and then correctly rehashed are rejected by the semantic
  checks in 32 cases. Tests cover direction on biased sites, an analytic near-zero for equal
  frequencies, exact additivity and wild-type mismatch at preparation. Leakage
  and invariance tests show labels do not change predictions, extra fields are
  rejected, and results do not depend on batch or order. Invalid models,
  manifests and artifacts are rejected.
- **FIMO parity.** Hand-authored motifs (a directional site, a palindrome and
  poly-A) and 350-base sequences with planted forward, reverse, palindromic,
  ambiguous, no-hit, unscorable and high-hit cases.
  `scripts/baseline_parity/fimo_receipt.py` invokes FIMO 5.5.9 directly; its
  receipt includes H12CORE hits on seeded random sequences. With `REWIRE_FIMO` and
  `REWIRE_H12CORE` set, the tests compare each adapter tuple (motif, start, stop,
  strand, score, p-value) with a separate FIMO call and with the receipt. They
  also check the following:
  - Threshold equality on a controlled site. A lone 8-base poly-A match has
    p = 4^-8, which FIMO rounds to 1.525878906e-05, the same double as that
    threshold. It is emitted at `--thresh 1.525878906e-05` and not at
    `1.525878905e-05`.
  - Hits at 1e-4 are the subset of a `--thresh 1` scan.
  - Text mode with `--max-stored-scores 50` still emits all 343 poly-A hits.
  - Scores do not change with batching, order, ID renaming or reverse
    complement.
  - Labels, pair roles and source indices never reach scoring.
  - A bare `fimo` name runs the resolved local file, not a different file on
    PATH.
  - A motif file or executable replaced after construction stops the scan.

  These scanner checks use synthetic motifs; the registered threshold stays
  1e-4. Without the environment variables the live tests skip, and a skip is not
  parity evidence. Failure handling uses scripted stand-in executables, which
  are not parity evidence either.
- **Receipt build provenance.** `fimo_receipt.py` records build provenance only
  from a build receipt whose `fimo_sha256` matches the executable it runs. It
  refuses a mismatched receipt. Without a receipt it lists reference build
  instructions, labelled as not observed. The committed receipt is bound to
  `fimo-build-receipt.json`, which describes the local macOS arm64 build of
  24 September 2026.

## Proposed later validation (not executed)

These runs need separate approval and must not be run as part of this issue.

| Candidate | Bound and conditions |
|---|---|
| ProteinGym | At most one preselected assay. Prefer the 2,972-variant AMFR cohort only after obtaining a reviewed EVCouplings model with MSA provenance and auditing its coordinates. With no model, do not run. If AMFR coverage is incomplete, keep the partial status and compare exact common IDs with the null separately, or preselect one fully covered assay before seeing predictions. One assay is not a suite result. |
| DART Task 1 | At most 1,000 source pairs from an authenticated, already available `syn64314109.1` HDF5 file, chosen by a documented rule before inspecting scores. Reuse the supplied controls and IDs; do not regenerate shuffles. This is smoke scope under the current SDK and is not an official Task 1 result. Report tie frequency. Stop if provenance or prerequisites are missing. |
