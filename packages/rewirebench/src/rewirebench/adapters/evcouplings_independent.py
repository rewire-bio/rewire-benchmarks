"""EVCouplings independent (site-wise) model with ProteinGym's scoring convention.

Method identity: ``proteingym-evcouplings-independent-v1``. ProteinGym's pinned
``EVmutation/score_mutants.py`` loads a plmc_v2 ``CouplingsModel``, calls
``to_independent_model()`` and records ``prediction_independent``
(``Site_Independent``, directionality +1). ``to_independent_model`` refits the
fields at every site from the stored single-site frequencies f_i, N_eff and
lambda_h by zero-initialised BFGS, minimising

    N_eff * (log sum_a exp h(a) - sum_a f_i(a) h(a)) + lambda_h * sum_a h(a)^2

and then clears J. The score of a variant is sum_i h*_i(mut) - h*_i(wt). It is
not a pseudocount profile and not the epistatic model with J cleared.

Two separate steps keep provenance and protocol boundaries explicit:

1. ``prepare_artifact`` reads an external, label-free plmc_v2 model named in a
   reviewed manifest, checks its hash, metadata and the assay coordinate map
   against the pinned ProteinGym reference, refits the independent fields and
   writes a derived JSON artifact. No DMS labels or benchmark variants are read.
2. ``EVCouplingsIndependent`` loads that artifact (hash required) and scores
   protocol inputs. Variants touching a position outside the model index list or
   a residue outside the model alphabet are unscored with a reason, never zero.

Pins: EVCouplings e1362407a0b65d63ca07df55f44cb17b0a3722b7
(evcouplings/couplings/model.py, sha256 6422cfc7...fa050) and ProteinGym
144fe22b07dfaeec2b366f2346203a9838a55b4c. The plmc_v2 reader and independent-field
objective below are adapted from EVCouplings, Copyright (c) 2017 EVcouplings
development team, MIT licence; the full notice is packaged as
``resources/proteingym/EVcouplings-LICENSE.upstream`` and receipted in
``resources/proteingym/sources.json``.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np

from rewirebench.protocols.proteingym import (
    REFERENCE_SHA256,
    UPSTREAM_REVISION,
    _sha,
    _substitutions,
    resource_path,
)

BASELINE_ID = "proteingym-evcouplings-independent-v1"
MANIFEST_SCHEMA = "rewire-evcouplings-independent-manifest-v1"
ARTIFACT_SCHEMA = "rewire-evcouplings-independent-fields-v1"
EVCOUPLINGS_REVISION = "e1362407a0b65d63ca07df55f44cb17b0a3722b7"
UPSTREAM_SOURCES = {
    "evcouplings/couplings/model.py": {
        "revision": EVCOUPLINGS_REVISION,
        "sha256": "6422cfc7892cb5d076194354b05fe91fe40a49cee28548e62c4e211f32cfa050",
    },
    "proteingym/baselines/EVmutation/score_mutants.py": {
        "revision": UPSTREAM_REVISION,
        "sha256": "9802fcd2e50ac0396a445b3c0bf9e4cfb865f1e57ae0c6382a12214ee0336050",
    },
    "proteingym/baselines/EVmutation/calculations.py": {
        "revision": UPSTREAM_REVISION,
        "sha256": "07e2059186bdf2f673c46817c0a2c630658269008474b3c05f10585b8d246c6e",
    },
    "config.json": {
        "revision": UPSTREAM_REVISION,
        "sha256": "7cd239d1e4c8b474f6ce2bd6574423b53ab48204f852f219e09a36f25d513608",
    },
}
# Declared by the manifest author; recorded, not verified by this package.
PROVENANCE_FIELDS = (
    "model_source", "model_retrieval_receipt", "alignment_release", "alignment_sha256",
    "query_interval", "filtering_policy", "sequence_weights", "model_generation_config",
)
TRAINING_OVERLAP = (
    "No DMS labels. External evolutionary information: a plmc_v2 model inferred from a "
    "multiple sequence alignment of natural homologues; independent fields refitted from its "
    "stored frequencies, N_eff and lambda_h. Alignment homologues may include benchmark "
    "proteins; overlap is not quantified."
)
METHOD = "EVCouplings CouplingsModel.to_independent_model(); ProteinGym prediction_independent"
SCORE_DIRECTION = "higher_is_fitter"
PROVENANCE_STATUS = "declared_by_manifest_not_verified_by_rewirebench"
# Stationarity residual max|gradient| / N_eff, in site-frequency units, admitted for
# SciPy's precision-loss status. Synthetic surveys at N_eff up to 5e5 gave <= 3e-9.
STATIONARITY_TOLERANCE = 1e-6
BFGS_GTOL = 1e-5  # SciPy fmin_bfgs default, used unchanged by upstream
SITE_KEYS = frozenset({"warnflag", "status", "objective", "gradient_max_abs", "stationarity_residual"})
OPTIMIZER = {
    "routine": "scipy.optimize.fmin_bfgs", "initial_fields": "zeros",
    "acceptance": "warnflag 0 with finite objective and gradient and max|gradient| <= 1e-5 "
                  "(SciPy default gtol); warnflag 2 only if finite and max|gradient|/N_eff <= "
                  "stationarity_tolerance; any other state fails",
    "stationarity_tolerance": STATIONARITY_TOLERANCE,
}
ARTIFACT_KEYS = frozenset({"schema", "baseline_id", "method", "score_direction", "reference_sha256",
                           "upstream_sources", "environment", "manifest_sha256", "dms_labels_used",
                           "assays"})
ASSAY_KEYS = frozenset({"wild_type_sequence", "msa_start", "model_position_offset", "model_id",
                        "model_sha256", "model_format", "precision", "model_metadata",
                        "model_provenance", "provenance_status", "alphabet", "fields", "optimizer"})
METADATA_KEYS = frozenset({"L", "num_symbols", "N_valid", "N_invalid", "num_iter", "theta",
                           "lambda_h", "lambda_J", "lambda_group", "N_eff", "alphabet",
                           "target_seq", "index_list"})
UNSCORED_POSITION = "position_outside_evcouplings_model_index_list"
UNSCORED_RESIDUE = "residue_outside_evcouplings_model_alphabet"
UNSCORED_ASSAY = "no_evcouplings_model_for_assay"


def _hex(value, length=64):
    return isinstance(value, str) and len(value) == length and all(
        c in "0123456789abcdef" for c in value)


def read_plmc_v2(path: str | Path, precision: str = "float32") -> dict:
    """Read a plmc_v2 binary model with the EVCouplings layout, strictly.

    The byte length must match the declared header exactly, so plmc_v1 files or a
    wrong precision fail instead of being reinterpreted. Mean-field models
    (lambda_h < 0) and nonfinite parameters are rejected.
    """
    if precision not in {"float32", "float64"}:
        raise ValueError("plmc_v2 precision must be float32 or float64")
    data = Path(path).read_bytes()
    size = np.dtype(precision).itemsize
    if len(data) < 20 + 5 * size:
        raise ValueError("File is too short to be a plmc_v2 model")
    L, q, n_valid, n_invalid, n_iter = (int(v) for v in np.frombuffer(data, "<i4", 5))
    if L < 1 or q < 2 or n_valid < 0 or n_invalid < 0 or n_iter < 0:
        raise ValueError("Invalid plmc_v2 header dimensions")
    pairs = L * (L - 1) // 2
    expected = (20 + 5 * size + q + (n_valid + n_invalid) * size + L + 4 * L
                + 2 * L * q * size + 2 * pairs * q * q * size)
    if len(data) != expected:
        raise ValueError("plmc_v2 size mismatch; wrong format, precision or truncated file")
    offset = 20
    def take(dtype, count):
        nonlocal offset
        values = np.frombuffer(data, dtype, count, offset)
        offset += values.nbytes
        return values
    theta, lambda_h, lambda_J, lambda_group, n_eff = take(precision, 5)
    try:
        alphabet = take("S1", q).tobytes().decode("ascii")
        weights = take(precision, n_valid + n_invalid)
        target = take("S1", L).tobytes().decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("plmc_v2 alphabet and target must be ASCII") from exc
    index_list = take("<i4", L).astype(int)
    f_i = take(precision, L * q).reshape(L, q)
    h_i = take(precision, L * q).reshape(L, q)
    f_ij = take(precision, pairs * q * q)
    J_ij = take(precision, pairs * q * q)
    if len(set(alphabet)) != q:
        raise ValueError("plmc_v2 alphabet symbols must be unique")
    if any(symbol not in alphabet or symbol in "-." for symbol in target):
        raise ValueError("Model target sequence must use non-gap model alphabet symbols")
    if len(set(index_list.tolist())) != L or np.any(np.diff(index_list) <= 0):
        raise ValueError("plmc_v2 index list must be strictly increasing")
    for name, values in (("header", np.array([theta, lambda_h, lambda_J, lambda_group, n_eff])),
                         ("weights", weights), ("f_i", f_i), ("h_i", h_i),
                         ("f_ij", f_ij), ("J_ij", J_ij)):
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite plmc_v2 {name} parameter")
    if lambda_h < 0:
        raise ValueError("Mean-field model (lambda_h < 0) is not a plmc pseudolikelihood model")
    if not lambda_h > 0 or not n_eff > 0:
        raise ValueError("Independent refit requires positive lambda_h and N_eff")
    if np.any(f_i < 0) or np.any(np.abs(f_i.sum(axis=1) - 1) > 1e-3):
        raise ValueError("Single-site frequencies must be nonnegative and sum to one")
    return {
        "L": L, "num_symbols": q, "N_valid": n_valid, "N_invalid": n_invalid,
        "num_iter": n_iter, "theta": theta, "lambda_h": lambda_h, "lambda_J": lambda_J,
        "lambda_group": lambda_group, "N_eff": n_eff, "alphabet": alphabet,
        "target_seq": target, "index_list": index_list.tolist(), "f_i": f_i,
        "precision": precision, "format": "plmc_v2",
    }


def fit_independent_fields(f_i, lambda_h, n_eff) -> tuple[np.ndarray, list[dict]]:
    """EVCouplings ``to_independent_model`` field estimate, site by site.

    Arguments keep the model's stored precision, as upstream does. The objective,
    gradient, zero start and ``fmin_bfgs`` call are unchanged from upstream, which
    ignores the optimizer's terminal status. Here each site's status is checked:

    - warnflag 0 (SciPy gradient tolerance met) is accepted if the terminal
      objective and gradient, recomputed at the returned fields, are finite and
      the recomputed max|gradient| meets SciPy's own default gtol (1e-5);
    - warnflag 2 (precision loss, routine at large N_eff) is accepted only if they
      are finite and the stationarity residual max|gradient| / N_eff is at most
      ``STATIONARITY_TOLERANCE`` (units of site frequency);
    - any other status (iteration limit, NaN) or nonfinite value fails.

    Returns the fields and a status record per site.
    """
    from scipy.optimize import fmin_bfgs

    def _log_post(x, *args):
        fi, lam, N = args
        logZ = np.log(np.exp(x).sum())
        return N * (logZ - (fi * x).sum()) + lam * ((x ** 2).sum())

    def _gradient(x, *args):
        fi, lam, N = args
        Z = np.exp(x).sum()
        P = np.exp(x) / Z
        return N * (P - fi) + lam * 2 * x

    L, q = f_i.shape
    h_i = np.zeros((L, q))
    sites = []
    for i in range(L):
        args = (f_i[i], lambda_h, n_eff)
        with np.errstate(all="ignore"):
            result = fmin_bfgs(_log_post, np.zeros(q), _gradient, args=args,
                               disp=False, full_output=True)
            x = np.asarray(result[0], dtype=float)
            objective = float(_log_post(x, *args)) if np.isfinite(x).all() else math.nan
            gradient = _gradient(x, *args) if np.isfinite(x).all() else np.full(q, math.nan)
        flag = int(result[6])
        values = [objective, float(result[1]), *gradient, *np.asarray(result[2], dtype=float)]
        if x.shape != (q,) or not np.isfinite(x).all() or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Independent field optimisation at model site {i} ended with nonfinite values")
        gradient_max = float(np.abs(gradient).max())
        residual = gradient_max / float(n_eff)
        if flag == 0 and gradient_max <= BFGS_GTOL:
            status = "converged"
        elif flag == 2 and residual <= STATIONARITY_TOLERANCE:
            status = "precision_loss_within_stationarity_tolerance"
        else:
            raise ValueError(f"Independent field optimisation at model site {i} failed "
                             f"(warnflag {flag}, stationarity residual {residual:.3g})")
        h_i[i] = x
        sites.append({"warnflag": flag, "status": status, "objective": objective,
                      "gradient_max_abs": gradient_max, "stationarity_residual": residual})
    return h_i, sites


def proteingym_model_id(dms_id: str, uniprot_id: str) -> str:
    """Model job prefix chosen by the pinned score_mutants.py (lines 37-45)."""
    model_id = "RASK_HUMAN_Ursu_2020" if dms_id == "RASK_HUMAN_Ursu_2020" else uniprot_id
    return "F7YBW8_MESOW" if model_id == "F7YBW7_MESOW" else model_id


def _reference():
    path = resource_path("DMS_substitutions.csv")
    if _sha(path) != REFERENCE_SHA256:
        raise ValueError("Packaged ProteinGym reference does not match its pin")
    with path.open(newline="") as stream:
        return {row["DMS_id"]: row for row in csv.DictReader(stream)}


def build_assay_fields(model: dict, *, wild_type_sequence: str, msa_start: int) -> dict:
    """Map model positions to assay positions and refit the independent fields.

    ProteinGym passes offset -(MSA_start - 1), so model index m is assay position
    m + MSA_start - 1. Every model position must fall inside the assay sequence
    and match its wild type; otherwise the coordinate map is wrong and preparation
    fails. Positions absent from the index list remain unscorable.
    """
    if type(msa_start) is not int or msa_start < 1:
        raise ValueError("msa_start must be a positive integer")
    positions = [m + msa_start - 1 for m in model["index_list"]]
    if positions[0] < 1 or positions[-1] > len(wild_type_sequence):
        raise ValueError("Model index list maps outside the assay wild-type sequence")
    mismatches = [p for p, residue in zip(positions, model["target_seq"])
                  if wild_type_sequence[p - 1] != residue]
    if mismatches:
        raise ValueError(f"Model target disagrees with assay wild type at {len(mismatches)} positions")
    fields, sites = fit_independent_fields(model["f_i"], model["lambda_h"], model["N_eff"])
    return {
        "fields": {str(p): [float(v) for v in row] for p, row in zip(positions, fields)},
        "optimizer": {**OPTIMIZER, "sites": {str(p): site for p, site in zip(positions, sites)}},
    }


def _environment():
    versions = {}
    for package in ("numpy", "scipy", "rewirebench"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unreported"
    return {"python": platform.python_version(), "platform": platform.platform(), **versions}


def prepare_artifact(manifest: str | Path, output: str | Path) -> dict:
    """Validate a reviewed manifest and write the derived independent-field artifact.

    The manifest lists, per ProteinGym assay, a local plmc_v2 model path, its
    SHA256, precision, model ID, MSA_start and declared preparation provenance.
    Relative model paths resolve against the manifest directory. The output must
    be a new file. Returns a summary including the artifact SHA256.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text())
    if spec.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest schema must be {MANIFEST_SCHEMA}")
    if spec.get("dms_labels_used") is not False:
        raise ValueError("Manifest must declare dms_labels_used: false")
    entries = spec.get("assays")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Manifest requires a nonempty assays mapping")
    reference = _reference()
    assays = {}
    for assay_id, entry in sorted(entries.items()):
        if assay_id not in reference:
            raise ValueError(f"Unknown ProteinGym assay: {assay_id}")
        ref = reference[assay_id]
        expected_id = proteingym_model_id(assay_id, ref["UniProt_ID"])
        if entry.get("model_id") != expected_id:
            raise ValueError(f"{assay_id}: model_id must be {expected_id} (pinned scorer rule)")
        if entry.get("format") != "plmc_v2":
            raise ValueError(f"{assay_id}: only plmc_v2 models are accepted")
        msa_start = entry.get("msa_start")
        if msa_start != int(ref["MSA_start"]):
            raise ValueError(f"{assay_id}: msa_start must equal the pinned reference MSA_start")
        provenance = entry.get("provenance", {})
        _check_provenance(provenance, assay_id)
        path = Path(entry.get("model_path", ""))
        path = path if path.is_absolute() else manifest.parent / path
        digest = _sha(path)
        if not _hex(entry.get("model_sha256")) or digest != entry["model_sha256"]:
            raise ValueError(f"{assay_id}: model SHA256 mismatch")
        model = read_plmc_v2(path, entry.get("precision", ""))
        derived = build_assay_fields(model, wild_type_sequence=ref["target_seq"],
                                     msa_start=msa_start)
        assays[assay_id] = {
            "wild_type_sequence": ref["target_seq"], "msa_start": msa_start,
            "model_position_offset": 1 - msa_start, "model_id": expected_id,
            "model_sha256": digest, "model_format": "plmc_v2", "precision": model["precision"],
            "model_metadata": {k: (float(model[k]) if isinstance(model[k], np.floating) else model[k])
                               for k in ("L", "num_symbols", "N_valid", "N_invalid", "num_iter",
                                         "theta", "lambda_h", "lambda_J", "lambda_group", "N_eff",
                                         "alphabet", "target_seq", "index_list")},
            "model_provenance": {k: provenance[k] for k in PROVENANCE_FIELDS},
            "provenance_status": PROVENANCE_STATUS,
            "alphabet": model["alphabet"], **derived,
        }
    artifact = {
        "schema": ARTIFACT_SCHEMA, "baseline_id": BASELINE_ID, "method": METHOD,
        "score_direction": SCORE_DIRECTION, "reference_sha256": REFERENCE_SHA256,
        "upstream_sources": UPSTREAM_SOURCES, "environment": _environment(),
        "manifest_sha256": _sha(manifest), "dms_labels_used": False, "assays": assays,
    }
    validate_artifact(artifact)
    text = json.dumps(artifact, sort_keys=True, indent=1, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return {"baseline_id": BASELINE_ID, "artifact": str(output), "assays": sorted(assays),
            "artifact_sha256": hashlib.sha256(text.encode()).hexdigest()}


def _check_provenance(provenance, assay_id):
    if not isinstance(provenance, dict) or set(provenance) - set(PROVENANCE_FIELDS):
        raise ValueError(f"{assay_id}: model provenance must contain only the declared fields")
    missing = [k for k in PROVENANCE_FIELDS
               if not isinstance(provenance.get(k), str) or not provenance[k].strip()]
    if missing:
        raise ValueError(f"{assay_id}: missing model provenance {', '.join(missing)}")
    if not _hex(provenance["alignment_sha256"]):
        raise ValueError(f"{assay_id}: alignment_sha256 must be a SHA256 digest")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_optimizer(optimizer, positions, assay_id, n_eff):
    if not isinstance(optimizer, dict) or {k: optimizer.get(k) for k in OPTIMIZER} != OPTIMIZER:
        raise ValueError(f"{assay_id}: optimizer record does not match the pinned routine and acceptance rule")
    sites = optimizer.get("sites")
    if set(optimizer) != {*OPTIMIZER, "sites"} or not isinstance(sites, dict) or set(sites) != positions:
        raise ValueError(f"{assay_id}: optimizer status must cover exactly the field positions")
    for position, site in sites.items():
        if (not isinstance(site, dict) or set(site) != SITE_KEYS
                or not all(_finite_number(site[k]) for k in SITE_KEYS - {"warnflag", "status"})):
            raise ValueError(f"{assay_id}: malformed optimizer status at position {position}")
        # Recorded norms must be nonnegative and mutually consistent with the recorded N_eff.
        if site["gradient_max_abs"] < 0 or site["stationarity_residual"] < 0:
            raise ValueError(f"{assay_id}: optimizer norms must be nonnegative at position {position}")
        if not math.isclose(site["stationarity_residual"], site["gradient_max_abs"] / n_eff,
                            rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"{assay_id}: optimizer residual contradicts gradient_max_abs / N_eff "
                             f"at position {position}")
        accepted = (site["warnflag"] == 0 and site["status"] == "converged"
                    and site["gradient_max_abs"] <= BFGS_GTOL) or (
            site["warnflag"] == 2 and site["status"] == "precision_loss_within_stationarity_tolerance"
            and site["stationarity_residual"] <= STATIONARITY_TOLERANCE)
        if type(site["warnflag"]) is not int or not accepted:
            raise ValueError(f"{assay_id}: unaccepted optimizer terminal state at position {position}")


def validate_artifact(data: dict) -> None:
    """Check the semantic contract of a prepared artifact, not only its bytes.

    Everything that preparation guarantees is rechecked: the no-label
    declaration and fixed direction; pinned reference, source pins and method;
    each assay's pinned wild type, MSA_start, offset and model identity; model
    metadata consistency (alphabet, target, index list, N_eff, lambda_h); field
    coordinates that are exactly the canonical mapped model positions; finite field
    rows; accepted optimizer states; and the declared preparation provenance.
    Declared MSA provenance is still a declaration, not authentication.
    """
    if not isinstance(data, dict) or set(data) != ARTIFACT_KEYS:
        raise ValueError("Artifact must contain exactly the prepared-artifact fields")
    if data["schema"] != ARTIFACT_SCHEMA or data["baseline_id"] != BASELINE_ID or data["method"] != METHOD:
        raise ValueError("Not an EVCouplings independent-field artifact for this method")
    if data["dms_labels_used"] is not False:
        raise ValueError("Artifact must declare dms_labels_used: false")
    if data["score_direction"] != SCORE_DIRECTION:
        raise ValueError(f"Artifact score direction must be {SCORE_DIRECTION}")
    if data["reference_sha256"] != REFERENCE_SHA256 or data["upstream_sources"] != UPSTREAM_SOURCES:
        raise ValueError("Artifact was prepared against different reference or source pins")
    environment = data["environment"]
    if not isinstance(environment, dict) or not all(
            isinstance(environment.get(k), str) and environment[k] for k in ("python", "numpy", "scipy")):
        raise ValueError("Artifact must record its Python, NumPy and SciPy versions")
    if not _hex(data["manifest_sha256"]):
        raise ValueError("Artifact must record the manifest SHA256")
    assays = data["assays"]
    if not isinstance(assays, dict) or not assays:
        raise ValueError("Artifact requires a nonempty assays mapping")
    reference = _reference()
    for assay_id, entry in assays.items():
        if assay_id not in reference:
            raise ValueError(f"Unknown ProteinGym assay: {assay_id}")
        if not isinstance(entry, dict) or set(entry) != ASSAY_KEYS:
            raise ValueError(f"{assay_id}: artifact entry must contain exactly the prepared fields")
        ref = reference[assay_id]
        wild_type, msa_start = ref["target_seq"], int(ref["MSA_start"])
        if entry["wild_type_sequence"] != wild_type:
            raise ValueError(f"{assay_id}: wild type differs from the pinned reference")
        if type(entry["msa_start"]) is not int or entry["msa_start"] != msa_start:
            raise ValueError(f"{assay_id}: msa_start differs from the pinned reference")
        if entry["model_position_offset"] != 1 - msa_start or type(entry["model_position_offset"]) is not int:
            raise ValueError(f"{assay_id}: model_position_offset must be 1 - MSA_start")
        if entry["model_id"] != proteingym_model_id(assay_id, ref["UniProt_ID"]):
            raise ValueError(f"{assay_id}: model_id differs from the pinned scorer rule")
        if not _hex(entry["model_sha256"]) or entry["model_format"] != "plmc_v2" \
                or entry["precision"] not in {"float32", "float64"}:
            raise ValueError(f"{assay_id}: invalid model hash, format or precision")
        if entry["provenance_status"] != PROVENANCE_STATUS:
            raise ValueError(f"{assay_id}: provenance status must be {PROVENANCE_STATUS}")
        _check_provenance(entry["model_provenance"], assay_id)
        meta = entry["model_metadata"]
        if not isinstance(meta, dict) or set(meta) != METADATA_KEYS:
            raise ValueError(f"{assay_id}: model metadata must contain exactly the recorded fields")
        alphabet, target, index_list = meta["alphabet"], meta["target_seq"], meta["index_list"]
        if (entry["alphabet"] != alphabet or not isinstance(alphabet, str)
                or len(set(alphabet)) != len(alphabet) or meta["num_symbols"] != len(alphabet)):
            raise ValueError(f"{assay_id}: alphabet is duplicated or disagrees with model metadata")
        if (not isinstance(target, str) or not isinstance(index_list, list) or meta["L"] != len(target)
                or len(index_list) != len(target) or not target
                or any(type(m) is not int for m in index_list)
                or any(b <= a for a, b in itertools.pairwise(index_list))):
            raise ValueError(f"{assay_id}: target sequence and index list are inconsistent")
        if any(c not in alphabet or c in "-." for c in target):
            raise ValueError(f"{assay_id}: model target uses symbols outside the non-gap alphabet")
        # The same header guarantees read_plmc_v2 enforces on the binary model.
        counts = {k: meta[k] for k in ("L", "num_symbols", "N_valid", "N_invalid", "num_iter")}
        minimum = {"L": 1, "num_symbols": 2, "N_valid": 0, "N_invalid": 0, "num_iter": 0}
        if any(type(v) is not int or v < minimum[k] for k, v in counts.items()):
            raise ValueError(f"{assay_id}: model header counts must be integers within plmc_v2 limits")
        if not all(_finite_number(meta[k]) for k in ("theta", "lambda_J", "lambda_group")):
            raise ValueError(f"{assay_id}: theta, lambda_J and lambda_group must be finite")
        if not (_finite_number(meta["N_eff"]) and meta["N_eff"] > 0
                and _finite_number(meta["lambda_h"]) and meta["lambda_h"] > 0):
            raise ValueError(f"{assay_id}: N_eff and lambda_h must be positive and finite")
        positions = [m + msa_start - 1 for m in index_list]
        if positions[0] < 1 or positions[-1] > len(wild_type):
            raise ValueError(f"{assay_id}: model index list maps outside the assay wild type")
        if any(wild_type[p - 1] != residue for p, residue in zip(positions, target)):
            raise ValueError(f"{assay_id}: model target disagrees with the assay wild type")
        fields = entry["fields"]
        expected = {str(p) for p in positions}
        if not isinstance(fields, dict) or set(fields) != expected:
            raise ValueError(f"{assay_id}: field coordinates are not the canonical mapped model positions")
        for position, row in fields.items():
            if (not isinstance(row, list) or len(row) != len(alphabet)
                    or not all(_finite_number(v) for v in row)):
                raise ValueError(f"{assay_id}: invalid field row at position {position}")
        _check_optimizer(entry["optimizer"], expected, assay_id, meta["N_eff"])


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in artifact: {key!r}")
        result[key] = value
    return result


class EVCouplingsIndependent:
    """Score ProteinGym substitutions from a prepared, hash-pinned field artifact.

    There is no ``fit`` method: the SDK's zero-shot protocol never supplies labels.
    """

    INPUT_FIELDS = frozenset({"id", "assay_id", "wild_type_sequence", "mutant", "mutated_sequence"})

    def __init__(self, artifact: str | Path, artifact_sha256: str):
        raw = Path(artifact).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not _hex(artifact_sha256) or digest != artifact_sha256:
            raise ValueError("EVCouplings independent artifact does not match artifact_sha256")
        data = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
        validate_artifact(data)
        self._assays, optimizer = {}, {}
        for assay_id, entry in data["assays"].items():
            alphabet = entry["alphabet"]
            fields = {int(p): dict(zip(alphabet, row)) for p, row in entry["fields"].items()}
            self._assays[assay_id] = (entry["wild_type_sequence"], fields)
            sites = entry["optimizer"]["sites"]
            optimizer[assay_id] = {
                "converged_sites": sum(s["status"] == "converged" for s in sites.values()),
                "precision_loss_positions": sorted(
                    (int(p) for p, s in sites.items() if s["warnflag"] == 2)),
                "max_stationarity_residual": max(s["stationarity_residual"] for s in sites.values()),
            }
        self.provenance = {
            "baseline_id": BASELINE_ID, "configuration_sha256": digest,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evcouplings_revision": EVCOUPLINGS_REVISION, "proteingym_revision": UPSTREAM_REVISION,
            "assays": sorted(self._assays), "training_overlap": TRAINING_OVERLAP,
            "optimizer": {"stationarity_tolerance": STATIONARITY_TOLERANCE, "assays": optimizer},
            "scipy_version_at_preparation": data["environment"]["scipy"],
            "published_result_reproduction": False,
        }

    def predict(self, inputs):
        results = {}
        for row in inputs:
            if set(row) - self.INPUT_FIELDS or not {"id", "assay_id", "wild_type_sequence", "mutant"} <= set(row):
                raise ValueError("Unexpected or missing input fields; labels must not reach adapters")
            ident = row["id"]
            if ident in results:
                raise ValueError("Duplicate input ID")
            if row["assay_id"] not in self._assays:
                results[ident] = {"score": None, "reason": UNSCORED_ASSAY}
                continue
            wild_type, fields = self._assays[row["assay_id"]]
            if row["wild_type_sequence"] != wild_type:
                raise ValueError("Input wild type differs from the artifact's assay wild type")
            mutations = _substitutions(row["mutant"], wild_type)
            if "mutated_sequence" in row:
                mutated = list(wild_type)
                for index, _, alt in mutations:
                    mutated[index] = alt
                if row["mutated_sequence"] != "".join(mutated):
                    raise ValueError("Mutated sequence disagrees with mutant notation")
            score, reason = 0.0, None
            for index, wt, alt in mutations:
                site = fields.get(index + 1)
                if site is None:
                    reason = UNSCORED_POSITION
                    break
                if wt not in site or alt not in site:
                    reason = UNSCORED_RESIDUE
                    break
                score += site[alt] - site[wt]
            results[ident] = {"score": None, "reason": reason} if reason else score
        return results
