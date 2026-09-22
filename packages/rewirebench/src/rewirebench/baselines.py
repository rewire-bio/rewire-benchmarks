"""Versioned, inspectable local baselines for the protocols supported by the SDK.

Registry coverage is not execution evidence. Unimplemented meaningful zero-shot
references stay blocked and never become scores. Running never exports or submits.
"""
from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path

from rewirebench import sdk

REGISTRY_VERSION = "1"
_CODE_SOURCE = "https://github.com/rewire-bio/rewire-benchmarks"


def _entry(ident, role, title, configuration=None, *, prediction_type="scalar", gap=None):
    return {
        "baseline_id": ident, "version": "1", "role": role, "title": title,
        "status": "blocked" if gap else "implemented",
        "execution_status": "not_established_by_registry",
        "prediction_type": prediction_type,
        "configuration": configuration or {}, "gap": gap,
        "implementation_source": _CODE_SOURCE,
        "scientific_claim": "Rewire reference; not a reproduced published baseline",
    }


_PRIOR = _entry("training-prior-v1", "null", "Training class prior / majority",
                {"binary": "positive-class prevalence", "multiclass": "majority; lowest-class tie"})
_MEAN = _entry("training-mean-v1", "null", "Training target mean")
_RANDOM = _entry("seeded-random-v1", "null", "Seeded random ranking control",
                 {"seed": 0, "method": "SHA256(seed:prepared_id), first 64 bits / 2**64",
                  "limitation": "One seeded control is not an uncertainty estimate; IDs bind the run"})
_NGRAM = {"ngram_range": [1, 3], "normalisation": "l2", "vocabulary_fit": "train only",
          "classification": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "seed": 0},
          "regression": {"alpha": 1.0, "solver": "lsqr", "tol": 1e-8, "max_iter": 10000}}

REGISTRY = {
    "mfass-v2": [
        _PRIOR,
        _entry("mfass-kmer-position-v2", "conventional", "Corrected MFASS k-mer/position baseline",
               {"window": 21, "k": 3, "orientation": "validated assay-oriented mutant",
                "max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31,
                "l2_regularization": 1.0, "seed": 20260914,
                "additional_inputs": ["position", "region", "phyloP", "phastCons"],
                "reference_revision": "bee9133b83f3aedaf2bbb9013f1875515845607e"}),
    ],
    "mfass-v2-frozen-encoder": [
        _entry("constant-paired-probe-v1", "null", "Constant paired features + fixed balanced probe",
               {"features": "one zero per sequence", "head": "MFASS fixed balanced logistic probe",
                "limitation": "Balanced-head control; not training prevalence"}, prediction_type="embedding"),
        _entry("paired-composition-probe-v1", "conventional", "Sequence composition + fixed MFASS probe",
               {"alphabet": "ACGT", "features": "log1p length, base fractions, unknown fraction",
                "head": "protocol-owned fixed logistic probe"}, prediction_type="embedding"),
    ],
    "flip2-fitness-v1": [
        _MEAN,
        _entry("protein-composition-probe-v1", "conventional", "Protein composition + fixed ridge probe",
               {"alphabet": "ACDEFGHIKLMNPQRSTVWY", "head": "protocol-owned alpha-10 ridge",
                "features": "log1p length, residue fractions, unknown fraction"},
               prediction_type="embedding"),
    ],
    "mrnabench-sample-mrl-v1": [
        _MEAN,
        _entry("rna-composition-probe-v1", "conventional", "RNA composition + train-only RidgeCV",
               {"alphabet": "ACGTU", "features": "log1p length, base fractions, unknown fraction",
                "head": "protocol-owned RidgeCV"}, prediction_type="embedding"),
    ],
    "genomic-benchmarks-v2": [
        _PRIOR,
        _entry("dna-ngram-logistic-v1", "conventional", "DNA character n-grams + logistic regression",
               {**_NGRAM, "case": "uppercase"}),
    ],
    "tdc-admet-group-v1": [
        _entry("training-control-v1", "null", "Training mean (regression) / prevalence (classification)"),
        _entry("smiles-ngram-linear-v1", "conventional", "SMILES character n-grams + regularised linear head",
               {**_NGRAM, "case": "preserved", "limitation": "String reference, not molecular descriptors; not SMILES-invariant"}),
    ],
    "proteingym-v1.3-dms-substitutions": [
        _RANDOM,
        _entry("proteingym-conventional-pending-v1", "conventional", "Label-free conventional reference pending",
               gap="Requires a reviewed alignment/frequency method, permitted MSA inputs and pinned implementation. No fitting to DMS labels."),
    ],
    "dart-eval-task1-zero-shot-v1": [
        _RANDOM,
        _entry("dart-conventional-pending-v1", "conventional", "Label-free regulatory reference pending",
               gap="Requires a source-reviewed label-free method with declared reference data. Training on element/control labels is forbidden."),
    ],
}


def describe_baselines(protocol: str) -> dict:
    """List implemented and blocked baseline roles without executing or downloading."""
    if protocol not in REGISTRY:
        raise ValueError(f"Unsupported baseline protocol: {protocol}")
    return {"registry_version": REGISTRY_VERSION, "protocol_id": protocol,
            "baselines": copy.deepcopy(REGISTRY[protocol])}


def _adapter(entry, data):
    from rewirebench.adapters.baselines import (
        PairedComposition,
        StringNgramReference,
        TrainingPrior,
    )
    from rewirebench.adapters.mfass import KmerBaseline
    from rewirebench.adapters.sequence import SeededRandomScore, SequenceComposition, TrainMean

    ident = entry["baseline_id"]
    if ident == "training-prior-v1":
        return TrainingPrior(multiclass=len(data.get("metadata", {}).get("classes", [])) > 2)
    if ident == "training-mean-v1":
        return TrainMean()
    if ident == "seeded-random-v1":
        return SeededRandomScore(seed=0)
    if ident == "mfass-kmer-position-v2":
        return KmerBaseline()
    if ident in {"constant-paired-probe-v1", "paired-composition-probe-v1"}:
        return PairedComposition(constant=ident == "constant-paired-probe-v1")
    if ident in {"protein-composition-probe-v1", "rna-composition-probe-v1"}:
        return SequenceComposition(entry["configuration"]["alphabet"])
    if ident == "dna-ngram-logistic-v1":
        return StringNgramReference(multiclass=len(data["metadata"]["classes"]) > 2)
    if ident in {"training-control-v1", "smiles-ngram-linear-v1"}:
        task = data.get("metadata", {}).get("task")
        if task not in {"classification", "regression"}:
            raise ValueError("TDC prepared data must declare classification or regression task")
        classification = task == "classification"
        if ident == "training-control-v1":
            return TrainingPrior() if classification else TrainMean()
        return StringNgramReference(field="smiles", classification=classification)
    raise ValueError("Baseline has no executable adapter")


def run_baselines(prepared: dict | str | Path, *, output: str | Path,
                  baseline_ids: list[str] | None = None, batch_size: int = 32,
                  fail_fast: bool = False) -> dict:
    """Execute fixed reference methods through SDK validation, never overwrite.

    All registry entries are included by default, including explicit blocked
    entries. The returned manifest distinguishes batch success from full-cohort
    completion; smoke/subset reports retain their SDK labels. Per-baseline errors
    are recorded by exception class only, avoiding private values or paths in a
    coordinator manifest. Keyboard interrupts terminate rather than being hidden.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    data = sdk._load(prepared, "prepared.json")
    sdk._validate_prepared(data)
    entries = describe_baselines(data["protocol_id"])["baselines"]
    if baseline_ids is not None:
        if not isinstance(baseline_ids, list) or not baseline_ids or any(not isinstance(i, str) for i in baseline_ids):
            raise ValueError("baseline_ids must be a nonempty list of IDs")
        if len(set(baseline_ids)) != len(baseline_ids):
            raise ValueError("Duplicate baseline IDs")
        if set(baseline_ids) - {e["baseline_id"] for e in entries}:
            raise ValueError("Unknown or incompatible baseline ID")
        indexed = {e["baseline_id"]: e for e in entries}
        entries = [indexed[i] for i in baseline_ids]
    path = sdk._new_output(output)
    manifest = {
        "schema_version": "1.0", "kind": "rewire_baseline_batch", "registry_version": REGISTRY_VERSION,
        "protocol_id": data["protocol_id"], "dataset_id": data["dataset_id"],
        "prepared_sha256": data["prepared_sha256"], "scope": data["scope"],
        "sdk_code_sha256": sdk._code_digest(), "created_at": datetime.now(UTC).isoformat(),
        "publication": "local_only_not_submitted", "baselines": [],
    }
    # Persist intentions before any model executes, so interruption is inspectable.
    sdk._write(path / "baseline-plan.json", {**manifest, "selected": entries})
    for entry in entries:
        ident = entry["baseline_id"]
        record = {"baseline_id": ident, "role": entry["role"], "configuration": entry["configuration"]}
        if entry["status"] == "blocked":
            record.update(status="blocked", reason=entry["gap"])
        else:
            try:
                model = {
                    "name": entry["title"],
                    "training_overlap": "no labelled fitting" if ident == "seeded-random-v1" else "protocol-permitted training only; no pretraining",
                    "configuration": {"baseline_id": ident, "baseline_version": entry["version"], **entry["configuration"]},
                    "input_information": "protocol allowlisted biological inputs; see baseline configuration for extra inputs",
                }
                report = sdk.run(data, _adapter(entry, data), output=path / ident, model=model,
                                 batch_size=batch_size, allow_partial=False, prediction_type=entry["prediction_type"])
                record.update(status="evaluated", report=f"{ident}/report.json",
                              completion=report["completion"], coverage=report["coverage"],
                              scientific_reproduction="not_established")
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                sdk._write(path / f"{ident}.status.json", record)
                manifest["baselines"].append(record)
                if fail_fast:
                    manifest["status"] = "interrupted_by_failure"
                    sdk._write(path / "baseline-manifest.json", manifest)
                    raise
                continue
        sdk._write(path / f"{ident}.status.json", record)
        manifest["baselines"].append(record)
    statuses = [r["status"] for r in manifest["baselines"]]
    manifest["status"] = "evaluated" if all(s == "evaluated" for s in statuses) else "incomplete"
    sdk._write(path / "baseline-manifest.json", manifest)
    return manifest
