"""Local, protocol-owned preparation, inference, scoring and explicit export.

No model code is downloaded or imported until the caller supplies an adapter.
Prepared data includes public assay labels for scoring; adapters receive only the
protocol's allowlisted inputs. This is leakage prevention, not a security sandbox.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import numbers
import os
import platform
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

PROTOCOLS = {
    "mfass-v2": "mfass",
    "mfass-v2-frozen-encoder": "mfass",
    "proteingym-v1.3-dms-substitutions": "proteingym",
    "tdc-admet-group-v1": "tdc_admet",
    "genomic-benchmarks-v1": "genomic_benchmarks",
}


class ScoreAdapter(Protocol):
    """Inputs include stable IDs and protocol-approved biological fields only."""

    def predict(self, inputs: list[dict[str, Any]]) -> Mapping[str, Any]: ...


class EmbeddingAdapter(Protocol):
    def embed(self, inputs: list[dict[str, Any]]) -> Mapping[str, Any]: ...


def _plugin(protocol: str):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported protocol {protocol!r}; choose {', '.join(PROTOCOLS)}")
    return importlib.import_module(f"rewirebench.protocols.{PROTOCOLS[protocol]}")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _load(value: dict | str | Path, filename: str) -> dict:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    path = Path(value)
    return json.loads((path / filename if path.is_dir() else path).read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, including same-run reruns: never overwrite evidence.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(value))


def _new_output(output: str | Path) -> Path:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _validate_prepared(data: dict) -> None:
    _plugin(data["protocol_id"])
    if data.get("scope") not in {"full", "subset", "smoke"}:
        raise ValueError("Prepared data must declare full, subset or smoke scope")
    rows = data.get("rows", [])
    ids = [r.get("id") for r in rows]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Prepared rows must have unique nonempty string IDs")
    for row in rows:
        if row.get("split") not in {"train", "test"} or not isinstance(row.get("inputs"), dict):
            raise ValueError("Invalid prepared split or inputs")
        target = row.get("target")
        if not isinstance(target, numbers.Real) or not math.isfinite(target):
            raise ValueError("Invalid prepared target")
    expected = data.get("prepared_sha256")
    if expected is None or expected != _digest(
        {k: v for k, v in data.items() if k != "prepared_sha256"}
    ):
        raise ValueError("Prepared artifact checksum mismatch; run prepare again")


def prepare(protocol: str, *, source: str | Path, output: str | Path, **options) -> dict:
    """Validate local upstream resources, snapshot protocol data and hash it.

    Resource acquisition commands and licences are documented per protocol.
    Preparation never executes user model code or uploads data.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    plugin = _plugin(protocol)
    data = plugin.prepare(Path(source), protocol_id=protocol, **options)
    if data["protocol_id"] != protocol:
        raise ValueError("Plugin returned a different protocol")
    data["schema_version"] = "1.0"
    data["prepared_sha256"] = _digest(data)
    _validate_prepared(data)
    path = _new_output(output)
    _write(path / "prepared.json", data)
    return data


def _inputs(rows: list[dict]) -> list[dict]:
    return [dict(copy.deepcopy(row["inputs"]), id=row["id"]) for row in rows]


def _checked_predictions(
    values: Mapping, expected: set[str], *, partial: bool
) -> tuple[dict, dict]:
    if not isinstance(values, Mapping):
        raise ValueError("Adapter must return a mapping keyed by stable input ID")
    unknown = set(values) - expected
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} unknown IDs")
    scores, failures = {}, {}
    for ident, value in values.items():
        if isinstance(value, Mapping):
            if value.get("score") is None:
                reason = value.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"Unscored ID {ident} needs a reason")
                failures[ident] = reason
                continue
            value = value["score"]
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Score for {ident} must be finite numeric data")
        scores[ident] = float(value)
    missing = expected - set(values)
    if missing and not partial:
        raise ValueError(
            f"Missing {len(missing)} prediction IDs; use explicit unscored reasons or allow_partial"
        )
    failures.update({ident: "missing_prediction" for ident in sorted(missing)})
    return scores, failures


def read_predictions(path: str | Path) -> dict:
    """Read keyed JSON or CSV/TSV with id,score and optional reason columns."""
    path = Path(path)
    if path.suffix == ".json":

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        data = json.loads(path.read_text(), object_pairs_hook=pairs)
        return data
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ","):
            ident = row.get("id")
            if not ident or ident in result:
                raise ValueError("Prediction file contains a missing or duplicate ID")
            value = row.get("score")
            result[ident] = (
                {"score": None, "reason": row.get("reason", "")}
                if value in (None, "", "NA")
                else float(value)
            )
    return result


def _model_info(model: dict | None) -> dict:
    model = model or {}
    return {
        "name": str(model.get("name", "private model")),
        "training_overlap": str(model.get("training_overlap", "unreported")),
    }


def evaluate(
    prepared: dict | str | Path,
    predictions: Mapping | str | Path,
    *,
    output: str | Path,
    model: dict | None = None,
    allow_partial: bool = False,
    execution: dict | None = None,
) -> dict:
    """Score supplied predictions. Imported scores do not prove model execution."""
    if Path(output).exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    data = _load(prepared, "prepared.json")
    _validate_prepared(data)
    values = read_predictions(predictions) if isinstance(predictions, (str, Path)) else predictions
    expected = {row["id"] for row in data["rows"] if row["split"] == "test"}
    scores, failures = _checked_predictions(values, expected, partial=allow_partial)
    normalized = {
        **scores,
        **{key: {"score": None, "reason": reason} for key, reason in failures.items()},
    }
    if not scores:
        raise ValueError("No scored predictions")
    result = _plugin(data["protocol_id"]).score(data, scores)
    # Core coverage has precedence; plugins may provide additional assay coverage.
    denominator = result.get("coverage", {}).get("denominator", len(expected))
    if type(denominator) is not int or denominator < len(expected):
        raise ValueError("Protocol denominator cannot shrink below the selected evaluation cohort")
    coverage = {
        "denominator": denominator,
        "scored": len(scores),
        "unscored": denominator - len(scores),
    }
    full = data["scope"] == "full" and coverage["unscored"] == 0
    report = {
        "schema_version": "1.0",
        "kind": "rewire_local_evaluation",
        "protocol_id": data["protocol_id"],
        "protocol_version": data["protocol_version"],
        "dataset_id": data["dataset_id"],
        "scope": data["scope"],
        "completion": "complete" if full else "partial",
        "model": _model_info(model),
        "model_configuration": copy.deepcopy((model or {}).get("configuration", {})),
        "input_information": str((model or {}).get("input_information", "unreported")),
        "protocol_configuration": copy.deepcopy(data.get("metadata", {})),
        "metrics": result.get("metrics", result),
        "protocol_results": result,
        "coverage": coverage,
        "coverage_context": {
            "selected": len(expected),
            "not_selected": denominator - len(expected),
            "unscored_file_scope": "selected inputs only; not_selected counted separately",
        },
        "prepared_sha256": data["prepared_sha256"],
        "predictions_sha256": _digest(normalized),
        "provenance": data.get("provenance", {}),
        "execution": execution or {"mode": "imported_predictions", "inference_seconds": None},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "rewirebench": _version(),
            "sdk_code_sha256": _code_digest(),
            "container_digest": os.environ.get("REWIRE_CONTAINER_DIGEST", "unreported"),
            "sif_sha256": os.environ.get("REWIRE_SIF_SHA256", "unreported"),
            "container_runtime": os.environ.get("REWIRE_CONTAINER_RUNTIME", "unreported"),
        },
        "timing_seconds": {"evaluation": time.perf_counter() - start},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "unreviewed_local_run",
        "independently_reproduced": False,
    }
    path = _new_output(output)
    _write(path / "report.json", report)
    _write(path / "predictions.json", normalized)
    _write(path / "unscored.json", failures)
    return report


def _code_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _version() -> str:
    try:
        return importlib.metadata.version("rewirebench")
    except importlib.metadata.PackageNotFoundError:
        return "development"


def run(
    prepared: dict | str | Path,
    adapter: ScoreAdapter | EmbeddingAdapter,
    *,
    output: str | Path,
    model: dict | None = None,
    batch_size: int = 32,
    allow_partial: bool = False,
) -> dict:
    """Run caller-supplied Python code locally, then evaluate its outputs."""
    if Path(output).exists():
        raise FileExistsError(output)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    data = _load(prepared, "prepared.json")
    _validate_prepared(data)
    start = time.perf_counter()
    train = [r for r in data["rows"] if r["split"] == "train"]
    test = [r for r in data["rows"] if r["split"] == "test"]
    embedded = data["protocol_id"] == "mfass-v2-frozen-encoder"
    if embedded:
        if not hasattr(adapter, "embed"):
            raise ValueError("Frozen encoder protocol requires adapter.embed(inputs)")
        rows, method = data["rows"], adapter.embed
    else:
        if not hasattr(adapter, "predict"):
            raise ValueError("Scalar protocol requires adapter.predict(inputs)")
        if hasattr(adapter, "fit"):
            if data["protocol_id"] != "mfass-v2":
                raise ValueError("Fitting is forbidden by the selected zero-shot protocol")
            adapter.fit(_inputs(train), [r["target"] for r in train])
        rows, method = test, adapter.predict
    outputs = {}
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        values = method(_inputs(batch))
        if not isinstance(values, Mapping) or set(values) - {r["id"] for r in batch}:
            raise ValueError("Adapter returned invalid or out-of-batch IDs")
        if outputs.keys() & values.keys():
            raise ValueError("Adapter returned duplicate IDs")
        outputs.update(values)
    if embedded:
        if set(outputs) != {r["id"] for r in rows}:
            raise ValueError("Embedding head requires complete train/test embeddings")
        outputs = _plugin(data["protocol_id"]).fit_embeddings(data, outputs)
    execution = {
        "mode": "local_adapter",
        "adapter": type(adapter).__name__,
        "inference_and_fit_seconds": time.perf_counter() - start,
        "batch_size": batch_size,
    }
    # Adapter declarations are local provenance, never automatic verified claims.
    declared = getattr(adapter, "provenance", None)
    if callable(declared):
        declared = declared()
    if isinstance(declared, dict):
        execution["adapter_provenance"] = declared
    return evaluate(
        data, outputs, output=output, model=model, allow_partial=allow_partial, execution=execution
    )


def _numeric_tree(value):
    """Export metrics only: exclude arbitrary strings, paths or raw model config."""
    if value is None:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Nonfinite metric")
        return value
    if isinstance(value, dict):
        return {
            str(k): _numeric_tree(v)
            for k, v in value.items()
            if isinstance(v, (dict, numbers.Real)) or v is None
        }
    raise ValueError("Export metrics must be numeric mappings")


def export(report: dict | str | Path, *, output: str | Path) -> dict:
    """Write a small, inspectable aggregate bundle. Does not upload anything."""
    data = _load(report, "report.json")
    if data.get("kind") != "rewire_local_evaluation":
        raise ValueError("Expected a local evaluation report")
    if data.get("scope") == "smoke":
        raise ValueError("Smoke tests cannot be exported as benchmark contributions")
    evidence_path = Path(output).with_suffix(".evidence.json")
    if Path(output).exists() or evidence_path.exists():
        raise FileExistsError("Export and evidence paths must both be new")
    keys = ("protocol_id", "protocol_version", "dataset_id", "scope", "completion", "coverage")
    bundle = {key: copy.deepcopy(data[key]) for key in keys}
    bundle.update(
        {
            "schema_version": "1.0",
            "kind": "rewire_benchmark_submission",
            "model": _model_info(data.get("model")),
            "metrics": _numeric_tree(
                data["metrics"]
                or {
                    "per_assay": {
                        key: {
                            "metrics": value["metrics"],
                            "scored": value["scored"],
                            "denominator": value["eligible"],
                        }
                        for key, value in data.get("protocol_results", {})
                        .get("per_assay", {})
                        .items()
                        if value.get("metrics")
                    }
                }
            ),
            "review_status": "unreviewed_contribution",
            "execution_status": data["execution"]["mode"],
            "independently_reproduced": False,
            "prepared_sha256": data["prepared_sha256"],
            "predictions_sha256": data["predictions_sha256"],
        }
    )
    # Only known digest/revision fields leave the local report. Never dump config.
    provenance = dict(data.get("provenance", {}))
    provenance["sdk_code_sha256"] = data.get("environment", {}).get("sdk_code_sha256")
    environment = data.get("environment", {})
    if _is_sha256(environment.get("sif_sha256")):
        provenance["sif_sha256"] = environment["sif_sha256"]
    image_digest = str(environment.get("container_digest", ""))
    if image_digest.startswith("sha256:") and _is_sha256(image_digest[7:]):
        provenance["oci_image_sha256"] = image_digest[7:]
    for key, value in data.get("execution", {}).get("adapter_provenance", {}).items():
        if key.endswith(("_sha256", "_revision")):
            provenance["model_" + key] = value
    bundle["provenance"] = {
        k: v
        for k, v in provenance.items()
        if (k.endswith("_sha256") or k.endswith("_revision"))
        and isinstance(v, str)
        and len(v) in {40, 64}
        and all(c in "0123456789abcdef" for c in v)
    }
    evidence = {"schema_version": "1.0", "assays": {}, "public_model_artifacts": {}}
    if data["protocol_id"] == "proteingym-v1.3-dms-substitutions":
        from rewirebench.protocols.proteingym import resource_path

        with resource_path("DMS_substitutions.csv").open(newline="") as source:
            allowed_assays = {row["DMS_id"] for row in csv.DictReader(source)}
        for name, digest in data.get("provenance", {}).get("assay_sha256", {}).items():
            if name in allowed_assays and _is_sha256(digest):
                evidence["assays"][name] = digest
    from rewirebench.adapters.mfass import DNABERT2

    for name, digest in (
        data.get("execution", {}).get("adapter_provenance", {}).get("artifact_sha256", {}).items()
    ):
        if name in DNABERT2.ARTIFACTS and digest == DNABERT2.ARTIFACTS[name]:
            evidence["public_model_artifacts"][name] = digest
    bundle["provenance"]["evidence_manifest_sha256"] = _digest(evidence)
    verification = data.get("provenance", {}).get("data_verification", "unreported")
    if data["protocol_id"].startswith("mfass-v2"):
        from rewirebench.protocols.mfass import COHORT_SHA256

        if bundle["provenance"].get("cohort_sha256") == COHORT_SHA256:
            verification = "pinned_source_bytes"
    bundle["data_verification"] = (
        verification
        if verification
        in {"pinned_source_bytes", "local_bytes_hashed_not_independently_source_verified"}
        else "unreported"
    )
    _write(Path(output), bundle)
    _write(evidence_path, evidence)
    return bundle


def _is_sha256(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )
