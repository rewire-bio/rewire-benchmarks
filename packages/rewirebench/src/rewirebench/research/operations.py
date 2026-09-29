"""Registered scientific operations. No model-supplied code, imports or commands."""
from __future__ import annotations

import importlib.metadata
import inspect
import ipaddress
import itertools
import json
import math
import platform
import socket
import sys
import tomllib
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

from rewirebench import sdk
from rewirebench.metrics import precision_at_n, recall_at_n

from .contracts import (
    OPERATION_SCHEMA,
    RECIPES,
    digest,
    file_sha,
    read_json,
    validate,
    validate_manifest,
    validate_table,
)


class BlockedOperation(ValueError):
    """The evidence cannot support this operation under its registered assumptions."""


def _public_url(url):
    parsed = urllib.parse.urlparse(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.port not in {None, 443}):
        raise BlockedOperation("Artifact retrieval permits public HTTPS URLs only")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise BlockedOperation("Artifact host must resolve to public addresses")


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def materialize_artifacts(manifest, resolver, cache, identifiers, max_bytes):
    """Fill absent resolver entries only. Never replace or modify supplied files."""
    result = dict(resolver)
    for artifact in manifest["artifacts"]:
        if artifact["id"] not in identifiers or artifact["id"] in result or not artifact["uri"]:
            continue
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache/artifact["sha256"]
        if not destination.exists():
            _public_url(artifact["uri"])
            temporary = destination.with_suffix(".partial")
            try:
                # No credentials, cookies, arbitrary headers or proxy authentication.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _PublicRedirect())
                with opener.open(artifact["uri"], timeout=30) as response, temporary.open("xb") as stream:
                    size = 0
                    while chunk := response.read(1024*1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise BlockedOperation("Artifact exceeds campaign storage budget")
                        stream.write(chunk)
                if file_sha(temporary) != artifact["sha256"]:
                    raise BlockedOperation("Downloaded artifact checksum mismatch")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if file_sha(destination) != artifact["sha256"]:
            raise BlockedOperation("Cached artifact checksum mismatch")
        result[artifact["id"]] = str(destination.resolve())
    return result


def resolve_artifact(manifest, resolver, identifier):
    artifact = next((a for a in manifest["artifacts"] if a["id"] == identifier), None)
    if artifact is None or identifier not in resolver:
        raise BlockedOperation(f"Artifact is not available locally: {identifier}")
    path = Path(resolver[identifier])
    if not path.is_absolute() or not path.is_file():
        raise BlockedOperation(f"Resolver must identify an existing absolute file: {identifier}")
    if file_sha(path) != artifact["sha256"]:
        raise BlockedOperation(f"Artifact checksum mismatch: {identifier}")
    return path


def load_table(manifest, resolver):
    path = resolve_artifact(manifest, resolver, manifest["table_artifact_id"])
    return validate_table(read_json(path), manifest)


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def metric_values(rows, method, metric_kind, *, capacity=100):
    selected = [r for r in rows if r["scores"][method] is not None]
    if not selected:
        return {}
    y = np.array([r["y"] for r in selected], dtype=float)
    p = np.array([r["scores"][method] for r in selected], dtype=float)
    varying = len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if metric_kind == "mfass":
            binary = len(np.unique(y)) == 2
            return {"precision_at_capacity": precision_at_n(y, p, capacity),
                    "recall_at_capacity": recall_at_n(y, p, capacity),
                    "average_precision_sklearn": float(average_precision_score(y, p))
                    if binary else None,
                    "auroc": float(roc_auc_score(y, p)) if binary else None}
        if metric_kind == "proteingym":
            from rewirebench.protocols.proteingym import _metrics, _ndcg
            b = [r["features"].get("binary_target") for r in selected]
            if all(v in {0, 1} for v in b):
                return _clean(_metrics(y, p, np.asarray(b)))
            top_y, top_p = y >= np.percentile(y, 90), p >= np.percentile(p, 90)
            return _clean({"Spearman": round(float(spearmanr(y, p).statistic), 3)
                           if varying else None, "NDCG": round(_ndcg(y, p), 3),
                           "Top_recall": round(float(np.sum(top_y & top_p) / np.sum(top_y)), 3)})
        return _clean({"mse": float(np.mean((p-y)**2)),
                       "pearson": float(pearsonr(y, p).statistic) if varying else None,
                       "spearman": float(spearmanr(y, p).statistic) if varying else None,
                       "ndcg": float(ndcg_score((y-y.min())[None, :], p[None, :]))
                       if len(y) >= 2 and np.ptp(y) > 0 else None})


def methods_for(operation, manifest):
    methods = operation["methods"] or list(manifest["expected_metrics"])
    if len(methods) != len(set(methods)) or set(methods) - set(manifest["expected_metrics"]):
        raise BlockedOperation("Unknown or repeated method; comparisons stay within this manifest")
    return methods


def validate_operation(operation, manifest):
    validate(operation, OPERATION_SCHEMA)
    methods = methods_for(operation, manifest)
    kind = operation["kind"]
    if kind in {"paired", "bootstrap"} and len(methods) < 2:
        raise BlockedOperation("A paired comparison requires at least two methods")
    if kind == "bootstrap" and not operation["metric"]:
        raise BlockedOperation("Bootstrap requires a preregistered metric")
    if kind == "subgroups" and operation["field"] not in manifest["semantics"]["subgroup_fields"]:
        raise BlockedOperation("Subgroup field is not registered in the manifest")
    if kind == "local_recipe" and (operation["recipe"] not in manifest["local_recipes"]
                                   or operation["recipe"] not in RECIPES):
        raise BlockedOperation("Recipe is not registered and permitted by this manifest")
    if kind != "local_recipe" and operation["recipe"] is not None:
        raise BlockedOperation("Recipe supplied to an unrelated operation")
    if operation["field"] is not None and kind != "subgroups":
        raise BlockedOperation("Field supplied to an unrelated operation")
    if operation["metric"] is not None and any(
        operation["metric"] not in manifest["expected_metrics"][m] for m in methods
    ):
        raise BlockedOperation("Metric is not registered for all selected methods")
    return methods


def replay(table, manifest, methods):
    metrics, checks, coverage = {}, [], {}
    for method in methods:
        metrics[method] = metric_values(table["rows"], method, table["metric_kind"])
        n = sum(r["scores"][method] is not None for r in table["rows"])
        coverage[method] = {"scored": n, "denominator": len(table["rows"]),
                            "missing": len(table["rows"])-n}
        for metric, expected in manifest["expected_metrics"][method].items():
            actual = metrics[method].get(metric)
            exists = metric in metrics[method]
            passed = exists and (actual is None and expected is None or
                     actual is not None and expected is not None and
                     abs(actual-expected) <= manifest["metric_tolerance"])
            checks.append({"method": method, "metric": metric, "expected": expected,
                           "actual": actual, "status": "passed" if passed else "failed",
                           "tolerance": manifest["metric_tolerance"], "metric_available": exists})
    return {"metrics": metrics, "checks": checks, "coverage": coverage}


def subgroup_bins(rows, field):
    """Fixed manifest labels, or fixed quartile rule using annotations only, never errors."""
    available = [r["features"].get(field) for r in rows if r["features"].get(field) is not None]
    if not available:
        raise BlockedOperation("No values for the registered annotation")
    if all(isinstance(v, (float, int)) for v in available) and len(set(available)) > 8:
        edges = np.unique(np.quantile(available, [.25, .5, .75])).tolist()
        groups = {}
        for row in rows:
            value = row["features"].get(field)
            label = "missing" if value is None else f"quartile-{np.searchsorted(edges, value, side='left')+1}"
            groups.setdefault(label, []).append(row)
        return groups, {"rule": "annotation-only quartiles; right-closed; duplicate edges removed",
                        "edges": edges}
    if len(set(map(str, available))) > 40:
        raise BlockedOperation("More than forty annotation categories; curate bins before analysis")
    groups = {}
    for row in rows:
        value = row["features"].get(field)
        label = "missing" if value is None else str(value)
        groups.setdefault(label, []).append(row)
    return groups, {"rule": "predeclared annotation categories", "edges": []}


def paired(table, methods, metric=None):
    pairs = []
    for a, b in itertools.combinations(methods, 2):
        common = [r for r in table["rows"] if all(r["scores"][m] is not None for m in (a, b))]
        ma = metric_values(common, a, table["metric_kind"])
        mb = metric_values(common, b, table["metric_kind"])
        deltas = {k: mb[k]-ma[k] if ma[k] is not None and mb.get(k) is not None else None
                  for k in ma if metric is None or metric == k}
        pairs.append({"reference": a, "candidate": b, "common_n": len(common),
                      "original_n": len(table["rows"]), "candidate_minus_reference": deltas,
                      "reference_metrics": ma, "candidate_metrics": mb})
    return {"pairs": pairs}


def bootstrap(table, manifest, methods, metric, draws=500):
    rows = table["rows"]
    if not manifest["semantics"]["independent_unit"] or any(r["group"] is None for r in rows):
        raise BlockedOperation("No documented independence unit; confirmatory inference is forbidden")
    results = []
    for a, b in itertools.combinations(methods, 2):
        common = [r for r in rows if all(r["scores"][m] is not None for m in (a, b))]
        groups = sorted({r["group"] for r in common})
        if len(groups) < 5:
            raise BlockedOperation("Fewer than five independent groups")
        index = {g: [r for r in common if r["group"] == g] for g in groups}
        rng, deltas, skipped = np.random.default_rng(20260923), [], 0
        observed = paired({**table, "rows": common}, [a, b], metric)["pairs"][0]
        if observed["candidate_minus_reference"].get(metric) is None:
            raise BlockedOperation("Metric is undefined on the common population")
        for _ in range(draws):
            sample = [r for g in rng.choice(groups, len(groups), replace=True) for r in index[g]]
            if table["metric_kind"] == "mfass" and len({r["y"] for r in sample}) < 2:
                skipped += 1
                continue
            cap = max(1, round(100*len(sample)/len(common)))
            ma = metric_values(sample, a, table["metric_kind"], capacity=cap).get(metric)
            mb = metric_values(sample, b, table["metric_kind"], capacity=cap).get(metric)
            if ma is None or mb is None:
                skipped += 1
            else:
                deltas.append(mb-ma)
        if skipped > .05*draws:
            raise BlockedOperation("More than five percent of grouped bootstrap draws are undefined")
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        results.append({"reference": a, "candidate": b, "metric": metric,
                        "observed_delta": observed["candidate_minus_reference"][metric],
                        "ci95_low": float(lo), "ci95_high": float(hi), "draws": draws,
                        "skipped": skipped, "independent_groups": len(groups),
                        "common_n": len(common), "seed": 20260923,
                        "capacity_rule": "100 / common_n held proportional within resamples"})
    return {"bootstrap": results, "inference": "exploratory unadjusted intervals; no significance claim"}


def local_recipe(manifest, resolver, recipe, output):
    if manifest.get("runner_code_sha256") != sdk._code_digest():
        raise BlockedOperation("Installed runner code differs from its pinned implementation hash")
    for role in ("recipe_code", "environment"):
        refs = [a for a in manifest["artifacts"] if a["role"] == role]
        if not refs:
            raise BlockedOperation(f"Pinned {role} evidence is required before local execution")
        for ref in refs:
            resolve_artifact(manifest, resolver, ref["id"])
    prepared_refs = [a for a in manifest["artifacts"] if a["role"] == "prepared"]
    if len(prepared_refs) != 1:
        raise BlockedOperation("Exactly one pinned SDK prepared snapshot is required")
    ref = prepared_refs[0]
    data = read_json(resolve_artifact(manifest, resolver, ref["id"]))
    sdk._validate_prepared(data)
    native_protocol = manifest.get("sdk_protocol_id")
    if data["protocol_id"] != native_protocol:
        raise BlockedOperation("Prepared protocol differs from the investigation")
    if ref.get("semantic_sha256") and data["prepared_sha256"] != ref["semantic_sha256"]:
        raise BlockedOperation("Prepared semantic checksum differs from the pinned snapshot")
    table = load_table(manifest, resolver)
    test = {r["id"]: r for r in data["rows"] if r["split"] == "test"}
    if set(test) != {r["id"] for r in table["rows"]}:
        raise BlockedOperation("Prepared and normalized identifiers differ; do not reprepare snapshots")
    if any(test[r["id"]]["target"] != r["y"] for r in table["rows"]):
        raise BlockedOperation("Prepared and normalized outcomes differ")
    from rewirebench.adapters.sequence import SeededRandomScore, SequenceComposition, TrainMean
    if recipe == "sdk:train-mean-v1":
        adapter = TrainMean()
    elif recipe == "sdk:sequence-composition-v1":
        protein = native_protocol.startswith("flip2")
        adapter = SequenceComposition("ACDEFGHIKLMNPQRSTVWY" if protein else "ACGT")
    elif recipe == "sdk:seeded-random-v1":
        adapter = SeededRandomScore(seed=0)
    elif recipe == "sdk:mfass-kmer-v2":
        if native_protocol != "mfass-v2":
            raise BlockedOperation("MFASS baseline requires corrected v2 protocol")
        from rewirebench.adapters.mfass import KmerBaseline
        adapter = KmerBaseline()
    elif recipe == "sdk:esm2-8m-v1":
        weights = [a for a in manifest["artifacts"] if a["role"] == "checkpoint"]
        if len(weights) != 1 or not native_protocol.startswith("proteingym"):
            raise BlockedOperation("ESM recipe requires ProteinGym and one pinned local checkpoint")
        from rewirebench.adapters.esm import ESM2Adapter
        adapter = ESM2Adapter(resolve_artifact(manifest, resolver, weights[0]["id"]), device="cpu")
    else:
        raise BlockedOperation("Unregistered local recipe")
    actual_source = inspect.getsourcefile(type(adapter))
    pinned_code = {a["sha256"] for a in manifest["artifacts"] if a["role"] == "recipe_code"}
    if not actual_source or file_sha(actual_source) not in pinned_code:
        raise BlockedOperation("Imported adapter implementation differs from pinned recipe code")
    env_ref = next(a for a in manifest["artifacts"] if a["role"] == "environment")
    locked = tomllib.loads(resolve_artifact(manifest, resolver, env_ref["id"]).read_text())
    required = ["numpy", "scipy", "scikit-learn"]
    if recipe == "sdk:esm2-8m-v1":
        required += ["torch", "fair-esm"]
    installed = {name: importlib.metadata.version(name) for name in required}
    for name, version in installed.items():
        if version not in {p.get("version") for p in locked.get("package", []) if p["name"] == name}:
            raise BlockedOperation(f"Installed {name} version differs from pinned environment lock")
    report = sdk.run(data, adapter, output=output, batch_size=16,
                     model={"name": recipe, "training_overlap": "protocol-managed; pretrained overlap unknown"})
    result = {k: report[k] for k in ("metrics", "coverage")}
    result.update(recipe=recipe, protocol_id=native_protocol,
                  scope=report.get("scope"), prepared_sha256=data["prepared_sha256"],
                  fitting=report.get("execution", {}).get("fitting", "unreported"),
                  report_sha256=file_sha(Path(output)/"report.json"))
    result["runtime"] = {"python": platform.python_version(), "platform": platform.system(),
                         "dependencies": installed, "adapter_sha256": file_sha(actual_source),
                         "runner_code_sha256": sdk._code_digest()}
    if "per_assay" in report:
        result["per_assay"] = report["per_assay"]
    return _clean(result)


def execute_operation(manifest, resolver, operation, output, *, cache=None, max_bytes=20*1024**3):
    validate_manifest(manifest)
    methods = validate_operation(operation, manifest)
    if cache:
        identifiers = ({a["id"] for a in manifest["artifacts"]}
                       if operation["kind"] in {"verify", "local_recipe"}
                       else {manifest["table_artifact_id"]})
        resolver = materialize_artifacts(manifest, resolver, Path(cache), identifiers, max_bytes)
    table = load_table(manifest, resolver)
    kind, rows = operation["kind"], table["rows"]
    limitations = ["Previously explored benchmark outcomes; this is not independent validation.",
                   "Descriptive analysis only; multiple comparisons are not confirmatory tests."]
    if kind == "verify":
        checks = []
        for a in manifest["artifacts"]:
            try:
                resolve_artifact(manifest, resolver, a["id"])
                checks.append({"artifact_id": a["id"], "status": "passed"})
            except BlockedOperation as exc:
                checks.append({"artifact_id": a["id"], "status": "failed", "reason": str(exc)})
        numerical = {"checks": checks, "rows": len(rows), "identifiers_unique": True,
                     "methods": list(manifest["expected_metrics"])}
    elif kind == "replay":
        numerical = replay(table, manifest, methods)
    elif kind == "coverage":
        numerical = {"coverage": replay(table, manifest, methods)["coverage"],
                     "common_all_methods": sum(all(r["scores"][m] is not None for m in methods)
                                               for r in rows), "original_n": len(rows)}
    elif kind == "paired":
        numerical = paired(table, methods, operation["metric"])
    elif kind == "bootstrap":
        numerical = bootstrap(table, manifest, methods, operation["metric"])
    elif kind == "subgroups":
        groups, binning = subgroup_bins(rows, operation["field"])
        numerical = {"field": operation["field"], "binning": binning,
                     "subgroups": [{"label": label, "n": len(sample),
                       "outcome_mean": float(np.mean([r["y"] for r in sample])),
                       "methods": {m: {"scored": sum(r["scores"][m] is not None for r in sample),
                                      "metrics": metric_values(sample, m, table["metric_kind"])}
                                   for m in methods}} for label, sample in sorted(groups.items())]}
    elif kind == "sensitivity":
        numerical = {"methods": {}}
        for method in methods:
            scored = [r for r in rows if r["scores"][method] is not None]
            reversed_rows = [{**r, "scores": {**r["scores"], method: -r["scores"][method]}}
                             for r in scored]
            constant_rows = [{**r, "scores": {**r["scores"], method: 0.0}} for r in scored]
            numerical["methods"][method] = {
                "original": metric_values(scored, method, table["metric_kind"]),
                "sign_reversed_diagnostic": metric_values(reversed_rows, method, table["metric_kind"]),
                "constant_ranking_control": metric_values(constant_rows, method, table["metric_kind"]),
                "unique_predictions": len({r["scores"][method] for r in scored}),
                "scored": len(scored)}
        limitations.append("Sign reversal is diagnostic and never changes the declared score direction.")
    else:
        numerical = local_recipe(manifest, resolver, operation["recipe"], output)
    table_ref = next(a for a in manifest["artifacts"] if a["id"] == manifest["table_artifact_id"])
    return {"operation_id": operation["id"], "kind": kind, "manifest_sha256": digest(manifest),
            "table_sha256": table_ref["sha256"], "code_sha256": sdk._code_digest(),
            "numerical": _clean(numerical), "limitations": limitations}


def main():
    """Private worker entry point, called with argv by the resource supervisor."""
    request, destination = read_json(sys.argv[1]), Path(sys.argv[2])
    try:
        if request.get("expected_code_sha256") != sdk._code_digest():
            raise BlockedOperation("Runner implementation changed during campaign")
        value = execute_operation(request["manifest"], request["resolver"], request["operation"],
                                  destination.parent/"sdk-output", cache=request.get("cache"),
                                  max_bytes=request.get("max_bytes", 20*1024**3))
        result = {"status": "completed", "receipt": value, "error": None}
        if any(check.get("status") == "failed" for check in value["numerical"].get("checks", [])):
            result.update(status="blocked", error="Evidence or metric replay failed; dependent analyses blocked")
    except (ValueError, OSError, ImportError) as exc:
        result = {"status": "blocked" if isinstance(exc, BlockedOperation) else "failed",
                  "receipt": None, "error": str(exc)}
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
