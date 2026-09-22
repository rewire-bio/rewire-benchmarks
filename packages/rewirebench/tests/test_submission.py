import io
import json
import urllib.error
from unittest.mock import patch

import pytest
from rewirebench.submission import SubmissionError, submit


def bundle():
    return {
        "schema_version": "1.0",
        "kind": "rewire_benchmark_submission",
        "protocol_id": "mfass-v2",
        "protocol_version": "2",
        "dataset_id": "mfass",
        "scope": "full",
        "completion": "complete",
        "model": {"name": "private method", "training_overlap": "unreported"},
        "metrics": {"auroc": 0.7},
        "coverage": {"denominator": 2, "scored": 2, "unscored": 0},
        "provenance": {"code_revision": "a" * 40},
        "execution_status": "local_adapter",
        "review_status": "unreviewed_contribution",
        "independently_reproduced": False,
        "prepared_sha256": "b" * 64,
        "predictions_sha256": "c" * 64,
    }


def arguments():
    return {
        "title": "Example evaluation",
        "summary": "An explicitly submitted local evaluation.",
        "source_url": "https://example.org/paper",
        "metric": "auroc",
        "value": "0.7",
        "source_locator": "Table 1",
    }


def test_dry_run_is_offline_and_retry_key_stable():
    with patch("urllib.request.build_opener", side_effect=AssertionError("network forbidden")):
        first = submit(bundle(), **arguments(), dry_run=True)
        second = submit(bundle(), **arguments(), dry_run=True)
    assert first == second
    assert "token" not in json.dumps(first)
    assert (
        first["contribution"]["details"]["rewire_bundle"]["review_status"]
        == "unreviewed_contribution"
    )


@pytest.mark.parametrize(
    "field,value",
    [("weights", "private.pt"), ("scope", "smoke"), ("independently_reproduced", True)],
)
def test_private_or_unearned_claims_refused(field, value):
    data = bundle()
    data[field] = value
    with pytest.raises(ValueError):
        submit(data, **arguments(), dry_run=True)


def test_verified_token_required():
    with pytest.raises(ValueError, match="verified-email"):
        submit(bundle(), **arguments())


@pytest.mark.parametrize("status,text", [(503, "disabled"), (401, "Verify"), (429, "rate limit")])
def test_service_gates_and_token_redaction(status, text):
    class Opener:
        def open(self, request, timeout):
            assert request.headers["Authorization"] == "Bearer private-token"
            raise urllib.error.HTTPError(request.full_url, status, "private-token", {}, None)

    with (
        patch("urllib.request.build_opener", return_value=Opener()),
        pytest.raises(SubmissionError, match=text) as error,
    ):
        submit(bundle(), **arguments(), token="private-token")
    assert "private-token" not in str(error.value)


def test_success_remains_pending_review():
    class Opener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            assert payload["contribution"]["type"] == "result"
            return io.BytesIO(b'{"result":{"data":{"id":"submission-id","status":"submitted"}}}')

    with patch("urllib.request.build_opener", return_value=Opener()):
        result = submit(bundle(), **arguments(), token="verified")
    assert result["publication_status"] == "pending_review"


def test_remote_http_refused():
    with pytest.raises(ValueError, match="HTTPS"):
        submit(bundle(), **arguments(), token="secret", endpoint="http://example.org/trpc")


@pytest.mark.parametrize("response", [b"null", b'{"result":null}', b'{"result":{}}'])
def test_malformed_acknowledgement_is_uncertain(response):
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(response)

    with (
        patch("urllib.request.build_opener", return_value=Opener()),
        pytest.raises(SubmissionError, match="not acknowledged"),
    ):
        submit(bundle(), **arguments(), token="verified")


@pytest.mark.parametrize("metrics", [{"auroc": None}, {"n": 2, "positives": 1, "capacity": 100, "prevalence": 0.5}])
def test_empty_numerical_metrics_rejected(metrics):
    payload = bundle()
    payload["metrics"] = metrics
    with pytest.raises(ValueError, match="No numerical"):
        submit(payload, **arguments(), dry_run=True)


@pytest.mark.parametrize("metrics", [
    {"predictions": {"private-variant": 0.3}},
    {"/private/model.pt": 1},
    {"auroc": 0.7, "credentials": 0},
    {"auroc": {"raw": 0.7}},
    {"auroc": True},
    {"auroc": float("nan")},
    {"auroc": float("inf")},
])
def test_unknown_nested_and_private_metric_fields_refused(metrics):
    payload = bundle()
    payload["metrics"] = metrics
    with pytest.raises(ValueError):
        submit(payload, **arguments(), dry_run=True)


def test_unsupported_protocol_refused():
    payload = bundle()
    payload["protocol_id"] = "private-unregistered-protocol"
    with pytest.raises(ValueError, match="Unsupported protocol"):
        submit(payload, **arguments(), dry_run=True)


def proteingym_bundle():
    from rewirebench.submission import _proteingym_assays

    assay_id, denominator = next(iter(_proteingym_assays().items()))
    payload = bundle()
    payload.update({
        "protocol_id": "proteingym-v1.3-dms-substitutions", "protocol_version": "1.3",
        "dataset_id": "proteingym-dms-substitutions-v1.3", "scope": "subset",
        "completion": "partial",
        "metrics": {"per_assay": {assay_id: {
            "metrics": {"Spearman": 0.5, "AUC": None, "MCC": None,
                        "NDCG": None, "Top_recall": None},
            "scored": 2, "denominator": denominator,
        }}},
        "coverage": {"denominator": denominator, "scored": 2, "unscored": denominator - 2},
    })
    return payload, assay_id


def test_official_proteingym_assay_summary_accepted():
    payload, _ = proteingym_bundle()
    result = submit(payload, **arguments(), dry_run=True)
    assert result["contribution"]["details"]["rewire_bundle"]["metrics"] == payload["metrics"]


@pytest.mark.parametrize("replacement", ["/private/weights.pt", "private-variant", "predictions"])
def test_proteingym_only_official_assay_names(replacement):
    payload, assay_id = proteingym_bundle()
    payload["metrics"]["per_assay"][replacement] = payload["metrics"]["per_assay"].pop(assay_id)
    with pytest.raises(ValueError, match="official ProteinGym assay IDs"):
        submit(payload, **arguments(), dry_run=True)


def test_proteingym_counts_do_not_substitute_for_metric_values():
    payload, assay_id = proteingym_bundle()
    summary = payload["metrics"]["per_assay"][assay_id]
    summary["metrics"] = dict.fromkeys(summary["metrics"])
    with pytest.raises(ValueError, match="No numerical"):
        submit(payload, **arguments(), dry_run=True)


@pytest.mark.parametrize("extra", [
    {"predictions": {"private-variant": 1}}, {"path": 1}, {"embeddings": [1, 2]},
])
def test_proteingym_summary_extra_fields_rejected(extra):
    payload, assay_id = proteingym_bundle()
    payload["metrics"]["per_assay"][assay_id].update(extra)
    with pytest.raises(ValueError, match="allow only"):
        submit(payload, **arguments(), dry_run=True)


def test_proteingym_scoped_coverage_must_reconcile():
    payload, assay_id = proteingym_bundle()
    payload["metrics"]["per_assay"][assay_id]["scored"] = 1
    with pytest.raises(ValueError, match="reconcile"):
        submit(payload, **arguments(), dry_run=True)


def test_proteingym_denominator_cannot_shrink():
    payload, assay_id = proteingym_bundle()
    payload["metrics"]["per_assay"][assay_id]["denominator"] = 2
    with pytest.raises(ValueError, match="official denominator"):
        submit(payload, **arguments(), dry_run=True)


def test_proteingym_suite_scores_require_full_track():
    payload, _ = proteingym_bundle()
    payload["metrics"] = {"Spearman": 0.5}
    with pytest.raises(ValueError, match="full-track"):
        submit(payload, **arguments(), dry_run=True)
    payload["scope"] = "full"
    payload["completion"] = "complete"
    payload["coverage"] = {"denominator": 2, "scored": 2, "unscored": 0}
    with pytest.raises(ValueError, match="full-track"):
        submit(payload, **arguments(), dry_run=True)
    from rewirebench.submission import _proteingym_assays

    total = sum(_proteingym_assays().values())
    payload["coverage"] = {"denominator": total, "scored": total, "unscored": 0}
    assert submit(payload, **arguments(), dry_run=True)


def test_proteingym_full_assay_wrapper_requires_all_assays():
    payload, _ = proteingym_bundle()
    payload["scope"] = "full"
    with pytest.raises(ValueError, match="all official assays"):
        submit(payload, **arguments(), dry_run=True)


@pytest.mark.parametrize("key", [
    "/private/customer/checkpoint_sha256", "model_/private/checkpoint_sha256",
    "private_customer_revision", "unknown_sha256",
])
def test_provenance_keys_are_allowlisted_before_submission(key):
    payload = bundle()
    payload["provenance"][key] = "a" * (40 if key.endswith("_revision") else 64)
    with pytest.raises(ValueError, match="allowlisted"):
        submit(payload, **arguments(), dry_run=True)


def local_copy_bundle(protocol="tdc-admet-group-v1", dataset="caco2_wang"):
    from rewirebench.protocols import genomic_benchmarks as gb
    from rewirebench.protocols import tdc_admet as tdc

    payload = bundle()
    is_tdc = protocol == tdc.PROTOCOL_ID
    payload.update({
        "protocol_id": protocol,
        "protocol_version": tdc.UPSTREAM_REVISION if is_tdc else gb.PROTOCOL_VERSION,
        "dataset_id": ("tdc-admet-" if is_tdc else "genomic-benchmarks-") + dataset,
        "evaluation_claim": "local_evaluation_not_paper_reproduction",
        "data_verification": "local_bytes_hashed_not_independently_source_verified",
        "metrics": {tdc.ADMET_METRICS[dataset]: 0.5, "n": 2} if is_tdc else (
            {"accuracy": 0.5, "f1_macro": 0.5, "f1_weighted": 0.5, "n": 2}
            if dataset == "human_ensembl_regulatory" else {"accuracy": 0.5, "f1": 0.5, "n": 2}
        ),
        "provenance": {
            "upstream_revision": tdc.UPSTREAM_REVISION if is_tdc else gb.UPSTREAM_REVISION,
            "test_sha256": "d" * 64, "train_sha256": "e" * 64,
        },
    })
    return payload


@pytest.mark.parametrize("protocol,dataset", [
    ("tdc-admet-group-v1", "caco2_wang"),
    ("tdc-admet-group-v1", "hia_hou"),
    ("tdc-admet-group-v1", "cyp2c9_veith"),
    ("tdc-admet-group-v1", "vdss_lombardo"),
    ("genomic-benchmarks-v2", "human_nontata_promoters"),
    ("genomic-benchmarks-v2", "human_ensembl_regulatory"),
])
def test_local_copy_known_metrics_and_partial_coverage_accepted(protocol, dataset):
    payload = local_copy_bundle(protocol, dataset)
    for partial in (False, True):
        if partial:
            payload["completion"] = "partial"
            payload["coverage"] = {"denominator": 5, "scored": 2, "unscored": 3}
        result = submit(payload, **arguments(), dry_run=True)
        assert result["contribution"]["details"]["rewire_bundle"] == payload


@pytest.mark.parametrize("field,value", [
    ("protocol_version", "unreviewed"),
    ("dataset_id", "tdc-admet-private_dataset"),
    ("evaluation_claim", "paper_reproduction"),
    ("data_verification", "pinned_source_bytes"),
    ("provenance", {"test_sha256": "d" * 64}),
    ("provenance", {}),
    ("metrics", {"roc-auc": 0.5, "n": 2}),
    ("metrics", {"mae": 0.5, "n": 1}),
    ("metrics", {"mae": 0.5, "n": 2.0}),
    ("metrics", {"mae": -0.1, "n": 2}),
    ("metrics", {"mae": None, "n": 2}),
    ("metrics", {"mae": 0.5}),
    ("metrics", {"mae": 0.5, "n": 2, "private_sequence": 1}),
    ("metrics", {"mae": {"private_sequence": 1}, "n": 2}),
])
def test_local_copy_unknown_fields_wrong_metrics_and_unsupported_claims_refused(field, value):
    payload = local_copy_bundle()
    payload[field] = value
    with pytest.raises(ValueError):
        submit(payload, **arguments(), dry_run=True)


@pytest.mark.parametrize("dataset,metric,value", [
    ("hia_hou", "roc-auc", 1.1),
    ("cyp2c9_veith", "pr-auc", -0.1),
    ("vdss_lombardo", "spearman", -1.1),
])
def test_tdc_metric_ranges_refused(dataset, metric, value):
    payload = local_copy_bundle(dataset=dataset)
    payload["metrics"][metric] = value
    with pytest.raises(ValueError, match="valid range"):
        submit(payload, **arguments(), dry_run=True)


@pytest.mark.parametrize("dataset,metrics", [
    ("human_nontata_promoters", {"accuracy": 1.1, "f1": 1.0, "n": 2}),
    ("human_nontata_promoters", {"accuracy": 0.5, "f1_macro": 0.5, "n": 2}),
    ("human_ensembl_regulatory", {"accuracy": 0.5, "f1": 0.5, "n": 2}),
])
def test_genomic_metric_ranges_and_averaging_are_dataset_specific(dataset, metrics):
    payload = local_copy_bundle("genomic-benchmarks-v2", dataset)
    payload["metrics"] = metrics
    with pytest.raises(ValueError):
        submit(payload, **arguments(), dry_run=True)


def test_genomic_v1_submission_is_retired():
    payload = local_copy_bundle("genomic-benchmarks-v2", "human_nontata_promoters")
    payload["protocol_id"] = "genomic-benchmarks-v1"
    with pytest.raises(ValueError, match="Unsupported protocol"):
        submit(payload, **arguments(), dry_run=True)


@pytest.mark.parametrize("status", ["submitted", "in_review", "changes_requested", "rejected", "accepted", "published"])
def test_acknowledgement_retains_only_known_identity_and_status(status):
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(json.dumps({"result": {"data": {
                "id": "private-id", "status": status, "email": "private@example.org",
                "provider_body": "private-secret",
            }}}).encode())
    with patch("urllib.request.build_opener", return_value=Opener()):
        result = submit(bundle(), **arguments(), token="verified")
    assert result["submission"] == {"id": "private-id", "status": status}
    assert "private-secret" not in json.dumps(result)


@pytest.mark.parametrize("bad", [
    {"status": "private-secret"}, {"status": ["submitted"]}, {"status": "pending"},
    {"id": "x" * 201}, {"id": "../private"}, {"id": "name@example.org"},
    {"id": "with whitespace"}, {"id": "not.as.jwt"},
])
def test_invalid_success_receipts_are_redacted(bad):
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(json.dumps({"result": {"data": {
                "id": "private-id", "status": "submitted", **bad,
            }}}).encode())
    with (
        patch("urllib.request.build_opener", return_value=Opener()),
        pytest.raises(SubmissionError, match="not acknowledged") as caught,
    ):
        submit(bundle(), **arguments(), token="verified")
    assert "private-secret" not in str(caught.value)
