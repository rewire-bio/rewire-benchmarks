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
