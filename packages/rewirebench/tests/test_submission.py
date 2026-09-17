import io
import json
import urllib.error
from unittest.mock import patch
import pytest
from rewirebench.submission import submit, SubmissionError


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
    return dict(
        title="Example evaluation",
        summary="An explicitly submitted local evaluation.",
        source_url="https://example.org/paper",
        metric="auroc",
        value="0.7",
        source_locator="Table 1",
    )


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

    with patch("urllib.request.build_opener", return_value=Opener()):
        with pytest.raises(SubmissionError, match=text) as error:
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

    with patch("urllib.request.build_opener", return_value=Opener()):
        with pytest.raises(SubmissionError, match="not acknowledged"):
            submit(bundle(), **arguments(), token="verified")


@pytest.mark.parametrize("metrics", [{"per_assay": {}}, {"AUROC": None}])
def test_empty_numerical_metrics_rejected(metrics):
    payload = bundle()
    payload["metrics"] = metrics
    with pytest.raises(ValueError, match="No numerical"):
        submit(payload, **arguments(), dry_run=True)
