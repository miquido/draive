import importlib
import sys
import types

import pytest
from haiway import MISSING


def _load_cloudwatch_module(monkeypatch: pytest.MonkeyPatch):
    fake_boto = types.SimpleNamespace(
        Session=type(
            "FakeSession",
            (),
            {
                "__init__": lambda self, **kwargs: None,
                "client": lambda self, service_name: object(),
            },
        ),
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto)

    if "botocore.exceptions" not in sys.modules:
        fake_exceptions = types.SimpleNamespace(
            ClientError=type("ClientError", (Exception,), {}),
        )
        monkeypatch.setitem(
            sys.modules,
            "botocore",
            types.SimpleNamespace(exceptions=fake_exceptions),
        )
        monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exceptions)

    module = importlib.import_module("draive.aws.cloudwatch")
    return importlib.reload(module)


def test_sanitize_attributes_filters_missing_and_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloudwatch = _load_cloudwatch_module(monkeypatch)
    attributes = {
        "name": "draive",
        "empty": None,
        "missing": MISSING,
        "numbers": [1, 2],
        "mixed": ["a", None, "b"],
        "blank": [],
    }

    sanitized = cloudwatch._sanitize_attributes(attributes)

    assert sanitized == {
        "name": "draive",
        "numbers": [1, 2],
        "mixed": ["a", None, "b"],
    }


def test_format_metric_dimensions_respects_sequence_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloudwatch = _load_cloudwatch_module(monkeypatch)
    attributes = {
        "service": "draive",
        "count": 5,
        "tags": ["alpha", "beta"],
        "skip": None,
    }

    dimensions = cloudwatch._format_metric_dimensions(attributes)

    assert dimensions == [
        {"Name": "service", "Value": "draive"},
        {"Name": "count", "Value": "5"},
        {"Name": "tags", "Value": "alpha,beta"},
    ]


def test_translate_cloudwatch_error_includes_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloudwatch = _load_cloudwatch_module(monkeypatch)
    response = {
        "Error": {"Code": "MalformedDetail", "Message": "Detail is not valid JSON"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    error = types.SimpleNamespace(response=response)

    exc = cloudwatch._translate_cloudwatch_error(
        error=error,
        service="events",
        operation="put_events",
        resource="default",
    )

    assert isinstance(exc, cloudwatch.AWSError)
    assert exc.code == "MalformedDetail"


def test_translate_cloudwatch_error_access_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloudwatch = _load_cloudwatch_module(monkeypatch)
    response = {
        "Error": {"Code": "AccessDeniedException", "Message": "Access denied"},
        "ResponseMetadata": {"HTTPStatusCode": 403},
    }
    error = types.SimpleNamespace(response=response)

    exc = cloudwatch._translate_cloudwatch_error(
        error=error,
        service="events",
        operation="put_events",
        resource="default",
    )

    assert isinstance(exc, cloudwatch.AWSAccessDenied)
