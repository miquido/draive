from typing import Any

import pytest
from google.genai.errors import APIError
from haiway import ctx

from draive.gemini.utils import quota_limit
from draive.models import ModelQuotaLimit, ModelRateLimit, model_rate_limit


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("value", ["0", 0, "20", None, "invalid", False, -1])
async def test_only_explicit_zero_quota_is_non_retryable(value: Any, wrapped: bool) -> None:
    violation = {} if value is None else {"quotaValue": value}
    payload = {
        "details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"},
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaValue": "100"}, violation],
            },
        ]
    }
    error = APIError(429, {"error": payload} if wrapped else payload)
    async with ctx.scope("test"):
        translated = model_rate_limit(
            provider="gemini", model="test", retry_after=1.0, quota_limit=quota_limit(error)
        )

    if value in ("0", 0) and not isinstance(value, bool):
        assert isinstance(translated, ModelQuotaLimit)
    else:
        assert isinstance(translated, ModelRateLimit)


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "quota limit: 0"},
        {"details": [{"quotaValue": "0"}]},
        {"details": [{"@type": "unknown", "violations": [{"quotaValue": "0"}]}]},
        {"details": [{"@type": "google.rpc.QuotaFailure", "violations": None}]},
    ],
)
def test_unstructured_or_unrecognized_details_do_not_establish_zero_quota(payload: Any) -> None:
    assert quota_limit(APIError(429, payload)) is None
