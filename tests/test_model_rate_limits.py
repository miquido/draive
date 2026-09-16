import pytest
from haiway import ctx

from draive import ModelQuotaLimit, ModelRateLimit
from draive.models import model_rate_limit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, quota, code",
    [
        ("openai", None, "insufficient_quota"),
        ("vllm", None, "billing_hard_limit_reached"),
        ("test", 0, None),
    ],
)
async def test_explicit_quota_signals_override_retry_after(
    provider: str, quota: int | None, code: str | None
) -> None:
    async with ctx.scope("test"):
        error = model_rate_limit(
            provider=provider, model="test", retry_after="15", quota_limit=quota, error_code=code
        )

    assert isinstance(error, ModelQuotaLimit)
    assert not isinstance(error, ModelRateLimit)
    assert error.provider == provider
    assert error.model == "test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quota",
    [None, 100],
)
async def test_no_explicit_quota_signal_preserves_rate_limit(quota: int | None) -> None:
    async with ctx.scope("test"):
        error = model_rate_limit(
            provider="test",
            model="test",
            retry_after="15",
            quota_limit=quota,
            error_code="rate_limit_exceeded",
        )

    assert isinstance(error, ModelRateLimit)
    assert error.retry_after == 15


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", [None, "invalid"])
async def test_missing_retry_delay_keeps_randomized_backoff(retry_after: str | None) -> None:
    async with ctx.scope("test"):
        error = model_rate_limit(provider="test", model="test", retry_after=retry_after)

    assert isinstance(error, ModelRateLimit)
    assert 0.3 <= error.retry_after <= 3
