import asyncio

import pytest
from haiway import ctx

from draive.models import ModelOutputFailed, ModelRateLimit, provider_failure, provider_gather
from draive.utils import gather_all


@pytest.mark.asyncio
async def test_provider_failure_translates_unexpected_errors() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed, match="embedding request failed - RuntimeError"):
            with provider_failure(provider="test", model="m", operation="embedding"):
                raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_provider_failure_passes_model_exceptions_through() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ModelRateLimit):
            with provider_failure(provider="test", model="m", operation="embedding"):
                raise ModelRateLimit(provider="test", model="m", retry_after=1.0)


@pytest.mark.asyncio
async def test_gather_all_cancels_siblings_and_raises_original() -> None:
    cancelled = asyncio.Event()

    async def slow() -> int:
        try:
            await asyncio.sleep(10)
            return 1

        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def failing() -> int:
        await asyncio.sleep(0)
        raise ValueError("first")

    with pytest.raises(ValueError, match="first"):
        await gather_all(slow(), failing())

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_gather_all_preserves_order() -> None:
    async def value(number: int) -> int:
        await asyncio.sleep(0.01 * (3 - number))
        return number

    assert await gather_all(value(1), value(2), value(3)) == (1, 2, 3)


@pytest.mark.asyncio
async def test_provider_gather_translates_failures() -> None:
    async def failing() -> int:
        raise RuntimeError("boom")

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed):
            await provider_gather(failing(), provider="test", model="m", operation="embedding")
