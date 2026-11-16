import asyncio

import pytest
from haiway import State, ctx

from draive.steps import Step, StepRetriesExhausted, StepState, StepStream, StepTimeout


class Marker(State):
    value: int = 0


async def _raising_timeout_error(state: StepState) -> StepStream:
    raise TimeoutError("raised by the step itself")
    yield  # pyright: ignore[reportUnreachable]


@pytest.mark.asyncio
async def test_timeout_cleans_up_cancelled_step() -> None:
    cleaned: list[bool] = []

    async def slow(state: StepState) -> StepStream:
        try:
            await asyncio.sleep(5)
            yield state

        finally:
            cleaned.append(True)

    async with ctx.scope("test"):
        with pytest.raises(StepTimeout):
            await Step(slow).with_timeout(0.02).process()

    assert cleaned == [True]


@pytest.mark.asyncio
async def test_timeout_wrapper_does_not_alter_fast_step_state() -> None:
    async with ctx.scope("test"):
        state = await Step.updating_artifacts(Marker(value=2)).with_timeout(5).process()

    assert state.get(Marker) == Marker(value=2)


@pytest.mark.asyncio
async def test_inner_timeout_error_is_not_converted_to_step_timeout() -> None:
    async with ctx.scope("test"):
        with pytest.raises(BaseException) as error:
            await Step(_raising_timeout_error).with_timeout(30).process()

    assert not isinstance(error.value, StepTimeout)


@pytest.mark.asyncio
async def test_retry_exhaustion_wraps_last_error_in_sequence() -> None:
    attempts: list[int] = []

    async def failing(state: StepState) -> StepStream:
        attempts.append(1)
        raise ValueError("boom")
        yield  # pyright: ignore[reportUnreachable]

    async with ctx.scope("test"):
        with pytest.raises(StepRetriesExhausted) as error:
            await Step.sequence(Step(failing).with_retry(limit=2)).process()

    assert len(attempts) == 3
    assert isinstance(error.value.__cause__, ValueError)
