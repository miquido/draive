import math

import pytest
from haiway import State

from draive.steps import Step, StepState


class Payload(State, serializable=True):
    value: int


@pytest.mark.asyncio
async def test_process_with_plain_state_gives_clear_error() -> None:
    with pytest.raises(TypeError, match="StepState"):
        await Step.noop.process(Payload(value=1))  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_step_state_updating_artifacts_accepts_self_key() -> None:
    state = StepState.of().updating_artifacts(**{"self": Payload(value=1)})

    assert state.artifacts["self"].to_state(Payload) == Payload(value=1)


def test_step_state_of_accepts_cls_and_context_like_keys() -> None:
    state = StepState.of((), **{"cls": Payload(value=1), "context": Payload(value=2)})

    assert set(state.artifacts) == {"cls", "context"}


@pytest.mark.asyncio
async def test_step_updating_artifacts_and_process_accept_self_key() -> None:
    step = Step.updating_artifacts(**{"self": Payload(value=1), "cls": Payload(value=2)})

    state = await step.process((), **{"self": Payload(value=0)})

    assert state.artifacts["self"].to_state(Payload) == Payload(value=1)
    assert state.artifacts["cls"].to_state(Payload) == Payload(value=2)


@pytest.mark.parametrize("seconds", (math.nan, 0, -1.0, math.inf, -math.inf))
def test_with_timeout_rejects_non_finite_or_non_positive(seconds: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        Step.noop.with_timeout(seconds)


def test_with_timeout_accepts_finite_positive() -> None:
    assert Step.noop.with_timeout(0.5) is not None
