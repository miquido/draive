import asyncio
from typing import Any

import pytest
from haiway import ctx

from draive.guardrails import (
    GuardrailsException,
    GuardrailsFailure,
    GuardrailsInputModerationException,
    GuardrailsModeration,
    GuardrailsModerationException,
    GuardrailsOutputModerationException,
)
from draive.multimodal import MultimodalContent


async def _violating(content: MultimodalContent, /, **extra: Any) -> None:
    raise GuardrailsModerationException(
        "bad",
        violations={"hate": 0.9},
        content=content,
        replacement=MultimodalContent.of("redacted"),
        meta={"source": "unit"},
    )


async def _generic(content: MultimodalContent, /, **extra: Any) -> None:
    raise GuardrailsException("generic block")


async def _crashing(content: MultimodalContent, /, **extra: Any) -> None:
    raise RuntimeError("provider exploded")


async def _provider_down(content: MultimodalContent, /, **extra: Any) -> None:
    raise GuardrailsFailure("provider down", cause=RuntimeError("network"))


@pytest.mark.asyncio
async def test_moderation_forwards_content_and_extra() -> None:
    seen: dict[str, Any] = {}

    async def checking(content: MultimodalContent, /, **extra: Any) -> None:
        seen["content"] = content.to_str()
        seen["extra"] = extra

    moderation = GuardrailsModeration(input_checking=checking, output_checking=checking)
    async with ctx.scope("test"):
        await moderation.check_input("hello", flag=1)
        assert seen == {"content": "hello", "extra": {"flag": 1}}
        await moderation.check_output("bye")

    assert seen == {"content": "bye", "extra": {}}


@pytest.mark.asyncio
async def test_moderation_defaults_allow_everything() -> None:
    async with ctx.scope("test"):
        await GuardrailsModeration.check_input("anything")
        await GuardrailsModeration.check_output("anything")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("check_input", GuardrailsInputModerationException),
        ("check_output", GuardrailsOutputModerationException),
    ),
)
async def test_moderation_translates_violation_keeping_details(
    method: str,
    expected: type[GuardrailsModerationException],
) -> None:
    moderation = GuardrailsModeration(input_checking=_violating, output_checking=_violating)
    async with ctx.scope("test"):
        with pytest.raises(expected) as raised:
            await getattr(moderation, method)("text")

    assert dict(raised.value.violations) == {"hate": 0.9}
    assert raised.value.replacement is not None
    assert raised.value.replacement.to_str() == "redacted"
    assert raised.value.content.to_str() == "text"
    assert raised.value.meta["source"] == "unit"


@pytest.mark.asyncio
async def test_moderation_reraises_matching_exception_unchanged() -> None:
    original = GuardrailsInputModerationException(
        "already typed",
        violations={},
        content=MultimodalContent.of("x"),
    )

    async def checking(content: MultimodalContent, /, **extra: Any) -> None:
        raise original

    async with ctx.scope("test"):
        with pytest.raises(GuardrailsInputModerationException) as raised:
            await GuardrailsModeration(input_checking=checking).check_input("x")

    assert raised.value is original


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("check_input", "check_output"))
async def test_moderation_wraps_unexpected_error_as_failure(method: str) -> None:
    moderation = GuardrailsModeration(input_checking=_crashing, output_checking=_crashing)
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsFailure) as raised:
            await getattr(moderation, method)("text")

    assert isinstance(raised.value.cause, RuntimeError)
    assert raised.value.meta["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_moderation_does_not_swallow_cancellation() -> None:
    async def checking(content: MultimodalContent, /, **extra: Any) -> None:
        raise asyncio.CancelledError

    async with ctx.scope("test"):
        with pytest.raises(asyncio.CancelledError):
            await GuardrailsModeration(input_checking=checking).check_input("x")


@pytest.mark.asyncio
async def test_moderation_generic_guardrails_exception_blocks_content() -> None:
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsInputModerationException) as raised:
            await GuardrailsModeration(input_checking=_generic).check_input("x")

    assert dict(raised.value.violations) == {"generic block": 1.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("check_input", "check_output"))
async def test_moderation_checker_failure_stays_a_failure(method: str) -> None:
    moderation = GuardrailsModeration(
        input_checking=_provider_down,
        output_checking=_provider_down,
    )
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsFailure):
            await getattr(moderation, method)("text")
