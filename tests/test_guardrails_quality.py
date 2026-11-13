import pytest
from haiway import ctx

from draive.evaluation import evaluator
from draive.guardrails import GuardrailsFailure, GuardrailsQualityVerification
from draive.guardrails.quality.types import GuardrailsQualityException
from draive.multimodal import MultimodalContent


@evaluator(name="broken", threshold="perfect")
async def broken_evaluator(value: MultimodalContent) -> float:
    raise RuntimeError("judge unavailable")


@evaluator(name="strict", threshold="perfect")
async def strict_evaluator(value: MultimodalContent) -> float:
    return 0.1


@pytest.mark.asyncio
async def test_quality_guardrail_reports_evaluator_error_as_failure() -> None:
    verification = GuardrailsQualityVerification.of(broken_evaluator.prepared())
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsFailure, match="judge unavailable"):
            await verification.verify(MultimodalContent.of("text"))


@pytest.mark.asyncio
async def test_quality_guardrail_reports_low_quality_with_message() -> None:
    verification = GuardrailsQualityVerification.of(strict_evaluator.prepared())
    async with ctx.scope("test"):
        with pytest.raises(GuardrailsQualityException, match="strict"):
            await verification.verify(MultimodalContent.of("text"))
