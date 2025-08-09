from typing import Any, Self, overload

from haiway import State, statemethod

from draive.evaluation import (
    EvaluatorResult,
    EvaluatorScenarioResult,
    PreparedEvaluator,
    PreparedEvaluatorScenario,
)
from draive.guardrails.quality.types import GuardrailsQualityException, GuardrailsQualityVerifying
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("GuardrailsQualityVerification",)


async def _no_verification(
    content: MultimodalContent,
    /,
    **extra: Any,
) -> None:
    pass


class GuardrailsQualityVerification(State):
    @classmethod
    def of(
        cls,
        evaluator: PreparedEvaluatorScenario | PreparedEvaluator,
    ) -> Self:
        async def verifying(
            content: MultimodalContent,
            /,
            **extra: Any,
        ) -> None:
            result: EvaluatorScenarioResult | EvaluatorResult = await evaluator(content)
            if result.passed:
                return  # verification passed

            if isinstance(result, EvaluatorResult):
                raise GuardrailsQualityException(
                    reason=result.evaluator,
                    content=content,
                    meta={
                        "performance": result.performance,
                        "report": result.report(detailed=True),
                    },
                )

            else:
                raise GuardrailsQualityException(
                    reason=result.scenario,
                    content=content,
                    meta={
                        "performance": result.performance,
                        "report": result.report(detailed=True),
                    },
                )

        return cls(verifying=verifying)

    @overload
    @classmethod
    async def verify(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None: ...

    @overload
    async def verify(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def verify(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        await self.verifying(
            MultimodalContent.of(content),
            **extra,
        )

    verifying: GuardrailsQualityVerifying = _no_verification
