from typing import Any, Self

from haiway import State, ctx

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.evaluation.scenario import PreparedScenarioEvaluator, ScenarioEvaluatorResult
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
        evaluator: PreparedScenarioEvaluator | PreparedEvaluator,
    ) -> Self:
        async def verifying(
            content: MultimodalContent,
            /,
            **extra: Any,
        ) -> None:
            result: ScenarioEvaluatorResult | EvaluatorResult = await evaluator(content)
            if result.passed:
                return  # verification passed

            match result:
                case ScenarioEvaluatorResult() as result:
                    raise GuardrailsQualityException(
                        reason=result.scenario,
                        content=content,
                        meta=result.meta,
                    )

                case EvaluatorResult() as result:
                    raise GuardrailsQualityException(
                        reason=result.evaluator,
                        content=content,
                        meta=result.meta,
                    )

        return cls(verifying=verifying)

    @classmethod
    async def verify(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        await ctx.state(cls).verifying(MultimodalContent.of(content))

    verifying: GuardrailsQualityVerifying = _no_verification
