from typing import Any, Self, overload

from haiway import State, statemethod

from draive.evaluation import (
    EvaluatorResult,
    EvaluatorScenarioResult,
    PreparedEvaluator,
    PreparedEvaluatorScenario,
)
from draive.guardrails.quality.types import GuardrailsQualityException, GuardrailsQualityVerifying
from draive.guardrails.types import GuardrailsException, GuardrailsFailure
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
        evaluator: PreparedEvaluatorScenario[MultimodalContent]
        | PreparedEvaluator[MultimodalContent],
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
        content = MultimodalContent.of(content)
        try:
            await self.verifying(
                content,
                **extra,
            )

        except GuardrailsQualityException:
            raise

        except GuardrailsException as exc:
            raise GuardrailsQualityException(
                f"Quality guardrails triggered: {exc}",
                content=content,
                reason=str(exc),
                meta=exc.meta,
            ) from exc

        except Exception as exc:
            raise GuardrailsFailure(
                f"Quality guardrails failed: {exc}",
                cause=exc,
                meta={"error_type": exc.__class__.__name__},
            ) from exc

    verifying: GuardrailsQualityVerifying = _no_verification
