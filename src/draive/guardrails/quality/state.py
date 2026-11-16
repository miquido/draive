from typing import Any, Self, overload

from haiway import State, statemethod

from draive.evaluation import (
    EvaluatorResult,
    EvaluatorScenarioResult,
    PreparedEvaluator,
    PreparedEvaluatorScenario,
)
from draive.guardrails.metrics import guardrails_verdict
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

            # an evaluation that errored is an infrastructure failure, not a quality verdict
            for evaluated in (result,) if isinstance(result, EvaluatorResult) else result.results:
                if (error := evaluated.meta.error) is not None:
                    raise GuardrailsFailure(
                        f"Quality guardrails failed: {error}",
                        cause=error,
                        meta={"error_type": error.__class__.__name__},
                    ) from error

            if isinstance(result, EvaluatorResult):
                raise GuardrailsQualityException(
                    f"Quality guardrails triggered: {result.evaluator}",
                    reason=result.evaluator,
                    content=content,
                    meta={
                        "performance": result.performance,
                        "report": result.report(detailed=True),
                    },
                )

            else:
                raise GuardrailsQualityException(
                    f"Quality guardrails triggered: {result.scenario}",
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
    @guardrails_verdict("quality")
    async def verify(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> None:
        """Verify the quality of the content.

        Parameters
        ----------
        content : Multimodal
            Content to verify.
        **extra : Any
            Additional options forwarded to the guardrails implementation.

        Raises
        ------
        GuardrailsQualityException
            When the content does not meet the quality requirements.
        GuardrailsFailure
            When the verification itself fails, i.e. the evaluator errors.
        """
        content = MultimodalContent.of(content)
        try:
            await self._verifying(
                content,
                **extra,
            )

        except GuardrailsQualityException, GuardrailsFailure:
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

    _verifying: GuardrailsQualityVerifying

    def __init__(
        self,
        verifying: GuardrailsQualityVerifying = _no_verification,
    ) -> None:
        super().__init__(_verifying=verifying)
