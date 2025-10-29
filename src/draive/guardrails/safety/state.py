from typing import Any, overload

from haiway import State, statemethod

from draive.guardrails.safety.regex import guardrails_regex_sanitizer
from draive.guardrails.safety.types import (
    GuardrailsSafetyException,
    GuardrailsSafetySanitization,
)
from draive.guardrails.types import GuardrailsException, GuardrailsFailure
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("GuardrailsSafety",)


class GuardrailsSafety(State):
    @overload
    @classmethod
    async def sanitize(
        cls,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> MultimodalContent: ...

    @overload
    async def sanitize(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> MultimodalContent: ...

    @statemethod
    async def sanitize(
        self,
        content: Multimodal,
        /,
        **extra: Any,
    ) -> MultimodalContent:
        content = MultimodalContent.of(content)
        try:
            return await self.sanitization(
                content,
                **extra,
            )

        except GuardrailsSafetyException:
            raise

        except GuardrailsException as exc:
            raise GuardrailsSafetyException(
                f"Safety guardrails triggered: {exc}",
                content=content,
                reason=str(exc),
                meta=exc.meta,
            ) from exc

        except Exception as exc:
            raise GuardrailsFailure(
                f"Safety guardrails failed: {exc}",
                cause=exc,
                meta={"error_type": exc.__class__.__name__},
            ) from exc

    # use default set of regex rules for sanitization
    sanitization: GuardrailsSafetySanitization = guardrails_regex_sanitizer()
