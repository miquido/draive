from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from haiway import Meta, MetaValues

from draive.guardrails.types import GuardrailsException
from draive.multimodal import MultimodalContent

__all__ = (
    "GuardrailsInputModerationException",
    "GuardrailsModerationChecking",
    "GuardrailsModerationException",
    "GuardrailsOutputModerationException",
)


class GuardrailsModerationException(GuardrailsException):
    __slots__ = ("content", "replacement", "violations")

    def __init__(
        self,
        *args: object,
        violations: Mapping[str, float],
        content: MultimodalContent,
        replacement: MultimodalContent | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args, meta=meta)
        self.violations: Mapping[str, float] = violations
        self.content: MultimodalContent = content
        self.replacement: MultimodalContent | None = replacement


class GuardrailsInputModerationException(GuardrailsModerationException):
    def __init__(
        self,
        *args: object,
        violations: Mapping[str, float],
        content: MultimodalContent,
        replacement: MultimodalContent | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(
            *args,
            violations=violations,
            content=content,
            replacement=replacement,
            meta=meta,
        )


class GuardrailsOutputModerationException(GuardrailsModerationException):
    def __init__(
        self,
        *args: object,
        violations: Mapping[str, float],
        content: MultimodalContent,
        replacement: MultimodalContent | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(
            *args,
            violations=violations,
            content=content,
            replacement=replacement,
            meta=meta,
        )


@runtime_checkable
class GuardrailsModerationChecking(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> None: ...
