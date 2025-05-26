from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from draive.commons import Meta, MetaValues
from draive.multimodal import MultimodalContent

__all__ = (
    "GuardrailsInputModerationException",
    "GuardrailsModerationChecking",
    "GuardrailsModerationException",
    "GuardrailsOutputModerationException",
)


class GuardrailsModerationException(Exception):
    def __init__(
        self,
        *args: object,
        violations: Sequence[str],
        content: MultimodalContent,
        replacement: MultimodalContent | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args)
        self.violations: Sequence[str] = violations
        self.content: MultimodalContent = content
        self.replacement: MultimodalContent | None = replacement
        self.meta: Meta = Meta.of(meta)


class GuardrailsInputModerationException(GuardrailsModerationException):
    def __init__(
        self,
        *args: object,
        violations: Sequence[str],
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
        violations: Sequence[str],
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
