from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from draive.multimodal import MultimodalContent

__all__ = (
    "GuardrailsContentException",
    "GuardrailsContentVerifying",
    "GuardrailsInputException",
    "GuardrailsOutputException",
)


class GuardrailsContentException(Exception):
    def __init__(
        self,
        *args: object,
        violations: Sequence[str],
        content: MultimodalContent,
        replacement: MultimodalContent | None = None,
    ) -> None:
        super().__init__(*args)
        self.violations: Sequence[str] = violations
        self.content: MultimodalContent = content
        self.replacement: MultimodalContent | None = replacement


class GuardrailsInputException(GuardrailsContentException):
    def __init__(
        self,
        *args: object,
        violations: Sequence[str],
        content: MultimodalContent,
        replacement: MultimodalContent | None = None,
    ) -> None:
        super().__init__(
            *args,
            violations=violations,
            content=content,
            replacement=replacement,
        )


class GuardrailsOutputException(GuardrailsContentException):
    def __init__(
        self,
        *args: object,
        violations: Sequence[str],
        content: MultimodalContent,
        replacement: MultimodalContent | None = None,
    ) -> None:
        super().__init__(
            *args,
            violations=violations,
            content=content,
            replacement=replacement,
        )


@runtime_checkable
class GuardrailsContentVerifying(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> None: ...
