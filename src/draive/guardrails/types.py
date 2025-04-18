from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from draive.multimodal import MultimodalContent

__all__ = (
    "ContentGuardrailsException",
    "GuardrailsContentVerifying",
)


class ContentGuardrailsException(Exception):
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


@runtime_checkable
class GuardrailsContentVerifying(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> None: ...
