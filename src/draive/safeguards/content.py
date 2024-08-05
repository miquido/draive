from typing import Any, Protocol, runtime_checkable

from draive.types import MultimodalContent

__all__ = [
    "ContentGuardrails",
]


@runtime_checkable
class ContentGuardrails(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> MultimodalContent: ...
