from typing import Any, Protocol, runtime_checkable

from draive.commons import META_EMPTY, Meta
from draive.multimodal import MultimodalContent

__all__ = (
    "GuardrailsQualityException",
    "GuardrailsQualityVerifying",
)


class GuardrailsQualityException(Exception):
    def __init__(
        self,
        *args: object,
        reason: str,
        content: MultimodalContent,
        meta: Meta | None = None,
    ) -> None:
        super().__init__(*args)
        self.reason: str = reason
        self.content: MultimodalContent = content
        self.meta: Meta = meta if meta is not None else META_EMPTY


@runtime_checkable
class GuardrailsQualityVerifying(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> None: ...
