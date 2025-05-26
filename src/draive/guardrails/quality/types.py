from typing import Any, Protocol, runtime_checkable

from draive.commons import Meta, MetaValues
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
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args)
        self.reason: str = reason
        self.content: MultimodalContent = content
        self.meta: Meta = Meta.of(meta)


@runtime_checkable
class GuardrailsQualityVerifying(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> None: ...
