from typing import Any, Protocol, runtime_checkable

from haiway import Meta, State

from draive.multimodal import MultimodalContent

__all__ = (
    "GuardrailsAnonymizedContent",
    "GuardrailsContentAnonymizing",
    "GuardrailsContentDeanonymizing",
)


@runtime_checkable
class GuardrailsContentDeanonymizing(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
    ) -> MultimodalContent: ...


class GuardrailsAnonymizedContent(State):
    content: MultimodalContent
    deanonymize: GuardrailsContentDeanonymizing | None = None
    meta: Meta = Meta.empty


@runtime_checkable
class GuardrailsContentAnonymizing(Protocol):
    async def __call__(
        self,
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> GuardrailsAnonymizedContent: ...
