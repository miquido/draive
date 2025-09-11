from typing import Any, Protocol, runtime_checkable

from draive.models import ResolveableInstructions
from draive.multimodal import MultimodalContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("ImageGenerating",)


@runtime_checkable
class ImageGenerating(Protocol):
    async def __call__(
        self,
        *,
        instructions: ResolveableInstructions,
        input: MultimodalContent,  # noqa: A002
        **extra: Any,
    ) -> ResourceContent | ResourceReference: ...
